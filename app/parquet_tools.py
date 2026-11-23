"""Parquet 读取与单位换算。

全部基于 pyarrow 分批扫描，不使用 pandas，内存占用与批大小相关，
不随文件总行数增长。
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .constants import OUTPUT_COLUMNS, PREVIEW_ROWS, READ_BATCH_SIZE


def parquet_columns(path: str) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def is_numeric_column(path: str, column: str) -> bool:
    schema = pq.ParquetFile(path).schema_arrow
    return pa.types.is_integer(schema.field(column).type) or pa.types.is_floating(
        schema.field(column).type
    )


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # NaN 等一律视为缺项
    if f != f:
        return None
    return f


def convert_height(value, unit: str) -> float | None:
    """高度统一成米；厘米除以 100，缺项返回 None（输出留空）。"""
    f = _to_float(value)
    if f is None:
        return None
    return f / 100.0 if unit == "cm" else f


def format_height(value: float | None) -> str:
    if value is None:
        return ""
    return format(value, ".6f").rstrip("0").rstrip(".")


def iter_output_rows(
    path: str,
    tree_id_column: str,
    metric_sources: dict[str, tuple[str, str]],
    batch_no: str,
    version_no: int,
) -> Iterator[dict[str, str]]:
    """按批扫描 Parquet，产出统一口径的输出行（生成器，不整表载入）。

    metric_sources: {metric_key: (源列名, 单位)}，没有配置的指标输出空串。
    保留 Parquet 中全部样木行，不做任何过滤。
    """
    wanted = [tree_id_column]
    for source_col, _unit in metric_sources.values():
        if source_col not in wanted:
            wanted.append(source_col)

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=READ_BATCH_SIZE, columns=wanted):
        columns = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        id_values = columns[tree_id_column]
        for i in range(batch.num_rows):
            row = {
                "batch_no": batch_no,
                "version_no": str(version_no),
                "tree_id": "" if id_values[i] is None else str(id_values[i]),
            }
            for metric_key, out_col in (
                ("tree_height", "tree_height_m"),
                ("under_branch_height", "under_branch_height_m"),
            ):
                if metric_key in metric_sources:
                    source_col, unit = metric_sources[metric_key]
                    row[out_col] = format_height(
                        convert_height(columns[source_col][i], unit)
                    )
                else:
                    row[out_col] = ""
            yield row


def preview_rows(
    path: str,
    tree_id_column: str,
    metric_sources: dict[str, tuple[str, str]],
    batch_no: str,
    version_no: int,
    limit: int = PREVIEW_ROWS,
) -> list[dict[str, str]]:
    """只取前 limit 行：iter_batches 读到足够行即停止，不读完整批。"""
    rows: list[dict[str, str]] = []
    for row in iter_output_rows(
        path, tree_id_column, metric_sources, batch_no, version_no
    ):
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def csv_chunks(rows: Iterable[dict[str, str]], include_bom: bool = True) -> Iterator[bytes]:
    """把输出行流式编码为 CSV 块。带 UTF-8 BOM，Excel 直接打开不乱码。"""
    if include_bom:
        yield "\ufeff".encode("utf-8")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()
    yield buffer.getvalue().encode("utf-8")
    buffer.seek(0)
    buffer.truncate(0)
    for row in rows:
        writer.writerow(row)
        if buffer.tell() >= 64 * 1024:
            yield buffer.getvalue().encode("utf-8")
            buffer.seek(0)
            buffer.truncate(0)
    if buffer.tell():
        yield buffer.getvalue().encode("utf-8")


# ---------- 残件抢救：按行组试读，整组成败 ----------

@dataclass
class DroppedGroup:
    """被舍弃的原件行组（行组号从 1 起）。"""

    row_group_no: int
    source_rows: int
    error: str


@dataclass
class SalvageResult:
    kept_groups: list[int] = field(default_factory=list)  # 原件行组号（从 1 起），按原顺序
    dropped_groups: list[DroppedGroup] = field(default_factory=list)
    kept_rows: int = 0
    output_path: str = ""


def salvage_row_groups(
    source_path: str,
    output_path: str,
    wanted_columns: list[str],
    batch_size: int = READ_BATCH_SIZE,
) -> SalvageResult:
    """抢救行组目录仍可读、部分数据已损坏的 Parquet 暂存件。

    仅按所选列探测：每个行组完整读到结尾才算可读，读到一半报错则**整组舍弃**。
    可读行组按原件中的顺序保留，流式重写成只含所选列的新 Parquet。

    两遍扫描都只持有 ``batch_size`` 大小的 RecordBatch 缓冲，
    内存不随文件或单个行组的行数增长。

    派生件写入失败会清理半成品并向上抛异常，由调用方拒绝发布。
    """
    pf = pq.ParquetFile(source_path)
    schema = pa.schema([pf.schema_arrow.field(name) for name in wanted_columns])
    total_groups = pf.num_row_groups

    # 第一遍：逐行组试读所选列；中途任何异常都让该组整体落选
    kept_indexes: list[int] = []
    dropped: list[DroppedGroup] = []
    for rg_idx in range(total_groups):
        source_rows = pf.metadata.row_group(rg_idx).num_rows
        try:
            for batch in pf.iter_batches(
                batch_size=batch_size,
                columns=wanted_columns,
                row_groups=[rg_idx],
            ):
                # 触发整批解码：to_pylist 确保列数据页真正被读取，
                # 而不是只生成惰性引用
                for name in wanted_columns:
                    batch.column(name).to_pylist()
        except Exception as exc:  # 损坏、解压失败、IO 错误等
            dropped.append(DroppedGroup(rg_idx + 1, source_rows, f"{type(exc).__name__}: {exc}"))
            continue
        kept_indexes.append(rg_idx)

    if not kept_indexes:
        # 无可用行：不生成派生件
        return SalvageResult(dropped_groups=dropped)

    # 第二遍：可读行组按原顺序流式重写；写失败则清理半成品并抛错
    kept_rows = 0
    try:
        with pq.ParquetWriter(output_path, schema) as writer:
            for rg_idx in kept_indexes:
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    columns=wanted_columns,
                    row_groups=[rg_idx],
                ):
                    writer.write_batch(batch)
                    kept_rows += batch.num_rows
    except Exception:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise

    return SalvageResult(
        kept_groups=[i + 1 for i in kept_indexes],
        dropped_groups=dropped,
        kept_rows=kept_rows,
        output_path=output_path,
    )
