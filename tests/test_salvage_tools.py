"""残件抢救纯函数测试：按行组试读、整组舍弃、顺序保留、缓冲不随行数增长。"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from app.parquet_tools import salvage_row_groups

SCHEMA = pa.schema([("tid", pa.string()), ("h", pa.float64())])


def _write_groups(path, sizes):
    writer = pq.ParquetWriter(path, SCHEMA, compression="snappy")
    for g, size in enumerate(sizes):
        writer.write_table(
            pa.table(
                {
                    "tid": [f"g{g}-r{i}" for i in range(size)],
                    "h": [g * 1000 + i for i in range(size)],
                },
                schema=SCHEMA,
            )
        )
    writer.close()


def _corrupt_row_group(path, row_group_index, column_index=0):
    """破坏指定行组某列的数据页字节，页脚/行组目录保持可读。"""
    pf = pq.ParquetFile(path)
    offset = pf.metadata.row_group(row_group_index).column(column_index).data_page_offset
    with open(path, "r+b") as fh:
        fh.seek(offset + 4)
        fh.write(b"\xff" * 16)


def test_salvage_drops_whole_group_and_keeps_order(tmp_path):
    src = tmp_path / "broken.parquet"
    dst = tmp_path / "salvaged.parquet"
    _write_groups(str(src), [10, 10, 10, 10])
    _corrupt_row_group(str(src), 1)  # 第 2 组损坏

    result = salvage_row_groups(str(src), str(dst), ["tid", "h"])

    assert result.kept_groups == [1, 3, 4]          # 原顺序，行组号从 1 起
    assert [g.row_group_no for g in result.dropped_groups] == [2]
    assert result.dropped_groups[0].source_rows == 10  # 目录所记行数
    assert result.kept_rows == 30

    out = pq.ParquetFile(str(dst))
    data = out.read().to_pydict()
    assert len(data["tid"]) == 30
    # 可读组按原件顺序首尾相接，没有来自坏组的 g1-* 行
    assert data["tid"][0] == "g0-r0"
    assert data["tid"][10] == "g2-r0"
    assert data["tid"][-1] == "g3-r9"
    assert not any(t.startswith("g1-") for t in data["tid"])
    # 派生件只含所选列
    assert out.schema_arrow.names == ["tid", "h"]


def test_mid_group_failure_still_drops_whole_group(tmp_path):
    src = tmp_path / "mid.parquet"
    dst = tmp_path / "mid_out.parquet"
    # 单组 50 行、组内多页；破坏数据页靠后的字节，使前几批能读、组尾才报错
    _write_groups(str(src), [50, 20])
    pf = pq.ParquetFile(str(src))
    offset = pf.metadata.row_group(0).column(0).data_page_offset
    with open(str(src), "r+b") as fh:
        fh.seek(offset + 120)
        fh.write(b"\xee" * 8)

    result = salvage_row_groups(str(src), str(dst), ["tid", "h"], batch_size=5)

    # 第 1 组读到一半报错 → 整组舍弃，不留半成品行；第 2 组完好
    assert result.kept_groups == [2]
    assert [g.row_group_no for g in result.dropped_groups] == [1]
    assert result.kept_rows == 20
    out = pq.ParquetFile(str(dst)).read().to_pydict()
    assert all(t.startswith("g1-") for t in out["tid"])


def test_unselected_corrupt_column_does_not_drop_group(tmp_path):
    src = tmp_path / "other_col.parquet"
    dst = tmp_path / "other_out.parquet"
    _write_groups(str(src), [10, 10])
    _corrupt_row_group(str(src), 0, column_index=1)  # 坏的是 h 列

    # 只抢救 tid 列：坏列不在所选列中，该组仍可读
    result = salvage_row_groups(str(src), str(dst), ["tid"])
    assert result.kept_groups == [1, 2]
    assert result.dropped_groups == []
    assert pq.ParquetFile(str(dst)).schema_arrow.names == ["tid"]


def test_all_groups_broken_produces_no_output(tmp_path):
    src = tmp_path / "allbad.parquet"
    dst = tmp_path / "allbad_out.parquet"
    _write_groups(str(src), [5, 5])
    _corrupt_row_group(str(src), 0)
    _corrupt_row_group(str(src), 1)

    result = salvage_row_groups(str(src), str(dst), ["tid", "h"])
    assert result.kept_groups == []
    assert result.kept_rows == 0
    assert len(result.dropped_groups) == 2
    assert not dst.exists()  # 无可用行不生成派生件


def test_buffer_does_not_grow_with_row_group_size(tmp_path, monkeypatch):
    """批大小固定时，无论行组多大，iter_batches 每次请求的行数都不超过批大小。"""
    import app.parquet_tools as pt

    src = tmp_path / "large.parquet"
    dst = tmp_path / "large_out.parquet"
    _write_groups(str(src), [5_000, 3_000])  # 大行组

    seen_sizes: list[int] = []
    real_iter = pq.ParquetFile.iter_batches

    def tracking_iter(self, *args, **kwargs):
        seen_sizes.append(kwargs.get("batch_size"))
        return real_iter(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", tracking_iter)
    result = pt.salvage_row_groups(str(src), str(dst), ["tid", "h"], batch_size=64)

    assert result.kept_rows == 8_000
    assert seen_sizes  # 确实走了分批接口
    assert set(seen_sizes) == {64}  # 缓冲恒定，不随行组/文件行数增长
