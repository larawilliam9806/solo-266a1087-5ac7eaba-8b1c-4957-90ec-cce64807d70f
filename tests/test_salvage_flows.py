"""残件抢救端到端流程：换交页抢救 → 确认清单 → 发布/拒绝 → 版号详情可查。"""
from __future__ import annotations

import csv
import io
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import publish_form, stage_token_from_redirect, upload  # noqa: E402

SCHEMA = pa.schema([("tid", pa.string()), ("h", pa.float64())])


def _parse_csv(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


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


def _corrupt(path, row_group_index, column_index=0, seek_extra=4):
    pf = pq.ParquetFile(path)
    offset = pf.metadata.row_group(row_group_index).column(column_index).data_page_offset
    with open(path, "r+b") as fh:
        fh.seek(offset + seek_extra)
        fh.write(b"\xff" * 16)


def _publish_first(client, parquet_factory, batch_no="PLOT-1"):
    path = parquet_factory(
        "v1.parquet",
        {"tid": [f"old-{i}" for i in range(5)], "h": [1.0 + i for i in range(5)]},
    )
    token = stage_token_from_redirect(upload(client, path))
    client.post(
        "/batches/new",
        data=publish_form(
            token,
            tree_id_column="tid",
            metrics={"tree_height": ("h", "m")},
            batch_no=batch_no,
            time_point="2024-09",
        ),
    )


def test_salvage_publish_flow(client, parquet_factory, tmp_path):
    _publish_first(client, parquet_factory)

    # 残件：4 个行组，第 2 组损坏
    damaged = str(tmp_path / "damaged.parquet")
    _write_groups(damaged, [10, 10, 10, 10])
    _corrupt(damaged, 1)

    # 损坏文件仍能通过上传（页脚/行组目录可读，列名可解析）
    resp = upload(client, damaged, target="replace:1")
    token = stage_token_from_redirect(resp)

    # 换交页有抢救入口
    page = client.get(f"/batches/1/replace?token={token}").text
    assert 'action="/batches/1/salvage"' in page

    resp = client.post(
        "/batches/1/salvage",
        data=publish_form(
            token, tree_id_column="tid", metrics={"tree_height": ("h", "m")}
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 302
    job_url = resp.headers["Location"]
    preview = client.get(job_url).text

    # 确认页列出舍弃行组号（从 1 起）与目录所记行数
    assert "残件抢救确认" in preview
    assert "<td>2</td>" in preview          # 舍弃的是第 2 组
    assert "10" in preview                  # 目录所记行数
    assert "g0-r0" in preview and "g2-r0" in preview
    assert "g1-r0" not in preview           # 坏组数据不出现
    assert "确认发布为新正式版" in preview

    job_id = job_url.rstrip("/").split("/")[-1]
    resp = client.post(f"/salvage/{job_id}/publish", follow_redirects=True)
    assert "抢救发布成功" in resp.text

    # v2 成为正式版，v1 保留
    detail = client.get("/batches/1").text
    assert "v1" in detail and "v2" in detail

    # 正式版数据 = 30 行可读行，顺序保留且不含坏组
    rows = _parse_csv(client.get("/merge/stream?batch_ids=1").data)
    assert len(rows) == 30
    assert rows[0]["tree_id"] == "g0-r0"
    assert rows[10]["tree_id"] == "g2-r0"
    assert rows[-1]["tree_id"] == "g3-r9"
    assert all(not r["tree_id"].startswith("g1-") for r in rows)
    assert {r["version_no"] for r in rows} == {"2"}

    # 版号详情能查到弃组清单
    version_page = client.get("/versions/2").text
    assert "残件抢救清单" in version_page
    assert "<td>2</td>" in version_page

    # 抢救成功后暂存原件已清理
    saved_token = token
    assert client.get(f"/batches/1/replace?token={saved_token}").status_code == 200


def test_salvage_no_usable_rows_rejected_and_original_kept(client, parquet_factory, tmp_path):
    _publish_first(client, parquet_factory)

    damaged = str(tmp_path / "allbad.parquet")
    _write_groups(damaged, [5, 5])
    _corrupt(damaged, 0)
    _corrupt(damaged, 1)
    token = stage_token_from_redirect(upload(client, damaged, target="replace:1"))

    resp = client.post(
        "/batches/1/salvage",
        data=publish_form(
            token, tree_id_column="tid", metrics={"tree_height": ("h", "m")}
        ),
        follow_redirects=True,
    )
    assert "无可用行" in resp.text
    assert "拒绝发布" in resp.text

    # 当前正式版不变（仍是 v1 的 5 行），待交原件仍在暂存
    detail = client.get("/batches/1").text
    assert "v2" not in detail
    rows = _parse_csv(client.get("/versions/1/download").data)
    assert len(rows) == 5
    # 暂存记录还在，换交页仍带着该 token
    page = client.get(f"/batches/1/replace?token={token}").text
    assert token in page
    # 没有留下抢救任务
    from app.db import get_db  # noqa: E402
    with client.application.app_context():
        jobs = get_db().execute("SELECT COUNT(*) AS c FROM salvage_jobs").fetchone()
        assert jobs["c"] == 0


def test_salvage_rejected_when_official_changed_after_start(client, parquet_factory, tmp_path):
    _publish_first(client, parquet_factory)

    # 开始一次抢救
    damaged = str(tmp_path / "broken.parquet")
    _write_groups(damaged, [8, 8])
    _corrupt(damaged, 0)
    token = stage_token_from_redirect(upload(client, damaged, target="replace:1"))
    resp = client.post(
        "/batches/1/salvage",
        data=publish_form(
            token, tree_id_column="tid", metrics={"tree_height": ("h", "m")}
        ),
        follow_redirects=False,
    )
    job_id = resp.headers["Location"].rstrip("/").split("/")[-1]

    # 抢救未确认期间，另一份完好换交件正常发布为 v2
    good = parquet_factory(
        "good.parquet",
        {"tid": [f"new-{i}" for i in range(3)], "h": [2.0 + i for i in range(3)]},
    )
    token2 = stage_token_from_redirect(upload(client, good, target="replace:1"))
    resp = client.post(
        "/batches/1/replace",
        data=publish_form(
            token2, tree_id_column="tid", metrics={"tree_height": ("h", "m")}
        ),
        follow_redirects=True,
    )
    assert "换交成功" in resp.text

    # 再确认旧抢救：正式版已换交，拒绝发布，保留 v2 与待交原件
    resp = client.post(f"/salvage/{job_id}/publish", follow_redirects=True)
    assert "正式版已换交" in resp.text
    rows = _parse_csv(client.get("/merge/stream?batch_ids=1").data)
    assert len(rows) == 3
    assert rows[0]["tree_id"] == "new-0"
    # 抢救任务与待交原件仍保留，可放弃后重来
    page = client.get(f"/salvage/{job_id}")
    assert page.status_code == 200
    resp = client.post(f"/salvage/{job_id}/discard", follow_redirects=True)
    assert "放弃本次抢救" in resp.text
    # 放弃后暂存原件仍在
    assert token in client.get(f"/batches/1/replace?token={token}").text


def test_salvage_validation_failure_keeps_original(client, parquet_factory, tmp_path):
    _publish_first(client, parquet_factory)
    damaged = str(tmp_path / "broken2.parquet")
    _write_groups(damaged, [6, 6])
    _corrupt(damaged, 1)
    token = stage_token_from_redirect(upload(client, damaged, target="replace:1"))

    # 列对照不合法（样木号列不存在）→ 抢救入口直接沿用换交校验，拒绝
    resp = client.post(
        "/batches/1/salvage",
        data=publish_form(
            token, tree_id_column="missing_col",
            metrics={"tree_height": ("h", "m")},
        ),
        follow_redirects=True,
    )
    assert "不在文件列中" in resp.text
    detail = client.get("/batches/1").text
    assert "v2" not in detail


def test_parallel_batch_salvage_and_normal_replace_are_independent(
    client, parquet_factory, tmp_path
):
    # 同批：一次抢救失败（无可用行）不影响随后另一份换交正常发布
    _publish_first(client, parquet_factory)
    allbad = str(tmp_path / "allbad2.parquet")
    _write_groups(allbad, [4, 4])
    _corrupt(allbad, 0)
    _corrupt(allbad, 1)
    bad_token = stage_token_from_redirect(upload(client, allbad, target="replace:1"))
    resp = client.post(
        "/batches/1/salvage",
        data=publish_form(
            bad_token, tree_id_column="tid", metrics={"tree_height": ("h", "m")}
        ),
        follow_redirects=True,
    )
    assert "无可用行" in resp.text

    good = parquet_factory(
        "good2.parquet",
        {"tid": [f"ok-{i}" for i in range(7)], "h": [0.5 * i for i in range(7)]},
    )
    good_token = stage_token_from_redirect(upload(client, good, target="replace:1"))
    resp = client.post(
        "/batches/1/replace",
        data=publish_form(
            good_token, tree_id_column="tid", metrics={"tree_height": ("h", "m")}
        ),
        follow_redirects=True,
    )
    assert "换交成功" in resp.text
    rows = _parse_csv(client.get("/versions/2/download").data)
    assert len(rows) == 7 and rows[0]["tree_id"] == "ok-0"
