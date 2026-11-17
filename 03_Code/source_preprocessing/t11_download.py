# -*- coding: utf-8 -*-
"""
任务 1.1 · 下载与完整性校验
==========================
用法（用项目内运行时执行）：
  PY = runtime/python/python.exe
  "$PY" t11_download.py --dataset sa      # Search Arena 24K（小，先跑通）
  "$PY" t11_download.py --dataset wc      # WildChat-4.8M（86 分片，长任务）
  "$PY" t11_download.py --dataset wc --workers 8

动作：
  1. 经 hf-mirror 镜像 snapshot_download 拉取全量仓库到 data/raw/<name>
  2. 用 parquet 元数据统计总行数（不读数据体，秒级）
  3. 行数对齐官方数（3,199,860 / 24,069）→ 不符即报错退出
  4. schema 快照存档到 01_数据台账/schema_<name>.json
  5. 文件清单 + 下载记录写入 01_数据台账/（CSV + Markdown 台账）

产出（脚本名 = 产出名前缀）：t11_下载记录.csv、t11_文件清单.csv、数据台账.md
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# 嵌入式 Python 为隔离模式（._pth），需手动把脚本目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_config as C  # noqa: E402  (必须最先 import：设 HF_ENDPOINT/HF_HOME)


def sizeof_fmt(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def download(repo: str, local_dir: Path, max_workers: int, use_hf_transfer: bool) -> str:
    """下载仓库全量，返回快照 revision（commit sha）"""
    if use_hf_transfer:
        import hf_transfer  # noqa: F401
        import os
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download, HfApi

    t0 = time.time()
    C.rt(f"开始下载 {repo} → {local_dir}")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(local_dir),
        max_workers=max_workers,
    )
    sha = HfApi().dataset_info(repo_id=repo).sha
    C.rt(f"下载完成，耗时 {time.time()-t0:.0f}s | revision {sha[:12]}")
    return sha


def verify(local_dir: Path, expected_rows: int, name: str) -> dict:
    """parquet 元数据级校验：行数、分片数、schema 快照"""
    import pyarrow.parquet as pq

    files = sorted(local_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"[FATAL] {local_dir} 下没有 parquet 文件")

    total_rows = 0
    total_bytes = 0
    schema = None
    per_file = []
    for f in files:
        meta = pq.read_metadata(f)
        total_rows += meta.num_rows
        total_bytes += f.stat().st_size
        per_file.append({"file": f.name, "rows": meta.num_rows,
                         "bytes": f.stat().st_size})
        if schema is None:
            schema = pq.read_schema(f)  # Arrow 逻辑类型，比 parquet 物理类型可读

    ok = (total_rows == expected_rows)
    C.rt(f"[{name}] 分片 {len(files)} 个 | 总行数 {total_rows:,} "
         f"(期望 {expected_rows:,}) | {sizeof_fmt(total_bytes)} | "
         f"{'✅ 一致' if ok else '❌ 不一致'}")

    # schema 快照
    schema_json = [{"name": fl.name, "type": str(fl.type)}
                   for fl in schema]
    out = C.LEDGER_DIR / f"schema_{name}.json"
    out.write_text(json.dumps(schema_json, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    return {"rows": total_rows, "ok": ok, "files": per_file,
            "bytes": total_bytes, "schema_path": str(out)}


def sample_heads(local_dir: Path, name: str, n: int = 20) -> Path:
    """抽前 n 段存 JSON 供人工目检（对照 HF 页面）"""
    import pyarrow.parquet as pq

    f = sorted(local_dir.rglob("*.parquet"))[0]
    tbl = pq.read_table(f).slice(0, n)
    out = C.LEDGER_DIR / f"t11_抽查样例_{name}.json"
    # 转成可序列化结构；binary/toxic 等按原样
    rows = tbl.to_pylist()
    for r in rows:
        r["_source_file"] = f.name
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    return out


def write_ledger(name: str, repo: str, rev: str, info: dict, sample_path: Path):
    """追加/更新 01_数据台账/ 下的记录"""
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lic = {"wildchat": "ODC-By（署名即可，仓库非 gated，无需勾选协议）",
           "search_arena": "prompts: CC BY 4.0；模型输出依提供方 ToS（仓库非 gated）"}[name]

    # CSV 下载记录（按数据集名 upsert，防重跑追加重复行）
    rec = C.LEDGER_DIR / "t11_下载记录.csv"
    existing = []
    if rec.exists():
        with rec.open(encoding="utf-8-sig") as fh:
            existing = [r for r in csv.DictReader(fh) if r["数据集"] != name]
    with rec.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "数据集", "repo", "revision", "下载完成时间", "总行数", "期望行数",
            "校验", "分片数", "总体积", "本地路径", "许可"])
        w.writeheader()
        for r in existing:
            w.writerow(r)
        w.writerow({"数据集": name, "repo": repo, "revision": rev,
                    "下载完成时间": ts, "总行数": info["rows"],
                    "期望行数": (C.WILDCHAT_EXPECTED_ROWS if name == "wildchat"
                                 else C.SEARCH_ARENA_EXPECTED_ROWS),
                    "校验": "PASS" if info["ok"] else "FAIL",
                    "分片数": len(info["files"]),
                    "总体积": sizeof_fmt(info["bytes"]),
                    "本地路径": str(C.RAW_DIR / {"wildchat": "wildchat-4.8m",
                                                 "search_arena": "search-arena-24k"}[name]),
                    "许可": lic})

    # 文件清单
    inv = C.LEDGER_DIR / "t11_文件清单.csv"
    new = not inv.exists()
    rows = [(name, r["file"], r["rows"], r["bytes"]) for r in info["files"]]
    with inv.open("a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["数据集", "文件", "行数", "字节数"])
        w.writerows(rows)

    # Markdown 台账（每次从 CSV 全量重渲染，保持两数据集都在）
    md = C.LEDGER_DIR / "数据台账.md"
    with rec.open(encoding="utf-8-sig") as fh:
        ledger_rows = list(csv.DictReader(fh))
    body = "\n".join(
        f"| {r['数据集']} | {r['repo']} | {r['revision'][:12]} | {r['下载完成时间']} "
        f"| {r['总行数']} / {r['期望行数']} | {r['校验']} | {r['分片数']} 片 / {r['总体积']} "
        f"| {r['本地路径']} | {r['许可']} |" for r in ledger_rows)
    md.write_text(f"""# 原始数据与许可台账

> 由 t11_download.py 自动维护；人工补充请另加节，勿改自动节。
> 生成时间：{ts}

| 数据集 | repo | revision | 下载时间 | 行数（实测/期望） | 校验 | 分片/体积 | 存储路径 | 许可 |
|---|---|---|---|---|---|---|---|---|
{body}

- schema 快照：`01_数据台账/schema_*.json`
- 抽查样例（各 20 段，供与 HF 页面人工比对）：`01_数据台账/t11_抽查样例_*.json`
- 文件级清单：`01_数据台账/t11_文件清单.csv`
- WildChat 下载方式：t11_parallel_download.sh（hf-mirror 直链、锁定 revision、
  断点续传、PAR1 魔数校验；huggingface.co 本机不可达）

## 协议接受说明

两个仓库在 Hugging Face 上均为**非 gated** 仓库（API `gated: false`，
2026-10-02 实测），下载无需登录或勾选协议；许可以仓库声明为准（见上表）。
""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sa", "wc"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-hf-transfer", action="store_true",
                    help="关掉 hf_transfer（兼容性兜底）")
    ap.add_argument("--no-download", action="store_true",
                    help="跳过下载（文件已由 t11_parallel_download.sh 拉好），"
                         "只做行数校验 + schema 快照 + 台账")
    a = ap.parse_args()

    if a.dataset == "sa":
        repo, d, exp, name = (C.SEARCH_ARENA_REPO, C.SEARCH_ARENA_DIR,
                              C.SEARCH_ARENA_EXPECTED_ROWS, "search_arena")
    else:
        repo, d, exp, name = (C.WILDCHAT_REPO, C.WILDCHAT_DIR,
                              C.WILDCHAT_EXPECTED_ROWS, "wildchat")

    if a.no_download:
        # 文件已由 t11 / t11_parallel_download.sh 拉好的锁定快照
        rev = {"wildchat": "c827c6df8fcf008219ffaffa4d1dd77491099367",
               "search_arena": "037967b4bcc793ef73e3edf8220b2ff5d6e1a65a"}[name]
    else:
        rev = download(repo, d, a.workers, use_hf_transfer=not a.no_hf_transfer)
    info = verify(d, exp, name)
    sp = sample_heads(d, name)
    write_ledger(name, repo, rev, info, sp)
    C.rt(f"台账已写入 {C.LEDGER_DIR}")

    if not info["ok"]:
        sys.exit(2)  # 决策门硬要求：行数对不上即失败


if __name__ == "__main__":
    main()
