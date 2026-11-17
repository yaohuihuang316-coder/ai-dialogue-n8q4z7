# -*- coding: utf-8 -*-
"""
任务 1.2 · 统一数据模型与解析器
==============================
用法：
  "$PY" t12_parse.py --source wildchat          # 逐分片解析 WildChat → data/interim/wildchat/
  "$PY" t12_parse.py --source search_arena      # 解析 SA → data/interim/search_arena/
  "$PY" t12_parse.py --source wildchat --only 0 # 只解析第 0 号分片（开发/比对用）

统一中间格式（parquet, zstd 压缩，两来源同 schema）：

  conv_id          string   wc_<hash> | sa_<hex>_a/_b（SA 按模型侧两行展开）
  source           string   wildchat | search_arena
  model            string
  language         string   会话级语言代码/名称
  script           string   zh-Hans / zh-Hant / null（本脚本置 null，t13 填充）
  n_turns          int32    消息条数（user+assistant）
  n_user_turns     int32    用户轮数
  ts_start/ts_end  ts[us]   首末消息时间；SA 只有会话级时间则两值相同
  turns            list<struct{turn_id:int32, role:string, content:string, ts:timestamp[us]}>
  flag_dedup       bool     本脚本恒 False，t13 置位
  flag_redacted    bool     WC=redacted；SA 恒 False（官方已 DLP 清理）
  flag_toxic       bool     WC=toxic（本版本应恒 False）；SA 恒 False
  meta             string   JSON，来源特有字段（见下）

  WC meta: {country, state, hashed_ip, accept_language}
  SA meta: {conv_id_orig, side, winner, judge, primary_intent, secondary_intent,
            languages[], client_country, citation_format, n_citations,
            citations_per_turn[], n_llm_trace}

产出：data/interim/<source>/t12_<source>_NNNNN.parquet（逐原始分片一一对应）
      04_核验与决策/t12_比对样本_<source>.json（50 段原始 vs 中间格式对照，人工核验用）
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_config as C  # noqa: E402

import pyarrow as pa
import pyarrow.parquet as pq

INTERIM_SCHEMA = pa.schema([
    ("conv_id", pa.string()),
    ("source", pa.string()),
    ("model", pa.string()),
    ("language", pa.string()),
    ("script", pa.string()),
    ("n_turns", pa.int32()),
    ("n_user_turns", pa.int32()),
    ("ts_start", pa.timestamp("us")),
    ("ts_end", pa.timestamp("us")),
    ("turns", pa.list_(pa.struct([
        ("turn_id", pa.int32()), ("role", pa.string()),
        ("content", pa.string()), ("ts", pa.timestamp("us"))]))),
    ("flag_dedup", pa.bool_()),
    ("flag_redacted", pa.bool_()),
    ("flag_toxic", pa.bool_()),
    ("meta", pa.string()),
])


def _ts(v):
    """parquet 读出的 datetime / int / None → 一律转 datetime（或 None）"""
    return v if hasattr(v, "year") else None


# ---------------------------------------------------------------- WildChat
def parse_wildchat_row(r):
    conv = r["conversation"] or []
    turns = []
    for i, el in enumerate(conv):
        ts = _ts(el.get("timestamp"))
        turns.append({"turn_id": i, "role": el.get("role") or "",
                      "content": el.get("content") or "", "ts": ts})
    n_user = sum(1 for t in turns if t["role"] == "user")
    sess_ts = _ts(r.get("timestamp"))
    ts_list = [t["ts"] for t in turns if t["ts"] is not None]
    ts_start = min(ts_list) if ts_list else sess_ts
    ts_end = max(ts_list) if ts_list else sess_ts
    hdr = r.get("header") or {}
    meta = {"country": r.get("country"), "state": r.get("state"),
            "hashed_ip": r.get("hashed_ip"),
            "accept_language": (hdr.get("accept-language") if isinstance(hdr, dict)
                                else None)}
    return {"conv_id": f"wc_{r['conversation_hash']}", "source": "wildchat",
            "model": r.get("model") or "", "language": r.get("language") or "",
            "script": None, "n_turns": len(turns), "n_user_turns": n_user,
            "ts_start": ts_start, "ts_end": ts_end, "turns": turns,
            "flag_dedup": False,
            "flag_redacted": bool(r.get("redacted")),
            "flag_toxic": bool(r.get("toxic")),
            "meta": json.dumps(meta, ensure_ascii=False)}


def run_wildchat(only=None, batch=10000):
    out_dir = C.INTERIM_DIR / "wildchat"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((C.WILDCHAT_DIR / "data").glob("train-*.parquet"))
    if not files:
        raise SystemExit("[FATAL] WildChat 原始分片未就绪")
    if only is not None:
        files = [files[only]]
    total = 0
    for fi, f in enumerate(files):
        idx = f.stem.split("-")[1]
        out = out_dir / f"t12_wildchat_{idx}.parquet"
        if out.exists():
            C.rt(f"[{fi+1}/{len(files)}] {out.name} 已存在，跳过")
            total += pq.read_metadata(out).num_rows
            continue
        writer = pq.ParquetWriter(out, INTERIM_SCHEMA, compression="zstd")
        n_this = 0
        for batch_tbl in pq.ParquetFile(f).iter_batches(
                batch_size=batch,
                columns=["conversation_hash", "model", "timestamp", "conversation",
                         "turn", "language", "toxic", "redacted", "state",
                         "country", "hashed_ip", "header"]):
            rows = [parse_wildchat_row(r)
                    for r in batch_tbl.to_pylist()]
            writer.write_table(pa.Table.from_pylist(rows, schema=INTERIM_SCHEMA))
            n_this += len(rows)
        writer.close()
        total += n_this
        C.rt(f"[{fi+1}/{len(files)}] {f.name} → {out.name}：{n_this:,} 段")
    C.rt(f"WildChat 解析完成，共 {total:,} 段")
    return out_dir


# ------------------------------------------------------------- Search Arena
def _sa_trace_stats(wst):
    """web_search_trace: [轮][引用对, ...]，引用对 = [标记, URL]"""
    if not wst:
        return None, None
    per_turn = [len(x) if isinstance(x, list) else 0 for x in wst]
    return sum(per_turn), per_turn


def parse_sa_row(r, side, row_idx=None):
    """row_idx：原始 parquet 行号 —— 会话级唯一键。

    重要（2026-10-02 实测）：
    - system_a/b_metadata.conv_id 是**每侧独立**的端点会话 ID，两侧永远不同，
      不能当会话键；会话键 = 原始行号
    - `turn` 字段 = 用户在 UI 实际发出的提问数；messages 只收录到最后一个
      完成的问答对。turn > messages 用户轮数时，多出的提问文本不可恢复
      （两侧 llm_trace 里也没有），记 n_dangling 作终止行为信号
    """
    msgs = r[f"messages_{side}"] or []
    turns = [{"turn_id": i, "role": (m.get("role") or ""),
              "content": (m.get("content") or ""), "ts": None}
             for i, m in enumerate(msgs)]
    n_user = sum(1 for t in turns if t["role"] == "user")
    md = r.get(f"system_{side}_metadata") or {}
    md_other = r.get(f"system_{'b' if side == 'a' else 'a'}_metadata") or {}
    n_cit, cit_per_turn = _sa_trace_stats(md.get("web_search_trace"))
    sess_ts = _ts(r.get("timestamp"))
    turn_field = r.get("turn")
    meta = {"conv_id_orig": md.get("conv_id"),
            "conv_id_other_side": md_other.get("conv_id"),
            "side": side, "row_idx": row_idx,
            "winner": r.get("winner"), "judge": r.get("judge"),
            "primary_intent": r.get("primary_intent"),
            "secondary_intent": r.get("secondary_intent"),
            "languages": r.get("languages") or [],
            "client_country": md.get("client_country"),
            "citation_format": md.get("citation_format_standardized"),
            "n_citations": n_cit, "citations_per_turn": cit_per_turn,
            "n_llm_trace": len(md.get("llm_trace") or []),
            "turn_field": turn_field,
            "n_dangling": (turn_field - n_user)
            if isinstance(turn_field, int) else None}
    return {"conv_id": f"sa_{row_idx:05d}_{side}" if row_idx is not None
            else f"sa_{md.get('conv_id')}_{side}",
            "source": "search_arena",
            "model": r.get(f"model_{side}") or "",
            "language": (r.get("languages") or [""])[0] or "",
            "script": None, "n_turns": len(turns), "n_user_turns": n_user,
            "ts_start": sess_ts, "ts_end": sess_ts, "turns": turns,
            "flag_dedup": False, "flag_redacted": False, "flag_toxic": False,
            "meta": json.dumps(meta, ensure_ascii=False)}


def run_search_arena(batch=2000):
    out_dir = C.INTERIM_DIR / "search_arena"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = sorted((C.SEARCH_ARENA_DIR / "data").glob("*.parquet"))[0]
    out = out_dir / "t12_search_arena_0000.parquet"
    writer = pq.ParquetWriter(out, INTERIM_SCHEMA, compression="zstd")
    n = 0
    for bt in pq.ParquetFile(f).iter_batches(batch_size=batch):
        buf = []
        for row_idx, r in enumerate(bt.to_pylist(), start=n // 2):
            for side in ("a", "b"):
                buf.append(parse_sa_row(r, side, row_idx))
        writer.write_table(pa.Table.from_pylist(buf, schema=INTERIM_SCHEMA))
        n += len(buf)
    writer.close()
    C.rt(f"Search Arena 解析完成：{n:,} 段（= 2 × 会话数）→ {out}")
    return out_dir


# ------------------------------------------------------------- 比对样本抽取
def sample_compare(source, n=50, seed=42):
    """随机抽 n 段，把『原始行』与『中间格式行』并排放 JSON，供人工逐字段比对"""
    out = C.VERIFY_DIR / f"t12_比对样本_{source}.json"
    rnd = random.Random(seed)
    if source == "wildchat":
        raw_files = sorted((C.WILDCHAT_DIR / "data").glob("train-*.parquet"))
        idx = rnd.randrange(len(raw_files))
        tbl = pq.read_table(raw_files[idx]).slice(0, 2000).to_pylist()
        pairs = []
        for r in tbl:
            if rnd.random() < 0.05 and len(pairs) < n:
                pairs.append({"原始": {k: r[k] for k in
                                      ("conversation_hash", "model", "timestamp",
                                       "turn", "language", "country", "redacted",
                                       "toxic", "conversation")},
                              "中间格式": parse_wildchat_row(r)})
    else:
        f = sorted((C.SEARCH_ARENA_DIR / "data").glob("*.parquet"))[0]
        tbl = pq.read_table(f).slice(0, 2000).to_pylist()
        pairs = []
        for row_idx, r in enumerate(tbl):
            if rnd.random() < 0.05 and len(pairs) < n:
                pairs.append({"原始": {k: r[k] for k in
                                      ("model_a", "model_b", "winner",
                                       "turn", "timestamp", "messages_a",
                                       "primary_intent", "languages")},
                              "中间格式": parse_sa_row(r, "a", row_idx)})
    out.write_text(json.dumps(pairs[:n], ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    C.rt(f"比对样本 {len(pairs[:n])} 段 → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wildchat", "search_arena"], required=True)
    ap.add_argument("--only", type=int, default=None,
                    help="只解析该序号的原始分片（开发用）")
    ap.add_argument("--compare", action="store_true", help="只生成比对样本")
    a = ap.parse_args()
    if a.source == "wildchat":
        if a.compare:
            sample_compare("wildchat")
        else:
            run_wildchat(only=a.only)
            sample_compare("wildchat")
    else:
        if a.compare:
            sample_compare("search_arena")
        else:
            run_search_arena()
            sample_compare("search_arena")


if __name__ == "__main__":
    main()
