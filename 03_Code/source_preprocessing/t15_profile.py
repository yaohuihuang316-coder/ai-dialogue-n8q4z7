# -*- coding: utf-8 -*-
"""
任务 1.5 · WildChat 总体画像 + 中文漏斗（+ 任务 1.15 冻结语料库 v1.0）
====================================================================
用法：
  "$PY" t15_profile.py                # 画像 + 漏斗 + 汇总表（审计报告素材）
  "$PY" t15_profile.py --freeze      # 额外生成冻结语料库 v1.0（is_info_seek 层）

产出：
  03_审计报告/t15_五维分布.csv（语言/轮数/模型/月份/国家 各一表）
  03_审计报告/t15_中文漏斗.md（叙事决策依据）
  data/corpus/analysis_corpus_v1.0/  冻结快照（--freeze 时）
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_config as C  # noqa: E402

import pyarrow as pa
import pyarrow.parquet as pq


def iter_wc(columns, batch=50000):
    for f in sorted((C.INTERIM_DIR / "wildchat").glob("t12_wildchat_*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=batch, columns=columns):
            yield from b.to_pylist()


def month_of(ts):
    return f"{ts.year}-{ts.month:02d}" if ts else "NA"


def main(freeze=False):
    zh_flags = {}
    fz = C.INTERIM_DIR / "wildchat" / "t13_zh_flags.parquet"
    if fz.exists():
        t = pq.read_table(fz).to_pylist()
        zh_flags = {r["conv_id"]: r for r in t}
    dup_ids = set()
    dq = C.INTERIM_DIR / "wildchat" / "t13_dupids.parquet"
    if dq.exists():
        dup_ids = set(pq.read_table(dq)["conv_id"].to_pylist())
    filt = {}
    for p in sorted((C.INTERIM_DIR / "wildchat").glob("t14_filter_flags_[0-9][0-9][0-9].parquet")):
        for r in pq.read_table(p).to_pylist():
            filt[r["conv_id"]] = r

    lang_c, turn_c, model_c, month_c, country_c = (Counter() for _ in range(5))
    zh_total = zh_multi = zh_info = zh_info_bd = 0
    zh_script_c, zh_country_c = Counter(), Counter()
    all_multi = all_info = total = 0
    C.rt("扫描中间层……")
    for r in iter_wc(["conv_id", "language", "n_user_turns", "model", "ts_start",
                      "meta"]):
        total += 1
        lang_c[r["language"] or "NA"] += 1
        turn_c[min(r["n_user_turns"], 10)] += 1
        model_c[r["model"] or "NA"] += 1
        month_c[month_of(r["ts_start"])] += 1
        m = json.loads(r["meta"] or "{}")
        country_c[m.get("country") or "NA"] += 1
        multi = r["n_user_turns"] >= 2
        if multi: all_multi += 1
        fi = filt.get(r["conv_id"])
        isinfo = fi and fi["is_info_seek"]
        if isinfo: all_info += 1
        is_zh = (r["language"] or "").lower() in {
            "chinese", "zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant"}
        if is_zh:
            zh_total += 1
            if multi:
                zh_multi += 1
                if isinfo:
                    zh_info += 1
                    if fi["borderline"]: zh_info_bd += 1
                    zf = zh_flags.get(r["conv_id"])
                    if zf:
                        zh_script_c[zf["script"] or "未定"] += 1
                        zh_country_c[m.get("country") or "NA"] += 1

    C.rt(f"总 {total:,} | 多轮 {all_multi:,} | 信息获取 {all_info:,}")
    C.rt(f"中文漏斗：总量 {zh_total:,} → 多轮 {zh_multi:,} → 信息获取 "
         f"{zh_info:,}（其中边界 {zh_info_bd:,}）")

    out = C.AUDIT_DIR / "t15_五维分布.json"
    out.write_text(json.dumps({
        "language_top30": lang_c.most_common(30),
        "user_turns_hist(min10)": sorted(turn_c.items()),
        "model": model_c.most_common(),
        "month": sorted(month_c.items()),
        "country_top30": country_c.most_common(30),
        "zh_script": dict(zh_script_c),
        "zh_country_top15": zh_country_c.most_common(15),
        "funnel": {"total": total, "multi": all_multi, "info": all_info,
                   "zh_total": zh_total, "zh_multi": zh_multi,
                   "zh_info": zh_info, "zh_info_bd": zh_info_bd,
                   "dup_ids": len(dup_ids)}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    C.rt(f"分布 JSON → {out}")

    md = C.AUDIT_DIR / "t15_中文漏斗.md"
    md.write_text(f"""# 中文子样本漏斗（叙事决策依据）

| 层 | 段数 | 占上一层 |
|---|---|---|
| WildChat 中文会话总量 | {zh_total:,} | — |
| ∧ 多轮（≥2 用户轮） | {zh_multi:,} | {zh_multi/max(1,zh_total):.1%} |
| ∧ 信息获取（过滤器纳入，含边界） | {zh_info:,} | {zh_info/max(1,zh_multi):.1%} |
|   其中边界暂纳 | {zh_info_bd:,} | — |

简繁：{dict(zh_script_c)}
国家 Top15：{zh_country_c.most_common(15)}

**叙事判定（对照实施报告任务 1.7 三选一）：**
- ≥15,000 → 叙事一；5,000–15,000 → 叙事二；<5,000 → 叙事三
- 实测 {zh_info:,} → **{"叙事一" if zh_info >= 15000 else ("叙事二" if zh_info >= 5000 else "叙事三")}**
""", encoding="utf-8")
    C.rt(f"漏斗 MD → {md}")

    if freeze:
        freeze_corpus()


def freeze_corpus():
    """冻结分析语料库 v1.0：中间层 + 全部侧表标志 → data/corpus/analysis_corpus_v1.0/"""
    out = C.CORPUS_DIR / "analysis_corpus_v1.0"
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    # 1. 中间层（硬链接拷贝，省空间）
    for src in C.INTERIM_DIR.rglob("t12_*.parquet"):
        dst = out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    # 2. 侧表
    for name in ("t13_dupids.parquet", "t13_zh_flags.parquet"):
        for src in (C.INTERIM_DIR / "wildchat").glob(name):
            shutil.copy2(src, out / f"wildchat_{name}")
        for src in (C.INTERIM_DIR / "search_arena").glob(name):
            shutil.copy2(src, out / f"search_arena_{name}")
    for src in (C.INTERIM_DIR / "wildchat").glob("t14_filter_flags_[0-9][0-9][0-9].parquet"):
        shutil.copy2(src, out / f"wildchat_{src.name}")
    for src in (C.INTERIM_DIR / "search_arena").glob("t14_filter_flags_[0-9][0-9][0-9].parquet"):
        shutil.copy2(src, out / f"search_arena_{src.name}")
    # 3. 版本记录
    rows = {}
    for f in sorted(out.glob("*.parquet")):
        rows[f.name] = pq.read_metadata(f).num_rows
    (out / "FREEZE.manifest.json").write_text(json.dumps({
        "version": "1.0", "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "files_rows": rows,
        "note": "冻结后任何改动须升版本号；生成脚本=02_代码/t12_parse.py "
                "t13_clean.py t14_filter.py t15_profile.py；上游 revision 见 01_数据台账"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    C.rt(f"语料库 v1.0 冻结 → {out}（{len(rows)} 文件）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    main(a.freeze)
