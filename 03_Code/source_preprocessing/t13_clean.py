# -*- coding: utf-8 -*-
"""
任务 1.3 · 清洗管线
==================
用法：
  "$PY" t13_clean.py --source wildchat      # 全量清洗 WC（流式，内存友好）
  "$PY" t13_clean.py --source search_arena
  "$PY" t13_clean.py --source wildchat --max-files 2   # 试跑前 2 个分片

动作（对应实施报告任务 1.3 五项）：
  1. conv_id（= conversation_hash）精确去重 → 重复 ID 清单表（保留首见）
  2. toxic 残留统计（本版本应已滤，只报数不删行）
  3. redacted 沿用中间层 flag_redacted（保留不删）
  4. 中文行复核：fastText 复验首条用户消息 + OpenCC 判简繁
  5. 字段缺失率统计

产出（侧表模式，不重写中间层）：
  data/interim/<source>/t13_dupids.parquet        重复 conv_id 清单
  data/interim/<source>/t13_zh_flags.parquet      中文行复核结果
  tmp/t13_清洗报告_<source>.md                     清洗报告（正式版并入审计报告 v1）
冻结语料库（t15）时合并：flag_dedup = conv_id ∈ dupids；script 直接并入。
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_config as C  # noqa: E402

import pyarrow as pa
import pyarrow.parquet as pq

# WC language 字段中文取值集合（分片落地后实测校准；先按已知可能值放全）
ZH_VALUES = {"chinese", "zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant",
             "chinese (simplified)", "chinese (traditional)"}


def first_user_text(turns):
    for t in turns or []:
        if t["role"] == "user":
            return t["content"] or ""
    return ""


def cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(1, len(s))


class LID:
    """语言识别：优先 lid176 ONNX 版（等价官方 lid.176.bin，见 lid_onnx.py 头注）；
    模型文件缺失时降级为 CJK 字符比例启发"""

    def __init__(self):
        self.m = None
        try:
            from lid_onnx import Lid176
            self.m = Lid176()
        except Exception as e:  # noqa: BLE001
            C.rt(f"LID ONNX 初始化失败（{e}）→ 降级启发式")

    def predict(self, text: str):
        if not text:
            return None, 0.0
        if self.m is not None:
            return self.m.predict(text)
        r = cjk_ratio(text)  # 降级
        return ("zh" if r > 0.3 else "other"), r


_T2S = None


def detect_script(text: str):
    """OpenCC 判简繁：繁→简转换后显著变化 → zh-Hant，否则 zh-Hans"""
    global _T2S
    if not text or cjk_ratio(text) < 0.15:
        return None
    if _T2S is None:
        from opencc import OpenCC
        _T2S = OpenCC("t2s")
    s = text[:3000]
    converted = _T2S.convert(s)
    diff = sum(1 for a, b in zip(s, converted) if a != b)
    return "zh-Hant" if diff / max(1, len(s)) > 0.02 else "zh-Hans"


def run(source: str, max_files=None):
    in_dir = C.INTERIM_DIR / source
    files = sorted(in_dir.glob("t12_*.parquet"))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"[FATAL] {in_dir} 无中间层分片，先跑 t12")

    lid = LID()
    C.rt(f"LID 模型：{'已加载' if lid.m is not None else '未找到 → 降级启发式'}")

    # ---- 阶段 1：流式扫描 —— 去重集 / 缺失率 / 中文行全局行号 ----
    C.rt(f"[1/2] 扫描 {len(files)} 个分片")
    seen = set()
    dup_ids = set()
    zh_rows = {}            # global_ri -> (file_i, conv_id)
    n_total = n_toxic = n_redacted = 0
    lang_counter = Counter()
    missing = Counter()
    g = 0                   # 全局行号
    for fi, f in enumerate(files):
        for batch in pq.ParquetFile(f).iter_batches(
                batch_size=50000,
                columns=["conv_id", "language", "flag_toxic",
                         "flag_redacted", "model", "ts_start"]):
            ids = batch["conv_id"].to_pylist()
            langs = batch["language"].to_pylist()
            n = len(ids)
            n_total += n
            n_toxic += sum(batch["flag_toxic"].to_pylist())
            n_redacted += sum(batch["flag_redacted"].to_pylist())
            lang_counter.update(langs)
            missing["model 空"] += sum(1 for m in batch["model"].to_pylist() if not m)
            missing["ts_start 空"] += sum(
                1 for t in batch["ts_start"].to_pylist() if t is None)
            for ri in range(n):
                cid = ids[ri]
                if cid in seen:
                    dup_ids.add(cid)
                else:
                    seen.add(cid)
                if (langs[ri] or "").strip().lower() in ZH_VALUES:
                    zh_rows[g + ri] = (fi, cid)
            g += n
        C.rt(f"  分片 {fi+1}/{len(files)} 完成（累计 {g:,} 行）")

    C.rt(f"总行 {n_total:,} | 唯一 {len(seen):,} | 重复 "
         f"{n_total - len(seen):,}（{(n_total-len(seen))/max(1,n_total):.2%}）"
         f" | toxic {n_toxic:,} | redacted {n_redacted:,} | 中文行 {len(zh_rows):,}")

    # 重复 ID 表
    pq.write_table(pa.table({"conv_id": pa.array(sorted(dup_ids), pa.string())}),
                   in_dir / "t13_dupids.parquet", compression="zstd")

    # ---- 阶段 2：中文行复核（按全局行号流式取 turns） ----
    C.rt(f"[2/2] 中文行 LID+简繁复核 {len(zh_rows):,} 行")
    side_schema = pa.schema([
        ("conv_id", pa.string()), ("lid_pred", pa.string()),
        ("lid_conf", pa.float32()), ("lid_agree", pa.bool_()),
        ("script", pa.string()), ("first_user_len", pa.int32()),
        ("cjk_ratio", pa.float32())])
    out_rows = []
    g = 0
    for fi, f in enumerate(files):
        for batch in pq.ParquetFile(f).iter_batches(
                batch_size=20000, columns=["conv_id", "turns"]):
            ids = batch["conv_id"].to_pylist()
            hit = [ri for ri in range(len(ids)) if (g + ri) in zh_rows]
            for ri in hit:
                cid = ids[ri]
                fu = first_user_text(batch["turns"][ri].as_py())
                pred, conf = lid.predict(fu)
                agree = (pred or "").startswith("zh") or pred == "chinese"
                script = detect_script(fu) if agree else None
                out_rows.append({"conv_id": cid, "lid_pred": pred,
                                 "lid_conf": conf, "lid_agree": agree,
                                 "script": script, "first_user_len": len(fu),
                                 "cjk_ratio": round(cjk_ratio(fu), 4)})
            g += len(ids)
        if out_rows and (fi + 1) % 10 == 0:
            pq.write_table(pa.Table.from_pylist(out_rows, schema=side_schema),
                           in_dir / "t13_zh_flags.parquet",
                           compression="zstd")  # 先写一版，最终覆盖
            C.rt(f"  分片 {fi+1} 累计中文复核 {len(out_rows):,}（中间落盘）")
    if out_rows:
        pq.write_table(pa.Table.from_pylist(out_rows, schema=side_schema),
                       in_dir / "t13_zh_flags.parquet", compression="zstd")

    # ---- 清洗报告（tmp 版） ----
    rep = C.TMP_DIR / f"t13_清洗报告_{source}.md"
    zh_agree_n = sum(1 for r in out_rows if r["lid_agree"])
    rep.write_text(f"""# 清洗报告（{source} · tmp 版，正式版并入审计报告 v1）

生成：t13_clean.py | 分片 {len(files)} 个 | {('试跑：前 ' + str(max_files) + ' 个分片') if max_files else '全量'}

| 项 | 数值 |
|---|---|
| 总行数 | {n_total:,} |
| 唯一 conv_id | {len(seen):,} |
| 重复行（flag_dedup 置位） | {n_total - len(seen):,}（{(n_total-len(seen))/max(1,n_total):.2%}） |
| toxic 残留 | {n_toxic:,} |
| redacted 标记（保留） | {n_redacted:,} |
| 中文行 | {len(zh_rows):,} |
| 中文 LID 复核一致 | {zh_agree_n:,}（{zh_agree_n/max(1,len(out_rows)):.1%}） |

字段缺失：{dict(missing)}

语言分布 Top20：{lang_counter.most_common(20)}
""", encoding="utf-8")
    C.rt(f"清洗报告 → {rep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wildchat", "search_arena"], required=True)
    ap.add_argument("--max-files", type=int, default=None)
    a = ap.parse_args()
    run(a.source, a.max_files)


if __name__ == "__main__":
    main()
