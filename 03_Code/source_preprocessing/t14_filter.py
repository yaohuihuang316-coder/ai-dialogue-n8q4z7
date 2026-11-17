# -*- coding: utf-8 -*-
"""
任务 1.4 · 信息获取类过滤器（方案二：规则 + 轻量分类器级联）
==========================================================
子命令流（按序执行）：
  1. rules           规则层全量扫描 → t14_ruleverdict.parquet（高置信排除 + 候选集）
  2. export-labels   分层抽样导出粗标文件（CSV，含建议标签，人工/复核者填 label 列）
  3. train           用填好的标注训练 fastText 三分类（in/out/bd）→ runtime/models/
  4. apply           级联应用全量 → t14_filter_flags.parquet（最终 is_info_seek）
  5. export-verify   500 段核验抽样导出（决策门 A 证据材料）
  6. verify-report   读回人工核验 → 精确率 + 混淆矩阵（≥90% 才过决策门 A）

操作性定义 v1（判定表见实施报告任务 1.4；核验时允许微调一次并记版本）：
  排除：代码生成与调试 / 翻译 / 纯创作生成 / 闲聊陪伴 / 格式改写
  纳入：事实与解释查询 / 求解与建议寻求 / 信息搜寻表达
  边界：学习辅导、带查询的混合任务 → 暂纳 + 标记 bd
"""
import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_config as C  # noqa: E402

import pyarrow as pa
import pyarrow.parquet as pq

FT_BIN = None  # fasttext python 模块（训练用）

# ------------------------------------------------------------------ 规则层
RE_CODE_FENCE = re.compile(r"```|~~~")
RE_CODE_STRONG = re.compile(
    r"(Traceback \(most recent call last\)|^\s*(def |class |import |from \S+ import "
    r"|public static void|function |const |let |var |#include|</?\w+>|SELECT .* FROM)"
    r"|npm install|pip install|报错|编译错误|段错误|syntax error|Exception in thread)",
    re.I | re.M)
RE_TRANSLATE = re.compile(
    r"(翻译(成|一下|为)|translate (this |the following )?(text |passage )?"
    r"(in)?to | Übersetze|traduce|번역|中译英|英译中|日语翻译"
    r"|переведи|traduis|翻訳し(て|てください)|dịch sang|çevir|ترجم|translate chinese|翻译中文)", re.I)
RE_CREATIVE = re.compile(
    r"(写(一(篇|首|个|则)|帮我写|个)(故事|小说|诗|词|歌|文案|剧本|段子|随笔)|"
    r"write (a |an )?(story|poem|song|novel|essay|screenplay|lyrics|fiction)|"
    r"compose (a |an )?(poem|song|story)|角色扮演|role-?play|续写|扮演|你扮演|"
    r"pretend to be|imagine you are|草拟一份情书|起个名字|取个名字|brand name ideas"
    # 多语言生成动词（2026-10-02 二次复核误纳诊断后补充；保守的"给我生成X"模式）
    r"|만들어\s*줘|캐릭터\s*만들|작성해\s*줘|이미지\s*만들|그려\s*줘"
    r"|キャラクター.{0,6}(作成|生成|描いて)|(小説|詩|物語)を書いて"
    r"|(crea|crear)\s+(un|una)?\s*(prompt|personaje|historia|poema|imagen|cuento|guion)"
    r"|genera\s+(un|una)?\s*(prompt|historia|imagen|poema)"
    r"|écris\s+(un|une)?\s*(histoire|poème)|rédige\s+(un|une)?\s*(texte|article)"
    r"|schreibe\s+(ein|eine)?\s*(Geschichte|Gedicht)|erstelle\s+(ein|eine)?\s*(Profil|Text)"
    r"|напиши( мне)?( историю| стих| текст| эссе| письмо| статью)|сочини"
    r"|اكتب( لي)?( قصة| قصيدة| مقالا)|أنشئ( لي)?( شخصية| قصة)"
    r"|viết( giúp)?( một)?( bài văn| câu chuyện| thơ| đoạn văn)"
    r"|escreva\s+(um|uma)?\s*(história|poema|texto)|cria\s+(um|uma)?\s*(história|personagem)"
    r"|bana\s+bir\s+(hikaye|şiir|metin)\s+yaz)", re.I)
RE_CODE_STRONG = re.compile(
    r"(Traceback \(most recent call last\)|^\s*(def |class |import |from \S+ import "
    r"|public static void|function |const |let |var |#include|</?\w+>|SELECT .* FROM)"
    r"|npm install|pip install|报错|编译错误|段错误|syntax error|Exception in thread)",
    re.I | re.M)
RE_FORMAT = re.compile(
    r"(帮我)?(总结|概括|摘要|润色|改写|重写|扩写|缩写|校对|纠错|润饰)(一下|这篇|这段|这个)?"
    r"|(summarize|summaries|polish|proofread|paraphrase|rewrite|reword|rephrase)"
    r"( this| the following| this text)?( text| passage| article| essay| paragraph)?"
    r"|改进这段|improve this (writing|paragraph|text)|使(更|更加)流畅", re.I)

# v1.1 手术刀清单（2026-10-03 终验 58 误纳裁决后标定；硬约束=对 352 个裁决口径真纳入段零碰撞，
# 模拟精确率 94.9%。钝模式（演讲稿/营销方案/字数页数泛匹配等）已实测倒退并撤除，勿再加回。）
RE_SURGICAL_V11 = [re.compile(x, re.I | re.M) for x in (
    r"responde como (un|una)", r"actúa como", r"QUERO UMA FALA",
    r"(propose|proposez|propón|proposer)\s+(un|une)?\s*titre",
    r"other\s+\d+\s+titles", r"俳句を作って", r"\bcmd\b.{0,20}\bbat\b",
    r"GUI[：:].{0,40}(创建|窗口)", r"用.{0,10}pyqt.{0,20}编写",
    r"构建一个\s*json\s*结构", r"字数[：:]?\s*\d+",
    r"\d+\s*字(左右|以内)", r"不超过\s*\d+\s*字", r"on \d+ pages",
    r"resume (la siguiente|este|esta)", r"mettre (ce|cette) (message|texte)",
    r"ubah teks menjadi", r"perbaiki (judulnya|teksnya)",
    r"buat lebih detail", r"换种说法", r"σενάριο|σεναριο",
    r"Haga una publicación", r"^script (about|for)",
    r"mi scrivi una frase", r"apa yang salah dengan judul",
    r"اريد كلام لمعلمتي", r"apakah aku cantik",
    r"^play .{2,30} by .{2,30}$", r"^(do you |are you )?(love|like) (me|ù|u)[?？.!. ]*$",
    r"用爽文小说", r"provide an idea",
    r"رد على رسالة حسب معلومات", r"Identify the core topics that someone needs to learn",
)]
RE_INFO_POS = re.compile(
    r"(什么|为什么|怎么|怎样|如何|哪些|哪个|是否|多少|何时|哪里|请问|帮我(查|找|搜)|推荐|建议|区别|原因|影响|"
    r"what|why|how|when|where|which|who|whom|whose|is there|are there|does|do |did |can |could |"
    r"explain|recommend|suggest|difference between|compare|帮我看看|我想(知道|了解|查询)|"
    r"tell me about|looking for|search for|find (me |out )?)", re.I)

VERDICT_SCHEMA = pa.schema([
    ("conv_id", pa.string()),
    ("verdict", pa.string()),      # keep_candidate | excl_code | excl_translate
                                   # | excl_creative | excl_chitchat | excl_format
    ("n_user_turns", pa.int32()),
    ("code_density", pa.float32()),
    ("code_marker", pa.bool_()),
    ("first2_chars", pa.int32()),
    ("has_question", pa.bool_()),
    ("is_zh", pa.bool_()),
    ("text12", pa.string()),       # 前两条用户消息拼接（分类器输入，截 1200 字符）
])


def user_turns(turns, k=2):
    out = []
    for t in turns or []:
        if t["role"] == "user":
            out.append(t["content"] or "")
            if len(out) >= k:
                break
    return out


def rule_one(n_user_turns, turns):
    us = user_turns(turns)
    text12 = "\n".join(us)
    joined = (text12 or "")[:4000]
    fences = len(RE_CODE_FENCE.findall(joined))
    code_density = min(1.0, fences * 40 / max(1, len(joined)))
    code_marker = bool(RE_CODE_STRONG.search(joined)) or code_density >= 0.15
    is_zh = sum(1 for ch in joined[:300] if "\u4e00" <= ch <= "\u9fff") > \
        0.15 * max(1, min(len(joined), 300))

    if code_marker or code_density >= 0.3:
        v = "excl_code"
    elif RE_TRANSLATE.search(joined) and len(joined) > 40:
        v = "excl_translate"
    elif RE_CREATIVE.search(joined):
        v = "excl_creative"
    elif (len(us) == 1 and RE_CHITCHAT.search(joined.strip())) or (
            # 裸短文本排除只对非 CJK 生效（二次复核发现 CJK 短真问题被误杀，如"红木家具公司有哪些"）
            len(joined.strip()) <= 12
            and not any("\u4e00" <= ch <= "\u9fff" for ch in joined)
            and "?" not in joined and "？" not in joined):
        v = "excl_chitchat"
    elif RE_FORMAT.search(joined) and not RE_INFO_POS.search(
            re.sub(RE_FORMAT, "", joined)):
        v = "excl_format"
    else:
        v = "keep_candidate"
    return {"verdict": v, "n_user_turns": n_user_turns,
            "code_density": round(code_density, 4), "code_marker": code_marker,
            "first2_chars": len(joined), "has_question": ("?" in joined or "？" in joined),
            "is_zh": is_zh, "text12": joined[:1200]}


def cmd_rules(source, max_files=None):
    in_dir = C.INTERIM_DIR / source
    files = sorted(in_dir.glob("t12_*.parquet"))
    if max_files:
        files = files[:max_files]
    out_dir = in_dir
    for fi, f in enumerate(files):
        out = out_dir / f"t14_ruleverdict_{f.stem.split('_')[-1]}.parquet"
        if out.exists():
            C.rt(f"[{fi+1}] {out.name} 已存在跳过"); continue
        rows = []
        for batch in pq.ParquetFile(f).iter_batches(
                batch_size=20000, columns=["conv_id", "n_user_turns", "turns"]):
            for r in batch.to_pylist():
                d = rule_one(r["n_user_turns"], r["turns"])
                d["conv_id"] = r["conv_id"]
                rows.append(d)
        pq.write_table(pa.Table.from_pylist(rows, schema=VERDICT_SCHEMA),
                       out, compression="zstd")
        C.rt(f"[{fi+1}/{len(files)}] {out.name}：{len(rows):,} 段")
    # 分布汇总
    cnt = Counter()
    for f in sorted(out_dir.glob("t14_ruleverdict_*.parquet")):
        t = pq.read_table(f, columns=["verdict"])
        cnt.update(t["verdict"].to_pylist())
    C.rt("规则层分布：" + json.dumps(cnt, ensure_ascii=False, indent=0))
    return cnt


# ------------------------------------------------------------- 标注导出
def cmd_export_labels(source, n, out_name, seed=42, zh_multi=False):
    """分层抽样：规则判定 × 语言(zh/en/其他) × 单轮/多轮 → CSV 待标。
    建议标签列 = 规则判定映射（keep_candidate→待判，其余→排除对应类）。"""
    import csv
    import random
    rnd = random.Random(seed)
    in_dir = C.INTERIM_DIR / source
    files = sorted(in_dir.glob("t14_ruleverdict_*.parquet"))
    rows = []
    for f in files:
        t = pq.read_table(f).to_pylist()
        if zh_multi:
            t = [r for r in t if r["is_zh"] and r["n_user_turns"] >= 2
                 and r["verdict"] == "keep_candidate"]
        rnd.shuffle(t)
        rows.extend(t[:math.ceil(n / len(files)) + 4])
    rnd.shuffle(rows)
    out = C.ANNOT_DIR / out_name
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "conv_id", "is_zh", "n_user_turns", "text12",
                    "建议标签", "label(填:in/out/bd)", "备注"])
        for i, r in enumerate(rows[:n]):
            sug = {"keep_candidate": "?", "excl_code": "out", "excl_translate": "out",
                   "excl_creative": "out", "excl_chitchat": "out",
                   "excl_format": "out"}.get(r["verdict"], "?")
            w.writerow([i, r["conv_id"], r["is_zh"], r["n_user_turns"],
                        r["text12"][:800], sug, "", ""])
    C.rt(f"标注文件 {min(n,len(rows))} 段 → {out}")
    return out


# ------------------------------------------------------------- 训练/应用
def _texts_and_labels(label_csv):
    import csv as _csv
    paths = [label_csv] if isinstance(label_csv, str) else label_csv
    X, y = [], []
    for _p in paths:
      with open(_p, encoding="utf-8-sig") as fh:
        for r in _csv.DictReader(fh):
            lab = (r.get("label(填:in/out/bd)") or "").strip()
            if lab in ("in", "out", "bd"):
                X.append(r["text12"].replace("\n", " "))
                y.append(f"__label__{lab}")
    return X, y


def cmd_train(label_csv, model="t14_clf"):
    import fasttext
    X, y = _texts_and_labels(label_csv)
    if len(X) < 50:
        raise SystemExit(f"[FATAL] 有效标注仅 {len(X)} 条（<50），先完成标注")
    cnt = Counter(y)
    C.rt(f"标注分布：{dict(cnt)}")
    import random
    import tempfile
    rnd = random.Random(7)
    idx = list(range(len(X))); rnd.shuffle(idx)
    cut = int(len(idx) * 0.9)
    tr = [f"{y[i]} {X[i]}" for i in idx[:cut]]
    va = [f"{y[i]} {X[i]}" for i in idx[cut:]]
    # fastText C++ 层无法打开含 CJK 的 Windows 路径，训练文件走 ASCII 临时目录
    tmp_tr = Path(tempfile.gettempdir()) / "t14_train.txt"
    tmp_va = Path(tempfile.gettempdir()) / "t14_valid.txt"
    tmp_tr.write_text("\n".join(tr), encoding="utf-8")
    tmp_va.write_text("\n".join(va), encoding="utf-8")
    m = fasttext.train_supervised(
        input=str(tmp_tr), epoch=40, lr=0.5, wordNgrams=2, minn=2, maxn=5,
        dim=100, minCount=2, verbose=2)
    n, pr, rec = m.test(str(tmp_va))
    C.rt(f"验证集 {n} 条 | P@1={pr:.3f}")
    import tempfile
    import shutil
    stage = Path(tempfile.gettempdir()) / "t14_clf_stage.bin"  # CJK 路径规避
    m.save_model(str(stage))
    out = C.PROJECT / "runtime" / "models" / model
    shutil.copy2(stage, str(out) + ".bin")
    stage.unlink()
    C.rt(f"分类器 → {out}.bin")
    return pr


def _load_clf(model="t14_clf"):
    """fastText 模型须先复制到 ASCII 路径再加载（CJK 路径规避）"""
    import shutil
    import tempfile
    import fasttext
    stage = Path(tempfile.gettempdir()) / "t14_clf_load.bin"
    shutil.copy2(str(C.PROJECT / "runtime" / "models" / f"{model}.bin"), stage)
    return fasttext.load_model(str(stage))


def cmd_apply(source, model="t14_clf", thresh=0.55):
    """级联：规则排除的直接 out；候选用分类器；低置信 → bd（暂纳+标记）。"""
    m = _load_clf(model)
    in_dir = C.INTERIM_DIR / source
    files = sorted(in_dir.glob("t14_ruleverdict_*.parquet"))
    out_schema = pa.schema([
        ("conv_id", pa.string()), ("is_info_seek", pa.bool_()),
        ("borderline", pa.bool_()), ("rule_verdict", pa.string()),
        ("clf_label", pa.string()), ("clf_prob", pa.float32())])
    all_rows = []
    for f in files:
        recs = pq.read_table(f).to_pylist()
        # 批量预测（比逐条快一个数量级），再线性合并
        cand_txt = [(r["text12"] or "").replace("\n", " ")[:1000]
                    for r in recs if r["verdict"] == "keep_candidate"]
        cand_txt = [t if t.strip() else " " for t in cand_txt]
        preds = m.predict(cand_txt) if cand_txt else ([], [])
        # 批量入参时返回 (labels 的二维列表, probs 的二维列表)，各取首元素
        if cand_txt and isinstance(preds[0][0], (list, tuple)):
            labels = [ls[0].replace("__label__", "") for ls in preds[0]]
            probs = [float(ps[0]) for ps in preds[1]]
        else:
            labels = [l.replace("__label__", "") for l in preds[0]]
            probs = [float(p) for p in preds[1]]
        ci = 0
        for r in recs:
            if r["verdict"] != "keep_candidate":
                all_rows.append({"conv_id": r["conv_id"], "is_info_seek": False,
                                 "borderline": False, "rule_verdict": r["verdict"],
                                 "clf_label": None, "clf_prob": None})
                continue
            txt = (r["text12"] or "").replace("\n", " ")[:1000]
            if not txt.strip():
                all_rows.append({"conv_id": r["conv_id"], "is_info_seek": False,
                                 "borderline": True,
                                 "rule_verdict": r["verdict"],
                                 "clf_label": None, "clf_prob": None})
            else:
                lab, p = labels[ci], probs[ci]
                if lab == "in" and p >= thresh:
                    isinfo, bd = True, False
                elif lab == "out" and p >= thresh:
                    isinfo, bd = False, False
                else:
                    isinfo, bd = True, True   # 边界暂纳，标记待核验
                all_rows.append({"conv_id": r["conv_id"], "is_info_seek": isinfo,
                                 "borderline": bd, "rule_verdict": r["verdict"],
                                 "clf_label": lab, "clf_prob": round(p, 4)})
            ci += 1
        C.rt(f"  {f.name} 完成（累计 {len(all_rows):,}）")
        if len(all_rows) > 2_000_000:  # 分批落盘防内存
            part = in_dir / f"t14_filter_flags_{len(list(in_dir.glob('t14_filter_flags_[0-9][0-9][0-9].parquet'))):03d}.parquet"
            pq.write_table(pa.Table.from_pylist(all_rows, schema=out_schema),
                           part, compression="zstd")
            all_rows = []
    if all_rows:
        part = in_dir / f"t14_filter_flags_{len(list(in_dir.glob('t14_filter_flags_[0-9][0-9][0-9].parquet'))):03d}.parquet"
        pq.write_table(pa.Table.from_pylist(all_rows, schema=out_schema),
                       part, compression="zstd")
    cnt = Counter()
    for p in sorted(in_dir.glob("t14_filter_flags_[0-9][0-9][0-9].parquet")):
        t = pq.read_table(p, columns=["is_info_seek", "borderline"])
        for a, b in zip(t["is_info_seek"].to_pylist(), t["borderline"].to_pylist()):
            cnt["纳入" if a and not b else ("边界暂纳" if a else "排除")] += 1
    total = sum(cnt.values())
    C.rt(f"过滤结果：{json.dumps(cnt, ensure_ascii=False)} | "
         f"保留率 {(cnt['纳入']+cnt['边界暂纳'])/max(1,total):.1%}")


# ------------------------------------------------------------- 核验
def cmd_export_verify(source, n=500, seed=2026):
    """从最终过滤结果分层抽 500 段（纳入 60% / 边界 20% / 排除 20%）导出核验表。"""
    import csv
    import random
    rnd = random.Random(seed)
    in_dir = C.INTERIM_DIR / source
    flags = []
    for p in sorted(in_dir.glob("t14_filter_flags_[0-9][0-9][0-9].parquet")):
        flags.extend(pq.read_table(p).to_pylist())
    texts = {}
    for f in sorted(in_dir.glob("t14_ruleverdict_*.parquet")):
        for r in pq.read_table(f, columns=["conv_id", "text12"]).to_pylist():
            texts[r["conv_id"]] = r["text12"]
    buckets = {"in": [], "bd": [], "out": []}
    for r in flags:
        k = "bd" if r["borderline"] else ("in" if r["is_info_seek"] else "out")
        buckets[k].append(r)
    quota = {"in": int(n * 0.6), "bd": int(n * 0.2), "out": n - int(n * 0.6) - int(n * 0.2)}
    sample = []
    for k, q in quota.items():
        rnd.shuffle(buckets[k])
        sample.extend(buckets[k][:q])
    rnd.shuffle(sample)
    out = C.ANNOT_DIR / "t14_核验表_500段.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "conv_id", "过滤器判定", "text12",
                    "人工判定(填:in/out/bd)", "备注"])
        for i, r in enumerate(sample):
            k = "边界暂纳" if r["borderline"] else ("纳入" if r["is_info_seek"] else "排除")
            w.writerow([i, r["conv_id"], k, (texts.get(r["conv_id"]) or "")[:800], "", ""])
    C.rt(f"核验表 {len(sample)} 段 → {out}")


def cmd_verify_report(verify_csv):
    """读回人工判定 → 过滤器精确率（纳入类的查准）+ 混淆矩阵。"""
    import csv as _csv
    tp = fp = fn = tn = 0
    conf = Counter()
    with open(verify_csv, encoding="utf-8-sig") as fh:
        for r in _csv.DictReader(fh):
            human = (r.get("人工判定(填:in/out/bd)") or "").strip()
            if human not in ("in", "out", "bd"):
                continue
            filt = r["过滤器判定"]
            pred_in = filt in ("纳入", "边界暂纳")
            # 人工 in/bd 都算真纳入（bd=边界暂纳计划内保留）
            true_in = human in ("in", "bd")
            conf[(filt, human)] += 1
            if pred_in and true_in: tp += 1
            elif pred_in and not true_in: fp += 1
            elif not pred_in and true_in: fn += 1
            else: tn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    lines = [f"# 过滤器核验报告（决策门 A 证据）",
             f"",
             f"- 有效核验：{tp+fp+fn+tn} 段",
             f"- **纳入精确率：{prec:.1%}**（门槛 ≥90%：{'✅ 通过' if prec >= 0.9 else '❌ 未达标'}）",
             f"- 纳入召回（核验样本内）：{rec:.1%}",
             f"- 混淆：TP={tp} FP={fp} FN={fn} TN={tn}",
             f"",
             f"## 逐格计数（过滤器判定 × 人工判定）",
             f""]
    for (f, h), c in sorted(conf.items()):
        lines.append(f"- {f} | 人工={h} : {c}")
    out = C.VERIFY_DIR / "t14_核验报告.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    C.rt(f"精确率 {prec:.1%} → {out}")
    return prec


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rules"); p.add_argument("--source", required=True)
    p.add_argument("--max-files", type=int, default=None)
    p = sub.add_parser("export-labels"); p.add_argument("--source", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", default="t14_粗标批次1.csv")
    p.add_argument("--zh-multi", action="store_true")
    p = sub.add_parser("train"); p.add_argument("--labels", nargs="+",
                                                default=[str(C.ANNOT_DIR / "t14_粗标批次1.csv")])
    p = sub.add_parser("apply"); p.add_argument("--source", required=True)
    p = sub.add_parser("export-verify"); p.add_argument("--source", required=True)
    p.add_argument("--n", type=int, default=500)
    p = sub.add_parser("verify-report"); p.add_argument("--csv",
                                                       default=str(C.ANNOT_DIR / "t14_核验表_500段.csv"))
    a = ap.parse_args()
    if a.cmd == "rules":
        cmd_rules(a.source, a.max_files)
    elif a.cmd == "export-labels":
        cmd_export_labels(a.source, a.n, a.out, zh_multi=a.zh_multi)
    elif a.cmd == "train":
        cmd_train(a.labels)
    elif a.cmd == "apply":
        cmd_apply(a.source)
    elif a.cmd == "export-verify":
        cmd_export_verify(a.source, a.n)
    elif a.cmd == "verify-report":
        cmd_verify_report(a.csv)


if __name__ == "__main__":
    main()
