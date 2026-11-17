# -*- coding: utf-8 -*-
"""
研究四 · 共享环境配置
====================
所有阶段一脚本在 import huggingface_hub / pyarrow 之前先 `import env_config`，
即可获得统一的路径与网络配置。修改此处 = 修改全部脚本行为。

封装原则：一切（缓存、数据、代码）都在 研究四/ 文件夹内。
"""
import os
import sys
from pathlib import Path

# ---- 项目根与关键路径（按本文件位置反推，脚本放哪里都不怕） ----
HERE = Path(__file__).resolve().parent            # 第一阶段/02_代码
PHASE1 = HERE.parent                              # 第一阶段
PROJECT = PHASE1.parent                           # 研究四

DATA_DIR = PROJECT / "data"                       # 数据总目录（项目内）
RAW_DIR = DATA_DIR / "raw"                        # 原始快照
HF_CACHE = DATA_DIR / "hf_cache"                  # HF 下载缓存
INTERIM_DIR = DATA_DIR / "interim"                # 统一中间层 parquet
CORPUS_DIR = DATA_DIR / "corpus"                  # 冻结的分析语料库
TMP_DIR = HERE / "tmp"                            # 脚本中间文件（冻结前清空）

LEDGER_DIR = PHASE1 / "01_数据台账"
AUDIT_DIR = PHASE1 / "03_审计报告"
VERIFY_DIR = PHASE1 / "04_核验与决策"
ANNOT_DIR = VERIFY_DIR / "标注工作台"      # 所有标注/核验材料的集中地

for d in (DATA_DIR, RAW_DIR, HF_CACHE, INTERIM_DIR, CORPUS_DIR, TMP_DIR,
          LEDGER_DIR, AUDIT_DIR, VERIFY_DIR, ANNOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- 网络配置（必须在 import huggingface_hub 之前设置） ----
# 本机实测 huggingface.co 不可达，统一走 hf-mirror 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
# hf_transfer 由各脚本按需开启（大文件快，但断点续传粒度为一个分片）

# ---- 数据集锚点 ----
WILDCHAT_REPO = "allenai/WildChat-4.8M"
WILDCHAT_DIR = RAW_DIR / "wildchat-4.8m"
WILDCHAT_EXPECTED_ROWS = 3_199_860

SEARCH_ARENA_REPO = "amdsf/search-arena-24k"
SEARCH_ARENA_DIR = RAW_DIR / "search-arena-24k"
SEARCH_ARENA_EXPECTED_ROWS = 24_069

# ---- 工具 ----
def rt(msg: str) -> None:
    """带时间戳的进度输出"""
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
