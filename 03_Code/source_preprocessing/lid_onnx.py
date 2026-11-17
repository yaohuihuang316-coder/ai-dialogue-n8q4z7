# -*- coding: utf-8 -*-
"""
lid176 ONNX 推理封装
====================
背景：官方 fastText Python 库（Windows 预编译 0.9.2）只支持模型格式 v11；
HF 上现成的 lid.176.bin 均为 v12 重存版，加载失败；本机无 MSVC 无法编译新版。
改用 TigreGotico/lid176-onnx（CC-BY-SA-3.0）的 ONNX 移植 + 原版 fastText
特征哈希，逻辑等价于 lid.176.bin（176 语种，含 __label__zh）。

依赖：onnxruntime、numpy；模型文件在 runtime/models/lid176_*
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lid176_hash import FastTextFeaturizer, HSCombiner  # noqa: E402

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "runtime" / "models"


class Lid176:
    def __init__(self):
        import onnxruntime as ort
        self.feat = FastTextFeaturizer.from_files(
            str(_MODEL_DIR / "lid176_vocab.txt"),
            str(_MODEL_DIR / "lid176_config.json"))
        self.hs = HSCombiner.from_file(str(_MODEL_DIR / "lid176_hs_tree.json"))
        self.labels = json.loads(
            (_MODEL_DIR / "lid176_labels.json").read_text(encoding="utf-8"))
        self.sess = ort.InferenceSession(
            str(_MODEL_DIR / "lid176_lid176.int8.onnx"),
            providers=["CPUExecutionProvider"])
        self._in_name = self.sess.get_inputs()[0].name

    def predict(self, text: str):
        """→ (label 去掉 __label__ 前缀, 概率)"""
        if not text or not text.strip():
            return None, 0.0
        ids = self.feat(text[:2000].replace("\n", " "))
        if len(ids) == 0:
            return None, 0.0
        node_probs = self.sess.run(None, {self._in_name: ids})[0]
        probs = self.hs(node_probs)
        i = int(np.argmax(probs))
        return self.labels[i].replace("__label__", ""), float(probs[i])


if __name__ == "__main__":
    lid = Lid176()
    tests = ["这是一个中文测试句子，我想查询一些信息",
             "The quick brown fox jumps over the lazy dog",
             "今天天氣很好，請問一下台北的景點",
             "こんにちは世界、教えてください",
             "Как дела, друг мой?",
             "Qu'est-ce que c'est que ça?"]
    for t in tests:
        lab, p = lid.predict(t)
        print(f"  {t[:22]:<26} → {lab} ({p:.3f})")
