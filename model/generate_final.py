#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由基线结果与三个只用训练集学得的NN成员生成决赛 results.csv。

该脚本不读取测试标签，不使用输入指纹，不在不同测试组之间聚合话务统计。
所有超参都由训练集的按小区留出验证确定。
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


P = argparse.ArgumentParser()
P.add_argument("--base-results", required=True)
P.add_argument("--cache-root", required=True)
P.add_argument("--out", required=True)
A = P.parse_args()

BASE_MULTS = np.array([0.905, 0.950, 0.950, 1.000], dtype=np.float64)
FINAL_MULTS = np.array([0.905, 0.950, 0.955, 1.000], dtype=np.float64)
ATT_DELTA_WEIGHT = 0.15
ATT_STRENGTH = 1.5
METRIC2_WEIGHT = 0.0215
CLAMP = math.log(6.0)
EPS = 1e-6

cache = Path(A.cache_root)
with Path(A.base_results).open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))
header, body = rows[0], rows[1:]
if len(body) % 24:
    raise ValueError("base result row count must be a multiple of 24")
n_groups = len(body) // 24
base = np.array([[float(x) for x in row[2:6]] for row in body], dtype=np.float64)
base = base.reshape(n_groups, 24, 4)

bl = np.load(cache / "test_windows.npz", allow_pickle=True)["BL"].astype(np.float64)
att = np.load(cache / "export_att314_test_best.npy").astype(np.float64)
conv107 = np.load(cache / "export_conv107_test_final.npy").astype(np.float64)
metric2 = np.load(cache / "export_conv_metric2_s314_test_best.npy").astype(np.float64)
for name, arr in (("BL", bl), ("att314", att), ("conv107", conv107), ("metric2", metric2)):
    if arr.shape != base.shape:
        raise ValueError(f"{name} shape {arr.shape} != base shape {base.shape}")

# 用 attention314 替换基线集成中权重0.15的 conv107。
base_log = np.log(np.maximum(base / BASE_MULTS[None, None, :], EPS))
att_log = np.clip(base_log + ATT_DELTA_WEIGHT * (att - conv107), bl - CLAMP, bl + CLAMP)
att_endpoint = np.exp(att_log) * FINAL_MULTS[None, None, :]
# 历史最优链先将 attention 替换端点写为4位小数 CSV，再做强度外推。
# 在这里显式保留同一边界，以便与审计分步链逐行复现。
att_endpoint = np.round(att_endpoint, 4)

# 在训练集留出验证选定的 attention 强度。
parent = np.exp(
    np.log(np.maximum(base, EPS))
    + ATT_STRENGTH * (np.log(np.maximum(att_endpoint, EPS)) - np.log(np.maximum(base, EPS)))
)

# 只对第3个指标加入小权重专家，其余指标不变。
metric2_endpoint = np.exp(np.clip(metric2, bl - CLAMP, bl + CLAMP)) * FINAL_MULTS[None, None, :]
pred = parent.copy()
pred[:, :, 2] = np.exp(
    (1.0 - METRIC2_WEIGHT) * np.log(np.maximum(parent[:, :, 2], EPS))
    + METRIC2_WEIGHT * np.log(np.maximum(metric2_endpoint[:, :, 2], EPS))
)
pred = np.maximum(pred * 1.03, 1e-4)

with Path(A.out).open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    flat = pred.reshape(-1, 4)
    for i, row in enumerate(body):
        w.writerow(row[:2] + [f"{flat[i, k]:.4f}" for k in range(4)])
print("written", A.out, len(body), "rows")
