# -*- coding: utf-8 -*-
"""Combine FABLE base predictions with the att314-conv107 log-domain delta.

This script is part of the audit reproduction chain for
`att314_replace107`. It reads:
  1. a FABLE base `results.csv`,
  2. `test_windows.npz` containing the statistical baseline BL,
  3. exported log predictions for manual-attn314 and conv107,
and writes the final `results.csv`.
"""
import argparse
import csv
import math
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np


P = argparse.ArgumentParser()
P.add_argument("--base-results", required=True)
P.add_argument("--cache-root", default=os.environ.get("P1_CACHE", os.path.join(os.getcwd(), "cache")))
P.add_argument("--att-test", required=True)
P.add_argument("--conv-test", required=True)
P.add_argument("--out", required=True)
A = P.parse_args()

BASE_MULTS = np.array([0.905, 0.950, 0.950, 1.000], dtype=np.float64)
FINAL_MULTS = np.array([0.905, 0.950, 0.955, 1.000], dtype=np.float64)
DELTA_WEIGHT = 0.15
CLAMP = math.log(6.0)

with open(A.base_results, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))
header, data = rows[0], rows[1:]
if len(data) % 24 != 0:
    raise ValueError("base results row count must be a multiple of 24")

n_groups = len(data) // 24
base = np.zeros((n_groups, 24, 4), dtype=np.float64)
for idx, row in enumerate(data):
    g, h = divmod(idx, 24)
    base[g, h] = [float(x) for x in row[2:6]]

z = np.load(os.path.join(A.cache_root, "test_windows.npz"), allow_pickle=True)
BL = z["BL"].astype(np.float64)
att = np.load(A.att_test).astype(np.float64)
conv = np.load(A.conv_test).astype(np.float64)
if BL.shape != att.shape or BL.shape != conv.shape or BL.shape != base.shape:
    raise ValueError("shape mismatch among BL/base/att/conv arrays")

base_log = np.log(np.maximum(base / BASE_MULTS[None, None, :], 1e-6))
logp = base_log + DELTA_WEIGHT * (att - conv)
logp = np.clip(logp, BL - CLAMP, BL + CLAMP)
pred = np.maximum(np.exp(logp) * FINAL_MULTS[None, None, :], 1e-4)

with open(A.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for idx, row in enumerate(data):
        g, h = divmod(idx, 24)
        w.writerow(row[:2] + ["%.4f" % pred[g, h, k] for k in range(4)])

print("written", A.out, len(data), "rows")
