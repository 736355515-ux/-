# -*- coding: utf-8 -*-
"""单线程构建原始 14 天窗口缓存 raw_windows.npy。"""
import os
import time

import numpy as np

import common as C

os.makedirs(C.CACHE, exist_ok=True)
t0 = time.time()
per, order = C.load_train_per_cell()
raws = []
for cell in order:
    rows = per[cell]
    arr = np.array([v for _, v in rows], dtype=np.float32)
    n_days = len(rows) // C.HOURS
    A = arr[:n_days * C.HOURS].reshape(n_days, C.HOURS, C.NUM_METRICS)
    for st in range(0, n_days - C.DAYS):
        raws.append(A[st:st + C.DAYS])
np.save(os.path.join(C.CACHE, "raw_windows.npy"), np.stack(raws).astype(np.float32))
print("saved", len(raws), "raw windows, elapsed", time.time() - t0)
