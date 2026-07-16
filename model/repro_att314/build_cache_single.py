# -*- coding: utf-8 -*-
"""单线程构建全部训练窗口特征并缓存 npz (stride=1)。"""
import numpy as np, os, time
import common as C

os.makedirs(C.CACHE, exist_ok=True)
t0 = time.time()
per, order = C.load_train_per_cell()
print('cells:', len(order), 'elapsed', time.time() - t0)

results = [C.cell_windows((c, per[c])) for c in order]
print('windows built, elapsed', time.time() - t0)

Xs, BLs, Ys, cells, dates, wds, starts = [], [], [], [], [], [], []
for cell, wins in results:   # results 保持 order 顺序 (Pool.map 保序)
    for st, d, wd, X, bl, Y in wins:
        Xs.append(X); BLs.append(bl); Ys.append(Y)
        cells.append(cell); dates.append(d); wds.append(wd); starts.append(st)
np.savez_compressed(
    f'{C.CACHE}/train_windows.npz',
    X=np.stack(Xs), BL=np.stack(BLs), Y=np.stack(Ys),
    cells=np.array(cells), dates=np.array(dates),
    wds=np.array(wds, dtype=np.int8), starts=np.array(starts, dtype=np.int16))
print('saved', len(Xs), 'windows, elapsed', time.time() - t0)
