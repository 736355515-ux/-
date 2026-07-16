# -*- coding: utf-8 -*-
"""单线程构建测试集窗口特征缓存。"""
import numpy as np, csv, os, time
import common as C
from predict import parse_dt, build_window_feats
import datetime

t0 = time.time()
groups = []
with open(f'{C.DATA}/test_data.csv', encoding='utf-8-sig') as f:
    r = csv.reader(f); next(r)
    buf = []
    for line in r:
        buf.append(line)
        if len(buf) == 14 * 24:
            cell = buf[0][1]; t0g = parse_dt(buf[0][0])
            arr = np.empty((14 * 24, 4))
            for i, row in enumerate(buf):
                for k in range(4):
                    v = row[2 + k]
                    arr[i, k] = np.nan if v == 'NIL' else float(v)
            groups.append((cell, t0g, arr.reshape(14, 24, 4)))
            buf = []
print('groups:', len(groups))

def build(g):
    cell, t0g, hist = g
    first = t0g + datetime.timedelta(days=14)
    X, bl = build_window_feats(hist, first.weekday())
    return cell, first.strftime('%Y/%m/%d %H:%M'), X.astype(np.float32), bl.astype(np.float32)

res = [build(g) for g in groups]
np.savez_compressed(f'{C.CACHE}/test_windows.npz',
                    X=np.stack([r[2] for r in res]),
                    BL=np.stack([r[3] for r in res]),
                    cells=np.array([r[0] for r in res]),
                    firsts=np.array([r[1] for r in res]))
print('saved', time.time() - t0)
