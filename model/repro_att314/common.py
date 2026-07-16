# -*- coding: utf-8 -*-
"""P1 attention 增量复现公共库。"""
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import csv
import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, MODEL_DIR)
from predict import build_window_feats, parse_dt

DATA = os.environ.get('P1_DATA', os.getcwd())
CACHE = os.environ.get('P1_CACHE', os.path.join(os.getcwd(), 'cache'))
NUM_METRICS, HOURS, DAYS = 4, 24, 14
THRESH = (0.2, 0.3, 0.4, 0.5)


def load_train_per_cell():
    per, order = {}, []
    with open(f'{DATA}/train_data.csv', encoding='utf-8-sig') as f:
        r = csv.reader(f); next(r)
        for line in r:
            t, cell = line[0], line[1]
            vals = [np.nan if x == 'NIL' else float(x) for x in line[2:6]]
            if cell not in per:
                per[cell] = []; order.append(cell)
            per[cell].append((t, vals))
    return per, order


def cell_windows(args):
    """一个小区的全部 stride-1 窗口 -> (X, bl, Y, meta)"""
    cell, rows = args
    arr = np.array([v for _, v in rows]); times = [t for t, _ in rows]
    n_days = len(times) // HOURS
    A = arr[:n_days * HOURS].reshape(n_days, HOURS, NUM_METRICS)
    t0 = parse_dt(times[0])
    out = []
    for st in range(0, n_days - DAYS):
        tgt = t0 + datetime.timedelta(days=st + DAYS)
        wd = tgt.weekday()
        X, bl = build_window_feats(A[st:st + DAYS], wd)
        out.append((st, tgt.strftime('%Y%m%d'), wd,
                    X.astype(np.float32), bl.astype(np.float32),
                    A[st + DAYS].astype(np.float32)))
    return cell, out


def get_split(cells_all):
    """与 predict.py 完全一致: RandomState(0) 从 sorted 小区抽 1/4 为验证"""
    cells_sorted = sorted(cells_all)
    val = set(np.random.RandomState(0).choice(
        cells_sorted, size=len(cells_sorted) // 4, replace=False))
    return val


def load_cache():
    z = np.load(f'{CACHE}/train_windows.npz', allow_pickle=True)
    return (z['X'], z['BL'], z['Y'], z['cells'], z['dates'], z['wds'], z['starts'])


def judge_filter(Yflat):
    """判题两阶段过滤: 任一指标NIL剔除; 任一指标低于其5%分位剔除.
    Yflat: (T,4) 真实值. 返回 keep 布尔掩码."""
    valid = ~np.isnan(Yflat).any(axis=1)
    q = np.percentile(Yflat[valid], 5, axis=0)
    keep = valid.copy()
    keep[valid] &= (Yflat[valid] >= q).all(axis=1)
    return keep


def mapeauc(Yflat, Pflat, keep=None):
    """Yflat/Pflat: (T,4). 返回 MAPEAUC 与各阈值比例."""
    if keep is None:
        keep = judge_filter(Yflat)
    ape = np.abs(Yflat[keep] - Pflat[keep]) / Yflat[keep]
    m = ape.mean(axis=1)
    rs = [(m < t).mean() for t in THRESH]
    return float(np.mean(rs)), rs


def postprocess(logp, BL, mults, clamp=np.log(3.0)):
    """logp/BL: (...,24,4) -> 预测值(钳位+乘子)"""
    lp = np.clip(logp, BL - clamp, BL + clamp)
    return np.exp(lp) * np.asarray(mults)


def tune_mults(Yflat, Pflat_raw, keep, init=(0.895, 0.935, 0.935, 0.995),
               grid=None, rounds=3):
    """坐标搜索分指标乘子. Pflat_raw: 钳位后未乘乘子的预测 (T,4)."""
    if grid is None:
        grid = np.arange(0.82, 1.06, 0.005)
    mults = list(init)
    best, _ = mapeauc(Yflat, Pflat_raw * np.array(mults), keep)
    for _ in range(rounds):
        improved = False
        for k in range(4):
            cur = mults[k]
            for g in grid:
                mults[k] = g
                s, _ = mapeauc(Yflat, Pflat_raw * np.array(mults), keep)
                if s > best + 1e-9:
                    best = s; cur = g; improved = True
            mults[k] = cur
        if not improved:
            break
    return best, mults
