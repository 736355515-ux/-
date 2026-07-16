# -*- coding: utf-8 -*-
"""Export validation/test log predictions for P1 NN members.

This helper trains from cached training windows and predicts validation/test
windows. It does not use test labels or input fingerprints.
"""
import argparse
import copy
import csv
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn as nn

import common as C


P = argparse.ArgumentParser()
P.add_argument("--arch", choices=["conv", "manual_attn"], required=True)
P.add_argument("--tag", required=True)
P.add_argument("--epochs", type=int, default=30)
P.add_argument("--bs", type=int, default=4096)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--hid", type=int, default=512)
P.add_argument("--ch", type=int, default=64)
P.add_argument("--heads", type=int, default=4)
P.add_argument("--layers", type=int, default=2)
P.add_argument("--drop", type=float, default=0.1)
P.add_argument("--aug", type=float, default=0.08)
P.add_argument("--noise", type=float, default=0.04)
P.add_argument("--seed", type=int, default=107)
P.add_argument("--save-prefix", required=True)
P.add_argument("--train-hours", default="", help="Optional comma-separated hour filter, e.g. 0,1,2,3")
P.add_argument("--eval-hours", default="", help="Optional comma-separated hour filter for checkpoint selection")
P.add_argument("--train-metrics", default="", help="Optional comma-separated metric filter, e.g. 0,1")
P.add_argument("--eval-metrics", default="", help="Optional comma-separated metric filter for checkpoint selection")
P.add_argument("--data-root", default=os.environ.get("P1_DATA", os.getcwd()))
P.add_argument("--cache-root", default=os.environ.get("P1_CACHE", os.path.join(os.getcwd(), "cache")))
P.add_argument("--device", default="cuda")
A = P.parse_args()
C.DATA = A.data_root
C.CACHE = A.cache_root


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch = A.ch
        self.conv = nn.Sequential(
            nn.Conv1d(9, ch, 3, padding=1), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(43 + 4 + ch * 2, A.hid), nn.GELU(), nn.Dropout(A.drop),
            nn.Linear(A.hid, A.hid), nn.GELU(), nn.Dropout(A.drop),
            nn.Linear(A.hid, 256), nn.GELU(), nn.Linear(256, 1),
        )

    def forward(self, xe, raw, mk, koh):
        z = torch.cat([raw, mk, (koh.unsqueeze(1) * raw).sum(-1, keepdim=True)], -1)
        h = self.conv(z.transpose(1, 2))
        e = torch.cat([h.mean(-1), h[:, :, -3:].mean(-1)], -1)
        return self.head(torch.cat([xe, koh, e], -1)).squeeze(1)


class AttnBlock(nn.Module):
    def __init__(self, ch, heads, drop):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.dh = ch // heads
        self.n1 = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)
        self.drop = nn.Dropout(drop)
        self.n2 = nn.LayerNorm(ch)
        self.ff = nn.Sequential(
            nn.Linear(ch, ch * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(ch * 2, ch), nn.Dropout(drop),
        )

    def forward(self, x):
        b, t, c = x.shape
        h = self.n1(x)
        qkv = self.qkv(h).view(b, t, 3, self.heads, self.dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) * (self.dh ** -0.5)
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.drop(self.proj(out))
        x = x + self.ff(self.n2(x))
        return x


class ManualAttnNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch = A.ch
        self.inp = nn.Linear(9, ch)
        self.pos = nn.Parameter(torch.randn(1, 14, ch) * 0.02)
        self.blocks = nn.ModuleList([AttnBlock(ch, A.heads, A.drop) for _ in range(A.layers)])
        self.head = nn.Sequential(
            nn.Linear(43 + 4 + ch * 3, A.hid), nn.GELU(), nn.Dropout(A.drop),
            nn.Linear(A.hid, A.hid), nn.GELU(), nn.Dropout(A.drop),
            nn.Linear(A.hid, 256), nn.GELU(), nn.Linear(256, 1),
        )

    def forward(self, xe, raw, mk, koh):
        z = torch.cat([raw, mk, (koh.unsqueeze(1) * raw).sum(-1, keepdim=True)], -1)
        h = self.inp(z) + self.pos
        for blk in self.blocks:
            h = blk(h)
        e = torch.cat([h.mean(1), h[:, -3:].mean(1), h[:, -1]], -1)
        return self.head(torch.cat([xe, koh, e], -1)).squeeze(1)


def load_test_raw(data_root, n_groups):
    groups = []
    path = os.path.join(data_root, "test_data.csv")
    with open(path, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r)
        buf = []
        for line in r:
            buf.append(line)
            if len(buf) == 14 * 24:
                arr = np.empty((14 * 24, 4), dtype=np.float32)
                for i, row in enumerate(buf):
                    for k in range(4):
                        v = row[2 + k]
                        arr[i, k] = np.nan if v == "NIL" else float(v)
                groups.append(arr.reshape(14, 24, 4))
                buf = []
    assert len(groups) == n_groups
    return np.stack(groups, axis=0)


def make_rel(raw, bl):
    lh = np.log(np.maximum(raw, 1e-6))
    msk = np.isnan(raw)
    lh[msk] = 0.0
    rel = np.clip(lh - bl[:, None, :, :] * (~msk), -4, 4).astype(np.float32)
    return rel, msk.astype(np.float32)


def build_all(X, rel, mk):
    n = len(X)
    iw, ih, ik = np.meshgrid(np.arange(n), np.arange(24), np.arange(4), indexing="ij")
    iw, ih, ik = iw.ravel(), ih.ravel(), ik.ravel()
    xe = X[iw, ih, ik, :].astype(np.float32)
    raw = rel[iw, :, ih, :]
    m = mk[iw, :, ih, :]
    koh = np.zeros((len(iw), 4), dtype=np.float32)
    koh[np.arange(len(iw)), ik] = 1.0
    return xe, raw, m, koh


def predict_array(net, dev, xe, raw, mk, koh, mu, sd):
    net.eval()
    xe = ((xe - mu) / sd).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(xe), 65536):
            out.append(net(
                torch.from_numpy(xe[i:i + 65536]).to(dev),
                torch.from_numpy(raw[i:i + 65536]).to(dev),
                torch.from_numpy(mk[i:i + 65536]).to(dev),
                torch.from_numpy(koh[i:i + 65536]).to(dev),
            ).cpu().numpy())
    return np.concatenate(out)


def tune_mults_any(Yflat, Pflat_raw, keep, init=None, grid=None, rounds=2):
    """Coordinate multiplier search for any metric count."""
    nmet = Pflat_raw.shape[1]
    if init is None:
        init = [1.0] * nmet
    else:
        init = list(init[:nmet])
        while len(init) < nmet:
            init.append(1.0)
    if grid is None:
        grid = np.arange(0.84, 1.08, 0.005)
    mults = list(init)
    best, _ = C.mapeauc(Yflat, Pflat_raw * np.array(mults), keep)
    for _ in range(rounds):
        improved = False
        for k in range(nmet):
            cur = mults[k]
            for g in grid:
                mults[k] = g
                s, _ = C.mapeauc(Yflat, Pflat_raw * np.array(mults), keep)
                if s > best + 1e-9:
                    best = s
                    cur = g
                    improved = True
            mults[k] = cur
        if not improved:
            break
    return best, mults


def main():
    t0 = time.time()
    X, BL, Y, cells, dates, wds, starts = C.load_cache()
    R = np.load(f"{C.CACHE}/raw_windows.npy")
    val_cells = C.get_split(set(cells.tolist()))
    is_va = np.array([c in val_cells for c in cells])
    rel, mk = make_rel(R, BL)
    ok = (~np.isnan(Y)) & (Y > 1e-6)
    q5 = np.nanpercentile(Y[~is_va].reshape(-1, 4), 5, axis=0)
    ok = ok & (Y >= q5[None, None, :])

    trW = np.where(~is_va)[0]
    vaW = np.where(is_va)[0]
    iw, ih, ik = np.where(ok[trW])
    if A.train_hours:
        allowed = set(int(x) for x in A.train_hours.split(",") if x != "")
        mask_h = np.array([int(h) in allowed for h in ih])
        iw, ih, ik = iw[mask_h], ih[mask_h], ik[mask_h]
    if A.train_metrics:
        allowed_m = set(int(x) for x in A.train_metrics.split(",") if x != "")
        mask_k = np.array([int(k) in allowed_m for k in ik])
        iw, ih, ik = iw[mask_k], ih[mask_k], ik[mask_k]
    gw = trW[iw]
    XE = X[gw, ih, ik, :].astype(np.float32)
    RAW = rel[gw, :, ih, :]
    MK = mk[gw, :, ih, :]
    KOH = np.zeros((len(gw), 4), dtype=np.float32)
    KOH[np.arange(len(gw)), ik] = 1
    ytr = np.log(np.maximum(np.nan_to_num(Y[gw, ih, ik], nan=1.0), 1e-6)).astype(np.float32)
    mu, sd = XE.mean(0), XE.std(0) + 1e-6
    XEn = (XE - mu) / sd
    print(
        "train", XEn.shape, RAW.shape,
        "hours", A.train_hours or "all",
        "metrics", A.train_metrics or "all",
        f"{time.time()-t0:.0f}s",
        flush=True,
    )

    XEv, RAWv, MKv, KOHv = build_all(X[vaW], rel[vaW], mk[vaW])
    eval_hours = None
    if A.eval_hours:
        eval_hours = [int(x) for x in A.eval_hours.split(",") if x != ""]
    eval_metrics = None
    if A.eval_metrics:
        eval_metrics = [int(x) for x in A.eval_metrics.split(",") if x != ""]
    zt = np.load(f"{C.CACHE}/test_windows.npz", allow_pickle=True)
    Xte, BLte = zt["X"], zt["BL"]
    Rte = load_test_raw(A.data_root, len(Xte))
    relte, mkte = make_rel(Rte, BLte)
    XEt, RAWtst, MKtst, KOHtst = build_all(Xte, relte, mkte)

    dev = A.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    torch.manual_seed(A.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(A.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    net = ConvNet() if A.arch == "conv" else ManualAttnNet()
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=A.lr, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=0.4)
    XEt_train = torch.from_numpy(XEn).to(dev)
    RAW_train = torch.from_numpy(RAW).to(dev)
    MK_train = torch.from_numpy(MK).to(dev)
    KOH_train = torch.from_numpy(KOH).to(dev)
    yt = torch.from_numpy(ytr).to(dev)
    n = len(yt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs * (n // A.bs + 1))
    gen = torch.Generator(device=dev).manual_seed(A.seed)

    def val_score():
        logp = predict_array(net, dev, XEv, RAWv, MKv, KOHv, mu, sd).reshape(len(vaW), 24, 4)
        yh = Y[vaW]
        blh = BL[vaW]
        lph = logp
        if eval_hours is not None:
            yh = yh[:, eval_hours, :]
            blh = blh[:, eval_hours, :]
            lph = lph[:, eval_hours, :]
        if eval_metrics is not None:
            yh = yh[:, :, eval_metrics]
            blh = blh[:, :, eval_metrics]
            lph = lph[:, :, eval_metrics]
        nmet = yh.shape[-1]
        Yv = yh.reshape(-1, nmet).astype(float)
        BLs = blh
        LPs = lph
        keep = C.judge_filter(Yv)
        lp = np.clip(LPs, BLs - np.log(6), BLs + np.log(6))
        init = (0.9, 0.95, 0.95, 1.0)
        if eval_metrics is not None:
            init = tuple(init[k] for k in eval_metrics)
        s, mults = tune_mults_any(
            Yv, np.exp(lp).reshape(-1, nmet), keep,
            init=init, rounds=2,
        )
        return s, mults, logp

    best_s, best_m, best_state = -1.0, None, None
    for ep in range(A.epochs):
        net.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        total, cnt = 0.0, 0
        for i in range(0, n, A.bs):
            idx = perm[i:i + A.bs]
            rawb = RAW_train[idx]
            mkb = MK_train[idx]
            if A.aug > 0:
                dm = (torch.rand(len(idx), 14, 1, device=dev, generator=gen) < A.aug).float()
                rawb = rawb * (1 - dm)
                mkb = torch.clamp(mkb + dm, max=1)
            if A.noise > 0:
                rawb = rawb + torch.randn(rawb.shape, device=dev, generator=gen) * A.noise
            opt.zero_grad()
            loss = lossf(net(XEt_train[idx], rawb, mkb, KOH_train[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
            cnt += 1
        if ep >= 5 and (ep + 1) % 2 == 0:
            s, m, _ = val_score()
            print(f"ep{ep+1} loss={total/cnt:.4f} val={s:.5f} {time.time()-t0:.0f}s", flush=True)
            if s > best_s:
                best_s, best_m = s, m
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})

    s_fin, m_fin, val_fin = val_score()
    np.save(A.save_prefix + "_val_final.npy", val_fin)
    test_fin = predict_array(net, dev, XEt, RAWtst, MKtst, KOHtst, mu, sd).reshape(len(Xte), 24, 4)
    np.save(A.save_prefix + "_test_final.npy", test_fin)

    if best_state is not None:
        net.load_state_dict(best_state)
        _, _, val_best = val_score()
        test_best = predict_array(net, dev, XEt, RAWtst, MKtst, KOHtst, mu, sd).reshape(len(Xte), 24, 4)
        np.save(A.save_prefix + "_val_best.npy", val_best)
        np.save(A.save_prefix + "_test_best.npy", test_best)

    print({
        "tag": A.tag,
        "final": round(float(s_fin), 5),
        "best": round(float(best_s), 5),
        "best_m": None if best_m is None else [round(float(x), 3) for x in best_m],
        "t": int(time.time() - t0),
    }, flush=True)


if __name__ == "__main__":
    main()
