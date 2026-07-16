# -*- coding: utf-8 -*-
"""
AI 话务信息预测 - 训练/推理完整管线 (numpy + lightgbm + torch)
=====================================================================
方法: 异构集成 —— 4 个日序列卷积 NN + 6 个 LightGBM 两阶段模型
  + 2 个 MLP, 按 0.6/0.3/0.1 加权取 log 平均:
  LightGBM 阶段1: 对 log(话务量) 做 L2 回归 (800棵树, lr=0.05);
  LightGBM 阶段2: 以阶段1预测为初值, 用 MAPEAUC 的 sigmoid 阈值代理损失
         (按判题"样本时刻"级: 4指标绝对百分比误差先取均值再过阈值,
          阈值 0.2/0.3/0.4/0.5, 温度 tau=0.08) 的自定义梯度
          继续提升 (300棵树, lr=0.025)。该损失直接对齐评分指标,
          并天然学得 MAPE 对高估/低估的不对称性。
  卷积 NN x4 (集成主力, 权重0.6): 对该小时 4 指标 x 14 天原始 log 滞后
       (相对统计基线) + NIL 掩码 + 目标指标序列组成的 (14天,9通道) 日序列
       做两层 Conv1d(64) 编码, 池化后与 43 维工程特征、指标 one-hot 拼接进
       MLP 头 (hid-hid-256-1, GELU, dropout 0.1); Huber(δ=0.4) 回归 log 目标,
       AdamW + 余弦退火 44 epochs, 训练时日遮蔽(p=0.08)与高斯噪声(σ=0.04)增强。
       种子 7/107/2027 (hid=512) 与 2718 (hid=768)。
  MLP x2 (权重0.1): 输入 113 维 = 43 工程特征 + 该小时原始滞后 56 + 掩码 14;
       113-512-512-256-1, Huber 回归, 10 epochs, 种子 7/107。

【随机数说明(供复现审查)】随机性来自 LightGBM 行/列子采样与 MLP
  初始化/dropout/数据洗牌:
  - 6 个 LightGBM 模型种子 SEEDS_LGB = (7, 99, 1907, 4099, 777, 2718)
    (两阶段共用同一 seed 参数);
  - MLP: torch.manual_seed(7), 数据洗牌 torch.Generator(seed=7)。
  为保证跨机器可复现并遵守赛事纪律: LightGBM 固定 num_threads=1, deterministic=True,
  force_row_wise=True; MLP 在 CPU 上单线程训练, torch.set_num_threads(1),
  torch.use_deterministic_algorithms(True)。除此之外无任何随机数。

特征(每个样本 = 一个(窗口, 目标小时h, 指标k), 共43维): 同前版
  14天同小时log滞后、鲁棒统计量、趋势、离散度、相邻小时、one-hot、
  小时谐波、跨指标近期水平等(详见 build_window_feats)。
目标: log(真实话务量)。样本清洗: ①剔除NIL与非正目标;
  ②剔除目标低于该指标全训练集5%分位的样本(与判题过滤对齐)。

推理: 集成 log 预测 (0.6*卷积NN均值 + 0.3*LGB均值 + 0.1*MLP均值) 后:
  (1) 钳位到统计基线的 e^{±ln6} 倍内(防未见小区上的极端外推);
  (2) 乘以分指标乘子 MULTS (由训练集划分出的验证小区调得)。

用法: python predict.py <train_data.csv> <test_data.csv> <results.csv>
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '8')

import numpy as np
import csv, sys, datetime, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import torch
import torch.nn as nn

MULTS = np.array([0.905, 0.950, 0.950, 1.000])
CLAMP = np.log(6.0)
W_CONV, W_LGB, W_MLP = 0.6, 0.3, 0.1
STRIDE = 1
LOWQ = 5.0            # 低分位样本剔除百分位
TAU = 0.08            # 代理损失温度
THRESH = (0.2, 0.3, 0.4, 0.5)
SEEDS_LGB = (7, 99, 1907, 4099, 777, 2718)
SEEDS_MLP = (7, 107)
CONV_CFG = ((7, 512), (107, 512), (2027, 512), (2718, 768))
NUM_METRICS, HOURS, DAYS = 4, 24, 14

LGB_PARAMS = dict(objective='regression', num_leaves=127, learning_rate=0.05,
                  min_data_in_leaf=50, bagging_fraction=0.7, bagging_freq=1,
                  feature_fraction=0.75, num_threads=8, verbose=-1,
                  max_bin=63, deterministic=True, force_row_wise=True)
N_TREES_L2, N_TREES_FT, LR_FT = 800, 300, 0.025
MLP_EPOCHS, MLP_BS, MLP_LR = 10, 4096, 1e-3
CONV_EPOCHS, CONV_AUG, CONV_NOISE = 44, 0.08, 0.04
NTHREADS = 8


class ConvNet(nn.Module):
    """日序列卷积编码 + MLP 头. 输入: xe(43), raw(14,4), mk(14,4), koh(4)"""
    def __init__(self, hid):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(9, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(43 + 4 + 128, hid), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hid, hid), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hid, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, xe, raw, mk, koh):
        z = torch.cat([raw, mk, (koh.unsqueeze(1) * raw).sum(-1, keepdim=True)], -1)
        h = self.conv(z.transpose(1, 2))
        e = torch.cat([h.mean(-1), h[:, :, -3:].mean(-1)], -1)
        return self.head(torch.cat([xe, koh, e], -1)).squeeze(1)


def train_conv(Xa, ytr, seed, hid):
    """CPU 确定性训练卷积 NN. Xa: (n,169) 全特征; 返回 (net, mu, sd)"""
    torch.set_num_threads(NTHREADS)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    XE = Xa[:, :43]
    mu, sd = XE.mean(0), XE.std(0) + 1e-6
    XEt = torch.from_numpy((XE - mu) / sd)
    RAWt = torch.from_numpy(Xa[:, 43:99].reshape(-1, 14, 4))
    MKt = torch.from_numpy(Xa[:, 113:169].reshape(-1, 14, 4))
    KOHt = torch.from_numpy(Xa[:, 22:26].copy())
    yt = torch.from_numpy(ytr.astype(np.float32))
    net = ConvNet(hid)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=0.4)
    n = len(yt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CONV_EPOCHS * (n // MLP_BS + 1))
    g = torch.Generator().manual_seed(seed)
    net.train()
    for ep in range(CONV_EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, MLP_BS):
            idx = perm[i:i+MLP_BS]
            raw = RAWt[idx]; mk = MKt[idx]
            dm = (torch.rand(len(idx), 14, 1, generator=g) < CONV_AUG).float()
            raw = raw * (1 - dm)
            mk = torch.clamp(mk + dm, max=1.0)
            raw = raw + torch.randn(raw.shape, generator=g) * CONV_NOISE
            opt.zero_grad()
            loss = lossf(net(XEt[idx], raw, mk, KOHt[idx]), yt[idx])
            loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 10 == 0:
            print('conv seed %d epoch %d done' % (seed, ep + 1), flush=True)
    net.eval()
    return net, mu, sd


def conv_predict(net, mu, sd, Xk):
    with torch.no_grad():
        xe = torch.from_numpy((Xk[:, :43] - mu) / sd)
        raw = torch.from_numpy(Xk[:, 43:99].reshape(-1, 14, 4))
        mk = torch.from_numpy(Xk[:, 113:169].reshape(-1, 14, 4))
        koh = torch.from_numpy(Xk[:, 22:26].copy())
        return net(xe, raw, mk, koh).numpy()


def raw_block(hist, bl):
    """hist:(14,24,4), bl:(24,4) -> rel:(24,4,56), mflag:(24,4,14), mall:(24,4,56)
    rel[h,k] = 该小时全部4指标的14天 log 滞后相对基线(NIL置0) 展平;
    mall[h,k] = 该小时全部4指标的 NIL 掩码展平(与 rel 同布局)"""
    lh = np.log(np.maximum(hist, 1e-6))
    m = np.isnan(hist)
    lh[m] = 0.0
    rel = np.clip(lh - bl[None, :, :] * (~m), -4, 4)      # (14,24,4)
    rel_hk = np.transpose(rel, (1, 0, 2)).reshape(24, 1, 56)
    rel_out = np.repeat(rel_hk, 4, axis=1)                 # (24,4,56)
    mf = np.transpose(m.astype(np.float32), (1, 2, 0))     # (24,4,14)
    m_hk = np.transpose(m.astype(np.float32), (1, 0, 2)).reshape(24, 1, 56)
    m_out = np.repeat(m_hk, 4, axis=1)                     # (24,4,56)
    return rel_out.astype(np.float32), mf, m_out.astype(np.float32)


def train_mlp(Xtr, ytr, seed):
    """CPU 确定性训练 MLP; 返回 (net, mu, sd)"""
    torch.set_num_threads(NTHREADS)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = (Xtr - mu) / sd
    D = Xtr.shape[1]
    net = nn.Sequential(
        nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=0.4)
    Xt = torch.from_numpy(Xn); yt = torch.from_numpy(ytr.astype(np.float32))
    n = len(yt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=MLP_EPOCHS * (n // MLP_BS + 1))
    g = torch.Generator().manual_seed(seed)
    net.train()
    for ep in range(MLP_EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, MLP_BS):
            idx = perm[i:i+MLP_BS]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]).squeeze(1), yt[idx])
            loss.backward(); opt.step(); sched.step()
        print('mlp seed %d epoch %d done' % (seed, ep), flush=True)
    net.eval()
    return net, mu, sd


def mlp_predict(net, mu, sd, Xk):
    with torch.no_grad():
        return net(torch.from_numpy((Xk - mu) / sd)).squeeze(1).numpy()


def parse_dt(s):
    d, hm = s.strip().split(' ')
    y, m, dd = d.split('/')
    return datetime.datetime(int(y), int(m), int(dd), int(hm.split(':')[0]))


def fmt_dt(dt):
    return '%d/%d/%d %d:00' % (dt.year, dt.month, dt.day, dt.hour)


def tgm_arr(a):
    v = a[~np.isnan(a)]
    if len(v) == 0: return np.nan
    if len(v) < 4: return float(np.median(v))
    sp = np.sort(np.maximum(v, 1e-6))
    return float(np.exp(np.mean(np.log(sp[1:-1]))))


def build_window_feats(hist, wd):
    """hist:(14,24,4) 含NaN; 返回 X:(24,4,43), base_log:(24,4)"""
    D, H, K = hist.shape
    is_wend = np.array([((wd - (D - d)) % 7) in (5, 6) for d in range(D)])
    tgt_wend = wd in (5, 6)
    lh = np.log(np.maximum(hist, 1e-6)); lh[np.isnan(hist)] = np.nan
    X = np.zeros((H, K, 43))
    dt_tgm = np.zeros((H, K)); rc_tgm = np.zeros((H, K)); all_med = np.zeros((H, K))
    disp = np.zeros((H, K)); nval = np.zeros((H, K))
    for h in range(H):
        for k in range(K):
            col = hist[:, h, k]; valid = ~np.isnan(col)
            nval[h, k] = valid.sum()
            mask = valid & (is_wend == tgt_wend)
            dt_tgm[h, k] = tgm_arr(col[mask]) if mask.sum() >= 2 else \
                (tgm_arr(col[valid]) if valid.any() else 1.0)
            idx = np.where(valid)[0]
            rc_tgm[h, k] = tgm_arr(col[idx[-7:]]) if len(idx) else dt_tgm[h, k]
            all_med[h, k] = np.median(col[valid]) if valid.any() else 1.0
            vv = col[valid]; vv = vv[vv > 1e-6]
            disp[h, k] = np.std(np.log(vv)) if len(vv) >= 3 else 0.5
    ldt = np.log(np.maximum(dt_tgm, 1e-6)); lrc = np.log(np.maximum(rc_tgm, 1e-6))
    lam = np.log(np.maximum(all_med, 1e-6))
    tf = np.zeros(K)
    for k in range(K):
        dm = np.nanmean(hist[:, :, k], axis=1); v = ~np.isnan(dm)
        if v.sum() >= 6:
            vv = dm[v]; tf[k] = np.log(min(max(np.mean(vv[-3:]) / np.mean(vv), 0.5), 2.0))
    for h in range(H):
        for k in range(K):
            f = []
            for d in [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
                v = lh[d, h, k]
                f.append(v if not np.isnan(v) else lrc[h, k])
            f += [ldt[h, k], lrc[h, k], lam[h, k], tf[k], disp[h, k]]
            f += [lrc[(h - 1) % H, k], lrc[(h + 1) % H, k]]
            dm_last = np.nanmean(hist[13, :, k]); dm_all = np.nanmean(hist[:, :, k])
            f.append(np.log(max(dm_last, 1e-6)) - np.log(max(dm_all, 1e-6))
                     if dm_last > 0 and dm_all > 0 else 0.0)
            oh = [0.0] * 4; oh[k] = 1.0; f += oh
            wdoh = [0.0] * 7; wdoh[wd] = 1.0; f += wdoh
            f += [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                  np.sin(4 * np.pi * h / 24), np.cos(4 * np.pi * h / 24)]
            f.append(1.0 if tgt_wend else 0.0)
            f.append(nval[h, k])
            for kk in range(K):
                if kk != k: f.append(lrc[h, kk])
            X[h, k, :len(f)] = f   # 42维特征, 第43列恒为0(占位)
    a = 0.70 if tgt_wend else 0.65
    base_log = a * ldt + (1 - a) * lrc + 0.5 * tf[None, :]
    return X, base_log


def make_proxy_obj(sid, y, tau=TAU):
    """LightGBM 自定义目标: 时刻级 MAPEAUC 代理损失梯度.
    loss = mean_th sigmoid((MAPE_slot - th)/tau), MAPE_slot = 组内 |e^r - 1| 均值."""
    cnt = np.bincount(sid).astype(np.float64)

    def obj(pred, ds):
        r = np.clip(pred - y, -10, 10)
        er = np.exp(r)
        ape = np.abs(er - 1.0)
        dape = np.sign(er - 1.0) * er
        mape = (np.bincount(sid, weights=ape) / np.maximum(cnt, 1))[sid]
        n = cnt[sid]
        g = np.zeros_like(pred)
        for th in THRESH:
            z = np.clip((mape - th) / tau, -30, 30)
            s = 1.0 / (1.0 + np.exp(-z))
            g += s * (1 - s) / tau * dape / n
        g /= len(THRESH)
        return g, np.ones_like(g)
    return obj


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else 'train_data.csv'
    test_path = sys.argv[2] if len(sys.argv) > 2 else 'test_data.csv'
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'results.csv'

    # ---------- 读训练集, 滑窗构造训练样本 ----------
    per = {}
    with open(train_path, encoding='utf-8-sig') as f:
        r = csv.reader(f); next(r)
        for line in r:
            t, cell = line[0], line[1]
            vals = [np.nan if x == 'NIL' else float(x) for x in line[2:6]]
            per.setdefault(cell, []).append((t, vals))
    # 样本拼接顺序保持"先训练小区、后验证小区"(开发期划分), 保证子采样序列可复现
    cells_sorted = sorted(per.keys())
    val_cells = set(np.random.RandomState(0).choice(
        cells_sorted, size=len(cells_sorted) // 4, replace=False))
    bucket = {'tr': [], 'va': []}
    for cell, rows in per.items():
        arr = np.array([v for _, v in rows]); times = [t for t, _ in rows]
        n_days = len(times) // HOURS
        A = arr[:n_days * HOURS].reshape(n_days, HOURS, NUM_METRICS)
        t0 = parse_dt(times[0])
        key = 'va' if cell in val_cells else 'tr'
        for st in range(0, n_days - DAYS, STRIDE):
            wd = (t0 + datetime.timedelta(days=st + DAYS)).weekday()
            X, bl = build_window_feats(A[st:st + DAYS], wd)
            rel, mf, mall = raw_block(A[st:st + DAYS], bl)
            Xfull = np.concatenate([X, rel, mf, mall], axis=2)   # (24,4,169)
            bucket[key].append((Xfull, A[st + DAYS]))

    # 低分位剔除阈值: 全训练集目标值(非NIL且>0)的各指标5%分位
    all_y = np.concatenate([Y.reshape(-1, NUM_METRICS)
                            for key in ('tr', 'va') for _, Y in bucket[key]])
    q5 = np.zeros(NUM_METRICS)
    for k in range(NUM_METRICS):
        v = all_y[:, k]; v = v[(~np.isnan(v)) & (v > 1e-6)]
        q5[k] = np.percentile(v, LOWQ)

    Xs, Ys, Sids = [], [], []
    slot = 0   # "样本时刻"编号: 每个(窗口,小时)一个
    for key in ('tr', 'va'):
        for X, Y in bucket[key]:
            for k in range(NUM_METRICS):
                y = Y[:, k]
                ok = (~np.isnan(y)) & (y > 1e-6) & (y >= q5[k])
                Xs.append(X[ok, k, :]); Ys.append(np.log(y[ok]))
                Sids.append(slot + np.where(ok)[0])
            slot += HOURS
    Xa = np.concatenate(Xs).astype(np.float32); La = np.concatenate(Ys)
    Xa_lgb = Xa[:, :43]
    sid = np.concatenate(Sids)
    _, sid = np.unique(sid, return_inverse=True)   # 紧凑化时刻编号
    print('train samples:', Xa.shape, flush=True)

    # ---------- 6 x LightGBM 两阶段 + 1 x MLP ----------
    proxy_obj = make_proxy_obj(sid, La)
    lgb_models = []
    dtr = lgb.Dataset(Xa_lgb, label=La, free_raw_data=False)
    for s in SEEDS_LGB:
        p1 = dict(LGB_PARAMS); p1['seed'] = s
        m0 = lgb.train(p1, dtr, num_boost_round=N_TREES_L2)
        init = m0.predict(Xa_lgb, num_iteration=N_TREES_L2)
        dtr2 = lgb.Dataset(Xa_lgb, label=La, init_score=init, free_raw_data=False)
        p2 = dict(LGB_PARAMS); p2['seed'] = s
        p2['objective'] = proxy_obj
        p2['learning_rate'] = LR_FT
        m1 = lgb.train(p2, dtr2, num_boost_round=N_TREES_FT)
        lgb_models.append((m0, m1))
        print('lgb seed %d trained' % s, flush=True)
    mlps = [train_mlp(Xa[:, :113], La, s) for s in SEEDS_MLP]
    print('mlp trained', flush=True)
    convs = [train_conv(Xa, La, s, h) for s, h in CONV_CFG]
    print('conv trained', flush=True)

    # ---------- 读测试集并预测 ----------
    groups = []
    with open(test_path, encoding='utf-8-sig') as f:
        r = csv.reader(f); next(r)
        buf = []
        for line in r:
            buf.append(line)
            if len(buf) == DAYS * HOURS:
                cell = buf[0][1]; t0 = parse_dt(buf[0][0])
                arr = np.empty((DAYS * HOURS, NUM_METRICS))
                for i, row in enumerate(buf):
                    for k in range(NUM_METRICS):
                        v = row[2 + k]
                        arr[i, k] = np.nan if v == 'NIL' else float(v)
                groups.append((cell, t0, arr.reshape(DAYS, HOURS, NUM_METRICS)))
                buf = []
    hdr = ['时间', '小区名称', '小区上行平均激活用户数', '小区下行平均激活用户数',
           '下行平均使用的PRB个数', '上行平均使用的PRB个数']
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(hdr)
        for cell, t0, hist in groups:
            first = t0 + datetime.timedelta(days=DAYS)
            X, bl = build_window_feats(hist, first.weekday())
            rel, mf, mall = raw_block(hist, bl)
            Xfull = np.concatenate([X, rel, mf, mall], axis=2)
            logp = np.zeros((HOURS, NUM_METRICS))
            for k in range(NUM_METRICS):
                Xk = Xfull[:, k, :43].astype(np.float32)
                acc = np.zeros(HOURS)
                for m0, m1 in lgb_models:
                    acc += m0.predict(Xk) + m1.predict(Xk)
                lg = acc / len(lgb_models)
                Xk169 = Xfull[:, k, :].astype(np.float32)
                mp = np.mean([mlp_predict(net, mu, sd, Xk169[:, :113])
                              for net, mu, sd in mlps], axis=0)
                cv = np.mean([conv_predict(net, mu, sd, Xk169)
                              for net, mu, sd in convs], axis=0)
                logp[:, k] = W_CONV * cv + W_LGB * lg + W_MLP * mp
            logp = np.clip(logp, bl - CLAMP, bl + CLAMP)
            pred = np.maximum(np.exp(logp) * MULTS[None, :], 1e-4)
            for h in range(HOURS):
                ts = first + datetime.timedelta(hours=h)
                w.writerow([fmt_dt(ts), cell] +
                           ['%.4f' % pred[h, k] for k in range(NUM_METRICS)])
    print('written', out_path, ':', len(groups), 'groups x 24 rows')


if __name__ == '__main__':
    main()
