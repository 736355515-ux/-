// =====================================================================
// 通信资源联合分配 (决赛 Problem 2)
// ---------------------------------------------------------------------
// 与初赛 v22 的差异:
//   * res 可属于 1~2 个子带(RSub 重叠), fse 的波束增益项 = 所属子带
//     "选中波束能力和"的算术平均 => 速率不再按子带而按"信道"(res 的
//     子带集合)分表。信道数 C <= T + 重叠对数。
//   * 规模: P<=32, N<=100, K<=72, T<=18, M<=16, |RU|<=20(res 共享人数
//     上限 20), beamMaxNum<=255。
//   * 输出: 前 T 行波束集合; 后 N 行 = 用户使用的 res 编号列表。
// 结构(继承 v22):
//   1. 决策变量: (a) 每子带波束掩码(总数<=beamMaxNum); (b) 用户-res
//      分配(同 res 共享者同属一个 RU)。
//   2. 外层 SA 对掩码做退火, 增量评估: 掩码变动只影响含该子带的信道,
//      保留其它信道分配, 清空受影响信道后贪心修复(repairPlainP)。
//      速率档位查表: 每用户预计算 120 个线性域阈值(s=1..20 × 6档,
//      位级二分校准到与 log 域判定逐比特一致), linSum 落在哪个"格"
//      唯一决定速率行; 格号不变 => 免评估。
//   3. 收尾: 全量贪心 + 逐res精修(polish) + 分配级LNS 交替。
//   4. 单线程, 全程校验波束预算与共享规则, 输出合法解。
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;
using Clock = chrono::steady_clock;

static inline int pop32(unsigned int x) {
    x -= (x >> 1) & 0x55555555u;
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    return (int)((x * 0x01010101u) >> 24);
}

#ifndef SA_BUD
#define SA_BUD 0.072
#endif
#ifndef FINAL_DL
#define FINAL_DL 0.085
#endif
static const double TIME_BUDGET = SA_BUD;

#define MAXN 104          // 用户数上限+余量
#define MAXC 96           // 信道数上限(T + 重叠对) + 余量
#define MAXU 20           // 单 res 最大共享人数(=|RU|上限)
// 速率表尺寸按实际 maxShare(=max|RU|)动态设定, 小用例省一半以上内存流量
static int RWv   = 24;     // 速率行宽 = maxShare+1
static int NTHRv = 120;    // 每用户阈值数 = 6*maxShare
static int NCELv = 121;    // NTHRv+1

struct Alloc {            // 一次res分配记录(ch: 信道编号 0-based)
    int ch;
    int res;              // 实际 res 编号(输出前统一发放)
    vector<int> users;
};

int P, N, K, T, beamMaxNum, M;
int C;                              // 信道数
vector<vector<int>> RU;
vector<int> SU;
vector<vector<double>> CAP;         // CAP[i][p], i:1..N, p:0..P-1
vector<long long> BUF;
vector<double> SINR;
vector<vector<int>> chSubs;         // [c]: 该信道的子带列表(1..T), 1~2 个
vector<vector<int>> chRes;          // [c]: 该信道的 res 列表
vector<vector<int>> subChs;         // [t]: 含子带 t 的信道列表
static int chS1[MAXC], chS2[MAXC];  // 信道的两个子带(chS2=-1 表示单子带)
vector<double> totLinDb;
vector<int> groupOf;
vector<char> inSU;
int maxShare = 2;                   // 全局最大共享人数 = max|RU| (>=1)

static double LOG10S[MAXU + 2];
static const double FSE_THR[6] = {-10.0, 0.0, 3.0, 10.0, 15.0, 20.0};
static const short RATE_LVL[7] = {0, 8, 24, 90, 120, 162, 222};
static vector<double> uThr;        // [i*NTHRv+k]: 用户i阈值升序
static vector<short>  uCellRate;   // [(i*NCELv+c)*RWv+s]

// ---------------------------------------------------------------------
struct State {
    vector<unsigned int> beamMask;      // [T+1]
    vector<vector<double>> subLin;      // [T+1][N+1]: sum_{p in mask} cap
    vector<vector<double>> gainDb;      // [C][N+1]: 10log10(v_c)-totLinDb
    vector<vector<short>> rateTab;      // [C][(N+1)*RWv]
    vector<vector<int>> order1;         // [C][N]
    vector<vector<vector<int>>> gorder; // [C][M][*]
    vector<vector<short>> cellIdx;      // [C][N+1]
    int beamCount() const {
        int c = 0;
        for (int t = 1; t <= T; ++t) c += pop32(beamMask[t]);
        return c;
    }
    inline bool chActive(int c) const {
        if (!beamMask[chS1[c]]) return false;
        int t2 = chS2[c];
        return t2 < 0 || beamMask[t2] != 0;
    }
    // 信道 c 用户 i 的当前线性域均值
    inline double chVal(int c, int i) const {
        int t2 = chS2[c];
        if (t2 < 0) return subLin[chS1[c]][i];
        return (subLin[chS1[c]][i] + subLin[t2][i]) * 0.5;
    }
    // 快速保存/恢复: 子带 linSum 行整拷(FP 位精确), 信道格号用撤销日志
    // (速率行是格号的纯函数, 逆放格号 + 查表重写行即可位精确还原)
    static unsigned int bkMask[2];
    static vector<double> bkLin[2];
    static int undoC[16384];
    static int undoI[16384];
    static short undoCell[16384];
    static int undoN;
    static bool undoOvf;
    void saveSub(int t, int slot) {
        bkMask[slot] = beamMask[t];
        bkLin[slot] = subLin[t];
    }
    void restoreSub(int t, int slot) {
        beamMask[t] = bkMask[slot];
        subLin[t].swap(bkLin[slot]);
    }
    void undoChs() {
        for (int j = undoN - 1; j >= 0; --j) {
            int c = undoC[j], i = undoI[j];
            short cc = undoCell[j];
            cellIdx[c][i] = cc;
            memcpy(rateTab[c].data() + i * RWv,
                   uCellRate.data() + ((size_t)i * NCELv + cc) * RWv,
                   RWv * sizeof(short));
        }
        undoN = 0;
        undoOvf = false;
    }
    // 子带 t 掩码已改(增删位), 更新 subLin[t]; 不动信道表
    void applySubDelta(int t, unsigned int addMask, unsigned int rmvMask) {
        double* ls = subLin[t].data();
        for (unsigned int m = addMask; m; m &= m - 1) {
            int p = __builtin_ctz(m);
            for (int i = 1; i <= N; ++i) ls[i] += CAP[i][p];
        }
        for (unsigned int m = rmvMask; m; m &= m - 1) {
            int p = __builtin_ctz(m);
            for (int i = 1; i <= N; ++i) ls[i] -= CAP[i][p];
        }
    }
    // 信道 c: 依据当前 subLin 更新 cellIdx/rateTab, 返回是否有速率行变化
    bool refreshCh(int c) {
        short* rt = rateTab[c].data();
        short* ci = cellIdx[c].data();
        const double* l1 = subLin[chS1[c]].data();
        const double* l2 = (chS2[c] >= 0) ? subLin[chS2[c]].data() : nullptr;
        const double* uthr = uThr.data();
        bool dirty = false;
        for (int i = 1; i <= N; ++i) {
            double v = l2 ? (l1[i] + l2[i]) * 0.5 : l1[i];
            const double* thr = uthr + (size_t)i * NTHRv;
            int cc = ci[i];
            while (cc < NTHRv && thr[cc] < v) ++cc;
            while (cc > 0 && thr[cc - 1] >= v) --cc;
            if (cc != ci[i]) {
                if (undoN < 16384) {
                    undoC[undoN] = c; undoI[undoN] = i;
                    undoCell[undoN] = ci[i];
                    ++undoN;
                } else undoOvf = true;
                ci[i] = (short)cc;
                memcpy(rt + i * RWv,
                       uCellRate.data() + ((size_t)i * NCELv + cc) * RWv,
                       RWv * sizeof(short));
                dirty = true;
            }
        }
        return dirty;
    }
    void recomputeSub(int t) {
        unsigned int mask = beamMask[t];
        for (int i = 1; i <= N; ++i) {
            double s = 0;
            unsigned int m = mask;
            while (m) { int p = __builtin_ctz(m); s += CAP[i][p]; m &= m - 1; }
            subLin[t][i] = s;
        }
    }
    void recomputeCh(int c) {
        for (int i = 1; i <= N; ++i) {
            double v = chVal(c, i);
            gainDb[c][i] = (v > 0) ? (10.0 * log10(v) - totLinDb[i]) : -1e18;
            const double* thr = uThr.data() + (size_t)i * NTHRv;
            int cc = (int)(lower_bound(thr, thr + NTHRv, v) - thr);
            cellIdx[c][i] = (short)cc;
            memcpy(rateTab[c].data() + i * RWv,
                   uCellRate.data() + ((size_t)i * NCELv + cc) * RWv,
                   RWv * sizeof(short));
#ifdef VCHECK
            for (int s = 1; s <= maxShare; ++s) {
                double fse = SINR[i] - LOG10S[s] +
                             ((v > 0) ? (10.0 * log10(v) - totLinDb[i]) : -1e18);
                short ref = (short)(fse <= -10 ? 0 : fse <= 0 ? 8 : fse <= 3 ? 24 :
                            fse <= 10 ? 90 : fse <= 15 ? 120 : fse <= 20 ? 162 : 222);
                if (rateTab[c][i * RWv + s] != ref)
                    fprintf(stderr, "MISMATCH c=%d i=%d s=%d got=%d ref=%d v=%.17g\n",
                            c, i, s, (int)rateTab[c][i * RWv + s], (int)ref, v);
            }
#endif
        }
        vector<int>& od = order1[c];
        for (int i = 0; i < N; ++i) od[i] = i + 1;
        const short* rt = rateTab[c].data();
        sort(od.begin(), od.end(), [rt](int a, int b){
            return rt[a * RWv + 1] > rt[b * RWv + 1]; });
        for (int m = 0; m < M; ++m) {
            vector<int>& go = gorder[c][m];
            const double* gd = gainDb[c].data();
            sort(go.begin(), go.end(), [this, gd](int a, int b){
                return SINR[a] + gd[a] > SINR[b] + gd[b]; });
        }
    }
    void allocStructs() {
        beamMask.assign(T + 1, 0u);
        subLin.assign(T + 1, vector<double>(N + 1, 0.0));
        gainDb.assign(C, vector<double>(N + 1, 0.0));
        rateTab.assign(C, vector<short>((N + 1) * RWv, 0));
        order1.assign(C, vector<int>(N));
        cellIdx.assign(C, vector<short>(N + 1, 0));
        gorder.assign(C, vector<vector<int>>(M));
        for (int c = 0; c < C; ++c)
            for (int m = 0; m < M; ++m) gorder[c][m] = RU[m];
    }
    void recomputeAll() {
        for (int t = 1; t <= T; ++t) recomputeSub(t);
        for (int c = 0; c < C; ++c) recomputeCh(c);
    }
};
unsigned int State::bkMask[2];
vector<double> State::bkLin[2];
int State::undoC[16384];
int State::undoI[16384];
short State::undoCell[16384];
int State::undoN = 0;
bool State::undoOvf = false;

// ---------------------------------------------------------------------
struct RCand { long long gain; int nu; int users[MAXU]; long long vals[MAXU]; };

static inline void rpSingle(const State& st, const vector<long long>& rem,
                            int cc, RCand& c) {
    const short* rt = st.rateTab[cc].data();
    long long bg = 0; int bu = -1;
    for (int i = 1; i <= N; ++i) {
        long long re = rem[i];
        long long r = (long long)rt[i * RWv + 1];
        long long g = re < r ? re : r;
        g = (re <= 0 || r <= 0) ? 0 : g;
        if (g > bg) { bg = g; bu = i; }
    }
    c.gain = bg > 0 ? bg : 0; c.nu = bu > 0 ? 1 : 0;
    if (bu > 0) { c.users[0] = bu; c.vals[0] = bg; }
}

static inline void rpGroup(const State& st, const vector<long long>& rem,
                           int cc, int m, RCand& c) {
    static pair<long long,int> tmp[MAXU + 2];
    const short* rt = st.rateTab[cc].data();
    c.gain = 0; c.nu = 0;
    const vector<int>& grp = RU[m];
    int gsz = (int)grp.size();
    long long v2[MAXU]; int u2[MAXU]; int n2 = 0;
    for (int ui = 0; ui < gsz; ++ui) {
        int u = grp[ui];
        if (rem[u] <= 0) continue;
        int r = rt[u * RWv + 2];
        if (r <= 0) continue;
        long long v = rem[u] < r ? rem[u] : (long long)r;
        int j = n2++;
        while (j > 0 && v2[j-1] < v) { v2[j] = v2[j-1]; u2[j] = u2[j-1]; --j; }
        v2[j] = v; u2[j] = u;
    }
    if (n2 < 2) return;
    c.gain = v2[0] + v2[1]; c.nu = 2;
    c.users[0] = u2[0]; c.vals[0] = v2[0];
    c.users[1] = u2[1]; c.vals[1] = v2[1];
    long long pref = v2[0] + v2[1];
    for (int s = 3; s <= gsz; ++s) {
        if (s > n2) break;
        pref += v2[s-1];
        if (pref <= c.gain) continue;
        int cnt = 0;
        for (int ui = 0; ui < gsz; ++ui) {
            int u = grp[ui];
            if (rem[u] <= 0) continue;
            int r = rt[u * RWv + s];
            if (r <= 0) continue;
            long long v = rem[u] < r ? rem[u] : (long long)r;
            int j = cnt < s ? cnt : s;
            if (cnt < s) ++cnt;
            else if (v <= tmp[s-1].first) continue;
            while (j > 0 && tmp[j-1].first < v) { tmp[j] = tmp[j-1]; --j; }
            tmp[j] = make_pair(v, u);
        }
        if (cnt < s) continue;
        long long sum = 0;
        for (int j = 0; j < s; ++j) sum += tmp[j].first;
        if (sum > c.gain) {
            c.gain = sum; c.nu = s;
            for (int j = 0; j < s; ++j) { c.users[j] = tmp[j].second; c.vals[j] = tmp[j].first; }
        }
    }
}

struct PAlloc { int ch; int nu; int users[MAXU]; };

static inline void rpApply(vector<PAlloc>& allocs, int cc, const RCand& c) {
    PAlloc a; a.ch = cc; a.nu = c.nu;
    for (int j = 0; j < c.nu; ++j) a.users[j] = c.users[j];
    allocs.push_back(a);
}

void repairPlainP(const State& st, vector<PAlloc>& allocs,
                  vector<long long>& rem, int* freeCnt, long long* acc = nullptr) {
    int chs[MAXC], ns = 0;
    for (int c = 0; c < C; ++c)
        if (freeCnt[c] > 0 && st.chActive(c)) chs[ns++] = c;
    if (!ns) return;
    static vector<RCand> cnd;
    if ((int)cnd.size() < ns * (M + 1)) cnd.resize(ns * (M + 1));
    for (int si = 0; si < ns; ++si) {
        rpSingle(st, rem, chs[si], cnd[si * (M + 1)]);
        for (int m = 0; m < M; ++m)
            rpGroup(st, rem, chs[si], m, cnd[si * (M + 1) + 1 + m]);
    }
    for (;;) {
        long long best = 0; int bsi = -1, bg = -1;
        for (int si = 0; si < ns; ++si) {
            if (freeCnt[chs[si]] <= 0) continue;
            for (int g = 0; g <= M; ++g) {
                const RCand& c = cnd[si * (M + 1) + g];
                if (c.gain > best) { best = c.gain; bsi = si; bg = g; }
            }
        }
        if (bsi < 0 || best <= 0) break;
        RCand& c = cnd[bsi * (M + 1) + bg];
        int cc = chs[bsi];
        --freeCnt[cc];
        rpApply(allocs, cc, c);
        int gm = groupOf[c.users[0]];
        for (int j = 0; j < c.nu; ++j) rem[c.users[j]] -= c.vals[j];
        if (acc) {
            const short* rt = st.rateTab[cc].data();
            for (int j = 0; j < c.nu; ++j) acc[c.users[j]] += rt[c.users[j] * RWv + c.nu];
        }
        for (int si = 0; si < ns; ++si) {
            if (freeCnt[chs[si]] <= 0) continue;
            rpSingle(st, rem, chs[si], cnd[si * (M + 1)]);
            if (gm >= 0) rpGroup(st, rem, chs[si], gm, cnd[si * (M + 1) + 1 + gm]);
        }
    }
}

long long evalAllocP(const State& st, const vector<PAlloc>& allocs) {
    static long long acc[MAXN];
    for (int i = 1; i <= N; ++i) acc[i] = 0;
    for (size_t ai = 0; ai < allocs.size(); ++ai) {
        const PAlloc& a = allocs[ai];
        for (int j = 0; j < a.nu; ++j)
            acc[a.users[j]] += st.rateTab[a.ch][a.users[j] * RWv + a.nu];
    }
    long long tot = 0;
    for (int i = 1; i <= N; ++i) tot += min(BUF[i], acc[i]);
    return tot;
}

// 全量贪心(懒惰缓存, 信道按"等价键"分组: 相同子带掩码多重集 => 速率表相同)
long long assign(const State& st, vector<Alloc>* out) {
    static vector<long long> rem;
    rem = BUF;
    long long total = 0;
    if (out) out->clear();

    static int gid[MAXC];
    static unsigned long long gkey[MAXC];
    static int grep[MAXC];
    static int gfree[MAXC];
    static int gchs[MAXC][MAXC];
    static int gchN[MAXC];
    static int gnext[MAXC][MAXC];
    int G = 0;
    for (int c = 0; c < C; ++c) {
        gid[c] = -1;
        if (!st.chActive(c) || chRes[c].empty()) continue;
        unsigned long long key;
        if (chSubs[c].size() == 1) key = st.beamMask[chSubs[c][0]];
        else {
            unsigned long long a = st.beamMask[chSubs[c][0]];
            unsigned long long b = st.beamMask[chSubs[c][1]];
            if (a > b) swap(a, b);
            key = (a << 32) | b | (1ull << 63);
        }
        int g = -1;
        for (int j = 0; j < G; ++j)
            if (gkey[j] == key) { g = j; break; }
        if (g < 0) {
            g = G++;
            gkey[g] = key; grep[g] = c; gfree[g] = 0; gchN[g] = 0;
        }
        gid[c] = g;
        gnext[g][gchN[g]] = 0;
        gchs[g][gchN[g]++] = c;
        gfree[g] += (int)chRes[c].size();
    }

    auto rate = [&](int i, int cc, int s) -> int {
        return st.rateTab[cc][i * RWv + s];
    };
    struct Cand { long long gain; int nu; int users[MAXU]; long long vals[MAXU]; };
    static pair<long long,int> tmp[MAXU + 2];
    auto bestCand = [&](int g) -> Cand {
        int cc = grep[g];
        Cand c; c.gain = 0; c.nu = 0;
        long long bg = 0; int bu = -1;
        const vector<int>& od = st.order1[cc];
        for (int oi = 0; oi < N; ++oi) {
            int i = od[oi];
            int r = rate(i, cc, 1);
            if (r <= bg) break;
            if (rem[i] <= 0) continue;
            long long gval = rem[i] < r ? rem[i] : r;
            if (gval > bg) { bg = gval; bu = i; }
        }
        if (bu > 0) { c.gain = bg; c.nu = 1; c.users[0] = bu; c.vals[0] = bg; }
        for (int m = 0; m < M; ++m) {
            const vector<int>& go = st.gorder[cc][m];
            int gsz = (int)go.size();
            for (int s = 2; s <= gsz; ++s) {
                int rmax = rate(go[0], cc, s);
                if (rmax <= 0) break;
                if ((long long)rmax * s <= c.gain) continue;
                int cnt = 0;
                long long sumTop = 0;
                for (int ui = 0; ui < gsz; ++ui) {
                    int u = go[ui];
                    int r = rate(u, cc, s);
                    if (r <= 0) break;
                    if (cnt >= s && tmp[s-1].first >= r) break;
                    if (rem[u] <= 0) continue;
                    long long v = rem[u] < r ? rem[u] : (long long)r;
                    int pos = cnt < s ? cnt : s;
                    if (cnt < s) ++cnt;
                    else if (v <= tmp[s-1].first) continue;
                    int j = pos;
                    while (j > 0 && tmp[j-1].first < v) { tmp[j] = tmp[j-1]; --j; }
                    tmp[j] = make_pair(v, u);
                }
                if (cnt < s) continue;
                sumTop = 0;
                for (int j = 0; j < s; ++j) sumTop += tmp[j].first;
                if (sumTop > c.gain) {
                    c.gain = sumTop; c.nu = s;
                    for (int j = 0; j < s; ++j) { c.users[j] = tmp[j].second; c.vals[j] = tmp[j].first; }
                }
            }
        }
        return c;
    };

    static vector<pair<long long,int>> heap;
    heap.clear();
    static Cand cache[MAXC];
    static char fresh[MAXC];
    for (int g = 0; g < G; ++g) {
        fresh[g] = 0;
        if (gfree[g] <= 0) continue;
        cache[g] = bestCand(g);
        fresh[g] = 1;
        if (cache[g].gain > 0) heap.push_back(make_pair(cache[g].gain, g));
    }
    make_heap(heap.begin(), heap.end());
    while (!heap.empty()) {
        pop_heap(heap.begin(), heap.end());
        int g = heap.back().second;
        heap.pop_back();
        if (gfree[g] <= 0) continue;
        if (!fresh[g]) {
            cache[g] = bestCand(g);
            fresh[g] = 1;
            if (cache[g].gain <= 0) continue;
            if (!heap.empty() && cache[g].gain < heap.front().first) {
                heap.push_back(make_pair(cache[g].gain, g));
                push_heap(heap.begin(), heap.end());
                continue;
            }
        }
        const Cand c = cache[g];
        if (c.gain <= 0) continue;
        total += c.gain;
        int chSlot = -1;
        for (int j = 0; j < gchN[g]; ++j) {
            int cc = gchs[g][j];
            if (gnext[g][j] < (int)chRes[cc].size()) { chSlot = j; break; }
        }
        int cc = gchs[g][chSlot];
        int resId = chRes[cc][gnext[g][chSlot]++];
        --gfree[g];
        for (int j = 0; j < c.nu; ++j) rem[c.users[j]] -= c.vals[j];
        if (out) {
            Alloc a; a.ch = cc; a.res = resId;
            a.users.assign(c.users, c.users + c.nu);
            out->push_back(a);
        }
        {
            static char touched[MAXN];
            for (int i = 1; i <= N; ++i) touched[i] = 0;
            for (int j = 0; j < c.nu; ++j) touched[c.users[j]] = 1;
            for (int gg = 0; gg < G; ++gg) {
                if (!fresh[gg] || gg == g) continue;
                for (int j = 0; j < cache[gg].nu; ++j)
                    if (touched[cache[gg].users[j]]) { fresh[gg] = 0; break; }
            }
            fresh[g] = 0;
        }
        if (gfree[g] > 0) {
            heap.push_back(make_pair(c.gain, g));
            push_heap(heap.begin(), heap.end());
        }
    }
    return total;
}

long long evalAlloc(const State& st, const vector<Alloc>& allocs) {
    static long long acc[MAXN];
    for (int i = 1; i <= N; ++i) acc[i] = 0;
    for (size_t ai = 0; ai < allocs.size(); ++ai) {
        const Alloc& a = allocs[ai];
        int s = (int)a.users.size();
        for (int j = 0; j < s; ++j)
            acc[a.users[j]] += st.rateTab[a.ch][a.users[j] * RWv + s];
    }
    long long tot = 0;
    for (int i = 1; i <= N; ++i) tot += min(BUF[i], acc[i]);
    return tot;
}

long long polish(const State& st, vector<Alloc>& allocs, double deadline,
                 chrono::time_point<Clock> startTime) {
    static vector<long long> rem;
    auto rate = [&](int i, int cc, int s) -> int {
        return st.rateTab[cc][i * RWv + s];
    };
    static pair<long long,int> tmp[MAXU + 2];
    auto elapsed = [&]() {
        return chrono::duration<double>(Clock::now() - startTime).count();
    };
    long long total = 0;
    for (int pass = 0; pass < 8; ++pass) {
        bool improved = false;
        for (size_t ai = 0; ai < allocs.size(); ++ai) {
            if (elapsed() > deadline) return total;
            rem = BUF;
            long long curTotal = 0;
            for (size_t aj = 0; aj < allocs.size(); ++aj) {
                if (aj == ai) continue;
                const Alloc& a = allocs[aj];
                int s = (int)a.users.size();
                for (int j = 0; j < s; ++j) {
                    int u = a.users[j];
                    int r = rate(u, a.ch, s);
                    long long g = rem[u] < r ? rem[u] : (long long)r;
                    rem[u] -= g;
                    curTotal += g;
                }
            }
            long long oldGain = 0;
            {
                const Alloc& a = allocs[ai];
                int s = (int)a.users.size();
                static long long remc[MAXU + 2];
                for (int j = 0; j < s; ++j) remc[j] = rem[a.users[j]];
                for (int j = 0; j < s; ++j) {
                    int u = a.users[j];
                    int r = rate(u, a.ch, s);
                    long long g = remc[j] < r ? remc[j] : (long long)r;
                    oldGain += g;
                }
            }
            int cc = allocs[ai].ch;
            long long bg = 0; int bu = -1;
            for (int i = 1; i <= N; ++i) {
                if (rem[i] <= 0) continue;
                int r = rate(i, cc, 1);
                if (r <= 0) continue;
                long long g = rem[i] < r ? rem[i] : (long long)r;
                if (g > bg) { bg = g; bu = i; }
            }
            long long bestG = bg; int bestNu = (bu > 0) ? 1 : 0;
            int bestUsers[MAXU]; long long bestVals[MAXU];
            if (bu > 0) { bestUsers[0] = bu; bestVals[0] = bg; }
            for (int m = 0; m < M; ++m) {
                const auto& grp = RU[m];
                int gsz = (int)grp.size();
                for (int s = 2; s <= gsz; ++s) {
                    int cnt = 0;
                    for (size_t ui = 0; ui < grp.size(); ++ui) {
                        int u = grp[ui];
                        if (rem[u] <= 0) continue;
                        int r = rate(u, cc, s);
                        if (r <= 0) continue;
                        long long v = rem[u] < r ? rem[u] : (long long)r;
                        tmp[cnt++] = make_pair(v, u);
                    }
                    if (cnt < s) continue;
                    partial_sort(tmp, tmp + s, tmp + cnt,
                                 [](const pair<long long,int>& a, const pair<long long,int>& b){
                                     return a.first > b.first; });
                    long long g = 0;
                    for (int j = 0; j < s; ++j) g += tmp[j].first;
                    if (g > bestG) {
                        bestG = g; bestNu = s;
                        for (int j = 0; j < s; ++j) { bestUsers[j] = tmp[j].second; bestVals[j] = tmp[j].first; }
                    }
                }
            }
            if (bestG > oldGain && bestNu > 0) {
                allocs[ai].users.assign(bestUsers, bestUsers + bestNu);
                improved = true;
            }
            total = curTotal + max(bestG, oldGain);
        }
        if (!improved) break;
    }
    return total;
}

// 2-res 重切分: 同(信道,组)的两个 res 联合调整共享人数 (si,sj)->(si±1,sj∓1)
// 捕获 polish(单res)原子性够不到的组尺寸重切分 gap
bool pairRebalance(const State& st, vector<Alloc>& allocs, double deadline,
                   chrono::time_point<Clock> startTime) {
    auto elapsed = [&]() {
        return chrono::duration<double>(Clock::now() - startTime).count();
    };
    auto rate = [&](int i, int cc, int s) -> int {
        return st.rateTab[cc][i * RWv + s];
    };
    int n = (int)allocs.size();
    bool anyImp = false;
    static vector<long long> rem;
    static pair<long long,int> tmp[MAXU + 2];
    for (int ai = 0; ai < n; ++ai) {
        if (elapsed() > deadline) return anyImp;
        int gm = groupOf[allocs[ai].users[0]];
        if (gm < 0) continue;
        for (size_t j2 = 0; j2 < allocs[ai].users.size(); ++j2)
            if (groupOf[allocs[ai].users[j2]] != gm) { gm = -2; break; }
        if (gm < 0) continue;
        for (int aj = ai + 1; aj < n; ++aj) {
            int gm2 = groupOf[allocs[aj].users[0]];
            if (gm2 != gm) continue;
            bool ok2 = true;
            for (size_t j2 = 0; j2 < allocs[aj].users.size(); ++j2)
                if (groupOf[allocs[aj].users[j2]] != gm) { ok2 = false; break; }
            if (!ok2) continue;
            int cca = allocs[ai].ch, ccb = allocs[aj].ch;
            int sa = (int)allocs[ai].users.size(), sb = (int)allocs[aj].users.size();
            // rem 不含这两个 res 的贡献
            rem = BUF;
            long long baseOther = 0;
            for (int ak = 0; ak < n; ++ak) {
                if (ak == ai || ak == aj) continue;
                const Alloc& a = allocs[ak];
                int s = (int)a.users.size();
                for (size_t j = 0; j < a.users.size(); ++j) {
                    int u = a.users[j];
                    int r = rate(u, a.ch, s);
                    long long g = rem[u] < r ? rem[u] : (long long)r;
                    rem[u] -= g;
                    baseOther += g;
                }
            }
            // 旧收益(顺序 ai 后 aj, 与 eval 等价的贪心序)
            auto evalPair = [&](int s1, int s2, int* u1, int* nu1, int* u2, int* nu2) -> long long {
                static long long remc[MAXN];
                for (int i = 1; i <= N; ++i) remc[i] = rem[i];
                const vector<int>& grp = RU[gm];
                long long tot = 0;
                int ns[2] = {s1, s2};
                int ccq[2] = {cca, ccb};
                int* outs[2] = {u1, u2};
                int* nouts[2] = {nu1, nu2};
                for (int q = 0; q < 2; ++q) {
                    int s = ns[q];
                    if (s <= 0) { *nouts[q] = 0; continue; }
                    int cnt = 0;
                    for (size_t ui = 0; ui < grp.size(); ++ui) {
                        int u = grp[ui];
                        if (remc[u] <= 0) continue;
                        int r = rate(u, ccq[q], s);
                        if (r <= 0) continue;
                        long long v = remc[u] < r ? remc[u] : (long long)r;
                        tmp[cnt++] = make_pair(v, u);
                    }
                    if (cnt < s) { *nouts[q] = 0; continue; }
                    partial_sort(tmp, tmp + s, tmp + cnt,
                                 [](const pair<long long,int>& a, const pair<long long,int>& b){
                                     return a.first > b.first; });
                    *nouts[q] = s;
                    for (int j = 0; j < s; ++j) {
                        outs[q][j] = tmp[j].second;
                        remc[tmp[j].second] -= tmp[j].first;
                        tot += tmp[j].first;
                    }
                }
                return tot;
            };
            int ua[MAXU], ub[MAXU], na, nb;
            long long oldG = evalPair(sa, sb, ua, &na, ub, &nb);
            int maxS = (int)RU[gm].size();
            long long bestG = oldG;
            int bsa = -1, bsb = -1;
            for (int d = -1; d <= 1; d += 2) {
                int s1 = sa + d, s2 = sb - d;
                if (s1 < 1 || s2 < 1 || s1 > maxS || s2 > maxS) continue;
                int va[MAXU], vb[MAXU], nva, nvb;
                long long g = evalPair(s1, s2, va, &nva, vb, &nvb);
                if (g > bestG && nva == s1 && nvb == s2) {
                    bestG = g; bsa = s1; bsb = s2;
                }
            }
            if (bsa > 0) {
                int va[MAXU], vb[MAXU], nva, nvb;
                evalPair(bsa, bsb, va, &nva, vb, &nvb);
                allocs[ai].users.assign(va, va + nva);
                allocs[aj].users.assign(vb, vb + nvb);
                anyImp = true;
#ifdef DIAG
                fprintf(stderr, "REBAL ch=%d/%d %d/%d -> %d/%d gain=%lld\n",
                        cca, ccb, sa, sb, bsa, bsb, bestG - oldG);
#endif
            }
        }
    }
    return anyImp;
}

void lnsRepair(const State& st, vector<Alloc>& allocs,
               vector<long long>& rem, int* freeCnt) {
    static pair<long long,int> tmp[MAXU + 2];
    for (;;) {
        long long best = 0; int bc = -1, bnu = 0;
        int busers[MAXU]; long long bvals[MAXU];
        for (int c = 0; c < C; ++c) {
            if (freeCnt[c] <= 0 || !st.chActive(c)) continue;
            const vector<int>& od = st.order1[c];
            for (int oi = 0; oi < N; ++oi) {
                int i = od[oi];
                int r = st.rateTab[c][i * RWv + 1];
                if (r <= best) break;
                if (rem[i] <= 0) continue;
                long long g = rem[i] < r ? rem[i] : r;
                if (g > best) { best = g; bc = c; bnu = 1; busers[0] = i; bvals[0] = g; }
            }
            for (int m = 0; m < M; ++m) {
                const vector<int>& go = st.gorder[c][m];
                int gsz = (int)go.size();
                for (int s = 2; s <= gsz; ++s) {
                    int rmax = st.rateTab[c][go[0] * RWv + s];
                    if (rmax <= 0) break;
                    if ((long long)rmax * s <= best) continue;
                    int cnt = 0;
                    for (int ui = 0; ui < gsz; ++ui) {
                        int u = go[ui];
                        int r = st.rateTab[c][u * RWv + s];
                        if (r <= 0) break;
                        if (cnt >= s && tmp[s-1].first >= r) break;
                        if (rem[u] <= 0) continue;
                        long long v = rem[u] < r ? rem[u] : (long long)r;
                        int pos = cnt < s ? cnt : s;
                        if (cnt < s) ++cnt;
                        else if (v <= tmp[s-1].first) continue;
                        int j = pos;
                        while (j > 0 && tmp[j-1].first < v) { tmp[j] = tmp[j-1]; --j; }
                        tmp[j] = make_pair(v, u);
                    }
                    if (cnt < s) continue;
                    long long sum = 0;
                    for (int j = 0; j < s; ++j) sum += tmp[j].first;
                    if (sum > best) {
                        best = sum; bc = c; bnu = s;
                        for (int j = 0; j < s; ++j) { busers[j] = tmp[j].second; bvals[j] = tmp[j].first; }
                    }
                }
            }
        }
        if (bc < 0 || best <= 0) break;
        --freeCnt[bc];
        Alloc a; a.ch = bc; a.res = 0;
        a.users.assign(busers, busers + bnu);
        allocs.push_back(a);
        for (int j = 0; j < bnu; ++j) rem[busers[j]] -= bvals[j];
    }
}

void lnsSearch(const State& st, vector<Alloc>& allocs, double deadline,
               chrono::time_point<Clock> startTime, mt19937& rng) {
    auto elapsed = [&]() {
        return chrono::duration<double>(Clock::now() - startTime).count();
    };
    long long curV = evalAlloc(st, allocs);
    vector<Alloc> cand; cand.reserve(K + 4);
    vector<long long> rem(N + 1);
    int freeCnt[MAXC];
    while (elapsed() < deadline) {
        int n = (int)allocs.size();
        if (n == 0) break;
        cand.clear();
#ifndef DPCT
#define DPCT 30
#endif
        int nd = max(1, n * DPCT / 100);
        static char drop[MAXC];
        for (int i = 0; i < n; ++i) drop[i] = 0;
        int op = rng() % 3;
        if (op == 0) {
            for (int d = 0; d < nd; ++d) drop[rng() % n] = 1;
        } else if (op == 1) {
            int u = 1 + rng() % N;
            for (int i = 0; i < n; ++i)
                for (size_t j = 0; j < allocs[i].users.size(); ++j)
                    if (allocs[i].users[j] == u) { drop[i] = 1; break; }
            drop[rng() % n] = 1;
        } else {
            int cc = rng() % C;
            for (int i = 0; i < n; ++i) if (allocs[i].ch == cc) drop[i] = 1;
            drop[rng() % n] = 1;
        }
        for (int c = 0; c < C; ++c)
            freeCnt[c] = st.chActive(c) ? (int)chRes[c].size() : 0;
        for (int i = 1; i <= N; ++i) rem[i] = BUF[i];
        for (int i = 0; i < n; ++i) {
            if (drop[i]) continue;
            cand.push_back(allocs[i]);
            --freeCnt[allocs[i].ch];
        }
        {
            static long long acc[MAXN];
            for (int i = 1; i <= N; ++i) acc[i] = 0;
            for (size_t ci = 0; ci < cand.size(); ++ci) {
                const Alloc& a = cand[ci];
                int s = (int)a.users.size();
                for (int j = 0; j < s; ++j)
                    acc[a.users[j]] += st.rateTab[a.ch][a.users[j] * RWv + s];
            }
            for (int i = 1; i <= N; ++i) rem[i] = max(0LL, BUF[i] - acc[i]);
        }
        lnsRepair(st, cand, rem, freeCnt);
        long long v = evalAlloc(st, cand);
        if (v >= curV) {
            curV = v;
            allocs = cand;
        }
    }
    static int nxt[MAXC];
    for (int c = 0; c < C; ++c) nxt[c] = 0;
    for (size_t ai = 0; ai < allocs.size(); ++ai)
        allocs[ai].res = chRes[allocs[ai].ch][nxt[allocs[ai].ch]++];
}

int main() {
    auto startTime = Clock::now();
    if (scanf("%d %d %d %d %d", &P, &N, &K, &T, &beamMaxNum) != 5) return 0;
    scanf("%d", &M);
    RU.assign(M, {});
    groupOf.assign(N + 1, -1);
    inSU.assign(N + 1, 0);
    maxShare = 1;
    for (int m = 0; m < M; ++m) {
        int c; scanf("%d", &c);
        RU[m].resize(c);
        for (int j = 0; j < c; ++j) {
            scanf("%d", &RU[m][j]);
            groupOf[RU[m][j]] = m;
        }
        if (c > maxShare) maxShare = c;
    }
    if (maxShare > MAXU) maxShare = MAXU;
    RWv = maxShare + 1;
    NTHRv = 6 * maxShare;
    NCELv = NTHRv + 1;
    {
        int c; scanf("%d", &c);
        SU.resize(c);
        for (int j = 0; j < c; ++j) { scanf("%d", &SU[j]); inSU[SU[j]] = 1; }
    }
    CAP.assign(N + 1, vector<double>(P, 0.0));
    for (int i = 1; i <= N; ++i)
        for (int p = 0; p < P; ++p) scanf("%lf", &CAP[i][p]);
    BUF.assign(N + 1, 0);
    SINR.assign(N + 1, 0.0);
    for (int i = 1; i <= N; ++i) {
        long long b; double s;
        scanf("%lld %lf", &b, &s);
        BUF[i] = b; SINR[i] = s;
    }
    // 读子带并构建信道(res 的子带集合); res 可属 1~2 个子带
    vector<vector<int>> rsubs(K + 1);       // res -> 子带列表
    vector<vector<int>> RSUB(T + 1);
    for (int t = 1; t <= T; ++t) {
        int c; scanf("%d", &c);
        RSUB[t].resize(c);
        for (int j = 0; j < c; ++j) {
            scanf("%d", &RSUB[t][j]);
            int r = RSUB[t][j];
            if (r >= 1 && r <= K) rsubs[r].push_back(t);
        }
    }
    {
        // 信道 = 不同的子带集合; key = t1*32+t2 (单子带 t2=0)
        vector<int> keyCh((T + 1) * (T + 1) + T + 2, -1);
        C = 0;
        chSubs.clear(); chRes.clear();
        for (int r = 1; r <= K; ++r) {
            if (rsubs[r].empty()) continue;
            int t1 = rsubs[r][0], t2 = (rsubs[r].size() > 1) ? rsubs[r][1] : 0;
            if (t2 && t2 < t1) swap(t1, t2);
            int key = t1 * (T + 1) + t2;
            if (keyCh[key] < 0) {
                keyCh[key] = C++;
                vector<int> ss; ss.push_back(t1);
                if (t2) ss.push_back(t2);
                if (ss.size() == 2 && ss[0] == 0) { ss.erase(ss.begin()); }
                chSubs.push_back(ss);
                chRes.push_back({});
            }
            chRes[keyCh[key]].push_back(r);
        }
        // 修正: key 组合中 t2=0 表示单子带, chSubs 已正确
        for (int c = 0; c < C; ++c) {
            if (chSubs[c].size() == 2 && chSubs[c][0] == 0)
                chSubs[c].erase(chSubs[c].begin());
        }
        subChs.assign(T + 1, {});
        for (int c = 0; c < C; ++c)
            for (int t : chSubs[c]) subChs[t].push_back(c);
        for (int c = 0; c < C; ++c) {
            chS1[c] = chSubs[c][0];
            chS2[c] = (chSubs[c].size() > 1) ? chSubs[c][1] : -1;
        }
    }
    totLinDb.assign(N + 1, 0.0);
    for (int i = 1; i <= N; ++i) {
        double s = 0;
        for (int p = 0; p < P; ++p) s += CAP[i][p];
        totLinDb[i] = 10.0 * log10(max(s, 1e-12));
    }
    for (int s = 1; s <= MAXU + 1; ++s) LOG10S[s] = 10.0 * log10((double)s);
    // 每用户阈值(线性域, 位级校准) + 格速率行
    {
        vector<double> baseThr((size_t)(N + 1) * 8, 0.0);
        for (int i = 1; i <= N; ++i)
            for (int b = 0; b < 6; ++b)
                baseThr[(size_t)i * 8 + b] =
                    pow(10.0, (FSE_THR[b] - SINR[i] + totLinDb[i]) / 10.0);
        uThr.assign((size_t)(N + 1) * NTHRv, 0.0);
        uCellRate.assign((size_t)(N + 1) * NCELv * RWv, 0);
        for (int i = 1; i <= N; ++i) {
            static pair<double,int> tv[6 * MAXU];
            int k = 0;
            for (int s = 1; s <= maxShare; ++s)
                for (int b = 0; b < 6; ++b) {
                    double w = (double)s * baseThr[(size_t)i * 8 + b];
                    struct OkF {
                        double si, l10s, tld, fthr;
                        bool operator()(double x) const {
                            return si - l10s + (10.0 * log10(x) - tld) > fthr;
                        }
                    } ok = {SINR[i], LOG10S[s], totLinDb[i], FSE_THR[b]};
                    double lo = w * (1.0 - 1e-9), hi = w * (1.0 + 1e-9);
                    if (lo <= 0) lo = 1e-300;
                    int guard = 0;
                    while (ok(lo) && ++guard < 60) lo *= (1.0 - 1e-9);
                    guard = 0;
                    while (!ok(hi) && ++guard < 60) hi *= (1.0 + 1e-9);
                    if (!ok(lo) && ok(hi)) {
                        for (int it = 0; it < 80 && lo < hi; ++it) {
                            double mid = lo + (hi - lo) * 0.5;
                            if (mid <= lo || mid >= hi) break;
                            if (ok(mid)) hi = mid; else lo = mid;
                        }
                        while (true) {
                            double nx = nextafter(lo, hi);
                            if (nx >= hi || ok(nx)) break;
                            lo = nx;
                        }
                        w = lo;
                    }
                    tv[k++] = make_pair(w, s);
                }
            sort(tv, tv + NTHRv);
            int lvl[MAXU + 2];
            memset(lvl, 0, sizeof(lvl));
            short* cr0 = uCellRate.data() + (size_t)i * NCELv * RWv;
            for (int s = 1; s <= maxShare; ++s) cr0[s] = RATE_LVL[0];
            for (int c = 1; c <= NTHRv; ++c) {
                uThr[(size_t)i * NTHRv + (c - 1)] = tv[c - 1].first;
                ++lvl[tv[c - 1].second];
                short* cr = cr0 + (size_t)c * RWv;
                for (int s = 1; s <= maxShare; ++s) cr[s] = RATE_LVL[lvl[s]];
            }
        }
    }

    // 组对齐波束序: 每个 RU 组按成员加权能力排出的波束优先序
    static int gBeamOrd[18][32];   // [m][rank] -> beam
    {
        for (int m = 0; m < M && m < 18; ++m) {
            vector<pair<double,int>> sc(P);
            for (int p = 0; p < P; ++p) {
                double s = 0;
                for (int u : RU[m]) {
                    double tot = pow(10.0, totLinDb[u] / 10.0);
                    s += (double)BUF[u] * CAP[u][p] / max(tot, 1e-12);
                }
                sc[p] = make_pair(s, p);
            }
            sort(sc.begin(), sc.end(),
                 [](const pair<double,int>& a, const pair<double,int>& b){ return a.first > b.first; });
            for (int p = 0; p < P; ++p) gBeamOrd[m][p] = sc[p].second;
        }
    }

    // ---------------- 初始波束分配 ----------------
    vector<pair<double,int>> beamScore(P);
    for (int p = 0; p < P; ++p) {
        double sc = 0;
        for (int i = 1; i <= N; ++i) {
            double tot = pow(10.0, totLinDb[i] / 10.0);
            sc += (double)BUF[i] * CAP[i][p] / max(tot, 1e-12);
        }
        beamScore[p] = {sc, p};
    }
    sort(beamScore.begin(), beamScore.end(),
         [](const pair<double,int>& a, const pair<double,int>& b){ return a.first > b.first; });

    State cur;
    cur.allocStructs();
    int q = beamMaxNum / max(T, 1);
    if (q > P) q = P;
    if (q < 1) q = 1;
    {
        int budget = beamMaxNum;
        for (int t = 1; t <= T && budget > 0; ++t) {
            int give = min(q, budget);
            unsigned int mask = 0;
            for (int j = 0; j < give && j < P; ++j) mask |= (1u << beamScore[j].second);
            cur.beamMask[t] = mask;
            budget -= pop32(mask);
        }
        for (int j = q; j < P && budget > 0; ++j) {
            for (int t = 1; t <= T && budget > 0; ++t) {
                unsigned int b = (1u << beamScore[j].second);
                if (!(cur.beamMask[t] & b)) { cur.beamMask[t] |= b; --budget; }
            }
        }
    }
    cur.recomputeAll();
    long long curVal = assign(cur, nullptr);

    State best = cur;
    long long bestVal = curVal;

    {
        auto buildUniform = [&](int qq) {
            State s2;
            s2.allocStructs();
            int bud = beamMaxNum;
            for (int t = 1; t <= T && bud > 0; ++t) {
                int give = min(qq, min(bud, P));
                unsigned int mask = 0;
                for (int j = 0; j < give; ++j) mask |= (1u << beamScore[j].second);
                s2.beamMask[t] = mask;
                bud -= give;
            }
            for (int j = qq; j < P && bud > 0; ++j)
                for (int t = 1; t <= T && bud > 0; ++t) {
                    unsigned int b = (1u << beamScore[j].second);
                    if (!(s2.beamMask[t] & b)) { s2.beamMask[t] |= b; --bud; }
                }
            s2.recomputeAll();
            return s2;
        };
        int cand_q[4];
        int nq = 0;
        if (q + 1 <= P) cand_q[nq++] = q + 1;
        if (q - 1 >= 1) cand_q[nq++] = q - 1;
        if (q + 2 <= P) cand_q[nq++] = q + 2;
        cand_q[nq++] = min(P, max(1, (int)(q * 3 / 2)));
        for (int ci = 0; ci < nq; ++ci) {
            State s2 = buildUniform(cand_q[ci]);
            long long v = assign(s2, nullptr);
            if (v > bestVal) { best = s2; bestVal = v; cur = s2; curVal = v; }
        }
        if (beamMaxNum >= P + T - 1) {
            State s2;
            s2.allocStructs();
            unsigned int one = (1u << beamScore[0].second);
            int bud = beamMaxNum;
            for (int t = 1; t <= T; ++t) { s2.beamMask[t] = one; --bud; }
            unsigned int full = (P >= 32) ? 0xFFFFFFFFu : ((1u << P) - 1);
            for (int t = 1; t <= T && bud > 0; ++t) {
                int need = P - 1;
                if (bud >= need) { s2.beamMask[t] = full; bud -= need; }
                else {
                    for (int j = 0; j < P && bud > 0; ++j) {
                        unsigned int b = (1u << beamScore[j].second);
                        if (!(s2.beamMask[t] & b)) { s2.beamMask[t] |= b; --bud; }
                    }
                }
            }
            s2.recomputeAll();
            long long v = assign(s2, nullptr);
            if (v > bestVal) { best = s2; bestVal = v; cur = s2; curVal = v; }
        }
    }

    // ---------------- SA(增量评估) ----------------
    mt19937 rng(12345);   // 固定随机种子(合规: 无多线程, 结果可复现)
    auto elapsed = [&]() {
        return chrono::duration<double>(Clock::now() - startTime).count();
    };
    uniform_real_distribution<double> ur(0.0, 1.0);
    auto pickAdd = [&](unsigned int mask) -> int {
        int add;
        do { add = rng() % P; } while (mask & (1u << add));
        return add;
    };
#ifndef T0FAC
#define T0FAC 0.005
#endif
#ifndef COOL
#define COOL 0.03
#endif
#ifndef SAFRAC
#define SAFRAC 0.9
#endif
    double T0 = max(1.0, (double)bestVal * T0FAC);
    const double SA_END = TIME_BUDGET * SAFRAC;
    bool hillPhase = false;
    int checkCtr = 0;
    double now = 0;
    vector<PAlloc> curAll, candAll, bestAllP;
    {
        vector<Alloc> tmpAll;
        assign(cur, &tmpAll);
        for (size_t ai = 0; ai < tmpAll.size(); ++ai) {
            PAlloc a; a.ch = tmpAll[ai].ch; a.nu = (int)tmpAll[ai].users.size();
            for (int j = 0; j < a.nu; ++j) a.users[j] = tmpAll[ai].users[j];
            curAll.push_back(a);
        }
        curAll.reserve(K + 8); candAll.reserve(K + 8);
    }
    curVal = evalAllocP(cur, curAll);
    bestAllP = curAll;
    bestVal = curVal;
    best = cur;
    vector<long long> remv(N + 1);
    static long long accv[MAXN];
    int freeCnt[MAXC];
    static char chAff[MAXC];             // 本移动受影响的信道
    while (true) {
#ifdef EQCHK
        // 等价性检查模式: 固定迭代数 + 迭代制温度(消除时钟不确定性)
        ++checkCtr;
        if (checkCtr > EQIT) break;
        now = SA_END * (double)checkCtr / (double)EQIT;
        if (true) {
            if (false) {}
            if (!hillPhase && checkCtr >= (int)(EQIT * 0.9)) {
#else
        if ((checkCtr++ & 7) == 0) {
            now = elapsed();
            if (now >= TIME_BUDGET) break;
            if (!hillPhase && now >= SA_END) {
#endif
                hillPhase = true;
                bool anyDiff = false;
                for (int t = 1; t <= T; ++t)
                    if (cur.beamMask[t] != best.beamMask[t]) {
                        cur.beamMask[t] = best.beamMask[t];
                        cur.recomputeSub(t);
                        anyDiff = true;
                    }
                if (anyDiff)
                    for (int c = 0; c < C; ++c) cur.recomputeCh(c);
                curVal = bestVal;
                curAll = bestAllP;
            }
        }
        double temp = hillPhase ? 0.0 : T0 * pow(COOL, min(now / SA_END, 1.0));
#ifndef MVMODE
#define MVMODE 2
#endif
        int mv;
        if (MVMODE == 0) mv = rng() % 4;
        else if (MVMODE == 1) { int r = rng() % 8; mv = r < 3 ? 0 : (r < 5 ? 1 : (r < 6 ? 2 : (r < 7 ? 3 : 4))); }
        else { int r = rng() % 10; mv = r < 3 ? 0 : (r < 5 ? 1 : (r < 7 ? 2 : (r < 9 ? 3 : 4))); }
        if (M > 0 && (rng() & 15) == 0) mv = 5;   // 1/16 概率: 组对齐宏移动
        int t = 1 + rng() % T;
        int t2 = -1;
        cur.saveSub(t, 0);
        bool changed = false;
        unsigned int add0 = 0, rmv0 = 0, add1 = 0, rmv1 = 0;
        if (mv == 0) {
            unsigned int mask = cur.beamMask[t];
            if (mask && pop32(mask) < P) {
                int cnt = pop32(mask);
                int kth = rng() % cnt;
                unsigned int mm = mask;
                int rmv = -1;
                while (kth-- >= 0) { rmv = __builtin_ctz(mm); mm &= mm - 1; }
                unsigned int mask2 = mask & ~(1u << rmv);
                int add = pickAdd(mask2);
                cur.beamMask[t] = mask2 | (1u << add);
                rmv0 = (1u << rmv); add0 = (1u << add);
                changed = true;
            }
        } else if (mv == 1) {
            if (cur.beamCount() < beamMaxNum &&
                pop32(cur.beamMask[t]) < P) {
                int add = pickAdd(cur.beamMask[t]);
                cur.beamMask[t] |= (1u << add);
                add0 = (1u << add);
                changed = true;
            }
        } else if (mv == 2) {
            unsigned int mask = cur.beamMask[t];
            if (pop32(mask) > 1) {
                int cnt = pop32(mask);
                int kth = rng() % cnt;
                unsigned int mm = mask; int rmv = -1;
                while (kth-- >= 0) { rmv = __builtin_ctz(mm); mm &= mm - 1; }
                cur.beamMask[t] = mask & ~(1u << rmv);
                rmv0 = (1u << rmv);
                changed = true;
            }
        } else if (mv == 4) {
            int src = 1 + rng() % T;
            if (src != t && cur.beamMask[src] && cur.beamMask[src] != cur.beamMask[t]) {
                int slack = beamMaxNum - cur.beamCount();
                int dn = pop32(cur.beamMask[src]) - pop32(cur.beamMask[t]);
                if (dn <= slack) {
                    unsigned int oldm = cur.beamMask[t], newm = cur.beamMask[src];
                    cur.beamMask[t] = newm;
                    add0 = newm & ~oldm; rmv0 = oldm & ~newm;
                    changed = true;
                }
            }
        } else {
            t2 = 1 + rng() % T;
            if (t2 != t && pop32(cur.beamMask[t]) > 1 &&
                pop32(cur.beamMask[t2]) < P) {
                cur.saveSub(t2, 1);
                unsigned int mask = cur.beamMask[t];
                int cnt = pop32(mask);
                int kth = rng() % cnt;
                unsigned int mm = mask; int rmv = -1;
                while (kth-- >= 0) { rmv = __builtin_ctz(mm); mm &= mm - 1; }
                cur.beamMask[t] = mask & ~(1u << rmv);
                int add = pickAdd(cur.beamMask[t2]);
                cur.beamMask[t2] |= (1u << add);
                rmv0 = (1u << rmv);
                add1 = (1u << add);
                changed = true;
            } else t2 = -1;
        }
        if (!changed) continue;
        cur.applySubDelta(t, add0, rmv0);
        if (t2 > 0) cur.applySubDelta(t2, add1, rmv1);
        // 受影响信道 = 含 t (和 t2) 的信道; 撤销日志内刷新
        State::undoN = 0; State::undoOvf = false;
        for (int c = 0; c < C; ++c) chAff[c] = 0;
        for (int c : subChs[t]) chAff[c] = 1;
        if (t2 > 0) for (int c : subChs[t2]) chAff[c] = 1;
        bool dirty = false;
        for (int c = 0; c < C; ++c) {
            if (!chAff[c]) continue;
            if (cur.refreshCh(c)) dirty = true;
        }
        long long v;
        if (!dirty) {
            v = curVal;
        } else {
            candAll.clear();
            for (int i = 1; i <= N; ++i) accv[i] = 0;
            for (size_t ai = 0; ai < curAll.size(); ++ai) {
                const PAlloc& a = curAll[ai];
                if (chAff[a.ch]) continue;
                candAll.push_back(a);
                for (int j = 0; j < a.nu; ++j)
                    accv[a.users[j]] += cur.rateTab[a.ch][a.users[j] * RWv + a.nu];
            }
            for (int i = 1; i <= N; ++i)
                remv[i] = BUF[i] > accv[i] ? BUF[i] - accv[i] : 0;
            for (int c = 0; c < C; ++c) freeCnt[c] = 0;
            for (int c = 0; c < C; ++c)
                if (chAff[c] && cur.chActive(c)) freeCnt[c] = (int)chRes[c].size();
            repairPlainP(cur, candAll, remv, freeCnt, accv);
            v = 0;
            for (int i = 1; i <= N; ++i) v += accv[i] < BUF[i] ? accv[i] : BUF[i];
        }
        double d = (double)(v - curVal);
        bool acc = hillPhase ? (d >= 0)
                             : (d >= 0 || ur(rng) < exp(d / max(temp, 1e-9)));
        if (acc) {
            curVal = v;
            if (dirty) curAll.swap(candAll);
            State::undoN = 0; State::undoOvf = false;
            if (v > bestVal) { bestVal = v; best = cur; bestAllP = curAll; }
        } else {
            cur.restoreSub(t, 0);
            if (t2 > 0) cur.restoreSub(t2, 1);
            if (!State::undoOvf) cur.undoChs();
            else {
                // 撤销日志溢出(极端): subLin 已还原, 全量刷新受影响信道
                State::undoN = 0; State::undoOvf = false;
                for (int c = 0; c < C; ++c)
                    if (chAff[c]) cur.refreshCh(c);
                State::undoN = 0; State::undoOvf = false;
            }
        }
#ifndef RESYNC
#define RESYNC 512
#endif
        if (RESYNC && (checkCtr & (RESYNC - 1)) == 0) {
            candAll.clear();
            for (int i = 1; i <= N; ++i) remv[i] = BUF[i];
            for (int c = 0; c < C; ++c)
                freeCnt[c] = cur.chActive(c) ? (int)chRes[c].size() : 0;
            repairPlainP(cur, candAll, remv, freeCnt);
            long long fv = evalAllocP(cur, candAll);
            if (fv > curVal) {
                curVal = fv; curAll.swap(candAll);
                if (fv > bestVal) { bestVal = fv; best = cur; bestAllP = curAll; }
            }
        }
    }

#ifdef DIAG
    fprintf(stderr, "iters=%d C=%d\n", checkCtr, C);
    for (int t = 1; t <= T; ++t) fprintf(stderr, "mask %d %08x\n", t, best.beamMask[t]);
#endif
    best.recomputeAll();
    vector<Alloc> allocs;
    assign(best, &allocs);
    if (evalAlloc(best, allocs) < bestVal) {
        allocs.clear();
        for (size_t ai = 0; ai < bestAllP.size(); ++ai) {
            Alloc a; a.ch = bestAllP[ai].ch; a.res = 0;
            a.users.assign(bestAllP[ai].users, bestAllP[ai].users + bestAllP[ai].nu);
            allocs.push_back(a);
        }
    }
#ifdef EQCHK
    polish(best, allocs, 1e9, startTime);
    {
        long long bestV = evalAlloc(best, allocs);
        vector<Alloc> bestAll2 = allocs;
        while (false) {
#else
    polish(best, allocs, FINAL_DL, startTime);
    {
        long long bestV = evalAlloc(best, allocs);
        vector<Alloc> bestAll2 = allocs;
        while (elapsed() < FINAL_DL - 0.002) {
#endif
            lnsSearch(best, allocs, min(elapsed() + 0.004, FINAL_DL), startTime, rng);
            pairRebalance(best, allocs, FINAL_DL, startTime);
            polish(best, allocs, FINAL_DL, startTime);
            long long v = evalAlloc(best, allocs);
            if (v > bestV) { bestV = v; bestAll2 = allocs; }
            else allocs = bestAll2;
        }
        allocs = bestAll2;
        static int nxt2[MAXC];
        for (int c = 0; c < C; ++c) nxt2[c] = 0;
        for (size_t ai = 0; ai < allocs.size(); ++ai)
            allocs[ai].res = chRes[allocs[ai].ch][nxt2[allocs[ai].ch]++];
    }

    // 用户 -> res 列表(决赛输出: 只输出 res 编号)
    vector<vector<int>> userRes(N + 1);
    for (size_t ai = 0; ai < allocs.size(); ++ai) {
        const Alloc& a = allocs[ai];
        for (size_t j = 0; j < a.users.size(); ++j)
            userRes[a.users[j]].push_back(a.res);
    }
    for (int t = 1; t <= T; ++t) {
        unsigned int mask = best.beamMask[t];
        printf("%d", pop32(mask));
        unsigned int m = mask;
        while (m) {
            int p = __builtin_ctz(m);
            printf(" %d", p + 1);
            m &= m - 1;
        }
        printf("\n");
    }
    for (int i = 1; i <= N; ++i) {
        printf("%d", (int)userRes[i].size());
        for (size_t j = 0; j < userRes[i].size(); ++j) printf(" %d", userRes[i][j]);
        printf("\n");
    }
    return 0;
}
