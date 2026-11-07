"""Vectorised, strictly causal features for every machine at every tick.

All arrays are laid out [episodes, 8 machines, T ticks]. The feature value at
tick t uses only observations at ticks < t, so the same code builds training
rows (label = payout at t) and the query row (t = cutoff).
"""

import numpy as np

T = 257  # cutoffs are 192..256, so ticks 0..256 cover every prediction point
PAYOUTS = (0, 1, 2, 5, 20)
TYPES = ("steady", "balanced", "jackpot")
LAGS = tuple(range(1, 9))
ONEHOT_LAGS = tuple(range(1, 17))
WINDOWS = (8, 16, 32, 64, T)
EWMA_HL = (3, 8, 24, 72)
MIX_HL = (8, 24, 72, 1e9)
SELF_HL = (1, 1.5, 2, 3, 4, 6, 8, 10, 16, 32, 64, 128)
PAIR_HL = (2, 6, 24)


def shift(a, k, fill=0.0):
    """out[..., t] = a[..., t-k] (fill where t-k < 0)."""
    out = np.full_like(a, fill)
    if k < a.shape[-1]:
        out[..., k:] = a[..., : a.shape[-1] - k]
    return out


def excum(a):
    """Exclusive prefix sum: out[..., t] = sum_{s<t} a[..., s]."""
    out = np.zeros_like(a)
    np.cumsum(a[..., :-1], axis=-1, out=out[..., 1:])
    return out


def disc(a, half_life):
    """Discounted exclusive sum: out[t] = sum_{s<t} d^(t-1-s) a[s]."""
    d = 0.5 ** (1.0 / half_life)
    out = np.zeros_like(a)
    for t in range(1, a.shape[-1]):
        out[..., t] = d * out[..., t - 1] + a[..., t - 1]
    return out


def lag_effect(src_w, src_l, dst_w, dst_p, base, k, c=4.0, half_life=None):
    """Shrunk estimate of P(dst wins | src won k ticks earlier) - P(... | src lost)."""
    sw, sl = shift(src_w, k), shift(src_l, k)
    acc = excum if half_life is None else (lambda x: disc(x, half_life))
    nww, nw = acc(sw * dst_w), acc(sw * dst_p)
    nlw, nl = acc(sl * dst_w), acc(sl * dst_p)
    return (nww + c * base) / (nw + c) - (nlw + c * base) / (nl + c)


def mix_llr(x, tables_lr, half_life):
    """Log-likelihood ratio (table 1 vs table 0) of past win sizes, discounted."""
    llr = np.zeros(x.shape, dtype=np.float64)
    for c in range(1, 5):
        llr += tables_lr[..., c - 1][..., None] * (disc((x == c).astype(np.float64), half_life)
                                                   if half_life < 1e8 else excum((x == c).astype(np.float64)))
    return llr


def build(x, types, signals, short, tables):
    """x: int [E, 8, T] payout class 0..4, -1 = missing. types/signals: [E, 8].
    short: bool [E]. tables: float [3 types, 2, 4] win-size mix per table.
    Returns (features float32 [E, 8, T, F], names)."""
    x = x.astype(np.int64)
    E, M, TT = x.shape
    P = (x >= 0).astype(np.float64)
    W = (x > 0).astype(np.float64)
    L = (x == 0).astype(np.float64)
    feats, names = [], []

    def add(name, arr):
        feats.append(np.broadcast_to(arr, (E, M, TT)).astype(np.float32))
        names.append(name)

    # 1. recent outcomes by tick lag (missing is its own category)
    for k in ONEHOT_LAGS:
        v = shift(x.astype(np.float64), k, fill=-1.0)
        for c in range(-1, 5):
            add(f"lag{k}_c{c}", (v == c).astype(np.float64))

    # 2. windowed class frequencies
    cum = {c: excum((x == c).astype(np.float64)) for c in range(5)}
    cumP = excum(P)
    for w in WINDOWS:
        n = cumP - shift(cumP, w)
        for c in range(5):
            add(f"freq{w}_c{c}", (cum[c] - shift(cum[c], w) + 0.5) / (n + 2.5))
        add(f"n{w}", n)

    # 3. missing-aware EWMA win rates and their gaps (drift detectors)
    rate = {}
    for hl in EWMA_HL:
        rate[hl] = (disc(W, hl) + 0.5) / (disc(P, hl) + 1.0)
        add(f"ewma{hl}", rate[hl])
    add("gap3_24", rate[3] - rate[24])
    add("gap8_72", rate[8] - rate[72])
    add("gap24_72", rate[24] - rate[72])
    base = (excum(W) + 1.0) / (cumP + 2.0)
    add("winrate_all", base)

    # 4. this machine's own signed lag effects (all-history and recent), times current state
    self_sum = np.zeros((E, M, TT))
    recent_sum = {hl: np.zeros((E, M, TT)) for hl in SELF_HL}
    for k in LAGS:
        state = shift(W, k) - shift(L, k)
        eff = lag_effect(W, L, W, P, base, k)
        add(f"self_eff{k}", eff)
        add(f"self_effx{k}", eff * state)
        self_sum += eff * state
        for hl in SELF_HL:
            eff_r = lag_effect(W, L, W, P, base, k, half_life=hl)
            add(f"self_effr{hl}x{k}", eff_r * state)
            recent_sum[hl] += eff_r * state
    add("self_sum", self_sum)
    for hl in SELF_HL:
        add(f"self_sum_recent{hl}", recent_sum[hl])

    # 5. pair contexts: recent win rate on past ticks whose (lag j, lag k) win/lose
    #    states matched the current ones -- "interacting" lags, rules that change over time
    S = {k: shift(W, k) - shift(L, k) for k in LAGS}
    for a in LAGS:
        for b in LAGS:
            if b <= a:
                continue
            series = []
            for sa in (1, -1):
                for sb in (1, -1):
                    ind = ((S[a] == sa) & (S[b] == sb)).astype(np.float64)
                    series += [ind * W, ind * P]
            stack = np.stack(series)  # [8, E, M, T]
            cur = (S[a] == 1) * 0 + (S[a] == -1) * 2 + (S[b] == -1) * 1  # combo index 0..3
            ok = (S[a] != 0) & (S[b] != 0)
            for hl in PAIR_HL:
                d = disc(stack, hl)
                wins = np.take_along_axis(d[0::2], cur[None], 0)[0]
                tries = np.take_along_axis(d[1::2], cur[None], 0)[0]
                prate = np.where(ok, (wins + 2.0 * base) / (tries + 2.0), np.nan)
                add(f"pair{a}_{b}_hl{hl}", prate)

    # 6. episode peers: recent vs long-run win rates, shared EWMA gaps
    epW, epP = W.sum(1, keepdims=True), P.sum(1, keepdims=True)
    pW, pP = excum(epW - W), excum(epP - P)
    for w in (8, 32, T):
        add(f"peer_rate{w}", (pW - shift(pW, w) + 1.0) / (pP - shift(pP, w) + 2.0))
    for hl_a, hl_b in ((3, 24), (8, 72)):
        g = rate[hl_a] - rate[hl_b]
        add(f"peer_gap{hl_a}_{hl_b}", (g.sum(1, keepdims=True) - g) / (M - 1))

    # 7. payout-table evidence (table 1 vs 0) at several memory lengths
    tix = np.array([[TYPES.index(t) for t in row] for row in types])
    lr = np.log(np.maximum(tables[:, 1], 1e-4)) - np.log(np.maximum(tables[:, 0], 1e-4))  # [3, 4]
    lr_m = lr[tix]  # [E, M, 4]
    for hl in MIX_HL:
        add(f"mix_llr{int(min(hl, 9999))}", mix_llr(x, lr_m, hl))

    # 8. static / bookkeeping
    add("type", tix[..., None].astype(np.float64))
    add("signal", np.asarray(signals, dtype=np.float64)[..., None])
    add("tick", np.arange(TT, dtype=np.float64))
    add("n_obs", cumP)
    add("short", np.asarray(short, dtype=np.float64)[:, None, None])

    return np.stack(feats, axis=-1), names
