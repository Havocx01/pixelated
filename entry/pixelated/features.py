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
PAIR_HL = (1, 2, 4, 6, 12, 24)
PAIR_EVID_HL = (2, 6, 24)
TRIPLE_HL = ()  # tested: +0.15, not significant, doubles runtime
RULE_SETTINGS = ((0.05, 0.02), (0.05, 0.1), (0.15, 0.05))
TABLE_RHO = ()  # tested: no gain


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


def ruleInputs(maps, context):
    bits = np.moveaxis(maps[:, np.clip(context, 0, 3)], 0, -1)
    known = np.broadcast_to((context >= 0)[..., None], bits.shape)
    episodes, machines = context.shape[:2]
    return bits.reshape(episodes, machines, -1), known.reshape(episodes, machines, -1)


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

    for k in ONEHOT_LAGS:
        v = shift(x.astype(np.float64), k, fill=-1.0)
        for c in range(-1, 5):
            add(f"lag{k}_c{c}", (v == c).astype(np.float64))

    cum = {c: excum((x == c).astype(np.float64)) for c in range(5)}
    cumP = excum(P)
    for w in WINDOWS:
        n = cumP - shift(cumP, w)
        for c in range(5):
            add(f"freq{w}_c{c}", (cum[c] - shift(cum[c], w) + 0.5) / (n + 2.5))
        add(f"n{w}", n)

    rate = {}
    for hl in EWMA_HL:
        rate[hl] = (disc(W, hl) + 0.5) / (disc(P, hl) + 1.0)
        add(f"ewma{hl}", rate[hl])
    add("gap3_24", rate[3] - rate[24])
    add("gap8_72", rate[8] - rate[72])
    add("gap24_72", rate[24] - rate[72])
    base = (excum(W) + 1.0) / (cumP + 2.0)
    add("winrate_all", base)

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

    # Compare past ticks whose lag-pair states match the current context.
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
            pr = {}
            for hl in PAIR_HL:
                d = disc(stack, hl)
                wins = np.take_along_axis(d[0::2], cur[None], 0)[0]
                tries = np.take_along_axis(d[1::2], cur[None], 0)[0]
                pr[hl] = np.where(ok, (wins + 2.0 * base) / (tries + 2.0), np.nan)
                add(f"pair{a}_{b}_hl{hl}", pr[hl])
                if hl in PAIR_EVID_HL:
                    add(f"pairn{a}_{b}_hl{hl}", np.where(ok, tries, np.nan))
            add(f"pairdiff{a}_{b}", pr[PAIR_HL[0]] - pr[PAIR_HL[-1]])
            # most recent tick with the same context: how long ago, and did it win?
            ticks = np.arange(TT)
            last = np.stack([np.maximum.accumulate(np.where(stack[2 * c + 1] > 0, ticks, -1), axis=-1)
                             for c in range(4)])
            last = np.concatenate([np.full(last.shape[:-1] + (1,), -1), last[..., :-1]], axis=-1)
            lsel = np.take_along_axis(last, cur[None], 0)[0]
            found = ok & (lsel >= 0)
            add(f"pairage{a}_{b}", np.where(found, ticks - lsel, np.nan))
            lwin = np.take_along_axis(W, np.clip(lsel, 0, None), -1)
            add(f"pairlast{a}_{b}", np.where(found, lwin, np.nan))

    for a in LAGS:
        for b in LAGS:
            for c in LAGS:
                if not a < b < c or not TRIPLE_HL:
                    continue
                series = []
                for sa in (1, -1):
                    for sb in (1, -1):
                        for sc in (1, -1):
                            ind = ((S[a] == sa) & (S[b] == sb) & (S[c] == sc)).astype(np.float64)
                            series += [ind * W, ind * P]
                stack = np.stack(series)
                cur = (S[a] == -1) * 4 + (S[b] == -1) * 2 + (S[c] == -1) * 1
                ok = (S[a] != 0) & (S[b] != 0) & (S[c] != 0)
                for hl in TRIPLE_HL:
                    d = disc(stack, hl)
                    wins = np.take_along_axis(d[0::2], cur[None], 0)[0]
                    tries = np.take_along_axis(d[1::2], cur[None], 0)[0]
                    add(f"tri{a}_{b}_{c}_hl{hl}", np.where(ok, (wins + 2.0 * base) / (tries + 2.0), np.nan))

    # Each lag pair has 16 candidate Boolean rules. The posterior allows occasional switches.
    if RULE_SETTINGS:
        pairs = [(a, b) for a in LAGS for b in LAGS if a < b]
        combo = np.stack([np.where((S[a] != 0) & (S[b] != 0),
                                   (S[a] == -1) * 2 + (S[b] == -1) * 1, -1) for a, b in pairs], -1)  # [E,M,T,28]
        maps = np.array([[(m >> c) & 1 for c in range(4)] for m in range(16)], dtype=np.float64)  # [16,4]
        # A full timeline of rule inputs can require several GiB.
        contexts = np.moveaxis(combo, 2, 0)
        R = len(pairs) * 16
        for eps, rho in RULE_SETTINGS:
            post = np.full((E, M, R), 1.0 / R)
            pred = np.zeros((E, M, TT))
            conf = np.zeros((E, M, TT))
            for t, context in enumerate(contexts):
                bits, known = ruleInputs(maps, context)
                q = np.where(known, np.where(bits > 0, 1 - eps, eps), 0.5)
                pred[:, :, t] = (post * q).sum(-1)
                conf[:, :, t] = post.max(-1)
                obs = P[:, :, t:t + 1] > 0
                lik = np.where(W[:, :, t:t + 1] > 0, q, 1 - q)
                post = np.where(obs, post * lik, post)
                post /= post.sum(-1, keepdims=True)
                post = (1 - rho) * post + rho / R
            add(f"rule_e{eps}_r{rho}", pred)
            add(f"rule_conf_e{eps}_r{rho}", conf)

        # Only a win reveals which payout table is more likely.
        tix0 = np.array([[TYPES.index(t) for t in row] for row in types])
        tab = tables[tix0]  # [E,M,2,4]
        size_lik = np.ones((E, M, TT, 2))
        for z in (0, 1):
            for c in range(1, 5):
                size_lik[..., z] = np.where(x == c, tab[:, :, z, c - 1][..., None], size_lik[..., z])
        for rho in TABLE_RHO:
            post = np.full((E, M, R), 1.0 / R)
            pz = np.zeros((E, M, TT))
            for t, context in enumerate(contexts):
                bits, known = ruleInputs(maps, context)
                pz[:, :, t] = (post * np.where(known, bits, 0.5)).sum(-1)
                won = W[:, :, t:t + 1] > 0
                l0, l1 = size_lik[:, :, t, 0:1], size_lik[:, :, t, 1:2]
                lik = np.where(known, np.where(bits > 0, l1, l0), 0.5 * (l0 + l1))
                post = np.where(won, post * lik, post)
                post /= post.sum(-1, keepdims=True)
                post = (1 - rho) * post + rho / R
            add(f"table_rule_r{rho}", pz)

    epW, epP = W.sum(1, keepdims=True), P.sum(1, keepdims=True)
    pW, pP = excum(epW - W), excum(epP - P)
    for w in (8, 32, T):
        add(f"peer_rate{w}", (pW - shift(pW, w) + 1.0) / (pP - shift(pP, w) + 2.0))
    for hl_a, hl_b in ((3, 24), (8, 72)):
        g = rate[hl_a] - rate[hl_b]
        add(f"peer_gap{hl_a}_{hl_b}", (g.sum(1, keepdims=True) - g) / (M - 1))

    tix = np.array([[TYPES.index(t) for t in row] for row in types])
    lr = np.log(np.maximum(tables[:, 1], 1e-4)) - np.log(np.maximum(tables[:, 0], 1e-4))  # [3, 4]
    lr_m = lr[tix]  # [E, M, 4]
    for hl in MIX_HL:
        add(f"mix_llr{int(min(hl, 9999))}", mix_llr(x, lr_m, hl))

    add("type", tix[..., None].astype(np.float64))
    add("signal", np.asarray(signals, dtype=np.float64)[..., None])
    add("tick", np.arange(TT, dtype=np.float64))
    add("n_obs", cumP)
    add("short", np.asarray(short, dtype=np.float64)[:, None, None])

    return np.stack(feats, axis=-1), names
