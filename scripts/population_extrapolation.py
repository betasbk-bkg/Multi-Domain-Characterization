"""Family-free extrapolation of the aggregate curves.

This is a check on the fitted saturation families, not a replacement for them: it asks where
the curves saturate when no curve family is assumed, under the stated sampling model.

Each item's observed label distribution is treated as its annotator population, and N labels
are drawn with replacement, so the aggregate can be computed for N far beyond the observed
pool without assuming any curve family.

  accuracy mode      C(N) = mean over items of the expected plurality accuracy, ties scored 1/m
  distribution mode  C(N) = 1 - mean over items of JSD_2(empirical_N, item distribution)

The asymptote is exact: as N grows the plurality converges to the item's modal label, so the
accuracy asymptote is the mean of [gold is the unique mode] + [gold tied among m modes]/m;
the distribution asymptote is 1.

Outputs the population curve, its exact asymptote, the population N95, the stopping counts at
the four representative weights under each dataset's N_support budget, and the log-log slope
of the remaining gain C(inf) - C(N), which is -1 for a Michaelis-Menten tail and -h for the
nested four-parameter form.

Sampling model and its limits. Each item's observed pool (about 51 labels on CIFAR-10H, 100 on
ChaosNLI, median 26 on Snapshot Serengeti) stands in for its annotator population; annotators
are exchangeable and labels are drawn with replacement. The accuracy asymptote is therefore the
plurality of the observed pool, so any gain a fitted family places beyond the pool is zero here
by construction. The distribution-mode curve is computed against the item's full distribution
rather than a fixed reference half, which shifts its level but not its shape. Monte Carlo draws
use a fixed seed.

    python scripts/population_extrapolation.py

Writes expected/diagnostics/population_curves.csv and expected/diagnostics/population_summary.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260709)
NS = list(range(1, 26)) + [30, 35, 40, 45, 50, 60, 70, 80, 100, 125, 150, 200, 300, 500]


def load(ds):
    f = f'data/processed/{ds}/labels_long.csv' + ('.gz' if ds != 'Snapshot_Serengeti' else '')
    lab = pd.read_csv(f, dtype=str, usecols=['item_id', 'label'])
    return lab


def draw_counts(P, N, T):
    """Multinomial counts for every item and trial via a binomial chain. P: (I, S)."""
    I, S = P.shape
    counts = np.zeros((I, T, S), dtype=np.int32)
    remaining = np.full((I, T), N, dtype=np.int64)
    rest = np.ones(I)
    for s in range(S):
        ps = P[:, s]
        q = np.where(rest > 1e-12, np.clip(ps / np.maximum(rest, 1e-12), 0, 1), 0.0)
        if s == S - 1:
            c = remaining
        else:
            c = RNG.binomial(remaining, q[:, None])
        counts[:, :, s] = c
        remaining = remaining - c
        rest = rest - ps
    return counts


def accuracy_curve(ds):
    lab = load(ds)
    gold = pd.read_csv(f'data/processed/{ds}/gold.csv', dtype=str).set_index('item_id').gold_label
    lab = lab[lab.item_id.isin(gold.index)]
    tab = lab.groupby(['item_id', 'label']).size().unstack(fill_value=0)
    items = tab.index
    g = gold.loc[items].values
    # put the gold label in column 0 of a per-item support, other labels after it
    rows, S = [], 0
    for it, gl in zip(items, g):
        r = tab.loc[it]; r = r[r > 0]
        others = r.drop(gl, errors='ignore').sort_values(ascending=False)
        vec = [r.get(gl, 0)] + list(others.values)
        rows.append(np.asarray(vec, float)); S = max(S, len(vec))
    P = np.zeros((len(rows), S))
    for i, v in enumerate(rows):
        P[i, :len(v)] = v / v.sum()
    # exact asymptote
    mx = P.max(axis=1)
    tied = (np.isclose(P, mx[:, None])).sum(axis=1)
    c_inf = float(np.mean(np.where(np.isclose(P[:, 0], mx), 1.0 / tied, 0.0)))
    # only items with more than one label value need simulation
    pure = (P[:, 0] == 1.0) | (P[:, 0] == 0.0) & (P[:, 1:].max(axis=1) == 1.0)
    fixed = np.where(P[:, 0] == 1.0, 1.0, 0.0)
    mixed = ~((P[:, 0] == 1.0) | (P[:, 0] == 0.0))
    # items whose gold never appears contribute 0 at every N; items all-gold contribute 1
    Pm = P[mixed]
    out = []
    for N in NS:
        T = 1500
        acc = np.zeros(Pm.shape[0])
        for chunk in range(0, Pm.shape[0], 400):
            sub = Pm[chunk:chunk + 400]
            c = draw_counts(sub, N, T)
            m = c.max(axis=2)
            ntie = (c == m[:, :, None]).sum(axis=2)
            score = np.where(c[:, :, 0] == m, 1.0 / ntie, 0.0)
            acc[chunk:chunk + 400] = score.mean(axis=1)
        total = (fixed[~mixed].sum() + acc.sum()) / len(P)
        out.append((N, total))
        print(f'  {ds} N={N:4d} C={total:.5f}', flush=True)
    return pd.DataFrame(out, columns=['N', 'C_pop']), c_inf, int(mixed.sum()), len(P)


def jsd2(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(a > 0, a * np.log2(a / np.where(b > 0, b, 1)), 0.0)
        return t.sum(axis=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def distribution_curve(ds):
    lab = load(ds)
    tab = lab.groupby(['item_id', 'label']).size().unstack(fill_value=0)
    P = tab.values.astype(float); P = P / P.sum(axis=1, keepdims=True)
    out = []
    for N in NS:
        T = 400
        tot = 0.0
        for chunk in range(0, P.shape[0], 500):
            sub = P[chunk:chunk + 500]
            c = draw_counts(sub, N, T).astype(float) / N
            d = jsd2(c, sub[:, None, :])
            tot += d.mean(axis=1).sum()
        C = 1 - tot / P.shape[0]
        out.append((N, C))
        print(f'  {ds} N={N:4d} C={C:.5f}', flush=True)
    return pd.DataFrame(out, columns=['N', 'C_pop']), 1.0


REP_LAMBDAS = (0.25, 0.50, 0.75, 0.90)
N_SUPPORT = {'CIFAR-10H': 50, 'ChaosNLI': 50, 'Snapshot_Serengeti': 21}


def stopping_counts(df, c_inf, n_support):
    full = np.arange(1, 501)
    C = np.interp(full, df.N.values, df.C_pop.values)
    S = (C - C[0]) / (c_inf - C[0])
    n95 = int(full[np.argmax(S >= 0.95)])
    out = {}
    for lam in REP_LAMBDAS:
        eta = (1 - lam) / (lam * n_support)
        k = np.arange(1, n_support + 1)
        out[lam] = int(k[np.argmax(S[:n_support] - eta * k)])
    return n95, out, float(1 - S[3]), float(100 * (c_inf - C[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('expected/diagnostics'))
    args = ap.parse_args()
    curves, rows = [], []
    for ds, kind in [('CIFAR-10H', 'accuracy'), ('Snapshot_Serengeti', 'accuracy'), ('ChaosNLI', 'distribution')]:
        if kind == 'accuracy':
            df, cinf, nmix, nall = accuracy_curve(ds)
        else:
            (df, cinf), nmix, nall = distribution_curve(ds), None, None
        df = df.assign(dataset=ds, C_inf=cinf)
        curves.append(df)
        n95, ns, sf, sp = stopping_counts(df, cinf, N_SUPPORT[ds])
        c1 = df.C_pop.iloc[0]; rem = cinf - df.C_pop
        slopes = {}
        for lo in (25, 50, 100):
            t = df[(df.N >= lo) & (rem > 0)]
            slopes[lo] = float(np.polyfit(np.log(t.N), np.log(rem[t.index]), 1)[0])
        row = {'dataset': ds, 'mode': kind, 'C_1': round(float(c1), 5), 'C_inf': round(float(cinf), 5),
               'population_N95': n95, 'tail_slope_N25_500': round(slopes[25], 3),
               'tail_slope_N50_500': round(slopes[50], 3), 'tail_slope_N100_500': round(slopes[100], 3),
               'shortfall_at_4': round(sf, 4), 'shortfall_at_4_points': round(sp, 3)}
        for lam in REP_LAMBDAS:
            row[f'n_star_lambda_{lam:.2f}'] = ns[lam]
        if nmix is not None:
            row['items_simulated'] = nmix; row['items_total'] = nall
        rows.append(row)
        print(row, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(curves).to_csv(args.out / 'population_curves.csv', index=False)
    pd.DataFrame(rows).to_csv(args.out / 'population_summary.csv', index=False)
    print('wrote', args.out / 'population_summary.csv')


if __name__ == '__main__':
    main()
