"""
========================================================================
Paper B — Nitti et al. (2013) Analytical Experiments
"Utility-Based Stopping Interpretation in Bounded Collective Systems"

BongKeun Song | FAU | 2026.05

데이터 소스:
  Nitti et al. (2013) supplementary xlsx
  SCM / PSO / MIPA trajectory data
  Expected format: sheet per algorithm/function combo
  Columns: [N (swarm size), performance_metric]

Utility 정의 (Bingöl / Snow 코드와 완전 통일):
  U(N) = λ·C(N) - (1-λ)·N/N_max
  λ  : performance weight ∈ {0.3, 0.5, 0.7, 0.9}
  N_max : sheet별 max observed N (domain ceiling)

실험 구성:
  PN_N1 : Ceiling fit + AIC/BIC (inverse-sqrt vs 경쟁 모델)
  PN_N2 : Utility-optimal N* computation
  PN_N3 : ε-stopping ↔ utility-stopping bridge
  PN_N4 : Failure regime analysis (landscape별 붕괴 조건)
  PN_N5 : GO/NO-GO checklist

Output: paperB_nitti_results.json + paperB_nitti_figures.png

Dependencies: numpy, scipy, matplotlib, pandas, openpyxl

사용법:
  python paperB_nitti.py --data ./nitti_data.xlsx
  python paperB_nitti.py  (DATA_PATH 기본값 사용)
========================================================================
"""

import numpy as np
import json
import sys
import time
import argparse
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'reproduced' / 'legacy_components'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit, brentq

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not found. Install: pip install pandas openpyxl")

warnings.filterwarnings('ignore')


# ====================================================================
# SECTION 0: CONFIG
# ====================================================================

DATA_PATH = './nitti_data.xlsx'

# Utility parameters — UNIFIED with Bingöl / Snow codes
LAMBDA_VALUES  = [0.25, 0.5, 0.75, 0.9]
EPSILON_THRESHOLD = 0.01    # Nitti data has wider spacing → 1% threshold

MIN_R2_ACCEPT  = 0.70       # Paper B pilot 기준 (42/63 = 0.67 → 0.70)

# Landscape classification (for failure regime analysis)
# Nitti covers: Sphere, Ackley, Griewank, Schwefel, Styblinski, Rastrigin
EASY_LANDSCAPES  = {'Sphere', 'Ackley', 'Griewank', 'Styblinski'}
HARD_LANDSCAPES  = {'Schwefel', 'Rastrigin'}   # 고차원 rugged → 붕괴 예상


# ====================================================================
# SECTION 1: MODEL FUNCTIONS (Bingöl / Snow 코드와 동일)
# ====================================================================

def ceiling_model(N, a, b):
    """Paper B core: C(N) = a - b/√N  (a=ceiling, b>0)"""
    return a - b / np.sqrt(N)

def exponential_sat(N, a, b):
    """C(N) = a(1 - e^{-bN})"""
    return a * (1.0 - np.exp(-b * N))

def logistic_3p(N, L, k, N0):
    return L / (1.0 + np.exp(-k * (N - N0)))

def michaelis(N, Vmax, Km):
    return Vmax * N / (Km + N)

def logarithmic(N, a, b):
    return a + b * np.log(np.maximum(N, 1e-6))

def aic_val(n_pts, k_params, rss):
    if rss <= 0: return -np.inf
    return n_pts * np.log(rss / n_pts) + 2.0 * k_params

def bic_val(n_pts, k_params, rss):
    if rss <= 0: return -np.inf
    return n_pts * np.log(rss / n_pts) + k_params * np.log(n_pts)

def _mm(n, c0, amp, k):     return c0 + amp * n / (k + n)
def _logsat(n, c0, amp, k): x = np.log1p(n); return c0 + amp * x / (k + x)
def _invsqrt(n, L, amp, k): return L - amp / np.sqrt(n + k)
_RULE1_FAMILIES = {
    "michaelis":      (_mm,     [0.5, 0.5, 2.0], ([0, 0, 1e-6], [1.2, 1.2, 1000.0])),
    "log_saturating": (_logsat, [0.5, 0.5, 1.0], ([0, 0, 1e-6], [1.2, 1.2, 1000.0])),
    "inverse_sqrt":   (_invsqrt,[1.0, 0.5, 1.0], ([0, 0, 1e-6], [1.2, 1.2, 1000.0])),
}
def _rule1_asym(fam, p):
    return float(p[0] + p[1]) if fam in ("michaelis", "log_saturating") else float(p[0])

def fit_primary_family(ns, y):
    """Refit with the same three admissible families as the primary closure.
    Returns (C_func, ceiling, admissible_flag) or (None, None, False)."""
    ns = np.asarray(ns, float); y = np.asarray(y, float)
    cands = []
    for fam, (f, p0, b) in _RULE1_FAMILIES.items():
        try:
            p, _ = curve_fit(f, ns, y, p0=p0, bounds=b, maxfev=200000)
            rss = float(np.sum((y - f(ns, *p)) ** 2))
            aic = len(ns) * np.log(max(rss, 1e-300) / len(ns)) + 6.0
            g = np.arange(1, int(ns.max()) + 1, dtype=float)
            mono = bool(np.all(np.diff(f(g, *p)) >= -1e-9))
            a = _rule1_asym(fam, p)
            cands.append((aic, fam, p, bool(mono and a >= y.max() - 1e-4), a))
        except Exception:
            pass
    if not cands:
        return None, None, False
    adm = [c for c in cands if c[3]]
    best = min(adm or cands)
    _, fam, p, admissible, a = best
    f = _RULE1_FAMILIES[fam][0]
    return (lambda n, _f=f, _p=p: _f(np.asarray(n, float), *_p)), float(a), bool(admissible)

def n95_from_fit(C_func, ceiling, search=1000):
    c1 = float(C_func(1.0))
    if not np.isfinite(ceiling) or ceiling <= c1 + 1e-12:
        return None
    target = c1 + 0.95 * (ceiling - c1)
    g = np.arange(1, search + 1, dtype=float)
    hit = np.where(C_func(g) >= target)[0]
    return None if len(hit) == 0 else float(g[hit[0]])

def s_normalized(C_func, N, N95):
    c1 = float(C_func(1.0)); den = float(C_func(float(N95))) - c1
    if abs(den) < 1e-12: return 0.0
    return float(np.clip((float(C_func(float(N))) - c1) / den, 0.0, 1.5))


def utility(C_N, N, lam, N_max):
    """
    U(N) = λ·C(N) - (1-λ)·N/N_max
    UNIFIED with Bingöl and Snow codes.
    """
    return lam * C_N - (1.0 - lam) * N / N_max

def marginal_gain_discrete(C_arr, idx):
    """δ(N_i) = (C(N_{i+1}) - C(N_i)) / |C(N_i)|"""
    if idx >= len(C_arr) - 1:
        return 0.0
    denom = abs(C_arr[idx])
    if denom < 1e-12:
        return 0.0
    return (C_arr[idx + 1] - C_arr[idx]) / denom


# ====================================================================
# SECTION 2: DATA LOADING
# ====================================================================

def load_nitti_xlsx(path):
    """
    Load Nitti supplementary xlsx.

    Expected format (each sheet):
      Column 0: N (swarm size, integer)
      Column 1: performance metric (e.g., best fitness, accuracy)

    Preprocessing:
      - Drop NaN rows
      - Sort by N ascending
      - Remove duplicate N values (keep mean)
      - Normalize performance to [0, 1] if not already bounded

    Returns: dict {sheet_name: {'N': array, 'C': array, 'N_max': float}}
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for xlsx loading")

    xls = pd.ExcelFile(path)
    sheets = {}

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=0)
            df = df.dropna()

            if df.shape[1] < 2:
                print(f"  SKIP {sheet}: insufficient columns")
                continue

            N_raw = df.iloc[:, 0].values.astype(float)
            C_raw = df.iloc[:, 1].values.astype(float)

            # Sort
            order = np.argsort(N_raw)
            N_raw = N_raw[order]
            C_raw = C_raw[order]

            # Deduplicate N: keep mean per unique N
            unique_N = np.unique(N_raw)
            C_mean = np.array([np.mean(C_raw[N_raw == n]) for n in unique_N])

            # Normalize performance to [0, 1]
            # Nitti uses fitness error (lower = better) or accuracy (higher = better)
            # Auto-detect: if most values > 1 → likely error → invert + normalize
            if np.median(C_mean) > 1.0:
                # Assume error metric: normalize and invert
                C_norm = 1.0 - (C_mean - C_mean.min()) / (C_mean.max() - C_mean.min() + 1e-12)
            elif C_mean.max() > 1.0:
                # Scale to [0,1]
                C_norm = C_mean / C_mean.max()
            else:
                C_norm = C_mean.copy()

            # Ensure monotonically non-decreasing (performance improves with N)
            # If not: flag as non-cooperative regime
            is_monotone = all(C_norm[i] <= C_norm[i+1] + 0.02
                              for i in range(len(C_norm)-1))

            sheets[sheet] = {
                'N':          unique_N,
                'C':          C_norm,
                'N_max':      float(unique_N.max()),
                'is_monotone': is_monotone,
                'raw_metric': 'inverted_error' if np.median(C_mean) > 1.0 else 'direct',
            }

        except Exception as e:
            print(f"  ERROR loading {sheet}: {e}")

    print(f"  Loaded {len(sheets)} sheets from {path}")
    return sheets


# ====================================================================
# SECTION 3: CORE FIT FUNCTION
# ====================================================================

def fit_all_models(N, C, alpha_upper=2.0):
    """
    Fit all 5 models, return dict of results.
    alpha_upper: upper bound for ceiling parameter.
    """
    n_pts = len(N)
    results = {}

    specs = [
        ('InvSqrt',    ceiling_model,  2,
         [max(C)*1.05, 0.5],  ([0.0, 0.0], [alpha_upper, 20.0])),
        ('Exponential',exponential_sat,2,
         [max(C)*1.05, 0.1],  ([0.0, 0.0], [alpha_upper, 5.0])),
        ('Logistic',   logistic_3p,    3,
         [max(C)*1.05, 0.1, np.median(N)],
         ([0.0, 0.0, N.min()], [alpha_upper, 5.0, N.max()*2])),
        ('Michaelis',  michaelis,      2,
         [max(C)*1.05, np.median(N)], ([0.0, 0.01], [alpha_upper, N.max()*5])),
        ('Logarithmic',logarithmic,    2,
         [0.5*max(C), 0.05],  ([-2.0, 0.0], [alpha_upper, 2.0])),
    ]

    for name, func, kp, p0, bounds_ in specs:
        try:
            popt, _ = curve_fit(func, N, C, p0=p0, bounds=bounds_,
                                maxfev=15000)
            pred = func(N, *popt)
            rss  = float(np.sum((C - pred)**2))
            ss_t = float(np.sum((C - np.mean(C))**2))
            r2   = 1.0 - rss/ss_t if ss_t > 1e-12 else 0.0
            a_v  = aic_val(n_pts, kp, rss)
            b_v  = bic_val(n_pts, kp, rss)
            results[name] = {
                'params': [float(p) for p in popt],
                'r2':     float(r2),
                'aic':    float(a_v),
                'bic':    float(b_v),
                'ceiling': float(popt[0]),
            }
        except Exception:
            results[name] = None

    return results


# ====================================================================
# SECTION 4: EXPERIMENT PN_N1 — Ceiling Fit + AIC/BIC
# ====================================================================
"""
목적: Nitti 각 sheet(algorithm × landscape × dimension)에서
     ceiling model fit + competing model AIC/BIC 비교.

왜 필요한가:
  Nitti 파일럿에서 42/63 conditions이 R²>0.7이었다.
  어떤 조건에서 inverse-sqrt가 경쟁 모델보다 나은지,
  어떤 조건에서 붕괴하는지 체계적으로 매핑한다.

무엇을 증명하는가:
  ① cooperative regime(Ackley, Griewank)에서 inverse-sqrt R²>0.7
  ② rugged landscape(Schwefel)에서 ceiling fit 붕괴 → failure regime
  ③ AIC/BIC 기준으로 inverse-sqrt가 경쟁 모델 대비 어느 위치인가
"""
def run_PN_N1(sheets):
    print("=" * 68)
    print("PN_N1: Ceiling Fit + AIC/BIC per condition")
    print(f"  Acceptance threshold: R² > {MIN_R2_ACCEPT}")
    print("=" * 68)

    results = {}

    accept_count = 0
    total_count  = 0

    print(f"\n  {'Sheet':<30} {'best':>10} {'InvSqrt R²':>11} "
          f"{'ΔAIC':>7} {'regime'}")
    print(f"  {'-'*72}")

    for sheet, d in sheets.items():
        N, C = d['N'], d['C']

        if len(N) < 4:
            print(f"  {sheet:<30} SKIP (n<4)")
            continue

        fits = fit_all_models(N, C, alpha_upper=2.0)

        # Best by AIC
        valid_aics = {k: v['aic'] for k, v in fits.items()
                      if v is not None and not np.isinf(v['aic'])}
        if not valid_aics:
            print(f"  {sheet:<30} ALL FITS FAILED")
            continue

        best_name  = min(valid_aics, key=valid_aics.get)
        best_aic   = valid_aics[best_name]

        inv = fits.get('InvSqrt')
        inv_r2   = inv['r2']   if inv else 0.0
        inv_daic = (inv['aic'] - best_aic) if inv else np.inf

        # Regime classification
        if not d['is_monotone']:
            regime = '✗ non-monotone (retrograde/rugged)'
        elif inv and inv_r2 >= MIN_R2_ACCEPT:
            regime = '✓ cooperative'
        else:
            regime = '⚠ poor fit'

        total_count += 1
        if inv and inv_r2 >= MIN_R2_ACCEPT:
            accept_count += 1

        print(f"  {sheet:<30} {best_name:>10} {inv_r2:>11.4f} "
              f"{inv_daic:>7.2f} {regime}")

        results[sheet] = {
            'fits':        fits,
            'best_model':  best_name,
            'best_aic':    float(best_aic),
            'is_monotone': d['is_monotone'],
            'N_max':       float(d['N_max']),
        }

    print(f"\n  Acceptance rate: {accept_count}/{total_count} "
          f"({100*accept_count/max(total_count,1):.1f}%) "
          f"InvSqrt R²>{MIN_R2_ACCEPT}")

    return results


# ====================================================================
# SECTION 5: EXPERIMENT PN_N2 — Utility-Optimal N*
# ====================================================================
"""
목적: Nitti 각 조건에서 utility-optimal N* 계산.

무엇을 증명하는가:
  ① cooperative regime에서 N* < N_max (stopping 발생)
  ② λ에 따른 N* 변화 (robustness)
  ③ 고차원 rugged landscape: N* 불안정 → limitation
"""
def run_PN_N2(pn_n1_results, sheets):
    print("=" * 68)
    print("PN_N2: Utility-Optimal N* per condition")
    print("  U(N) = λ·C(N) - (1-λ)·N/N_max  [Bingöl-unified]")
    print("=" * 68)

    results = {}
    count_early = {str(lam): 0 for lam in LAMBDA_VALUES}
    count_total = 0

    for sheet, d in pn_n1_results.items():
        inv = d['fits'].get('InvSqrt')
        if inv is None:
            continue

        a, b   = inv['params'][0], inv['params'][1]
        N_max  = d['N_max']
        N_range = np.arange(1, int(N_max) + 1, dtype=float)
        C_func  = lambda n, _a=a, _b=b: ceiling_model(n, _a, _b)

        lam_res = {}
        count_total += 1

        for lam in LAMBDA_VALUES:
            U_vals = np.array([utility(C_func(n), n, lam, N_max)
                               for n in N_range])
            idx    = np.argmax(U_vals)
            n_star = float(N_range[idx])
            u_star = float(U_vals[idx])

            if n_star < N_max * 0.95:
                count_early[str(lam)] += 1

            lam_res[str(lam)] = {
                'N_star': n_star,
                'U_star': u_star,
                'early_stop': bool(n_star < N_max * 0.95),
            }

        results[sheet] = {
            'ceiling_a': float(a),
            'beta_b':    float(b),
            'N_max':     N_max,
            'lambda_results': lam_res,
        }

    print(f"\n  Early stop rate (N* < 0.95·N_max):")
    for lam in LAMBDA_VALUES:
        rate = count_early[str(lam)] / max(count_total, 1)
        print(f"    λ={lam}: {count_early[str(lam)]}/{count_total} "
              f"({100*rate:.1f}%)")

    return results


# ====================================================================
# SECTION 6: EXPERIMENT PN_N3 — ε-stopping ↔ Utility Bridge
# ====================================================================
"""
목적: Nitti 조건별 ε-stopping과 utility-stopping 수치 비교.
Bingöl (PB_B5) / Snow (PS_S3)와 동일한 방법으로 λ_equiv 역산.
"""
def run_PN_N3(pn_n2_results, sheets):
    print("=" * 68)
    print("PN_N3: ε-stopping ↔ Utility-stopping Bridge")
    print(f"  ε threshold: {EPSILON_THRESHOLD}")
    print("=" * 68)

    results = {}
    valid_bridges = 0
    total_bridges = 0

    for sheet, d in pn_n2_results.items():
        # Rule (1): refit with the primary 3 families; budget and ε-search are
        # confined to the observed range (Nitti N_max is already the per-sheet
        # observed maximum, so no extrapolation is involved).
        src = sheets[sheet]
        Ns = np.asarray(src['N'], float); Cs = np.asarray(src['C'], float)
        N_obs_max = int(np.max(Ns))
        N_budget = float(N_obs_max)
        total_bridges += 1

        C_func, ceiling, admissible = fit_primary_family(Ns, Cs)
        if C_func is None or not admissible:
            results[sheet] = {'N_eps_observed': None, 'lambda_equiv': None,
                              'verdict': 'inadmissible fit'}
            continue
        N95 = n95_from_fit(C_func, ceiling)
        if N95 is None:
            results[sheet] = {'N_eps_observed': None, 'lambda_equiv': None,
                              'verdict': 'N95 not identifiable'}
            continue

        # ε-stopping on the fitted curve, absolute marginal gain, observed range
        eps_stop = None
        for n in range(1, N_obs_max):
            if float(C_func(n + 1) - C_func(n)) < EPSILON_THRESHOLD:
                eps_stop = n
                break

        lam_equiv = np.nan; verdict = 'no ε in observed range'
        if eps_stop is not None and eps_stop >= 2:
            dS_prev = s_normalized(C_func, eps_stop, N95)     - s_normalized(C_func, eps_stop - 1, N95)
            dS_at   = s_normalized(C_func, eps_stop + 1, N95) - s_normalized(C_func, eps_stop, N95)
            if dS_prev > 0 and dS_at > 0:
                lam_equiv = 0.5 * (1.0 / (1.0 + N_budget * dS_prev) + 1.0 / (1.0 + N_budget * dS_at))
                verdict = '✓ valid' if 0.2 <= lam_equiv <= 0.99 else '⚠ edge'
                if 0.2 <= lam_equiv <= 0.99:
                    valid_bridges += 1

        results[sheet] = {
            'N_eps_observed': int(eps_stop) if eps_stop is not None else None,
            'N95_fit':        float(N95),
            'lambda_equiv':   float(lam_equiv) if not np.isnan(lam_equiv) else None,
            'verdict':        verdict,
        }

    print(f"\n  Valid bridges (λ_equiv ∈ [0.2, 0.99]): "
          f"{valid_bridges}/{total_bridges} "
          f"({100*valid_bridges/max(total_bridges,1):.1f}%)")
    return results


# ====================================================================
# SECTION 7: EXPERIMENT PN_N4 — Failure Regime Analysis
# ====================================================================
"""
목적: cooperative vs rugged landscape 조건별 inverse-sqrt 성립 여부.

Nitti 파일럿 발견: Schwefel / high-dim rugged에서 inverse-sqrt 붕괴.
이걸 boundary condition evidence로 명시한다.
"""
def run_PN_N4(pn_n1_results):
    print("=" * 68)
    print("PN_N4: Failure Regime Analysis")
    print("  Cooperative vs rugged landscape characterization")
    print("=" * 68)

    results = {}

    # Classify by landscape keyword in sheet name
    easy_r2s = []
    hard_r2s = []

    for sheet, d in pn_n1_results.items():
        inv = d['fits'].get('InvSqrt')
        r2  = inv['r2'] if inv else 0.0
        sheet_upper = sheet.upper()

        is_hard = any(h.upper() in sheet_upper for h in HARD_LANDSCAPES)
        is_easy = any(e.upper() in sheet_upper for e in EASY_LANDSCAPES)

        regime = 'hard' if is_hard else ('easy' if is_easy else 'unknown')

        if regime == 'easy':   easy_r2s.append(r2)
        elif regime == 'hard': hard_r2s.append(r2)

        results[sheet] = {
            'regime':    regime,
            'inv_r2':    float(r2),
            'is_monotone': d['is_monotone'],
            'pass':      bool(r2 >= MIN_R2_ACCEPT and d['is_monotone']),
        }

    # Summary
    print(f"\n  Cooperative landscapes: n={len(easy_r2s)}")
    if easy_r2s:
        print(f"    mean R²={np.mean(easy_r2s):.4f}, "
              f"pass rate={sum(r>=MIN_R2_ACCEPT for r in easy_r2s)}/{len(easy_r2s)}")

    print(f"\n  Rugged landscapes: n={len(hard_r2s)}")
    if hard_r2s:
        print(f"    mean R²={np.mean(hard_r2s):.4f}, "
              f"pass rate={sum(r>=MIN_R2_ACCEPT for r in hard_r2s)}/{len(hard_r2s)}")

    print(f"\n  → Failure condition: high-dimensional rugged landscape")
    print(f"    → boundary condition evidence for Paper B Section: Limitation")

    return results


# ====================================================================
# SECTION 8: VISUALIZATION
# ====================================================================
def make_figures(n1, n2, n3, n4, sheets):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

    # ── Panel 1: R² distribution by model ──
    ax1 = fig.add_subplot(gs[0, 0])
    model_names = ['InvSqrt', 'Exponential', 'Logistic', 'Michaelis', 'Logarithmic']
    r2_by_model = {m: [] for m in model_names}
    for sheet, d in n1.items():
        for m in model_names:
            if d['fits'].get(m):
                r2_by_model[m].append(d['fits'][m]['r2'])

    bp_data = [r2_by_model[m] for m in model_names if r2_by_model[m]]
    bp_labels = [m[:7] for m in model_names if r2_by_model[m]]
    ax1.boxplot(bp_data, labels=bp_labels)
    ax1.axhline(MIN_R2_ACCEPT, color='red', ls='--', lw=1.2,
                label=f'R²={MIN_R2_ACCEPT}')
    ax1.set_ylabel('R²'); ax1.set_title('PN_N1: R² Distribution\nper model', fontsize=9)
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3, axis='y')
    plt.setp(ax1.get_xticklabels(), rotation=20, fontsize=8)

    # ── Panel 2: AIC win rate ──
    ax2 = fig.add_subplot(gs[0, 1])
    from collections import Counter
    best_cnt = Counter(d['best_model'] for d in n1.values())
    names_ = list(best_cnt.keys())
    vals_  = [best_cnt[n] for n in names_]
    bar_cols = ['#2ecc71' if n == 'InvSqrt' else '#3498db' for n in names_]
    ax2.bar(range(len(names_)), vals_, color=bar_cols)
    ax2.set_xticks(range(len(names_)))
    ax2.set_xticklabels([n[:7] for n in names_], rotation=20, fontsize=8)
    ax2.set_ylabel('# conditions (best AIC)')
    ax2.set_title('PN_N1: Best Model by AIC\n(green=InvSqrt)', fontsize=9)
    ax2.grid(alpha=0.3, axis='y')

    # ── Panel 3: N* distribution by λ ──
    ax3 = fig.add_subplot(gs[0, 2])
    for lam in LAMBDA_VALUES:
        n_stars = [d['lambda_results'][str(lam)]['N_star']
                   for d in n2.values()
                   if str(lam) in d.get('lambda_results', {})]
        n_maxes = [d['N_max'] for d in n2.values()
                   if str(lam) in d.get('lambda_results', {})]
        ratios  = [ns/nm for ns, nm in zip(n_stars, n_maxes) if nm > 0]
        if ratios:
            ax3.hist(ratios, bins=15, alpha=0.5, label=f'λ={lam}')
    ax3.axvline(1.0, color='black', ls='--', lw=1.2)
    ax3.set_xlabel('N*/N_max ratio')
    ax3.set_ylabel('Count')
    ax3.set_title('PN_N2: N*/N_max distribution\n(<1 = early stop)', fontsize=9)
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    # ── Panel 4: λ_equiv distribution ──
    ax4 = fig.add_subplot(gs[1, 0])
    leqs = [v['lambda_equiv'] for v in n3.values()
            if v['lambda_equiv'] is not None]
    if leqs:
        ax4.hist(leqs, bins=15, color='#3498db', alpha=0.8)
    ax4.axvline(0.2,  color='green', ls='--', lw=1.2, label='0.2')
    ax4.axvline(0.99, color='red',   ls='--', lw=1.2, label='0.99')
    ax4.set_xlabel('λ_equiv')
    ax4.set_ylabel('Count')
    ax4.set_title('PN_N3: λ_equiv distribution\n(valid if 0.2–0.99)', fontsize=9)
    ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

    # ── Panel 5: Failure regime R² comparison ──
    ax5 = fig.add_subplot(gs[1, 1])
    easy_r2s_ = [d['inv_r2'] for d in n4.values() if d['regime'] == 'easy']
    hard_r2s_ = [d['inv_r2'] for d in n4.values() if d['regime'] == 'hard']
    unkn_r2s_ = [d['inv_r2'] for d in n4.values() if d['regime'] == 'unknown']
    data_box  = [x for x in [easy_r2s_, hard_r2s_, unkn_r2s_] if x]
    labels_bx = [l for l, x in
                 zip(['Easy', 'Hard(rugged)', 'Unknown'],
                     [easy_r2s_, hard_r2s_, unkn_r2s_]) if x]
    if data_box:
        ax5.boxplot(data_box, labels=labels_bx)
    ax5.axhline(MIN_R2_ACCEPT, color='red', ls='--', lw=1.2,
                label=f'R²={MIN_R2_ACCEPT}')
    ax5.set_ylabel('InvSqrt R²')
    ax5.set_title('PN_N4: Failure Regime\nR² by landscape type', fontsize=9)
    ax5.legend(fontsize=7); ax5.grid(alpha=0.3, axis='y')

    # ── Panel 6: Representative fit curve (first passing sheet) ──
    ax6 = fig.add_subplot(gs[1, 2])
    plotted = 0
    for sheet, d in n1.items():
        inv = d['fits'].get('InvSqrt')
        if inv is None or inv['r2'] < MIN_R2_ACCEPT:
            continue
        N_sh = sheets[sheet]['N']
        C_sh = sheets[sheet]['C']
        a, b = inv['params'][:2]
        N_fine = np.linspace(N_sh.min(), N_sh.max() * 1.5, 200)
        ax6.scatter(N_sh, C_sh, s=25, alpha=0.7)
        ax6.plot(N_fine, ceiling_model(N_fine, a, b), lw=1.5,
                 label=f"{sheet[:12]} R²={inv['r2']:.3f}")
        plotted += 1
        if plotted >= 5:
            break
    ax6.set_xlabel('N (swarm size)')
    ax6.set_ylabel('Performance C(N)')
    ax6.set_title('PN_N1: Sample Fits\n(InvSqrt, top passing)', fontsize=9)
    ax6.legend(fontsize=6); ax6.grid(alpha=0.3)

    plt.suptitle(
        "Paper B — Nitti et al. Analytical Experiments\n"
        "Swarm Optimization Domain: Utility-Based Stopping Framework",
        fontsize=12, fontweight='bold', y=1.01)

    out_path = str(OUTPUT_DIR / 'paperB_nitti_figures.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figures saved: {out_path}")
    return out_path


# ====================================================================
# SECTION 9: MAIN + GO/NO-GO
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DATA_PATH,
                        help='Path to nitti xlsx file')
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "=" * 68)
    print("  Paper B — Nitti Analytical Experiments")
    print("  Utility definition UNIFIED with Bingöl / Snow codes")
    print("=" * 68 + "\n")

    # Load
    print(f"  Loading: {args.data}")
    sheets = load_nitti_xlsx(args.data)
    if not sheets:
        print("  ERROR: No sheets loaded. Check DATA_PATH and xlsx format.")
        return
    print()

    n1 = run_PN_N1(sheets);       print()
    n2 = run_PN_N2(n1, sheets);   print()
    n3 = run_PN_N3(n2, sheets);   print()
    n4 = run_PN_N4(n1);           print()

    fig_path = make_figures(n1, n2, n3, n4, sheets)

    output = {
        'metadata': {
            'source':      'Nitti et al. workbook bundled as data/legacy_components/nitti_data.xlsx',
            'timestamp':   '1970-01-01 00:00:00',
            'runtime_sec': 0.0,
            'utility_definition':
                'U(N) = λC(N) - (1-λ)N/N_max [Bingöl-unified]',
            'lambda_values':   LAMBDA_VALUES,
            'epsilon_threshold': EPSILON_THRESHOLD,
            'min_r2_accept':   MIN_R2_ACCEPT,
        },
        'PN_N1_ceiling_fit':   n1,
        'PN_N2_utility_Nstar': n2,
        'PN_N3_eps_bridge':    n3,
        'PN_N4_failure':       n4,
    }

    json_path = str(OUTPUT_DIR / 'paperB_nitti_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False,
                  default=lambda x: (float(x) if isinstance(x, (np.floating, np.float64))
                                     else int(x) if isinstance(x, (np.integer,))
                                     else x))
    print(f"\n  Results saved: {json_path}")

    # ── GO/NO-GO ──
    print("\n" + "=" * 68)
    print("  PAPER B GO/NO-GO — Nitti Domain")
    print("=" * 68)

    all_inv_r2 = [d['fits']['InvSqrt']['r2']
                  for d in n1.values()
                  if d['fits'].get('InvSqrt')]
    accept_rate = (sum(r >= MIN_R2_ACCEPT for r in all_inv_r2)
                   / max(len(all_inv_r2), 1))

    checks = []
    checks.append(('PN_N1 InvSqrt accept rate ≥ 50%',
                   accept_rate >= 0.5))

    # Early stop rate at λ=0.5
    early_05 = sum(
        d['lambda_results'].get('0.5', {}).get('early_stop', False)
        for d in n2.values()
    )
    checks.append(('PN_N2 early stop rate ≥ 50% at λ=0.5',
                   early_05 / max(len(n2), 1) >= 0.5))

    # Bridge valid rate
    valid_br = sum(1 for v in n3.values()
                   if v.get('lambda_equiv') is not None
                   and 0.2 <= v['lambda_equiv'] <= 0.99)
    checks.append(('PN_N3 valid bridges ≥ 40%',
                   valid_br / max(len(n3), 1) >= 0.4))

    # Failure regime: easy > hard
    easy_mean = np.mean([d['inv_r2'] for d in n4.values()
                         if d['regime'] == 'easy']) if any(
        d['regime']=='easy' for d in n4.values()) else 0.0
    hard_mean = np.mean([d['inv_r2'] for d in n4.values()
                         if d['regime'] == 'hard']) if any(
        d['regime']=='hard' for d in n4.values()) else 1.0
    checks.append(('PN_N4 easy R² > hard R² (failure regime confirmed)',
                   easy_mean > hard_mean))

    for label, ok in checks:
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {label}")

    all_pass = all(ok for _, ok in checks)
    print(f"\n  Note: Nitti is 'boundary condition evidence', NOT main backbone.")
    print(f"        Partial failure → failure regime section in paper.")
    print(f"\n  Overall: {'GO (boundary evidence)' if all_pass else 'CONDITIONAL — check failures'}")
    print(f"  Runtime: {time.time()-t0:.1f}s\n")


if __name__ == '__main__':
    main()
