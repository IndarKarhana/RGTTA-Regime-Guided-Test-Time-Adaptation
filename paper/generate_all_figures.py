#!/usr/bin/env python3
"""
Phase A: Generate all figures + statistical tests for the RGTTA paper.

Outputs:
  - paper/figures/  (all figure PNGs/PDFs)
  - paper/statistical_tests.txt  (all p-values, effect sizes, rankings)

Data source: benchmarks/results/unified_v2_8pol/unified_results_full.csv
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "results" / "unified_v2_8pol" / "unified_results_full.csv"
BATCH_DATA = ROOT / "benchmarks" / "results" / "unified_v2_8pol" / "unified_batch_detail.csv"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'font.family': 'serif',
    'figure.constrained_layout.use': False,
})

# ── Policy configuration ──────────────────────────────────────────────
POLICIES_6 = ['tta', 'ewc', 'dynatta', 'rgtta', 'rgtta_ewc', 'rgtta_dynatta']
DISPLAY = {
    'tta': 'TTA', 'ewc': 'EWC', 'dynatta': 'DynaTTA',
    'rgtta': 'RG-TTA', 'rgtta_ewc': 'RG-EWC', 'rgtta_dynatta': 'RG-DynaTTA',
}
COLORS = {
    'tta': '#4e79a7', 'ewc': '#f28e2b', 'dynatta': '#e15759',
    'rgtta': '#59a14f', 'rgtta_ewc': '#76b7b2', 'rgtta_dynatta': '#edc948',
}
PAIRS = [('tta', 'rgtta'), ('ewc', 'rgtta_ewc'), ('dynatta', 'rgtta_dynatta')]

REAL_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange']
SYNTH_DATASETS = ['synth_stable', 'synth_trend_break', 'synth_slow_drift',
                  'synth_fast_switch', 'synth_recurring', 'synth_volatility',
                  'synth_shock_recovery', 'synth_multi_regime']


# ── Load data ──────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA)
    df = df[df['policy'].isin(POLICIES_6)].copy()
    print(f"Loaded {len(df)} rows, {df['policy'].nunique()} policies, "
          f"{df['dataset'].nunique()} datasets, {df['seed'].nunique()} seeds")
    return df


def seed_average(df):
    """Average over seeds → 224 experiments per policy."""
    group_cols = ['model', 'dataset', 'horizon', 'policy']
    agg = df.groupby(group_cols).agg(
        mse=('mse_mean', 'mean'),
        mse_std=('mse_mean', 'std'),
        time=('total_time', 'mean'),
        time_std=('total_time', 'std'),
    ).reset_index()
    return agg


# ══════════════════════════════════════════════════════════════════════
# A2: STATISTICAL SIGNIFICANCE TESTS
# ══════════════════════════════════════════════════════════════════════

def run_statistical_tests(df, out_file):
    """Run all statistical tests and write results."""
    sa = seed_average(df)
    lines = ["=" * 70, "RGTTA STATISTICAL SIGNIFICANCE TESTS", "=" * 70, ""]

    # ── A2.1: Wilcoxon signed-rank tests (pair-wise) ──────────────────
    lines.append("─" * 50)
    lines.append("A2.1: WILCOXON SIGNED-RANK TESTS (pair-wise)")
    lines.append("─" * 50)

    wilcoxon_results = {}
    for base, rg in PAIRS:
        base_mse = sa[sa['policy'] == base].set_index(['model', 'dataset', 'horizon'])['mse']
        rg_mse = sa[sa['policy'] == rg].set_index(['model', 'dataset', 'horizon'])['mse']
        common_idx = base_mse.index.intersection(rg_mse.index)
        b = base_mse.loc[common_idx].values
        r = rg_mse.loc[common_idx].values

        stat, p = stats.wilcoxon(b, r, alternative='two-sided')
        # Also one-sided: is RG significantly better (lower)?
        stat1, p1 = stats.wilcoxon(b, r, alternative='greater')

        # Effect size: r = Z / sqrt(N)
        n = len(common_idx)
        z = stats.norm.ppf(1 - p / 2)  # approximate Z from p-value
        effect_r = z / np.sqrt(n)

        # Win count
        rg_wins = np.sum(r < b)
        ties = np.sum(r == b)
        base_wins = np.sum(r > b)

        lines.append(f"\n{DISPLAY[base]} vs {DISPLAY[rg]} (N={n}):")
        lines.append(f"  Two-sided: W={stat:.1f}, p={p:.2e}")
        lines.append(f"  One-sided (RG better): W={stat1:.1f}, p={p1:.2e}")
        lines.append(f"  Effect size r = {effect_r:.3f}")
        lines.append(f"  RG wins: {rg_wins}/{n} ({100*rg_wins/n:.1f}%), "
                     f"Base wins: {base_wins}, Ties: {ties}")

        wilcoxon_results[(base, rg)] = {
            'p_two': p, 'p_one': p1, 'stat': stat,
            'effect_r': effect_r, 'n': n, 'rg_wins': rg_wins
        }

    # Bonferroni correction (3 tests)
    lines.append(f"\nBonferroni-corrected α = 0.05/3 = {0.05/3:.4f}")
    for (base, rg), res in wilcoxon_results.items():
        sig = "✓ SIGNIFICANT" if res['p_two'] < 0.05 / 3 else "✗ not significant"
        lines.append(f"  {DISPLAY[base]}→{DISPLAY[rg]}: p={res['p_two']:.2e} → {sig}")

    # ── A2.2: Friedman test + Nemenyi post-hoc ────────────────────────
    lines.append("\n" + "─" * 50)
    lines.append("A2.2: FRIEDMAN TEST + NEMENYI POST-HOC")
    lines.append("─" * 50)

    # Pivot: each experiment → rank across 6 policies
    pivot = sa.pivot_table(index=['model', 'dataset', 'horizon'],
                           columns='policy', values='mse')
    pivot = pivot[POLICIES_6].dropna()  # ensure all 6 present
    n_exp = len(pivot)
    lines.append(f"\nExperiments with all 6 policies: {n_exp}")

    # Rank each row (1 = best = lowest MSE)
    ranks = pivot.rank(axis=1, method='average')
    avg_ranks = ranks.mean()

    lines.append("\nAverage ranks (lower is better):")
    for pol in POLICIES_6:
        lines.append(f"  {DISPLAY[pol]:12s}: {avg_ranks[pol]:.3f}")

    # Friedman test
    policy_arrays = [pivot[pol].values for pol in POLICIES_6]
    friedman_stat, friedman_p = stats.friedmanchisquare(*policy_arrays)
    lines.append(f"\nFriedman χ² = {friedman_stat:.2f}, p = {friedman_p:.2e}")
    lines.append(f"  {'✓ SIGNIFICANT' if friedman_p < 0.05 else '✗ not significant'} at α=0.05")

    # Nemenyi post-hoc: CD = q_α * sqrt(k*(k+1) / (6*N))
    k = 6  # number of policies
    N = n_exp
    # Critical values for Nemenyi test (k=6, α=0.05): q_α ≈ 2.850
    q_alpha = 2.850  # from Nemenyi table for k=6, α=0.05
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    lines.append(f"\nNemenyi Critical Difference (CD) at α=0.05: {cd:.3f}")
    lines.append(f"  (q_α={q_alpha}, k={k}, N={N})")

    lines.append("\nPairwise rank differences vs CD:")
    nemenyi_results = {}
    for i, j in combinations(range(k), 2):
        p1_name, p2_name = POLICIES_6[i], POLICIES_6[j]
        diff = abs(avg_ranks[p1_name] - avg_ranks[p2_name])
        sig = "SIG" if diff > cd else "n.s."
        lines.append(f"  {DISPLAY[p1_name]:12s} vs {DISPLAY[p2_name]:12s}: "
                     f"|Δrank|={diff:.3f} {'> ' if diff > cd else '< '}{cd:.3f} → {sig}")
        nemenyi_results[(p1_name, p2_name)] = {'diff': diff, 'sig': diff > cd}

    # ── A2.4: Real-world vs Synthetic win rates ───────────────────────
    lines.append("\n" + "─" * 50)
    lines.append("A2.4: REAL-WORLD vs SYNTHETIC WIN RATES")
    lines.append("─" * 50)

    for subset_name, datasets in [("Real-world (6)", REAL_DATASETS),
                                   ("Synthetic (8)", SYNTH_DATASETS)]:
        sub = sa[sa['dataset'].isin(datasets)]
        pivot_sub = sub.pivot_table(index=['model', 'dataset', 'horizon'],
                                     columns='policy', values='mse')
        pivot_sub = pivot_sub[POLICIES_6].dropna()
        n_sub = len(pivot_sub)

        # Count wins
        wins = {}
        for pol in POLICIES_6:
            wins[pol] = (pivot_sub[pol] == pivot_sub.min(axis=1)).sum()

        rg_total = sum(wins[p] for p in ['rgtta', 'rgtta_ewc', 'rgtta_dynatta'])
        base_total = sum(wins[p] for p in ['tta', 'ewc', 'dynatta'])

        lines.append(f"\n{subset_name} — {n_sub} experiments:")
        for pol in POLICIES_6:
            lines.append(f"  {DISPLAY[pol]:12s}: {wins[pol]:3d} wins ({100*wins[pol]/n_sub:.1f}%)")
        lines.append(f"  RG total: {rg_total}/{n_sub} ({100*rg_total/n_sub:.1f}%)")
        lines.append(f"  Baseline total: {base_total}/{n_sub} ({100*base_total/n_sub:.1f}%)")

    # ── A2.3: Standard deviations across seeds ────────────────────────
    lines.append("\n" + "─" * 50)
    lines.append("A2.3: STANDARD DEVIATIONS ACROSS SEEDS")
    lines.append("─" * 50)

    # Model-level MSE ± std (for Table 4)
    model_stats = sa.groupby(['policy', 'model']).agg(
        mse_mean=('mse', 'mean'),
        mse_seed_std=('mse_std', 'mean'),  # avg of per-experiment stds
    ).reset_index()
    lines.append("\nModel-level MSE (mean ± avg_seed_std):")
    for model in ['gru_small', 'itransformer', 'patchtst', 'dlinear']:
        lines.append(f"\n  {model}:")
        for pol in POLICIES_6:
            row = model_stats[(model_stats['policy'] == pol) & (model_stats['model'] == model)]
            if len(row) > 0:
                m, s = row['mse_mean'].values[0], row['mse_seed_std'].values[0]
                lines.append(f"    {DISPLAY[pol]:12s}: {m:>12,.0f} ± {s:>10,.0f}")

    # Real-world dataset MSE ± std (for Table 5)
    lines.append("\nReal-world MSE ± std (avg across models, horizons, seeds):")
    for ds in REAL_DATASETS:
        ds_data = sa[sa['dataset'] == ds]
        lines.append(f"\n  {ds}:")
        for pol in POLICIES_6:
            pol_data = ds_data[ds_data['policy'] == pol]
            if len(pol_data) > 0:
                m = pol_data['mse'].mean()
                s = pol_data['mse_std'].mean()
                lines.append(f"    {DISPLAY[pol]:12s}: {m:>10.2f} ± {s:>8.2f}")

    # Write results
    with open(out_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n✅ Statistical tests written to {out_file}")

    return avg_ranks, cd, wilcoxon_results, nemenyi_results


# ══════════════════════════════════════════════════════════════════════
# A1: FIGURES
# ══════════════════════════════════════════════════════════════════════

def fig_critical_difference(avg_ranks, cd, save_path):
    """A1.2: Demšar-style Critical Difference diagram."""
    k = len(avg_ranks)
    sorted_policies = sorted(POLICIES_6, key=lambda p: avg_ranks[p])
    sorted_ranks = [avg_ranks[p] for p in sorted_policies]
    sorted_labels = [DISPLAY[p] for p in sorted_policies]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Horizontal axis: rank scale
    low_rank = 1
    high_rank = k
    ax.set_xlim(low_rank - 0.6, high_rank + 0.6)
    ax.set_ylim(-0.8, 2.0)

    # Draw rank axis at top
    ax.hlines(1.2, low_rank, high_rank, color='black', linewidth=1.5)
    for r in range(1, k + 1):
        ax.vlines(r, 1.15, 1.25, color='black', linewidth=1.5)
        ax.text(r, 1.35, str(r), ha='center', va='bottom', fontsize=11)

    # CD bar
    ax.hlines(1.6, low_rank, low_rank + cd, color='red', linewidth=2.5)
    ax.text(low_rank + cd / 2, 1.68, f'CD={cd:.2f}', ha='center', va='bottom',
            fontsize=10, color='red', fontweight='bold')

    # Place policies: top 3 on left, bottom 3 on right
    half = k // 2
    left_policies = sorted_policies[:half]
    right_policies = sorted_policies[half:]

    y_positions = {}

    # Left side (best ranks) — labels on left, wider spacing
    for i, pol in enumerate(left_policies):
        rank = avg_ranks[pol]
        y = 0.85 - i * 0.50
        ax.plot(rank, 1.2, 'ko', markersize=6, zorder=5)
        ax.vlines(rank, y, 1.2, color='gray', linewidth=0.8, linestyle='-')
        is_rg = pol.startswith('rgtta')
        ax.text(low_rank - 0.5, y, f'{DISPLAY[pol]} ({rank:.2f})',
                ha='right', va='center', fontsize=11,
                fontweight='bold' if is_rg else 'normal',
                color=COLORS[pol])
        y_positions[pol] = y

    # Right side (worst ranks) — labels on right, wider spacing
    for i, pol in enumerate(right_policies):
        rank = avg_ranks[pol]
        y = 0.85 - i * 0.50
        ax.plot(rank, 1.2, 'ko', markersize=6, zorder=5)
        ax.vlines(rank, y, 1.2, color='gray', linewidth=0.8, linestyle='-')
        is_rg = pol.startswith('rgtta')
        ax.text(high_rank + 0.5, y, f'({rank:.2f}) {DISPLAY[pol]}',
                ha='left', va='center', fontsize=11,
                fontweight='bold' if is_rg else 'normal',
                color=COLORS[pol])
        y_positions[pol] = y

    # Draw cliques (groups not significantly different)
    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(avg_ranks[sorted_policies[i]] - avg_ranks[sorted_policies[j]]) < cd:
                merged = False
                for clique in cliques:
                    if sorted_policies[i] in clique or sorted_policies[j] in clique:
                        clique.add(sorted_policies[i])
                        clique.add(sorted_policies[j])
                        merged = True
                        break
                if not merged:
                    cliques.append({sorted_policies[i], sorted_policies[j]})

    # Draw thick lines connecting clique members
    clique_y = -0.15
    for clique in cliques:
        members = sorted(clique, key=lambda p: avg_ranks[p])
        r_min = avg_ranks[members[0]]
        r_max = avg_ranks[members[-1]]
        ax.hlines(clique_y, r_min, r_max, color='black', linewidth=3.5)
        clique_y -= 0.20

    ax.set_axis_off()
    ax.set_title('Critical Difference Diagram (Nemenyi, α=0.05)', fontsize=14, pad=35)

    fig.savefig(save_path, format='pdf')
    fig.savefig(str(save_path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  ✅ CD diagram → {save_path}")


def fig_pairwise_waterfall(sa, wilcoxon_results, save_path):
    """A1.3: Pair-wise improvement grouped bar chart with p-values."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    pair_labels = []
    deltas_avg = []
    deltas_med = []
    win_rates = []
    p_values = []

    for base, rg in PAIRS:
        base_mse = sa[sa['policy'] == base].set_index(['model', 'dataset', 'horizon'])['mse']
        rg_mse = sa[sa['policy'] == rg].set_index(['model', 'dataset', 'horizon'])['mse']
        common = base_mse.index.intersection(rg_mse.index)

        b = base_mse.loc[common].values
        r = rg_mse.loc[common].values
        rel_change = (r - b) / b * 100

        pair_labels.append(f'{DISPLAY[base]} → {DISPLAY[rg]}')
        deltas_avg.append(np.mean(rel_change))
        deltas_med.append(np.median(rel_change))
        win_rates.append(np.sum(r < b) / len(common) * 100)
        p_values.append(wilcoxon_results[(base, rg)]['p_one'])

    x = np.arange(len(pair_labels))
    width = 0.3

    bars_avg = ax.bar(x - width / 2, deltas_avg, width, label='Mean Δ%',
                      color=['#59a14f', '#76b7b2', '#edc948'], alpha=0.85,
                      edgecolor='black', linewidth=0.5)
    bars_med = ax.bar(x + width / 2, deltas_med, width, label='Median Δ%',
                      color=['#59a14f', '#76b7b2', '#edc948'], alpha=0.45,
                      edgecolor='black', linewidth=0.5, hatch='///')

    # Add p-value + stars ABOVE the bars (with enough offset)
    for i, (ba, bm) in enumerate(zip(bars_avg, bars_med)):
        # Place annotation above the tallest bar (or below if negative)
        heights = [ba.get_height(), bm.get_height()]
        if max(heights) >= 0:
            y_pos = max(heights) + 1.5
            va = 'bottom'
        else:
            y_pos = min(heights) - 3.0
            va = 'top'
        p = p_values[i]
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.text(i, y_pos, f'p={p:.1e} {stars}', ha='center', va=va,
                fontsize=9, fontweight='bold' if stars != 'n.s.' else 'normal')

    # Add win rate labels INSIDE bars (centered)
    for i, wr in enumerate(win_rates):
        mid_y = min(deltas_avg[i], deltas_med[i]) / 2
        ax.text(i, mid_y, f'{wr:.0f}% wins', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_ylabel('Relative MSE Change (%)', fontsize=12)
    ax.set_title('Pair-wise Effect of Regime-Guidance on MSE', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add vertical margin so annotations don't clip
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 3, ymax + 4)

    fig.tight_layout()
    fig.savefig(save_path, format='pdf')
    fig.savefig(str(save_path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  ✅ Pairwise waterfall → {save_path}")


def fig_dataset_heatmap(sa, save_path):
    """A1.4: Per-dataset win-rate heatmap."""
    pivot = sa.pivot_table(index=['model', 'dataset', 'horizon'],
                           columns='policy', values='mse')
    pivot = pivot[POLICIES_6].dropna()

    # Count wins per dataset per policy (use float64 from the start)
    all_datasets = sorted(sa['dataset'].unique())
    display_cols = [DISPLAY[p] for p in POLICIES_6]
    win_matrix = pd.DataFrame(0.0, index=all_datasets, columns=display_cols)

    total_per_ds = {}
    for ds in all_datasets:
        ds_rows = pivot.loc[pivot.index.get_level_values('dataset') == ds]
        total_per_ds[ds] = len(ds_rows)
        row_mins = ds_rows.min(axis=1)
        for pol in POLICIES_6:
            wins = (ds_rows[pol] == row_mins).sum()
            win_matrix.loc[ds, DISPLAY[pol]] = float(wins)

    # Convert to win rate %
    win_rate = win_matrix.copy()
    for ds in all_datasets:
        if total_per_ds[ds] > 0:
            win_rate.loc[ds] = win_matrix.loc[ds] / total_per_ds[ds] * 100.0

    # Sort datasets: real-world first, then synthetic
    real_order = [d for d in REAL_DATASETS if d in all_datasets]
    synth_order = [d for d in SYNTH_DATASETS if d in all_datasets]
    order = real_order + synth_order
    win_rate = win_rate.loc[order]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    # Use annot=False — seaborn has a bug that stops annotating mid-grid.
    # We add text manually below.
    sns.heatmap(win_rate, annot=False, cmap='RdYlGn',
                center=16.7, vmin=0, vmax=100,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Win Rate (%)', 'shrink': 0.75},
                ax=ax)

    # Manually annotate every cell (bypasses seaborn annot bug)
    nrows, ncols = win_rate.shape
    for i in range(nrows):
        for j in range(ncols):
            val = win_rate.iloc[i, j]
            label = '—' if val == 0 else f'{val:.0f}%'
            ax.text(j + 0.5, i + 0.5, label,
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color='black')

    # Add separator between real and synth
    ax.axhline(len(real_order), color='black', linewidth=2.5)

    ax.set_title('Per-Dataset Win Rate by Policy (%)', fontsize=14, pad=12)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=10, rotation=30)
    ax.tick_params(axis='y', labelsize=10)

    # Annotate real/synth groups — move further left to avoid tick overlap
    ax.text(-1.3, len(real_order) / 2, 'Real', ha='right', va='center',
            fontsize=12, fontweight='bold', rotation=90)
    ax.text(-1.3, len(real_order) + len(synth_order) / 2, 'Synth', ha='right',
            va='center', fontsize=12, fontweight='bold', rotation=90)

    fig.tight_layout(rect=[0.04, 0, 1, 1])  # left margin for group labels
    fig.savefig(save_path, format='pdf')
    fig.savefig(str(save_path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  ✅ Dataset heatmap → {save_path}")


def fig_mse_vs_time(sa, save_path):
    """A1.6: MSE vs wall-clock time scatter (Pareto front)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Overall (all datasets)
    ax = axes[0]
    for pol in POLICIES_6:
        pol_data = sa[sa['policy'] == pol]
        mean_mse = pol_data['mse'].mean()
        mean_time = pol_data['time'].mean()
        mse_se = pol_data['mse'].std() / np.sqrt(len(pol_data))
        time_se = pol_data['time'].std() / np.sqrt(len(pol_data))

        is_rg = pol.startswith('rgtta')
        marker = 's' if is_rg else 'o'
        ax.errorbar(mean_time, mean_mse, xerr=time_se, yerr=mse_se,
                    fmt=marker, markersize=10, color=COLORS[pol],
                    label=DISPLAY[pol], capsize=3, linewidth=1.5,
                    markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Average Adaptation Time (s)')
    ax.set_ylabel('Average MSE')
    ax.set_title('All 14 Datasets')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Real-world only
    ax = axes[1]
    sa_real = sa[sa['dataset'].isin(REAL_DATASETS)]
    for pol in POLICIES_6:
        pol_data = sa_real[sa_real['policy'] == pol]
        if len(pol_data) == 0:
            continue
        mean_mse = pol_data['mse'].mean()
        mean_time = pol_data['time'].mean()
        mse_se = pol_data['mse'].std() / np.sqrt(len(pol_data))
        time_se = pol_data['time'].std() / np.sqrt(len(pol_data))

        is_rg = pol.startswith('rgtta')
        marker = 's' if is_rg else 'o'
        ax.errorbar(mean_time, mean_mse, xerr=time_se, yerr=mse_se,
                    fmt=marker, markersize=10, color=COLORS[pol],
                    label=DISPLAY[pol], capsize=3, linewidth=1.5,
                    markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Average Adaptation Time (s)')
    ax.set_ylabel('Average MSE')
    ax.set_title('Real-World Datasets Only', fontsize=13)
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('MSE vs Adaptation Time Trade-off', fontsize=14)
    fig.savefig(save_path, format='pdf')
    fig.savefig(str(save_path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  ✅ MSE vs Time → {save_path}")


def fig_horizon_analysis(sa, save_path):
    """Additional: MSE by horizon for each policy (line chart)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (title, datasets) in zip(axes, [('All Datasets', None),
                                              ('Real-World Only', REAL_DATASETS)]):
        sub = sa if datasets is None else sa[sa['dataset'].isin(datasets)]
        for pol in POLICIES_6:
            pol_data = sub[sub['policy'] == pol]
            horizon_mse = pol_data.groupby('horizon')['mse'].mean()
            is_rg = pol.startswith('rgtta')
            ax.plot(horizon_mse.index, horizon_mse.values,
                    marker='s' if is_rg else 'o',
                    color=COLORS[pol], label=DISPLAY[pol],
                    linewidth=2 if is_rg else 1.5,
                    linestyle='-' if is_rg else '--',
                    markersize=7)

        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Average MSE')
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_xticks([96, 192, 336, 720])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('MSE Scaling with Forecast Horizon', fontsize=14)
    fig.savefig(save_path, format='pdf')
    fig.savefig(str(save_path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  ✅ Horizon analysis → {save_path}")


def fig_model_comparison(sa, save_path):
    """Bar chart: MSE by model and policy (real-world only)."""
    sa_real = sa[sa['dataset'].isin(REAL_DATASETS)]
    models = ['gru_small', 'itransformer', 'patchtst', 'dlinear']
    model_labels = ['GRU-Small', 'iTransformer', 'PatchTST', 'DLinear']

    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(models))
    width = 0.13
    offsets = np.arange(len(POLICIES_6)) - (len(POLICIES_6) - 1) / 2

    for i, pol in enumerate(POLICIES_6):
        vals = []
        for m in models:
            v = sa_real[(sa_real['policy'] == pol) & (sa_real['model'] == m)]['mse'].mean()
            vals.append(v)
        is_rg = pol.startswith('rgtta')
        ax.bar(x + offsets[i] * width, vals, width,
               label=DISPLAY[pol], color=COLORS[pol],
               edgecolor='black', linewidth=0.5,
               alpha=0.85 if is_rg else 0.6)

    ax.set_ylabel('Average MSE', fontsize=12)
    ax.set_title('Real-World MSE by Model Architecture', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=12)
    ax.legend(ncol=3, fontsize=10, loc='upper right',
              framealpha=0.9, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format='pdf')
    fig.savefig(str(save_path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  ✅ Model comparison → {save_path}")


def fig_rank_distribution(sa, save_path):
    """Box plot of rank distributions across experiments."""
    pivot = sa.pivot_table(index=['model', 'dataset', 'horizon'],
                           columns='policy', values='mse')
    pivot = pivot[POLICIES_6].dropna()
    ranks = pivot.rank(axis=1, method='average')

    fig, ax = plt.subplots(figsize=(9, 5.5))

    rank_data = []
    labels = []
    for pol in POLICIES_6:
        rank_data.append(ranks[pol].values)
        labels.append(DISPLAY[pol])

    bp = ax.boxplot(rank_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops=dict(color='red', linewidth=2))

    for patch, pol in zip(bp['boxes'], POLICIES_6):
        patch.set_facecolor(COLORS[pol])
        patch.set_alpha(0.7)

    ax.set_ylabel('Rank (1=best, 6=worst)')
    ax.set_title('Distribution of Ranks Across 224 Experiments')
    ax.axhline(3.5, color='gray', linestyle='--', alpha=0.5, label='Median rank')
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(save_path, format='pdf')
    fig.savefig(str(save_path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  ✅ Rank distribution → {save_path}")


def fig_adaptation_behavior(batch_df, save_path):
    """A1.5: Adaptation behavior over batches for one representative experiment.
    Columns: model, dataset, horizon, seed, policy, batch, mse, mae, rmse,
             mape, smape, direction_acc, tier, similarity
    """
    target_configs = [
        ('gru_small', 'ETTh2', 96),
        ('gru_small', 'ETTh1', 96),
        ('gru_small', 'synth_recurring', 96),
        ('itransformer', 'ETTh2', 96),
    ]

    for model, dataset, horizon in target_configs:
        sub = batch_df[(batch_df['model'] == model) &
                       (batch_df['dataset'] == dataset) &
                       (batch_df['horizon'] == horizon) &
                       (batch_df['seed'] == 0)]
        if len(sub) == 0:
            continue

        rgtta_data = sub[sub['policy'] == 'rgtta']
        if len(rgtta_data) == 0 or rgtta_data['similarity'].isna().all():
            continue

        print(f"  Using example: {model}/{dataset}/H={horizon}")

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        # Panel 1: MSE across batches for all policies
        ax = axes[0]
        for pol in POLICIES_6:
            pol_data = sub[sub['policy'] == pol].sort_values('batch')
            if len(pol_data) > 0:
                is_rg = pol.startswith('rgtta')
                ax.plot(pol_data['batch'], pol_data['mse'],
                        marker='s' if is_rg else 'o',
                        color=COLORS[pol], label=DISPLAY[pol],
                        linewidth=2 if is_rg else 1.2,
                        linestyle='-' if is_rg else '--',
                        markersize=6, alpha=0.8)
        ax.set_ylabel('Batch MSE', fontsize=11)
        ax.set_title(f'Adaptation Behavior: {model} / {dataset} / H={horizon}', fontsize=14)
        ax.legend(ncol=3, fontsize=9, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3)

        # Panel 2: RG-TTA similarity + threshold lines
        ax = axes[1]
        rgtta_sorted = rgtta_data.sort_values('batch')
        sims = rgtta_sorted['similarity'].values
        batches = rgtta_sorted['batch'].values
        tiers = rgtta_sorted['tier'].values

        ax.plot(batches, sims, 'g-s', label='Similarity', linewidth=2, markersize=7)

        # Color points by tier
        tier_colors = {'easy': '#59a14f', 'hard': '#e15759', 'medium': '#f28e2b'}
        for b, s, t in zip(batches, sims, tiers):
            c = tier_colors.get(str(t).lower().strip(), 'gray')
            ax.plot(b, s, 'o', color=c, markersize=9, zorder=5)

        ax.axhline(0.75, color='red', linestyle='--', alpha=0.7, label='Ckpt threshold (0.75)')
        ax.axhline(0.85, color='orange', linestyle=':', alpha=0.7, label='HIGH (0.85)')
        ax.axhline(0.55, color='blue', linestyle=':', alpha=0.7, label='LOW (0.55)')
        ax.set_ylabel('Regime Similarity', fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.legend(ncol=2, fontsize=9, loc='lower right', framealpha=0.9)
        ax.grid(alpha=0.3)

        # Panel 3: Computed LR from similarity (α = α_base × (1 + 0.67 × (1 - sim)))
        ax = axes[2]
        lr_base = 3e-4
        lr_sim_scale = 0.67
        lr_values = lr_base * (1 + lr_sim_scale * (1 - sims))
        tta_fixed_lr = lr_base

        ax.bar(batches, lr_values * 1e4, color='#59a14f', alpha=0.7,
               label='RG-TTA adaptive LR (×10⁴)')
        ax.axhline(tta_fixed_lr * 1e4, color='#4e79a7', linestyle='--', linewidth=2,
                   label=f'TTA fixed LR ({tta_fixed_lr:.0e})')
        ax.set_ylabel('Learning Rate (×10⁴)', fontsize=11)
        ax.set_xlabel('Batch', fontsize=11)
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3)

        fig.tight_layout(h_pad=1.5)
        fig.savefig(save_path, format='pdf')
        fig.savefig(str(save_path).replace('.pdf', '.png'))
        plt.close(fig)
        print(f"  ✅ Adaptation behavior → {save_path}")
        return

    print(f"  ⚠️ No suitable adaptation behavior data found")


def generate_latex_snippets(wilcoxon_results, avg_ranks, cd, sa):
    """Generate LaTeX-ready snippets for inserting into paper."""
    snippets = []

    # Wilcoxon p-values for Table 3 (pairwise)
    snippets.append("\n% === WILCOXON P-VALUES FOR TABLE 3 (pairwise) ===")
    for (base, rg), res in wilcoxon_results.items():
        p = res['p_one']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        snippets.append(f"% {DISPLAY[base]}→{DISPLAY[rg]}: p={p:.2e} {stars}")

    # Updated Table 3 row (with p-values)
    snippets.append("\n% === UPDATED PAIRWISE TABLE WITH P-VALUES ===")
    snippets.append(r"""% Replace Table 3 (tab:pairwise) with:
\begin{tabular}{llcccc}
\toprule
Baseline & RG-Variant & $\Delta$MSE (avg) & $\Delta$MSE (median) & RG Wins & $p$-value \\
\midrule""")

    for base, rg in PAIRS:
        res = wilcoxon_results[(base, rg)]
        base_mse = sa[sa['policy'] == base].set_index(['model', 'dataset', 'horizon'])['mse']
        rg_mse = sa[sa['policy'] == rg].set_index(['model', 'dataset', 'horizon'])['mse']
        common = base_mse.index.intersection(rg_mse.index)
        b = base_mse.loc[common].values
        r = rg_mse.loc[common].values
        rel = (r - b) / b * 100
        avg_delta = np.mean(rel)
        med_delta = np.median(rel)
        n = len(common)
        rg_wins = np.sum(r < b)

        p = res['p_one']
        if p < 0.001:
            p_str = f"${p:.1e}$***"
        elif p < 0.01:
            p_str = f"${p:.1e}$**"
        elif p < 0.05:
            p_str = f"${p:.2e}$*"
        else:
            p_str = f"${p:.2e}$"

        sign = '+' if avg_delta > 0 else ''
        msign = '+' if med_delta > 0 else ''
        snippets.append(
            f"{DISPLAY[base]} & {DISPLAY[rg]} & ${sign}{avg_delta:.1f}\\%$ & "
            f"${msign}{med_delta:.1f}\\%$ & {rg_wins}/{n} ({100*rg_wins/n:.1f}\\%) & {p_str} \\\\"
        )

    snippets.append(r"""\bottomrule
\end{tabular}""")

    # Real vs Synth summary
    snippets.append("\n% === REAL-WORLD vs SYNTHETIC WIN SUMMARY ===")
    for subset_name, datasets in [("Real-world", REAL_DATASETS),
                                   ("Synthetic", SYNTH_DATASETS)]:
        sub = sa[sa['dataset'].isin(datasets)]
        pivot_sub = sub.pivot_table(index=['model', 'dataset', 'horizon'],
                                     columns='policy', values='mse')
        pivot_sub = pivot_sub[POLICIES_6].dropna()
        n_sub = len(pivot_sub)
        rg_wins = 0
        for _, row in pivot_sub.iterrows():
            winner = row.idxmin()
            if winner in ['rgtta', 'rgtta_ewc', 'rgtta_dynatta']:
                rg_wins += 1
        snippets.append(f"% {subset_name}: RG wins {rg_wins}/{n_sub} ({100*rg_wins/n_sub:.1f}%)")

    out = ROOT / "paper" / "latex_snippets.txt"
    with open(out, 'w') as f:
        f.write('\n'.join(snippets))
    print(f"\n✅ LaTeX snippets written to {out}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def fig_workflow(save_path):
    """System overview: RG-TTA v2 pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-1.2, 5.8)
    ax.axis('off')

    # ── Box styles ────────────────────────────────────────
    input_sty = dict(boxstyle='round,pad=0.4', facecolor='#E8EAF6', edgecolor='#283593', linewidth=1.4)
    feat_sty  = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.4)
    dec_sty   = dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1.4)
    ckpt_sty  = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.4)
    lr_sty    = dict(boxstyle='round,pad=0.4', facecolor='#FCE4EC', edgecolor='#C62828', linewidth=1.4)
    adapt_sty = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.4)
    store_sty = dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1.4)
    out_sty   = dict(boxstyle='round,pad=0.4', facecolor='#ECEFF1', edgecolor='#455A64', linewidth=1.4)
    mem_sty   = dict(boxstyle='round,pad=0.5', facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1.8, linestyle='--')

    # ── Positions — wider spacing to avoid all overlaps ───
    pos = {
        'batch':  (0.8,  2.8),
        'feat':   (3.0,  2.8),
        'sim':    (5.4,  2.8),
        'gate':   (7.8,  4.6),
        'lr':     (7.8,  2.8),
        'adapt':  (10.2, 2.8),
        'store':  (12.6, 2.8),
        'output': (12.6, 0.5),
        'memory': (5.4,  0.0),
    }

    # ── Draw boxes ────────────────────────────────────────
    box_labels = [
        ('batch',  'New Batch\n$B_t$',                      input_sty, 10),
        ('feat',   'Extract\nFeatures $\\mathbf{r}$',       feat_sty,  10),
        ('sim',    'Compute\nSimilarity',                    dec_sty,   10),
        ('gate',   'Loss-Gated\nCkpt Load',                 ckpt_sty,  10),
        ('lr',     'Smooth LR\nModulation',                  lr_sty,    10),
        ('adapt',  'Adapt Head\n(early stop)',               adapt_sty, 10),
        ('store',  'Save to\nMemory',                        store_sty, 10),
        ('output', 'Forecast\n$\\hat{y}_{t+H}$',            out_sty,   10),
        ('memory', 'Checkpoint Memory $\\mathcal{M}$  (5 slots, FIFO)', mem_sty, 9),
    ]

    for key, label, style, fs in box_labels:
        x, y = pos[key]
        ax.text(x, y, label, ha='center', va='center', fontsize=fs,
                bbox=style, fontweight='medium')

    # ── Arrow style ───────────────────────────────────────
    akw = dict(arrowstyle='-|>', color='#444444', linewidth=1.4, mutation_scale=14)

    # Main flow: batch → feat → sim → lr → adapt → store
    for k1, k2 in [('batch','feat'), ('feat','sim'), ('sim','lr'), ('lr','adapt'), ('adapt','store')]:
        ax.annotate('', xy=pos[k2], xytext=pos[k1], arrowprops=akw)

    # Store → output (downward)
    ax.annotate('', xy=pos['output'], xytext=pos['store'], arrowprops=akw)

    # Sim → gate (upward-right)
    ax.annotate('', xy=(pos['gate'][0]-0.8, pos['gate'][1]-0.3),
                xytext=(pos['sim'][0]+0.8, pos['sim'][1]+0.3),
                arrowprops={**akw, 'connectionstyle': 'arc3,rad=-0.15'})

    # Gate → adapt (rightward-down, green)
    ax.annotate('', xy=(pos['adapt'][0]-0.7, pos['adapt'][1]+0.4),
                xytext=(pos['gate'][0]+0.8, pos['gate'][1]-0.4),
                arrowprops={**akw, 'connectionstyle': 'arc3,rad=0.15', 'color': '#2E7D32'})

    # Memory ↔ sim (bidirectional vertical)
    ax.annotate('', xy=(pos['sim'][0]-0.3, pos['sim'][1]-0.6),
                xytext=(pos['memory'][0]-0.3, pos['memory'][1]+0.5),
                arrowprops={**akw, 'arrowstyle': '<->', 'color': '#6A1B9A', 'linewidth': 1.2})

    # Store → memory (save arc)
    ax.annotate('', xy=(pos['memory'][0]+2.0, pos['memory'][1]+0.3),
                xytext=(pos['store'][0]-0.3, pos['store'][1]-0.6),
                arrowprops={**akw, 'connectionstyle': 'arc3,rad=0.35', 'color': '#6A1B9A', 'linewidth': 1.2})

    # ── Labels on arrows / near boxes (all well-spaced) ──

    # Gate condition labels — above the gate box
    ax.text(6.2, 4.1, 'sim $\\geq$ 0.75?', fontsize=8.5,
            color='#2E7D32', fontstyle='italic', ha='center')
    ax.text(9.5, 4.1, '$\\ell_{ckpt} < 0.7\\,\\ell_{curr}$', fontsize=8.5,
            color='#2E7D32', fontstyle='italic', ha='center')

    # Sim detail — below sim box, above memory
    ax.text(5.4, 1.65, 'KS + Wasserstein +\nFeature + Variance', fontsize=8,
            ha='center', color='#666666', fontstyle='italic')

    # LR formula — below LR box
    ax.text(7.8, 1.65, '$\\alpha = \\alpha_{base}(1 + \\gamma(1{-}\\mathrm{sim}))$', fontsize=9,
            ha='center', color='#C62828', fontstyle='italic')

    # Adapt detail — below adapt box
    ax.text(10.2, 1.65, 'Frozen backbone\nmax 25, patience 3', fontsize=8,
            ha='center', color='#666666', fontstyle='italic')

    # Memory labels
    ax.text(4.5, 1.15, 'query', fontsize=8, color='#6A1B9A', fontstyle='italic', ha='center')
    ax.text(9.5, 0.3, 'store ckpt', fontsize=8, color='#6A1B9A', fontstyle='italic', ha='center')

    # Title
    ax.set_title('RG-TTA: Regime-Guided Test-Time Adaptation Pipeline',
                 fontsize=14, pad=18)

    fig.savefig(save_path, format='pdf')
    fig.savefig(str(save_path).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  ✅ Workflow diagram → {save_path}")


def main():
    print("=" * 60)
    print("RGTTA Paper — Phase A: Statistical Tests + Figures")
    print("=" * 60)

    df = load_data()
    sa = seed_average(df)

    # ── A2: Statistical tests ─────────────────────────────────────────
    print("\n📊 Running statistical tests...")
    stats_file = ROOT / "paper" / "statistical_tests.txt"
    avg_ranks, cd, wilcoxon_results, nemenyi_results = run_statistical_tests(df, stats_file)

    # ── Generate LaTeX snippets ───────────────────────────────────────
    print("\n📝 Generating LaTeX snippets...")
    generate_latex_snippets(wilcoxon_results, avg_ranks, cd, sa)

    # ── A1: Figures ───────────────────────────────────────────────────
    print("\n🎨 Generating figures...")

    fig_workflow(FIG_DIR / "workflow.pdf")
    fig_critical_difference(avg_ranks, cd, FIG_DIR / "critical_difference.pdf")
    fig_pairwise_waterfall(sa, wilcoxon_results, FIG_DIR / "pairwise_improvement.pdf")
    fig_dataset_heatmap(sa, FIG_DIR / "dataset_heatmap.pdf")
    fig_mse_vs_time(sa, FIG_DIR / "mse_vs_time.pdf")
    fig_horizon_analysis(sa, FIG_DIR / "horizon_analysis.pdf")
    fig_model_comparison(sa, FIG_DIR / "model_comparison.pdf")
    fig_rank_distribution(sa, FIG_DIR / "rank_distribution.pdf")

    # Adaptation behavior (needs batch detail CSV)
    if BATCH_DATA.exists():
        print("\n🔬 Loading batch detail data for adaptation behavior figure...")
        batch_df = pd.read_csv(BATCH_DATA)
        fig_adaptation_behavior(batch_df, FIG_DIR / "adaptation_behavior.pdf")
    else:
        print(f"\n⚠️ Batch detail CSV not found: {BATCH_DATA}")

    print("\n" + "=" * 60)
    print("✅ Phase A complete!")
    print(f"   Figures:  {FIG_DIR}/")
    print(f"   Stats:    {stats_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
