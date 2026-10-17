"""Redraw compact main figures at their final single-column dimensions.

Only saved metrics are read. No model fitting or signal processing is performed.
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent
if (R / 'public_package').exists():
    D = R / 'public_package'
    F = R / 'figures'
else:
    D = R.parents[1]
    F = D / 'reproduction/figures'
F.mkdir(parents=True, exist_ok=True)
M = pd.read_csv(D / 'metrics/summary.csv')
CO = ['full', 'low_ber']
COL = ['#BA7D7A', '#617F77']
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    'axes.linewidth': .65, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'pdf.fonttype': 42,
    'axes.axisbelow': True, 'grid.color': '#D5DAD8', 'grid.linewidth': .5,
    'legend.frameon': False,
})

def row(c, experiment, fold='pooled'):
    return M[(M.experiment == c + '/' + experiment) & (M.fold.astype(str) == str(fold))].iloc[0]

def legend(fig):
    fig.legend([Line2D([], [], color=COL[k], marker=['o', 's'][k], ls='none', ms=4)
                for k in range(2)], ['Full primary', 'Low BER'],
               ncol=2, loc='upper center', bbox_to_anchor=(.54, 1),
               handletextpad=.45, columnspacing=1.2)

def save(fig, name):
    # Fixed physical canvas: no tight bounding box that changes final scale.
    for ext in ['pdf', 'png']:
        fig.savefig(F / f'{name}.{ext}', dpi=360, facecolor='white')
    plt.close(fig)

# Two aligned panels preserve both metrics without a full-width figure.
fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.65), sharex=True)
for ax, key, letter, title in zip(axes, ['macro_f1', 'accuracy'], 'ab', ['Macro-F1', 'Accuracy']):
    for k, c in enumerate(CO):
        rs = [row(c, 'baseline_' + n) for n in ['logistic_regression', 'random_forest', 'lightgbm', 'mlp']]
        v = np.array([r[key] for r in rs]); x = np.arange(4) + (k-.5)*.36
        ax.bar(x, v, width=.32, color=COL[k], alpha=.88, edgecolor='white', linewidth=.5)
        hi = np.array([r[key+'_high'] for r in rs]); lo = np.array([r[key+'_low'] for r in rs])
        ax.errorbar(x, v, yerr=[v-lo, hi-v], fmt='none', ecolor='#414947', lw=.65, capsize=1.8)
        for xx, vv, hh in zip(x, v, hi):
            ax.text(xx, hh + (.065 if k == 0 else .012), f'{vv:.3f}', color=COL[k],
                    ha='center', va='bottom', fontsize=8.1)
    ax.set(ylim=(0, 1.11), ylabel=title, yticks=[0, .25, .5, .75, 1])
    ax.set_title(f'({letter}) {title}', loc='left', pad=4)
    ax.grid(axis='y', alpha=.65)
axes[-1].set_xticks(range(4), ['LR', 'RF', 'LightGBM', 'MLP'])
legend(fig); fig.subplots_adjust(left=.17, right=.98, bottom=.085, top=.86, hspace=.38)
save(fig, '01_baselines')

# Compact forests retain row labels and numeric estimates at readable size.
fig, ax = plt.subplots(figsize=(3.4, 4.4))
folds = M[(M.experiment == 'full/holdout_exact_cell') & (M.fold.astype(str) != 'pooled')].fold.tolist()
for j, fold in enumerate(folds):
    if j % 2 == 0: ax.axhspan(j-.49, j+.49, color='#F1F4F2', zorder=0)
    for k, c in enumerate(CO):
        r = row(c, 'holdout_exact_cell', fold); y = j+(k-.5)*.32
        ax.errorbar(r.macro_f1, y, xerr=[[r.macro_f1-r.macro_f1_low], [r.macro_f1_high-r.macro_f1]],
                    fmt=['o', 's'][k], color=COL[k], ms=3.5, capsize=1.6, lw=.8)
        ax.text(1.035, y, f'{r.macro_f1:.3f}', color=COL[k], va='center', clip_on=False, fontsize=8.3)
labels = [re.sub(r'(?<![0-9])4QAM', 'QPSK', str(s)).replace('|', ' / ') for s in folds]
ax.set(yticks=range(len(folds)), yticklabels=labels, xlabel='Macro-F1', xlim=(0, 1.02),
       ylim=(len(folds)-.4, -.6), xticks=[0, .25, .5, .75, 1])
ax.tick_params(axis='y', length=0, pad=4); ax.grid(axis='x', alpha=.7)
legend(fig); fig.subplots_adjust(left=.37, right=.865, bottom=.105, top=.90)
save(fig, 'exact_cell_holdout')

fig, ax = plt.subplots(figsize=(3.4, 3.35))
reps = ['modality_iq_ds', 'modality_amp_seq', 'modality_phase_seq', 'modality_psd_seq',
        'baseline_logistic_regression', 'modality_F32_plus_psd']
names = ['I/Q + PCA', 'Intensity + PCA', 'Phase + PCA', 'PSD + PCA', 'Structured', 'Structured +\nPSD fusion']
for j, n in enumerate(reps):
    ax.axhspan(j-.49, j+.49, color=['#F4F0EC', '#EEF3F0'][j//3], zorder=0)
    for k, c in enumerate(CO):
        r = row(c, n); y = j+(k-.5)*.34
        ax.errorbar(r.macro_f1, y, xerr=[[r.macro_f1-r.macro_f1_low], [r.macro_f1_high-r.macro_f1]],
                    fmt=['o', 's'][k], color=COL[k], ms=3.6, capsize=1.8, lw=.9)
        ax.text(1.035, y, f'{r.macro_f1:.3f}', va='center', color=COL[k], clip_on=False, fontsize=8.3)
ax.set(yticks=range(6), yticklabels=names, xlim=(0, 1.02), xlabel='Macro-F1',
       ylim=(5.65, -.65), xticks=[0, .25, .5, .75, 1])
ax.tick_params(axis='y', length=0, pad=4); ax.grid(axis='x', alpha=.7)
legend(fig); fig.subplots_adjust(left=.35, right=.865, top=.87, bottom=.14)
save(fig, 'representation_comparison')
print('Redrew three figures at 3.4-inch width; unchanged saved metrics.')
