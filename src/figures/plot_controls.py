"""Supplementary control evidence from saved metrics, with no model fitting."""
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
if (HERE/'public_package').exists():
    root=HERE/'public_package';dest=HERE/'figures'
else:
    root=HERE.parents[1];dest=root/'figures/supplementary'
dest.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(root/'metrics/summary.csv');m=m[m.fold.astype(str).eq('pooled')].set_index('experiment')
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,'axes.labelsize':11,'axes.titlesize':11,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'pdf.fonttype':42})
def save(fig,name):
    fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');fig.savefig(dest/(name+'.png'),dpi=250,bbox_inches='tight');plt.close(fig)
names=['metadata_control','ablation_power_only','ablation_no_power','baseline_lightgbm']
labels=['Condition metadata only','Three power descriptors','Without power descriptors','All 32 signal descriptors']
fig,ax=plt.subplots(figsize=(7.2,3.15),layout='constrained')
for y,(n,label,c) in enumerate(zip(names,labels,['#7D8794','#B4A17B','#6F9D92','#B86978'])):
    r=m.loc['full/'+n];v=r.macro_f1
    ax.errorbar(v,y,xerr=[[v-r.macro_f1_low],[r.macro_f1_high-v]],fmt='o',color=c,markersize=7,capsize=3,lw=1.5)
    ax.text(.975,y,f'{v:.4f}',ha='right',va='center',color=c)
ax.set(yticks=range(4),yticklabels=labels,xlim=(0,1),xlabel='Five-class macro-F1');ax.invert_yaxis();ax.grid(axis='x',alpha=.16);save(fig,'signal_controls')
fig,ax=plt.subplots(figsize=(7.2,3.0),layout='constrained')
for j,(co,c,mark,label) in enumerate([('full','#B86978','o','Full primary'),('low_ber','#6F9D92','s','Low BER')]):
    for k,n in enumerate(['pairwise_distance_F32','pairwise_distance_no_spectral']):
        r=m.loc[co+'/'+n];v=r.macro_f1;y=k+(.12 if j else -.12)
        ax.errorbar(v,y,xerr=[[v-r.macro_f1_low],[r.macro_f1_high-v]],fmt=mark,color=c,capsize=3,markersize=7,label=label if k==0 else None)
        ax.text(v+.025,y,f'{v:.4f}',color=c,va='center')
ax.set(yticks=[0,1],yticklabels=['All 32 descriptors','Without six spectral descriptors'],xlim=(.35,1),xlabel='Pooled bidirectional distance-transfer macro-F1');ax.invert_yaxis();ax.legend(loc='lower left',frameon=False);ax.grid(axis='x',alpha=.16);save(fig,'distance_spectral_control')
print(dest)
