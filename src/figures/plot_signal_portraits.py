"""Manuscript signal portraits from released derived arrays and tables."""
from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--tensor',type=Path,default=ROOT/'data/tensors/shard.h5')
parser.add_argument('--output',type=Path,default=ROOT/'reproduction/figures')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
CL=['normal','attenuation','edfa','wss_offset','wss_bandwidth']
N=['Normal','VOA','EDFA','WSS offset','WSS bandwidth']
COL=['#697A76','#BA7D7A','#B59A6D','#648BA6','#79A790']
CH=['#648BA6','#BA7D7A','#79A790','#B59A6D']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':9.5,'legend.fontsize':9,
 'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.65,
 'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
 'pdf.fonttype':42,'mathtext.fontset':'stix','legend.frameon':False,
 'axes.axisbelow':True,'grid.alpha':.3,'grid.linewidth':.4})
d=pd.read_csv(ROOT/'data/structured_features_3355.csv').merge(
 pd.read_csv(ROOT/'data_dictionary/capture_index.csv'),on=['sample_id','diagnosis_label'],validate='one_to_one')
freq=pd.read_csv(ROOT/'data_dictionary/psd_frequency_axis.csv').frequency_center_ghz.to_numpy()
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(args.output/f'{name}.{ext}',dpi=320,facecolor='white')
 plt.close(fig)
with h5py.File(args.tensor) as h:
 ids=[s.decode() if isinstance(s,bytes) else str(s) for s in h['sample_id'][:]];lookup={s:i for i,s in enumerate(ids)}
 picked=d[d.modulation.eq('16QAM')&d.distance_km.eq(640)&d.launch_power_dbm.eq(1)].sort_values('sample_id').groupby('diagnosis_label').first().reindex(CL)
 fig,axs=plt.subplots(5,3,figsize=(7.1,5.8))
 for i,c in enumerate(CL):
  ix=lookup[picked.loc[c,'sample_id']]
  for j,mode in enumerate(['iq_ds','amp_seq','psd_seq']):
   ax=axs[i,j];a=h[mode][ix]
   for k,line in enumerate(a):ax.plot(freq if j==2 else np.arange(len(line)),line,color=CH[k],lw=.55,alpha=.85)
   if j==0:ax.set_ylabel(N[i],fontsize=9)
   if i==0:
    ax.set_title(['(a) I/Q (a.u.)','(b) Normalized intensity','(c) Log-spectrum (dB)'][j],loc='left',pad=25)
    labels=[['$I_X$','$Q_X$','$I_Y$','$Q_Y$'],['X','Y'],['X','Y','Mixed']][j]
    ax.legend([Line2D([],[],color=CH[k],lw=1) for k in range(len(labels))],labels,
       ncol=len(labels),loc='lower center',bbox_to_anchor=(.5,1.01),columnspacing=.65,handlelength=1,handletextpad=.3,borderaxespad=0)
   if j==2:ax.set(xlim=(-128,128),xticks=[-100,0,100])
   if i==4:ax.set_xlabel('Frequency (GHz)' if j==2 else 'Array index')
   ax.grid()
 fig.subplots_adjust(left=.145,right=.985,bottom=.085,top=.88,wspace=.38,hspace=.40);save(fig,'representative_modalities')
 cells=d.groupby(['distance_km','modulation','launch_power_dbm']).diagnosis_label.nunique()
 common=set(cells[cells.eq(5)].index)
 support=d.apply(lambda r:(r.distance_km,r.modulation,r.launch_power_dbm) in common,axis=1)
 fig,axs=plt.subplots(1,3,figsize=(7.1,2.65),sharey=True)
 for i,c in enumerate(CL):
  ix=sorted(lookup[s] for s in d.loc[support&d.diagnosis_label.eq(c),'sample_id']);a=h['psd_seq'][ix]
  for j,ax in enumerate(axs):ax.plot(freq,a[:,j].mean(0),color=COL[i],ls=['-','--',':','-.',(0,(3,1,1,1))][i],label=N[i],lw=1.2)
 for j,ax in enumerate(axs):
  ax.set(title=['(a) X polarization','(b) Y polarization','(c) Coherent mixture'][j],xlabel='Frequency (GHz)',xlim=(-128,128),xticks=[-100,-50,0,50,100]);ax.grid()
 axs[0].set_ylabel('Digital log-spectrum (dB)')
 fig.legend(*axs[0].get_legend_handles_labels(),ncol=5,loc='upper center',bbox_to_anchor=(.53,1),columnspacing=1.05,handlelength=2)
 fig.subplots_adjust(left=.10,right=.99,bottom=.22,top=.73,wspace=.18);save(fig,'psd_class_curves')
fig,axs=plt.subplots(2,3,figsize=(7.1,4.25))
keys=['power_total_db','kurt_i1','corr_amp_xy','amp_std_x','psd_centroid_norm','psd_ripple_std_db']
titles=['Total digital power (dB)','$I_X$ kurtosis','X–Y intensity correlation','X-intensity standard deviation','Spectral centroid ($f/F_s$)','Full-grid log-power spread (dB)']
for j,(ax,key,title) in enumerate(zip(axs.flat,keys,titles)):
 bp=ax.boxplot([d.loc[d.diagnosis_label.eq(c),key].dropna().to_numpy() for c in CL],positions=range(5),widths=.55,whis=(5,95),showfliers=False,patch_artist=True,medianprops={'color':'#222','linewidth':1},boxprops={'edgecolor':'#555'},whiskerprops={'color':'#555'})
 for b,c in zip(bp['boxes'],COL):b.set(facecolor=c,alpha=.65)
 ax.set_xticks(range(5),['Normal','VOA','EDFA','Offset','Bandwidth'],rotation=30,ha='right')
 ax.set_title(f'({chr(97+j)}) '+title,loc='left',pad=7,fontsize=9);ax.grid(axis='y')
fig.subplots_adjust(left=.085,right=.99,bottom=.14,top=.93,wspace=.42,hspace=.68);save(fig,'09_feature_distributions')
print(args.output)
