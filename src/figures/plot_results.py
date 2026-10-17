"""Data-driven PDFs from public tables, saved predictions and derived tensors only."""
from pathlib import Path
import json,string
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproduction/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':10,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'legend.frameon':False})
COL=['#697A76','#BA7D7A','#B59A6D','#648BA6','#79A790']
CL=['normal','attenuation','edfa','wss_offset','wss_bandwidth'];DISPLAY=['Normal','VOA','EDFA','WSS offset','WSS bandwidth']
CO=['full','low_ber'];CC=['#BA7D7A','#617F77']
F=json.loads((ROOT/'data_dictionary/feature_columns.json').read_text())
d=pd.read_csv(ROOT/'data/structured_features_3355.csv').merge(pd.read_csv(ROOT/'data_dictionary/capture_index.csv'),on=['sample_id','diagnosis_label'],validate='one_to_one')
ms=pd.read_csv(ROOT/'metrics/summary.csv');ms=ms[ms.fold.astype(str).eq('pooled')].set_index('experiment')
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def finish(ax):
 for s in ax.spines.values():s.set_visible(True);s.set_linewidth(.7)
 ax.tick_params(which='both',direction='in',top=True,right=True)
def bars():
 fig,axs=plt.subplots(1,2,figsize=(7.4,3.6),layout='constrained')
 for ax,key in zip(axs,['macro_f1','accuracy']):
  for k,c in enumerate(CO):
   r=ms.loc[[c+'/baseline_'+n for n in ['logistic_regression','random_forest','lightgbm','mlp']]]
   x=np.arange(4)+(k-.5)*.34;v=r[key].to_numpy()
   ax.bar(x,v,.30,color=CC[k],label='Full primary' if k==0 else 'Low BER')
   ax.errorbar(x,v,yerr=[v-r[key+'_low'],r[key+'_high']-v],fmt='none',color='#444444',capsize=2)
   for xx,y in zip(x,v):ax.text(xx,y+.075,f'{y:.3f}',ha='center',color=CC[k],fontsize=9)
  ax.set(xticks=np.arange(4),xticklabels=['LR','RF','LGBM','MLP'],ylim=(0,1.12),ylabel='Macro-F1' if key=='macro_f1' else 'Accuracy');finish(ax)
 axs[0].legend(loc='upper left',fontsize=8);save(fig,'baseline_comparison')
def confusion():
 fig,axs=plt.subplots(1,2,figsize=(8.5,4.3),layout='constrained')
 for ax,c in zip(axs,CO):
  p=pd.read_csv(ROOT/'predictions'/c/'baseline_lightgbm/predictions.csv')
  cm=pd.crosstab(p.y_true,p.y_pred).reindex(index=CL,columns=CL,fill_value=0).to_numpy();ratio=cm/cm.sum(1,keepdims=True)
  ax.imshow(ratio,cmap='PuBu' if c=='full' else 'BuGn',vmin=0,vmax=1)
  for i in range(5):
   for j in range(5):ax.text(j,i,f'{ratio[i,j]:.2f}\n({cm[i,j]})',ha='center',va='center',fontsize=8,color='white' if ratio[i,j]>.6 else '#222222')
  ax.set(xticks=range(5),yticks=range(5),xticklabels=DISPLAY,yticklabels=DISPLAY,title='Full primary' if c=='full' else 'Low BER',xlabel='Predicted',ylabel='True');ax.tick_params(axis='x',rotation=45);finish(ax)
 save(fig,'confusion_matrices')
def coverage_features():
 fig,ax=plt.subplots(figsize=(4.8,3.1),layout='constrained');counts=d.diagnosis_label.value_counts().reindex(CL)
 ax.barh(DISPLAY,counts,color=COL)
 for j,n in enumerate(counts):ax.text(n+8,j,str(n),va='center')
 ax.set(xlim=(0,1000),xlabel='Captures');ax.invert_yaxis();finish(ax);save(fig,'coverage')
 corr=np.column_stack([d[F].corrwith(d.diagnosis_label.eq(c).astype(float)) for c in CL])
 fig,ax=plt.subplots(figsize=(6,9),layout='constrained');im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto');ax.set(yticks=range(32),yticklabels=F,xticks=range(5),xticklabels=DISPLAY);ax.tick_params(axis='x',rotation=30);fig.colorbar(im,ax=ax,label='Pearson r');save(fig,'feature_label_correlations')
 fig,axs=plt.subplots(2,3,figsize=(8,5.2),layout='constrained')
 for ax,col in zip(axs.flat,['power_x_db','crest_i1','amp_std_x','psd_centroid_norm','psd_lr_diff_db','psd_ripple_std_db']):
  bp=ax.boxplot([d.loc[d.diagnosis_label.eq(c),col] for c in CL],tick_labels=['N','V','E','O','B'],whis=(5,95),showfliers=False,patch_artist=True)
  for a,c in zip(bp['boxes'],COL):a.set_facecolor(c);a.set_alpha(.65)
  ax.set_title(col);finish(ax)
 save(fig,'feature_distributions')
def transfer():
 fig,ax=plt.subplots(figsize=(7.4,3.3),layout='constrained')
 names=['holdout_exact_cell','holdout_lp_id','holdout_modulation','holdout_distance_km']
 for k,c in enumerate(CO):
  r=ms.loc[[c+'/'+n for n in names]];v=r.macro_f1.to_numpy();y=np.arange(4)+(k-.5)*.16
  ax.errorbar(v,y,xerr=[v-r.macro_f1_low,r.macro_f1_high-v],fmt='o' if k==0 else 's',color=CC[k],capsize=3,label='Full primary' if k==0 else 'Low BER')
 ax.set(yticks=np.arange(4),yticklabels=['Complete condition','Launch power','Modulation','Distance'],xlim=(0,1),xlabel='Macro-F1');ax.invert_yaxis();ax.legend(loc='lower left');finish(ax);save(fig,'condition_transfer')
def magnitudes():
 p=ROOT/'metrics/fiveclass_magnitude_recall.csv'
 if not p.exists():return
 m=pd.read_csv(p);m=m[m['mode'].eq('common_all_settings')]
 fig,axs=plt.subplots(1,4,figsize=(10,3.2),sharey=True,layout='constrained')
 for ax,fam,label in zip(axs,CL[1:],DISPLAY[1:]):
  for co,c in zip(CO,CC):
   a=m[m.cohort.eq(co)&m.fault.eq(fam)].sort_values('setting');ax.plot(a.setting,a.recall,'o-',color=c,label='Full primary' if co=='full' else 'Low BER');ax.fill_between(a.setting,a.recall_low,a.recall_high,color=c,alpha=.14)
  ax.set(title=label,ylim=(0,1.04),xlabel='Setting (dB)' if fam in ['attenuation','edfa'] else 'Setting (GHz)');finish(ax)
 axs[0].set_ylabel('Five-class recall');axs[-1].legend(loc='lower right',fontsize=8);save(fig,'magnitude_recall')
def ber():
 q=d.merge(pd.read_csv(ROOT/'data/qot_annotations_3355.csv').drop(columns=['ber_mean','dsp_valid']),on='sample_id',validate='one_to_one')
 q=q[q.dsp_valid.astype(str).str.lower().eq('true')]
 settings={'attenuation':'severity_value_db','edfa':'severity_value_db','wss_offset':'wss_offset_ghz','wss_bandwidth':'wss_bandwidth_ghz'}
 fig,axs=plt.subplots(4,4,figsize=(11,9),layout='constrained')
 for row,(mod,dist) in enumerate([('QPSK',640),('QPSK',960),('16QAM',640),('16QAM',960)]):
  for col,fam in enumerate(CL[1:]):
   ax=axs[row,col];a=q[q.modulation.eq(mod)&q.distance_km.eq(dist)&q.diagnosis_label.eq(fam)]
   for lp,c in zip([1,3,5],['#527D9B','#78A18E','#C49B62']):
    s=a[a.launch_power_dbm.eq(lp)].groupby(settings[fam]).ber_mean.agg(['mean',lambda x:x.quantile(.25),lambda x:x.quantile(.75)])
    if s.empty:continue
    ax.plot(s.index,s['mean'],'o-',color=c,label=f'{lp} dBm')
    ax.vlines(s.index,s.iloc[:,1],s.iloc[:,2],color=c,lw=1)
   ax.axhline(.024,color='#555555',ls='--',lw=.8);ax.set_yscale('symlog',linthresh=1e-4);ax.set_title(f'{mod}, {dist} km | {DISPLAY[col+1]}',fontsize=9);ax.set_ylim(0,.5);finish(ax)
 axs[0,-1].legend(fontsize=8);save(fig,'ber_by_condition')
def spectra():
 import h5py
 target=ROOT/'data/tensors/shard.h5'
 if not target.exists():return
 with h5py.File(target) as h:
  ids=[x.decode() if isinstance(x,bytes) else str(x) for x in h['sample_id'][:]];lookup={v:i for i,v in enumerate(ids)};z=h['psd_seq'][:]
 cells=d.groupby(['distance_km','modulation','launch_power_dbm']).diagnosis_label.nunique();cells=set(cells[cells.eq(5)].index)
 mask=d.apply(lambda r:(r.distance_km,r.modulation,r.launch_power_dbm) in cells,axis=1)
 axis=pd.read_csv(ROOT/'data_dictionary/psd_frequency_axis.csv').frequency_center_ghz
 fig,axs=plt.subplots(1,3,figsize=(9,3),sharey=True,layout='constrained')
 for j,ax in enumerate(axs):
  for lab,c,ls in zip(CL,COL,['-','--','-.',':',(0,(3,1,1,1))]):
   idx=[lookup[v] for v in d.loc[mask&d.diagnosis_label.eq(lab),'sample_id']];ax.plot(axis,z[idx,j].mean(0),color=c,ls=ls,label=DISPLAY[CL.index(lab)])
  ax.set(xlabel='Frequency (GHz)',title=['X','Y','Coherent mixture'][j],xlim=(-60,60));finish(ax)
 axs[0].set_ylabel('Digital log-spectrum (dB)');axs[-1].legend(fontsize=8);save(fig,'spectral_class_means')
def constellations():
 cases=json.loads((ROOT/'data/representative_constellations.json').read_text());z=np.load(ROOT/'data/representative_constellations.npz',allow_pickle=False)
 groups=[['normal','wss_bandwidth'],['voa','edfa'],['wss_offset']]
 fig=plt.figure(figsize=(10.6,5.4));gs=fig.add_gridspec(3,6,hspace=.45,wspace=.12);letter=0
 for row,fams in enumerate(groups):
  selected=[c for c in cases if c['family'] in fams]
  for col,c in enumerate(selected):
   ax=fig.add_subplot(gs[row,col]);a=z[c['coordinates_key']];lim=np.max(np.abs(np.r_[a.real,a.imag]))*1.015;lim=max(lim,3.8)
   rng=np.random.default_rng(20260913);v=a[np.sort(rng.choice(len(a),min(30000,len(a)),replace=False))]
   ax.scatter(v.real,v.imag,s=.3,alpha=.17,color='#648BA6',rasterized=True,linewidths=0)
   H,xe,ye=np.histogram2d(a.real,a.imag,bins=140,range=[[-lim,lim],[-lim,lim]]);H=gaussian_filter(H,2)
   if H.max()>0:ax.contour((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,H.T,levels=H.max()*np.array([.1,.5,.9]),colors='#355674',linewidths=.45)
   ax.set(aspect='equal',xlim=(-lim,lim),ylim=(-lim,lim));ax.axis('off');label=c['label'].replace('wss_bandwidth','Bandwidth')
   ax.set_title(f'({string.ascii_lowercase[letter]}) {label}\nBER={c["sample_ber"]:.2e}',fontsize=9,pad=2);letter+=1
 save(fig,'constellations_all_settings')
if __name__=='__main__':
 for fn in [bars,confusion,coverage_features,transfer,magnitudes,ber,spectra,constellations]:fn()
 print(OUT)
