"""Main-paper control and magnitude figures from saved metrics; no model fitting."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=ROOT/'figures/main')
args=parser.parse_args();OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':9.2,'axes.labelsize':9.2,'axes.titlesize':9.6,'xtick.labelsize':8.8,
    'ytick.labelsize':9.2,'legend.fontsize':9.2,'legend.frameon':False,
    'pdf.fonttype':42,'mathtext.fontset':'stix','axes.linewidth':.65,
    'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
COL=['#BA7D7A','#617F77'];CO=['full','low_ber'];MARK=['o','s'];LAB=['Full primary','Low BER']
def finish(ax):
    for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#39433F')
    ax.tick_params(which='both',direction='in',top=True,right=True,length=3,pad=3)
def save(fig,name):
    for ext in ['pdf','png']:fig.savefig(OUT/(name+'.'+ext),dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)
handles=[Line2D([],[],color=c,marker=mk,lw=1.3,markersize=4,label=l) for c,mk,l in zip(COL,MARK,LAB)]
m=pd.read_csv(ROOT/'metrics/summary.csv');m=m[m.fold.astype(str).eq('pooled')].set_index('experiment')
fig,axs=plt.subplots(2,1,figsize=(7.25,4.3),gridspec_kw={'height_ratios':[1.5,1]})
groups=[(['metadata_control','ablation_power_only','ablation_no_power','baseline_lightgbm'],
         ['Condition metadata only','Power descriptors (3)','Without power descriptors (29)','All descriptors (32)'],'(a) Input controls: full primary'),
        (['pairwise_distance_F32','pairwise_distance_no_spectral'],
         ['All descriptors (32)','Without spectral descriptors (26)'],'(b) Paired distance transfer')]
inputs=[]
for ax,(names,labels,title) in zip(axs,groups):
    for y in range(len(names)):
        if y%2==0:ax.axhspan(y-.43,y+.43,color='#F2F5F3',zorder=0)
    cohorts=list(zip(CO,COL,MARK)) if len(names)==2 else [(CO[0],COL[0],MARK[0])]
    for k,(co,c,mk) in enumerate(cohorts):
        for j,name in enumerate(names):
            row=m.loc[co+'/'+name];v=row.macro_f1;y=j+(0 if len(cohorts)==1 else -.13 if k==0 else .13)
            ax.errorbar(v,y,xerr=[[v-row.macro_f1_low],[row.macro_f1_high-v]],
                        fmt=mk,color=c,capsize=2.4,lw=1.05,markersize=4.2,zorder=3)
            ax.text(1.025,y,f'{v:.4f}',transform=ax.get_yaxis_transform(),
                    color=c,ha='left',va='center',fontsize=9.2,clip_on=False)
            inputs.append({'panel':title,'cohort':co,'experiment':name,'n':int(row.n),
                           'groups':int(row.groups),'macro_f1':v,'low':row.macro_f1_low,'high':row.macro_f1_high})
    ax.set(yticks=range(len(names)),yticklabels=labels,xlim=(0,1),ylim=(len(names)-.5,-.5),xlabel='Five-class macro-F1')
    ax.set_title(title,loc='left',pad=7);ax.grid(axis='x',alpha=.15);finish(ax)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.62,1.015),ncol=2)
fig.subplots_adjust(left=.345,right=.87,top=.88,bottom=.10,hspace=.65)
save(fig,'signal_ablation_summary')
pd.DataFrame(inputs).to_csv(ROOT/'metrics/core_control_figure_inputs.csv',index=False)

data=pd.read_csv(ROOT/'metrics/magnitude_matched.csv')
fig,axs=plt.subplots(3,4,figsize=(7.25,5.45),sharey='row')
families=['attenuation','edfa','wss_offset','wss_bandwidth']
titles=['(a) VOA','(b) EDFA','(c) WSS offset','(d) WSS bandwidth']
metrics=[('fault_f1','Fault-class F1'),('normal_false_alarm','False-alarm rate'),('fault_missed_detection','Missed-detection rate')]
for col,(fam,title) in enumerate(zip(families,titles)):
    axs[0,col].set_title(title,pad=7)
    for row,(metric,label) in enumerate(metrics):
        ax=axs[row,col]
        for co,color,marker in zip(CO,COL,MARK):
            z=data[(data.cohort==co)&(data.fault==fam)&(data['mode']=='common_all_settings')].sort_values('setting')
            assert len(z) and z.setting.is_unique,(co,fam)
            ax.plot(z.setting,z[metric],color=color,marker=marker,lw=1.25,ms=3.4)
            ax.fill_between(z.setting,z[metric+'_low'],z[metric+'_high'],color=color,alpha=.14,lw=0)
        ticks=sorted(z.setting.unique());ax.set_xticks(ticks)
        ax.set(ylim=(-.025,1.025),yticks=[0,.25,.5,.75,1]);ax.grid(axis='y',alpha=.15)
        if fam=='wss_bandwidth':ax.set_xlim(72,53)
        else:
            span=max(ticks)-min(ticks);pad=max(.4,.08*span);ax.set_xlim(min(ticks)-pad,max(ticks)+pad)
        if col==0:ax.set_ylabel(label,labelpad=5)
        if row<2:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel(['Added loss (dB)','Gain reduction (dB)','Offset (GHz)','Bandwidth (GHz)'][col])
        finish(ax)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,1.005),ncol=2)
fig.subplots_adjust(left=.10,right=.985,top=.90,bottom=.095,wspace=.16,hspace=.20)
save(fig,'binary_magnitude_comparison')
print('Core figures saved:',OUT)
