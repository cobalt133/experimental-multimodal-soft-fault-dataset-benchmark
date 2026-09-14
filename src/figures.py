"""Redraw real post-DSP X symbols and update descriptive five-label correlations."""
from pathlib import Path
import json,hashlib,math
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from publication_style import finish
R=Path(__file__).resolve().parent;F=R/'figures';D=R.parent/'dataset_revision_worktree/revision_v3_closure'
S=R.parent/'ALL_AMPLITUDE_CONSTELLATIONS_XPOL_MINIMAL_20260913'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'pdf.fonttype':42,'axes.spines.top':True,'axes.spines.right':True,'xtick.direction':'in','ytick.direction':'in','legend.frameon':False})
m=json.loads((S/'selection_manifest_all_amplitudes_xpol.json').read_text());z=np.load(S/'all_amplitude_constellation_coordinates_xpol.npz')
cases=[x for x in m['cases'] if not(x['family']=='wss_bandwidth' and x['fault_setting']=='wss_bandwidth_75GHz')]
assert len(cases)==17 and len({x['coordinates_key'] for x in cases})==17
old=pd.read_csv(R.parent/'FINAL_TRANSFER_WSS75_QOT_R5_FULL_20260911/qot_full_r5/qot_sample_level_main_3355_r5.csv')
for x in cases:
    a=z[x['coordinates_key']];assert a.ndim==1 and np.iscomplexobj(a) and np.isfinite(a).all()
    b=old[old.sample_id==x['sample_id']];assert len(b)==1 and abs(float(b.ber_mean.iloc[0])-x['sample_ber'])<1e-12
    assert abs((x['ber_x']+x['ber_y'])/2-x['sample_ber'])<1e-12
def label(x,short=False):
    v=x['fault_setting'].split('_')[-1].replace('GHz','').replace('dB','')
    return {'normal':'Normal (75 GHz)','voa':f'VOA: {v} dB','edfa':f'EDFA: {v} dB reduction','wss_offset':f'WSS offset: {v} GHz','wss_bandwidth':f'WSS bandwidth: {v} GHz'}[x['family']]
def save(fig,name):
    finish(fig)
    for ext in ['pdf','png']:fig.savefig(F/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.04,facecolor='white')
    plt.close(fig)
main=[next(x for x in cases if x['family']=='normal')]
for fam in ['voa','edfa','wss_offset','wss_bandwidth']:
    # A fixed, non-visual choice: the largest representative BER still below the reference.
    main.append(max((x for x in cases if x['family']==fam and x['sample_ber']<.024),key=lambda x:x['sample_ber']))
drawn=[]
COLORS={'normal':'#5B708F','voa':'#8FA58A','edfa':'#B97870','wss_offset':'#9A8298','wss_bandwidth':'#C19A68'}
def panel(ax,x,lim,j,title=None):
    raw=z[x['coordinates_key']];rng=np.random.default_rng(20260913);idx=np.sort(rng.choice(len(raw),min(30000,len(raw)),replace=False));a=raw[idx];color=COLORS[x['family']]
    # Adapted from the supplied render_all_amplitudes_xpol_from_npz.py density().
    # Preserve its scatter/contour palette and smoothing, but expand the grid to
    # include all actual points. Constant bin width keeps the smoothing scale fixed.
    ax.scatter(a.real,a.imag,s=1.15,c=color,alpha=.21,linewidths=0,rasterized=True)
    bins=math.ceil(2*lim*14)
    H,xe,ye=np.histogram2d(raw.real,raw.imag,bins=bins,range=[[-lim,lim],[-lim,lim]])
    H=gaussian_filter(H,2.0);xc=(xe[:-1]+xe[1:])/2;yc=(ye[:-1]+ye[1:])/2
    lev=np.linspace(H.max()*.10,H.max()*.90,5)
    density_cmap=LinearSegmentedColormap.from_list('density',['#ffffff',color],N=256)
    ax.contourf(xc,yc,H.T,levels=lev,cmap=density_cmap,alpha=.34)
    ax.contour(xc,yc,H.T,levels=lev[::2],colors=color,linewidths=.38,alpha=.8)
    ax.grid(color='#d9e1ec',linewidth=.25,alpha=.35)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal',xticks=[],yticks=[],xlabel='',ylabel='')
    ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
                   labelbottom=False,labelleft=False)
    ber=f"{x['sample_ber']:.2e}" if x['sample_ber'] else '0 (observed)'
    display=title or label(x).replace(': ','\n').replace('Normal (75 GHz)','Normal\n75 GHz')
    ax.set_title(f'({chr(97+j)}) '+display+'\nBER '+ber,fontsize=8.2,pad=1.5,linespacing=.9)
    drawn.append({'figure_label':label(x),'sample_id':x['sample_id'],'coordinates_key':x['coordinates_key'],'ber_mean':x['sample_ber'],'ber_x':x['ber_x'],'displayed_symbols':len(idx),'available_symbols':len(z[x['coordinates_key']]),'axis_limit':lim,'outside':int(np.sum((abs(z[x['coordinates_key']].real)>lim)|(abs(z[x['coordinates_key']].imag)>lim))),'index_sha256':hashlib.sha256(idx.tobytes()).hexdigest()})
def limit(items):return math.ceil(max(max(abs(z[x['coordinates_key']].real).max(),abs(z[x['coordinates_key']].imag).max()) for x in items)*1.05)
# Optional five-class preview; the manuscript now includes ALL 17 settings in two figures.
fig,axs=plt.subplots(1,5,figsize=(7.4,2.05));titles=['Normal\n75 GHz','VOA\n5 dB','EDFA\n3 dB reduction','WSS offset\n12 GHz','WSS bandwidth\n60 GHz']
for j,(ax,x,t) in enumerate(zip(axs,main,titles)):panel(ax,x,limit(main),j,t)
fig.subplots_adjust(left=.065,right=.995,top=.65,bottom=.21,wspace=.40);save(fig,'constellation_five_class_preview')
# Full-amplitude evidence: centered rows with real panels only. Normal is not a bandwidth fault.
for name,groups in [('constellation_loss_gain',[cases[:3],cases[3:7]]),('constellation_wss',[cases[7:10],cases[10:13],cases[13:]])]:
    fig=plt.figure(figsize=(7.4,2.55*len(groups)));lim=limit([x for g in groups for x in g]);j=0
    for row,g in enumerate(groups):
        width=.19;gap=.052;left=(1-(len(g)*width+(len(g)-1)*gap))/2
        for col,x in enumerate(g):
            ax=fig.add_axes([left+col*(width+gap),1-(row+1)/len(groups)+.065,width,.60/len(groups)])
            panel(ax,x,lim,j);j+=1
    save(fig,name)
    if name=='constellation_loss_gain':
        import shutil
        for ext in ['pdf','png']:shutil.copy2(F/f'{name}.{ext}',F/f'10_constellations.{ext}')
# Compact complete matrix with three semantic rows and no panel frames.
fig=plt.figure(figsize=(7.4,3.35)); width=.105; gap=.008
row_cases=[[x for x in cases if x['family']=='normal']+[x for x in cases if x['family']=='wss_bandwidth'],
           [x for x in cases if x['family'] in ('voa','edfa')],
           [x for x in cases if x['family']=='wss_offset']]
def matrix_title(x):
    if x['family']=='normal': return 'Normal'
    if x['family']=='voa': return 'VOA '+x['fault_setting'].split('_')[-1].replace('dB','')
    if x['family']=='edfa': return 'EDFA '+x['fault_setting'].split('_')[-1].replace('dB','')
    if x['family']=='wss_offset': return 'Off '+x['fault_setting'].split('_')[-1].replace('GHz','')
    return 'BW '+x['fault_setting'].split('_')[-1].replace('GHz','')
j=0
for row,items in enumerate(row_cases):
    row_width=len(items)*width+(len(items)-1)*gap; left0=(1-row_width)/2
    for col,x in enumerate(items):
        raw=z[x['coordinates_key']]; lim=max(np.max(np.abs(raw.real)),np.max(np.abs(raw.imag)))*1.10
        bottom=[.70,.375,.05][row]
        ax=fig.add_axes([left0+col*(width+gap),bottom,width,.22])
        panel(ax,x,lim,j)
        # Labels remain above each constellation, matching the manuscript's earlier style.
        ax.set_title(f'({chr(97+j)}) {matrix_title(x)}\n{x["sample_ber"]:.2e}',fontsize=6.0,
                     color='#202020',fontweight='semibold',pad=1.5,linespacing=.85)
        ax.set_axis_off()
        j+=1
save(fig,'constellation_all_17_compact')
# Per-family strips use consistent axes within each strip and no unused cells.
for fam in m['families']:
    its=[x for x in cases if x['family']==fam];n=len(its);cols=min(3,n);rows=math.ceil(n/cols)
    fig=plt.figure(figsize=(7.4,2.45*rows));lim=limit(its)
    for row in range(rows):
        g=its[row*cols:(row+1)*cols];width=.235;gap=.065;left=(1-(len(g)*width+(len(g)-1)*gap))/2
        for col,x in enumerate(g):panel(fig.add_axes([left+col*(width+gap),1-(row+1)/rows+.08,width,.60/rows]),x,lim,row*cols+col)
    save(fig,'constellation_'+fam)
p=pd.read_csv(D/'linked_primary.csv');features=p.columns[p.columns.get_loc('power_x_db'):].tolist();assert len(features)==32
classes=['normal','attenuation','edfa','wss_offset','wss_bandwidth'];names=['Normal','VOA','EDFA','Offset','Bandwidth']
def abbrev(f):
    if f.startswith('amp_'):
        return f.replace('amp_std_','Int. SD ').replace('amp_skew_','Int. skew ').replace('amp_kurt_','Int. kurt ')
    return f.replace('power_','Power ').replace('crest_','Crest ').replace('skew_','Skew ').replace('kurt_','Kurt ').replace('corr_amp_xy','Int. corr. XY').replace('corr_','Corr ').replace('psd_','PSD ').replace('ripple_std','ripple').replace('lr_diff','L/R diff.').replace('_norm','').replace('_db',' (dB)').replace('_',' ')
cmap=LinearSegmentedColormap.from_list('signed',['#527D9B','#FCFCFA','#BA7D7A'])
numbers={}
for tag,cohort in [('feature_label_correlations',p),('feature_label_correlations_common',p[p.common_support])]:
    y=pd.DataFrame({c:(cohort.label5==c).astype(float) for c in classes},index=cohort.index)
    corr=np.array([[cohort[f].corr(y[c]) for c in classes] for f in features]);assert np.isfinite(corr).all()
    pd.DataFrame(corr,index=features,columns=classes).to_csv(R/'audit'/f'{tag}.csv')
    fig,ax=plt.subplots(figsize=(3.75,6.7));im=ax.imshow(corr,vmin=-1,vmax=1,cmap=cmap,aspect='auto')
    ax.set(yticks=range(32),yticklabels=[abbrev(f) for f in features],xticks=range(5),xticklabels=names)
    ax.tick_params(axis='x',labelrotation=45)
    for i in range(32):
        for j in range(5):ax.text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr[i,j])>.8 else '#303030')
    fig.subplots_adjust(left=.40,right=.98,bottom=.15,top=.99)
    ca=fig.add_axes([.41,.055,.55,.013]);fig.colorbar(im,cax=ca,orientation='horizontal',label='Pearson r');save(fig,tag)
    numbers[tag]={'n':len(cohort),'max_abs':float(abs(corr).max()),'top':[(features[i],classes[j],float(corr[i,j])) for i,j in sorted(np.ndindex(corr.shape),key=lambda ij:abs(corr[ij]),reverse=True)[:8]]}
# Descriptor redundancy is a different question from label association: retain separately.
c=p[features].corr();c.to_csv(R/'audit/feature_feature_correlations.csv')
fig,ax=plt.subplots(figsize=(7.4,6.8));im=ax.imshow(c,vmin=-1,vmax=1,cmap=cmap);ax.set(xticks=range(32),yticks=range(32),xticklabels=range(1,33),yticklabels=range(1,33),xlabel='Feature index',ylabel='Feature index');ax.tick_params(axis='x',labelrotation=90);fig.colorbar(im,ax=ax,fraction=.035,pad=.03,label='Pearson r');fig.subplots_adjust(left=.07,right=.91,bottom=.09,top=.98);save(fig,'feature_feature_correlations')
pd.DataFrame({'index':range(1,33),'feature':features,'display':[abbrev(f) for f in features]}).to_csv(R/'audit/feature_index.csv',index=False)
(R/'audit/xpol_redraw.json').write_text(json.dumps({'input_npz_sha256':hashlib.sha256((S/'all_amplitude_constellation_coordinates_xpol.npz').read_bytes()).hexdigest(),'main_selection_rule':'All 17 unique supplied setting representatives, split into loss/gain and WSS panels; no BER-based setting exclusion','unique_settings':17,'normal_in_bandwidth_fault_panels':False,'smoothing':True,'smoothing_scope':'Density display only; Gaussian sigma=2 histogram bins, 5 relative-density levels (10-90% of panel peak); not probability-mass contours','server_renderer_sha256':hashlib.sha256((S/'render_all_amplitudes_xpol_from_npz.py').read_bytes()).hexdigest(),'per_panel_normalization':False,'raw_mat_reads':0,'new_fits':0,'source_ber_verified':True,'plots':drawn,'correlations':numbers},indent=2))
print(json.dumps(numbers,indent=2));print('Redrew X-only coordinates: 17 unique settings, no synthetic empty panels.')
