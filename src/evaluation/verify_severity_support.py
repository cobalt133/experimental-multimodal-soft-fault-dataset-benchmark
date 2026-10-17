"""Rebuild magnitude support and intervals from public metadata and OOF predictions."""
from pathlib import Path
import numpy as np,pandas as pd
try:from bootstrap_ci import intervals
except ImportError:from public_bootstrap import intervals
HERE=Path(__file__).resolve().parent
P=HERE/'public_package' if (HERE/'public_package').exists() else HERE.parents[1]
META=pd.read_csv(P/'data_dictionary/capture_index.csv').drop(columns='g_config')
META['cell']=[str((int(r.distance_km),r.modulation,float(r.launch_power_dbm) if pd.notna(r.launch_power_dbm) else -999.0)) for r in META.itertuples()]
SETTING={'attenuation':'severity_value_db','edfa':'severity_value_db','wss_offset':'wss_offset_ghz','wss_bandwidth':'wss_bandwidth_ghz'}
rows=[]
for c in ['full','low_ber']:
 p=pd.read_csv(P/'predictions'/c/'baseline_lightgbm/predictions.csv').merge(META,on='sample_id',validate='one_to_one')
 table=pd.read_csv(P/'metrics/fiveclass_magnitude_recall.csv');table=table[table.cohort.eq(c)]
 for fault,col in SETTING.items():
  z=p[p.y_true.eq(fault)];shared=set.intersection(*[set(g.cell) for _,g in z.groupby(col)])
  for r in table[table.fault.eq(fault)].itertuples():
   g=z[z[col].eq(r.setting)]
   if r.mode=='common_all_settings':g=g[g.cell.isin(shared)]
   assert len(g)==r.n and g.g_config.nunique()==r.groups and g.cell.nunique()==r.n_cells
   a=g.assign(correct=g.y_true.eq(g.y_pred)).groupby('g_config').correct.agg(['sum','count']).to_numpy()
   rng=np.random.default_rng(20260911);b=a[rng.integers(len(a),size=(2000,len(a)))].sum(1);low,high=np.quantile(b[:,0]/b[:,1],[.025,.975])
   delta=max(abs(g.y_true.eq(g.y_pred).mean()-r.recall),abs(low-r.recall_low),abs(high-r.recall_high));assert delta<1e-10
   rows.append({'cohort':c,'task':'five_class_recall','fault':fault,'setting':r.setting,'support_mode':r.mode,'captures':len(g),'cells':g.cell.nunique(),'max_difference':delta,'passed':True})
  bp=pd.read_csv(P/'predictions'/c/('binary_'+fault)/'predictions.csv').merge(META,on='sample_id',validate='one_to_one')
  pos=bp[bp.diagnosis_label.eq(fault)];normal=bp[bp.diagnosis_label.eq('normal')]
  shared=set(normal.cell).intersection(*[set(g.cell) for _,g in pos.groupby(col)])
  bt=pd.read_csv(P/'metrics/magnitude_matched.csv');bt=bt[bt.cohort.eq(c)&bt.fault.eq(fault)]
  for r in bt.itertuples():
   g=pos[pos[col].eq(r.setting)];support=shared if r.mode=='common_all_settings' else set(g.cell)&set(normal.cell)
   a=g[g.cell.isin(support)];b=normal[normal.cell.isin(support)];z=pd.concat([a,b])
   assert len(a)==r.n_fault and len(b)==r.n_normal and len(support)==r.n_cells and z.g_config.nunique()==r.groups
   out=intervals(z,int(r.bootstrap_requested));delta=max(abs(float(getattr(r,k))-v) for k,v in out.items() if hasattr(r,k));assert delta<1e-10,(c,fault,r.setting,delta)
   rows.append({'cohort':c,'task':'binary_detection','fault':fault,'setting':r.setting,'support_mode':r.mode,'captures':len(z),'cells':len(support),'max_difference':delta,'passed':True})
 print(c,'magnitude checks passed',flush=True)
pd.DataFrame(rows).to_csv(P/'reproduction/severity_support_and_intervals.csv',index=False)
print(len(rows),'setting-level support/interval rows verified',flush=True)
