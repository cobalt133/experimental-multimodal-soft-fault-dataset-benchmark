"""Portable derived-data reproduction. Copied into the staged public package."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score,accuracy_score

ROOT=Path(__file__).resolve().parents[2]
SEED=20260911
def model(name):
    if name=='logistic_regression':return make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight='balanced',random_state=SEED))
    if name=='random_forest':return RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=4,random_state=SEED)
    if name=='mlp':return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),alpha=.001,batch_size=64,learning_rate_init=.001,max_iter=1000,early_stopping=False,random_state=SEED))
    if name=='lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=15,class_weight='balanced',n_jobs=4,random_state=SEED,verbosity=-1)
    raise ValueError(name)

def scores(p):
    n=p.y_true.eq('normal');pn=p.y_pred.eq('normal')
    return {'n':len(p),'macro_f1':f1_score(p.y_true,p.y_pred,average='macro'),'accuracy':accuracy_score(p.y_true,p.y_pred),'normal_false_alarm':float((~pn[n]).mean()),'fault_missed_detection':float(pn[~n].mean())}

def run(cohort,experiment,name='lightgbm'):
    frame=pd.read_csv(ROOT/'data'/('structured_features_3355.csv' if cohort=='full' else 'structured_features_2224_low_ber.csv')).set_index('sample_id')
    meta=pd.read_csv(ROOT/'data_dictionary/capture_index.csv').set_index('sample_id')
    expected=pd.read_csv(ROOT/'predictions'/cohort/experiment/'predictions.csv')
    members=pd.read_csv(ROOT/'splits'/cohort/experiment/'members.csv',dtype={'fold':str})
    features=json.loads((ROOT/'data_dictionary/feature_columns.json').read_text())
    # Exact membership replay; never regenerate or optimize the split.
    labels=expected.set_index('sample_id').y_true
    if not labels.index.is_unique:raise ValueError('Replay requires one held-out prediction per capture.')
    frame['target']=frame.diagnosis_label
    if set(labels).issubset({'normal','fault'}):frame['target']=np.where(frame.diagnosis_label.eq('normal'),'normal','fault')
    frame.loc[labels.index,'target']=labels
    if experiment.endswith('metadata_control'):
        x=meta.reindex(frame.index)[['distance_km','modulation','launch_power_dbm']].copy()
        x['modulation']=x.modulation.map({'QPSK':0,'16QAM':1,'64QAM':2})
        x=x.fillna(-99)
    else:
        if experiment=='ablation_power_only':features=[c for c in features if c.startswith('power_')]
        if experiment=='ablation_no_power':features=[c for c in features if not c.startswith('power_')]
        if experiment=='pairwise_distance_no_spectral':features=[c for c in features if not c.startswith('psd_')]
        x=frame[features]
    arrays=None
    if experiment.startswith('modality_'):
        import h5py
        mode=experiment[len('modality_'):]
        target=ROOT/'data/tensors/shard.h5'
        if not target.exists():raise FileNotFoundError('First run scripts/rebuild_tensor_shard.py')
        with h5py.File(target) as h:
            ids=[v.decode() if isinstance(v,bytes) else str(v) for v in h['sample_id'][:]]
            lookup={v:i for i,v in enumerate(ids)}
            arrays=h['psd_seq' if mode=='F32_plus_psd' else mode][:]
    records=[]
    for fold,m in members.groupby('fold',sort=False):
        tr=m.loc[m.partition.eq('train'),'sample_id'];te=m.loc[m.partition.eq('test'),'sample_id']
        if 'random' not in experiment and set(meta.loc[tr].g_config)&set(meta.loc[te].g_config):raise ValueError('Configuration leakage')
        y=frame.loc[tr,'target']
        if y.isna().any():raise ValueError('Missing training target; this experiment needs a task-specific target table.')
        if arrays is not None:
            from sklearn.decomposition import PCA
            xt=arrays[[lookup[s] for s in tr]].reshape(len(tr),-1)
            xe=arrays[[lookup[s] for s in te]].reshape(len(te),-1)
            sc=StandardScaler();xt=sc.fit_transform(xt);xe=sc.transform(xe)
            pc=PCA(n_components=min(64,len(tr)-1,xt.shape[1]),svd_solver='randomized',random_state=SEED)
            xt=pc.fit_transform(xt);xe=pc.transform(xe)
            if mode=='F32_plus_psd':
                sf=StandardScaler();xt=np.c_[xt,sf.fit_transform(frame.loc[tr,features])];xe=np.c_[xe,sf.transform(frame.loc[te,features])]
            est=LogisticRegression(max_iter=5000,class_weight='balanced',random_state=SEED);est.fit(xt,y);yp=est.predict(xe)
        else:
            est=model(name);kw={'mlpclassifier__sample_weight':compute_sample_weight('balanced',y)} if name=='mlp' else {}
            est.fit(x.loc[tr],y,**kw)
            yp=est.predict(x.loc[te])
        records.extend({'sample_id':sid,'fold':fold,'y_true':frame.loc[sid,'target'],'y_pred':pred} for sid,pred in zip(te,yp))
    p=pd.DataFrame(records);dest=ROOT/'reproduction/reruns'/cohort/experiment;dest.mkdir(parents=True,exist_ok=True)
    p.to_csv(dest/'predictions.csv',index=False)
    result=scores(p);(dest/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return result

def verify():
    rows=[]
    for f in sorted((ROOT/'predictions').glob('*/*/predictions.csv')):
        p=pd.read_csv(f);r=scores(p);r['experiment']=str(f.relative_to(ROOT/'predictions').parent)
        if not p.sample_id.is_unique:raise ValueError(f'Duplicate OOF ids: {f}')
        metric=ROOT/'metrics'/f.relative_to(ROOT/'predictions').parent/'metrics.csv'
        if metric.exists():
            m=pd.read_csv(metric);m=m[m.fold.astype(str).eq('pooled')].iloc[0]
            assert abs(r['macro_f1']-m.macro_f1)<1e-10
        rows.append(r)
    out=ROOT/'reproduction/verification_metrics.csv';pd.DataFrame(rows).to_csv(out,index=False)
    print(f'Verified {len(rows)} saved prediction sets; no model fitting.');return rows

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--cohort',choices=['full','low_ber'],default='full');a.add_argument('--experiment',default='baseline_lightgbm');a.add_argument('--model',default='lightgbm');a.add_argument('--verify',action='store_true');args=a.parse_args()
    verify() if args.verify else run(args.cohort,args.experiment,args.model)
