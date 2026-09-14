"""Fixed LightGBM binary, magnitude and recorded-condition holdout analyses."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from train_revision import ROOT, SEED, model

OUT = ROOT/'results'/'targeted'

def run(d, features, name, split_pairs=None):
    d = d.reset_index(drop=True)
    labels = sorted(d.target.unique())
    if split_pairs is None:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        split_pairs = [(str(i),tr,te) for i,(tr,te) in enumerate(cv.split(d,d.target,d.g_config))]
    rows, status = [], []
    for fold,tr,te in split_pairs:
        train, test = d.iloc[tr], d.iloc[te]
        assert not set(train.g_config)&set(test.g_config)
        if set(train.target) != set(labels) or set(test.target) != set(labels):
            status.append(dict(fold=fold,status='NOT_ESTIMABLE_CLASS_SUPPORT',n_train=len(tr),n_test=len(te)))
            continue
        est = model('lightgbm')
        est.fit(train[features],train.target)
        yp = est.predict(test[features])
        for (_,r), pred in zip(test.iterrows(),yp):
            rows.append(dict(sample_id=r.sample_id,fold=fold,y_true=r.target,y_pred=pred,
                             diagnosis_label=r.diagnosis_label,nuisance_cell=r.nuisance_cell,
                             severity_value_db=r.severity_value_db,wss_offset_ghz=r.wss_offset_ghz,
                             wss_bandwidth_ghz=r.wss_bandwidth_ghz,g_config=r.g_config))
        status.append(dict(fold=fold,status='COMPLETE',n_train=len(tr),n_test=len(te)))
    pd.DataFrame(status).to_csv(OUT/f'{name}_status.csv',index=False)
    p = pd.DataFrame(rows)
    if p.empty: return
    assert p.sample_id.is_unique
    p.to_csv(OUT/f'{name}_predictions.csv',index=False)
    metrics = []
    for fold,part in [('pooled',p)] + list(p.groupby('fold')):
        cm = confusion_matrix(part.y_true,part.y_pred,labels=labels)
        normal = part.y_true.eq('normal')
        metrics.append(dict(fold=fold,n=len(part),accuracy=accuracy_score(part.y_true,part.y_pred),
            macro_f1=f1_score(part.y_true,part.y_pred,labels=labels,average='macro',zero_division=0),
            fault_f1=f1_score(part.y_true,part.y_pred,pos_label='fault',average='binary') if labels==['fault','normal'] else None,
            normal_false_alarm=(part.loc[normal,'y_pred']!='normal').mean(),
            fault_missed_detection=(part.loc[~normal,'y_pred']=='normal').mean()))
        pd.DataFrame(cm,index=labels,columns=labels).to_csv(OUT/f'{name}_{fold}_confusion.csv')
    pd.DataFrame(metrics).to_csv(OUT/f'{name}_metrics.csv',index=False)
    print(name,metrics[0],flush=True)
    return p

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(ROOT/'samples.csv'); f=pd.read_csv(ROOT/'features_F32.csv')
    features=json.loads((ROOT/'audit.json').read_text())['features']
    d=s[s.primary].merge(f,on=['sample_id','record_id'],validate='one_to_one')
    d['target']=d.diagnosis_label
    for condition in ['lp_id','modulation','distance_km']:
        # Unknown LP cannot be interpreted as a held-out physical LP setting.
        x=d[d[condition].notna()].reset_index(drop=True)
        pairs=[]
        for value in sorted(x[condition].unique()):
            te=np.flatnonzero(x[condition].eq(value)); tr=np.flatnonzero(~x[condition].eq(value))
            pairs.append((str(value),tr,te))
        run(x,features,'holdout_'+condition,pairs)
    x=d.copy(); x['target']=np.where(x.diagnosis_label.eq('normal'),'normal','fault')
    run(x,features,'binary_any_fault')
    for fault in ['attenuation','edfa','wss_offset','wss_bandwidth']:
        x=d[d.diagnosis_label.isin(['normal',fault])].copy()
        # Require normal and this fault to be present in each condition cell.
        supported=x.groupby('nuisance_cell').diagnosis_label.nunique()
        x=x[x.nuisance_cell.isin(supported[supported.eq(2)].index)].copy()
        x['target']=np.where(x.diagnosis_label.eq('normal'),'normal','fault')
        p=run(x,features,'binary_'+fault)
        setting={'attenuation':'severity_value_db','edfa':'severity_value_db',
                 'wss_offset':'wss_offset_ghz','wss_bandwidth':'wss_bandwidth_ghz'}[fault]
        # A single cross-magnitude model is evaluated by setting; no setting-specific retraining.
        results=[]
        for value,g in p[p.diagnosis_label.eq(fault)].groupby(setting):
            normals=p[p.diagnosis_label.eq('normal') & p.nuisance_cell.isin(g.nuisance_cell)]
            z=pd.concat([g,normals])
            results.append(dict(setting=value,n_fault=len(g),n_normal=len(normals),
                fault_recall=(g.y_pred=='fault').mean(),missed_rate=(g.y_pred=='normal').mean(),
                false_alarm=(normals.y_pred=='fault').mean(),
                fault_f1=f1_score(z.y_true,z.y_pred,pos_label='fault'),
                accuracy=accuracy_score(z.y_true,z.y_pred)))
        pd.DataFrame(results).to_csv(OUT/f'{fault}_magnitude.csv',index=False)

if __name__=='__main__': main()
