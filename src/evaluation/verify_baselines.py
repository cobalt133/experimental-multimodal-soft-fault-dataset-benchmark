"""Independent saved-OOF checks and configuration bootstrap uncertainty."""
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from train_revision import ROOT

def macro(cm):
    den=cm.sum(axis=0)+cm.sum(axis=1)
    return np.divide(2*np.diag(cm),den,out=np.zeros(len(den)),where=den>0).mean()

def main():
    labels=json.loads((ROOT/'audit.json').read_text())['classes']
    s=pd.read_csv(ROOT/'samples.csv')
    reports=[]
    for cohort in ['primary','common_support']:
        out=ROOT/'results'/cohort
        split=pd.read_csv(out/'frozen_test_folds.csv')
        expected=s[s[cohort]]
        assert set(split.sample_id)==set(expected.sample_id)
        assert split.sample_id.is_unique
        assert split.groupby('g_config').fold.nunique().max()==1
        for name in ['logistic_regression','random_forest','lightgbm','mlp']:
            p=pd.read_csv(out/f'{name}_oof.csv')
            assert p.sample_id.is_unique and set(p.sample_id)==set(expected.sample_id)
            j=p.merge(split,on='sample_id',validate='one_to_one',suffixes=('_pred','_split'))
            assert j.fold_pred.equals(j.fold_split)
            truth=p.merge(expected[['sample_id','diagnosis_label']],on='sample_id',validate='one_to_one')
            assert truth.y_true.equals(truth.diagnosis_label)
            cm=confusion_matrix(p.y_true,p.y_pred,labels=labels)
            saved=pd.read_csv(out/f'{name}_confusion.csv',index_col=0)
            assert np.array_equal(cm,saved.to_numpy())
            mats=np.stack([confusion_matrix(g.y_true,g.y_pred,labels=labels) for _,g in j.groupby('g_config')])
            rng=np.random.default_rng(20260911)
            boot=[]
            for _ in range(2000):
                sampled=mats[rng.integers(len(mats),size=len(mats))].sum(axis=0)
                if (sampled.sum(axis=1)==0).any(): continue
                boot.append(macro(sampled))
            lo,hi=np.quantile(boot,[.025,.975])
            reports.append(dict(cohort=cohort,model=name,macro_f1=macro(cm),ci_low=lo,ci_high=hi,
                valid_bootstrap=len(boot),requested_bootstrap=2000,n=len(p),groups=len(mats),
                verification='PASS',oof_sha256=hashlib.sha256((out/f'{name}_oof.csv').read_bytes()).hexdigest()))
    pd.DataFrame(reports).to_csv(ROOT/'results/baseline_independent_verification.csv',index=False)
    print(pd.DataFrame(reports).drop(columns='oof_sha256').to_string(index=False))

if __name__=='__main__': main()
