"""Conditional configuration-bootstrap intervals for saved held-out predictions."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.metrics import confusion_matrix
def metrics(cm,labels):
 tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(len(tp)),where=den>0)
 n=labels.index('normal');support=cm.sum(1);total=cm.sum()
 out={'macro_f1':float(f.mean()),'accuracy':float(tp.sum()/total),'normal_false_alarm':float((support[n]-cm[n,n])/support[n]),'fault_missed_detection':float((cm[:,n].sum()-cm[n,n])/(total-support[n]))}
 if 'fault' in labels:out['fault_f1']=float(f[labels.index('fault')])
 return out
def intervals(p,replicates=2000):
 labels=['normal','fault'] if set(p.y_true)=={'normal','fault'} else ['normal','attenuation','edfa','wss_offset','wss_bandwidth']
 out=metrics(confusion_matrix(p.y_true,p.y_pred,labels=labels),labels)
 mats=np.stack([confusion_matrix(x.y_true,x.y_pred,labels=labels) for _,x in p.groupby('g_config')]);rng=np.random.default_rng(20260911);samples=[]
 for _ in range(replicates):
  c=mats[rng.integers(len(mats),size=len(mats))].sum(0)
  if (c.sum(1)==0).any():continue
  samples.append(metrics(c,labels))
 for key in list(out):out[key+'_low'],out[key+'_high']=np.quantile([x[key] for x in samples],[.025,.975]).tolist()
 out.update(bootstrap_requested=replicates,bootstrap_valid=len(samples));return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('predictions',type=Path);a.add_argument('--replicates',type=int,default=2000);v=a.parse_args();print(json.dumps(intervals(pd.read_csv(v.predictions),v.replicates),indent=2))
