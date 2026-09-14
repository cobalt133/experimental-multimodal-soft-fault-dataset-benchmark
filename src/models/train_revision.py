"""Reference baselines. All preprocessing fitted inside training folds."""
from pathlib import Path
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1] / 'revision_v2'
SEED = 20260911

def model(name):
    if name == 'logistic_regression':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced', random_state=SEED))
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=4, random_state=SEED)
    if name == 'mlp':
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64,32), activation='relu', solver='adam',
            alpha=0.001, batch_size=64, learning_rate_init=0.001,
            max_iter=1000, early_stopping=False, random_state=SEED))
    if name != 'lightgbm':
        raise ValueError(name)
    return LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15,
                          class_weight='balanced', n_jobs=4, random_state=SEED, verbosity=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=['primary','common_support'], default='common_support')
    parser.add_argument('--models', nargs='+', default=['logistic_regression','random_forest','lightgbm','mlp'],
                        choices=['logistic_regression','random_forest','lightgbm','mlp'])
    args = parser.parse_args()
    s = pd.read_csv(ROOT/'samples.csv')
    f = pd.read_csv(ROOT/'features_F32.csv')
    features = json.loads((ROOT/'audit.json').read_text())['features']
    d = s[s[args.cohort]].merge(f, on=['sample_id','record_id'], validate='one_to_one').reset_index(drop=True)
    out = ROOT/'results'/args.cohort
    out.mkdir(parents=True, exist_ok=True)
    labels = json.loads((ROOT/'audit.json').read_text())['classes']
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(d, d.diagnosis_label, d.g_config))
    assignments = []
    for fold, (tr, te) in enumerate(splits):
        assert not set(d.iloc[tr].g_config) & set(d.iloc[te].g_config)
        assert set(d.iloc[tr].diagnosis_label) == set(labels)
        assert set(d.iloc[te].diagnosis_label) == set(labels)
        assignments.extend({'sample_id':d.iloc[i].sample_id,'g_config':d.iloc[i].g_config,'fold':fold} for i in te)
    assignment_frame = pd.DataFrame(assignments)
    frozen = out/'frozen_test_folds.csv'
    if frozen.exists():
        pd.testing.assert_frame_equal(pd.read_csv(frozen), assignment_frame)
    else:
        assignment_frame.to_csv(frozen, index=False)
    summaries = pd.read_csv(out/'summary.csv').to_dict('records') if (out/'summary.csv').exists() else []
    for name in args.models:
        predictions = []
        diagnostics = []
        for fold,(tr,te) in enumerate(splits):
            est = model(name)
            fit_kwargs = {}
            if name == 'mlp':
                fit_kwargs['mlpclassifier__sample_weight'] = compute_sample_weight('balanced', d.iloc[tr].diagnosis_label)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                est.fit(d.iloc[tr][features], d.iloc[tr].diagnosis_label, **fit_kwargs)
            diagnostics.append(dict(fold=fold, warnings=[str(w.message) for w in caught],
                n_iter=int(est.named_steps['mlpclassifier'].n_iter_) if name=='mlp' else None))
            yp = est.predict(d.iloc[te][features])
            for i, pred in zip(te, yp):
                predictions.append(dict(sample_id=d.iloc[i].sample_id, fold=fold,
                                        y_true=d.iloc[i].diagnosis_label, y_pred=pred))
        p = pd.DataFrame(predictions)
        assert p.sample_id.is_unique and len(p) == len(d)
        p.to_csv(out/f'{name}_oof.csv', index=False)
        cm = confusion_matrix(p.y_true,p.y_pred,labels=labels)
        pd.DataFrame(cm,index=labels,columns=labels).to_csv(out/f'{name}_confusion.csv')
        rates = []
        for i, label in enumerate(labels):
            tp = int(cm[i,i]); fn = int(cm[i,:].sum()-tp); fp = int(cm[:,i].sum()-tp)
            tn = int(cm.sum()-tp-fn-fp)
            rates.append(dict(label=label, TP=tp, FN=fn, FP=fp, TN=tn,
                              recall=tp/(tp+fn), missed_rate=fn/(tp+fn),
                              false_positive_rate=fp/(fp+tn), F1=2*tp/(2*tp+fp+fn)))
        pd.DataFrame(rates).to_csv(out/f'{name}_per_class.csv', index=False)
        normal = p.y_true.eq('normal'); predicted_normal = p.y_pred.eq('normal')
        row = dict(model=name, cohort=args.cohort, n=len(p), groups=d.g_config.nunique(),
                   macro_f1=f1_score(p.y_true,p.y_pred,average='macro'),
                   accuracy=accuracy_score(p.y_true,p.y_pred),
                   normal_false_alarm=float((~predicted_normal[normal]).mean()),
                   fault_missed_detection=float(predicted_normal[~normal].mean()))
        summaries = [r for r in summaries if r['model'] != name] + [row]
        (out/f'{name}_training_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
        print(json.dumps(row), flush=True)
    import importlib.metadata as md
    (out/'environment.json').write_text(json.dumps({x:md.version(x) for x in ['pandas','numpy','scikit-learn','lightgbm']},indent=2))

if __name__ == '__main__':
    main()
