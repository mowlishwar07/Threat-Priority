"""Train only from observed features; keep labels and scenario names outside X."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ml.schema import NUMERIC, CATEGORICAL, FEATURES, prepare, labels, read_csv
from ml.dataset import write_data
from ml.evaluation import evaluate

ROOT = Path(__file__).resolve().parent

def make_preprocessor():
    return ColumnTransformer([
        ('num',SimpleImputer(strategy='median',keep_empty_features=True),NUMERIC),
        ('missing',MissingIndicator(features='all'),NUMERIC),
        ('cat',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='__missing__')),
                         ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CATEGORICAL)
    ], verbose_feature_names_out=False)

def train(csv_path, output, provenance, mapping=None):
    raw,x,ts,warnings = prepare(read_csv(Path(csv_path).read_bytes()),mapping)
    y,reason = labels(raw)
    if y is None or len(set(y)) < 2:
        raise ValueError(reason or 'Training requires both benign and attack labels.')
    # Remove feature-identical records before splitting; a duplicated record cannot leak across sets.
    mask = ~x.astype(str).duplicated()
    removed = int((~mask).sum())
    x,y,ts = x.loc[mask].reset_index(drop=True),y[mask],ts.loc[mask].reset_index(drop=True)
    if len(x)<200 or min(np.bincount(y))<30:
        raise ValueError('Training needs at least 200 unique records and 30 examples of each class.')
    all_ids = np.arange(len(x))
    tr,rest = train_test_split(all_ids,test_size=.4,stratify=y,random_state=17)
    va,te = train_test_split(rest,test_size=.5,stratify=y[rest],random_state=18)
    pre = make_preprocessor()
    # Training-only masks teach missingness handling, without using validation/test statistics.
    augmented = x.iloc[tr].copy()
    rng = np.random.default_rng(19)
    for c in FEATURES:
        augmented.loc[rng.random(len(augmented))<.06,c] = np.nan
    fit_x = pd.concat([x.iloc[tr],augmented],ignore_index=True)
    fit_y = np.tile(y[tr],2)
    pre.fit(fit_x)
    z = pre.transform(fit_x)
    rf = RandomForestClassifier(n_estimators=120,max_depth=10,min_samples_leaf=5,random_state=20,n_jobs=2)
    rf.fit(z,fit_y)
    lr = Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=1200,random_state=20))])
    lr.fit(z,fit_y)
    validation = {}
    for title,model in [('Random Forest',rf),('Logistic Regression',lr)]:
        validation[title] = evaluate(y[va],model.predict_proba(pre.transform(x.iloc[va]))[:,1],ts.iloc[va],20)
    # Fixed 0.5 detection threshold; operational severity bands are independent of this decision.
    began = time.perf_counter()
    test_p = rf.predict_proba(pre.transform(x.iloc[te]))[:,1]
    batch_ms = (time.perf_counter()-began)*1000
    metrics = evaluate(y[te],test_p,ts.iloc[te],20)
    groups = NUMERIC + NUMERIC
    encoder = pre.named_transformers_['cat'].named_steps['encode']
    for field,cats in zip(CATEGORICAL,encoder.categories_):
        groups.extend([field]*len(cats))
    model_dir = Path(output)
    model_dir.mkdir(exist_ok=True,parents=True)
    background = pre.transform(x.iloc[rng.choice(tr,min(64,len(tr)),replace=False)])
    bundle = {'preprocessor':pre,'model':rf,'background':background,'groups':groups,
              'metadata':{'provenance':provenance,'model':'Random Forest','score_definition':'100 x model estimated attack probability (uncalibrated proxy)',
                          'feature_schema':FEATURES,'split':'60/20/20 stratified random, duplicate feature rows removed',
                          'threshold':.5,'training_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),
                          'duplicates_removed':removed,'library_versions':{'scikit-learn':'1.6.1','shap':'0.48.0'},
                          'selection':'Random Forest chosen for native Tree SHAP; Logistic Regression is a validation baseline.'},
              'validation':validation,'test_metrics':metrics}
    joblib.dump(bundle,model_dir/'risk_model.joblib',compress=3)
    report = {'metadata':bundle['metadata'],'validation_comparison':validation,'test':metrics,
              'batch_scoring_ms':batch_ms,'mean_batch_ms_per_event':batch_ms/len(te),'input_warnings':warnings}
    report_dir = ROOT/'reports' if model_dir==ROOT/'models' else model_dir
    report_dir.mkdir(exist_ok=True,parents=True)
    (report_dir/'metrics.json').write_text(json.dumps(report,indent=2))
    pd.concat([raw.loc[mask].iloc[te].reset_index(drop=True),pd.Series(test_p*100,name='risk_score')],axis=1).to_csv(report_dir/'held_out_predictions.csv',index=False)
    (ROOT/'data'/'held_out_test.csv' if model_dir==ROOT/'models' else model_dir/'held_out_test.csv').write_text(raw.loc[mask].iloc[te].to_csv(index=False))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train a binary risk model from the documented feature schema.')
    parser.add_argument('--csv',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'models')
    parser.add_argument('--provenance',help='Dataset title, source URL, and any sampling details')
    parser.add_argument('--mapping',type=Path,help='Optional JSON mapping of source headers to canonical names')
    args=parser.parse_args()
    if args.csv is None:
        write_data(ROOT/'data')
        args.csv=ROOT/'data'/'synthetic_events.csv'
        args.provenance=args.provenance or 'Synthetic scenario generator v2; not captured traffic'
    else:
        args.provenance=args.provenance or 'User-supplied labeled CSV; source not specified'
    mapping=json.loads(args.mapping.read_text()) if args.mapping else None
    train(args.csv,args.output,args.provenance,mapping)
