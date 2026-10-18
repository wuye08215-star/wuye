"""Compare three new regression families, selecting only by training CV."""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parent
with np.load(ROOT/'frozen_data.npz') as d:Z={k:d[k] for k in d.files}
X=Z['X'];Y=Z['y'];TRAIN=Z['train'];TEST=Z['test']
NAMES=json.loads((ROOT/'source_protocol.json').read_text())['features']
FEATURES={'four':[18,19,20,22],'six':[0,18,19,20,21,22]}
FOLDS=list(KFold(5,shuffle=True,random_state=20261018).split(TRAIN))

def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def arr(rows,cols):return np.ascontiguousarray(X[rows][:,cols])
def make_model(family,params):
    if family=='XGBoost':
        return XGBRegressor(objective='reg:squarederror',tree_method='hist',device='cpu',random_state=42,n_jobs=1,**params)
    if family=='GaussianProcess':
        base=RBF(params['length_scale']) if params['kernel']=='RBF' else Matern(params['length_scale'],nu=1.5)
        gp=GaussianProcessRegressor(kernel=base+WhiteKernel(params['noise']),optimizer=None,normalize_y=True,alpha=1e-8)
        return make_pipeline(StandardScaler(),gp)
    return make_pipeline(SplineTransformer(n_knots=params['n_knots'],degree=params['degree'],knots='quantile',include_bias=False),
        StandardScaler(),Ridge(alpha=params['alpha']))

def configs():
    xgb=list(ParameterSampler(dict(n_estimators=[150,300,600,1000],learning_rate=[.01,.03,.06,.1],max_depth=[1,2,3,4],
        min_child_weight=[3,10,30],reg_lambda=[1.,10.,50.,100.],reg_alpha=[0.,.1,1.],
        subsample=[.7,.85,1.],colsample_bytree=[.7,1.]),n_iter=24,random_state=20261021))
    gp=[dict(kernel=k,length_scale=l,noise=n) for k in ['RBF','Matern15'] for l in [.3,1.,3.,10.] for n in [.1,.5]]
    splines=list(ParameterSampler(dict(n_knots=[3,5,8,12,20],degree=[2,3],alpha=[.01,.1,1.,10.,100.]),n_iter=24,random_state=20261021))
    allconfigs=[]
    for family,items in [('XGBoost',xgb),('GaussianProcess',gp),('SplineRidge',splines)]:
        for feature,cols in FEATURES.items():
            for index,p in enumerate(items):allconfigs.append(dict(candidate=f'{family}_{feature}_{index:02d}',family=family,feature_set=feature,columns=cols,params=p))
    return allconfigs

def evaluate(c):
    rows=[]
    for fold,(tr,va) in enumerate(FOLDS,1):
        a=TRAIN[tr];b=TRAIN[va];model=make_model(c['family'],c['params'])
        with warnings.catch_warnings(record=True) as ww:
            warnings.simplefilter('always');model.fit(arr(a,c['columns']),Y[a]);p=np.maximum(model.predict(arr(b,c['columns'])),0)
        rows.append(dict(candidate=c['candidate'],family=c['family'],feature_set=c['feature_set'],fold=fold,
            R2=float(r2_score(Y[b],p)),warning=' | '.join(str(w.message) for w in ww)))
    return rows

def main():
    start=time.time();allconfigs=configs();save('candidates_before_scores.json',allconfigs)
    save('protocol_before_scores.json',dict(families=['XGBoost','GaussianProcess','SplineRidge'],candidate_count=len(allconfigs),
        training_n=1090,test_n=273,feature_sets=FEATURES,target='ln(1+Li)',original_filter='Li <= 2000',
        CV='KFold(5,shuffle=True,random_state=20261018)',selection='Highest mean training CV R2 within each family across 4/6 feature sets. Freeze all three before fixed test evaluation.',
        physical_bound='max(prediction_ln1p,0), same for CV/test',additional_sample_deletion=False,
        GP='Training-only StandardScaler; normalize_y=True; RBF or Matern1.5 plus white noise. Fixed hyperparameters searched by CV, no kernel optimizer.',
        SplineRidge='Additive B-spline bases plus Ridge. Knots, scaling and fit within each fold. No explicit interactions; GAM-like additive regression, not a full GAM implementation.',
        XGBoost='Official xgboost-cpu package 3.4.1, squared-error objective, CPU histogram trees.',
        caution='Four-feature choice was already selected on the training data in previous work; this CV is conditional development evidence. Fixed test has been viewed previously; it is not untouched external validation.',
        sources=['https://github.com/dmlc/xgboost','https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/gaussian_process','https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_polynomial.py']))
    print('START',len(allconfigs),'configurations',flush=True)
    rows=[]
    for i,batch in enumerate(Parallel(n_jobs=4,return_as='generator')(delayed(evaluate)(c) for c in allconfigs),1):
        rows+=batch
        if i%16==0:print('CV completed',i,'/',len(allconfigs),flush=True)
    folds=pd.DataFrame(rows);folds.to_csv(ROOT/'all_fold_scores.csv',index=False)
    summary=folds.groupby(['candidate','family','feature_set'],sort=False).agg(CV_R2=('R2','mean'),CV_std=('R2','std')).reset_index()
    summary.to_csv(ROOT/'training_summary.csv',index=False)
    selected=[]
    for family in ['XGBoost','GaussianProcess','SplineRidge']:
        sub=summary[summary.family==family];name=sub.loc[sub.CV_R2.idxmax(),'candidate'];c=next(c for c in allconfigs if c['candidate']==name)
        selected.append(dict(**c,CV_R2=float(sub.loc[sub.candidate==name,'CV_R2'].iloc[0]),CV_std=float(sub.loc[sub.candidate==name,'CV_std'].iloc[0])))
    save('selection_before_test.json',selected)
    results=[];prediction=pd.DataFrame(dict(excel_row=Z['excel_rows'][TEST],observed_ln1p=Y[TEST],observed_Li=Z['raw_y'][TEST]))
    for c in selected:
        model=make_model(c['family'],c['params']);model.fit(arr(TRAIN,c['columns']),Y[TRAIN]);raw=model.predict(arr(TEST,c['columns']));p=np.maximum(raw,0)
        row=dict(family=c['family'],candidate=c['candidate'],feature_set=c['feature_set'],features=' | '.join(NAMES[j] for j in c['columns']),
            CV_R2=c['CV_R2'],CV_std=c['CV_std'],test_R2=float(r2_score(Y[TEST],p)),
            test_RMSE=float(np.sqrt(mean_squared_error(Y[TEST],p))),test_MAE=float(mean_absolute_error(Y[TEST],p)),
            training_n=len(TRAIN),test_n=len(TEST),negative_raw_n=int((raw<0).sum()),delta_vs_RF4=float(r2_score(Y[TEST],p)-.5538115677138247))
        # XGBoost returns float32; export as float64 to retain its exact values in CSV.
        results.append(row);prediction[c['family']]=np.asarray(p,dtype=float)
        print('TEST',json.dumps(row,ensure_ascii=False),flush=True)
    pd.DataFrame(results).to_csv(ROOT/'selected_test_results.csv',index=False)
    prediction.to_csv(ROOT/'selected_test_predictions.csv',index=False)
    save('completion.json',dict(candidates=len(allconfigs),fold_scores=len(folds),elapsed_seconds=time.time()-start))
    print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
