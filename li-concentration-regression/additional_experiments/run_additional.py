"""Training-only iteration, robust-loss and outlier-sensitivity experiments.

All fold validation rows and all 273 fixed test rows are retained.
Run with the existing project venv, OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import KFold, ParameterSampler, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

ROOT=Path(__file__).resolve().parent
with np.load(ROOT/'frozen_data.npz') as archive:
    Z={key:archive[key] for key in archive.files}
TRAIN=Z['train']; TEST=Z['test']; Y=Z['y']
X=np.ascontiguousarray(Z['X'][:, Z['nonredundant']]); FULL=Z['X']
FOLDS=list(KFold(5, shuffle=True, random_state=20261018).split(TRAIN))
BASE=RandomForestRegressor(n_estimators=500,max_depth=8,min_samples_leaf=1,
    max_features=.3,ccp_alpha=.001,random_state=42,n_jobs=1)
SIZES=[50,100,300,500,1000,2000]

def output(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)))

def score(y,p): return float(r2_score(y,np.maximum(p,0)))

def make_lgb(kind,n):
    if kind=='LightGBM_original_iterations':
        return LGBMRegressor(n_estimators=n,learning_rate=.03,max_depth=5,num_leaves=20,
            subsample=.8,colsample_bytree=.8,random_state=42,verbosity=-1,n_jobs=1)
    return LGBMRegressor(n_estimators=n,learning_rate=.01,max_depth=-1,num_leaves=7,
        min_child_samples=5,reg_lambda=.1,reg_alpha=0,subsample=1.,subsample_freq=1,
        colsample_bytree=1.,min_split_gain=0,random_state=42,verbosity=-1,n_jobs=1)

def iteration_fold(i,tr,va):
    rows=[];a=TRAIN[tr];b=TRAIN[va]
    model=clone(BASE).set_params(n_estimators=50,warm_start=True,n_jobs=1)
    for n in SIZES:
        model.set_params(n_estimators=n);model.fit(X[a],Y[a])
        rows.append(dict(category='RF_iterations',candidate=f'RF_trees_{n}',fold=i,
            n_estimators=n,R2=score(Y[b],model.predict(X[b])),training_n=len(a),removed_n=0))
    for kind,array in [('LightGBM_original_iterations',FULL),('LightGBM_dedup_iterations',X)]:
        # Refit each size: exact original column-sampling and model initialization behavior.
        for n in SIZES:
            model=make_lgb(kind,n);model.fit(array[a],Y[a])
            rows.append(dict(category=kind,candidate=f'{kind}_{n}',fold=i,
                n_estimators=n,R2=score(Y[b],model.predict(array[b])),training_n=len(a),removed_n=0))
    print('iteration fold',i,'done',flush=True)
    return rows

def filtered_fit(kind,fraction,array,target,ids):
    """The only rows inspected here are the current training fold's rows."""
    keep=np.ones(len(target),dtype=bool);newy=target.copy()
    details={'method':kind,'fraction':fraction}
    if kind in ('target_trim','target_winsor'):
        low,high=np.quantile(target,[fraction/2,1-fraction/2])
        if kind=='target_trim':keep=(target>=low)&(target<=high)
        else:newy=np.clip(target,low,high)
        details.update(lower_ln1p=float(low),upper_ln1p=float(high))
    elif kind=='isolation_X':
        detector=IsolationForest(n_estimators=300,contamination=fraction,random_state=42,n_jobs=1)
        keep=detector.fit_predict(array)==1
    elif kind=='residual_OOF':
        # Inner cross-fitting avoids ranking samples by optimistic in-sample errors.
        estimator=clone(BASE).set_params(n_jobs=1)
        pp=cross_val_predict(estimator,array,target,
            cv=KFold(3,shuffle=True,random_state=20261019),n_jobs=1)
        residual=np.abs(target-np.maximum(pp,0))
        threshold=np.quantile(residual,1-fraction)
        keep=residual<=threshold;details['absolute_residual_cutoff']=float(threshold)
    elif kind!='none':raise ValueError(kind)
    model=clone(BASE).set_params(n_jobs=1);model.fit(array[keep],newy[keep])
    details.update(training_n=len(target),retained_n=int(keep.sum()),removed_n=int((~keep).sum()),
        changed_target_n=int((newy!=target).sum()),
        removed_excel_rows=Z['excel_rows'][ids[~keep]].tolist(),
        changed_target_excel_rows=Z['excel_rows'][ids[newy!=target]].tolist())
    return model,details

def outlier_fold(i,tr,va):
    a=TRAIN[tr];b=TRAIN[va];rows=[];audit=[]
    # Cache inner cross-fitted residuals once per fold for the three cutoffs.
    pp=cross_val_predict(clone(BASE).set_params(n_jobs=1),X[a],Y[a],
        cv=KFold(3,shuffle=True,random_state=20261019),n_jobs=1)
    residual=np.abs(Y[a]-np.maximum(pp,0))
    for kind in ['none','target_trim','target_winsor','isolation_X','residual_OOF']:
        for fraction in ([0.] if kind=='none' else [.01,.02,.05]):
            if kind=='residual_OOF':
                cutoff=np.quantile(residual,1-fraction);keep=residual<=cutoff
                model=clone(BASE).set_params(n_jobs=1);model.fit(X[a[keep]],Y[a[keep]])
                details=dict(method=kind,fraction=fraction,training_n=len(a),retained_n=int(keep.sum()),
                    removed_n=int((~keep).sum()),changed_target_n=0,absolute_residual_cutoff=float(cutoff),
                    removed_excel_rows=Z['excel_rows'][a[~keep]].tolist(),changed_target_excel_rows=[])
            else:model,details=filtered_fit(kind,fraction,X[a],Y[a],a)
            candidate=f'{kind}_{fraction:g}';audit.append(dict(fold=i,candidate=candidate,**details))
            rows.append(dict(category='outlier_sensitivity',candidate=candidate,fold=i,
                R2=score(Y[b],model.predict(X[b])),method=kind,fraction=fraction,
                training_n=len(a),removed_n=details['removed_n'],changed_target_n=details['changed_target_n']))
    print('outlier fold',i,'done',flush=True)
    return rows,audit

def robust_fold(i,tr,va,configs):
    a=TRAIN[tr];b=TRAIN[va];rows=[]
    for number,params in enumerate(configs):
        model=GradientBoostingRegressor(random_state=42,**params)
        model.fit(X[a],Y[a])
        rows.append(dict(category='robust_loss',candidate=f'GB_{number:02d}_{params["loss"]}',fold=i,
            R2=score(Y[b],model.predict(X[b])),training_n=len(a),removed_n=0,params=json.dumps(params)))
    print('robust fold',i,'done',flush=True)
    return rows

def main():
    start=time.time()
    # Frozen before any new CV or test scores: 16 Huber configurations + matched squared-loss controls.
    sampled=list(ParameterSampler(dict(n_estimators=[200,500,1000],learning_rate=[.02,.05,.1],
        max_depth=[1,2,3],min_samples_leaf=[5,15,30],alpha=[.8,.9,.95]),n_iter=16,random_state=20261019))
    configs=[]
    for p in sampled:
        configs.append(dict(p,loss='huber'));configs.append(dict(p,loss='squared_error'))
    protocol=dict(task='Additional sensitivity tests, no claim of exhaustive search or R2 upper bound',
        training_n=1090,test_n=273,fold_seed=20261018,folds=5,target='ln(1+Li)',
        iterations=SIZES,robust_configs=configs,
        outlier_methods=['target_trim','target_winsor','isolation_X','residual_OOF'],fractions=[.01,.02,.05],
        filtering='Fit each rule exclusively on that fold training subset. Never remove validation/test rows.',
        target_trim='Remove observations outside central 99%, 98% or 95% training target quantiles. Actual number depends on ties.',
        target_winsor='Clip training targets to same quantile bounds; do not change validation/test observed targets.',
        residual='3-fold training-only cross-fitting; trim largest absolute log-target residuals by training quantile.',
        scoring='R2 with max(prediction,0); same fixed physical lower limit for all candidates',
        selection='Freeze best candidate of each iteration family, each outlier method, and each loss based ONLY on mean training CV R2. Evaluate all selected candidates once on unchanged test. Also retain RF500 and outlier-none controls.',
        baseline_model=BASE.get_params(),
        warning='Fixed test has been inspected in prior development; it is internal validation, not new external validation.',
        github_sources=['https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_iforest.py',
            'https://github.com/scikit-learn/scikit-learn/blob/main/examples/ensemble/plot_gradient_boosting_regression.py'])
    output('protocol_frozen.json',protocol)
    rows=[]
    for batch in Parallel(n_jobs=4)(delayed(iteration_fold)(i,tr,va) for i,(tr,va) in enumerate(FOLDS,1)):rows+=batch
    pd.DataFrame(rows).to_csv(ROOT/'iteration_fold_scores.csv',index=False)
    audit=[]
    for batch,au in Parallel(n_jobs=4)(delayed(outlier_fold)(i,tr,va) for i,(tr,va) in enumerate(FOLDS,1)):
        rows+=batch;audit+=au
    output('outlier_training_audit.json',audit)
    pd.DataFrame(rows).to_csv(ROOT/'iteration_and_outlier_folds.csv',index=False)
    for batch in Parallel(n_jobs=4)(delayed(robust_fold)(i,tr,va,configs) for i,(tr,va) in enumerate(FOLDS,1)):rows+=batch
    folds=pd.DataFrame(rows);folds.to_csv(ROOT/'all_fold_scores.csv',index=False)
    summary=folds.groupby(['category','candidate'],sort=False).agg(CV_R2=('R2','mean'),CV_std=('R2',lambda x:np.std(x,ddof=1)),
        removed_mean=('removed_n','mean')).reset_index()
    summary.to_csv(ROOT/'training_summary.csv',index=False)
    selected=[]
    for category in ['RF_iterations','LightGBM_original_iterations','LightGBM_dedup_iterations']:
        selected.append(summary.loc[summary[summary.category==category].CV_R2.idxmax(),'candidate'])
    for method in ['target_trim','target_winsor','isolation_X','residual_OOF']:
        subset=summary[(summary.category=='outlier_sensitivity')&summary.candidate.str.startswith(method+'_')]
        selected.append(subset.loc[subset.CV_R2.idxmax(),'candidate'])
    for loss in ['huber','squared_error']:
        subset=summary[(summary.category=='robust_loss')&summary.candidate.str.endswith('_'+loss)]
        selected.append(subset.loc[subset.CV_R2.idxmax(),'candidate'])
    selected+=['none_0','RF_trees_500'];selected=list(dict.fromkeys(selected))
    output('selection_before_test.json',dict(selected=selected,elapsed_training_seconds=time.time()-start))
    # Test touched only after selected candidate names have been written to disk.
    results=[];prediction=pd.DataFrame(dict(excel_row=Z['excel_rows'][TEST],observed_ln1p=Y[TEST],observed_Li=Z['raw_y'][TEST]))
    (ROOT/'models').mkdir(exist_ok=True)
    for name in selected:
        row=folds[folds.candidate==name].iloc[0];category=row.category;array=X;details={}
        if category=='RF_iterations':model=clone(BASE).set_params(n_estimators=int(row.n_estimators),n_jobs=1);model.fit(X[TRAIN],Y[TRAIN])
        elif category.startswith('LightGBM'):
            model=make_lgb(category,int(row.n_estimators));array=FULL if category=='LightGBM_original_iterations' else X
            model.fit(array[TRAIN],Y[TRAIN])
        elif category=='robust_loss':model=GradientBoostingRegressor(random_state=42,**json.loads(row.params));model.fit(X[TRAIN],Y[TRAIN])
        else:
            model,details=filtered_fit(row.method,float(row.fraction),X[TRAIN],Y[TRAIN],TRAIN)
        p=np.maximum(model.predict(array[TEST]),0);cv=summary[summary.candidate==name].iloc[0]
        result=dict(candidate=name,category=category,CV_R2=cv.CV_R2,CV_std=cv.CV_std,
            test_R2=score(Y[TEST],p),test_RMSE=float(np.sqrt(mean_squared_error(Y[TEST],p))),test_MAE=float(mean_absolute_error(Y[TEST],p)),
            test_n=len(TEST),removed_n=details.get('removed_n',0),changed_target_n=details.get('changed_target_n',0))
        results.append(result);prediction[name]=p
        if details:output(name+'_full_training_audit.json',details)
        joblib.dump(model,ROOT/'models'/f'{name}.joblib')
        print('TEST',json.dumps(result),flush=True)
    pd.DataFrame(results).to_csv(ROOT/'selected_test_results.csv',index=False)
    prediction.to_csv(ROOT/'selected_test_predictions.csv',index=False)
    output('completion.json',dict(candidate_count=len(summary),fold_fit_scores=len(folds),
        elapsed_seconds=time.time()-start,selected_test_evaluations=len(results)))
    print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
