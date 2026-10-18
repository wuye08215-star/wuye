"""RF variable-count comparison; no outer-validation/fixed-test feature ranking.

The 63 subsets are explicitly enumerated because only 6 unique information
groups remain. Raw-column top-k ranking is re-estimated inside every training
fold, using an inner train/validation split and permutation importance.
"""
from pathlib import Path
from itertools import combinations
import json, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

R=Path(__file__).resolve().parent
with np.load(R/'frozen_data.npz') as archive:Z={k:archive[k] for k in archive.files}
X=Z['X'];Y=Z['y'];TRAIN=Z['train'];TEST=Z['test'];UNIQUE=Z['nonredundant'].tolist()
NAMES=json.loads((R/'source_protocol.json').read_text())['features']
SUBSETS=[list(c) for k in range(1,7) for c in combinations(UNIQUE,k)]
FOLDS=list(KFold(5,shuffle=True,random_state=20261018).split(TRAIN))

def original():return RandomForestRegressor(n_estimators=1000,max_depth=10,min_samples_leaf=3,random_state=42,n_jobs=1)
def modified():return RandomForestRegressor(n_estimators=500,max_depth=8,min_samples_leaf=1,max_features=.3,ccp_alpha=.001,random_state=42,n_jobs=1)
def array(rows,cols):return np.ascontiguousarray(X[rows][:,cols])
def score(estimator,x,y):return r2_score(y,np.maximum(estimator.predict(x),0))
def save(name,obj):(R/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def candidate(cols):return 'subset_'+'_'.join(str(c) for c in cols)

def rank_training(rows):
    inner_train,inner_val=train_test_split(rows,test_size=.2,random_state=20261020)
    model=original();model.fit(array(inner_train,list(range(23))),Y[inner_train])
    imp=permutation_importance(model,array(inner_val,list(range(23))),Y[inner_val],scoring=score,
        n_repeats=10,random_state=20261020,n_jobs=1)
    ranked=np.argsort(-imp.importances_mean,kind='stable').tolist()
    audit=dict(ranking_train_excel_rows=Z['excel_rows'][inner_train].tolist(),
        ranking_validation_excel_rows=Z['excel_rows'][inner_val].tolist(),
        ranked_columns=ranked,ranked_features=[NAMES[c] for c in ranked],
        permutation_mean=imp.importances_mean.tolist(),permutation_std=imp.importances_std.tolist())
    return ranked,audit

def fit_score(model,cols,a,b):
    model.fit(array(a,cols),Y[a]);return float(score(model,array(b,cols),Y[b]))

def fold_task(fold,tr,va):
    a=TRAIN[tr];b=TRAIN[va];rows=[]
    for cols in SUBSETS:
        val=fit_score(modified(),cols,a,b)
        rows.append(dict(candidate=candidate(cols),family='subset_modified_RF',fold=fold,
            n_features=len(cols),columns=json.dumps(cols),features=' | '.join(NAMES[c] for c in cols),R2=val))
    ranked,audit=rank_training(a);audit['fold']=fold
    for name,model,cols in [('original23',original(),list(range(23))),
        ('modified_params23',modified(),list(range(23))),
        ('training_rank_top6_original_RF',original(),ranked[:6]),
        ('training_rank_top7_original_RF',original(),ranked[:7])]:
        rows.append(dict(candidate=name,family='raw_column_comparison',fold=fold,
            n_features=len(cols),columns=json.dumps(cols),features=' | '.join(NAMES[c] for c in cols),
            R2=fit_score(model,cols,a,b)))
    print('fold',fold,'complete; top6',[NAMES[c] for c in ranked[:6]],flush=True)
    return rows,audit

def main():
    start=time.time()
    save('protocol_before_scores.json',dict(task='Assess retaining top 6/7 raw columns vs subsets of 6 nonredundant information groups',
        n=1363,training_n=1090,test_n=273,target='ln(1+Li)',original_filter='Li <= 2000',
        unique_column_indices=UNIQUE,unique_features=[NAMES[i] for i in UNIQUE],
        subsets=SUBSETS,subset_count=63,total_recipes=67,
        selection='Mean 5-fold TRAINING validation R2 only; select one subset per feature count 1..6. Break exact ties by original index order. Freeze names before fixed test evaluation.',
        fold_seed=20261018,ranking='Raw top6/top7: inner 80/20 split with seed20261020 within each outer TRAINING fold; original RF fitted on inner train, permutation R2 on inner validation, 10 repeats. Refit selected columns on all outer train. Validation/test are never used to rank.',
        subset_RF_params=modified().get_params(),original_RF_params=original().get_params(),
        no_new_sample_filtering=True,test_caution='Previously inspected internal development test; not untouched external validation.',
        source='https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/feature_selection'))
    rows=[];audits=[]
    for rr,au in Parallel(n_jobs=4)(delayed(fold_task)(i,tr,va) for i,(tr,va) in enumerate(FOLDS,1)):
        rows+=rr;audits.append(au)
    scores=pd.DataFrame(rows);scores.to_csv(R/'all_fold_scores.csv',index=False);save('fold_ranking_audit.json',audits)
    summary=scores.groupby(['candidate','family','n_features'],sort=False).agg(CV_R2=('R2','mean'),CV_std=('R2','std')).reset_index()
    summary.to_csv(R/'training_summary.csv',index=False)
    selected=[]
    for k in range(1,7):
        sub=summary[(summary.family=='subset_modified_RF')&(summary.n_features==k)]
        selected.append(sub.loc[sub.CV_R2.idxmax(),'candidate'])
    sub=summary[summary.family=='subset_modified_RF'];overall=sub.loc[sub.CV_R2.idxmax(),'candidate']
    selected+=['original23','modified_params23','training_rank_top6_original_RF','training_rank_top7_original_RF']
    ranked,audit=rank_training(TRAIN);save('full_training_ranking_audit.json',audit)
    save('selection_before_test.json',dict(selected_per_count=selected[:6],overall_training_CV_selected=overall,
        all_evaluated=selected,final_raw_top6=ranked[:6],final_raw_top7=ranked[:7]))
    results=[];predictions=pd.DataFrame(dict(excel_row=Z['excel_rows'][TEST],observed_ln1p=Y[TEST],observed_Li=Z['raw_y'][TEST]))
    for name in selected:
        if name.startswith('subset_'):cols=[int(s) for s in name.split('_')[1:]];model=modified()
        elif name=='modified_params23':cols=list(range(23));model=modified()
        elif name=='original23':cols=list(range(23));model=original()
        else:cols=ranked[:6 if 'top6' in name else 7];model=original()
        model.fit(array(TRAIN,cols),Y[TRAIN]);p=np.maximum(model.predict(array(TEST,cols)),0)
        cv=summary[summary.candidate==name].iloc[0]
        r=dict(candidate=name,n_features=len(cols),features=' | '.join(NAMES[c] for c in cols),
            columns=json.dumps(cols),CV_R2=float(cv.CV_R2),CV_std=float(cv.CV_std),
            test_R2=float(r2_score(Y[TEST],p)),test_RMSE=float(np.sqrt(mean_squared_error(Y[TEST],p))),
            test_MAE=float(mean_absolute_error(Y[TEST],p)),training_n=len(TRAIN),test_n=len(TEST),
            selected_by_training_CV=name==overall)
        results.append(r);predictions[name]=p
        print('TEST',json.dumps(r,ensure_ascii=False),flush=True)
    pd.DataFrame(results).to_csv(R/'selected_test_results.csv',index=False)
    predictions.to_csv(R/'selected_test_predictions.csv',index=False)
    save('completion.json',dict(recipes=67,fold_scores=len(scores),elapsed_seconds=time.time()-start))
    print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
