"""Optional RF + SVR averaging, weight chosen on training OOF folds only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R=Path(__file__).resolve().parent
z=np.load(R/'frozen_data.npz');a=z['train'];b=z['test'];y=z['y']
x=np.ascontiguousarray(z['X'][:,z['nonredundant']])
models={'RF':RandomForestRegressor(n_estimators=500,max_depth=8,min_samples_leaf=1,
    max_features=.3,ccp_alpha=.001,random_state=42,n_jobs=1),
    'SVR':make_pipeline(StandardScaler(),SVR(C=7.873647301696307,gamma=1.4490330125116613,epsilon=.05))}
weights=np.linspace(0,1,11)
(R/'ensemble_protocol_before_scores.json').write_text(json.dumps(dict(weights_RF=weights.tolist(),
    scoring='mean of five training validation fold R2 scores; all predictions max(raw,0)',
    selection='choose one RF weight before fixed test evaluation',
    models={name:str(model) for name,model in models.items()},CV_seed=20261018),indent=2))
folds=list(KFold(5,shuffle=True,random_state=20261018).split(a));oof={}
for family,estimator in models.items():
    model=clone(estimator)
    if family=='RF':model.set_params(n_jobs=1)
    oof[family]=np.maximum(cross_val_predict(model,x[a],y[a],cv=folds,n_jobs=4),0)
records=[]
for w in weights:
    p=w*oof['RF']+(1-w)*oof['SVR']
    scores=[r2_score(y[a[va]],p[va]) for _,va in folds]
    records.append(dict(weight_RF=w,weight_SVR=1-w,CV_R2=np.mean(scores),CV_std=np.std(scores,ddof=1),fold_R2=scores))
pd.DataFrame(records).to_json(R/'ensemble_training_scores.json',orient='records',indent=2)
best=max(records,key=lambda v:v['CV_R2'])
(R/'ensemble_selection_before_test.json').write_text(json.dumps(best,indent=2))
for estimator in models.values():estimator.fit(x[a],y[a])
p=sum(best['weight_'+family]*np.maximum(estimator.predict(x[b]),0) for family,estimator in models.items())
result=dict(**best,test_R2=r2_score(y[b],p),test_RMSE=float(np.sqrt(mean_squared_error(y[b],p))),
    test_MAE=mean_absolute_error(y[b],p),test_n=len(b))
(R/'ensemble_test_result.json').write_text(json.dumps(result,indent=2))
pd.DataFrame(dict(excel_row=z['excel_rows'][b],observed_ln1p=y[b],prediction_ln1p=p)).to_csv(R/'ensemble_test_predictions.csv',index=False)
pd.DataFrame(dict(excel_row=z['excel_rows'][a],observed_ln1p=y[a],RF_OOF=oof['RF'],SVR_OOF=oof['SVR'])).to_csv(R/'ensemble_training_oof.csv',index=False)
print(json.dumps(result),flush=True)
