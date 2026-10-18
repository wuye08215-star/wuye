"""Complementarity check: RF + newly selected GP; training-only weight choice."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from compare_new_methods import ROOT,Z,TRAIN,TEST,Y,FOLDS,arr,make_model

choices=json.loads((ROOT/'selection_before_test.json').read_text())
gp=next(c for c in choices if c['family']=='GaussianProcess');cols=gp['columns']
assert cols==[18,19,20,22]
rf=RandomForestRegressor(n_estimators=500,max_depth=8,min_samples_leaf=1,max_features=.3,ccp_alpha=.001,random_state=42,n_jobs=1)
gpr=make_model('GaussianProcess',gp['params']);weights=np.linspace(0,1,11)
(ROOT/'ensemble_protocol_before_scores.json').write_text(json.dumps(dict(weight_RF=weights.tolist(),
    method='VotingRegressor on four already-selected inputs; nonnegative lower bound on final prediction',
    GP_params=gp['params'],selection='Highest mean five-fold training CV R2. Freeze weight before fixed test prediction.'),indent=2))
rf_oof=cross_val_predict(rf,arr(TRAIN,cols),Y[TRAIN],cv=FOLDS,n_jobs=4)
gp_oof=cross_val_predict(gpr,arr(TRAIN,cols),Y[TRAIN],cv=FOLDS,n_jobs=4)
rows=[]
for w in weights:
    p=np.maximum(w*rf_oof+(1-w)*gp_oof,0)
    scores=[r2_score(Y[TRAIN[va]],p[va]) for _,va in FOLDS]
    rows.append(dict(weight_RF=float(w),weight_GP=float(1-w),CV_R2=float(np.mean(scores)),CV_std=float(np.std(scores,ddof=1)),fold_R2=scores))
pd.DataFrame(rows).to_json(ROOT/'ensemble_training_scores.json',orient='records',indent=2)
best=max(rows,key=lambda r:r['CV_R2'])
(ROOT/'ensemble_selection_before_test.json').write_text(json.dumps(best,indent=2))
model=VotingRegressor([('rf',rf),('gp',gpr)],weights=[best['weight_RF'],best['weight_GP']],n_jobs=1)
model.fit(arr(TRAIN,cols),Y[TRAIN]);p=np.maximum(model.predict(arr(TEST,cols)),0)
result=dict(**best,test_R2=r2_score(Y[TEST],p),test_RMSE=float(np.sqrt(mean_squared_error(Y[TEST],p))),
    test_MAE=mean_absolute_error(Y[TEST],p),training_n=len(TRAIN),test_n=len(TEST),delta_vs_RF4=float(r2_score(Y[TEST],p)-.5538115677138247))
(ROOT/'ensemble_test_result.json').write_text(json.dumps(result,indent=2))
pd.DataFrame(dict(excel_row=Z['excel_rows'][TEST],observed_ln1p=Y[TEST],prediction_ln1p=p)).to_csv(ROOT/'ensemble_test_predictions.csv',index=False)
pd.DataFrame(dict(excel_row=Z['excel_rows'][TRAIN],observed_ln1p=Y[TRAIN],RF=rf_oof,GP=gp_oof)).to_csv(ROOT/'ensemble_training_oof.csv',index=False)
print(json.dumps(result),flush=True)
