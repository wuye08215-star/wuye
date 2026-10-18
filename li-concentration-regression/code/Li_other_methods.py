"""Li浓度回归：XGBoost、GaussianProcess、样条加性回归、RF+GP。

参数与四因子输入均已按原QSH训练内验证确定；不新增筛除样本。
本轮固定测试结果均未超过四因子RF，作为可运行的对照试验。
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from RF_modified import clean_data_final

CONFIGS = {'XGBoost': {'subsample': 1.0,
             'reg_lambda': 1.0,
             'reg_alpha': 0.0,
             'n_estimators': 1000,
             'min_child_weight': 10,
             'max_depth': 3,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7},
 'GaussianProcess': {'kernel': 'Matern15', 'length_scale': 0.3, 'noise': 0.5},
 'SplineRidge': {'n_knots': 20, 'degree': 2, 'alpha': 10.0}}
RF_WEIGHT = 0.30000000000000004
FACTORS=['AET','AI','Alpha(AET/PET)','BIO7']

def make_model(method):
    if method=='xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(objective='reg:squarederror',tree_method='hist',device='cpu',random_state=42,n_jobs=1,**CONFIGS['XGBoost'])
    if method=='gpr':
        p=CONFIGS['GaussianProcess']
        kernel=RBF(p['length_scale']) if p['kernel']=='RBF' else Matern(p['length_scale'],nu=1.5)
        return make_pipeline(StandardScaler(),GaussianProcessRegressor(
            kernel=kernel+WhiteKernel(p['noise']),normalize_y=True,optimizer=None,alpha=1e-8))
    if method=='spline':
        p=CONFIGS['SplineRidge']
        return make_pipeline(SplineTransformer(n_knots=p['n_knots'],degree=p['degree'],knots='quantile',include_bias=False),
            StandardScaler(),Ridge(alpha=p['alpha']))
    rf=RandomForestRegressor(n_estimators=500,max_depth=8,min_samples_leaf=1,
        max_features=.3,ccp_alpha=.001,random_state=42,n_jobs=1)
    return VotingRegressor([('rf',rf),('gp',make_model('gpr'))],weights=[RF_WEIGHT,1-RF_WEIGHT],n_jobs=1)

def nonnegative_r2(estimator,x,y):return r2_score(y,np.maximum(estimator.predict(x),0))

def run(data_path,method,output):
    output.mkdir(parents=True,exist_ok=True)
    df=pd.read_excel(data_path)
    # 保持原代码的清洗范围与Li<=2000筛选，防止改变比较样本。
    all_features=[c for c in df.columns if c not in ['Li','ID','ID_1','E','W','1/0']]
    x,y,before,after=clean_data_final(df,all_features,'Li')
    mask=y<=2000;x=x.loc[mask];y=np.log1p(y.loc[mask])
    arrays=train_test_split(x[FACTORS].to_numpy(float),y.to_numpy(float),x.index.to_numpy()+2,test_size=.2,random_state=42)
    xtrain,xtest,ytrain,ytest,rows_train,rows_test=arrays
    xtrain=np.ascontiguousarray(xtrain);xtest=np.ascontiguousarray(xtest)
    model=make_model(method);model.fit(xtrain,ytrain)
    raw=model.predict(xtest);prediction=np.maximum(raw,0)
    metrics=dict(Model=method,R2=r2_score(ytest,prediction),RMSE=float(np.sqrt(mean_squared_error(ytest,prediction))),
        MAE=mean_absolute_error(ytest,prediction),NSE=r2_score(ytest,prediction),
        N_train=len(ytrain),N_test=len(ytest),N_features=len(FACTORS),N_negative_raw=int((raw<0).sum()))
    pd.DataFrame([metrics]).to_excel(output/'metrics.xlsx',index=False)
    pd.DataFrame(dict(Excel_row=rows_test,Observed_Li=np.expm1(ytest),Observed_ln1p=ytest,
        Raw_prediction_ln1p=raw,Prediction_ln1p=prediction,Prediction_Li=np.expm1(prediction))).to_excel(output/'predictions.xlsx',index=False)
    importance=permutation_importance(model,xtest,ytest,n_repeats=20,random_state=42,scoring=nonnegative_r2,n_jobs=1)
    imp=pd.DataFrame(dict(Factor=FACTORS,Importance_R2_drop=importance.importances_mean,Shuffle_STD=importance.importances_std)).sort_values('Importance_R2_drop',ascending=False)
    imp.to_excel(output/'feature_importance.xlsx',index=False)
    fig,ax=plt.subplots(figsize=(5,5));ax.scatter(ytest,prediction,s=15,alpha=.6)
    limits=[min(ytest.min(),prediction.min()),max(ytest.max(),prediction.max())];ax.plot(limits,limits,'r--')
    ax.set(xlabel='Observed ln(1+Li)',ylabel='Predicted ln(1+Li)',title=f'{method}: R2={metrics["R2"]:.4f}')
    fig.tight_layout();fig.savefig(output/'fit.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3));view=imp.iloc[::-1]
    ax.barh(view.Factor,view.Importance_R2_drop,xerr=view.Shuffle_STD,capsize=3)
    ax.set(xlabel='Decrease in test R2 after shuffling',title=method+' predictive importance')
    fig.tight_layout();fig.savefig(output/'importance.png',dpi=200);plt.close(fig)
    print(pd.DataFrame([metrics]).to_string(index=False));print('结果目录：',output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='三种新方法及RF/GP组合的Li浓度回归对照')
    parser.add_argument('data',nargs='?',default='QSH.xlsx')
    parser.add_argument('--method',choices=['xgboost','gpr','spline','rf_gp'],default='rf_gp')
    parser.add_argument('--output',default=None)
    args=parser.parse_args();run(Path(args.data).resolve(),args.method,Path(args.output or (args.method+'_results')).resolve())
