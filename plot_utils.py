import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm,chi2

def to_long_format(df):
    var = [s for s in df.columns if s[:3]=='var']
    no_var = [s for s in df.columns if s[:3]!='var']
    df1 = df[no_var]
    df2 = df1.melt(['n_sample','d','link','clf','index','joint regression'],var_name='estimator',value_name='estimate')
    df2['error'] = df2['estimate'].to_numpy()- df1['IM'].iloc[df2['index']].to_numpy()
    df2['error (scaled)'] = np.sqrt(df2['n_sample']) * np.abs(df2['error'])
    df2 = df2[df2['estimator']!='IM']
    df2['errorsq'] = df2['error']**2
    df2['var_est'] = df[var+['index']].melt(['index'],var_name='estimator',value_name='var_estimate')['var_estimate'].to_numpy()
    df2['se_est'] = np.sqrt(df2['var_est'])
    return df2

def compute_wald(df,alpha=0.05):
    df['cover'] = norm.ppf(1-0.5*alpha)*df['se_est'] > np.sqrt(df['n_sample'])*np.abs(df['error']) #nominal coverage is 1-alpha
    df['width'] = 2*norm.ppf(1-0.5*alpha)*df['se_est']/np.sqrt(df['n_sample'])
    df['zvalue'] = np.sqrt(df['n_sample'])*df['error']/df['se_est']
    df['pvalue'] = 1- chi2.cdf(df['zvalue']**2,df=1)
    return df

params = ['n_sample','estimator','d','link','clf','joint regression']

def average_over_settings(df,params=params):
    grp_df = df.groupby(params,as_index=False)['errorsq'].mean()
    grp_df['var'] = df.groupby(params,as_index=False)['error'].var()['error']
    grp_df['bias'] = df.groupby(params,as_index=False)['error'].mean()['error']
    grp_df['coverage'] = df.groupby(params,as_index=False)['cover'].mean()['cover']
    grp_df['width_avg'] = df.groupby(params,as_index=False)['width'].mean()['width']
    return grp_df


def rename(df):
    df['log(n)'] = df['n_sample'].apply(lambda x: np.emath.logn(3,x//300))
    df['scaled RMSE'] = np.sqrt(df['n_sample']) * np.sqrt(df['errorsq'])
    df['OR'] = df['estimator'].apply(lambda x: 'Neural net' if x[:2]=='SI' else 'Oracle' if x[-2:]=='_O' else 'OLS')
    df['Crossfit'] = df['estimator'].apply(lambda x: True if x[-2:]=='cf' else False)
    df['Estimator'] = df['estimator'].apply(lambda x: x[:-3] if x[-2:]=='cf' else x)
    df['Estimator'] = df['Estimator'].apply(lambda x: x[3:] if x[:2]=='SI' else x)
    df['Estimator'] = df['Estimator'].apply(lambda x: 'Regression' if x in ['R','Regr','Regr_O'] else x)
    df['Estimator'] = df['Estimator'].apply(lambda x: 'AIPW' if x=='AIPW_O' else x)
    df['Estimator'] = df['Estimator'].apply(lambda x: 'DOPE-IDX' if x in ['OAPW','OAPW_O'] else x)
    df['Estimator'] = df['Estimator'].apply(lambda x: 'DOPE-BCL' if x in ['OBPW','OBPW_O'] else x)
    #df['link'] = df['link'].apply(lambda x: 'cube root' if x=='cbrt' else x)
    df['link'] = df['link'].apply(lambda x: 'square' if x=='sqr' else x)
    df[''] = np.ones_like(df['n_sample'])
    return df



def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot_table(index=args[1],columns=args[0],values=args[2])
    sns.heatmap(d,annot=True,cbar=True,fmt='.2f',**kwargs)


def draw_heatmap_joint(*args, **kwargs):
    data = kwargs.pop('data')
    d1 = data.pivot_table(index=args[1],columns=args[0],values=args[2])
    ax = sns.heatmap(d1,annot=False,cbar=True,**kwargs)
    d2 = data.pivot_table(index=args[1],columns=args[0],values=args[3])
    d3 = d2.copy()
    for i in np.arange(4):
        for j in np.arange(3):
            d3.iloc[i,j] = f'{d1.iloc[i,j]:.2f} ({d2.iloc[i,j]:.2f})'
    ax = sns.heatmap(d1,ax=ax,annot=d3,fmt='',cbar=False,**kwargs)
    return ax
