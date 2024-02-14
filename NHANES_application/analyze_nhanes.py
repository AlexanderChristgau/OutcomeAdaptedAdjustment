import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from tqdm import tqdm

from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss

import sys; sys.path.append(".."); import estimators as es


### plot settings
from matplotlib import rc
rc('font',**{'size':30})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath,times}')
params = {'font.family':'serif','font.size':13,'axes.labelsize':14,'axes.titlesize':15}
sns.set_theme(style='whitegrid',palette='pastel',rc=params)


### auxiliary functions
hstack = lambda t,w: np.hstack([t.reshape(-1,1),w])
stack0 = lambda w: np.hstack([np.zeros((w.shape[0],1)),w])
stack1 = lambda w: np.hstack([np.ones((w.shape[0],1)),w])

def fitTmodel(name,W,T):
    return es.create_classifier(name).fit(W,T)


cv_score = lambda mod,X,y: np.round(cross_val_score(mod,X,y,scoring='neg_log_loss',cv=5).mean(),3)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_imputed',action=argparse.BooleanOptionalAction)
    parser.add_argument('--tune_hyperparameters',action=argparse.BooleanOptionalAction)
    settings = parser.parse_args()
    use_imputed = settings.use_imputed

    ### Load data
    if use_imputed:
        O = pd.read_pickle('NHANES_imputed.pkl')
    else:
        O = pd.read_pickle('NHANES_removed.pkl')

    On = O.to_numpy()
    T,W,Y = On[:,0],On[:,1:-1],On[:,-1]

    ### First find default initial index (for faster training in CV and bootstrapping)
    clf = es.IndexClassifier()
    if settings.tune_hyperparameters:
        idxs = []
        losses = []
        print('refitting neural net to find default initialization if on single-index')
        for _ in tqdm(range(5)):
            clf.fit(hstack(T,W),Y)
            y_pred = clf.predict_proba(hstack(T,W))
            losses.append(log_loss(Y,y_pred))
            idxs.append(clf.index.weight.detach().numpy().flatten())
        idx_init = idxs[np.argmin(losses)]
        print('random init losses:',losses)
        if use_imputed:
            np.save('results/best-idx_imputed',idx_init)
        else:
            np.save('results/best-idx_removed',idx_init)
    else:
        if use_imputed:
            idx_init = np.load('results/best-idx_imputed.npy')
        else:
            idx_init = np.load('results/best-idx_removed.npy')


    ### CV score hyperparameters for index model
    if settings.tune_hyperparameters:
        print('Finetuning single index model')
        grid = {
            'hidden_dim': [100,200],
            'n_iter': [1500,3000,5000],
            'l2pen':[0,1e-3,1e-4],
            'lr': [1e-1,1e-2,1e-3],
        }
        estimator_CV = RandomizedSearchCV(estimator=es.IndexClassifier(initial_idx=idx_init),param_distributions=grid,n_iter=6,cv=5,verbose=3,random_state=42,n_jobs=-1,scoring='neg_log_loss')
        estimator_CV.fit(hstack(T,W),Y)
        best_params = estimator_CV.best_params_
        print(best_params)
    else:
        best_params = {'n_iter': 1500, 'lr': 0.001, 'l2pen': 0.001, 'hidden_dim': 100}
        # The random CV may not always select these parameters, as it depends on rng and whether or not the completely imputed dataset is used.
        # However, this setting will achieve very similar loss and be much faster for training

    ### Check model fits and generate plot of propensity scores
    clf = es.IndexClassifier(**best_params)
    clf.fit(hstack(T,W),Y,initial=idx_init)
    Z = clf.partial_predict(W).reshape(-1,1)
    Q = clf.predict_conditional(1,W).reshape(-1,1)

    clfW = fitTmodel('Logistic',W,T)
    clfZ = fitTmodel('Logistic',Z,T)
    clfQ = fitTmodel('Logistic',Q,T)

    pW = clfW.predict_proba(W)[:,1]
    pZ = clfZ.predict_proba(Z)[:,1]
    pQ = clfQ.predict_proba(Q)[:,1]
    props = pd.DataFrame({
        r'$\widehat{P}(T_i=1 \mid \mathbf{W}_i)$':pW,
        r'$\widehat{P}(T_i=1 \mid \mathbf{W}_i^\top\hat{\theta}\,)$':pZ,
        r'$\widehat{P}(T_i=1 \mid \widehat{g}(1,\mathbf{W}_i))$':pQ,
    })
    sns.histplot(props,common_bins=False,binwidth=0.0125,element='step',**{'alpha':0.3,'linewidth':1.4})
    plt.xlabel('Propensity score estimate')
    if use_imputed:
        plt.savefig('../plots/propensities_imputed.pdf',bbox_inches='tight')
    else:
        plt.savefig('../plots/propensities_removed.pdf',bbox_inches='tight')


    ### Print scores for various regressions
    print('score T~Z MLP',cv_score(es.create_classifier('MLP'),Z,T))
    print('score T~Z Logist',cv_score(clfZ,Z,T))
    print('score T~W Logist',cv_score(es.create_classifier('Logistic'),W,T))
    print('score Y~T,W Logistic',cv_score(es.create_classifier('Logistic'),hstack(T,W),Y))
    print('score Y~T,W single-idx',cv_score(es.IndexClassifier(initial_idx=idx_init,**best_params),hstack(T,W),Y))


    ### Estimate interventional means and ATE
    im0s = []
    im1s = []
    Vars = []

    im0s.append(Y[T==0].mean())
    im1s.append(Y[T==1].mean())
    Vars.append(((2*T-1)*Y).var())

    # Classical estimators based on linear methods
    x,y,z = es.ATE_est_binY(T,W,Y,'Logistic','Logistic')
    im0s += list(x)
    im1s += list(y)
    Vars += list(z)

    # Estimators based on SI network
    x,y,z = es.SI_ATE_binY(T,W,Y,index_dim=1,clf='Logistic',initial=idx_init,
                        l2pen=best_params['l2pen'],n_iter=best_params['n_iter'],joint_training=True)
    im0s += list(x)
    im1s += list(y)
    Vars += list(z)

    ### Estimates for full data set
    IM0s = np.array(im0s)
    IM1s = np.array(im1s)
    ATEs = IM1s-IM0s
    Vars = np.array(Vars)


    ### compute bootstrap resamples
    ATE_raw, ATE_reg, ATE_ipw = ([] for _ in range(3))
    ATE_aipw, ATE_pru = ([] for _ in range(2))
    ATE_sir, ATE_siaipw, ATE_sioapw, ATE_siobpw, = ([] for _ in range(4))
    ATE_sir_cf, ATE_siw_cf, ATE_siz_cf = ([] for _ in range(3))
    reps = 100

    for ii in tqdm(range(reps)):
        T_,W_,Y_ = resample(T,W,Y,n_samples=len(T),random_state=ii)
        ATE_raw.append([es.unadjusted(T_,Y_,0),es.unadjusted(T_,Y_,1)])
        # Classical estimators based on linear methods
        x,y,_ = es.ATE_est_binY(T_,W_,Y_,'Logistic','Logistic')
        ATE_ipw.append([x[0],y[0]])
        ATE_reg.append([x[1],y[1]])
        ATE_aipw.append([x[2],y[2]])
        ATE_pru.append([x[3],y[3]])

        # Estimators based on SI network
        x,y,_ = es.SI_ATE_binY(T_,W_,Y_,clf='Logistic',initial=idx_init, **best_params)
        ATE_sir.append([x[0],y[0]])
        ATE_siaipw.append([x[1],y[1]])
        ATE_sioapw.append([x[2],y[2]])
        ATE_siobpw.append([x[3],y[3]])

    df = pd.DataFrame({
        'Naive contrast':ATE_raw,
        'IPW (Logistic)':ATE_ipw,
        'Regr. (Logistic)':ATE_reg,
        'AIPW (Logistic)':ATE_aipw,
        'DOPE-BCL (Logistic)': ATE_pru,
        'Regr. (NN)': ATE_sir,
        'AIPW (NN)': ATE_siaipw,
        'DOPE-IDX (NN)': ATE_sioapw,
        'DOPE-BCL (NN)': ATE_siobpw,
    })
    dfm = df.melt(var_name='estimator',value_name='intmeans')
    dfm['chi0'] = dfm['intmeans'].apply(lambda x: x[0])
    dfm['chi1'] = dfm['intmeans'].apply(lambda x: x[1])
    dfm['ATE'] = dfm['intmeans'].apply(lambda x: x[1]-x[0])

    # dfm.to_pickle('results/bootstrap_estimates.pkl')

    df_grp = pd.DataFrame(dfm.groupby('estimator')['ATE'].var().rename('BS var')*len(T))
    df_grp['BS lower'] = dfm.groupby('estimator')['ATE'].quantile(0.025)
    df_grp['BS upper'] = dfm.groupby('estimator')['ATE'].quantile(0.975)

    rename = {
    'AIPW':'AIPW (Logistic)', 
    'DOPE-BCL':'DOPE-BCL (Logistic)',
    'IPW':'IPW (Logistic)',
    'Regr':'Regr. (Logistic)',
    'SI_AIPW':'AIPW (NN)',
    'SI_DOPE-BCL':'DOPE-BCL (NN)',
    'SI_DOPE-IDX':'DOPE-IDX (NN)',
    'SI_R':'Regr. (NN)',
    'Unadjusted':'Naive contrast'
    }
    ATEs_pd = pd.Series(ATEs,index=rename.values())
    Vars_pd = pd.Series(Vars,index=rename.values())
    df_grp.insert(0,'estimate',ATEs_pd.reindex_like(df_grp))

    if use_imputed:
        df_grp.to_pickle('results/nhanes_ate_imputed.pkl')
    else:
        df_grp.to_pickle('results/nhanes_ate_removed.pkl')

    df_grp['BS CI'] =  list(zip(df_grp['BS lower'].round(3),df_grp['BS upper'].round(3)))
    df_table = df_grp.drop(['BS lower','BS upper'],axis=1)
    
    if use_imputed:
        df_table.sort_values(by='BS var').round(3).to_latex('results/nhanes_table_imputed')
    else:
        df_table.sort_values(by='BS var').round(3).to_latex('results/nhanes_table_removed')