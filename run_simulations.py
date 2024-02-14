import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
import argparse

import estimators as es
import samplescheme as ss


## Run a single setting
def run_single_setting(n_sample=500,reps=200,d=4,link='sin',pbar=None,classifier='Logistic', regressor='OLS',crossfit=True,joint_regression=True,rng=None):
    IM_con, IM_reg, IM_ipw, IM_aipw, IM_pru = ([] for _ in range(5))
    IM_sir, IM_siaipw, IM_sioapw, IM_siobpw = ([] for _ in range(4))
    IM_true = []

    var_con, var_reg, var_ipw, var_aipw, var_pru = ([] for _ in range(5))
    var_sir, var_siaipw, var_sioapw, var_siobpw = ([] for _ in range(4))

    if crossfit:
        IM_reg_cf, IM_ipw_cf, IM_aipw_cf, IM_pru_cf = ([] for _ in range(4))
        IM_sir_cf, IM_siaipw_cf, IM_sioapw_cf, IM_siobpw_cf = ([] for _ in range(4))
        var_reg_cf, var_ipw_cf, var_aipw_cf, var_pru_cf = ([] for _ in range(4))
        var_sir_cf, var_siaipw_cf, var_sioapw_cf, var_siobpw_cf = ([] for _ in range(4))

    if rng is None:
        rng = np.random.default_rng()
    
    for i in range(reps):
        idxY = ss.sample_idx(rng,d)
        IM_true.append(np.mean([ss.compute_IM(rng,indexY=idxY,link=link,N=10000) for _ in range(100)])) #Compute montecarlo estimate of groundtruth IM based 100000 samples
        W,T,Y = ss.sample_SI(rng,indexY=idxY,n=n_sample,link=link,Ystd=1)
        idx_approx = idxY + rng.uniform(-0.1,0.1,size=d)

        IM_con.append(Y[T==1].mean())
        var_con.append(Y[T==1].var())
        
        # Classical estimators based on linear methods
        x,y = es.IM_est(T,W,Y,'Logistic',regressor,joint_regression=joint_regression)
        IM_ipw.append(x[0]); var_ipw.append(y[0])
        IM_reg.append(x[1]); var_reg.append(y[1])
        IM_aipw.append(x[2]); var_aipw.append(y[2])
        IM_pru.append(x[3]); var_pru.append(y[3])

        # Estimators based on SI network
        x,y = es.SI_IM(T,W,Y,clf=classifier,joint_regression=joint_regression,initial_idx=idx_approx)
        IM_sir.append(x[0]); var_sir.append(y[0])
        IM_siaipw.append(x[1]); var_siaipw.append(y[1])
        IM_sioapw.append(x[2]); var_sioapw.append(y[2])
        IM_siobpw.append(x[3]); var_siobpw.append(y[3])

        if crossfit:
            # Classical and crossfit
            x,y = es.IM_est_cf(T,W,Y,'Logistic',regressor,joint_regression=joint_regression)
            IM_ipw_cf.append(x[0]); var_ipw_cf.append(y[0])
            IM_reg_cf.append(x[1]); var_reg_cf.append(y[1])
            IM_aipw_cf.append(x[2]); var_aipw_cf.append(y[2])
            IM_pru_cf.append(x[3]); var_pru_cf.append(y[3])

            # SI network and crossfit
            x,y = es.SI_IM_cf(T,W,Y,clf=classifier,joint_regression=joint_regression,initial_idx=idx_approx)
            IM_sir_cf.append(x[0]); var_sir_cf.append(y[0])
            IM_siaipw_cf.append(x[1]); var_siaipw_cf.append(y[1])
            IM_sioapw_cf.append(x[2]); var_sioapw_cf.append(y[2])
            IM_siobpw_cf.append(x[3]); var_siobpw_cf.append(y[3])

        if i<reps-1 and pbar:
            pbar.update(1)
    
    df = pd.DataFrame({
        'n_sample': [n_sample]*reps,
        'd': [d]*reps,
        'link': [link]*reps,
        'clf': [classifier]*reps,
        'joint regression':[joint_regression]*reps,
        'IM':IM_true,
        'Unadjusted':IM_con,
        'Regr':IM_reg,
        'IPW':IM_ipw,
        'AIPW':IM_aipw,
        'OBPW': IM_pru,
        'SI_R': IM_sir,
        'SI_AIPW': IM_siaipw,
        'SI_OAPW': IM_sioapw,
        'SI_OBPW': IM_siobpw,
        'var_Unadjusted':var_con,
        'var_Regr':var_reg,
        'var_IPW':var_ipw,
        'var_AIPW':var_aipw,
        'var_OBPW': var_pru,
        'var_SI_R': var_sir,
        'var_SI_AIPW': var_siaipw,
        'var_SI_OAPW': var_sioapw,
        'var_SI_OBPW': var_siobpw
    })
    if crossfit:
        df_cf = pd.DataFrame({
            'Regr_cf':IM_reg_cf,
            'IPW_cf':IM_ipw_cf,
            'AIPW_cf':IM_aipw_cf,
            'OBPW_cf': IM_pru_cf,
            'SI_R_cf': IM_sir_cf,
            'SI_AIPW_cf': IM_siaipw_cf,
            'SI_OAPW_cf': IM_sioapw_cf,
            'SI_OBPW_cf': IM_siobpw_cf,
            'var_Regr_cf':var_reg_cf,
            'var_IPW_cf':var_ipw_cf,
            'var_AIPW_cf':var_aipw_cf,
            'var_OBPW_cf': var_pru_cf,
            'var_SI_R_cf': var_sir_cf,
            'var_SI_AIPW_cf': var_siaipw_cf,
            'var_SI_OAPW_cf': var_sioapw_cf,
            'var_SI_OBPW_cf': var_siobpw_cf,
        })
        df = pd.concat([df, df_cf], axis=1)

    return df


## Run several settings
def run_experiment(
        n_samples,d_list,links,clfs,joint_regression,
        n_reps=100,crossfit=False,seed=42,tempsave=False
    ):
    df_list = []
    settings = product(*[n_samples,d_list,links,clfs,joint_regression])
    len_settings = n_reps*len(n_samples)*len(d_list)*len(links)*len(clfs)*len(joint_regression)
    
    print('Total number of settings:', len_settings//n_reps)
    pbar = tqdm(settings,total=len_settings)
    rng = np.random.default_rng(seed=seed)

    for n_sample,d,link,clf,jr in pbar:
        df_list.append(
            run_single_setting(
                n_sample=n_sample,
                reps=n_reps,
                d=d,
                link=link,
                classifier=clf,
                pbar=pbar,
                crossfit=crossfit,
                joint_regression=jr,
                rng=rng
            )
        )
        if tempsave:
            pd.concat(df_list).to_pickle('temp-save/'+str(seed)+'.pkl')
    else:
        pbar.close()
        
    df = pd.concat(df_list)
    
    return df


#run main experiment
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    filename = 'simdata/full_simulation.pkl'
    parser.add_argument('--file_save_path',default=filename,type=str)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--sample_sizes", nargs="+", default=[2700,900,300])
    parser.add_argument("--dims", nargs="+", default=[4,12,36])
    parser.add_argument("--links", nargs="+", default=['lin','sqr','cbrt','sin'])
    parser.add_argument("--clfs", nargs="+", default=['Logistic'])
    parser.add_argument("--joint", nargs="+", default=[False,True])
    parser.add_argument("--crossfit", nargs="+", default=False)
    parser.add_argument("--seed", nargs="+", type=int, default=42)

    settings = parser.parse_args()

    df = run_experiment(
        n_samples= settings.sample_sizes,
        d_list= settings.dims,
        links= settings.links,
        clfs=settings.clfs,
        joint_regression=settings.joint,
        n_reps = settings.repetitions,
        crossfit=settings.crossfit,
        seed=settings.seed
    )
    df.to_pickle(settings.file_save_path)
