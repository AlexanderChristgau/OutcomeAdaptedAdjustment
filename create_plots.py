import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc
from cmasher import get_sub_cmap
from tqdm.notebook import tqdm
from itertools import product

import plot_utils as pu


### plot settings
rc('font',**{'size':30})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath,times}')
params = {'font.family':'serif','font.size':13,'axes.labelsize':14,'axes.titlesize':15,'figure.figsize':(1,1)}
sns.set_theme(style='whitegrid',palette='colorblind',rc=params)
sns.set_context("paper", rc=params)   


# function that generates the rmse plots in the main paper
def rmse_plot(df,path=None,ymins=None,ymaxs=None,hue='Estimator',links = ['lin','square','cbrt','sin'],hue_order=['Regression','AIPW','DOPE-BCL','DOPE-IDX']):
    g = sns.FacetGrid(df,col='link',col_wrap=2,col_order=links,sharey=False,aspect=1.3,height=3)
    g.map(sns.lineplot,'log(n)','scaled RMSE','Estimator','',hue,**{'errorbar':'se','sizes':(2,1)},hue_order=hue_order)

    if '1' in g._legend_data: 
        g._legend_data.pop('1')
    g.add_legend()
    sns.move_legend(g, "upper right",bbox_to_anchor=(1.03, 0.65))

    #Legend texts
    for text in g.legend.texts:
        if text.get_text() in ['Estimator','clf','OR','joint regression','Crossfit']:
            text.set_fontsize(16)
        else:
            text.set_fontsize(12)
            if text.get_text() == 'OLS':
                text.set_text('Linear')
    if (ymins is not None) and (ymaxs is not None):
        for ax,ymin,ymax in zip(g.axes.flatten(),ymins,ymaxs):
            ax.set_ylim((ymin,ymax))

    g.set_xlabels(r'sample size $n$')
    g.set_ylabels(r'$\sqrt{n} \times\mathrm{RMSE}$')
    plt.xticks(ticks= np.arange(3), labels=[300*(3**l) for l in range(3)])

    if path is not None:
        plt.savefig(path,bbox_inches='tight')
    else:
        plt.show()


viridis_trim = get_sub_cmap('viridis', 0, 1)


def heatmap_plot(df,value='coverage',path=None):
    g = sns.FacetGrid(df,row='Estimator',col='Crossfit',sharey=False,aspect=1.3)
    cbar_ax = g.figure.add_axes([1.015,0.12, 0.03, 0.8])
    g.map_dataframe(pu.draw_heatmap,'n_sample','link',value,cbar_ax=cbar_ax,vmin=0.4, vmax=1,cmap=viridis_trim)
    g.set_xlabels(r'sample size $n$',**{'fontsize':16})
    g.set_ylabels('link',**{'fontsize':16})
    g.set_xticklabels(labels=df.n_sample.unique(),step=np.arange(1,4)-0.5,**{'fontsize':12})
    g.set_yticklabels(**{'fontsize':12})

    if path is not None:
        plt.savefig(path,bbox_inches='tight')
    else:
        plt.show()



if __name__=='__main__':
    print('loading and formating data...')
    # Data was collected over three different runs. For the first run, the prompt was
    # > python run_simulations.py --file_save_path simdata/full_simulation_seed1.pkl --seed 1
    # Similarly for seed 2 and seed 3
    dfs1 = pd.read_pickle('simdata/full_simulation_seed1.pkl')
    dfs2 = pd.read_pickle('simdata/full_simulation_seed2.pkl')
    dfs3 = pd.read_pickle('simdata/full_simulation_seed3.pkl')
    df = pd.concat([dfs1,dfs2,dfs3])
    
    df = df.reset_index(drop=True).reset_index()
    df = df.drop(['Unadjusted','IPW','var_Unadjusted','var_IPW'],axis=1)
    if 'IPW_cf' in df.columns:
        df = df.drop(['IPW_cf','var_IPW_cf'],axis=1)
    df2 = pu.to_long_format(df)
    df2 = pu.compute_wald(df2)
    grp_df = pu.average_over_settings(df2)
    grp_df = pu.rename(grp_df)
    df2 = pu.rename(df2)

    ## Create rmse line plots
    print('creating RMSE plots...')
    df_nocf = df2[df2['Crossfit']==False].copy()
    
    rmse_plot(df_nocf[df_nocf['joint regression']==False],
              path='paperplots/RMSE_lineplot_sepa_nocf.pdf',
              ymins=[2,2,2.5,2.5],ymaxs=[5.5,5.5,8,8])
    
    rmse_plot(df_nocf[df_nocf['joint regression']==True],
              path='paperplots/RMSE_lineplot_joint_nocf.pdf',
              ymins=[1.5,1.5,2.5,2.5],ymaxs=[5.5,5.5,8,8])
 
    ## Create heatmap plot
    print('creating heatmap coverage plot...')
    df_heatmap = grp_df[(grp_df['joint regression']==True)&(grp_df['Estimator']!='Regression')]
    heatmap_plot(df_heatmap,path='paperplots/Coverage.pdf')


    ## crossfitting comparison
    print('creating cross-fitting comparison plot...')
    df_cfcomparison = df2[(df2['Estimator']!='Regression')&(df2['Estimator']!='AIPW')&(df2['joint regression']==False)].copy()
    df_cfcomparison['Estimator'] = df_cfcomparison['estimator']
    df_cfcomparison['Estimator'] = df_cfcomparison['Estimator'].apply(lambda x: 'DOPE-IDX' if x in ['SI_OAPW','SI_OAPW_cf'] else x)
    df_cfcomparison['Estimator'] = df_cfcomparison['Estimator'].apply(lambda x: 'DOPE-BCL-lin' if x in ['OBPW','OBPW_cf'] else x)
    df_cfcomparison['Estimator'] = df_cfcomparison['Estimator'].apply(lambda x: 'DOPE-BCL-nn' if x in ['SI_OBPW','SI_OBPW_cf'] else x)

    rmse_plot(df_cfcomparison,'paperplots/RMSE_lineplot_cf.pdf',hue='Crossfit',hue_order=['DOPE-BCL-lin','DOPE-BCL-nn','DOPE-IDX'])