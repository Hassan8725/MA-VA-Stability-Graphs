"""
@author: Hassan

Variable Stability Functions
"""
import numpy as np
import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")


def attach_normalized_aps(df,agentid='agentid', nf='nf', uid='idntuniqueappel_vaca',
                          optimization_metric='multinomial_metric'):
    
    ag_cp=df.groupby([agentid,nf]).agg({uid: 'count',
                                       optimization_metric: 'mean'})
    ag_cp.rename(columns={uid:'Calls',optimization_metric:'CR'}, inplace=True)
    
    nf_cp=df.groupby([nf]).agg({uid: 'count',
                                optimization_metric: 'mean'})
    nf_cp.rename(columns={uid:'Nf_calls',optimization_metric:'Nf_CR'}, inplace=True)
    nf_cp=nf_cp[nf_cp['Nf_calls']>50]
    ag_cp.reset_index(drop=False,inplace=True)
    nf_cp.reset_index(drop=False,inplace=True)
    merged=ag_cp.merge(nf_cp,how='left',on='nf')
    merged['W_CR']=(merged['CR']/merged['Nf_CR'])*merged['Calls']
    
    agent_norm_crs=merged.groupby(agentid).agg({'Calls':sum,
                                                'W_CR':sum})
    agent_norm_crs.rename(columns={'Calls':'Sum_calls','W_CR':'Sum_W_CR'},inplace=True)
    agent_norm_crs['Norm_CR']=agent_norm_crs['Sum_W_CR']*1.0/agent_norm_crs['Sum_calls']
    
    #agent_norm_crs['AP_norm']=(agent_norm_crs['Norm_CR']-min(agent_norm_crs['Norm_CR'])+0.00001)/(max(agent_norm_crs['Norm_CR'])+0.00001-min(agent_norm_crs['Norm_CR'])+0.00001)
    agent_norm_crs['AP_norm']=agent_norm_crs['Norm_CR'].rank(pct=True)
    
    aps=agent_norm_crs[['AP_norm']].reset_index(drop=False)
    df_1=df.merge(aps, how='left', on='agentid')
    return df_1.copy()

def get_va_grouped_data(df,variable,gb_wise, ap_col='AP_norm', optimization_metric='multinomial_metric'):
    Var_df_th_bh=df[df[ap_col]<0.5].groupby([gb_wise,variable]).agg({optimization_metric: 'mean'
                                                                       }).rename(
        columns={optimization_metric: 'BH'}).merge(
        df[df[ap_col]>=0.5].groupby([gb_wise,variable]).agg(
            {optimization_metric: 'mean'}).rename(columns={optimization_metric: 'TH'}),
        how='left',on=[gb_wise,variable])
    
    Var_df_th_bh['VA_delta']=Var_df_th_bh['TH']-Var_df_th_bh['BH']
    
    Var_df_th_bh.reset_index(drop=False,inplace=True)
    
    return Var_df_th_bh.copy()

## Function for grouped data
def get_grouped_data(df,variable,gb_wise,uid,binary_metric,cont_metric,multinomial_metric):
    '''This function is used for getting grouped data
    gb_wise: Week Wise, Month Wise or Day wise
    Variable: Any categorical column
    '''
    Var_data = df.groupby([gb_wise,variable]).agg({'dt': 'min',
                                               uid: 'count',
                                               binary_metric: 'mean',
                                               multinomial_metric: 'mean',
                                               cont_metric: 'mean'})
    Var_data.rename(columns = {'dt': 'Min_Date',
                               uid: 'Calls',
                               binary_metric: 'CR',
                               multinomial_metric: 'CR_multinomial',
                               cont_metric: 'RPC'},
                    inplace=True)
    Var_data.reset_index(inplace=True)
    Var_data.sort_values(by='Min_Date',inplace=True, ascending=True)

    return Var_data


def get_stability_graph(df,variable,gb_wise,req_stats):
    '''This function will plot the line graph for the categorical column against gb_wise
    gb_wise: Week Wise, Month Wise or Day wise
    reg_stats: CR,CR_multinomial, RPC, ect 
    '''
    var_set=set(df[variable])
    fig = plt.figure()
    for cat in var_set:
        selected_data= df[df[variable]==cat]
        plt.plot(selected_data[gb_wise],selected_data[req_stats], label=cat)
    plt.legend()
    plt.title(variable)
    plt.xlabel(gb_wise)
    plt.xticks(rotation=90)
    plt.ylabel(req_stats)
    #plt.show()
    return fig

####
