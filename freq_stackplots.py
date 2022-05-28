"""
A tool for plotting SSC frequency across months, keeping here to help declutter other code

Can also also plot seasonal differences between models (or obsv) - freq or temp
    as in SSC_compareTabular (copying much of this from that)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
import scipy
from obsv_new import obsv_new
from wrf_new import wrf_new

fig_outpath_wrf = '/Users/birke111/Documents/ssc/SSC/Model comparisons/new'
fig_outpath_obsv = '/Users/birke111/Documents/ssc/freq_stackplots'

loc_dict = {'FAR':'Fargo', 'DLH':'Duluth', 
            'MSP':'Minneapolis', 'RST':'Rochester'} # ordered to loosely match geography

#loc_dict = {'MSP':'Minneapolis', 'DLH':'Duluth'}
model_dict = {'BC':'bcc-csm1-1', 'CC':'CCSM4', 'CM':'CMCC-CM', \
              'CN':'CNRM-CM5', 'GF':'GFDL-ESM2M', 'IP':'IPSL-CM5A-LR', \
              'MI':'MIROC5',  'MR':'MRI-CGCM3'}
model_dict2 = {'BC':'bcc-csm1-1', 'CC':'CCSM4', 'CM':'CMCC-CM', \
              'CN':'CNRM-CM5', 'GF':'GFDL-ESM2M', 'IP':'IPSL-CM5A-LR', \
              'MI':'MIROC5', 'MR':'MRI-CGCM3', 'ens':'Multi-model ensemble'}
    
# for inputs 
#year_dict = {'HIST':[1980,1999], 'MID':[2040,2059], 'END4.5':[2060,2079], 'END8.5':[2080,2099]}
year_dict = {'HIST':[1980,1999], 'MID':[1980,1999], 'END4.5':[1980,1999], 'END8.5':[1980,1999]}

scenario_namedict = {'HIST': 'Historical runs, 1980-1999',
                     'MID': 'RCP4.5, 2040-2059',
                     'END4.5': 'RCP4.5, 2080-2099',
                     'END8.5': 'RCP8.5, 2080-2099'}
hist_namedict = {'obsv': 'Station observations, 1948-2019',
                 'ens': 'Multi-model ensemble, 1980-1999'}

#SSClist = ['DM', 'DP', 'DT', 'MM', 'MP', 'MT', 'TR', 'DP+', 'DT+', 'MP+', 'MT+', 'TR+']
#SSClist_main = ['DP', 'DM', 'DT', 'MP', 'MM', 'MT', 'MT+']
SSClist_full =  ['DP', 'DM', 'DT', 'DT+', 'MP', 'MM', 'MT', 'MT+']
#SSClist_plus2 = ['DP', 'DM', 'DT', 'DT+', 'DT++', 'MP', 'MM', 'MT', 'MT+', 'MT++']

SSClist_obsv = ['DP', 'DM', 'DT', 'MP', 'MM', 'MT', 'MT+', 'TR']
#SSClist_full = ['DP', 'DM', 'DT', 'DT+', 'DT++', 'MP', 'MM', 'MT', 'MT+', 'MT++']
sscColors = {'DP':'wheat', 'DM':'sandybrown', 
             'DT':'orangered', 'DT+':'firebrick', 'DT++':'darkred',
             'MP':'paleturquoise', 'MM':'mediumturquoise', 
             'MT':'mediumseagreen', 'MT+':'green', 'MT++':'darkgreen',
             'TR':'grey'}


month_list = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']

var_dict = {'Freq':'Frequency (%)', 
            'AM_temp':'AM temp (F)', 
            'AM_dewpt':'AM dew point (F)', 
            'AM_slp':'AM sea level pressure (mb)',
            'AM_cc':'AM cloud cover (10ths)',
            'AM_wspd':'AM wind speed (mph)',
            'AM_apptemp':'AM apparent temp (F)',
            'PM_temp':'PM temp (F)', 
            'PM_dewpt':'PM dew point (F)', 
            'PM_slp':'PM sea level pressure (mb)',
            'PM_cc':'PM cloud cover (10ths)',
            'PM_wspd':'PM wind speed (mph)',
            'PM_apptemp':'PM apparent temp (F)'}


def TypeAgg(code, MTplus=1, DTplus=0): # number of "pluses" to leave unaggregated
    ssctype = int(code)    
    if ssctype in range(61,70):
        if ssctype > 60 + MTplus:
            rounddown = 60 + MTplus
        else:
            rounddown = ssctype
    elif ssctype in range(31,40):
        if ssctype > 30 + DTplus:
            rounddown = 30 + DTplus
        else:
            rounddown = ssctype        
    else:
        rounddown = int(ssctype/10) * 10
    return rounddown 


def ssc_decode(ssctype):
    typeagg = TypeAgg(ssctype, 0, 0)
    code_dict = {0:'', 10:'DM', 20:'DP', 30:'DT', 40:'MM', 50:'MP', 60:'MT', 70:'TR'}
    
    if ssctype == 99:
        return 'All'
    else:
        decode = code_dict[typeagg]
        pluses = ssctype-typeagg
        for p in range(pluses):
            decode = decode + '+'
        return decode

def Input(loc='MSP', model='obsv', scenario='HIST'):
    if model == 'obsv':
        df = obsv_new.SSCjoin('', loc, 'daily') 
        # could try hourly later, it's just slower, and doesn't affect SSC freq patterns
        
        df.SSC.fillna(0, inplace=True)
        
    else:
        df = wrf_new.SSCjoin(loc, scenario, model)   
    return df

def FreqPivot(loc='MSP', model='obsv', scenario='HIST', MTplus=2, DTplus=1):
    df = Input(loc, model, scenario)
    
    # aggregating non-MT/DT plus types by default - less interested in winter DP+ etc
    df['SSC'] = [TypeAgg(a, MTplus, DTplus) for a in df.SSC]
           
    # if model == 'obsv':
    #     df['Month'] = df.DateTime.dt.month
    # else:
    df['Month'] = df.DateTime.dt.month
    
    pivot = df.pivot_table(index='Month', columns='SSC', values='DateTime', 
                           aggfunc='count', fill_value=0)        
    
    if model == 'obsv':
        pivot = pivot.drop(columns=[0], errors='ignore')
        SSClist_to_use = SSClist_obsv
    else:
        pivot = pivot.drop(columns=[0,70], errors='ignore')
        SSClist_to_use = SSClist_full
        
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100 # express each as a % frequency
    pivot.columns = [ssc_decode(a) for a in pivot.columns.values]
    pivot = pivot[[a for a in SSClist_to_use if a in pivot.columns]]
    # if typeAgg:
    #     if plusTypes:
    #         pivot = pivot[SSClist_plus]
    #     else:
    #         pivot = pivot[SSClist_main]    
    return pivot

def EnsembleFreqPivot(loc='MSP', scenario='HIST', MTplus=2, DTplus=2):
    modelPivots = [FreqPivot(loc, M, scenario, MTplus, DTplus) for M in model_dict.keys()]
    df = pd.concat(modelPivots).groupby('Month').mean()
    df = df[[a for a in SSClist_full if a in df.columns]]
    return df

# to route to either regular FreqPivot or the ensemble mean
def FreqPivot_select(loc='MSP', model='obsv', scenario='HIST', MTplus=2, DTplus=1):
    if model == 'ens':
        return EnsembleFreqPivot(loc, scenario, MTplus, DTplus)
    else:
        return FreqPivot(loc, model, scenario, MTplus, DTplus)

def FreqStackplot_obsv(savefig=True):
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    
    for ax, loc in zip(axes.flatten(), loc_dict.keys()):
        df = FreqPivot(loc, 'obsv', '', 1, 0)
        df.index = month_list
        
        x = df.index
        y = df.T.values
        
        ax.stackplot(x, y, labels=df.columns, 
                     colors=[sscColors[a] for a in df.columns])
                
        ax.set_xticklabels(df.index, rotation=45, ha='center')
        ax.set_ylabel('Frequency (%)')
        ax.set_title(loc_dict[loc], fontsize=12)
        ax.grid(axis='both', linestyle='dashed', color='gray')
    
    Handles, Labels = ax.get_legend_handles_labels() # to be reversed so legend order matches plots
    fig.legend(reversed(Handles), reversed(Labels), bbox_to_anchor=(1.0,0.6)) 
    
    fig_outpath_full = os.path.join(fig_outpath_obsv, 'obsvUrban_SSCfreq.png')  
    if savefig:
        plt.savefig(fig_outpath_full, dpi=150)

# 9-panel, all models + ensemble for single scenario
def FreqStackplot_wrf(loc='MSP', scenario='HIST', MTplus=1, DTplus=0, savefig=True):
        
    fig, axes = plt.subplots(3,3, figsize=(15,11))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    
    for ax, model in zip(axes.flatten(), model_dict2.keys()):
        df = FreqPivot_select(loc, model, scenario, MTplus, DTplus)
        df.index = month_list
        
        x = df.index
        y = df.T.values
        
        ax.stackplot(x, y, labels=df.columns, 
                     colors=[sscColors[a] for a in df.columns])
                
        ax.set_xticklabels(df.index, rotation=45, ha='center')
        ax.set_ylabel('Frequency (%)')
        ax.set_title(model_dict2[model], fontsize=12)
        ax.grid(axis='both', linestyle='dashed', color='gray')
    
    Handles, Labels = ax.get_legend_handles_labels() # to be reversed so legend order matches plots
    fig.legend(reversed(Handles), reversed(Labels), bbox_to_anchor=(1.0,0.6)) 
    fig.suptitle('{0} - {1}'.format(loc_dict[loc], scenario_namedict[scenario]), 
                                    fontsize=16, y=0.95)
    
    fig_outpath_full = os.path.join(fig_outpath_wrf,
                                    '{0}_{1}_SSCfreq.png'.format(loc, scenario))  
    if savefig:
        plt.savefig(fig_outpath_full, dpi=150, bbox_inches='tight')
        
# 2-panel, observed + historical ensemble        
def FreqStackplot_HistComparison(loc='MSP', MTplus=1, DTplus=0, savefig=True):
    
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    
    for ax, model in zip(axes.flatten(), ['obsv', 'ens']):
        df = FreqPivot_select(loc, model, 'HIST', MTplus, DTplus)
        df.index = month_list
        
        x = df.index
        y = df.T.values
        
        ax.stackplot(x, y, labels=df.columns, 
                     colors=[sscColors[a] for a in df.columns])
                
        ax.set_xticklabels(df.index, rotation=45, ha='center')
        ax.set_ylabel('Frequency (%)')
        ax.set_title(hist_namedict[model], fontsize=12)
        ax.grid(axis='both', linestyle='dashed', color='gray')
        
    Handles, Labels = axes[0].get_legend_handles_labels() # to be reversed so legend order matches plots
    fig.legend(reversed(Handles), reversed(Labels), bbox_to_anchor=(1.0,0.7)) 
    fig.suptitle('{0} air mass frequency'.format(loc_dict[loc]), 
                                    fontsize=16, y=1.05)
    
    fig_outpath_full = os.path.join(fig_outpath_wrf,
                                    '{0}_SSCfreq_HistComparison.png'.format(loc))  
    if savefig:
        plt.savefig(fig_outpath_full, dpi=150, bbox_inches='tight')

    
def seasonalDiffPlot(loc='MSP', scenario='HIST', typeAgg=True, savefig=True):
    
    fig, axes = plt.subplots(3,3, figsize=(15,11))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    df_obsv = FreqPivot(loc, 'obsv', scenario)
    df_obsv.index = month_list
    
    for ax, model in zip(axes.flatten(), model_dict2.keys()):
        df2 = FreqPivot_select(loc, model, scenario, typeAgg)
        df2.index = month_list
        df = df2 - df_obsv
 
        for ssc in df.columns:
            ax.plot(df.index, df[ssc], linewidth=2.5, color=sscColors[ssc])
        
        ax.set_xticklabels(df.index, rotation=45, ha='center')
        ax.set_ylabel('Frequency (%) difference')
        #ax.set_ylim(-35,45)
        ax.set_title('{} - Historical'.format(model_dict2[model]), fontsize=12)
        ax.grid(axis='both', linestyle='dashed', color='gray')
    
    fig.legend(labels=df.columns, bbox_to_anchor=(1.0,0.6))
    
    fig_outpath_full = os.path.join(fig_outpath_wrf,
                                    '{0}_{1}_SSCfreq_comparison.png'.format(loc, scenario))  
    if savefig:
        plt.savefig(fig_outpath_full, dpi=150)
