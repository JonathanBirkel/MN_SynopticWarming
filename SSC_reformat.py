"""
Reformat hourly model outputs to Scott's SSC algorithm inputs.
(.csv to .dat)

Updated to accomodate Duluth inputs
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import datetime as dt
import metpy.calc as mpcalc
from metpy.units import units


model_dict = {'BC':'bcc-csm1-1', 'CC':'CCSM4', 'CM':'CMCC-CM', \
              'CN':'CNRM-CM5', 'GF':'GFDL-ESM2M', 'IP':'IPSL-CM5A-LR', \
              'MI':'MIROC5', 'MR':'MRI-CGCM3', }
model_dict2 = {'BC':'bcc-csm1-1', 'CC':'CCSM4', 'CM':'CMCC-CM', \
              'CN':'CNRM-CM5', 'GF':'GFDL-ESM2M', 'IP':'IPSL-CM5A-LR', \
              'MI':'MIROC5', 'MR':'MRI-CGCM3', 'ens':'Ensemble'}
scenario_dict = {'HIST':'historical', 'MID':'RCP4.5', 'END4.5':'RCP4.5', 'END8.5':'RCP8.5'}
filename_dict = {'HIST':'TC4', 'MID':'TC45', 'END4.5':'TC89', 'END8.5':'TC4'}

# year_dict = {'HIST':[1980,1999], 'MID':[2040,2059], 'END4.5':[2060,2079], 'END8.5':[2080,2099]}
# END4.5 years offset so all scenarios can be strung together without duplicate dates

# now resetting all to HIST timeframe 
year_dict = {'HIST':[1980,1999], 'MID':[1980,1999], 'END4.5':[1980,1999], 'END8.5':[1980,1999]}

var_dict = {'temp':'T2_biascorrected', 'prcp':'precip_biascorrected', 'RH':'RH'}


def InputHrly(loc='MSP', scenario='HIST', model='CM'):
    Scen = scenario_dict[scenario]  
    if loc=='MSP':
        dirc = '/Users/birke111/Documents/Hourly_CSV/all_models_{}'.format(Scen)
        filename_dict = {'HIST':'TC4', 'MID':'TC45', 'END4.5':'TC89', 'END8.5':'TC4'}
    elif loc=='DLH':
        dirc = '/Users/birke111/Documents/Hourly_CSV_DLH/DU_all_models_{}'.format(Scen)
        filename_dict = {'HIST':'DU89', 'MID':'DU45', 'END4.5':'DU89', 'END8.5':'DU89'}
 
    # if model=='ens':
    #     inputfile = 'ensemble_{}.csv'.format(scenario)
    #     fullpath = os.path.join(dirc, inputfile)
    
    # else:
    Model = model_dict[model]
    inputfile = '{0}_{1}_hourly{2}.csv'.format(Model, Scen, filename_dict[scenario])
    fullpath = os.path.join(dirc, Model, Scen, inputfile)
    
    pre_dates_dict = {'HIST':'1979-12-31', 'MID':'2039-12-31', 
                      'END4.5':'2079-12-31', 'END8.5':'2079-12-31'}
    pre_date = pre_dates_dict[scenario]
    
    col_names = pd.read_csv(fullpath, nrows=0).columns
    types_dict = {'Date': str, pre_date: str} # Date header missing on some DLH files
    types_dict.update({col: float for col in col_names if col not in types_dict})
    
    df = pd.read_csv(fullpath, dtype=types_dict)
    
    if 'Date' not in df.columns: # first heading would be pre_date instead
        df.rename(columns = {pre_date: 'Date'}, inplace=True)
        df['Date'] = df.Date.shift(1)
                
    df['Year'] = np.zeros(len(df), dtype='int')
    df['DateTime'] = np.zeros(len(df), dtype=object)
        
    for a in range(len(df.Date.values)):
        try:            
            date_time = dt.datetime.strptime(str(df.Date.values[a]), '%Y-%m-%d')  
            date_time = date_time.replace(hour = int(df.Time[a]))
            
            #if scenario == 'END4.5':
            #    date_time = date_time.replace(year = date_time.year - 20)
            if scenario == 'MID':
                date_time = date_time.replace(year = date_time.year - 60)
            if scenario in ['END4.5', 'END8.5']:
                date_time = date_time.replace(year = date_time.year - 100)
            
            df.DateTime.values[a] = date_time
            df.Year.values[a] = date_time.year
            df.Date.values[a] = date_time.strftime('%Y%m%d')
        except ValueError:
            df.Year.values[a] = 9999
           
    df = df[ (df.Year >= year_dict[scenario][0]) & (df.Year <= year_dict[scenario][1]) ]
    df = df[ df['Time'].isin([3, 9, 15, 21]) ]
    df = df.reset_index()
            
    df = df.drop(columns=['index', 'GRDFLX', 'H', 'LE', 'Rnet', 'SNOWH', 'SW', 
                            'SurfacePressure(calculated)'], errors='ignore')
    return df

# see what's missing from each
def checkVars():
    full_varlist = ['Date', 'Time', 'direction10', 'GRDFLX', 'H', 'LE',
                    'precip_biascorrected', 'RH', 'Rnet', 'SNOWH', 'speed10', 'SW',
                    'T2_biascorrected', 'CloudFraction', 'SurfacePressure(calculated)',
                    'SurfacePressure(corrected)']
    for M in model_dict.keys():
        for S in scenario_dict.keys():
            Scen = scenario_dict[S]    
            dirc = '/Users/birke111/Documents/Hourly_CSV/all_models_{}'.format(Scen)
         
            if M=='ens':
                inputfile = 'ensemble_{}.csv'.format(S)
                fullpath = os.path.join(dirc, inputfile)
            
            else:
                Model = model_dict[M]
                inputfile = '{0}_{1}_hourly{2}.csv'.format(Model, Scen, filename_dict[S])
                fullpath = os.path.join(dirc, Model, Scen, inputfile)
            
            col_names = pd.read_csv(fullpath, nrows=0).columns
            absentcols = np.setdiff1d(full_varlist, col_names)
            print('model:', M, '\t scenario:', S, '\n \t missing:', absentcols)


def windComponents(wspd, wdir):
    U, V = mpcalc.wind_components(wspd * units('m/s'), wdir * units.deg)    
    return U.magnitude, V.magnitude

def dewpt(temp, relh):
    Td = mpcalc.dewpoint_from_relative_humidity(temp * units.degC, relh * units.percent)
    return Td.magnitude


def recalc(loc='MSP', scenario='HIST', model='CM'):
    df = InputHrly(loc, scenario, model)
    df['Temp'] = df['T2_biascorrected'] - 273.15
    
    df['DP'] = dewpt(df['Temp'].values, df['RH'].values)    
    
    # df['SLP'] = need to convert???
    # if using metpy: use units.mbar or units.hPa NOT units.mb!
    
    df['SLP'] = df['SurfacePressure(corrected)'] / 100
    mean_shift = 1013.25 - np.mean(df.SLP)
    df.SLP = df.SLP + mean_shift
    
    df.loc[df.Time == 3, 'SLP'] = df.loc[df.Time == 3, 'SLP'] + 1
    df.loc[df.Time == 9, 'SLP'] = df.loc[df.Time == 9, 'SLP'] - 1
    
    if 'direction10' in df.columns:
        df['U'] = windComponents(df['speed10'].values, df['direction10'].values)[0]
        df['V'] = windComponents(df['speed10'].values, df['direction10'].values)[1]
    else:
        df['U'], df['V'] = 0, 0
    
    df['CloudFraction'] = df['CloudFraction'] * 10
    
    return df[['Date', 'Time', 'Temp', 'DP', 'SLP', 'U', 'V', 'CloudFraction']]

def reformat(loc='MSP', scenario='HIST', model='CM'):
    df = recalc(loc, scenario, model)
    
    variable_cols = ['Temp', 'DP', 'SLP', 'U', 'V', 'CloudFraction']
    for col in variable_cols:
        df[col] = [format(a, '6.1f') for a in df[col]]
    
    #df.columns = [str(a) + str(df.columns[a]) for a in range(len(df.columns))]
    df.columns = ['Date', 'Time', '0Temp', '1DP', '2SLP', '3U', '4V', '5CC']

    # temporarily transposing, seems more flexible with hierarchical index than cols
    
    pivot = df.pivot(index='Date', columns='Time').T
    pivot = pivot.sort_index(axis=0, level=1).T    
    
    return pivot.dropna() # remove Dec31 of final year, where later hours are unfilled

    
def export_dat(loc='MSP', scenario='MID', model='CC', runNo=1): 
    '''
    df = reformat(scenario, model) #.reset_index().astype('str')
    if scenario != 'HIST':
        df_hist = reformat('HIST', model) #.reset_index().astype('str')
        df = pd.concat([df_hist, df])
    '''
    
    #df_list = [reformat(S, model) for S in scenario_dict.keys()]
    #df = pd.concat(df_list)    
    
    df = reformat(loc, scenario, model)
    df = df.reset_index().astype('str')
    
    #outpath = os.path.join('/Users/birke111/Documents/Hourly_CSV/SSC_inputsgrouped', scenario)
    outpath = '/Users/birke111/Documents/Hourly_CSV_{}/SSC_inputs_yearsreset'.format(loc)
    outpath = os.path.join(outpath, scenario)
    
    Path(outpath).mkdir(parents=True, exist_ok=True)
    
    #outfile = os.path.join(outpath, 'MSP_{0}_{1}.dat'.format(scenario, model_dict[model]))
    outfile = os.path.join(outpath, '{0}{1:0>2d}.dat'.format(loc[0], runNo))
    #outfile = os.path.join(outpath, 'MS{}.dat'.format(runNo))
    
    np.savetxt(fname=outfile, X=df.values, fmt='%s', delimiter='')
    #df.to_csv(os.path.join(outpath, 'MSP_{0}_{1}.txt'.format(scenario, model_dict[model])), 
    #          sep = '', header=False)
    
def export_loop(loc='MSP'):
    runNo = 1
    for scenario in scenario_dict.keys():
        #for model in ['CC', 'CM', 'GF', 'IP', 'MR']:
        for model in model_dict.keys():
            export_dat(loc, scenario, model, runNo)
            runNo = runNo + 1
   