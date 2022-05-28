"""
Functions of this code, to split into smaller files later:
    
Input tabular bias-corrected WRF data, extract 3a/3p temps/dewpts
Input SSC calendars and merge with other data
Calculate time series for SSC frequency, temp/dewpt trends, consecutive episodes
Calculate EHF, identify heat events, calculate timeseries trends and aligns with SSC categories
Calculate model ensemble averages of a given variable of interest
Evaluate historical (1980-99) model biases/agreement with observations
Perform t-tests to evaluate statistical significance of model scenario differences
Produce boxplots and/or tabular outputs


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from pathlib import Path
import datetime as dt
import metpy.calc as mpcalc
from metpy.units import units
#import statsmodels.api as sm
#import pylab
from obsv import obsv_process
import seaborn as sns
import ttests

loc_dict = {'MSP':'Minneapolis', 'DLH':'Duluth'}

#obsvInputPath = '/Users/birke111/Documents/ssc/obsv_new'
sscCalendarPath = '/Users/birke111/Documents/ssc/SSC_calendars'
obsv_tabularPath = '/Users/birke111/Documents/ssc/obsv_new/result_tabular3'
obsv_tabularPath_ObsModel = '/Users/birke111/Documents/ssc/obsv_new/result_tabular4'

model_dict = {'BC':'bcc-csm1-1', 'CC':'CCSM4', 'CM':'CMCC-CM', \
              'CN':'CNRM-CM5', 'GF':'GFDL-ESM2M', 'IP':'IPSL-CM5A-LR', \
              'MI':'MIROC5',  'MR':'MRI-CGCM3'}
model_dict2 = {'BC':'bcc-csm1-1', 'CC':'CCSM4', 'CM':'CMCC-CM', \
              'CN':'CNRM-CM5', 'GF':'GFDL-ESM2M', 'IP':'IPSL-CM5A-LR', \
              'MI':'MIROC5', 'MR':'MRI-CGCM3', 'ens':'Ensemble'}
    
scenario_dict = {'HIST':'historical', 'MID':'RCP4.5', 'END4.5':'RCP4.5', 'END8.5':'RCP8.5'}

# actual year range of each scenario
year_dict = {'HIST':[1980,1999], 'MID':[2040,2059], 'END4.5':[2080,2099], 'END8.5':[2080,2099]}

# previously had all scenarios in one file, with only END4.5 years shifted 
# year_dict_forSSC = {'HIST':[1980,1999], 'MID':[2040,2059], 'END4.5':[2060,2079], 'END8.5':[2080,2099]}

# all years are now set to 1980-1999 so each scenario can set its own climatology
year_dict_forSSC = {'HIST':[1980,1999], 'MID':[1980,1999], 'END4.5':[1980,1999], 'END8.5':[1980,1999]}


SSClist_full = ['DP', 'DM', 'DT', 'DT+', 'DT++', 'MP', 'MM', 'MT', 'MT+', 'MT++']
sscColors = {'DP':'wheat', 'DM':'sandybrown', 
             'DT':'orangered', 'DT+':'firebrick', 'DT++':'darkred',
             'DT (all)': 'lightpink',
             'MP':'paleturquoise', 'MM':'mediumturquoise', 
             'MT':'mediumseagreen', 'MT+':'green', 'MT++':'darkgreen',
             'TR':'grey', 'All':'lightgray'}
hwColors = {'From HIST $T_{95}$': 'orangered', 'From own $T_{95}$': 'sandybrown'}

fig_outpath = '/Users/birke111/Documents/ssc/wrf_new/boxplots'
fig_outpath_miscWRF = '/Users/birke111/Documents/ssc/wrf_new'
stats_outpath = '/Users/birke111/Documents/ssc/wrf_new/stats'

SSCvar_list = ['TMAX', 'TMIN', 'TRANGE', 'D3', 'D15', 'SSCFREQ']
SSCrunvar_list = ['SSC_runcount', 'SSC_rundur', 'SSC_rundurmax']
HWvar_list = ['HWssnlength', 'HWstartdate', 'HWintensity', 'HWintensitySevere', 
                'HWdur', 'HWfreq', 'HW_TDPcount', 'HW_EHFmax', 'HW_EHFmaxdate',
                'HWdurmax', 'HWdurmaxSevere', 'HWdaycount', 'HWdaycountSevere', 
                'HW_runcount5', 'HW_runcount7']
HWvar_sublist = ['HWssnlength', 'HWstartdate', 'HWintensity', 'HWintensitySevere', 
                 'HWdur', 'HWfreq', 'HWdaycount', 'HWdaycountSevere']

def FloatConv(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False 

    
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
  
def PivotFill(pivot, yrInit, yrEnd, fillWith=np.nan):  # fill in missing years in time series
    # defaults to filling with nan but can also do 0, etc.
    for i in range(yrInit, yrEnd):
        if i not in pivot.index.values:
            pivot.loc[i] = [fillWith] * pivot.shape[1]
    pivot = pivot.sort_index()
    return pivot

def inDateRange(dt_input, mInit=5, dInit=20, mEnd=9, dEnd=10): # dt_input in datetime format    
    # default: May20-Sep10 inclusive   
    init = dt.datetime(dt_input.year, mInit, dInit, 0, 0, 0)
    end = dt.datetime(dt_input.year, mEnd, dEnd, 23, 59, 59)
    
    return (dt_input >= init and dt_input <= end)

# should work for either hourly or daily dataframe inputs - has to have 'DateTime' column!
def seasonalDF(df, mInit=5, dInit=20, mEnd=9, dEnd=10):
    df['InRange'] = [inDateRange(a, mInit, dInit, mEnd, dEnd) for a in df['DateTime']]    
    sub_df = df[ df['InRange'] == True]  
    sub_df = sub_df.drop(columns='InRange')
    
    return sub_df

def dewpt(temp, relh):
    Td = mpcalc.dewpoint_from_relative_humidity(temp * units.degC, relh * units.percent)
    return Td.magnitude


def InputHourly(loc='MSP', scenario='HIST', model='CM'):
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
                    
    df['DateTime'] = pd.to_datetime(df.Date, errors='coerce')
    df = df.dropna() # remove rows (at end, or first if Duluth RCP4.5) with NaT
    
    '''   
    df['Time'] = pd.to_datetime(df.Time).dt.time
    df['DateTime'] = [pd.Timestamp.combine(df.DateTime[a], df.Time[a]).round('H')
                      for a in range(len(df))]    
    df['Date'] = [a.floor('D') for a in df.DateTime]
    df['Year'] = df.DateTime.dt.year
    '''
    df['Year'] = df.DateTime.dt.year
    df = df[ (df.Year >= year_dict[scenario][0]) & (df.Year <= year_dict[scenario][1]) ]   
    
    #df['JulianDay'] = [a.timetuple().tm_yday for a in df.Date]  
    df['TEMP'] = df['T2_biascorrected'] - 273.15
    df['DEWPT'] = dewpt(df['TEMP'].values, df['RH'].values)           
    
    df = df[['DateTime', 'Time', 'Year', 'TEMP', 'DEWPT']]
    df = df.reset_index(drop=True)
    
    return df

# could do an InputDaily from netcdfs - but more likely in a separate file entirely
# that would be for across-grid operations, totally different animal than what's happening here


def HourlySummaries(loc='MSP', scenario='HIST', model='CM'):
    df = InputHourly(loc, scenario, model)
    
    df_3a3p = df[df.Time.isin([3,15])]
    pivot = df_3a3p.pivot(index='DateTime', columns='Time')
    pivot = pivot[['TEMP','DEWPT']]
    pivot.columns = ['TMIN', 'TMAX', 'D3', 'D15']
    
    #pivot['TRANGE'] = pivot.TMAX - pivot.TMIN.shift(-1) 
    
    pivot['TRANGE'] = pivot.TMAX - pivot.TMIN # as with obsv_new, removing the shift now
    pivot = pivot.dropna()
    
    return pivot.reset_index()



def SSCinput(loc='MSP', scenario='HIST', model='CM'):
    model_list = list(model_dict.keys())
    #cals_list = [loc[0:2] + str(i+1) for i in range(len(model_list))]
    
    scenNo_dict = {'HIST':0, 'MID':1, 'END4.5':2, 'END8.5':3}
    cals_list = ['M{:0>2d}'.format(i+1 + 8*scenNo_dict[scenario]) 
                 for i in range(len(model_list))]
    cals_dict = {model_list[i]: cals_list[i] for i in range(len(model_list))}
    
    txtfile = os.path.join(sscCalendarPath, 'fromWRF', loc, '{}.cal3notr'.format(cals_dict[model]))
    df = pd.read_csv(txtfile, sep=' ', header=None, names=['Station', 'Date', 'SSC'])
    
    df.index = pd.to_datetime(df.Date, format='%Y%m%d')    
    df['SSC'] = [(a if a in range(10,80) else 0) for a in df.SSC]
    
    df['InYearRange'] = [y in range(year_dict_forSSC[scenario][0], year_dict_forSSC[scenario][1]+1) 
                         for y in df.index.year]
    df = df[df.InYearRange == True]
    
    years_shift = {'HIST':0, 'MID':60, 'END4.5':100, 'END8.5':100}
    #if scenario == 'END4.5':
    #    df.index = [i.replace(year = i.year + 20) for i in df.index]
    df.index = [i.replace(year = i.year + years_shift[scenario]) for i in df.index]
    return df.drop(columns=['Station', 'Date', 'InYearRange'])

  
def SSCjoin(loc='MSP', scenario='HIST', model='CM'):   

    ssc_df = SSCinput(loc, scenario, model)
    
    wrf_df = HourlySummaries(loc, scenario, model)
    df = wrf_df.merge(ssc_df, how='left', left_on='DateTime', right_index=True)
    df['Year'] = df.DateTime.dt.year
    return df
    

def SSCtrends(loc='MSP', scenario='HIST', model='CM', trend='SSCFREQ', 
              ssctype=60, MTplus=1, DTplus=0, seasonal=True, yearAgg=True,
              customSeasonal=False, customDates=[6,1,6,30]):
    # now doing yearly pivot only by default for SSCFREQ
    # yearAgg argument dictates whether to pivot others by year
    yrInit, yrEnd = year_dict[scenario]
                    
    df = SSCjoin(loc, scenario, model)
    if customSeasonal:
        df = seasonalDF(df, customDates[0], customDates[1], customDates[2], customDates[3]) 
    elif seasonal:
        df = seasonalDF(df, 5, 20, 9, 10)
    df['SSC'] = df.SSC.fillna(0)
    #if typeAgg:
    df['SSC'] = [TypeAgg(a, MTplus, DTplus) for a in df.SSC]
    
    if trend == 'SSCFREQ':
        # some END8.5 models don't have DP at all - need an all-zeros column for that
        try:
            pivot = df.pivot_table(index='Year', columns='SSC', values='DateTime', 
                                   aggfunc='count', fill_value=0, margins=False)
            timeseries = pivot.loc[yrInit:yrEnd,ssctype]
        except KeyError: # i.e. if there is no 20 column in pivot
            pivot = PivotFill(df[df.SSC == ssctype], yrInit, yrEnd)
            pivot.index.name = 'Year'
            timeseries = pivot.SSC.fillna(0)
               
    elif trend in ['TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
        if ssctype==99:
            if yearAgg:
                pivot = df.pivot_table(index='Year', values=trend, 
                                   aggfunc='mean', fill_value=None, margins=False)
                pivot = PivotFill(pivot,yrInit,yrEnd)
                timeseries = pivot.loc[yrInit:yrEnd,trend]
            else:
                timeseries = df[trend]
        else:
            if yearAgg:
                try: 
                    pivot = df.pivot_table(index='Year', columns='SSC', values=trend, 
                                           aggfunc='mean', fill_value=None, margins=False)
                    pivot = PivotFill(pivot,yrInit,yrEnd)
                    timeseries = pivot.loc[yrInit:yrEnd,ssctype]
                except KeyError: # i.e. if there is no 20 column in pivot
                    pivot = PivotFill(df[df.SSC == ssctype], yrInit, yrEnd)
                    pivot.index.name = 'Year'
                    timeseries = pivot.SSC # seemingly no issues yet with NaN averaging
            else:    
                timeseries = df.loc[(df.SSC == ssctype), trend]
    
    return timeseries
       

   
def SSCruns(loc='MSP', scenario='HIST', model='CM', opp_types = [30,61], minrun=3, seasonal=True):
    df = SSCjoin(loc, scenario, model)
    df['SSC'] = df.SSC.fillna(0)
    df['SSC'] = [TypeAgg(a, True) for a in df.SSC]
    
    df['OppDay'] = [a in opp_types for a in df.SSC]
    df['Runstart'] = 0
    df['InRun'] = False
    
    for i in range(1, len(df)):
        runlength=0        
        if seasonal:
            is_seasonstart = inDateRange(df.DateTime[i], 5, 20, 5, 20)
        else:
            is_seasonstart = False
        possible_startday = not df.OppDay[i-1] or is_seasonstart
        
        if df.OppDay[i] and possible_startday:
            runlength = 1
            for j in range(1,200): # max for observations was 20, but models have longer runs!
                if df.OppDay[i+j]==1:
                    if seasonal and not inDateRange(df.DateTime[i], 5, 20, 9, 10):
                        break # eg. a 5-day run starting Sep 8 counts as only 3 days
                    else:
                        runlength = runlength+1
                else:
                    break
            df.Runstart[i] = runlength
    for m in range(len(df)):
        if df.Runstart[m] >= minrun:
            for n in range(df.Runstart[m]):
                df.InRun[m+n]=1   
                
    return df

def SSCruntrends(loc='MSP', scenario='HIST', model='CM', stat='SSC_runcount', 
                 seasonal=True, opp_types=[30,61], minrun=3, yearAgg=False):
    df = SSCruns(loc, scenario, model, opp_types, minrun, seasonal)
    if seasonal:
        df = seasonalDF(df, 5, 20, 9, 10)
    
    if stat == 'SSC_runcount':
        df['Runstart'] = df.Runstart > 2
        yearlydf = df.groupby('Year').agg('sum')
        return yearlydf.loc[:, 'Runstart']    
    
    elif stat == 'SSC_rundur':
        if yearAgg:    # unlikely to use it this way! just good to have flexibility to do so?
            yearlydf = df.loc[(df.Runstart > 2), :]
            yearlydf = yearlydf.groupby('Year').agg('mean')
            yearlydf = PivotFill(yearlydf, year_dict[scenario][0], year_dict[scenario][1]+1, 0)
            # should I fill with 0? or np.nan as in default? (or moot point anyway?)
            return yearlydf.loc[:, 'Runstart']
        else:
            return df.loc[(df.Runstart > 2), 'Runstart']
   
    elif stat == 'SSC_rundurmax':
        yearlydf = df.groupby('Year').agg('max')
        return yearlydf.loc[:, 'Runstart']
    
       
def heatwaves_setup(loc='MSP', scenario='HIST', model='CM'):
    df = HourlySummaries(loc, scenario, model)    
    df['TMEAN'] = (df.TMAX + df.TMIN.shift(-1)) / 2
    T95 = np.nanpercentile(df.TMEAN, 95)
    return df, T95    
    
def EHF_heatwaves(loc='MSP', scenario='HIST', model='CM', use_hist_t95=True):
    df, T95 = heatwaves_setup(loc, scenario, model)
    if use_hist_t95 and scenario != 'HIST':
        T95 = heatwaves_setup(loc, 'HIST', model)[1]
    
    df['PeriodAvg'] = [np.mean(df.TMEAN[i:i+3]) for i in range(len(df))]
    # could apply a shift if wanting to assign to 2nd or 3rd day of period instead of 1st
    df['AcclPeriodAvg'] = [np.mean(df.TMEAN[i-30:i]) for i in range(len(df))]

    df['EHIsig'] = df.PeriodAvg - T95
    df['EHIaccl'] = df.PeriodAvg - df.AcclPeriodAvg
    
    df.EHIsig[df.EHIsig < 0] = 0  # only care about positive values here   
    df.EHIaccl[(df.EHIaccl < 1) | (df.EHIaccl.isna())] = 1
    
    df['EHF'] = df['EHIsig'] * df['EHIaccl']
    EHF85 = np.nanpercentile(df.EHF[df.EHF > 0], 85) # "severity threshold"

    df['HeatWaveDay'] = [any(df.EHIsig[i-2:i+1] > 0) for i in range(len(df))]
    df['SevereHeatWaveDay'] = [any(df.EHF[i-2:i+1] > EHF85) for i in range(len(df))]
    
    df['Runstart'] = 0 # mark start of heat wave with # of days in it
    for i in range(len(df)):
        runlength = 0
        if df.HeatWaveDay[i] and not df.HeatWaveDay[i-1]:
            runlength = 1
            for j in range(1,200): # 40 wasn't enough!
                if df.HeatWaveDay[i+j]:
                    runlength = runlength+1
                else:
                    break
            df.Runstart[i] = runlength
    df['RunstartSevere'] = 0 # same, but only counting consecutive severe TDPs
    for i in range(len(df)):
        runlength = 0
        if df.SevereHeatWaveDay[i] and not df.SevereHeatWaveDay[i-1]:
            runlength = 1
            for j in range(1,50): # MSP's max is 20, leaving a safe margin for models later
                if df.SevereHeatWaveDay[i+j]:
                    runlength = runlength+1
                else:
                    break
            df.RunstartSevere[i] = runlength
    
    df = df[['DateTime', 'TMEAN', 'EHIsig', 'EHIaccl', 'EHF', 'HeatWaveDay', 
             'SevereHeatWaveDay', 'Runstart', 'RunstartSevere']]
    return df, T95, EHF85
  

def heatwaveSSC(loc='MSP', scenario='HIST', model='CM', use_hist_t95=True, severe=False):

    df, T95, EHF85 = EHF_heatwaves(loc, scenario, model, use_hist_t95)

    ssc_df = SSCinput(loc, scenario, model)
    ssc_df.index.name = 'DateTime'

    # timestamps are index - need to give index a name to pull for right_on
    
    df = df.merge(ssc_df, how='left', left_on='DateTime', right_on='DateTime')   
    df = df[df.HeatWaveDay == True] 
    df['SSC'] = [ssc_decode(TypeAgg(a, 1, 0)) for a in df.SSC] #.fillna(0)]
    
    if severe:
        df = df[df.SevereHeatWaveDay == True]
    
    SSCcounts = df['SSC'].value_counts()    
    for ssc in ['DP', 'DM', 'DT', 'MP', 'MM', 'MT', 'MT+', 'TR']:
        if ssc not in SSCcounts.index:
            SSCcounts[ssc] = 0
    
    return SSCcounts


def heatwaveSSC_tabular(loc='MSP', use_hist_t95=True): 
    df = pd.DataFrame()
    for S in scenario_dict.keys():
        DF = pd.DataFrame()
        DFsevere = pd.DataFrame()
        for M in model_dict.keys():
            counts = heatwaveSSC(loc, S, M, use_hist_t95, False)
            # counts_pct = (counts / sum(counts)) * 100
            countsSevere = heatwaveSSC(loc, S, M, use_hist_t95, True)
            # countsSevere_pct = (countsSevere / sum(countsSevere)) * 100
            DF[M] = counts
            DFsevere[M] = countsSevere
       
        counts = DF.sum(axis=1)
        countsSevere = DFsevere.sum(axis=1)
       
        df['{}_counts'.format(S)] = counts
        df['{}_counts_pct'.format(S)] = (counts / sum(counts)) * 100
        df['{}_countsSevere'.format(S)] = countsSevere
        df['{}_countsSevere_pct'.format(S)] = (countsSevere / sum(countsSevere)) * 100
        
    df = df.T
    df = df[['DP', 'DM', 'DT', 'MP', 'MM', 'MT', 'MT+']]
    
    if use_hist_t95:
        outpath = os.path.join(stats_outpath, 'heatwaveSSC_NoAccl.csv')
    else:
        outpath = os.path.join(stats_outpath, 'heatwaveSSC_FullAccl.csv')
    df.to_csv(outpath)

def heatwavetrends(trend='HWssnlength', loc='MSP', scenario='HIST', model='CM', 
                   use_hist_t95=True, yearAgg=False):
    #trends: HWssnlength, HWstartdate, HWintensity, HWintensitySevere,  
    #        HWdur, HWfreq, HW_TDPcount, HW_EHFmax,
    #        HWdurmax, HWdurmaxSevere, HWdaycount, HWdaycountSevere, 
    #        HW_runcount5, HW_runcount7
    df, T95, EHF85 = EHF_heatwaves(loc, scenario, model, use_hist_t95)
    
    df['JulianDay'] = [a.timetuple().tm_yday for a in df.DateTime]
    df['Year'] = df.DateTime.dt.year  
    # apparently only need .dt accessor if timestamps are a series rather than index
    
    if trend not in ['HWdur', 'HWintensity', 'HWintensitySevere']:
        yearAgg = True
   
    df = df[df.HeatWaveDay == True]
    if trend == 'HWssnlength':
        dfgroup = df[['Year', 'JulianDay']].groupby('Year').agg(['max','min'])
        dfgroup.columns = ['Jmax', 'Jmin']
        dfgroup[trend] = dfgroup.Jmax - dfgroup.Jmin + 1
    elif trend == 'HWstartdate':
        dfgroup = df[['Year', 'JulianDay']].groupby('Year').agg(['min'])
        dfgroup.columns = [trend]
    
    elif trend == 'HWintensity':
        # if taken in 3-day groups, this is really just EHIsig!
        '''
        DF = df.copy(deep=True)
        DF = DF[DF.EHIsig > 0]
        if yearAgg:
            dfgroup = df[['Year', 'EHIsig']].groupby('Year').agg('mean')
        else: 
           dfgroup = df.loc[:, 'EHIsig']
        '''
        # but this way shows more individual day character - 
        #    how many "heat wave days" aren't actually above T95 individually??
        if yearAgg:
            dfgroup = df[['Year', 'TMEAN']].groupby('Year').agg('mean') - T95
        else: 
           dfgroup = df.loc[:, 'TMEAN'] - T95
        
        dfgroup.columns = [trend]
               
    elif trend == 'HWintensitySevere':
        DF = df.copy(deep=True)
        DF = DF[DF.SevereHeatWaveDay == True]
        if yearAgg:
            dfgroup = DF[['Year', 'TMEAN']].groupby('Year').agg('mean') - T95
        else:
            dfgroup = DF.loc[:, 'TMEAN'] - T95
        dfgroup.columns = [trend]
   
    elif trend == 'HWdur':
        DF = df.copy(deep=True)
        DF = DF[DF.Runstart > 0]
        if yearAgg:
            dfgroup = DF[['Year', 'Runstart']].groupby('Year').agg('mean')
        else:
            dfgroup = DF.loc[:, 'Runstart']
        #dfgroup.name = trend
        dfgroup.columns = [trend]
    
    elif trend == 'HWfreq':
        DF = df.copy(deep=True)
        DF = DF[DF.Runstart > 0]
        dfgroup = DF[['Year', 'Runstart']].groupby('Year').agg('count')
        dfgroup.columns = [trend]
    elif trend == 'HW_TDPcount':
        DF = df.copy(deep=True)
        DF = DF[DF.EHF > 0]
        dfgroup = DF[['Year', 'EHF']].groupby('Year').agg('count')
        dfgroup.columns = [trend]
    elif trend == 'HW_EHFmax':
        DF = df.copy(deep=True)
        DF = DF[DF.EHF > 0]
        dfgroup = DF[['Year', 'EHF']].groupby('Year').agg('max')
        dfgroup.columns = [trend]
    elif trend == 'HW_EHFmaxdate':
        DF = df.copy(deep=True)
        DF = DF[DF.EHF > 0][['Year', 'EHF', 'JulianDay']]
        dfgroup = DF.loc[DF.groupby('Year')['EHF'].idxmax()]
        dfgroup = dfgroup.set_index('Year').drop(columns='EHF')
        dfgroup.columns = [trend]
    
    # new additions
    #HWdurmax, HWdurmaxSevere, HWdaycount, HWdaycountSevere, HW_runcount (runlength 5 or 7?)
    elif trend == 'HWdurmax':
        DF = df.copy(deep=True)
        DF = DF[DF.Runstart > 0]
        dfgroup = DF[['Year', 'Runstart']].groupby('Year').agg('max')
        dfgroup.columns = [trend]
    elif trend == 'HWdurmaxSevere':
        DF = df.copy(deep=True)
        DF = DF[DF.RunstartSevere > 0]
        dfgroup = DF[['Year', 'RunstartSevere']].groupby('Year').agg('max')
        dfgroup.columns = [trend]
    elif trend == 'HWdaycount':
        DF = df.copy(deep=True)
        dfgroup = DF[['Year', 'HeatWaveDay']].groupby('Year').agg('count')
        dfgroup.columns = [trend]
    elif trend == 'HWdaycountSevere':    
        DF = df.copy(deep=True)
        DF = DF[DF.SevereHeatWaveDay == True]
        dfgroup = DF[['Year', 'SevereHeatWaveDay']].groupby('Year').agg('count')
        dfgroup.columns = [trend]
        
    elif trend == 'HW_runcount5':
        DF = df.copy(deep=True)
        DF = DF[DF.Runstart >= 5]
        dfgroup = DF[['Year', 'Runstart']].groupby('Year').agg('count')        
        dfgroup.columns = [trend]
    elif trend == 'HW_runcount7':
        DF = df.copy(deep=True)
        DF = DF[DF.Runstart >= 7]
        dfgroup = DF[['Year', 'Runstart']].groupby('Year').agg('count')        
        dfgroup.columns = [trend]
        # could consolidate those 2 (add "runlength" as another argument)
        # but taking the quick n dirty approach right now
    
    
    if yearAgg:
        yrInit, yrEnd = year_dict[scenario]
        dfgroup = PivotFill(dfgroup, yrInit, yrEnd+1)  
        timeseries = dfgroup.loc[yrInit:yrEnd,trend]    
        if trend in ['HWssnlength', 'HWfreq', 'HW_TDPcount', 
                     'HWdaycount', 'HWdaycountSevere', 'HW_runcount5', 'HW_runcount7']: 
        #       I think these are the only ones where replacing NaN with 0 makes sense?
            timeseries = timeseries.fillna(0) 
    else:
        timeseries = dfgroup
    
    return timeseries, T95, EHF85


    # see if this works ok! yearAgg only affects 3(?) HW vars anyway
    #     but make sure format is workable downstream


##########################################################################

 
# average any 20-year time series across models
# use for obs-model agreement assessments, NOT for ttests
def EnsembleAvg(loc='MSP', scenario='HIST', trend='SSCFREQ', 
                ssctype=60, MTplus=1, DTplus=0, seasonal=True,
                customSeasonal=False, customDates=[6,1,6,30]):
    #if trend in ['SSCFREQ', 'TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
    modelSeries = [SSCtrends(loc, scenario, M, trend, ssctype, 
                             MTplus, DTplus, seasonal, True,
                             customSeasonal, customDates) for M in model_dict.keys()]
           
    df = pd.concat(modelSeries).groupby('Year').mean()
    return df   

# for model-observation comparisons - outputs both in same format
def ModelObsSeries(loc='MSP', var='TMAX', ssctype=99, month='Jun'):
    dates_dict = {'Jun': [6,1,6,30], 'Jul': [7,1,7,31], 'Aug': [8,1,8,31]} 
    
    if var == 'SSCFREQ':
        model_series = EnsembleAvg(loc, 'HIST', var, ssctype)
        obsv_file = os.path.join(obsv_tabularPath_ObsModel, var, loc,
                                 ssc_decode(ssctype), '1980-1999.csv')
    else:   # TMIN, TMAX, D3, D15
        model_series = EnsembleAvg(loc, 'HIST', var, ssctype, 
                                   customSeasonal=True, customDates=dates_dict[month])
        obsv_file = os.path.join(obsv_tabularPath_ObsModel, var, loc,
                                 ssc_decode(ssctype), month, '1980-1999.csv')
    obsv_series = pd.read_csv(obsv_file, index_col=0)[var]
    return model_series.values, obsv_series.values

def ModelAgreementStats(loc='MSP', var='T15', ssctype=99, month='Jun'):
    if var == 'T15':
        var = 'TMAX'
    elif var == 'T3':
        var = 'TMIN'
    
    model_series, obsv_series = ModelObsSeries(loc, var, ssctype, month)
    
    obsMean = np.mean(obsv_series)
    simsMean = np.mean(model_series)
    MBE = simsMean - obsMean
    RMSD = np.sqrt(np.mean((model_series - obsv_series)**2))
    PctError = 100 * (simsMean - obsMean) / obsMean
    
    statsdict = {'ObsMean': obsMean, 'SimsMean': simsMean, 'RMSD': RMSD, 'PctError': PctError}
    return statsdict

def ModelAgreementTable(loc='MSP'):
    statsdict = {}
    
    for var in ['T3', 'D3', 'T15', 'D15']:
        for month in ['Jun', 'Jul', 'Aug']:
            rowname = var + '_' + month
            statsdict[rowname] = ModelAgreementStats(loc, var, 99, month)
    for ssctype in [10, 20, 30, 40, 50, 60, 61]:
        rowname = 'SSCFREQ_' + ssc_decode(ssctype)
        statsdict[rowname] = ModelAgreementStats(loc, 'SSCFREQ', ssctype, '')
    
    df = pd.DataFrame.from_dict(statsdict, orient='index')
    file_outpath = os.path.join(stats_outpath, '{}_ModelAgreementTable.csv'.format(loc))
    df.to_csv(file_outpath)
    
def ModelAgreementPlot(loc='MSP'): #, var='T15', ssctype=99, month='Jun'):
    var_list = ['T3', 'D3', 'T15', 'D15'] * 3 + ['SSCFREQ'] * 7
    ssc_list = [99] * 12 + [20, 10, 30, 50, 40, 60, 61]
    mon_list = ['Jun'] * 4 + ['Jul'] * 4 + ['Aug'] * 4 + [''] * 7
    #lists = list(zip(var_list, ssc_list, mon_list))
    
    fig, axes = plt.subplots(5,4, figsize=(16,20))
    fig.subplots_adjust(hspace=0.45, wspace=0.25)
    for ax, var, ssctype, month in zip(axes.flatten(), var_list, ssc_list, mon_list):
            
        if var == 'T15':
            Var = 'TMAX'
        elif var == 'T3':
            Var = 'TMIN'   
        else:
            Var = var
        y, x = ModelObsSeries(loc, Var, ssctype, month)
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x,y)
        rsquared = rvalue**2
        linreg_y = slope*x + intercept
        
        maxValue = np.append(x,y).max()
        minValue = np.append(x,y).min()
        pltRange = maxValue - minValue
        pltMax = pltRange * 0.05 + maxValue
        pltMin = pltRange * -0.05 + minValue
        
        ax.scatter(x,y,s=25, c = 'black')
        ax.plot(x, linreg_y, c='red')
        
        if var == 'SSCFREQ':
            titlestr = 'SSCFREQ_' + ssc_decode(ssctype)
        else:
            titlestr = var + '_' + month
        
        ax.set_title(titlestr, fontsize=14)
        ax.set_xlabel('Observed', fontsize=12)
        ax.set_ylabel('Simulated', fontsize=12)
        ax.set_xlim(pltMin, pltMax)
        ax.set_ylim(pltMin, pltMax)
        ax.set_aspect('equal')
    axes[-1,-1].axis('off')
    fig.suptitle('Model-observation agreement, Minneapolis 1980-1999', fontsize=16, y=0.92)
        
    fig_outpath_full = os.path.join(fig_outpath_miscWRF, 'ModelAgreement.png')
    plt.savefig(fig_outpath_full, dpi=150, bbox_inches='tight')
    
def ModelAgreementPlot_temp(loc='MSP'): #, var='T15', ssctype=99, month='Jun'):
    var_list = ['T3', 'D3', 'T15', 'D15'] * 3
    ssc_list = [99] * 12
    mon_list = ['Jun'] * 4 + ['Jul'] * 4 + ['Aug'] * 4
    #lists = list(zip(var_list, ssc_list, mon_list))
    
    fig, axes = plt.subplots(3,4, figsize=(16,12))
    fig.subplots_adjust(hspace=0.45, wspace=0.25)
    for ax, var, ssctype, month in zip(axes.flatten(), var_list, ssc_list, mon_list):
            
        if var == 'T15':
            Var = 'TMAX'
        elif var == 'T3':
            Var = 'TMIN'   
        else:
            Var = var
        y, x = ModelObsSeries(loc, Var, ssctype, month)
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x,y)
        rsquared = rvalue**2
        linreg_y = slope*x + intercept
        
        maxValue = np.append(x,y).max()
        minValue = np.append(x,y).min()
        pltRange = maxValue - minValue
        pltMax = pltRange * 0.05 + maxValue
        pltMin = pltRange * -0.05 + minValue
        
        ax.scatter(x,y,s=25, c = 'black')
        ax.plot(x, linreg_y, c='red')
        
        var_dict = {'T3': '3:00 temperature', 'D3': '3:00 dew point', 
                    'T15': '15:00 temperature', 'D15': '15:00 dew point'}
        month_dict = {'Jun': 'June', 'Jul': 'July', 'Aug': 'August'}
        titlestr = var_dict[var] + ', ' + month_dict[month]
        
        ax.set_title(titlestr, fontsize=14)
        ax.set_xlabel('Observed', fontsize=12)
        ax.set_ylabel('Simulated', fontsize=12)
        ax.set_xlim(pltMin, pltMax)
        ax.set_ylim(pltMin, pltMax)
        ax.set_aspect('equal')

    fig.suptitle('Model-observation agreement, Minneapolis 1980-1999', fontsize=16, y=0.98)
        
    fig_outpath_full = os.path.join(fig_outpath_miscWRF, 'TEMP_ModelAgreement.png')
    plt.savefig(fig_outpath_full, dpi=150, bbox_inches='tight')
    
def ModelAgreementPlot_SSCFREQ(loc='MSP'): #, var='T15', ssctype=99, month='Jun'):
    var_list = ['SSCFREQ'] * 7
    ssc_list = [20, 10, 30, 50, 40, 60, 61]
    mon_list = [''] * 7
    #lists = list(zip(var_list, ssc_list, mon_list))
    
    fig, axes = plt.subplots(2,4, figsize=(16,8))
    fig.subplots_adjust(hspace=0.45, wspace=0.25)
    for ax, var, ssctype, month in zip(axes.flatten(), var_list, ssc_list, mon_list):

        y, x = ModelObsSeries(loc, var, ssctype, month)
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x,y)
        rsquared = rvalue**2
        linreg_y = slope*x + intercept
        
        maxValue = np.append(x,y).max()
        minValue = np.append(x,y).min()
        pltRange = maxValue - minValue
        pltMax = pltRange * 0.05 + maxValue
        pltMin = pltRange * -0.05 + minValue
        
        ax.scatter(x,y,s=25, c = 'black')
        ax.plot(x, linreg_y, c='red')
        
        titlestr = ssc_decode(ssctype) + ' frequency'
        
        ax.set_title(titlestr, fontsize=14)
        ax.set_xlabel('Observed', fontsize=12)
        ax.set_ylabel('Simulated', fontsize=12)
        ax.set_xlim(pltMin, pltMax)
        ax.set_ylim(pltMin, pltMax)
        ax.set_aspect('equal')
    axes[-1,-1].axis('off')
    fig.suptitle('Model-observation agreement, Minneapolis 1980-1999', fontsize=16, y=1.0)
        
    fig_outpath_full = os.path.join(fig_outpath_miscWRF, 'SSCFREQ_ModelAgreement.png')
    plt.savefig(fig_outpath_full, dpi=150, bbox_inches='tight')


##################################################################


# compile a single variable across models - 
# doesn't matter if yearly or daily values, all becoming one 1d array

# also outputs T95 and EHF85 as arrays of 8 (1 per model)

def EnsembleConcat(loc='MSP', scenario='HIST', trend='SSCFREQ', 
                ssctype=60, MTplus=1, DTplus=0, seasonal=True, yearAgg=True, use_hist_t95=True):
    T95, EHF85 = [], []
    model_list = list(model_dict.keys())
    # for HWvars these will become lists of 8 (1 for each model)
    
    if trend in SSCvar_list:
        modelSeries = [SSCtrends(loc, scenario, M, trend, ssctype, 
                                 MTplus, DTplus, seasonal, yearAgg) for M in model_list]
    elif trend in SSCrunvar_list:
        modelSeries = [SSCruntrends(loc, scenario, M, trend, seasonal, 
                                    [60+MTplus, 30+DTplus], 3, yearAgg) for M in model_list]

    elif trend in HWvar_list:
        modelSeries, T95, EHF85 = [], [], []
        for i in range(len(model_list)):
            MS, T, E = heatwavetrends(trend, loc, scenario, model_list[i], 
                                                              use_hist_t95, yearAgg)
            modelSeries.append(MS)
            T95.append(T)
            EHF85.append(E)
        # modelSeries = [heatwavetrends(trend, loc, scenario, M, use_hist_t95, yearAgg)[0]
        #                for M in model_dict.keys()]          
    df = pd.concat(modelSeries)
    return df, T95, EHF85


def combinebySSC(loc='MSP', trend='SSCFREQ', ssctype=60, includeObsv=False,
                 MTplus=1, DTplus=0, seasonal=True, yearAgg=True):
    df = pd.DataFrame()
    scenario_list = list(scenario_dict.keys())
    for scenario in scenario_list:
        series = EnsembleConcat(loc, scenario, trend, ssctype, MTplus, DTplus, 
                                seasonal, yearAgg)[0].reset_index(drop=True)
        df = pd.concat([df, series], ignore_index=True, axis=1)
    
    if includeObsv:
        
        #### still using yearly obsv means for everything here!
        ####        (adjust later? it's not part of ttests anyway...)
        
        if trend in ['SSCFREQ', 'SSC_runcount', 'D3', 'D15']:               
            obsv_input = os.path.join(obsv_tabularPath, trend, loc, ssc_decode(ssctype),
                                  '1948-2019.csv') 
        elif trend in ['TMAX', 'TMIN', 'TRANGE']:
            obsv_input = os.path.join(obsv_tabularPath, trend, loc, ssc_decode(ssctype),
                                  'hourly','1948-2019.csv') 
        # same year range here for all, just not for original station records
        
        station_series = pd.read_csv(obsv_input)[trend]
        
        # station_series = obsv_new.SSCtrends(DF, loc, trend, inputType, 
        #                                          1948, 2021, ssctype, True, True, seasonal).values
        df_obsv = pd.DataFrame(station_series) #, columns=['Station'])
        df = pd.concat([df_obsv, df], ignore_index=True, axis=1) 
        #   removes col names so have to reassign
        scenario_list.insert(0, 'Station')
    df.columns = scenario_list
    df['SSC'] = ssctype
    return df

# probably redundant with above!
def combine_forSSCruns(loc='MSP', trend='SSC_runcount', includeObsv=False, 
                       MTplus=1, DTplus=0, seasonal=True, yearAgg=False):
    df = pd.DataFrame()
    scenario_list = list(scenario_dict.keys())
    
    for scenario in scenario_list:
        series = EnsembleConcat(loc, scenario, trend, 60, MTplus, DTplus, 
                                seasonal, yearAgg)[0].reset_index(drop=True)
        df = pd.concat([df, series], ignore_index=True, axis=1)
    df.columns = scenario_list
    df['SSCRUN'] = True # doesn't do anything but adds consistency to formatting?
    return df

def combine_forHW(loc='MSP', trend='HWdur', includeObsv=False, 
                  use_hist_t95=True, yearAgg=False):
    
    df = pd.DataFrame()
    T95_df = pd.DataFrame()
    EHF85_df = pd.DataFrame()
    scenario_list = list(scenario_dict.keys())
    
    for scenario in scenario_list:
        ensConcat, T95, EHF85 = EnsembleConcat(loc, scenario, trend, 60, 1, 0,
                                               False, yearAgg, use_hist_t95)
        df[scenario] = ensConcat.reset_index(drop=True)
        T95_df[scenario] = T95
        EHF85_df[scenario] = EHF85
        
    if includeObsv:
        obsv_input = os.path.join(obsv_tabularPath, trend, loc, '1948-2019.csv') 
        station_series = pd.read_csv(obsv_input)[trend]
        df_obsv = pd.DataFrame(station_series)
        df = pd.concat([df_obsv, df], ignore_index=True, axis=1) 
        #   removes col names so have to reassign
        scenario_list.insert(0, 'Station')
        df.columns = scenario_list
    df['HIST_T95'] = use_hist_t95
    return df, T95_df, EHF85_df

    # df may not always return 160 rows if no scenario has events every year in all models
    # shouldn't be an issue if ttests don't require equal nobs?
    
    

def ttest_significance(loc='MSP', trend='SSCFREQ', ssctype=60,
                 MTplus=1, DTplus=0, seasonal=True, yearAgg=True, use_hist_t95=True):
    if trend in HWvar_list:
        df = combine_forHW(loc, trend, False, use_hist_t95, yearAgg)[0]
        
    # applies to SSCruns as well as other SSC-based vars
    else:
        df = combinebySSC(loc, trend, ssctype, False, MTplus, DTplus, seasonal, yearAgg)
    
    p1, p2, p3 = [ttests.main(df[scenario][~np.isnan(df[scenario])].values, 
                              df['HIST'][~np.isnan(df['HIST'])].values) 
                              for scenario in ['MID', 'END4.5', 'END8.5']]
    #p1, p2, p3 = [p < 0.05 for p in ttest_pvalues]
    sig_dict = {'HIST': 1, 'MID': p1, 'END4.5': p2, 'END8.5': p3}
    return sig_dict
    

def ttest_SSCtable(loc='MSP', yearAgg=False):
    #df = pd.DataFrame(index=['HIST', 'MID', 'END4.5', 'END8.5'])
    statsdict = {}
    
    for trend in ['TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
        for ssctype in [99, 20, 10, 30, 50, 40, 60, 61]:
            rowname = '{0}_{1}'.format(trend, ssc_decode(ssctype))
            statsdict[rowname] = ttest_significance(loc, trend, ssctype, 1, 0, True, yearAgg)

    for ssctype in [20, 10, 30, 50, 40, 60, 61]:
        rowname = 'SSCFREQ_{}'.format(ssc_decode(ssctype))
        statsdict[rowname] = ttest_significance(loc, 'SSCFREQ', ssctype, 1, 0, True, yearAgg)
    
    for trend in ['SSC_runcount', 'SSC_rundur', 'SSC_rundurmax']:
        statsdict[trend] = ttest_significance(loc, trend, 60, 1, 0, True, yearAgg)
    
    df = pd.DataFrame.from_dict(statsdict, orient='index')
    
    if yearAgg:
        fileEnd = 'YearAgg'
    else:
        fileEnd = 'NonYearAgg'
    df.to_csv(os.path.join(stats_outpath, '{0}_compiled_SSC_ttests_{1}.csv'.format(loc, fileEnd)))
    
def ttest_HWtable(loc='MSP', yrAgg=False):
    statsdict = {}   
    for var in HWvar_list:
        rowname1 = var + '_from_hist_t95'
        statsdict[rowname1] = ttest_significance(loc, var, yearAgg=yrAgg, use_hist_t95=True)
        rowname2 = var + '_from_own_t95'
        statsdict[rowname2] = ttest_significance(loc, var, yearAgg=yrAgg, use_hist_t95=False)
    df = pd.DataFrame.from_dict(statsdict, orient='index')
    
    if yrAgg:
        fileEnd = 'YearAgg'
    else:
        fileEnd = 'NonYearAgg'
    df.to_csv(os.path.join(stats_outpath, '{0}_compiled_HW_ttests_{1}.csv'.format(loc, fileEnd)))
        
    
    

def boxplotDF(loc='MSP', trend='SSCFREQ', includeObsv=False, ssctypes=[99, 20, 30, 60, 61],
                 MTplus=1, DTplus=0, seasonal=True, yearAgg=True):
    if includeObsv:
        scenario_list = ['Station', 'HIST', 'MID', 'END4.5', 'END8.5']
    else:
        scenario_list = ['HIST', 'MID', 'END4.5', 'END8.5']
            
    if trend in SSCvar_list:
        if trend=='SSCFREQ':
            if 99 in ssctypes:
                ssctypes.remove(99) # that would just give full length of designated season!
        
        SSCdf_list = [combinebySSC(loc, trend, ssctype, includeObsv, 
                                   MTplus, DTplus, seasonal, yearAgg) for ssctype in ssctypes]        
        dfconcat = pd.concat(SSCdf_list)
                
        df = pd.melt(dfconcat, id_vars='SSC', value_vars=scenario_list, 
                     var_name='Scenario', value_name=trend)
        df['SSC'] = [ssc_decode(a) for a in df.SSC]
    
    
    elif trend in SSCrunvar_list:
        df = combine_forSSCruns(loc, trend, includeObsv, MTplus, DTplus,
                                    seasonal, yearAgg)
        df = pd.melt(df, id_vars='SSCRUN', value_vars=scenario_list,
                     var_name='Scenario', value_name=trend)
    
    elif trend in HWvar_list:
        HWdf_list = [combine_forHW(loc, trend, includeObsv, use_hist_t95, yearAgg)[0]
                     for use_hist_t95 in [True, False]]
        dfconcat = pd.concat(HWdf_list)
        
        df = pd.melt(dfconcat, id_vars='HIST_T95', value_vars=scenario_list,
                     var_name='Scenario', value_name=trend)
        id_dict = {True: 'From HIST $T_{95}$', False: 'From own $T_{95}$'}
        df['HIST_T95'] = [id_dict[a] for a in df.HIST_T95]
    
    return df, scenario_list

def DTboxplotDF(loc='MSP', trend='SSCFREQ', seasonal=True):
    DT_agg = combinebySSC('', loc, trend, 30, False, 1, 0, seasonal)
    DT_agg['SSC'] = 'DT (all)'    
    DT_split = [combinebySSC('', loc, trend, ssctype, False,
                             1, 2, seasonal) for ssctype in [30, 31, 32]]
    DT_split.insert(0, DT_agg)
    df = pd.concat(DT_split)
    scenario_list = list(scenario_dict.keys())
    
    df = pd.melt(df, id_vars='SSC', value_vars=scenario_list, 
                 var_name='Scenario', value_name=trend)
    df['SSC'] = [ssc_decode(a) if FloatConv(a) else a for a in df.SSC]
    return df, scenario_list  
    

def boxplot(loc='MSP', trend='SSCFREQ', includeObsv=False, ssctypes=[99, 20, 30, 60, 61],
                 MTplus=1, DTplus=0, seasonal=True, yearAgg=True, DTsubtypesonly=False,
                 savefig=True):  
    
    if DTsubtypesonly:
        df, scenario_list = DTboxplotDF(loc, trend, seasonal)
        plotcolors = [sscColors[a] for a in ['DT (all)', 'DT', 'DT+', 'DT++']]
        Hue = 'SSC'
    
    else:
        df, scenario_list = boxplotDF(loc, trend, includeObsv, ssctypes, 
                   MTplus, DTplus, seasonal, yearAgg)
        if trend in SSCvar_list:
            plotcolors = [sscColors[ssc_decode(a)] for a in ssctypes]
            Hue = 'SSC'
            varCategory = 'SSCvar'
        elif trend in SSCrunvar_list:
            plotcolors = ['mediumseagreen'] 
            Hue = 'SSCRUN' # doesn't do anything but adds consistency to formatting?
            varCategory = 'SSCrunvar'
        elif trend in HWvar_list:
            # plotcolors = [hwColors[a] for a in ['From HIST T95', 'From own T95']]
            plotcolors = [hwColors[a] for a in ['From HIST $T_{95}$', 'From own $T_{95}$']]
            Hue = 'HIST_T95'
            varCategory = 'HWvar'
    sns.set_palette(plotcolors)
    
    titledict = {'TMAX': '15:00 air temperature',
                 'TMIN': '3:00 air temperature', 
                 'TRANGE': '15:00 - 3:00 temperature range',
                 'D3': '3:00 dew point temperature',
                 'D15': '15:00 dew point temperature',
                 'SSCFREQ': 'Air mass frequency',
                 'SSC_runcount': '3+ day episodes of MT+/DT',
                 'SSC_rundur': 'Duration of 3+ day episodes of MT+/DT',
                 'HWssnlength': 'Heat wave season length',
                 'HWstartdate': 'Heat wave season starting date',
                 'HWintensity': 'Exceedance of $T_{95}$ on heat wave days',
                 'HWintensitySevere': 'Exceedance of $T_{95}$ on severe heat wave days',
                 'HWdur': 'Heat wave duration',
                 'HWfreq': 'Heat wave frequency',
                 'HWdaycount': 'Number of heat wave days',
                 'HWdaycountSevere': 'Number of severe heat wave days'
                 }
    
    # plot element width and legend bbox_to_anchor x-position (no legend for SSCrunvars)
    dims_dict = {'SSCvar': [0.8, 1.2], 'SSCrunvar': [0.3, 0], 'HWvar': [0.6, 1.0]}
    dims = dims_dict[varCategory]
   
    # make figure wider if including all SSC types rather than just the main 4 + overall (99)
    if len(ssctypes) > 5:
        fig, ax = plt.subplots(figsize=(12,6))
    else:
        fig, ax = plt.subplots(figsize=(8,6))
    
    ax = sns.boxplot(x='Scenario', y=trend, data=df, hue=Hue, 
                     linewidth=0.8, fliersize=2, whis=1.5, width=dims[0])
    
    if trend in SSCrunvar_list:
        ax.get_legend().remove()
    else:
        Handles, Labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        plt.legend(handles=Handles, labels=Labels, bbox_to_anchor=(dims[1],0.6), fontsize=11)
    
    titlestr = titledict[trend] + ', ' + loc_dict[loc]
    if seasonal and varCategory != 'HWvar':
        titlestr = titlestr + ' (May 20 - Sep 10)'
    ax.set_title(titlestr, fontsize=14)
    ax.set_xticklabels(scenario_list, fontsize=12)
    ax.set_xlabel('', fontsize=1)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.4, axis='y', linestyle='-')
    
    if trend in ['TMAX', 'TMIN', 'TRANGE', 'D3', 'D15', 'HWintensity', 'HWintensitySevere']:
        ax.set_ylabel('Temperature (C)', fontsize=12)
    elif trend in ['SSC_runcount', 'HWfreq']:
        ax.set_ylabel('Occurrences per season', fontsize=12)
    elif trend in ['SSCFREQ', 'HWdaycount', 'HWdaycountSevere']:
        ax.set_ylabel('Days per season', fontsize=12) #need to adjust if ever using seasonal=False
    elif trend == 'HWssnlength':
        ax.set_ylabel('Days per year', fontsize=12)
    elif trend == 'HWstartdate':
        ax.set_ylabel('Julian day', fontsize=12)
    elif trend in ['HWdur', 'SSC_rundur']:
        ax.set_ylabel('Days per occurrence', fontsize=12)
    
    fpath = fig_outpath
    if DTsubtypesonly:
        fpath = os.path.join(fpath, 'DT subsets')
    elif yearAgg:
        fpath = os.path.join(fpath, 'YearAgg')
    else:
        fpath = os.path.join(fpath, 'NonYearAgg')
    
    if len(ssctypes) > 5:
        fig_outpath_full = os.path.join(fpath, 'wide',
                                    '{0}_{1}_boxplotsWide.png'.format(loc, trend))
    else:
        fig_outpath_full = os.path.join(fpath,
                                    '{0}_{1}_boxplots.png'.format(loc, trend))  
    #plt.gcf().set_size_inches(9.5,6)
    if savefig:
        plt.savefig(fig_outpath_full, dpi=300, bbox_inches='tight')
        
def boxplotLoop(loc='MSP'):    
    for yrAgg in [False]:

        for var in ['TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
            boxplot(loc, var, False, [99, 20, 30, 60, 61], yearAgg=yrAgg)        
        boxplot(loc, 'SSCFREQ', False, [20, 30, 60, 61])
        
        for var in ['SSC_runcount', 'SSC_rundur']:
            boxplot(loc, var, False, [20, 30, 60, 61], yearAgg=yrAgg)

        for var in HWvar_sublist:
            boxplot(loc, var, False, yearAgg=yrAgg)

# "wide" boxplots: include all SSC types instead of the main 4 addressed in thesis.        
def boxplotWide_Loop(loc='MSP'):
    for var in ['TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
        boxplot(loc, var, False, [99, 20, 10, 30, 50, 40, 60, 61], yearAgg=False)        
    boxplot(loc, 'SSCFREQ', False, [20, 10, 30, 50, 40, 60, 61])


# outputs one row for a table, to be compiled below
def tabular_output(loc='MSP', var='SSCFREQ', ssctype=60, use_hist_t95=True, yrAgg=False):
    stats_dict = {}
    
    if var == 'T15':
        var= 'TMAX'
    elif var == 'T3':
        var = 'TMIN'   
    
    if var in SSCvar_list:
        df = combinebySSC(loc, var, ssctype, yearAgg=yrAgg)
    elif var in ['SSC_runcount', 'SSC_rundur']:
        df = combine_forSSCruns(loc, var, yearAgg=yrAgg)
    elif var in HWvar_sublist:
        df = combine_forHW(loc, var, False, use_hist_t95, yrAgg)[0]
    
    means = df.mean(axis=0, skipna=True)
    stdevs = df.std(axis=0, skipna=True)
    for S in scenario_dict.keys():
        stats_dict['{}_mean'.format(S)] = means[S]
        stats_dict['{}_stdev'.format(S)] = stdevs[S]
        
    # if var in HWvar_sublist:
    #     return stats_dict, T95_df, EHF85_df
    # else:
    return stats_dict

def tabular_SSCcompiled(loc='MSP', yearAgg=False):
    statsdict = {}
    
    for ssctype in [20, 10, 30, 50, 40, 60, 61]:
        rowname = 'SSCFREQ_{}'.format(ssc_decode(ssctype))
        statsdict[rowname] = tabular_output(loc, 'SSCFREQ', ssctype, True, yearAgg)
    
    for trend in ['T3', 'D3', 'T15', 'D15', 'TRANGE']:
        for ssctype in [99, 20, 10, 30, 50, 40, 60, 61]:
            rowname = '{0}_{1}'.format(trend, ssc_decode(ssctype))
            statsdict[rowname] = tabular_output(loc, trend, ssctype, True, yearAgg)
    
    for trend in ['SSC_runcount', 'SSC_rundur']:
        rowname = trend
        statsdict[rowname] = tabular_output(loc, trend, 60, True, yearAgg)
    
    df = pd.DataFrame.from_dict(statsdict, orient='index')
        
    if yearAgg:
        fileEnd = 'YearAgg'
    else:
        fileEnd = 'NonYearAgg'
    df.to_csv(os.path.join(stats_outpath, '{0}_SSCstats_{1}.csv'.format(loc, fileEnd)))

def tabular_HWcompiled(loc='MSP', yearAgg=False):
    statsdict = {}
    t95_dict = {True: 'from_HIST_T95', False: 'from_own_T95'}
    
    for trend in HWvar_sublist:
        for use_hist_t95 in [True, False]:
            rowname = '{0}_{1}'.format(trend, t95_dict[use_hist_t95])
            statsdict[rowname] = tabular_output(loc, trend, 60, use_hist_t95, yearAgg)
    
    dfHist, T95Hist, EHFHist = combine_forHW(loc, use_hist_t95=True)
    dfOwn, T95Own, EHFOwn = combine_forHW(loc, use_hist_t95=False)
    
    statsdict['T95'] = tabular_HWfooterstats(T95Own)
    statsdict['EHF85_from_HIST_T95'] = tabular_HWfooterstats(EHFHist)
    statsdict['EHF85_from_own_T95'] = tabular_HWfooterstats(EHFOwn)
    
    df = pd.DataFrame.from_dict(statsdict, orient='index')
        
    if yearAgg:
        fileEnd = 'YearAgg'
    else:
        fileEnd = 'NonYearAgg'
    df.to_csv(os.path.join(stats_outpath, '{0}_HWstats_{1}.csv'.format(loc, fileEnd)))
    
def tabular_HWfooterstats(df):
    stats_dict = {}
    means = df.mean(axis=0, skipna=True)
    stdevs = df.std(axis=0, skipna=True)
    for S in scenario_dict.keys():
        stats_dict['{}_mean'.format(S)] = means[S]
        stats_dict['{}_stdev'.format(S)] = stdevs[S]
        
    return stats_dict
    
          

    