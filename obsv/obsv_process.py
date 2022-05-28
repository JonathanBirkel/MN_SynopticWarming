"""
A less messy recreation of previous daily/hourly historical processing - 
    more easily transferrable to model operations!
    daily and hourly go together, no messy separate scripts
    join ssc .txt files to observations - eliminate need for manual cleanup
    
automate cleanup:
    fill in gaps - Jordan is missing entire 2004-05
        but also single-day missing values
        and bad formats (number with letter included? non int convertable, etc)
    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from pathlib import Path
import datetime as dt
#import statsmodels.api as sm
#import pylab

loc_dict = {'MSP':'Minneapolis', 'JORM5':'Jordan',
            'RST':'Rochester', 'GMDM5':'Grand Meadow',
            'DLH':'Duluth', 'TOHM5':'Two Harbors',
            #'STC':'St. Cloud', 'MLCM5':'Milaca',
            'FAR':'Fargo', 'ADAM5':'Ada'}

#inputs
obsvInputPath = '/Users/birke111/Documents/ssc/obsv_new'
sscCalendarPath = '/Users/birke111/Documents/ssc/SSC_calendars'

#outputs
figurePath = '/Users/birke111/Documents/ssc/obsv_new/result_figures4'
tabularPath = '/Users/birke111/Documents/ssc/obsv_new/result_tabular5'

tminShift = False

def FloatConv(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False 
 
def TypeAgg(code, keep_MTplus=True):   # group 31,32 with 30 etc.       
    ssctype = int(code)
    if ssctype in range(61,70) and keep_MTplus:
        rounddown = 61
    else:
        rounddown = int(ssctype/10) * 10
    return rounddown

def ssc_decode(ssctype):
    typeagg = TypeAgg(ssctype, False)
    code_dict = {0:'', 10:'DM', 20:'DP', 30:'DT', 40:'MM', 50:'MP', 60:'MT', 70:'TR'}
    
    if ssctype == 99:
        return 'All'
    else:
        decode = code_dict[typeagg]
        pluses = ssctype-typeagg
        for p in range(pluses):
            decode = decode + '+'
        return decode
  
def PivotFill(pivot, yrInit, yrEnd):  # fill in missing years in time series
    for i in range(yrInit, yrEnd):
        if i not in pivot.index.values:
            pivot.loc[i] = [np.nan] * pivot.shape[1]
    pivot = pivot.sort_index()
    return pivot

def inDateRange(dt_input, mInit=5, dInit=20, mEnd=9, dEnd=10): # dt_input in datetime format    
    # default: May20-Sep10 inclusive   
    init = dt.datetime(dt_input.year, mInit, dInit, 0, 0, 0)
    end = dt.datetime(dt_input.year, mEnd, dEnd, 23, 59, 59)
    
    return (dt_input >= init and dt_input <= end)

# should work for either hourly or daily dataframe inputs
def seasonalDF(df, mInit=5, dInit=20, mEnd=9, dEnd=10):
    df['InRange'] = [inDateRange(a, mInit, dInit, mEnd, dEnd) for a in df['DateTime']]    
    sub_df = df[ df['InRange'] == True]  
    sub_df = sub_df.drop(columns='InRange')
    
    return sub_df


def InputDaily(loc='MSP'):
    fullpath = os.path.join(obsvInputPath, 'daily/{}.csv'.format(loc))
    df = pd.read_csv(fullpath, header=3, # index_col='Date', 
                     usecols=['Date', 'TMAX', 'TMIN'], 
                     skipfooter=17, engine='python')
    df['DateTime'] = pd.to_datetime(df.Date)
    df['Year'] = df.DateTime.dt.year
    #df['JulianDay'] = [a.timetuple().tm_yday for a in df.DateTime]
    
    df['TMAX'] = [(float(a) if FloatConv(a) else np.nan) for a in df.TMAX] 
    df['TMIN'] = [(float(a) if FloatConv(a) else np.nan) for a in df.TMIN]    
    df.TMAX, df.TMIN = (df.TMAX-32)/1.8, (df.TMIN-32)/1.8 
    
    if tminShift:
        df['TMIN'] = df.TMIN.shift(-1)
        df['TRANGE'] = df.TMAX - df.TMIN
    else:
        #df['TRANGE'] = df.TMAX - df.TMIN.shift(-1)
        df['TRANGE'] = df.TMAX - df.TMIN
    
    #df.set_index('DateTime', inplace=True)    
    return df.drop(columns='Date')


def InputHourly(loc='MSP', var='TEMP'):  
    path = os.path.join(obsvInputPath, 'hourly', var)
    
    yrInit_dict = {'MSP':1945,
                   'RST':1948,
                   'DLH':1948,
                   #'STC':1948,
                   'FAR':1947}
    # hourly data was subdivided into 3 intervals to avoid download failure  
    # intervals arbitrary but consistent across locations, aside from start year
    file1 = os.path.join(path, '{0}_{1}-1979.csv'.format(loc, yrInit_dict[loc]))
    file2 = os.path.join(path, '{}_1980-2000.csv'.format(loc))
    file3 = os.path.join(path, '{}_2001-2020.csv'.format(loc))
    file_list = [file1, file2, file3]
    
    var_dict = {'TEMP':'Temp (F)', 'DEWPT':'Dewpt (F)'}
    Var = var_dict[var]
    
    df = pd.DataFrame(columns=['Date', 'Time', Var])
    for file in file_list:
        df_subset = pd.read_csv(file, header=4, 
                                usecols=['Date', 'Time', Var],
                                skipfooter=15, engine='python')
        # python engine is slower but needed for skipfooter-
        #   unless there's another way around dealing with that?
        df = df.append(df_subset)
    df = df.reset_index(drop=True)
    
    df['DateTime'] = pd.to_datetime(df.Date)
    #df['Hour'] = pd.to_datetime(df.Time).round('H').dt.hour
    df['Time'] = pd.to_datetime(df.Time).dt.time

    df['DateTime'] = [pd.Timestamp.combine(df.DateTime[a], df.Time[a]).round('H')
                      for a in range(len(df))]    
    df['Date'] = [a.floor('D') for a in df.DateTime]
    #df['DateTime'] = [a.floor('D') for a in df.DateTime]
    #df['Year'] = df.DateTime.dt.year
    df['Hour'] = df.DateTime.dt.hour
    #df['JulianDay'] = [a.timetuple().tm_yday for a in df.DateTime]
    
    df[var] = [(float(a) if FloatConv(a) else np.nan) for a in df[Var]]   
    df[var] = (df[var] - 32) / 1.8
    return df.drop(columns=['Time', Var])

'''
InputHourly is very slow, though that probably could be improved - 
this is the point at which we decide whether to rerun it

DF will either be string 'refresh' or a pre-existing dataframe from InputHourly
'''
def HourlySummaries(DF, loc='MSP', var='TEMP', multiTypes=False, hrshift=-9):
    if type(DF) == str:
        if DF == 'refresh':
            df = InputHourly(loc, var)
        else:
            print('Invalid input')
    else:
        df = DF.copy(deep=True)
    
    df.rename(columns={'Date':'DateTime', 'DateTime':'DateTimeFull'}, inplace=True)
    df.drop_duplicates(subset='DateTimeFull', keep='last', inplace=True)
    
    df_3a3p = df[df.Hour.isin([3,15])]
    pivot = df_3a3p.pivot(index='DateTime', columns='Hour').loc[:,var]
    #pivot = df_3pivot['TEMP']
    pivot['Year'] = pivot.index.year
    
    if var == 'TEMP':
        pivot.columns = ['T3', 'T15', 'Year']
        
        if multiTypes: # for comparisons of different forms of tmin/tmax
            # otherwise just returns the 3a/3p temps
        
            df['TEMP_shift'] = df.TEMP.shift(hrshift)   
            df_tmax = df.groupby('DateTime').agg('max')[['TEMP','TEMP_shift']]
            df_tmax.columns = ['TMAX', 'TMAX_shift']
            df_tmin = df.groupby('DateTime').agg('min')[['TEMP','TEMP_shift']]
            df_tmin.columns = ['TMIN', 'TMIN_shift']
            
            dfjoin = df_tmax.merge(df_tmin, 'left', left_on='DateTime', right_on='DateTime')
            dfjoin = dfjoin.merge(pivot, 'left', left_on='DateTime', right_on='DateTime')
            df_out = dfjoin.dropna()     
        else:
            df_out = pivot
            df_out.columns = ['TMIN', 'TMAX', 'Year']
                        
            # 1. what I had before, shifting trange only
            #df_out['TRANGE'] = df_out.TMAX - df_out.TMIN.shift(-1)
            
            # 2. keeping everything within same calendar day
            #    (going with this option for simplicty, SOME improved results if not all)
            df_out['TRANGE'] = df_out.TMAX - df_out.TMIN
            
            # 3. shifting tmin AND trange
            #df_out['TMIN'] = df_out.TMIN.shift(-1)
            #df_out['TRANGE'] = df_out.TMAX - df_out.TMIN
            
    
    elif var == 'DEWPT':
        pivot.columns = ['D3', 'D15', 'Year']
        df_out = pivot
    
    return df_out.reset_index()


def tempcomparison(DF, loc='MSP', seasonal=True, temptype='TMAX', tempx='BASIC', tempy='EHF', EHFroll=True):
    df = DF.copy(deep=True)
    # df = HourlySummaries(loc, False)
    
    df.columns = ['TMAX_BASIC', 'TMAX_EHF', 'TMIN_BASIC', 'TMIN_EHF', 'TMIN_SSC', 'TMAX_SSC']
    df['DateTime'] = df.index 
    
    #shift EHF tmins back forward to next day (equivalent to shifting others backward)
    if EHFroll:
        df['TMIN_EHF'] = df.TMIN_EHF.shift(1)
    
    daily = InputDaily(loc)
    daily = daily[['DateTime', 'TMAX', 'TMIN']]
    daily.columns = ['DateTime', 'TMAX_DAILY', 'TMIN_DAILY']
    
    df = df.merge(daily, 'left', left_on='DateTime', right_on='DateTime')    
    if seasonal:
        df = seasonalDF(df, 5, 20, 9, 10)
    
    xlabel = '{0}_{1}'.format(temptype,tempx)
    ylabel = '{0}_{1}'.format(temptype,tempy)
    
    X = df[xlabel].values
    Y = df[ylabel].values
    mask = ~np.isnan(X) & ~np.isnan(Y)
    x = X[mask]
    y = Y[mask]
    
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x,y)
    rsquared = rvalue**2
    linreg_y = slope*x + intercept
    
    plt.figure(figsize=(6,6))
    plt.scatter(x,y,s=15,c='black')
    plt.plot(x, linreg_y, c='red')
    
    if seasonal:
        titlestr = '{} summer temperatures'.format(loc)
    else:
        titlestr = '{} all-year temperatures'.format(loc)
    
    plt.title(titlestr, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.5, linestyle='-')
    plt.figtext(0.05,0.00,
                'y = ' + str(round(slope,5)) + '*x + ' + str(round(intercept,5)) +
                '\nr^2 = ' + str(round(rsquared,5)),
                fontsize=10)
    
# how much of the time is offset TMIN occurring after midnight??? (summer-only or all-year)
# alternately: how much of the time is TMAX occurring between 9a-11p? (probably most)
def tempAgreement(DF, loc='MSP', seasonal=True, temptype='TMIN', hrshift=-9):
    df = HourlySummaries(DF, loc, 'TEMP', True, hrshift)
    
    df.columns = ['DateTime', 'TMAX_BASIC', 'TMAX_EHF', 'TMIN_BASIC', 
                  'TMIN_EHF', 'TMIN_SSC', 'TMAX_SSC', 'Year']
    #df['DateTime'] = df.index 
    #shift EHF tmins back forward to next day (equivalent to shifting others backward)
    df['TMIN_EHF'] = df.TMIN_EHF.shift(1)      
    if seasonal:
        df = seasonalDF(df, 5, 20, 9, 10)
        
    df['test'] = (df['{}_BASIC'.format(temptype)] == df['{}_EHF'.format(temptype)])
    return np.mean(df.test)

def tempAgreementSummary(DF_list, locs=['MSP', 'RST', 'DLH', 'FAR'], hrshift=-9):
    df = pd.DataFrame(index=['TMIN_summer', 'TMIN_allyear', 'TMAX_summer', 'TMAX_allyear'])
    for i in range(len(locs)):
        DF = DF_list[i]
        loc = locs[i]
        df[loc] = [tempAgreement(DF, loc, True, 'TMIN', hrshift), 
                   tempAgreement(DF, loc, False, 'TMIN', hrshift),
                   tempAgreement(DF, loc, True, 'TMAX', hrshift), 
                   tempAgreement(DF, loc, False, 'TMAX', hrshift)]
    df['mean'] = df.mean(axis=1)
    return df

def obsv_checkmissing(DF, loc='MSP', inputType='daily', var='TEMP', seasonal=True):
    if inputType == 'daily':
        df = InputDaily(loc)
        df['NaNcheck'] = df.TMAX - df.TMIN
        #   returns nan if either is nan - actual numbers don't matter
        col = 'NaNcheck'
    elif inputType == 'hourly':
        df = HourlySummaries(DF, loc, var, False) # so won't check for missing values at other times
        if var == 'TEMP':
            col = 'TRANGE' # catch if either TMAX or TMIN is missing this way
        elif var == 'DEWPT':
            df['DRANGE'] = df.D15 - df.D3 # again, just to catch if either is missing
            col = 'DRANGE'
    
    if seasonal:
        df = seasonalDF(df)
    
    # total NaNs (not by year), may not need
    #series = df[col]    
    #missingCount = series.isna().sum()
    #missingPct = 100 * (missingCount / len(series))
    
    # count NaNs by year
    yrGroup = df.groupby('Year')
    missingSeries = yrGroup.agg('size') - yrGroup.agg('count')[col]   
    
    PctMissingSeries = (missingSeries / yrGroup.agg('size')) * 100
    PctMissingSeries[PctMissingSeries == np.inf] = 100    
    return PctMissingSeries

def flagYears(DF, loc='MSP', inputType='daily', var='TEMP', seasonal=True, pct_threshold=20):
    series = obsv_checkmissing(DF, loc, inputType, var, seasonal)
    years = series[series >= pct_threshold].index.values
    return years
    




def SSCinput(loc='MSP'):
    txtfile = os.path.join(sscCalendarPath, '{}.txt'.format(loc))
    df = pd.read_csv(txtfile, sep=' ', header=None, names=['Station', 'Date', 'SSC'])
    
    df.index = pd.to_datetime(df.Date, format='%Y%m%d')    
    df['SSC'] = [(a if a in range(10,80) else 0) for a in df.SSC]
    
    return df.drop(columns=['Station', 'Date'])

  
def SSCjoin(DF, loc='MSP', inputType='daily', var='TEMP'): # join SSCs with main input (daily or hourly) by date    
    locpair_dict = {'MSP':'MSP', 'JORM5':'MSP',
                    'RST':'RST', 'GMDM5':'RST',
                    'DLH':'DLH', 'TOHM5':'DLH',
                    #'STC':'STC', 'MLCM5':'STC',
                    'FAR':'FAR', 'ADAM5':'FAR',
                    }
    ssc_df = SSCinput(locpair_dict[loc]) # rural sites don't have their own SSC
    
    if inputType == 'daily':
        obsv_df = InputDaily(loc)
        df = obsv_df.merge(ssc_df, how='left', left_on='DateTime', right_index=True)
    elif inputType == 'hourly':
        #obsv_df = InputHourly(loc)
        obsv_df = HourlySummaries(DF, loc, var, False)
        df = obsv_df.merge(ssc_df, how='left', left_on='DateTime', right_index=True)
    return df
    

def SSCtrends(DF, loc='MSP', trend='SSCFREQ', inputType='daily', yrInit=1948, yrEnd=2020, 
              ssctype=60, typeAgg=True, keep_MTplus=True, seasonal=True,
              customSeasonal=False, customDates=[6,1,6,30]):
    if trend in ['D3', 'D15']:
        var = 'DEWPT'
    else:
        var = 'TEMP'                     
    df = SSCjoin(DF, loc, inputType, var)
    
    if customSeasonal:
        df = seasonalDF(df, customDates[0], customDates[1], customDates[2], customDates[3]) 
    elif seasonal:
        df = seasonalDF(df, 5, 20, 9, 10)
    df['SSC'] = df.SSC.fillna(0)
    if typeAgg:
        df['SSC'] = [TypeAgg(a, keep_MTplus) for a in df.SSC]
    
    if trend == 'SSCFREQ':
        pivot = df.pivot_table(index='Year', columns='SSC', values='DateTime',
                               aggfunc='count', fill_value=0, margins=False)
        timeseries = pivot.loc[yrInit:yrEnd-1,ssctype]
        
    elif trend in ['TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
        if ssctype==99:
            pivot = df.pivot_table(index='Year', values=trend, 
                                   aggfunc='mean', fill_value=None, margins=False)
            pivot = PivotFill(pivot,yrInit,yrEnd)
            timeseries = pivot.loc[yrInit:yrEnd-1,trend]
        else:
            pivot = df.pivot_table(index='Year', columns='SSC', values=trend, 
                                   aggfunc='mean', fill_value=None, margins=False)
            pivot = PivotFill(pivot,yrInit,yrEnd)
            timeseries = pivot.loc[yrInit:yrEnd-1,ssctype]
    
    return timeseries
       
   
def SSCruns(loc='MSP', opp_types = [30,61], minrun=3, seasonal=True):
    df = SSCjoin('', loc, inputType='daily')
    df['SSC'] = df.SSC.fillna(0)
    df['SSC'] = [TypeAgg(a, True) for a in df.SSC]
    
    df['OppDay'] = [a in opp_types for a in df.SSC]
    
    # "old" method - mark start of consec runs of any length (non overlapping) with # days in it   
    df['Runstart'] = 0
    df['InRun'] = False
    for i in range(1, len(df)):
        runlength = 0
        # new addition: counting runs that start just before May 20, but including just the "summer" days
        # this makes more difference in models, where long runs are more frequent 
        # but Fargo misses a long one (May 1980) without this adjustment
        if seasonal:
            is_seasonstart = inDateRange(df.DateTime[i], 5, 20, 5, 20)
        else:
            is_seasonstart = False
        possible_startday = not df.OppDay[i-1] or is_seasonstart
        #if df.OppDay[i] and not df.OppDay[i-1]:
        if df.OppDay[i] and possible_startday:
            runlength = 1
            for j in range(1,20):
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
                df.InRun[m+n]=True   
                
    # "new" method - mark start of any TDP, doesn't matter if overlapping
    # is this what Vanos et al did? (update: seems maybe not?)   
    df['TDPstart'] = [all(df.OppDay[a:a+minrun]) for a in range(len(df))]
    
    return df

def SSCruntrends(loc='MSP', yrInit=1948, yrEnd=2020, seasonal=True, 
                 opp_types=[30,61], minrun=3, overlap=False, stat='SSC_runcount'):
    
    # adding new stat options, was only freq
    # now durmean and durmax too
    df = SSCruns(loc, opp_types, minrun, seasonal)
    if seasonal:
        df = seasonalDF(df, 5, 20, 9, 10)
        #df = seasonalDF(df, 6, 1, 8, 31)
    if overlap:
        yearlydf = df.groupby('Year').agg('sum')
        return yearlydf.loc[yrInit:yrEnd-1, 'TDPstart']
    else:
        if stat == 'SSC_runcount':
            df['Runstart'] = df.Runstart > 2
            yearlydf = df.groupby('Year').agg('sum')
        elif stat == 'SSC_rundurmean':
            df.loc[(df.Runstart < 3), 'Runstart'] = np.nan
            yearlydf = df.groupby('Year').agg('mean')
        elif stat == 'SSC_rundurmax':
            df.loc[(df.Runstart < 3), 'Runstart'] = np.nan
            yearlydf = df.groupby('Year').agg('max')
        return yearlydf.loc[yrInit:yrEnd-1, 'Runstart']
    
         
def EHF_heatwaves(DF, loc='MSP', pct_threshold=95, days_in_pd=3, temptype='TMEAN', each=False):

    '''
    This is set up in accordance with Nairn-Fawcett, except for usage of 3a/3p temps
    as tmin/tmax, rather than using actual tmin/tmax within each 9a-9a period.
    
    Other args that could be tweaked if not adhering to EHF parameters:                    
    temptype: TMEAN (as in EHF), TMAX or TMIN (self explanatory) - which to use for threshold
    each: do ALL (3) days have to exceed threshold individually, 
        or just the average across the (3)
    '''
    
    #df = HourlySummaries(loc, False)
    df = HourlySummaries(DF, loc, 'TEMP', False)
    
    df['TMEAN'] = (df.TMAX + df.TMIN.shift(-1)) / 2
    T95 = np.nanpercentile(df[temptype], pct_threshold)
    
    df['PeriodAvg'] = [np.mean(df[temptype][i:i+days_in_pd]) for i in range(len(df))]
    # could apply a shift if wanting to assign to 2nd or 3rd day of period instead of 1st
    df['AcclPeriodAvg'] = [np.mean(df[temptype][i-30:i]) for i in range(len(df))]

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
            for j in range(1,40):
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
    #df['TDPstart'] = df.EHF > 0 # gives a simple annual count of TDPs
    
    df = df[['DateTime', 'TMEAN', 'EHIsig', 'EHIaccl', 'EHF', 'HeatWaveDay', 
             'SevereHeatWaveDay', 'Runstart', 'RunstartSevere']]
    return df, T95, EHF85
  

def heatwaveSSC(DF, loc='MSP', severe=False, as_pct=False, 
                   pct_threshold=95, days_in_pd=3):
    #trends: HWssnlength, HWstartdate, HWintensity(severe), HHWdur, HWfreq
    
    df, T95, EHF85 = EHF_heatwaves(DF, loc, pct_threshold, days_in_pd, 'TMEAN', False)
    #df, T95, EHF85 = EHF_heatwaves(loc, pct_threshold, days_in_pd, 'TMEAN', False)
    
    ssc_df = SSCinput(loc)    
    df = df.merge(ssc_df, how='left', left_on='DateTime', right_on='Date')   
    df = df[df.HeatWaveDay == True] 
    df['SSC'] = [ssc_decode(TypeAgg(a, keep_MTplus=True)) for a in df.SSC] #.fillna(0)]
    
    if severe:
        df = df[df.SevereHeatWaveDay == True]
    
    SSCcounts = df['SSC'].value_counts()
    if as_pct:
        SSCcounts = (SSCcounts / sum(SSCcounts)) * 100
    return SSCcounts

def heatwaveSSC_tabular(DF_list): 
    #DF_list = a list of the four DF inputs (from InputHourly) that would be used individually
    locs = ['MSP', 'RST', 'DLH', 'FAR']
    df = pd.DataFrame()
    for i in range(4):
        counts = heatwaveSSC(DF_list[i], locs[i], False, False, 95, 3)
        counts_pct = (counts / sum(counts)) * 100
        countsSevere = heatwaveSSC(DF_list[i], locs[i], True, False, 95, 3)
        countsSevere_pct = (countsSevere / sum(countsSevere)) * 100
        df['{}_counts'.format(locs[i])] = counts
        df['{}_counts_pct'.format(locs[i])] = counts_pct
        df['{}_countsSevere'.format(locs[i])] = countsSevere
        df['{}_countsSevere_pct'.format(locs[i])] = countsSevere_pct
    df = df.T
    outpath = os.path.join(tabularPath, 'heatwaveSSC.csv')
    df.to_csv(outpath)

def heatwavetrends(DF, trend='HWssnlength', loc='MSP', yrInit=1948, yrEnd=2020, 
                   pct_threshold=95, days_in_pd=3):
    #trends: HWssnlength, HWstartdate, HWintensity, HWintensitySevere,  
    #        HWdur, HWfreq, HW_TDPcount, HW_EHFmax,
    #        HWdurmax, HWdurmaxSevere, HWdaycount, HWdaycountSevere, 
    #        HW_runcount5, HW_runcount7
    df, T95, EHF85 = EHF_heatwaves(DF, loc, pct_threshold, days_in_pd, 'TMEAN', False)
    #df, T95, EHF85 = EHF_heatwaves(loc, pct_threshold, days_in_pd, 'TMEAN', False)
    
    df['JulianDay'] = [a.timetuple().tm_yday for a in df.DateTime]
    df['Year'] = df.DateTime.dt.year  
    # apparently only need .dt accessor if timestamps are a series rather than index
   
    df = df[df.HeatWaveDay == True]
    if trend == 'HWssnlength':
        dfgroup = df[['Year', 'JulianDay']].groupby('Year').agg(['max','min'])
        dfgroup.columns = ['Jmax', 'Jmin']
        dfgroup[trend] = dfgroup.Jmax - dfgroup.Jmin + 1
    elif trend == 'HWstartdate':
        dfgroup = df[['Year', 'JulianDay']].groupby('Year').agg(['min'])
        dfgroup.columns = [trend]
    elif trend == 'HWintensity':
        # mean exceedance of T95 on hw days- some years might not at all?
        # so really just yearly mean positive EHIsig - a more direct/intuitive metric than EHF
        dfgroup = df[['Year', 'TMEAN']].groupby('Year').agg('mean') - T95
        dfgroup.columns = [trend]
    elif trend == 'HWintensitySevere':
        DF = df.copy(deep=True)
        DF = DF[DF.SevereHeatWaveDay == True]
        dfgroup = DF[['Year', 'TMEAN']].groupby('Year').agg('mean') - T95
        dfgroup.columns = [trend]
    elif trend == 'HWdur':
        DF = df.copy(deep=True)
        DF = DF[DF.Runstart > 0]
        dfgroup = DF[['Year', 'Runstart']].groupby('Year').agg('mean')
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
        # but taking the quick and dirty approach right now
           
    dfgroup = PivotFill(dfgroup, yrInit, yrEnd)  
    timeseries = dfgroup.loc[yrInit:yrEnd-1,trend]
    
    if trend in ['HWssnlength', 'HWfreq', 'HW_TDPcount', 
                 'HWdaycount', 'HWdaycountSevere', 'HW_runcount5', 'HW_runcount7']: 
    #       I think these are the only ones where replacing NaN with 0 makes sense?
        timeseries = timeseries.fillna(0)  
    
    return timeseries, T95, EHF85


###############################################################################
        

def plot(DF, loc='MSP', trend='SSCFREQ', inputType='daily', yrInit=1948, yrEnd=2020, 
              ssctype=60, typeAgg=True, keep_MTplus=True, seasonal=True,
              opp_types=[30,61], minrun=3, missing_pct=10,
              pct_threshold=95, days_in_pd=3,
              savefig=True):
    if trend in ['D3', 'D15']:
        var = 'DEWPT'
    else:
        var = 'TEMP' 
    
    if trend in ['SSCFREQ', 'TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
        series = SSCtrends(DF, loc, trend, inputType, yrInit, yrEnd,
                           ssctype, typeAgg, keep_MTplus, seasonal).values     
    elif trend == 'SSC_TDPcount':
        series = SSCruntrends(loc, yrInit, yrEnd, seasonal, 
                              opp_types, minrun, True).values
    
    elif trend in ['SSC_runcount', 'SSC_rundurmean', 'SSC_rundurmax']:
        series = SSCruntrends(loc, yrInit, yrEnd, seasonal, 
                              opp_types, minrun, False, trend).values
        
    elif trend in ['HWssnlength', 'HWstartdate', 'HWintensity', 'HWintensitySevere', 
                   'HWdur', 'HWfreq', 'HW_TDPcount', 'HW_EHFmax', 'HW_EHFmaxdate',
                   'HWdurmax', 'HWdurmaxSevere', 'HWdaycount', 'HWdaycountSevere', 
                   'HW_runcount5', 'HW_runcount7']:
        series = heatwavetrends(DF, trend, loc, yrInit, yrEnd, 
                                pct_threshold, days_in_pd)[0].values
         
    else:
        print('Invalid input')
    
    X = np.array(range(yrInit,yrEnd))
    mask = ~np.isnan(X) & ~np.isnan(series)
    x = X[mask]
    y = series[mask]
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x,y)
    rsquared = rvalue**2
    linreg_y = slope*x + intercept
    
        
    titledict = {#'TMAX': 'Daily highs',
                 #'TMIN': 'Daily lows', 
                 #'TRANGE': 'Diurnal temperature range',
                 'SSCFREQ': 'Air mass frequency',
                 'HWssnlength': 'Heat wave season length',
                 'HWstartdate': 'Heat wave season starting date',
                 'HWintensity': 'Heat wave exceedance of $T_{95}$',
                 'HWintensitySevere': 'Severe heat wave exceedance of $T_{95}$',
                 'HWdur': 'Heat wave duration',
                 'HWfreq': 'Heat wave frequency',
                 'HW_TDPcount': '3-day periods with positive EHF',
                 'HW_EHFmax': 'Annual maximum EHF',
                 'HW_EHFmaxdate': 'Date of annual maximum EHF',
                 'HWdurmax': 'Maximum heat wave duration', 
                 'HWdurmaxSevere': 'Maximum consecutive severe heat wave days',
                 'HWdaycount': 'Heat wave days', 
                 'HWdaycountSevere': 'Severe heat wave days', 
                 'HW_runcount5': '5+ day heat waves', 
                 'HW_runcount7': '7+ day heat waves',
                 'D3': '3am dew points',
                 'D15': '3pm dew points',
                 }
    if inputType == 'daily':
        titledict['TMAX'], titledict['TMIN'], titledict['TRANGE'] = \
            'Daily highs', 'Daily lows', 'Diurnal temperature range'
    elif inputType == 'hourly':
        titledict['TMAX'], titledict['TMIN'], titledict['TRANGE'] = \
            '3pm temperatures', '3am temperatures', '3pm-3am temperature range'
        
    
    if trend=='SSC_TDPcount':
        # for now: assuming inputs are either [30,60,61] or [30,61]
        # might also try for just 60?
        if 60 in opp_types:
            titlestr='{0}-day episodes of MT/MT+/DT ({1})'.format(minrun, loc_dict[loc])
        else:
            titlestr='{0}-day episodes of MT+/DT ({1})'.format(minrun, loc_dict[loc])
    elif trend in ['SSC_runcount', 'SSC_rundurmean', 'SSC_rundurmax']:
        # for now: assuming inputs are either [30,60,61] or [30,61]
        # might also try for just 60?
        SSCrun_dict = {'SSC_runcount': '', 
                            'SSC_rundurmean': 'Mean duration, ', 
                            'SSC_rundurmax': 'Max duration, '}
        if 60 in opp_types:
            titlestr='{0}{1}+ day episodes of MT/MT+/DT ({2})'.format(SSCrun_dict[trend], 
                                                                      minrun, loc_dict[loc])
        else:
            titlestr = '{0}{1}+ day episodes of MT+/DT ({2})'.format(SSCrun_dict[trend], 
                                                                     minrun, loc_dict[loc])
        
    elif trend in ['TMAX', 'TMIN', 'TRANGE', 'SSCFREQ', 'D3', 'D15']:
        titlestr=titledict[trend]+' ({0} days, {1})'.format(ssc_decode(ssctype), loc_dict[loc])
    else: 
        titlestr=titledict[trend]+' ({})'.format(loc_dict[loc])
        
    # plot flagged years (> _% missing data) in gray    
    FlagYears = flagYears(DF, loc, inputType, var, seasonal, missing_pct)
    FlagSeries = [year in FlagYears for year in x]
    ColorSeries = ['darkgray' if a else 'black' for a in FlagSeries]
    
    plt.figure(figsize=(7, 6))
    plt.scatter(x,y,s=25,c=ColorSeries)
    plt.plot(x, linreg_y, c='red') 
    plt.title(titlestr, fontsize=14)
    plt.xlabel('Year', fontsize=12)
    if trend in ['TMAX', 'TMIN', 'TRANGE', 'D3', 'D15','HWintensity', 'HWintensitySevere']:
        plt.ylabel('Temperature (C)', fontsize=12)
    elif trend in ['SSC_TDPcount', 'SSC_runcount', 'HWfreq', 'HW_TDPcount', 
                   'HW_runcount5', 'HW_runcount7']:
        plt.ylabel('Occurrences per season', fontsize=12)
    elif trend in ['SSCFREQ', 'HWdaycount', 'HWdaycountSevere']:
        plt.ylabel('Days per season', fontsize=12)
    elif trend == 'HWssnlength':
        plt.ylabel('Days per year', fontsize=12)
    elif trend in ['HWstartdate', 'HW_EHFmaxdate']:
        plt.ylabel('Julian day', fontsize=12)
    elif trend in ['HWdur', 'HWdurmax', 'HWdurmaxSevere', 'SSC_rundurmean', 'SSC_rundurmax']:
        plt.ylabel('Days per occurrence', fontsize=12)
    elif trend == 'HW_EHFmax':
        plt.ylabel('Excess Heat Factor', fontsize=12)
    
    
    #set as needed
    #plt.ylim(15,33)
    
    plt.grid(alpha=0.5, linestyle='-')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.figtext(0.08, 0.02,
                'y = ' + str(round(slope,5)) + '*x + ' + str(round(intercept,3)) + 
                '\nP = ' + str(round(pvalue,5)) + 
                '\t$R^{2}$ = ' + str(round(rsquared,5)),
                fontsize=10)                
    if savefig:
        if trend in ['SSCFREQ', 'TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
            fig_outpath = os.path.join(figurePath, trend, loc, ssc_decode(ssctype))
            if trend in ['TMAX', 'TMIN', 'TRANGE']:
                fig_outpath = os.path.join(fig_outpath, inputType)
        else:
            fig_outpath = os.path.join(figurePath, trend, loc)
        Path(fig_outpath).mkdir(parents=True, exist_ok=True)
        filestr = '{0}-{1}.png'.format(yrInit, yrEnd-1)
        
        plt.gcf().set_size_inches(7,6)
        # this is supposed to make fig sizes consistent but doesn't seem to work yet
        # probably isn't counting frame space etc
        
        plt.savefig(os.path.join(fig_outpath, filestr), dpi=120)


def tabular_output(DF, loc='MSP', trend='SSCFREQ', inputType='daily', yrInit=1948, yrEnd=2020, 
              ssctype=60, typeAgg=True, keep_MTplus=True, seasonal=True,
              opp_types=[30,61], minrun=3, missing_pct=10,
              pct_threshold=95, days_in_pd=3, 
              customSeasonal=False, month='Jun',
              savetab=True):
    if trend in ['D3', 'D15']:
        var = 'DEWPT'
    else:
        var = 'TEMP'   
    
    dates_dict = {'Jun': [6,1,6,30], 'Jul': [7,1,7,31], 'Aug': [8,1,8,31]}
    if trend in ['SSCFREQ', 'TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
        series = SSCtrends(DF, loc, trend, inputType, yrInit, yrEnd,
                           ssctype, typeAgg, keep_MTplus, seasonal,
                           customSeasonal, dates_dict[month])
        T95, EHF85 = 0, 0
    elif trend == 'SSC_TDPcount':
        series = SSCruntrends(loc, yrInit, yrEnd, seasonal, 
                              opp_types, minrun, True)
        T95, EHF85 = 0, 0
    elif trend in ['SSC_runcount', 'SSC_rundurmean', 'SSC_rundurmax']:
        series = SSCruntrends(loc, yrInit, yrEnd, seasonal, 
                              opp_types, minrun, False, trend)
        T95, EHF85 = 0, 0
        
    
    elif trend in ['HWssnlength', 'HWstartdate', 'HWintensity', 'HWintensitySevere', 
                   'HWdur', 'HWfreq', 'HW_TDPcount', 'HW_EHFmax', 'HW_EHFmaxdate',
                   'HWdurmax', 'HWdurmaxSevere', 'HWdaycount', 'HWdaycountSevere', 
                   'HW_runcount5', 'HW_runcount7']:
        series, T95, EHF85 = heatwavetrends(DF, trend, loc, yrInit, yrEnd, 
                                            pct_threshold, days_in_pd)
    df = pd.DataFrame(series)
    df.columns = [trend]
   
    FlagYears = flagYears(DF, loc, inputType, var, seasonal, missing_pct)
    df['{}pct_missing'.format(missing_pct)] = [year in FlagYears for year in df.index]
    
    if savetab:
        if trend in ['SSCFREQ', 'TMAX', 'TMIN', 'TRANGE', 'D3', 'D15']:
            tab_outpath = os.path.join(tabularPath, trend, loc, ssc_decode(ssctype))
            
            if customSeasonal:
                tab_outpath = os.path.join(tab_outpath, month)
            elif trend in ['TMAX', 'TMIN', 'TRANGE']:
                tab_outpath = os.path.join(tab_outpath, inputType)
        else:
            tab_outpath = os.path.join(tabularPath, trend, loc)
        Path(tab_outpath).mkdir(parents=True, exist_ok=True)
        filestr = '{0}-{1}.csv'.format(yrInit, yrEnd-1)
        df.to_csv(os.path.join(tab_outpath, filestr))
    
    X = np.array(range(yrInit,yrEnd))
    mask = ~np.isnan(X) & ~np.isnan(series)
    x = X[mask]
    y = series[mask]
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x,y)
    rsquared = rvalue**2
    #linreg_y = slope*x + intercept
    
    stats_dict = {'slope': slope, 'intercept': intercept, 
                  'rsquared': rsquared, 'pvalue': pvalue, 
                  'stderr': stderr, 'mean': np.mean(y),
                  'T95': T95, 'EHF85': EHF85} 
    # T95, EHF85 will be the same for all HW rows but seems easiest way to handle them
    return stats_dict

def outputLoop_forModelAgreement(DF_temp, DF_dewpt, loc='MSP'):
    for trend in ['TMAX', 'TMIN']:
        for Month in ['Jun', 'Jul', 'Aug']:
            for ssctype in [99]:
                tabular_output(DF_temp, loc, trend, 'hourly', 1980, 2000, ssctype,
                               customSeasonal=True, month=Month)
    for trend in ['D3', 'D15']:
        for Month in ['Jun', 'Jul', 'Aug']:
            for ssctype in [99]:
                tabular_output(DF_dewpt, loc, trend, 'hourly', 1980, 2000, ssctype,
                               customSeasonal=True, month=Month)
    #for ssctype in [20, 30, 60, 61, 70]:
    for ssctype in [10, 20, 30, 40, 50, 60, 61, 70]:
        tabular_output('', loc, 'SSCFREQ', 'daily', 1980, 2000, ssctype)




def outputLoop(df_T, df_D, loc='MSP', yrInit=1948, yrEnd=2020):
    # trendslist = ['TMAX', 'TMIN', 'TRANGE', 'D3', 'D15', 'SSCFREQ', 'SSC_TDPcount',
    #               'HWssnlength', 'HWstartdate', 'HWintensity', 'HWintensitySevere', 
    #               'HWdur', 'HWfreq', 'HW_TDPcount', 'HW_EHFmax', 'HW_EHFmaxdate']    
    statsdict = {}
    
    # SSC stats (urban)
    if loc in ['MSP', 'RST', 'DLH', 'FAR']:
        for trend in ['TMAX', 'TMIN', 'TRANGE']:
            for ssctype in [20, 30, 60, 61, 99]:
                for inputType in ['daily', 'hourly']:
                    plot(df_T, loc, trend, inputType, yrInit, yrEnd, ssctype)
                    rowname = '{0}_{1}_{2}'.format(trend, ssc_decode(ssctype), inputType)
                    statsdict[rowname] = tabular_output(df_T, loc, trend, inputType, 
                                                        yrInit, yrEnd, ssctype)
        for trend in ['D3', 'D15']:
            for ssctype in [20, 30, 60, 61, 99]:
                plot(df_D, loc, trend, 'hourly', yrInit, yrEnd, ssctype)
                rowname = '{0}_{1}'.format(trend, ssc_decode(ssctype))
                statsdict[rowname] = tabular_output(df_D, loc, trend, 'hourly', 
                                                    yrInit, yrEnd, ssctype)
        for ssctype in [20, 30, 60, 61]:
            plot(df_T, loc, 'SSCFREQ', 'daily', yrInit, yrEnd, ssctype)
            rowname = 'SSCFREQ_{}'.format(ssc_decode(ssctype))
            statsdict[rowname] = tabular_output(df_T, loc, 'SSCFREQ', 'daily', 
                                                yrInit, yrEnd, ssctype) 
        
        for trend in ['SSC_TDPcount', 'SSC_runcount', 'SSC_rundurmean', 'SSC_rundurmax']:
            plot(df_T, loc, trend, 'daily', yrInit, yrEnd)
            rowname = trend
            statsdict[rowname] = tabular_output(df_T, loc, trend, 'daily', 
                                                  yrInit, yrEnd)                   
    
        # heat wave stats
        for trend in ['HWssnlength', 'HWstartdate', 'HWintensity', 'HWintensitySevere', 
                      'HWdur', 'HWfreq', 'HW_TDPcount', 'HW_EHFmax', 'HW_EHFmaxdate',
                      'HWdurmax', 'HWdurmaxSevere', 'HWdaycount', 'HWdaycountSevere', 
                      'HW_runcount5', 'HW_runcount7']:
            plot(df_T, loc, trend, 'hourly', yrInit, yrEnd)
            rowname = trend
            statsdict[rowname] = tabular_output(df_T, loc, trend, 'hourly', 
                                                yrInit, yrEnd) 
            
    # SSC stats (rural)
    elif loc in ['JORM5', 'GMDM5', 'TOHM5', 'ADAM5']:
        for trend in ['TMAX', 'TMIN', 'TRANGE']:
            for ssctype in [20, 30, 60, 61, 99]:
                plot(df_T, loc, trend, 'daily', yrInit, yrEnd, ssctype)
                rowname = '{0}_{1}'.format(trend, ssc_decode(ssctype))
                statsdict[rowname] = tabular_output(df_T, loc, trend, 'daily', 
                                                    yrInit, yrEnd, ssctype)
                
    df = pd.DataFrame.from_dict(statsdict, orient='index')
    df.to_csv(os.path.join(tabularPath, '{}_compiledstats.csv'.format(loc)))
            
def multiLocLoop(yrInit=1948, yrEnd=2020):
    for loc in loc_dict.keys():
        if loc in ['MSP', 'RST', 'DLH', 'FAR']:
            df_T = InputHourly(loc, 'TEMP')
            df_D = InputHourly(loc, 'DEWPT')
        else:
            df_T, df_D = '', ''
        outputLoop(df_T, df_D, loc, yrInit, yrEnd)

        
# for after that's been run! pull variables across locations for side-by side comparison
def compiledUrbanRural():
    tempstats = ['TMAX', 'TMIN', 'TRANGE']
    SSClist = ['DP', 'DT', 'MT', 'MT+', 'All']
    # cols_list = ['slope', 'intercept', 'rsquared', 'pvalue',
    #              'stderr', 'mean', 'T95', 'EHF85']
    
    df = pd.DataFrame() #columns=cols_list)

    for loc in loc_dict.keys():
        tabular_inpath = os.path.join(tabularPath, loc + '_compiledstats.csv')
        df_in = pd.read_csv(tabular_inpath, index_col=0)
      
        if loc in ['MSP', 'RST', 'DLH', 'FAR']:
            dailytemp_index = ['{0}_{1}_daily'.format(t, ssc) for t in tempstats for ssc in SSClist]
        else:
            dailytemp_index = ['{0}_{1}'.format(t, ssc) for t in tempstats for ssc in SSClist]
        df_in = df_in.loc[dailytemp_index]
        #df_in.index = ['{}_'.format(loc) + a for a in df_in.index]
        df_in['station'] = loc
        df = df.append(df_in)

    #df.drop(columns = ['T95', 'EHF85'], inplace=True)
    df['linreg_start'] = (df.slope * 1948) + df.intercept
    df['linreg_end'] = (df.slope * 2019) + df.intercept    
    df['linreg_change'] = df.linreg_end - df.linreg_start
    df = df[['station', 'slope', 'rsquared', 'pvalue', 'stderr', 'mean', 
               'linreg_start', 'linreg_end', 'linreg_change']]
    df.index.rename('tempstat', inplace=True)
    outpath = os.path.join(tabularPath, 'compiledUrbanRural.csv')
    df.to_csv(outpath)
    
def SSCfreqplots_other(loc='MSP'):
    for ssctype in [10, 40, 50, 70]:
        plot('', loc, 'SSCFREQ', 'daily', 1948, 2020, 
              ssctype, savefig=True)
        