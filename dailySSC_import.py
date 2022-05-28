'''
Attempting to pull directly from online SSC table to create a map-applicable dataframe.

'''

import numpy as np
import pandas as pd 
from datetime import date
import os
import dailySSC_map as Map

path = 'http://sheridan.geog.kent.edu/ssc/today.html'
#path = '030221.html'

colNames = ['Yesterday SSC', 'Yesterday AM', 'Yesterday PM', 'Today SSC', 'Today AM', 'Today PM', 'Tomorrow SSC', 'Tomorrow AM', 'Tomorrow PM']
SSCdict = {10:'DM', 20:'DP', 30:'DT', 70:'TR', 40:'MM', 50:'MP', 60:'MT', 61:'M+'}
SSClist = list(SSCdict.values())

df = pd.read_html(path, header=0, index_col=0, flavor=None)[0]
df = df.drop( columns=['Unnamed: 4','Unnamed: 8','Unnamed: 12'], axis=1 )


#AM/PM are 900Z/2100Z
#AM and PM refer to morning (4am Eastern/1am Pacific) and afternoon (4pm/1pm) temperatures and dew points in °C.

#locDict = { 'DLH':130, 'MSP':132, 'RST':133 }

dateStr = date.today().isoformat()
#dateStr= '2021-03-02'


def makeCSV():
    outpath = dateStr + '.csv'
    if os.path.exists(outpath):
       print('Error: output file already exists')
    else:
        df.to_csv(outpath)


# MAKE THIS A SEPARATE .PY FILE?
        
#day = 0 to 2 (yesterday to tomorrow- yesterday data generally more comprehensive, no forecast)
#col = SSC, AM, PM
#temp = T (temp) or D (dewpt)
def extract(Loc='MN: Minneapolis-St. Paul', day=0, col='SSC', temp='T'):
    
    df.columns = colNames
    Days = ['Yesterday', 'Today', 'Tomorrow']
    colStr = Days[day] + ' ' + col
    DFcell = df.loc[Loc, colStr]
    
    if type(DFcell) != str:
        return np.nan
    
    if col=='SSC':
        SSCtype = DFcell[0:2]
        if SSCtype not in SSClist:
            SSCtype = np.nan
        return SSCtype    
    else:
        if temp=='T':
            Temp = DFcell.split()[0]
        #split function gives 3 values, middle one is '/'
        if temp=='D':
            Temp = DFcell.split()[2]
        return Temp
    
    
def today():
    makeCSV()
    Map.TodayMap(dateStr)
    