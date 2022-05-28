'''
Create maps to show air mass distributions more easily.
'''


import numpy as np
import pandas as pd 
import datetime as dt
import os
import geopandas
import matplotlib.pyplot as plt
#import Ngl
import plotly


colNames = ['Yesterday SSC', 'Yesterday AM', 'Yesterday PM', 'Today SSC', 'Today AM', 'Today PM', 'Tomorrow SSC', 'Tomorrow AM', 'Tomorrow PM']
SSCdict = {20:'DP', 10:'DM', 30:'DT', 50:'MP', 40:'MM', 60:'MT', 61:'M+', 70:'TR'}
SSClist = list(SSCdict.values())

sscfullnames = {'DP': 'Dry Polar',
                'DM': 'Dry Moderate', 
                'DT': 'Dry Tropical',
                'MP': 'Moist Polar',
                'MM': 'Moist Moderate',
                'MT': 'Moist Tropical',
                'MT+': 'Moist Tropical Plus',
                'TR': 'Transition'}

stations = '/Users/birke111/Documents/ssc/daily/ssc_stations.csv'
stationsDF = pd.read_csv(stations, index_col=0)

#most available: day BEFORE date given
def Input(date='2021-06-05'):
    df = pd.read_csv(date + '.csv', index_col=0)
    df.columns = colNames
    df = df[df.index.notnull()]
    return df


#day = 0 to 2 (yesterday to tomorrow- yesterday data generally more comprehensive, no forecast)
#col = SSC, AM, PM
#temp = T (temp) or D (dewpt)
def extract(data, Loc='MN: Minneapolis-St. Paul', day=0, col='SSC', temp='T'):

    df = data

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


# loc arg is temporary, will go thru all locs in loop
def Compile(Loc='MN: Minneapolis-St. Paul', date='2021-06-05'):
    
    sampleIndex = Input('2021-03-01').index # to copy indices from a known file where needed
    
    dateformat = '%Y-%m-%d'
    today = dt.datetime.strptime(date, dateformat) 
    dayBefore = dt.datetime.strftime((today - dt.timedelta(days=1)), dateformat)
    dayAfter  = dt.datetime.strftime((today + dt.timedelta(days=1)), dateformat)
    
    # can these be condensed into a loop?
    try:
        dayAfterDF = Input(dayAfter).loc[:,colNames[0:3]]
    except FileNotFoundError:
        dayAfterDF = pd.DataFrame([], index=sampleIndex, columns=colNames[0:3])    
    #dayafterBool = os.path.isfile(dayAfter + '.csv')
    try:
        dayOfDF = Input(dayAfter).loc[:,colNames[3:6]]
    except FileNotFoundError:
        dayOfDF = pd.DataFrame([], index=sampleIndex, columns=colNames[3:6])
    try:
        dayBeforeDF = Input(dayAfter).loc[:,colNames[6:9]]
    except FileNotFoundError:
        dayBeforeDF = pd.DataFrame([], index=sampleIndex, columns=colNames[6:9])
    
    fullDF = pd.concat([dayAfterDF, dayOfDF, dayBeforeDF], axis=1)
    #fullDF = dayAfterDF
    #fullDF.append([dayOfDF, dayBeforeDF], axis=1)
    return fullDF.loc[Loc]
    #return dayAfterDF
'''   
doesn't work (yet) if dayAfter file is missing, ok if other 2 are??
ignoring this for now though, want to map today
    
    read in entirety of all 3 (day after, of, before)
    
    make it all NaNs if file doesn't exist but leave in same column format
        then can ignore all NaNs equally?
    
    import just the relevant columns
        then I have all 3 together, can see how consistent they stayed!
        cols for each (so this x3): loc name, ssc, amtemp, amdewpt, pmtemp, pmdewpt
            day after first (0), then day of second (1), then day prev third (2)
            3d dataframe? can these cols be nested within "main" cols 
                        for each date? pull more easily that way?
    
    make a new df (cols: same list as above but x1)
    for each location: (can i avoid a for loop? maybe not...)
        attach associated lat/lon
        do basically extract (above), start w "day after" file if possible          
'''

def TodayMap(date='2021-06-14'):
    DF = Input(date).loc[::,colNames[3:6]]
    
    df = stationsDF.dropna()
    df['SSC'] = [extract(DF, Loc, 1, 'SSC') for Loc in df.index]
    
    
    sscColors = {'DP':'wheat', 'DM':'sandybrown', 'DT':'orangered', \
                'MP':'paleturquoise', 'MM':'mediumturquoise', 'MT':'mediumseagreen', \
                'M+':'darkgreen', 'TR':'grey', np.nan:'white'}
    df['colors'] = [sscColors[a] for a in df['SSC']]
    '''
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
    
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    ax = world[world.continent == 'North America'].plot(
        color='white', edgecolor='black')
    
    #gdf['color'] = ['#C62828', '#C62828', '#283593', '#FF9800', '#283593']
    
    gdf.plot(ax=ax, color='red')
    
    plt.show()
    '''
    
    

    #fig = plotly.express.scatter_geo(lat=df['lat'], lon=df['lon'], color=df['colors'], projection='natural earth')
    go = plotly.graph_objects
    
    fig = go.Figure()
    
    for SSC in SSClist:
        if SSC == 'M+':
            ssc = 'MT+'
        else:
            ssc = SSC
        
        DF = df[df['SSC'] == SSC]
        fig.add_trace(go.Scattergeo( 
            name = sscfullnames[ssc],            
            lat = DF['lat'], 
            lon = DF['lon'], 
            text = DF.index,
            #mode = 'markers', 
            marker = dict(color = sscColors[SSC], size=15),
            ))
    

    
    fig.update_layout(geo_scope='usa', title='{} Air Masses (SSC2)'.format(date), showlegend=True)
    
    plotly.offline.plot(fig, filename='maps/sscmap-{}.html'.format(date), auto_open=True)


    
    
    
    
    