import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import geopandas
from shapely.geometry import Point
import censusgeocode as cg
import datetime as dt


def pull_community(link):
    '''
    This function pulls in the community boundary area for Chicago.

    link: API link

    returns: dictionary with community number as key, community name as value
    '''

    response = requests.get(link)
    d = response.json()

    comm_dict = []

    for comm in d:
        k = comm['area_num_1']
        l = comm['community']
        new = {'community_area': k, 'name': l}
        comm_dict.append(new)

    return pd.DataFrame(comm_dict)

    
def pull_data(years, limit = 300000):
    '''
    This function pulls data from the api_link and organizes it into a pandas dataframe.

    years: list of years to pull crime data for
    limit: how many rows to pull 

    returns: list of dataframes, one for each year
    '''

    off = 0

    full = []

    for year in years:
        url = f'https://data.cityofchicago.org/resource/6zsd-86xi.json?year={year}&$limit={limit}'
        response = requests.get(url, timeout = 10)
        data = response.json()
        full.append(pd.DataFrame(data))

    return full

def avg_crime_nhood(*args):
    '''
    This function finds the mean number of crimes per community area.
    
    args: dataframes for each year 
    
    return: dataframe with community area and average crimes
    '''
    all_years = []
    
    for df in args:
        yr = df['year'].unique()[0]
        by_nhood = df.groupby(['community_area', 'primary_type']).count()[['id']]
        by_nhood = by_nhood.reset_index()
        col_name = f'{yr} Total'
        by_nhood.columns = ['community_area', 'Type', col_name]
        all_years.append(by_nhood)

    final_counts = pd.merge(all_years[0], all_years[1], on = ['community_area','Type'], how='outer')
    final_counts = final_counts.fillna(0)
    col1 = final_counts.columns[2]
    col2 = final_counts.columns[3]
    final_counts['Average'] = round((final_counts[col2]+final_counts[col1])/2,1)
    final_counts['Percent Change'] = round((final_counts[col2]/final_counts[col1] -1)*100,1)
    final_counts['Type']=final_counts['Type'].str.capitalize()
        
    return final_counts

def num_crimes_type(*args):
    '''
    This function calculates the total number of crimes committed across all 
    dataframes inputted.

    *args: dataframes containing crime records (each row is a crime)

    return: integer with number of crimes
    '''

    all_years = []

    for df in args:
        yr = df['year'].unique()[0]
        #print(yr)
        by_type = df.groupby('primary_type').count()[['id']]
        by_type = by_type.reset_index()
        col_name = f'{yr} Total'
        by_type.columns = ['Type', col_name]
        all_years.append(by_type)

    final_counts = pd.merge(all_years[0], all_years[1], on = 'Type')
    col1 = final_counts.columns[1]
    col2 = final_counts.columns[2]
    final_counts['Percent Change'] = round((final_counts[col2]/final_counts[col1]-1)*100,1)
    final_counts['Type']=final_counts['Type'].str.capitalize()

    return final_counts


def mk_table(data, save=False, filename = 'table.png', dpi = 800, scale=False, fontsize = False):
    '''
    This function makes a matplotlib table from a pandas dataframe.

    data: pandas dataframe
    filename: filename to save table png
    dpi = resolution

    '''
    plt.figure(figsize=(8,6))
    plt.axis('off')

    cell_text = []
    for row in range(len(data)):
        cell_text.append(data.iloc[row])

    tab = plt.table(cellText=cell_text, 
        colLabels=data.columns, 
        loc='center')

    if fontsize:
        tab.auto_set_font_size(False)
        tab.set_fontsize(5)
    if scale:
        tab.scale(1.25,3)
    
    if save:
        plt.savefig(filename, dpi = dpi, bbox_inces='tight')

def mk_bar(df, x_col, y_col, title, save=False, filename = 'bar.png', dpi = 500):
    '''
    This function makes a horizontal bar chart from a dataframe.

    df: dataframe
    x_col: name of column for the x-axis argument in the plot
    y_col: name of column for the y-axis argument in the plot
    filename: filename to save barchart png
    dpi = resolution

    return: None
    '''

    df = df.sort_values(y_col, ascending=True)
    df.plot.barh(x=x_col, y=y_col, legend = None)
    plt.title(title)
    plt.tick_params(axis='y', which='major', labelsize=6.5)
    if save:
        plt.savefig(filename, bbox_inches='tight', dpi=400)
    
def most_common_yr(df, year = 2018):
    '''
    This function builds a dataframe that finds the most common crime type for each community area and aggregates to be used
    in a pie chart.
    
    df: dataframe from avg_crime_nhood
    
    returns: dataframe
    '''
    
    dict_n = []
    tgt_col = f'{year} Total'

    for c in df['community_area'].unique():
        new = df[df['community_area']== c].sort_values(tgt_col, ascending=False)
        top = new.iloc[0]
        n = {'Type': top['Type'], tgt_col: top[tgt_col], 'area': top['community_area']}
        dict_n.append(n)
    
    return pd.DataFrame(dict_n)
    
    
def mk_pie(df, year, save=False, filename = 'pie.png', dpi=500):
    '''
    This function makes a pie chart from a given dataframe.
    
    df: dataframe from most_common_yr
    
    returns: None
    '''
    
    df.set_index('Type', inplace=True, drop=True)
    final = df.groupby('Type').count()
    
    tgt_col = f'{year} Total'
    
    final.plot.pie(y=tgt_col, legend=None,
                   autopct='%1.0f%%',pctdistance=0.75, 
                   title = f'Most Common Type of Crime in {year}\n(by Community Area)')
    plt.axis('off')
    if save:
        plt.savefig(filename, dpi=dpi)
    
def summary_nhood(df, area_num):
    '''
    This function creates a table summary of type of crime in 2017 and 2018 of a particular community area
    as well as a bar chart of the percent change.
    
    df: dataframe from avg_crime_nhood
    area_num: community_area number
    
    returns: None
    '''
    
    filtered = df[df['community_area'] == str(area_num)]
    filtered = filtered.set_index('community_area')
    
    f1 = f'nhood_{area_num}.png'
    mk_table(filtered, filename = f1)
    
    f2 = f'nhood_bar_{area_num}.png'
    title = f'Average Number of Crimes by Type (2017 to 2018)\nCommunity Area: {area_num}'
    mk_bar(filtered, 'Type', 'Average', title, f2)


def get_acs_blk_data(state, county, filename = 'data/census_data.csv'):
    '''
    This function retrieves block group data from the census_api and saves it to a csv.
    
    state: state code for api geographical hierarchy
    county: county code for api geographical hierarchy
    
    returns: dataframe with all block groups for the state and county
    '''

    all_tracts = f'https://api.census.gov/data/2016/acs/acs5?get=NAME&for=tract:*&in=state:{state}+county:{county}'
    
    response = requests.get(all_tracts)
    data = response.json()
    tracts_list = [x[-1] for x in data[1:]]

    full_census_data = 'empty'
    count=0
    for t in tracts_list:
        count +=1
        
        blkgrp = f'https://api.census.gov/data/2017/acs/acs5?get=NAME,B01002_001E,B19001_001E,B19001_002E,B19001_003E,B19001_004E,B19001_005E,B19001_006E,B19001_007E,B19001_008E,B19001_009E,B19001_010E,B19001_011E,B19001_012E,B19001_013E,B03002_012E,B03002_002E,B03002_003E,B03002_004E,B03002_006E&for=block+group:*&in=state:{state}+county:{county}+tract:{t}&key=48b45a2062a735ab2c6960def7d8cd8223041485'
        
        response = requests.get(blkgrp)
        data = response.json()
        

        df = pd.DataFrame(data)

        if type(full_census_data) == str:
            full_census_data = df
        else:
            full_census_data = pd.concat([full_census_data, df[1:]], ignore_index = True)

    full_census_data.to_csv(filename)
    return full_census_data

def find_tract_blk(lat, lng): 
    '''
    This function finds the census block info for a given latitude and longitude. Easy to use when only have one coordinate to check.
    
    lat: latitude
    lng: longitude
    
    return: FIPS census tract and block group code
    '''
    geo_info = cg.coordinates(x=lng,y=lat)
    block = geo_info['2010 Census Blocks'][0]['BLKGRP']

    tract = geo_info['2010 Census Blocks'][0]['TRACT']

    return tract+block



def ltlng_to_fips(df, geodf):
    '''
    This function uses spatial join to identify which the census blocks and tracts for data with latitude and longitude
    
    df: pandas dataframe with crime data
    geofile: geojson file with census block level data
    
    return: geodataframe 
    '''
    
    #add shapely Points to pandas dataframe
    df = df.dropna(subset=['latitude', 'longitude'])
    geometry = [Point(xy) for xy in zip(pd.to_numeric(df.longitude), pd.to_numeric(df.latitude))]
    df = df.drop(['longitude', 'latitude'], axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = geopandas.GeoDataFrame(df, crs=crs, geometry=geometry)
    
    #join
    return geopandas.sjoin(gdf, geodf, how="inner", op='intersects')
    
    
def multiple_joins(types, df_list, geofile = 'data/block_bounds.geojson'):
    '''This function returns a dictionary of multiple spatial joins
    
    types: list of types to filter by
    df_list: list of dfs to do the join on
    
    returns: dictionary
    
    '''
    
    final = {}
    
    #read in geojson
    geodf =geopandas.read_file(geofile)
    
    for t in types:
        #print(t)
        for df in df_list:
            t = t.upper()
            yr = df['year'].unique()[0]
            
            tmp = df[df['primary_type'] == t]
            
            
            joined = ltlng_to_fips(tmp, geodf)
            
            t = t.capitalize()
            key = f'{t} ({yr})'
            final[key] = joined
            
    return final

def build_summary_table(metrics, dict_df, census_df):
    '''
    This function builds a summary table of characteristics for each crime type in dict_df
    
    metrics: list of metrics for summary table
    dict_df: dictionary of df with census info for that crime (results of block_summary)
    census_df: census data dataframe
    
    returns: summary dataframe
    '''
    
    summary_table = pd.DataFrame()
    summary_table['Descriptive Metrics: Average'] = metrics
    
    for k, v in dict_df.items():
    
        blk_list = get_blk_list(v)
        blk_sum = block_summary(blk_list, census_df)
        
        values = get_metrics(blk_sum, metrics)
        
        summary_table[k] = values
        
    return summary_table

def get_metrics(df_block_info, metrics):
    '''
    Get summary metrics for summary table
    
    df_block_info: dataframe for crime type with block information
    metrics: columns to average
    
    return: list of values
    '''
    summary = []
    for m in metrics:
        val = round(df_block_info[m].mean(),2)
        if 'P' in m.upper():
            val = f'{val}%'
        summary.append(val)
        
    return summary

def get_blk_list(crime_df, q_thresh = 0.75):
    '''
    This function identifies the blocks that had a certain amount of crime (set by q_thresh)
    
    crime_df: df with specific type of crime (battery, homicide, etc) for one year
    q_thresh: quantile threshold to determine which blocks to look at
    
    return: list of block groups as strings
    '''
    
    crime_df['block'] = crime_df['tractce10'] + crime_df['blockce10'].str.slice(stop=1)
    by_block = crime_df.groupby('block').count()[['id']]
    q = by_block['id'].quantile(q_thresh)
    filtered = by_block[by_block['id'] > q]
    
    return list(filtered.index)
    
def block_summary(block_list, all_census):
    '''
    This function returns summary statistics for given blocks.
    
    block_list: list of blocks to summarize
    all_census: census dataframe
    
    return: dataframe with summary stats per block group
    '''
    summary = []
    
    for blk in block_list:
        tract = int(blk[0:-1])
        block = int(blk[-1])
        
        census_df = all_census[(all_census.tract == tract) & (all_census.block == block)]
        
        med_age = census_df['Median Age'].iloc[0]
        if med_age < 0:
            continue
        
        total_hh = census_df['Total Households'].iloc[0]
        if total_hh <= 0:
            hhinc_less_25 = np.NaN
            hhinc_25_50 = np.NaN
            hhinc_50_100 = np.NaN
        else:
            hhinc_less_25 = round((sum([census_df['Households < 10k'].iloc[0], census_df['10k < Household < 15k'].iloc[0], census_df['15k < Household < 20k'].iloc[0], census_df['20k < Household < 25k'].iloc[0]])/total_hh)*100,2)
            hhinc_25_50 = round((sum([census_df['25k < Household < 30k'].iloc[0], census_df['30k < Household < 35k'].iloc[0], census_df['35k < Household < 40k'].iloc[0], census_df['40k < Household < 45k'].iloc[0], census_df['45k < Household < 50k'].iloc[0]])/total_hh)*100,2)
            hhinc_50_100 = round((sum([census_df['50k < Household < 60k'].iloc[0], census_df['60k < Household < 75k'].iloc[0], census_df['75k < Household < 100k'].iloc[0]])/total_hh)*100,2)
            #hhinc_over_100 = round((1 - hhinc_less_25 - hhinc_25_50 - hhinc_50_100)*100,2)
        
        total_race = census_df['Total Hispanic'].iloc[0] + census_df['Total Not Hispanic'].iloc[0]
        pct_hisp = round((census_df['Total Hispanic'].iloc[0]/total_race)*100, 2)
        pct_white = round((census_df['Total White'].iloc[0]/total_race)*100, 2)
        pct_black = round((census_df['Total Black'].iloc[0]/total_race)*100, 2)
        pct_asian = round((census_df['Total Asian'].iloc[0]/total_race)*100, 2)
        
        
        tmp = {'Block': blk, 'Median Age': med_age, 'Pct household income < 25k': hhinc_less_25, '25k < Pct household income < 50k': hhinc_25_50,
              '50k < Pct household income < 100k': hhinc_50_100, 'Percent Hispanic': pct_hisp,
              'Percent White': pct_white, 'Percent Black': pct_black, 'Percent Asian': pct_asian}
        
        summary.append(tmp)
    
    return pd.DataFrame(summary)    


def compare_yr(df1, df2, y1, y2, wardnum, end1, end2, timedelta1, timedelta2, tgt_types, claims):
    '''
    This function returns the summary statistics for a given time period and ward.

    df1: year1 crime data
    df2: year2 crime data
    y1: year 1
    y2: year 2
    wardnum: ward number
    end1: end date for year 1
    end2: end date for year 2
    timedelta1: time delta to go back for year1
    timedelta2: time delta to go back for year2
    tgt_type: list of crime types
    claims: list of claimed changes

    returns: dataframe
    '''
    wardnum = str(wardnum)
    
    ward_43_17 = df1[df1['ward']== wardnum]
    ward_43_17.loc[:, 'date'] = ward_43_17['date'].apply(lambda x: pd.to_datetime(x))

    ward_43_18 = df2[df2['ward']==wardnum]
    ward_43_18.loc[:, 'date'] = ward_43_18['date'].apply(lambda x: pd.to_datetime(x))

    start2 = end2 - timedelta2
    start1 = end1 - timedelta1

    filter_17 = ward_43_17[(ward_43_17['date'] >= start1) & (ward_43_17['date'] <= end1)]
    filter_18 = ward_43_18[(ward_43_18['date'] >= start2) & (ward_43_18['date'] <= end2)]

    totl_17 = filter_17.shape[0]
    totl_18 = filter_18.shape[0]

    total = pd.DataFrame([['All Types', totl_17,totl_18]], columns = ['primary_type',f'{y1} Period Total', f'{y2} Period Total'])
    
    filter_17 = filter_17[filter_17['primary_type'].isin(tgt_types)]
    filter_18 = filter_18[filter_18['primary_type'].isin(tgt_types)]

    count_17 = filter_17.groupby('primary_type').count()[['id']]
    count_17.columns = [f'{y1} Period Total']
    count_18 = filter_18.groupby('primary_type').count()[['id']]
    count_18.columns = [f'{y2} Period Total']
    both = pd.merge(count_17, count_18, on='primary_type')
    both = both.reset_index()
    both = total.append(both, ignore_index = True)


    both['Pct Change'] = round((both[f'{y2} Period Total']/both[f'{y1} Period Total']-1)*100,2)
    both['Claimed Pct Change'] = claims

    both.columns = ['Category',f'{y1} Period Total', f'{y2} Period Total', 'Pct Change', 'Claimed Pct Change']

    return both

def nhood_count(df_summary, community_name, tgt_col, crime_type, linkage_url = 'https://data.cityofchicago.org/resource/igwz-8jzy.json'):
    '''
    This function aggregates across column names for a given community.
    
    df_summary: dataframe summary of crime type by neighborhood
    community_name: neighborhood name
    tgt_col: columns to sum over
    crime_type: crime type 
    
    returns: integer value
    '''
    
    nhood_dict = pull_community(linkage_url)
    
    crime = crime_type.capitalize()
    name = community_name.upper()
    nhood_num = list(nhood_dict[nhood_dict['name'].str.contains(name)]['community_area'])
    
    totl = 0
    
    for i in nhood_num:
        df = df_summary[df_summary['community_area'] == i]
        
        df_type = df[df['Type'] == crime]
        
        for j in tgt_col:
            totl += df_type.iloc[0][j]
            
    return totl

def address_info(block_address, df):
    '''
    This function provides summary of crime by type for a particular address for all years in the given dataframe.
    
    block_address: block address
    df: dataframe with all years of data to be examined

    returns: dataframe
    '''

    filter_address = df[df['block']==block_address]

    total_add = filter_address.shape[0]
    count_add = filter_address.groupby('primary_type').count()[['id']]
    count_add['percent'] = round((count_add.id/total_add)*100,2)
    count_add = count_add.reset_index()
    count_add.columns = ['Type', 'Frequency', 'Percent of Total']

    probs=count_add.sort_values('Percent of Total', ascending=False)
    return probs
    

def ytd_ward_df(df, wardnum, end):
    '''
    This function filters a dataframe by ward number and dates (beginning of year to end date)
    
    df: dataframe
    wardnum: ward number
    end: end date for YTD
    
    return: dataframe
    '''
    
    wardnum = str(wardnum)
    filtered = df[df['ward'] == wardnum]
    filtered.loc[:, 'date'] = filtered['date'].apply(lambda x: pd.to_datetime(x))
    
    return filtered[filtered['date'] <= end]


