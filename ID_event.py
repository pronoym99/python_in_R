import pandas as pd
import numpy as np
import sys
# import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
from haversine import haversine, Unit
from haversine import haversine_vector, Unit
from tqdm import tqdm
# from time import sleep
# import pyarrow.feather as feather
from pathlib import Path
# import pytz
import pyreadr
import time
from time import time as t
from itertools import combinations, permutations
# import math
# import os
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 150)
from multiprocess import cpu_count
# import gspread

def categorize_shift(hour: int) -> str:
    if 6 <= hour < 14:
        return 'A'
    elif 14 <= hour < 22:
        return 'B'
    return 'C'

def cons_id_grouping(list_):
    buckets = []
    current_bucket = []

    for index, value in enumerate(list_):
        if index == 0 or value != list_[index - 1]:
            if current_bucket:
                buckets.append(current_bucket)
            current_bucket = []
        current_bucket.append(index)

    # Add the last bucket if there's one
    if current_bucket:
        buckets.append(current_bucket)
    return buckets

def get_shift_timestamp(date_str):
    datetime_input = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    input_time = datetime_input.time()

    if input_time >= datetime.strptime('00:00:00', '%H:%M:%S').time() and input_time < datetime.strptime('06:00:00', '%H:%M:%S').time():
        shift_time = datetime_input.replace(hour=6, minute=0, second=0, microsecond=0)
    elif input_time >= datetime.strptime('06:00:00', '%H:%M:%S').time() and input_time < datetime.strptime('14:00:00', '%H:%M:%S').time():
        shift_time = datetime_input.replace(hour=14, minute=0, second=0, microsecond=0)
    elif input_time >= datetime.strptime('14:00:00', '%H:%M:%S').time() and input_time < datetime.strptime('22:00:00', '%H:%M:%S').time():
        shift_time = datetime_input.replace(hour=22, minute=0, second=0, microsecond=0)
    else:
        shift_time = (datetime_input + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)

    return shift_time

def row_split(start_time,end_time):
    end = str(get_shift_timestamp(start_time))
    start_list=[];end_list=[]
    while pd.to_datetime(end)<pd.to_datetime(end_time):
        start_list.append((start_time,end))
        start_time = end
        end = str(get_shift_timestamp(start_time))
    else:
        start_list.append((start_time,end_time))
    return start_list

def ign_time_cst(a,b): # output -> final ign time for each event
    # a = ignstatus column ;  b = consecutive Time difference column
    s_t=time.time()
    buckets = []
    start_index = None
    for i, value in enumerate(a):
        if value == 1:
            if start_index is None:
                start_index = i
        elif start_index is not None:
            buckets.append((start_index, i - 1))
            start_index = None
    if start_index is not None:
        buckets.append((start_index, len(a) - 1))
    ign_time=0
    for j in buckets:
        s = sum(b[(j[0]+1):(j[1]+1)])
        try:
            s = s+(b[j[0]]/5)+(b[j[1]+1]/2.5)
        except:
            s=s+(b[j[0]]/5)
        ign_time=ign_time+s
#     print(f'my ign_time_cst function execution time:{time.time()-s_t}s')
    return ign_time


def select_ign_time(row):     # row wise , Returns row
    if ((row['ign_time_igndata']/row['total_time'])*100 == 100)or((row['ign_time_igndata']/row['total_time'])*100 == 0):
        return row['ign_time_cst']
    else:
        return row['ign_time_igndata']

def Off_On_grouping(indicator):
    buckets = []
    start_position = None
    for i, value in enumerate(indicator):
        if value == 'end':
            if start_position is not None:
                end_position = i
                buckets.append((start_position, end_position))
            start_position = i
        elif value == 'strt':
            if start_position is not None:
                end_position = i
                buckets.append((start_position, end_position))
                start_position = None
    if start_position is not None:
        buckets.append((start_position, len(indicator)))
    return buckets


def From_Togrouping(indicator,from_,to_):
    buckets = []
    start_position = None
    for i, value in enumerate(indicator):
        if value == from_:
            if start_position is not None:
                end_position = i
                buckets.append((start_position, end_position))
            start_position = i
        elif value == to_:
            if start_position is not None:
                end_position = i
                buckets.append((start_position, end_position))
                start_position = None
    if start_position is not None:
        buckets.append((start_position, len(indicator)))
    return buckets

binary_to_id = {(1, 1, 1): 'id1',(0, 1, 0): 'id2',(1, 0, 1): 'id3',(0, 1, 1): 'id4',(1, 0, 0): 'id5',
(0, 0, 1): 'id6',(1, 1, 0): 'id7',(0, 0, 0): 'id8'}
def map_binary_to_id(row):
    return binary_to_id[tuple(row)]
def id_attachment(df):
    selected_columns = ['currentIgn', 'veh_movement_status', 'fuel_movement_status']
    df['ID_status'] = df[selected_columns].apply(map_binary_to_id, axis=1)
    return df

def event_creation(df):
    temp_dict={}
    df['new_time_diff'] = df['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
    df.loc[0,'con_cum_distance']=0    #Distance
    ign_cst = ign_time_cst(df['currentIgn'].tolist(),df['new_time_diff'].tolist())
    keys=['termid','regNumb','start_time','end_time','total_obs','initial_level','end_level','max_time_gap','ign_cst','total_dist','ID_status']
    values=[df.head(1)['termid'].item(),df.head(1)['regNumb'].item(),df.head(1)['ts'].item(),df.tail(1)['ts'].item(),len(df),
           df.head(1)['currentFuelVolumeTank1'].item(),df.tail(1)['currentFuelVolumeTank1'].item(),df['new_time_diff'].max(),
           ign_cst,df['con_cum_distance'].sum(),df.tail(1)['ID_status'].item()]     #Distance
    temp_dict.update(zip(keys,values))
    return temp_dict

def ign_exist(termid):
    veh_df = new_cst_1[new_cst_1['termid']==termid]
    veh_df.reset_index(drop=True,inplace=True)
    veh_df['ts'] = pd.to_datetime(veh_df['ts'])
    groups = From_Togrouping(veh_df['Indicator'].tolist(),'strt','end')
    for i in groups:
        veh_df.loc[i[0]:i[-1],'currentIgn']=1
    reverse_groups = From_Togrouping(veh_df['Indicator'].tolist(),'end','strt')
    for i in reverse_groups:
        veh_df.loc[i[0]+1:i[-1]-1,'currentIgn']=0
    if groups[0][0]!=0:
        veh_df.loc[:groups[0][0]-1,'currentIgn']=0
    combined=[]
    for i in range(len(groups)):
        combined.append(groups[i])
        combined.append(reverse_groups[i])
    combined.insert(0,(0,groups[0][0]))
    if combined[0][0]==combined[0][1]==0:
        combined.pop(0)
    final_term_df=pd.DataFrame()
    for i in combined:
        sample = veh_df.loc[i[0]:i[-1]]
        sample.reset_index(drop=True,inplace=True)
        indicator = sample.head(1)['Indicator'].item()
        last_ind = sample.tail(1)['Indicator'].item()
        if last_ind =='strt':
            sample.loc[sample.index[-1],'currentIgn']=0
        start_time=sample.head(1)['ts'].item()
        end_time=sample.tail(1)['ts'].item()
        sample_list = row_split(str(start_time),str(end_time))
        l=[]
        for k in range(len(sample_list)):
            sample2=sample[(sample['ts']>=pd.to_datetime(sample_list[k][0]))&(sample['ts']<=pd.to_datetime(sample_list[k][1]))]
            sample2.reset_index(drop=True,inplace=True)
            if (len(sample2)==0):
                temp_dict={}
                temp_dict['termid']=[termid];temp_dict['regNumb']=[sample.head(1)['regNumb'].item()]
                temp_dict['start_time']=[sample_list[k][0]];temp_dict['end_time']=[sample_list[k][1]]
                temp_dict['max_time_gap']=[(pd.to_datetime(sample_list[k][1])-pd.to_datetime(sample_list[k][0])).total_seconds()/60]
                temp_dict['dl_status']= ['Data_Loss']
                shift_df=pd.DataFrame(temp_dict)
            else:
#                 keys=['termid','regNumb','start_time','end_time','initial_level','end_level']
                sample2['Time_diff'] = sample2['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
                sample2['con_cum_distance'] = sample2['cum_distance'].diff().fillna(0)
                sample2['Cons_Speed'] = sample2['con_cum_distance']/sample2['Time_diff']     #Distance
                sample2['Cons_Speed'] = sample2['Cons_Speed'].fillna(0)
                sample2['veh_movement_status'] = 1
                sample2.loc[sample2['Cons_Speed']<50 , 'veh_movement_status'] = 0
                sample2['fuel_consumption']=sample2['currentFuelVolumeTank1'].diff().fillna(0)
                sample2['Cons_lph']=(sample2['fuel_consumption']/sample2['Time_diff'])*60
                sample2['fuel_movement_status'] = 1
                sample2.loc[abs(sample2['Cons_lph'])<10 , 'fuel_movement_status'] = 0
                sample2.loc[0,'fuel_movement_status']=0
                sample2=id_attachment(sample2)
                id_groups = cons_id_grouping(sample2['ID_status'].tolist())
                shift_wise_list = []
                for j in id_groups:
                    if (j[0]==0)&(len(j)==1):
                        pass
                    elif (j[0]==0)&(len(j)!=1):
                        sample3=sample2.loc[j[0]:j[-1]]
                        sample3.reset_index(drop=True,inplace=True)
                        t_dict = event_creation(sample3)
                        shift_wise_list.append(t_dict)
                    else:
                        sample3=sample2.loc[j[0]-1:j[-1]]
                        sample3.reset_index(drop=True,inplace=True)
                        t_dict = event_creation(sample3)
                        shift_wise_list.append(t_dict)
                shift_df=pd.DataFrame(shift_wise_list)
            l.append(shift_df)
        strt_end_df = pd.concat(l)
        if len(strt_end_df)!=0:
            strt_end_df['start_time']=pd.to_datetime(strt_end_df['start_time'])
            strt_end_df.sort_values(by=['start_time'],inplace=True)
            final_term_df=pd.concat([final_term_df,strt_end_df])
            final_term_df.reset_index(drop=True,inplace=True)

    veh_ign = ign[ign['termid']==termid]
    # if len(veh_ign)!=0:
    veh_ign = veh_ign.reset_index(drop=True)
    veh_f_df_dict = final_term_df.to_dict('records')
    for row in veh_f_df_dict:
        ign_ = veh_ign.loc[(((veh_ign['strt']<=row['end_time'])&(veh_ign['strt']>=row['start_time'])) | ((veh_ign['end']<=row['end_time'])&(veh_ign['end']>=row['start_time'])) | ((veh_ign['strt']<=row['start_time'])&(veh_ign['end']>=row['end_time'])))]
        ign_.loc[ign_['strt']<row['start_time'],'strt']=row['start_time']
        ign_.loc[ign_['end']>row['end_time'],'end']=row['end_time']
        ign_['dur(mins)']=(ign_['end']-ign_['strt'])/timedelta(minutes=1)
        row['ign_time_igndata'] = sum(ign_['dur(mins)'])
    final_term_df = pd.DataFrame(veh_f_df_dict)
    # else:
    #     final_term_df['ign_time_igndata'] = 0
    return final_term_df

def ign_not_exist(termid):
    veh_df = new_cst_1[new_cst_1['termid']==termid]
    veh_df.reset_index(drop=True,inplace=True)
    veh_df['ts'] = pd.to_datetime(veh_df['ts'])
    veh_df.sort_values(by=['ts'],ascending=True,inplace=True)
    veh_df['Time_diff'] = veh_df['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
    veh_df['con_cum_distance'] = veh_df['cum_distance'].diff().fillna(0)
    veh_df['Cons_Speed'] = veh_df['con_cum_distance']/veh_df['Time_diff']     #Distance
    veh_df['Cons_Speed'] = veh_df['Cons_Speed'].fillna(0)
    veh_df['veh_movement_status'] = 1
    veh_df.loc[veh_df['Cons_Speed']<50 , 'veh_movement_status'] = 0
    veh_df['fuel_consumption']=veh_df['currentFuelVolumeTank1'].diff().fillna(0)
    veh_df['Cons_lph']=(veh_df['fuel_consumption']/veh_df['Time_diff'])*60
    veh_df['fuel_movement_status'] = 1
    veh_df.loc[abs(veh_df['Cons_lph'])<10 , 'fuel_movement_status'] = 0
    veh_df.loc[0,'fuel_movement_status']=0
    veh_df['currentIgn'] = 0
    veh_df=id_attachment(veh_df)
    groups = cons_id_grouping(veh_df['ID_status'].tolist())
    groups=[sublist for sublist in groups if not (len(sublist) == 1 and sublist[0] == 0)]
    list_=[]
    for index,i in enumerate(groups):
        if (i[0]==0)&(len(i)!=1):
            sample = veh_df.loc[i[0]:i[-1]]
            id_=veh_df.loc[i[-1],'ID_status']
        elif (i[0]!=0)and(veh_df.loc[i[0],'ID_status'] in ['id1','id3','id7']):
            sample=veh_df.loc[i[0]-1:i[-1]]
            id_ = veh_df.loc[i[-1],'ID_status']
        elif (i[0]!=0)and(veh_df.loc[i[0],'ID_status'] =='id5'):
            if (veh_df.loc[i[0],'Indicator']=='strt')&(i[-1]+1<len(veh_df)):
                inc = groups[index+1]
                sample = veh_df.loc[i[0]-1:inc[-1]]
                id_ = veh_df.loc[inc[-1],'ID_status']
            else:
                sample = veh_df.loc[i[0]-1:i[-1]]
                id_ = veh_df.loc[i[-1],'ID_status']
        elif (i[0]!=0)and(veh_df.loc[i[0],'ID_status'] in ['id2','id4','id6','id8'])&(i[-1]+1<=len(veh_df)-1):
            if veh_df.loc[i[-1]+1,'ID_status'] in ['id1','id3','id5','id7']:
                sample=veh_df.loc[i[0]-1:i[-1]]
            else:
                sample=veh_df.loc[i[0]-1:i[-1]]
            id_=veh_df.loc[i[-1],'ID_status']
        sample = sample.reset_index(drop=True)
        sample['ts'] = pd.to_datetime(sample['ts'])
        start_time=sample.head(1)['ts'].item()
        end_time=sample.tail(1)['ts'].item()
        sample_list = row_split(str(start_time),str(end_time))
        l=[]
        for k in range(len(sample_list)):
            temp_dict={}
            sample2=sample[(sample['ts']>=pd.to_datetime(sample_list[k][0]))&(sample['ts']<=pd.to_datetime(sample_list[k][1]))]
            sample2.reset_index(drop=True,inplace=True)
            sample2.loc[0,'con_cum_distance']=0
            sample2['new_time_diff'] = sample2['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
            print(sample2['currentIgn'].tolist())
            ign_cst = ign_time_cst(sample2['currentIgn'].tolist(),sample2['new_time_diff'].tolist())
            keys2=['termid','reg_numb','start_time','end_time','total_obs','max_time_gap','initial_level','end_level',
                   'ign_time_cst','total_dist','ID_status','indicator']
            values2=[termid,sample2.head(1)['regNumb'].item(),sample_list[k][0],sample_list[k][1],
                     len(sample2),
                     sample2['new_time_diff'].max(),sample2.head(1)['currentFuelVolumeTank1'].item(),sample2.tail(1)['currentFuelVolumeTank1'].item(),
                     ign_cst,sample2['con_cum_distance'].sum(),id_,sample.head(1)['Indicator'].item()]
            temp_dict.update(zip(keys2,values2))
            l.append(temp_dict)
        within_df = pd.DataFrame(l)
        within_df = within_df.reset_index(drop=True)
        list_.append(within_df)
    ff=pd.concat(list_)
    ff['start_time'] = pd.to_datetime(ff['start_time'])
    ff['end_time']=pd.to_datetime(ff['end_time'])
#     ff.drop_duplicates(subset=['end_time'],keep='first',inplace=True)

    ff.reset_index(drop=True,inplace=True)
    ff['ign_time_igndata'] = 0

    return ff

def final_id_grouping(i):
#     for i in termid_list:
    if new_cst_1[new_cst_1['termid']==i]['Indicator'].nunique()!=0:
        result = ign_exist(i)
    else:
        result = ign_not_exist(i)

    return result


def additional_parameters(final_df):

    final_df=final_df.reset_index(drop=True)
    final_df['start_time'] = pd.to_datetime(final_df['start_time'])
    final_df['end_time']=pd.to_datetime(final_df['end_time'])
    final_df['total_cons']=final_df['initial_level']-final_df['end_level']
    final_df['lp100k'] = final_df.apply(lambda row: (row['total_cons']/row['total_dist'])*100000 if row['total_dist'] > 0 else 'NaN', axis=1)
    final_df['total_time'] = (final_df['end_time']-final_df['start_time']).dt.total_seconds()/60
    final_df['lph'] = final_df.apply(lambda row: (row['total_cons']/row['total_time'])*60 if row['total_time']>0 else 'NaN', axis=1)
    final_df['avg_speed'] = (final_df['total_dist']/final_df['total_time'])*0.06
    return final_df

def final_threshold_modification(i):
    if i['ID_status']=='id6':
        if abs(int(i['lph']))<10:
            i['ID_status']='id8'
    elif (i['ID_status']=='id2'):
        if ((i['total_time']<5)&(i['avg_speed']<10))or((i['total_time']>5)&(i['avg_speed']<3)):
            i['ID_status']='id8'
    elif (i['ID_status']=='id1'):
        if ((i['total_time']<5)&(i['avg_speed']<10))or((i['total_time']>5)&(i['avg_speed']<3)):
            i['ID_status']='id3'
    elif (i['ID_status']=='id4'):
        if ((i['total_time']<5)&(i['avg_speed']<10))or((i['total_time']>5)&(i['avg_speed']<3)):
            i['ID_status']='id6'
    elif (i['ID_status']=='id7'):
        if ((i['total_time']<5)&(i['avg_speed']<10))or((i['total_time']>5)&(i['avg_speed']<3)):
            i['ID_status']='id5'
    else:
        pass
    return i

def select_ign_time(row):
    if not row['total_time']:
        return np.nan
    if (row['ign_time_igndata']/row['total_time'])*100 == 0:
        return row['ign_cst']
    else:
        return row['ign_cst']

    # def print_eda(datam):
    #     datam['date'] = datam['start_time'].dt.date
    #     start_time = pd.to_datetime('22:00:00').time()
    #     datam['date'] = datam.apply(lambda row: row['date'] if start_time > row['start_time'].time() else (row['start_time'] + pd.DateOffset(days=1)).date(), axis=1)
    #     datam['tottime_move'] = datam.apply(lambda row: row['total_time'] if row['ID_status'] in  else 0,axis=1)
    #     datam['tottime_stop'] = datam.apply(lambda row: row['total_time'] if row['status']=='stationary' else 0,axis=1)
    #     datam['tottime_stop_ign_on'] = datam.apply(lambda row: row['final_ign_time'] if row['status']=='stationary' else 0,axis=1)
    #     datam['totdist_move'] = datam.apply(lambda row: row['total_dist'] if row['status']=='movement' else 0,axis=1)
    #     print_eda=datam.groupby(['reg_numb','date']).agg({'totdist_move':'sum','total_time':'sum','final_ign_time':'sum','tottime_move':'sum','tottime_stop':'sum','tottime_stop_ign_on':'sum'}).reset_index()
    #     print_eda.rename(columns={'date':'date1','totdist_move':'km_dist','total_time':'hours_timespan','final_ign_time':'hours_ign_on','tottime_move':'hours_move','tottime_stop':'hours_idle','tottime_stop_ign_on':'hours_idle_on'},inplace=True)
    #     print_eda['km_dist'] = print_eda['km_dist']/1000
    #     print_eda[['hours_timespan','hours_ign_on','hours_move','hours_idle','hours_idle_on']] /=60
    #     print_eda['hours_idle_off'] = print_eda['hours_idle']-print_eda['hours_idle_on']
    #     print_eda['idle_pct'] = print_eda['hours_idle']/print_eda['hours_timespan']
    #     print_eda['idleon_pct'] = print_eda['hours_idle_on']/print_eda['hours_idle']
    #     print_eda['mach_util_pct'] = print_eda['hours_ign_on']/print_eda['hours_timespan']
    #     return print_eda

desired_order = ['C', 'A', 'B']
def shift_order(x):
    return pd.Categorical(x, categories=desired_order, ordered=True)

def fresh_summary(datam):
    datam['tottime_move'] = datam.apply(lambda row: row['total_time'] if row['veh_status']=='movement' else 0,axis=1)
    datam['tottime_stop_ign_on'] = datam.apply(lambda row: row['final_ign_time'] if row['veh_status']=='stationary' else 0,axis=1)
    datam['totdist_move'] = datam.apply(lambda row: row['total_dist'] if row['veh_status']=='movement' else 0,axis=1)
    datam['totdist_stop'] = datam.apply(lambda row: row['total_dist'] if row['veh_status']=='stationary' else 0,axis=1)
    datam['totfuel_stop'] = datam.apply(lambda row: row['total_cons'] if row['veh_status']=='stationary' and row['total_cons']>-10 else 0,axis=1)
    datam['totfuel_move'] = datam.apply(lambda row: row['total_cons'] if row['veh_status']=='movement'and row['total_cons']>-10 else 0,axis=1)
    datam['hour'] = datam['start_time'].dt.hour
    datam['shift1'] = datam['hour'].progress_apply(categorize_shift)
    fresh_summary=datam.groupby(['regNumb','date1','shift1']).agg({'termid':'first','total_obs':'count','totdist_move':'sum','totdist_stop':'sum','tottime_move':'sum','tottime_stop_ign_on':'sum','totfuel_stop':'sum','totfuel_move':'sum','ign_time_igndata':'sum','final_ign_time':'sum','total_time':'sum'}).reset_index()
    fresh_summary.rename(columns={'regNumb':'reg_numb','total_obs':'N','ign_time_igndata':'tottime_ignevent_on','final_ign_time':'tottime_ign_on','total_time':'tottime_span'},inplace=True)
    fresh_summary['tottime_stop'] = fresh_summary['tottime_span'] - fresh_summary['tottime_move']
    fresh_summary['tottime_move_ign_on'] = fresh_summary['tottime_ign_on'] - fresh_summary['tottime_stop_ign_on']
    fresh_summary['tottime_stop_ign_off'] = fresh_summary['tottime_stop'] - fresh_summary['tottime_stop_ign_on']
    fresh_summary['idle_pct'] = fresh_summary['tottime_stop']/fresh_summary['tottime_span']
    fresh_summary['move_pct'] = fresh_summary['tottime_move']/fresh_summary['tottime_span']
    fresh_summary['ignon_move_pct'] = fresh_summary['tottime_move_ign_on']/fresh_summary['tottime_move']
    fresh_summary['ignon_idle_pct'] = fresh_summary['tottime_stop_ign_on']/fresh_summary['tottime_stop']
    fresh_summary['idle_ignon_pct'] = fresh_summary['tottime_stop_ign_on']/fresh_summary['tottime_ign_on']
    fresh_summary['shift1'] = fresh_summary.groupby(['reg_numb', 'date1'])['shift1'].transform(shift_order)
    fresh_summary = fresh_summary.sort_values(by=['reg_numb', 'date1', 'shift1']).reset_index(drop=True)
    return fresh_summary



if __name__ == '__main__':

    if (len(sys.argv) < 3) or (Path(sys.argv[1]).suffix!='.csv') or (Path(sys.argv[2]).suffix!='.RDS'):
        print('InputFileError: Kindly pass the Enriched cst in csv format followed by Ignition Master file in RDS format.\nExiting...')
        sys.exit(0)
    else:
        enriched_cst,ign_file = Path(sys.argv[1]),Path(sys.argv[2])

        new_cst_1 = pd.read_csv(enriched_cst)
        new_cst_1['ts'] = pd.to_datetime(new_cst_1['ts_unix'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        # new_cst_1 = new_cst_1[(new_cst_1['ts']>=pd.to_datetime('2023-09-01 21:00:00'))&(new_cst_1['ts']<=pd.to_datetime('2023-09-03 23:00:00'))]
        ign = pyreadr.read_r(ign_file)[None]
        ign.rename(columns={'stop':'end'},inplace=True)
        ign['strt'] = pd.to_datetime(ign['IgnON'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        ign['end'] = pd.to_datetime(ign['IgnOFF'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        ign = ign[(ign['strt']>=new_cst_1['ts'].min())&(ign['end']<=new_cst_1['ts'].max())]
        ign['termid'] = ign['termid'].astype(int)

        termid_list = new_cst_1[new_cst_1['regNumb'].str.startswith(tuple(['DJ-','DNP-','DNU-']))]['termid'].unique().tolist()  #new_cst_1['termid'].unique().tolist()   
        final_df = pd.concat([final_id_grouping(i) for i in tqdm(termid_list)])
        final_df1 = additional_parameters(final_df)
        final_df_dict=final_df1.to_dict('records')
        final_df2 = pd.DataFrame([final_threshold_modification(i) for i in tqdm(final_df_dict)])
        final_df2['final_ign_time'] = final_df2.apply(select_ign_time, axis=1)
        final_df2['date1'] = final_df2['start_time'].dt.date
        start_time = pd.to_datetime('22:00:00').time()
        final_df2['date1'] = final_df2.apply(lambda row: row['date1'] if start_time > row['start_time'].time() else (row['start_time'] + pd.DateOffset(days=1)).date(), axis=1)
        final_df2['veh_status'] = final_df2.apply(lambda x:'stationary' if x['ID_status'] in ('id3','id5','id6','id8') else 'movement',axis=1)
        fresh_summary_df = fresh_summary(final_df2)
        final_df2.rename(columns={'regNumb':'reg_numb','ign_time_igndata':'ign_time_ignMaster','ign_cst':'ign_time_cst'},inplace=True)
        final_df2['start_time'] = (final_df2['start_time'] - pd.Timestamp("1970-01-01 05:30:00")) // pd.Timedelta('1s')
        final_df2['end_time'] = (final_df2['end_time'] - pd.Timestamp("1970-01-01 05:30:00")) // pd.Timedelta('1s')
        
        if len(sys.argv) == 3:
            final_df2.to_csv('ID_event_data.csv')
            fresh_summary_df.to_csv('ID_fresh_summary.csv')
            print('ID Data saved successfully into your Working Directory.')
        elif len(sys.argv) == 4:
            print('OutPutFileError: Kindly pass two Output File Paths for Id_event data followed by the Fresh Summary in csv format\nExiting...')
            sys.exit(0)
            # print(str(outfile1).split('\\')[-1])
            # outfile2 = Path(sys.argv[4])
            # if (outfile1.suffix != '.csv')or(outfile2.suffix != '.csv'):
            #   print('OutputFilesFormatError: Need to write outputs to CSV files only\nExiting....')
            #   sys.exit(0)
            # elif (outfile1 == outfile2)or(str(outfile1).split('\\')[-1]==str(outfile2).split('\\')[-1]):
            #   print("OutputFilesNameError: Output file Paths or Names can't be same\nExiting...")
            #   sys.exit(0)
            # final_df1.to_csv(outfile2)
        elif len(sys.argv)==5:
            outfile1 = Path(sys.argv[3])
            outfile2 = Path(sys.argv[4])
            if (outfile1.suffix != '.csv')or(outfile2.suffix != '.csv'):
                print('Need to write both outputs to CSV files only\nExiting....')
                sys.exit(0)
            final_df2.to_csv(outfile1)
            fresh_summary_df.to_csv(outfile2)
            print(f' ID data folowed by fresh summary data are successfully saved to below path\n {outfile1}&{outfile2}.')
        # Check for extra args
        else:
            print('Supports atleast 1 or 2 file arguments.')
