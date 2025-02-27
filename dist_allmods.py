import pandas as pd
import numpy as np
import sys
# import matplotlib.pyplot as plt
# from d_config import cst_data_path,ign_data_path,output_data_path
from datetime import datetime, timedelta, time
from haversine import haversine_vector, Unit
from multiprocess import cpu_count
from p_tqdm import p_map
from pathlib import Path
from tqdm import tqdm
from time import sleep
import pyreadr
import pytz
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
import math
import os
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")


def categorize_shift(hour: int) -> str:
    if 6 <= hour < 14:
        return 'A'
    elif 14 <= hour < 22:
        return 'B'
    return 'C'

# def calculate_consecutive_haversine_distances(datam):
#     distances = []
#     for i in range(1, len(datam)):
#         lat1, lon1 = datam.at[i-1, 'lt'], datam.at[i-1, 'lg']
#         lat2, lon2 = datam.at[i, 'lt'], datam.at[i, 'lg']
#         distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
#         distances.append(distance)
#     distances.insert(0,0)
#     return distances

def continuous_position_wise_grouping(a):
    buckets = []
    start = 0
    is_zero_bucket = a[0] == 0
    for i, num in enumerate(a[1:], start=1):
        if num == 0 and not is_zero_bucket:
            end = i
            is_zero_bucket = True
            buckets.append((start, end))
            start = end
        elif num != 0 and is_zero_bucket:
            end = i
            is_zero_bucket = False
            buckets.append((start, end))
            start = end
    if is_zero_bucket:
        end = len(a)
        buckets.append((start, end))
    elif not is_zero_bucket and start != len(a):
        end = len(a)
        buckets.append((start, end))
    return buckets

def add_stationary_column(datam):
    datam['status'] = 'stationary'
    return datam
def add_movement_column(datam):
    datam['status'] = 'movement'
    return datam

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

def fuel_interpolation(initial_level, end_level, increments_list,total_time):

    step_size = (end_level - initial_level) / total_time
    in_list = increments_list.copy()
    buckets=[]
    if len(in_list)==1:
        buckets.append((initial_level,end_level))
        return buckets
    else:
        in_list.pop(-1)
        for increment in in_list:
            bucket_time = (pd.to_datetime(increment[1])-pd.to_datetime(increment[0])).total_seconds()/60
            bucket_start = initial_level
            bucket_end = initial_level + (bucket_time * step_size)
            buckets.append((bucket_start, bucket_end))
            initial_level = bucket_end
        buckets.append((buckets[-1][1], end_level))
        return buckets

def ign_time_cst(a,b):
    # a = ignstatus column ;  b = Time difference column
    buckets = [];start_index = None
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
            s = s+(b[j[0]]/2)+(b[j[1]+1]/2)
        except:
            s=s+(b[j[0]]/2)
        ign_time=ign_time+s
    return ign_time

def dist_allmods(i):
    term_df = df[df['termid']==i]
    term_df=term_df.reset_index(drop=True)
    term_df['shift'] = term_df['hour'].apply(categorize_shift)

    if len(term_df['lt']) == 1:
      term_df['Haversine_dist'] = 0.0
    else:
      coordinates = np.column_stack((term_df['lt'], term_df['lg']))
      haversine_distances = haversine_vector(coordinates[:-1], coordinates[1:], Unit.METERS)
      haversine_distances = np.concatenate(([0.0], haversine_distances))

      term_df['Haversine_dist'] = haversine_distances

    term_df.sort_values(by=['ts'],inplace=True)
    term_df['Time_diff'] = term_df['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
    term_df['Cons_Speed'] = term_df['Haversine_dist']/term_df['Time_diff']
    term_df['Cons_Speed'] = term_df['Cons_Speed'].fillna(0)
    term_df['status'] = 1
    term_df.loc[term_df['Cons_Speed']<50 , 'status'] = 0

    bucket = continuous_position_wise_grouping(term_df['status'].tolist())
    list_=[]
    for index,j in enumerate(bucket):
        if j[0]!=0:
            sample = term_df.iloc[j[0]-1:j[1]]
        else:
            sample = term_df.iloc[j[0]:j[1]]
            if len(sample)==1:
                try:
                    inc = bucket[index+1]
                    sample= term_df.iloc[j[0]:inc[1]]
                except:
                    pass
        sample = sample.reset_index(drop=True)
        sample['ts'] = pd.to_datetime(sample['ts'])
        sample['new_time_diff'] = sample['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
        start_d = sample.head(1)['date'].item();start_time=sample.head(1)['ts'].item()
        end_d = sample.tail(1)['date'].item();end_time=sample.tail(1)['ts'].item()

        if len(sample['lt']) == 1:
          sample['new_distance'] = 0.0
        else:
          # lt = sample['lt'].to_numpy()
          # lg = sample['lg'].to_numpy()
          # Calculate haversine distances using haversine_vector
          coordinates = np.column_stack((sample['lt'], sample['lg']))
          haversine_distances = haversine_vector(coordinates[:-1], coordinates[1:], Unit.METERS)
          haversine_distances = np.concatenate(([0.0], haversine_distances))
          sample['new_distance'] = haversine_distances

          # sample['new_distance']= calculate_consecutive_haversine_distances(sample)

        ig_time = sample[sample['currentIgn']==1]
        start_level=sample.head(1)['currentFuelVolumeTank1'].item()
        end_level=sample.tail(1)['currentFuelVolumeTank1'].item()
        if start_d == end_d:
            date = 'Same'
        else:
            date = 'Different'
        start_shift = sample.head(1)['shift'].item();end_shift=sample.tail(1)['shift'].item()
        total_time = (end_time-start_time).total_seconds()/60
        if (start_shift==end_shift)&(((date=='Same')&((start_shift=='B')or(start_shift=='A')or((start_shift=='C')&(total_time<480))))or((date=='Different')&(start_shift=='C')&(total_time<480))):
            term_dict={}
            ign_cst = ign_time_cst(sample['currentIgn'].tolist(),sample['new_time_diff'].tolist())
            keys = ['termid','reg_numb','start_time','end_time','total_obs','start_lt','start_lg','end_lt','end_lg',
                    'max_time_gap','initial_level','end_level','total_dist','ign_perc','ign_time_cst']
            values = [[sample.head(1)['termid'].item()],[sample.head(1)['regNumb'].item()],[start_time],[end_time],[len(sample)],
                      [sample.head(1)['lt'].item()],[sample.head(1)['lg'].item()],[sample.tail(1)['lt'].item()],
                      [sample.tail(1)['lg'].item()],[sample['new_time_diff'].max()],[sample.head(1)['currentFuelVolumeTank1'].item()],
                      [sample.tail(1)['currentFuelVolumeTank1'].item()],[sample['new_distance'].sum()],[(len(ig_time)/len(sample))*100],
                     [ign_cst]]
            term_dict.update(zip(keys,values))
            within_df = pd.DataFrame(term_dict)
            within_df['Interpolation_status'] = 'Both_Real'
        else:
            sample_list = row_split(str(start_time),str(end_time))
            l=[];
            fuel_inter = fuel_interpolation(start_level,end_level,sample_list,total_time)
            for k in range(len(sample_list)):
                temp_dict={}
                sample2=sample[(sample['ts']>=pd.to_datetime(sample_list[k][0]))&(sample['ts']<=pd.to_datetime(sample_list[k][1]))]
                sample2['new_time_diff'] = sample2['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
                sample2.loc[0,'new_distance']=0
                ign_cst = ign_time_cst(sample2['currentIgn'].tolist(),sample2['new_time_diff'].tolist())
                b_df = term_df[term_df['ts']<pd.to_datetime(sample_list[k][0])]
                a_df = term_df[term_df['ts']>pd.to_datetime(sample_list[k][1])]
                if (len(b_df)!=0) and (len(a_df)!=0):
                    b_sl=b_df.tail(1)['currentFuelVolumeTank1'].item();b_st=b_df.tail(1)['ts'].item()
                    a_el=a_df.head(1)['currentFuelVolumeTank1'].item();a_et=a_df.head(1)['ts'].item()
                else:
                    b_sl=0;b_st=0;a_el=0;a_et=0
                keys2=['termid','reg_numb','start_time','end_time','total_obs','max_time_gap','initial_level','end_level',
                'b_sl','b_st','a_sl','a_st','b_el','b_et','a_el','a_et','ign_time_cst','total_dist']
                values2=[i,sample.head(1)['regNumb'].item(),sample_list[k][0],sample_list[k][1],
                         len(sample[(sample['ts']>=pd.to_datetime(sample_list[k][0]))&(sample['ts']<=pd.to_datetime(sample_list[k][1]))]),
                         sample2['new_time_diff'].max(),fuel_inter[k][0],fuel_inter[k][1],
                         b_sl,b_st,term_df[term_df['ts']>pd.to_datetime(sample_list[k][0])].head(1)['currentFuelVolumeTank1'].item(),
                         term_df[term_df['ts']>pd.to_datetime(sample_list[k][0])].head(1)['ts'].item(),
                         term_df[term_df['ts']<pd.to_datetime(sample_list[k][1])].tail(1)['currentFuelVolumeTank1'].item(),
                         term_df[term_df['ts']<pd.to_datetime(sample_list[k][1])].tail(1)['ts'].item(),a_el,a_et,ign_cst,
                         sample2['new_distance'].sum()]
                temp_dict.update(zip(keys2,values2))
                l.append(temp_dict)
            within_df = pd.DataFrame(l)
            within_df['start_lt'] = sample.head(1)['lt'].item()
            within_df['start_lg'] = sample.head(1)['lg'].item()
            within_df['end_lt'] = sample.tail(1)['lt'].item()
            within_df['end_lg'] = sample.tail(1)['lg'].item()
            within_df = within_df.reset_index(drop=True)
            within_df.loc[0,'Interpolation_status'] = 'End_interpolated'
            within_df.loc[within_df.index[-1],'Interpolation_status'] = 'Start_interpolated'
        list_.append(within_df)
    list_[::2] = [add_stationary_column(df) for df in list_[::2]]
    list_[1::2] = [add_movement_column(df) for df in list_[1::2]]
    ff=pd.concat(list_)
    ff.loc[ff['Interpolation_status'].isnull()==True,'Interpolation_status']='Both_Interpolated'
    ff['start_time'] = pd.to_datetime(ff['start_time'])
    ff['end_time']=pd.to_datetime(ff['end_time'])
    ff.sort_values(by=['start_time'],inplace=True)
    ff['start_hour'] = ff['start_time'].dt.hour
    ff['end_hour'] = ff['end_time'].dt.hour
    ff['start_shift'] = ff['start_hour'].apply(categorize_shift)
    ff['end_shift'] = ff['end_hour'].apply(categorize_shift)
    sleep(.00001)

    return ff

# ign = pd.read_csv('data/dtignmast.csv', parse_dates=['strt','end'])


def ign_time_int(i):
    veh_f_df = final_df[final_df['termid']==i]
    veh_f_df = veh_f_df.reset_index(drop=True)
    veh_ign = ign[ign['termid']==i]
    veh_ign = veh_ign.reset_index(drop=True)
    for ind,row in veh_f_df.iterrows():
        ign_ = veh_ign.loc[(((veh_ign['strt']<=row['end_time'])&(veh_ign['strt']>=row['start_time'])) | ((veh_ign['end']<=row['end_time'])&(veh_ign['end']>=row['start_time'])) | ((veh_ign['strt']<=row['start_time'])&(veh_ign['end']>=row['end_time'])))]
        ign_.loc[ign_['strt']<pd.to_datetime(row['start_time']),'strt']=pd.to_datetime(row['start_time'])
        ign_.loc[ign_['end']>pd.to_datetime(row['end_time']),'end']=pd.to_datetime(row['end_time'])
        ign_['dur(mins)']=(ign_['end']-ign_['strt'])/timedelta(minutes=1)
        veh_f_df.loc[ind,'ign_time_ignMaster'] = sum(ign_['dur(mins)'])
    return veh_f_df

def final_data_f(datam):
    datam[['start_time', 'end_time']] = datam[['start_time', 'end_time']].apply(pd.to_datetime)
    datam['total_cons']=datam['initial_level']-datam['end_level']
    datam['lp100k'] = datam.apply(lambda row: (row['total_cons']/row['total_dist'])*100000 if row['total_dist'] > 0 else 'NaN', axis=1)
    datam['total_time'] = (datam['end_time']-datam['start_time']).dt.total_seconds()/60
    datam['lph'] = datam.apply(lambda row: (row['total_cons']/row['total_time'])*60 if row['total_time']>0 else 'NaN', axis=1)
    datam['avg_speed'] = (datam['total_dist']/datam['total_time'])*0.06
    datam.loc[(datam['Interpolation_status']!='Both_Real')&(datam['total_obs'].isin([0,1])),'max_time_gap'] = (datam['end_time']-datam['start_time']).dt.total_seconds()/60

    return datam

def new_fuel(s_time,e_time,s_level,e_level,date):
    total_time = (pd.to_datetime(e_time)-pd.to_datetime(s_time)).total_seconds()/60
    step_size=(e_level-s_level)/total_time
    bucket_size = (date - pd.to_datetime(s_time)).total_seconds()/60
    new_level = s_level+(bucket_size*step_size)
    return new_level

def custom_function(group):
    group_dict = group.to_dict('records')
    for row in group_dict:
        if (row['Interpolation_status']=='Both_Interpolated')&(row['total_obs']>1):
            row['initial_level'] = new_fuel(pd.to_datetime(row['b_st']),pd.to_datetime(row['a_st']),row['b_sl'],row['a_sl'],pd.to_datetime(row['start_time']))
            row['end_level'] = new_fuel(pd.to_datetime(row['b_et']),pd.to_datetime(row['a_et']),row['b_el'],row['a_el'],pd.to_datetime(row['end_time']))
        elif (row['Interpolation_status']=='Start_interpolated')&(row['total_obs']>1):
            row['initial_level'] = new_fuel(pd.to_datetime(row['b_st']),pd.to_datetime(row['a_st']),row['b_sl'],row['a_sl'],pd.to_datetime(row['start_time']))
        elif (row['Interpolation_status']=='End_interpolated')&(row['total_obs']>1):
            row['end_level'] = new_fuel(pd.to_datetime(row['b_et']),pd.to_datetime(row['a_et']),row['b_el'],row['a_el'],pd.to_datetime(row['end_time']))
    return pd.DataFrame(group_dict)

def select_ign_time(row):
    if not row['total_time']:
        return np.nan
    if ((row['ign_time_ignMaster']/row['total_time'])*100 == 100)or((row['ign_time_ignMaster']/row['total_time'])*100 == 0):
        return row['ign_time_cst']
    else:
        return row['ign_time_ignMaster']

# def print_eda(datam):
#     datam['date'] = datam['start_time'].dt.date
#     start_time = pd.to_datetime('22:00:00').time()
#     datam['date'] = datam.apply(lambda row: row['date'] if start_time > row['start_time'].time() else (row['start_time'] + pd.DateOffset(days=1)).date(), axis=1)
#     datam['tottime_move'] = datam.apply(lambda row: row['total_time'] if row['status']=='movement' else 0,axis=1)
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
    # datam['date1'] = datam['start_time'].dt.date
    # start_time = pd.to_datetime('22:00:00').time()
    # datam['date1'] = datam.apply(lambda row: row['date1'] if start_time > row['start_time'].time() else (row['start_time'] + pd.DateOffset(days=1)).date(), axis=1)
    datam['tottime_move'] = datam.apply(lambda row: row['total_time'] if row['status']=='movement' else 0,axis=1)
    datam['tottime_stop_ign_on'] = datam.apply(lambda row: row['final_ign_time'] if row['status']=='stationary' else 0,axis=1)
    datam['totdist_move'] = datam.apply(lambda row: row['total_dist'] if row['status']=='movement' else 0,axis=1)
    datam['totdist_stop'] = datam.apply(lambda row: row['total_dist'] if row['status']=='stationary' else 0,axis=1)
    datam['totfuel_stop'] = datam.apply(lambda row: row['total_cons'] if row['status']=='stationary' and row['total_cons']>-10 else 0,axis=1)
    datam['totfuel_move'] = datam.apply(lambda row: row['total_cons'] if row['status']=='movement'and row['total_cons']>-10 else 0,axis=1)
    datam['hour'] = datam['start_time'].dt.hour
    datam['shift1'] = datam['hour'].progress_apply(categorize_shift)
    fresh_summary=datam.groupby(['reg_numb','date1','shift1']).agg({'termid':'first','total_obs':'count','totdist_move':'sum','totdist_stop':'sum','tottime_move':'sum','tottime_stop_ign_on':'sum','totfuel_stop':'sum','totfuel_move':'sum','ign_time_ignMaster':'sum','final_ign_time':'sum','total_time':'sum'}).reset_index()
    fresh_summary.rename(columns={'total_obs':'N','ign_time_ignMaster':'tottime_ignevent_on','final_ign_time':'tottime_ign_on','total_time':'tottime_span'},inplace=True)
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
    # num_cores = cpu_count()
    # final_df_list = p_map(dist_allmods, termid_list, num_cpus=num_cores)
    # final_df=pd.concat(final_df_list)
    # final_df_dict=final_df.to_dict('records')
    # integrated_df_list = p_map(ign_time_int, final_df_dict, num_cpus=num_cores)
    # integrated_df=pd.DataFrame(integrated_df_list)
    # integrated_df1 = final_data_f(integrated_df)

    if len(sys.argv) < 3:
      print('NoPathError: You need to provide the path of the RDS files as input. CST data followed by ignition data.\nExiting...')
      sys.exit(0)
    else:
      infile_cst, infile_igtn = Path(sys.argv[1]), Path(sys.argv[2])

      # Check validity of both args at once
      if infile_cst.suffix == '.RDS':
        df = pyreadr.read_r(infile_cst)[None]
      else:
        df = pd.read_csv(infile_cst)
      if infile_igtn.suffix == '.RDS':
        ign = pyreadr.read_r(infile_igtn)[None]
      else:
        ign = pd.read_csv(infile_igtn)

      df.rename(columns={'latitude':'lt', 'longitude':'lg'}, inplace=True)
      df.dropna(subset=['termid', 'lt', 'lg'], inplace=True)
      df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
      df['date'] = df['ts'].dt.date.astype(str)
      df['hour'] = df['ts'].dt.hour
      termid_list = df['termid'].unique().tolist()
      ign['strt'] = pd.to_datetime(ign['IgnON'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
      ign['end'] = pd.to_datetime(ign['IgnOFF'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
      ign['termid'] = ign['termid'].astype(int)
      final_df = pd.concat([dist_allmods(termid) for termid in tqdm(termid_list)])
      final_df_dict=final_df.to_dict('records')
      integrated_df = pd.concat([ign_time_int(termid) for termid in tqdm(termid_list)])
      integrated_df.reset_index(drop=True, inplace=True)
      grouped = integrated_df.groupby('termid')
      integrated_df = grouped.progress_apply(custom_function)
      integrated_df=integrated_df.reset_index(drop=True)
      integrated_df = final_data_f(integrated_df)
      integrated_df['final_ign_time'] = integrated_df.apply(select_ign_time, axis=1)
      integrated_df['date1'] = integrated_df['start_time'].dt.date
      start_time = pd.to_datetime('22:00:00').time()
      integrated_df['date1'] = integrated_df.apply(lambda row: row['date1'] if start_time > row['start_time'].time() else (row['start_time'] + pd.DateOffset(days=1)).date(), axis=1)
      fresh_summary_df = fresh_summary(integrated_df)
      integrated_df['start_time'] = (integrated_df['start_time'] - pd.Timestamp("1970-01-01 05:30:00")) // pd.Timedelta('1s')
      integrated_df['end_time'] = (integrated_df['end_time'] - pd.Timestamp("1970-01-01 05:30:00")) // pd.Timedelta('1s')
      if 'b_sl' in integrated_df.columns:
        integrated_df.drop(['start_hour','end_hour','b_sl','b_st','a_sl','a_st','b_el','b_et','a_el','a_et'],axis=1,inplace=True)

      # When No Output Data Path is given ~ Save Files into Working Directory
      if len(sys.argv) == 3:
        integrated_df.to_csv('Integrated_dist_allmods.csv')
        fresh_summary_df.to_csv('fresh_summary.csv')
        print('Data saved successfully to the above path')

      # When only one output path is given s
      elif len(sys.argv) == 4:
        outfile1 = Path(sys.argv[3])
        # outfile2 = Path(sys.argv[4])
        if (outfile1.suffix != '.csv'):
          print('Need to write Output File to a CSV file only\nExiting....')
          sys.exit(0)
        integrated_df.to_csv(outfile1)
        # fresh_summary_df.to_csv(outfile2)
        print(f' Output Idling is saved successfully to below paths \n {outfile1}')

       # Two output paths are given in correct format
      elif len(sys.argv) == 5:
        outfile1 = Path(sys.argv[3])
        outfile2 = Path(sys.argv[4])
        if (outfile1.suffix != '.csv')or(outfile2.suffix != '.csv'):
          print('Need to write both outputs to a CSV file only\nExiting....')
          sys.exit(0)
        integrated_df.to_csv(outfile1)
        fresh_summary_df.to_csv(outfile2)
        print(f'Outputs saved successfully to below paths \n {outfile1} & {outfile2}')

      # Check for extra args
      else:
        print('Supports atleast 2 and atmost 4 file arguments')
