import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta, time
from haversine import haversine, Unit
from haversine import haversine_vector, Unit
from tqdm import tqdm
from time import sleep
from pathlib import Path
import pytz
import pyreadr
import time
from time import time as t
from itertools import combinations, permutations
import math
import os
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 150)
from multiprocess import cpu_count

def categorize_shift(hour: int) -> str:
    if 6 <= hour < 14:
        return 'A'
    elif 14 <= hour < 22:
        return 'B'
    return 'C'

# def calculate_consecutive_haversine_distances(df):
#     distances = []
#     for i in range(1, len(df)):
#         lat1, lon1 = df.at[i-1, 'lt'], df.at[i-1, 'lg']
#         lat2, lon2 = df.at[i, 'lt'], df.at[i, 'lg']
#         distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
#         distances.append(distance)
#     distances.insert(0,0)
#     return distances

def disp_cst(i):                              # Injection of Hecpoll refuel start timings into CST   => i : termid
    term_df = cst[cst['regNumb']==i]
    term_df.reset_index(drop=True,inplace=True)

    # if len(term_df['lt']) == 1:
    #     term_df['Distance'] = 0.0
    # else:
    #     coordinates = np.column_stack((term_df['lt'], term_df['lg']))
    #     haversine_distances = haversine_vector(coordinates[:-1], coordinates[1:], Unit.METERS)
    #     haversine_distances = np.concatenate(([0.0], haversine_distances))
    #     term_df['Distance'] = haversine_distances

    # term_df['cum_distance'] = term_df['Distance'].cumsum().fillna(0)
    disp_df = disp[disp['regNumb']==i]
    if len(disp_df)!=0:
        con = pd.concat([term_df,disp_df],axis=0)
        con.sort_values(by=['ts'],inplace=True)
        con.loc[con['termid'].isnull(),'termid']=term_df.head(1)['termid'].item()
        # con['mine'] = term_df.head(1)['mine'].item()
        # con['class'] = term_df.head(1)['class'].item()
        con.sort_values(by=['ts','Refuel_status'],na_position='first',inplace=True)
        con.reset_index(drop=True,inplace=True)
        duplicated = [int(i) - 1 for i in con[con['ts'].duplicated()].index.tolist()]
        for l in duplicated:
            con.loc[l,'Refuel_status'] = con.loc[l+1,'Refuel_status']
            con.loc[l,'Quantity'] = con.loc[l+1,'Quantity']
            con.loc[l,'TxId'] = con.loc[l+1,'TxId']
        con.drop_duplicates(subset=['ts'],keep='first',inplace=True)
        return con
    else:
        return term_df

def new_fuel(s_time,e_time,s_level,e_level,date):           # For single fuel interpolation based on before/after values
    #  =>  s_time : previous time , e_time : next time , s_level : previous fuel level , e_level : next level , date : Time for what this Interpolation to do

    total_time = (pd.to_datetime(e_time)-pd.to_datetime(s_time)).total_seconds()/60
    step_size=(e_level-s_level)/total_time
    bucket_size = (pd.to_datetime(date) - pd.to_datetime(s_time)).total_seconds()/60
    new_level = s_level+(bucket_size*step_size)
    return new_level

def find_first_peak_index(arr):
    for i in range(1, len(arr) - 1):
        if arr[i] >= arr[i - 1] and arr[i] > arr[i + 1]:
            return i
    return np.where(np.array(arr) == max(arr))[0][-1]

def refuel_end_injection(i):                        # Injection of Refuel-end points approx 20 mins apart from each start points
    #   => i : termid

    term_df = disp_cst[disp_cst['termid']==i]
    term_df.reset_index(drop=True,inplace=True)
    if (term_df.loc[0,'Refuel_status']=='Refuel')&(term_df.loc[term_df.index[-1],'Refuel_status']=='Refuel'):
        term_df.drop([0,term_df.index[-1]],axis=0,inplace=True)
    elif term_df.loc[0,'Refuel_status']=='Refuel':
        term_df.drop(0,axis=0,inplace=True)
    elif term_df.loc[term_df.index[-1],'Refuel_status']=='Refuel':
        term_df.drop(term_df.index[-1],inplace=True)    
    else:
        pass
    term_df.reset_index(drop=True,inplace=True)
#     count = count+len(term_df)
    injected_data=[]
    for ind,j in term_df.iterrows():
        if (j['Refuel_status']=='Refuel'):
            next_qty_list = term_df.loc[ind+1:ind+31]['currentFuelVolumeTank1'].tolist()
            first_peak_index = find_first_peak_index(next_qty_list)
            # refuel_end_time = term_df.loc[ind,'ts'] + timedelta(minutes=20)
            if ind !=0:
                level = term_df.loc[ind-2:ind,'currentFuelVolumeTank1'].min()#;term_df.loc[ind-2:ind,'currentFuelVolumeTank1']=level
                term_df.loc[ind-2:ind,'currentFuelVolumeTank1'] = level               # fuel fillup in refuel start times
                term_df.loc[ind+first_peak_index+1 , 'currentFuelVolumeTank1'] = level + term_df.loc[ind,'Quantity']     # fuel fillup in Refuel end
                term_df.loc[ind+first_peak_index+1 , 'Refuel_status'] = 'Refuel_end'                                     # status fillup in ''
                # refuel_end_level = level + term_df.loc[ind,'Quantity']

                refuel_start_cum_distance = new_fuel(term_df.loc[ind-1,'ts'],term_df.loc[ind+1,'ts'],
                                                    term_df['cum_distance'].iloc[:ind].dropna().iloc[-1],term_df['cum_distance'].iloc[ind:].dropna().iloc[0],
                                                    term_df.loc[ind,'ts'])
                term_df.loc[ind,'cum_distance'] = refuel_start_cum_distance

                # injected_data.append({'termid':i,'regNumb':term_df.head(1)['regNumb'].item(),'ts':refuel_end_time,
                #                     'currentFuelVolumeTank1':refuel_end_level,'Refuel_status':'Refuel_end'})

            else:
                level = term_df.loc[ind+1,'currentFuelVolumeTank1']
                term_df.loc[ind,'currentFuelVolumeTank1'] = level
                term_df.loc[ind+first_peak_index+1 , 'currentFuelVolumeTank1'] = level + term_df.loc[ind,'Quantity']
                term_df.loc[ind+first_peak_index+1 , 'Refuel_status'] = 'Refuel_end'
                term_df.loc[ind,'cum_distance'] = term_df.loc[ind+1,'cum_distance'].item()
                # refuel_end_level = level + term_df.loc[ind,'Quantity']
                # injected_data.append({'termid':i,'regNumb':term_df.head(1)['regNumb'],'ts':refuel_end_time,'currentFuelVolumeTank1':refuel_end_level,'Refuel_status':'Refuel_end'})
    # injected_df = pd.DataFrame(injected_data)
#     normal_df=normal_df.append(term_df)
    # concat_df = pd.concat([term_df,injected_df],axis=0,ignore_index=True)
    # concat_df['ts'] = pd.to_datetime(concat_df['ts'])
    term_df.sort_values(by=['ts'],inplace=True)     # concat_df , termid
    # term_df.reset_index(drop=True,inplace=True)          # concat_df
    # for ind,j in concat_df.iterrows():
    #     if j['Refuel_status']=='Refuel_end':
    #         a = concat_df[concat_df['ts']<pd.to_datetime(j['ts'])]
    #         b = concat_df[concat_df['ts']>pd.to_datetime(j['ts'])]
    #         if len(b)!=0:
    #             end_cum_distance = new_fuel(a.tail(1)['ts'].item(),b.head(1)['ts'].item(),a['cum_distance'].dropna().iloc[-1],
    #                                     b['cum_distance'].dropna().iloc[0],j['ts'])
    #         else:
    #             end_cum_distance = a['cum_distance'][a['cum_distance'] != 0].iloc[-1]
    #         concat_df.loc[ind,'cum_distance'] = end_cum_distance

    return term_df     # concat_df

# def refuel_end_cum_distance(i):                                  # Cumulative Distance interpolation for Refuel end points
#       # => i : termid
#     term_df = disp_cst1[disp_cst1['termid']==i]
#     term_df.reset_index(drop=True,inplace=True)
#     for ind,j in term_df.iterrows():
#         if j['Refuel_status']=='Refuel_end':
#             a = term_df[term_df['ts']<pd.to_datetime(j['ts'])]
#             b = term_df[term_df['ts']>pd.to_datetime(j['ts'])]
#             end_cum_distance = new_fuel(a.tail(1)['ts'].item(),b.head(1)['ts'].item(),a.tail(1)['cum_distance'].item(),
#                                        b.head(1)['cum_distance'].item(),j['ts'])
#             term_df.loc[ind,'cum_distance'] = end_cum_distance
#     return term_df

def melt_conc(i):                         # For injecting original ignition points into CST => i = termid;
    ign_term = ign[ign['termid']==i];cst_term=disp_cst2[disp_cst2['termid']==i]
    cst_term=cst_term.reset_index(drop=True);ign_term=ign_term.reset_index(drop=True)
    if len(ign_term)!=0:
        melt_ign = pd.melt(ign_term,value_vars=['strt','end'],var_name='Indicator',value_name='ts')
        melt_ign['termid']=str(i);melt_ign['regNumb']=ign_term.head(1)['veh'].item()
        # melt_ign.sort_values(by='ts',inplace=True)
        cst_1 = pd.concat([cst_term,melt_ign],ignore_index=True)
        cst_1['ts'] = pd.to_datetime(cst_1['ts'])
        cst_1.sort_values(by='ts',inplace=True)
        # cst_1.reset_index(drop=True,inplace=True)
        end_indices = cst_1[cst_1['Indicator'] == 'end'].index
    #     cst_1.loc[end_indices, 'Distance'] = cst_1['Distance'].shift(-1)
        cst_1.loc[end_indices, 'cum_distance'] = cst_1['cum_distance'].shift(-1)

        for ind,row in cst_1.iterrows():
            if (row['Indicator'] in ('end','strt'))&(str(row['cum_distance'])=='nan'):
                temp_df = cst_term[cst_term['ts']<pd.to_datetime(row['ts'])]
                a_df = cst_term[cst_term['ts']>pd.to_datetime(row['ts'])]
                if (len(temp_df)!=0)&(len(a_df)!=0):
                    s_time=temp_df.tail(1)['ts'].item()
                    e_time=a_df.head(1)['ts'].item()
    #                 s_level=temp_df.tail(1)['Distance'].item()
                    s_level_1=temp_df.tail(1)['cum_distance'].item()
    #                 e_level=a_df.head(1)['Distance'].item()
                    e_level_1=a_df.head(1)['cum_distance'].item()
    #                 cst_1.loc[ind,'Distance'] = new_fuel(s_time,e_time,s_level,e_level,row['ts'])
                    cst_1.loc[ind,'cum_distance'] = new_fuel(s_time,e_time,s_level_1,e_level_1,row['ts'])
                elif len(temp_df)==0:
    #                 cst_1.loc[ind,'Distance'] = a_df.head(1)['Distance'].item()
                    cst_1.loc[ind,'cum_distance'] = a_df.head(1)['cum_distance'].item()
                else:
    #                 cst_1.loc[ind,'Distance'] = temp_df.tail(1)['Distance'].item()
                    cst_1.loc[ind,'cum_distance'] = temp_df.tail(1)['cum_distance'].item()
        cst_1.sort_values(by=['ts','Indicator'],inplace=True)
        cst_1.reset_index(drop=True,inplace=True)
        fil_strt=cst_1.query("Indicator=='strt'")
        for ind,row in fil_strt.iterrows():
            temp_df = cst_term[cst_term['ts']<pd.to_datetime(row['ts'])]
            a_df = cst_term[cst_term['ts']>pd.to_datetime(row['ts'])]
            if (len(temp_df)!=0)&(len(a_df)!=0):
                s_time=temp_df.tail(1)['ts'].item()
                e_time=cst_term[cst_term['ts']>pd.to_datetime(row['ts'])].head(1)['ts'].item()
                s_level=temp_df.tail(1)['currentFuelVolumeTank1'].item()
                e_level=cst_term[cst_term['ts']>pd.to_datetime(row['ts'])].head(1)['currentFuelVolumeTank1'].item()
                cst_1.loc[ind,'currentFuelVolumeTank1']= new_fuel(s_time,e_time,s_level,e_level,row['ts'])
            elif len(temp_df)==0:
                cst_1.loc[ind,'currentFuelVolumeTank1'] = a_df.head(1)['currentFuelVolumeTank1'].item()
            else:
                cst_1.loc[ind,'currentFuelVolumeTank1'] = temp_df.tail(1)['currentFuelVolumeTank1'].item()
        end_indices = cst_1[cst_1['Indicator'] == 'end'].index
        cst_1.loc[end_indices, 'currentFuelVolumeTank1'] = cst_1['currentFuelVolumeTank1'].shift(-1)
        cst_1['termid'] = cst_1['termid'].astype(int)
        cst_1.sort_values(by=['ts','Indicator'],na_position='first',inplace=True)
        cst_1.reset_index(drop=True,inplace=True)
        duplicated = [int(i) - 1 for i in cst_1[cst_1['ts'].duplicated()].index.tolist()]
        for l in duplicated:
            cst_1.loc[l,'Indicator'] = cst_1.loc[l+1,'Indicator']
        cst_1.drop_duplicates(subset=['ts'],keep='first',inplace=True)
        cst_1.loc[(cst_1['Indicator']== 'strt')|(cst_1['Indicator']=='end'),'Is_Ignition']= 'TRUE'
        # cst_1['mine']=cst_term.head(1)['mine'].item()
        # cst_1['class'] = cst_term.head(1)['mine'].item()
    else:
        cst_1 = cst_term.copy()

    return cst_1

def find_contiguous_groups_indices(binary_list):     # consecutive 1s bucketings from 'currentIgn' column (PM) ; binary_list = currentIgn list
    groups_indices = []
    start_index = None
    for i, bit in enumerate(binary_list):
        if bit == 1:
            if start_index is None:
                start_index = i
        elif start_index is not None:
            if i - start_index > 1:
                groups_indices.append((start_index, i - 1))
            start_index = None

    # Check if the last group is also valid
    if start_index is not None and len(binary_list) - start_index > 1:
        groups_indices.append((start_index, len(binary_list) - 1))

    return groups_indices

def synthetic_ignition(datam):      # Filling up Indicator column with cst 'strt' 'end'
    # =>  datam : Cst data

    indices_strt_end = datam[~datam['Indicator'].isnull()].index.tolist()[1:] + [len(datam)-1]
    results = [(indices_strt_end[i] + 1,indices_strt_end[i+1] - 1) for i in range(0, len(indices_strt_end) - 1, 2) if (indices_strt_end[i+1] - indices_strt_end[i]) > 2]

    for x, y in results:
        res = [(l[0] + x, l[1] + x) for l in find_contiguous_groups_indices(datam.loc[x:y, 'currentIgn'])]
        datam.loc[[x for x, _ in res], 'Indicator'] = 'strt'
        datam.loc[[y for _, y in res], 'Indicator'] = 'end'

    return datam


def shift_custom_function(group):                    # Injection of shift points and Fuel/distances accordingly  ; Running for each date-group inside each termid group
    cst_term = new_cst[new_cst['termid']==group.head(1)['termid'].item()]
    group['ts']=pd.to_datetime(group['ts']);cst_term['ts']=pd.to_datetime(cst_term['ts'])
    group.sort_values(by=['ts'],inplace=True)
    time_1 = str(group.head(1)['date'].item())+' 06:00:00'
    temp = cst_term[cst_term['ts']<pd.to_datetime(time_1)];tem = cst_term[cst_term['ts']>pd.to_datetime(time_1)]
    if len(temp)==0:
        time_1_level=tem.head(1)['currentFuelVolumeTank1'].item()
#         time_1_dist=tem.head(1)['Distance'].item()
        time_1_cum_dist=tem.head(1)['cum_distance'].item()
    elif len(tem)==0:
        time_1_level=temp.tail(1)['currentFuelVolumeTank1'].item()
#         time_1_dist=tem.head(1)['Distance'].item()
        time_1_cum_dist=temp.tail(1)['cum_distance'].item()
    else:
        time_1_level = new_fuel(temp.tail(1)['ts'].item(),tem.head(1)['ts'].item(),
                               temp.tail(1)['currentFuelVolumeTank1'].item(),
                               tem.head(1)['currentFuelVolumeTank1'].item(),pd.to_datetime(time_1))
#         time_1_dist = new_fuel(temp.tail(1)['ts'].item(),tem.head(1)['ts'].item(),
#                                temp.tail(1)['Distance'].item(),
#                                tem.head(1)['Distance'].item(),pd.to_datetime(time_1))
        time_1_cum_dist=new_fuel(temp.tail(1)['ts'].item(),tem.head(1)['ts'].item(),
                               temp.tail(1)['cum_distance'].item(),
                               tem.head(1)['cum_distance'].item(),pd.to_datetime(time_1))
    time_2 = str(group.head(1)['date'].item())+' 14:00:00'
    temp_1 = cst_term[cst_term['ts']<pd.to_datetime(time_2)];tem_1 = cst_term[cst_term['ts']>pd.to_datetime(time_2)]
    if len(tem_1)==0:
        time_2_level=temp_1.tail(1)['currentFuelVolumeTank1'].item()
#         time_2_dist=temp_1.tail(1)['Distance'].item()
        time_2_cum_dist=temp_1.tail(1)['cum_distance'].item()
    elif len(temp_1)==0:
        time_2_level=tem_1.head(1)['currentFuelVolumeTank1'].item()
#         time_2_dist = tem_1.head(1)['Distance'].item()
        time_2_cum_dist=tem_1.head(1)['cum_distance'].item()
    else:
        time_2_level = new_fuel(temp_1.tail(1)['ts'].item(),tem_1.head(1)['ts'].item(),
                               temp_1.tail(1)['currentFuelVolumeTank1'].item(),
                               tem_1.head(1)['currentFuelVolumeTank1'].item(),pd.to_datetime(time_2))
#         time_2_dist = new_fuel(temp_1.tail(1)['ts'].item(),tem_1.head(1)['ts'].item(),
#                                temp_1.tail(1)['Distance'].item(),
#                                tem_1.head(1)['Distance'].item(),pd.to_datetime(time_2))
        time_2_cum_dist = new_fuel(temp_1.tail(1)['ts'].item(),tem_1.head(1)['ts'].item(),
                               temp_1.tail(1)['cum_distance'].item(),
                               tem_1.head(1)['cum_distance'].item(),pd.to_datetime(time_2))
#     else:
#         time_2_level = 0;time_2_dist=0
    time_3 = str(group.head(1)['date'].item())+' 22:00:00'
    temp_2 = cst_term[cst_term['ts']<pd.to_datetime(time_3)]
    tem_2 = cst_term[cst_term['ts']>pd.to_datetime(time_3)]
    if len(tem_2)==0:
        time_3_level = temp_2.tail(1)['currentFuelVolumeTank1'].item()
#         time_3_dist=temp_2.tail(1)['Distance'].item()
        time_3_cum_dist=temp_2.tail(1)['cum_distance'].item()
    elif len(temp_2)==0:
#         print(group.head(1)['termid'].item(),group.head(1)['date'].item())
        time_3_level=tem_2.head(1)['currentFuelVolumeTank1'].item()
#         time_3_dist=tem_2.head(1)['Distance'].item()
        time_3_cum_dist=tem_2.head(1)['cum_distance'].item()
    else:
        time_3_level = new_fuel(temp_2.tail(1)['ts'].item(),tem_2.head(1)['ts'].item(),
                               temp_2.tail(1)['currentFuelVolumeTank1'].item(),
                               tem_2.head(1)['currentFuelVolumeTank1'].item(),pd.to_datetime(time_3))
#         time_3_dist = new_fuel(temp_2.tail(1)['ts'].item(),tem_2.head(1)['ts'].item(),
#                                temp_2.tail(1)['Distance'].item(),
#                                tem_2.head(1)['Distance'].item(),pd.to_datetime(time_3))
        time_3_cum_dist = new_fuel(temp_2.tail(1)['ts'].item(),tem_2.head(1)['ts'].item(),
                               temp_2.tail(1)['cum_distance'].item(),
                               tem_2.head(1)['cum_distance'].item(),pd.to_datetime(time_3))
    temp_df = pd.DataFrame({'ts':[time_1,time_2,time_3],'currentFuelVolumeTank1':[time_1_level,time_2_level,time_3_level],
                           'cum_distance':[time_1_cum_dist,time_2_cum_dist,time_3_cum_dist]})
    temp_df['termid'] = group.head(1)['termid'].item()
    temp_df['regNumb'] = group.head(1)['regNumb'].item()
    temp_df['ts'] = pd.to_datetime(temp_df['ts'])
    # temp_df['mine'] = group.head(1)['mine'].item();temp_df['class'] = group.head(1)['class'].item()
    df = pd.concat([group,temp_df],axis=0)
    df.sort_values(by=['ts','direction'],na_position='last',inplace=True)
    df.drop_duplicates(subset=['ts'],inplace=True)
    return df

def custom_function(group):                         #  Running for each termid group

    start_time1 = group['ts'].min();end_time1 = group['ts'].max()
    group_1 = group.groupby('date')
    group_result = group_1.apply(shift_custom_function)
    group_result=group_result.reset_index(drop=True)
    # group_result['mine']= group.head(1)['mine'].item()
    # group_result['class'] = group.head(1)['mine'].item()
    group_result['Distance'] = group_result['cum_distance'].diff().fillna(0)
    group_result = group_result[(group_result['ts']>=start_time1)&(group_result['ts']<=end_time1)]
    return group_result


if __name__ == '__main__':

#     cst = pyreadr.read_r('../INPUT_DATA/data/oct/cst_1_6Oct.RDS')[None]
#     cst['ts'] = pd.to_datetime(cst['ts'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
#     cst['date'] = pd.to_datetime(cst['ts']).dt.date.astype(str)
#     cst.dropna(subset=('termid', 'currentIgn', 'cum_distance', 'currentFuelVolumeTank1'), inplace=True)
#     # faulty_fuel = cst[cst['currentFuelVolumeTank1'].isnull()]['regNumb'].unique().tolist()
#     # cst = cst[~cst['regNumb'].isin(faulty_fuel)]
#     start_time1 = cst['ts'].min();end_time1=cst['ts'].max()


#     # Ignition Master Data Read and Pre processing

#     ign = pyreadr.read_r('../INPUT_DATA/data/oct/dtign_upto9Oct.RDS')[None]
#     ign.rename(columns={'stop':'end'}, inplace=True)
#     ign['strt'] = pd.to_datetime(ign['IgnON'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
#     ign['end'] = pd.to_datetime(ign['IgnOFF'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
#     ign.sort_values(by=['termid','strt','end'],inplace=True)
#     ign.drop_duplicates(subset=['termid','strt'],keep='last',inplace=True)
#     ign.drop_duplicates(subset=['termid','end'],keep='first',inplace=True)
#     ign = ign[(ign['strt']>=cst['ts'].min())&(ign['end']<=cst['ts'].max())]
#     ign['termid'] = ign['termid'].astype(int)
#     ign = ign[['termid','veh','strt','end']]

#     termid_list = cst['termid'].unique().tolist()[:1]
#     regNumb_list = cst['regNumb'].unique().tolist()[0:1]
#     # termid_list = [1204000487];regNumb_list=['WSNP-2201']

#     # Hectronics Refuel Data Read

#     disp = pd.read_csv('../INPUT_DATA/data/oct/hecpoll_1_11Oct.csv')
#     disp.rename(columns={'Vehicle Number':'regNumb','Date':'date','Time Stamp':'ts'},inplace=True)
#     disp=disp[disp['regNumb'].isin(cst['regNumb'])][['ts','date','Station Name','regNumb','Quantity','TxId']]
# #   if len(disp)!=0:
#     disp['Refuel_status'] = 'Refuel'
#     disp['ts']=disp['ts'].str.replace(' IST', '')
#     disp['ts'] = pd.to_datetime(disp['ts'])
#     disp = disp[(disp['ts']>=cst['ts'].min())&(disp['ts']<=cst['ts'].max())]
#     disp['Quantity'] = disp['Quantity'].str.replace(',','').astype(float)
#     disp = disp[disp['Quantity']>20]
#     print("Iteration 1: Refuel concatenation to CST")
#     disp_cst = pd.concat([disp_cst(i) for i in tqdm(regNumb_list)])

#     print("Iteration 2: Refuel end times injection into CST")
#     disp_cst2 = pd.concat([refuel_end_injection(i) for i in tqdm(termid_list)])


    num_cores = cpu_count()
    if len(sys.argv) < 4:
      print('InputFilesError: You need to provide the path of RDS cst/ign files and Hectronic csv as input.\nCST data followed by ignition data followed by Hectronics Dispense Data.\nExiting....')
      sys.exit(0)
    else:
      infile_cst, infile_igtn,disp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

      # Check validity of both args at once
      if (infile_igtn.suffix != '.RDS') or (disp.suffix!='.csv'):
        print('FileFormatError: Only RDS files for cst/ign and csv for Hec data applicable as input\nExiting....')
        sys.exit(0)
      if infile_cst.suffix == '.RDS':
          cst = pyreadr.read_r(infile_cst)[None]
          cst['ts'] = pd.to_datetime(cst['ts'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)

      else:
          cst = pd.read_csv(infile_cst)
        #   cst['ts'] = cst['ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
          cst['ts'] = pd.to_datetime(cst['ts'])

      cst['date'] = pd.to_datetime(cst['ts']).dt.date.astype(str)
      cst.dropna(subset=('termid', 'currentIgn', 'cum_distance', 'currentFuelVolumeTank1'), inplace=True)
      # faulty_fuel = cst[cst['currentFuelVolumeTank1'].isnull()]['regNumb'].unique().tolist()
      # cst = cst[~cst['regNumb'].isin(faulty_fuel)]
      start_time1 = cst['ts'].min();end_time1=cst['ts'].max()


      # Ignition Master Data Read and Pre processing

      ign = pyreadr.read_r(infile_igtn)[None]
      ign.rename(columns={'stop':'end'}, inplace=True)
      ign['strt'] = pd.to_datetime(ign['IgnON'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
      ign['end'] = pd.to_datetime(ign['IgnOFF'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
      ign.sort_values(by=['termid','strt','end'],inplace=True)
      ign.drop_duplicates(subset=['termid','strt'],keep='last',inplace=True)
      ign.drop_duplicates(subset=['termid','end'],keep='first',inplace=True)
      ign = ign[(ign['strt']>=cst['ts'].min())&(ign['end']<=cst['ts'].max())]
      ign['termid'] = ign['termid'].astype(int)
      ign = ign[['termid','veh','strt','end']]

      termid_list = cst['termid'].unique().tolist()
      regNumb_list = cst['regNumb'].unique().tolist()

      # Hectronics Refuel Data Read

      disp = pd.read_csv(disp)
      disp.rename(columns={'Vehicle Number':'regNumb','Date':'date','Time Stamp':'ts'},inplace=True)
      disp=disp[disp['regNumb'].isin(cst['regNumb'])][['ts','date','Station Name','regNumb','Quantity','TxId']]
      if len(cst)==0:
        print('CstDataError: Fuel levels API values error/Blank. Kindly pass a valid cst data.\nExiting....')
        sys.exit(0)
    #   if len(disp)!=0:
      disp['Refuel_status'] = 'Refuel'
      disp['ts']=disp['ts'].str.replace(' IST', '')
      disp['ts'] = pd.to_datetime(disp['ts'])
      disp = disp[(disp['ts']>=cst['ts'].min())&(disp['ts']<=cst['ts'].max())]
      disp['Quantity'] = disp['Quantity'].str.replace(',','').astype(float)
      disp = disp[disp['Quantity']>20]


      # Synthetic Algorithm Iterations

      print("Iteration 1: Refuel concatenation to CST")
      disp_cst = pd.concat([disp_cst(i) for i in tqdm(regNumb_list)])

      print("Iteration 2: Refuel end times injection into CST")
      disp_cst2 = pd.concat([refuel_end_injection(i) for i in tqdm(termid_list)])

      print("Iteration 3: Ignition Concatenation to CST")
      new_cst = pd.concat([melt_conc(termid) for termid in tqdm(termid_list)])
      new_cst.sort_values(by=['termid','ts'],inplace=True)
      new_cst = new_cst.reset_index(drop=True)
      new_cst = synthetic_ignition(new_cst)
      new_cst['termid']=new_cst['termid'].astype(int)
      new_cst['date'] = new_cst['ts'].dt.date

      print("Iteration 4: Shift times injection to CST")
      grouped = new_cst.groupby('termid')
      new_cst_1=grouped.progress_apply(custom_function)

      new_cst_1=new_cst_1.reset_index(drop=True)
      new_cst_1.drop(['Station Name', 'date'],axis=1,inplace=True)
      new_cst_1['date1'] = new_cst_1['ts'].dt.date
      start_time = pd.to_datetime('22:00:00').time()
      new_cst_1['date1'] = new_cst_1.apply(lambda row: row['date1'] if start_time > row['ts'].time() else (row['ts'] + pd.DateOffset(days=1)).date(), axis=1)
      new_cst_1['hour'] = new_cst_1['ts'].dt.hour
      new_cst_1['shift1'] = new_cst_1['hour'].progress_apply(categorize_shift)
      new_cst_1.drop('hour', axis=1, inplace=True)
      new_cst_1['ts_unix'] = (new_cst_1['ts'] - pd.Timestamp("1970-01-01 05:30:00")) // pd.Timedelta('1s')
      print('Synthetic CST has been generated successfully! ')
      new_cst_1 = new_cst_1[(new_cst_1['ts']>=start_time1)&(new_cst_1['ts']<=end_time1)]
      new_cst_1.drop(['ts'],axis=1,inplace=True)


    # Error Logging for Output Files
      if len(sys.argv) == 4:
        new_cst_1.to_csv('New_Synthetic_CST.csv')
        print('Data saved successfully into your Working Directory.')
    
      elif len(sys.argv) == 5:
        outfile1 = Path(sys.argv[4])
        new_cst_1.to_csv(outfile1)
        print(f' Enriched CST is successfully saved to below path: \n{outfile1}.')

      # Check for extra args
      else:
        print('Supports atleast 3 or 4 file arguments.')
