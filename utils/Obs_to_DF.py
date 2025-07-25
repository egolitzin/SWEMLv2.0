# Import packages
# Dataframe Packages
import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Vector Packages
import geopandas as gpd
import shapely
from shapely import wkt
from shapely.geometry import Point, Polygon

# General Packages
import os
import re
from datetime import datetime
import glob
from pathlib import Path
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
import time
import requests
import concurrent.futures as cf
# import s3fs do not need this
# import get_InSitu_obs
import pickle as pkl
import warnings; warnings.filterwarnings("ignore")

#need to mamba install gdal, earthaccess 
#pip install pystac_client, richdem, planetary_computer, dask, distributed, retrying

#connecting to AWS
import boto3

#load access key
HOME = os.getcwd()
KEYPATH = "utils/AWSaccessKeys.csv"


if os.path.isfile(f"{HOME}/{KEYPATH}") == True:
    ACCESS = pd.read_csv(f"{HOME}/{KEYPATH}")

    #start session
    SESSION = boto3.Session(
        aws_access_key_id=ACCESS['Access key ID'][0],
        aws_secret_access_key=ACCESS['Secret access key'][0],
    )
    S3 = SESSION.resource('s3')
    #AWS BUCKET information
    BUCKET_NAME = 'national-snow-model'
    #S3 = boto3.resource('S3', config=Config(signature_version=UNSIGNED))
    BUCKET = S3.Bucket(BUCKET_NAME)
else:
    print(f"no AWS credentials present, {HOME}/{KEYPATH}")
    
#set multiprocessing limits
CPUS = len(os.sched_getaffinity(0))
CPUS = int((CPUS/2)-2)

#function for processing a single row -  test to see if this works
def cell_id_2_topography(row, timestamp,transposed_data,nearest_snotel, snotel_data):
    try:
        cell_id = row['cell_id']
        station_ids = nearest_snotel[cell_id]
        station_mapping = {old_id: f"ns_{i+1}" for i, old_id in enumerate(station_ids)}
        cols = ['station_id', timestamp]
        selected_snotel_data = snotel_data[cols]
        # # Rename the station IDs in the selected SNOTEL data
        selected_snotel_data['station_id'] = selected_snotel_data['station_id'].map(station_mapping)
        selected_snotel_data.dropna(subset =['station_id'], inplace = True)
        # # Transpose and set the index correctly
        transposed_data[cell_id] = selected_snotel_data.set_index('station_id').T

        transposed_data[cell_id].rename(columns = {'dates': 'Date'})

        
    except:
        print(f"{cell_id} throwing error, moving on...")
        

#function for processing a single timestamp
def process_single_timestamp(args):
    #get key variable from args
    aso_swe_files_folder_path, aso_swe_file, new_column_names, snotel_data, nearest_snotel , Obsdf, obsdf_path, output_res = args
    
    #Get timestamp
    timestamp = aso_swe_file.split('_')[-1].split('.')[0]
    #Get region
    region = aso_swe_file.split('_')[1]

    #load in SWE data from ASO
    aso_swe_data = pd.read_parquet(os.path.join(aso_swe_files_folder_path, aso_swe_file))

    #drop duplicate sites/spatial area, average the spatial area per cell_id
    aso_swe_data.reset_index(inplace=True)

    transposed_data = {}

    #print(new_column_names)

    if timestamp in new_column_names.values():
        print(f"Site processing complete, adding observtional data to {timestamp} df...")
        tqdm_notebook.pandas()
        aso_swe_data.progress_apply(lambda row: cell_id_2_topography(row, timestamp,transposed_data,nearest_snotel, snotel_data), axis =1)
        #print(transposed_data)
        #Convert dictionary of sites to dataframe
        transposed_df = pd.concat(transposed_data, axis=0) 

        #display(transposed_df)
        transposed_df.reset_index(inplace = True)
        transposed_df.rename(columns={'index': 'cell_id','station_id':"Date"}, inplace=True)
        transposed_df.rename(columns={'level_0': 'cell_id','dates':"Date"}, inplace=True)
        #display(transposed_df)
     

        # Reset index and rename columns
        #transposed_df.rename(columns={'level_0': 'cell_id', 'level_1': 'Date'}, inplace = True)
       # transposed_df.rename(columns={'index': 'cell_id'}, inplace = True)
        transposed_df['Date'] = pd.to_datetime(transposed_df['Date'])
        

        aso_swe_data['Date'] = pd.to_datetime(timestamp)
        aso_swe_data = aso_swe_data[['cell_id', 'Date', 'swe_m']]
        merged_df = pd.merge(aso_swe_data, transposed_df, how='left', on=['cell_id', 'Date'])

        Obsdf = pd.concat([Obsdf, merged_df], ignore_index=True)

    else:
        aso_swe_data['Date'] = pd.to_datetime(timestamp)
        aso_swe_data = aso_swe_data[['cell_id', 'Date', 'swe_m']]

        # No need to merge in this case, directly concatenate
        Obsdf = pd.concat([Obsdf, aso_swe_data], ignore_index=True)

    cols = [
    'cell_id', 'Date', 'swe_m', 'ns_1', 'ns_2', 'ns_3', 'ns_4', 'ns_5', 'ns_6'
    ]
    #display(Obsdf)
    Obsdf = Obsdf[cols]
    table = pa.Table.from_pandas(Obsdf)
    # Parquet with Brotli compression
    print(f"{obsdf_path}/{region}_{timestamp}_ObsDF.parquet")
    pq.write_table(table,f"{obsdf_path}/Obsdf_{region}_{timestamp}.parquet", compression='BROTLI')
  

def Nearest_Snotel_2_obs_MultiProcess(region, output_res, manual, dates):    

    print('Connecting site observations with nearest monitoring network obs')
    #get Snotel observations
    # snotel_path = f"{HOME}/SWEMLv2.0/data/SNOTEL_Data/"
    snotel_path = f"{HOME}/data/SNOTEL_Data/"
    #Snotelobs_path = f"{snotel_path}ground_measures.parquet"
    Snotelobs_path = f"{snotel_path}ground_measures_dp.parquet"
    # #nearest snotel path
    nearest_snotel_dict_path = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution"
    #ASO observations
    aso_swe_files_folder_path = f"{HOME}/data/ASO/{region}/{output_res}M_SWE_parquet"

     #Make folder for predictions
    # obsdf_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution/Obsdf"
    obsdf_path = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/Obsdf"

    if not os.path.exists(obsdf_path):
        os.makedirs(obsdf_path, exist_ok=True)

    #Get sites/snotel observations from 2013-2019
    print('Loading observations')
    try:
        snotel_data = pd.read_parquet(Snotelobs_path)
        snotel_data = snotel_data.T
        snotel_data.reset_index(inplace=True)
        snotel_data.rename(columns={'index':'station_id'}, inplace = True)
    except:
        print("Go run the get_InSitu_obs script above...")

    #Load dictionary of nearest sites
    print(f"Loading {output_res}M resolution grids for {region} region")
    with open(f"{nearest_snotel_dict_path}/nearest_SNOTEL.pkl", 'rb') as handle:
        nearest_snotel = pkl.load(handle)

    #Processing SNOTEL Obs to correct date/time
    print('Processing datetime component of SNOTEL observation dataframe')
    date_columns = snotel_data.columns[1:]
    new_column_names = {col: pd.to_datetime(col, format='%Y-%m-%d').strftime('%Y%m%d') for col in date_columns}
    snotel_data = snotel_data.rename(columns=new_column_names)
    
    #create dataframe
    if manual == False:
        aso_swe_files = [filename for filename in os.listdir(aso_swe_files_folder_path)]

    if manual == True:
        aso_swe_files = [f"ASO_{output_res}M_SWE_{date}.parquet" for date in dates]

    print(f"Loading {len(aso_swe_files)} processed ASO observations for the {region} at {output_res}M resolution")

    #Find out if we need to get more snotel obs...
    ts = []
    nots = []
    for aso_swe_file in aso_swe_files:
        timestamp = aso_swe_file.split('_')[-1].split('.')[0]

        if timestamp in new_column_names.values():
            ts.append(timestamp)
        else:
            nots.append(timestamp)
    nots = np.sort(nots)
    print(f"There are {len(ts)} aso dates in snotel obs")
    print(f"There are {len(nots)} missing snotel obs")
    
    #get missing snotel observations
    #create list of dates
    dates = []
    for date in nots:
        Y= date[:4]
        m = date[4:6]
        d= date[6:]
        dates.append(f"{Y}-{m}-{d}")

    #print(f"Getting CDEC and SNOTEL observations for the following dates: {dates}")
    #Getdata for missing dates
    #snotel_data = get_InSitu_obs.Get_Monitoring_Data_Threaded(dates)

    # date_columns = snotel_data.columns[1:]
    # new_column_names = {col: pd.to_datetime(col, format='%Y-%m-%d').strftime('%Y%m%d') for col in date_columns}
    # snotel_data_f = snotel_data.rename(columns=new_column_names)
    # snotel_data_f.reset_index(inplace=True)

    print(f"Connecting {len(aso_swe_files)} timesteps of observations for {region}")
    aso_swe_files.sort()
    Obsdf = pd.DataFrame()
    #using ProcessPool here because of the python function used (e.g., not getting data but processing it)
    with cf.ProcessPoolExecutor(max_workers=CPUS) as executor: #changed from none to 6 as it was overloading the cpu
        # Start the load operations and mark each future with its process function
        [executor.submit(process_single_timestamp, (aso_swe_files_folder_path, aso_swe_files[i], new_column_names, snotel_data, nearest_snotel, Obsdf, obsdf_path,output_res)) for i in tqdm(range(len(aso_swe_files)))]

    # for i in tqdm_notebook(range(len(aso_swe_files))):
    #     args = aso_swe_files_folder_path, aso_swe_files[i], new_column_names, snotel_data, nearest_snotel, Obsdf, obsdf_path,output_res
    #     process_single_timestamp(args)
                           


    print(f"Job complete for connecting SNOTEL obs to sites/dates")


