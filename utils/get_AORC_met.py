import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

#data packages
import s3fs

#multiprocessing
from tqdm.auto import tqdm
import concurrent.futures as cf


import os
from datetime import datetime, timedelta

HOME = os.getcwd()


def partition_precip(ds):
    """
    Partition APCP_surface into snow and rain using the Jennings et al. (2018)
    bivariate logistic regression on air temperature and relative humidity.

    Jennings, K.S., Winchell, T.S., Livneh, B., Molotch, N.P. (2018).
    Spatial variation of the rain–snow temperature threshold across the Northern Hemisphere.
    Nature Communications, 9, 1148. https://doi.org/10.1038/s41467-018-03629-7

    Model (Supplementary Table 2, all-sites):
        logit(P_snow) = -10.04 + 1.41*T(°C) + 0.09*RH(%)
    P_snow is used as a continuous fractional weight so APCP is split proportionally
    rather than assigned binary phase per hour.

    RH is derived from TMP_2maboveground, SPFH_2maboveground, and PRES_surface.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain APCP_surface, TMP_2maboveground, SPFH_2maboveground, PRES_surface.

    Returns
    -------
    snow, rain : xr.DataArray, xr.DataArray
    """
    T_C   = ds['TMP_2maboveground'] - 273.15       # K → °C
    P_hPa = ds['PRES_surface'] / 100.0             # Pa → hPa
    q     = ds['SPFH_2maboveground']               # specific humidity (kg/kg)

    # Saturation vapor pressure (hPa) — Bolton (1980)
    es = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    # Actual vapor pressure from specific humidity
    e  = q * P_hPa / (0.622 + 0.378 * q)
    RH = (e / es * 100.0).clip(0.0, 100.0)         # relative humidity (%)

    P_snow = 1.0 / (1.0 + np.exp(-10.04 + 1.41 * T_C + 0.09 * RH))

    snow = xr.where(P_snow>=0.5,ds['APCP_surface'],0)
    # rain = ds['APCP_surface'] * (1.0 - P_snow)
    rain = xr.where(P_snow<0.5,ds['APCP_surface'],0)
    return snow, rain


def partition_precip_wetbulb(ds):
    """
    Partition APCP_surface into snow and rain using wet-bulb temperature with a sigmoidal
    fraction of snow. 

    Wet-bulb temperature is estimated from air temperature and relative humidity
    using the Stull (2011) empirical approximation (valid for T in [-20, 50] °C,
    RH in [5, 99] %):

        Tw = T·atan[0.151977(RH+8.313659)^0.5] + atan(T+RH)
           - atan(RH-1.676331) + 0.00391838·RH^1.5·atan(0.023101·RH) - 4.686035

    Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity and Air Temperature.
    Journal of Applied Meteorology and Climatology, 50(11), 2267–2269.
    https://doi.org/10.1175/JAMC-D-11-0143.1

    Wang, Y.-H., Broxton, P., Fang, Y., Behrangi, A., Barlage, M., Zeng, X., & Niu, G.-Y. (2019). 
    A Wet-Bulb Temperature-Based Rain-Snow Partitioning Scheme Improves Snowpack Prediction Over 
    the Drier Western United States. Geophysical Research Letters, 46(23), 13825–13835. 
    https://doi.org/10.1029/2019GL085722

    Parameters
    ----------
    ds : xr.Dataset
        Must contain APCP_surface, TMP_2maboveground, SPFH_2maboveground, PRES_surface.

    Returns
    -------
    snow, rain : xr.DataArray, xr.DataArray
    """
    T_C   = ds['TMP_2maboveground'] - 273.15
    P_hPa = ds['PRES_surface'] / 100.0
    q     = ds['SPFH_2maboveground']

    # Relative humidity (%) 
    es = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    e  = q * P_hPa / (0.622 + 0.378 * q)
    RH = (e / es * 100.0).clip(0.0, 100.0)

    # Wet-bulb temperature (°C) - Stull (2011)
    Tw = (T_C * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
          + np.arctan(T_C + RH)
          - np.arctan(RH - 1.676331)
          + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
          - 4.686035)

    a = 6.99e-5         # dimensionless
    b = 2.0             # K^-1
    c = 3.97            # K
    P_snow = 1.0 / (1.0 + a*np.exp(b*(Tw + c)))

    snow = ds['APCP_surface'] * P_snow
    rain = ds['APCP_surface'] * (1.0 - P_snow)
    return snow, rain, Tw, P_snow


def get_aorc_wy_daily(WY, output_res, thresh, save_path=None):
    """
    Stream hourly AORC data for a water year from S3, partition precipitation into
    snow and rain using a temperature threshold, aggregate to daily resolution,
    and save the result as a zarr store.

    Daily output variables:
        APCP_surface        - Total precipitation daily sum (mm)
        SNOW_regression     - Snow daily sum via Jennings et al. (2018) logistic regression (mm)
        RAIN_regression     - Rain daily sum via Jennings et al. (2018) logistic regression (mm)
        SNOW_wetbulb        - Snow daily sum via Stull (2011) wet-bulb temp (mm)
        RAIN_wetbulb        - Rain daily sum via Stull (2011) wet-bulb temp (mm)
        DSWRF_surface       - Downward shortwave radiation daily mean (W/m^2)
        DLWRF_surface       - Downward longwave radiation daily mean (W/m^2)
        PRES_surface        - Pressure daily mean (Pa)
        TMP_2maboveground   - Air temperature daily mean (K)
        SPFH_2maboveground  - Specific humidity daily mean (kg/kg)
        UGRD_10maboveground - Zonal wind daily mean (m/s)
        VGRD_10maboveground - Meridional wind daily mean (m/s)

    The spatial domain is the union bounding box of all training DF files for the WY.
    Hourly data is streamed from S3 and never stored locally.

    Parameters
    ----------
    WY : int
        Water year 
    output_res : int
        Output spatial resolution in m 
    thresh : int
        fSCA threshold 
    save_path : str, optional
        Path for the output zarr store.
        Defaults to {HOME}/data/Precipitation/AORC/{WY}/AORC_daily_WY{WY}.zarr
    """
    WY_start = f"{WY - 1}-10-01"
    WY_end   = f"{WY}-09-30"
    yrs = [WY - 1, WY]

    if save_path is None:
        save_path = os.path.join(HOME, "data", "Precipitation", str(WY), "AORC", f"AORC_daily_WY{WY}.zarr")

    if os.path.exists(save_path):
        print(f"AORC daily WY{WY} dataset already exists at {save_path}")
        return xr.open_zarr(save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Derive unified bounding box from all training DF files 
    training_df_dir = f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/VIIRSGeoObsDFs/{thresh}_fSCA_Thresh"
    files = [f for f in os.listdir(training_df_dir) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No training DF parquet files found in {training_df_dir}")

    all_lons, all_lats = [], []
    for file in files:
        df = pd.read_parquet(os.path.join(training_df_dir, file), columns=['cen_lat', 'cen_lon'])
        all_lons.extend([df['cen_lon'].min(), df['cen_lon'].max()])
        all_lats.extend([df['cen_lat'].min(), df['cen_lat'].max()])

    left   = min(all_lons) - 0.1
    right  = max(all_lons) + 0.1
    bottom = min(all_lats) - 0.1
    top    = max(all_lats) + 0.1
    print(f"Derived bbox from {len(files)} training DFs: lon [{left:.2f}, {right:.2f}], lat [{bottom:.2f}, {top:.2f}]")

    print(f"Connecting to AORC S3 for WY{WY} ({WY_start} to {WY_end}) ...")
    s3 = s3fs.S3FileSystem(anon=True)
    fileset = [
        s3fs.S3Map(root=f"noaa-nws-aorc-v1-1-1km/{yr}.zarr", s3=s3, check=False)
        for yr in yrs
    ]

    # chunks={} uses native AORC zarr tile sizes w/ no S3 object sub-division
    ds = xr.open_mfdataset(fileset, engine="zarr", consolidated=True, chunks={})
    lat_asc = ds.latitude.values[0] < ds.latitude.values[-1]
    lat_s   = slice(bottom, top) if lat_asc else slice(top, bottom)
    ds_wy = ds.sel(
        time=slice(WY_start, WY_end),
        latitude=lat_s,
        longitude=slice(left, right)
    )

    # Snow/rain partitioning: Jennings et al. (2018) bivariate logistic regression
    ds_wy['SNOW_regression'], ds_wy['RAIN_regression'] = partition_precip(ds_wy)

    # Snow/rain partitioning: Wang et al (2019) Tw partitioning 
    # Tw, P_snow
    ds_wy['SNOW_wetbulb'], ds_wy['RAIN_wetbulb'], ds_wy['Tw'], ds_wy['P_snow'] = partition_precip_wetbulb(ds_wy)

    # Aggregate to daily: sums for precip, means for everything else
    precip_vars = ['APCP_surface',
                   'SNOW_regression', 'RAIN_regression',
                   'SNOW_wetbulb',    'RAIN_wetbulb']
    mean_vars   = ['DSWRF_surface', 'DLWRF_surface', 'PRES_surface',
                   'TMP_2maboveground', 'SPFH_2maboveground',
                   'UGRD_10maboveground', 'VGRD_10maboveground',
                   'Tw','P_snow']

    ds_daily_precip = ds_wy[precip_vars].resample(time='1D').sum()  # kg/m^2 = mm
    ds_daily = xr.merge([
        ds_daily_precip,
        ds_wy[mean_vars].resample(time='1D').mean(),
    ])

    # Rechunk to uniform sizes before writing.
    # Zarr requires uniform chunks on all but the last chunk,
    # so we consolidate here.  30-day time chunks keep peak memory ~1 GB/pass.
    # Spatial chunks (128 lat × 256 lon) match the AORC native zarr tile size.
    ds_daily = ds_daily.chunk({'time': 30, 'latitude': 128, 'longitude': 256})

    # Stream hourly data from S3, compute, and write daily zarr
    print(f"Processing and saving daily AORC dataset for WY{WY} to {save_path} ...")
    ds_daily.to_zarr(save_path, mode="w", consolidated=True)
    print("Done.")

    return xr.open_zarr(save_path)

## AORC functions adapted from CUAHSI HydroShare library - thank you!
def get_aorc_precip(WY, output_res, thresh):
    """
    Fetch AORC hourly precipitation for a water year, aggregate to daily sums,
    and write per-observation parquet files.

    Memory strategy: compute the union bounding box of all unprocessed training DFs
    upfront, load the entire WY daily APCP for that bbox into memory in one chunked
    S3 read (~0.5 GB for a typical western-US study area), then serve all per-file
    extractions from the in-memory array.
    """
    WY_start = datetime(WY-1, 10, 1)
    obs_start = WY_start.strftime('%Y-%m-%d')
    print("Water Year start date:", obs_start)

    yrs = [WY-1, WY]

    training_df_dir = f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/VIIRSGeoObsDFs/{thresh}_fSCA_Thresh"
    all_files = [f for f in os.listdir(training_df_dir) if f.endswith(".parquet")]
    precip_data_path = f"{HOME}/data/Precipitation/{WY}/{output_res}M_AORC_Precip"
    os.makedirs(precip_data_path, exist_ok=True)

    # Only work on files that haven't been written yet
    todo = [
        f for f in all_files
        if not os.path.exists(
            f"{precip_data_path}/AORC_{f.split('_')[-2]}_{f.split('_')[-1].split('.')[0]}.parquet"
        )
    ]
    if not todo:
        print("All files already processed.")
        return

    # --- Union bounding box across all remaining files ---
    all_lons, all_lats = [], []
    for file in todo:
        df = pd.read_parquet(os.path.join(training_df_dir, file), columns=['cen_lat', 'cen_lon'])
        all_lons.extend([df['cen_lon'].min(), df['cen_lon'].max()])
        all_lats.extend([df['cen_lat'].min(), df['cen_lat'].max()])
    bbox_left   = min(all_lons) - 0.1
    bbox_right  = max(all_lons) + 0.1
    bbox_bottom = min(all_lats) - 0.1
    bbox_top    = max(all_lats) + 0.1
    print(f"Union bbox: lon [{bbox_left:.3f}, {bbox_right:.3f}], lat [{bbox_bottom:.3f}, {bbox_top:.3f}]")

    # --- Single S3 read for the union bbox ---
    # chunks={} uses native AORC zarr tile sizes (avoids sub-dividing S3 objects).
    # .load() then materializes only the small union-bbox subset (~0.5 GB).
    base_url = 's3://noaa-nws-aorc-v1-1-1km'
    s3_out = s3fs.S3FileSystem(anon=True)
    fileset = [
        s3fs.S3Map(root=f"s3://{base_url}/{yr}.zarr", s3=s3_out, check=False)
        for yr in yrs
    ]
    print("Streaming AORC daily precip from S3 (union bbox, chunked) ...")
    ds_yrs = xr.open_mfdataset(fileset, engine='zarr', chunks={})
    lat_asc  = ds_yrs.latitude.values[0] < ds_yrs.latitude.values[-1]
    bbox_lat = slice(bbox_bottom, bbox_top) if lat_asc else slice(bbox_top, bbox_bottom)
    da_bbox_full = (
        ds_yrs['APCP_surface']
        .sel(latitude=bbox_lat,
             longitude=slice(bbox_left, bbox_right))
        .resample(time='1D').sum()
    )
    da_loaded = da_bbox_full.load()   # ~0.5 GB for typical region
    ds_yrs.close()
    print(f"Loaded daily APCP: shape {da_loaded.shape}, "
          f"{da_loaded.nbytes / 1e9:.2f} GB")

    # --- Per-file extraction from in-memory array ---
    for file in tqdm(todo, desc="Processing AORC precip files"):
        timestamp = file.split('_')[-1].split('.')[0]
        region    = file.split('_')[-2]
        obs_end   = f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:]}'
        out_path  = f"{precip_data_path}/AORC_{region}_{timestamp}.parquet"

        print(f"Getting precipitation data for {obs_end} at {region}")

        training_df = pd.read_parquet(os.path.join(training_df_dir, file))
        left   = training_df['cen_lon'].min() - 0.1
        right  = training_df['cen_lon'].max() + 0.1
        bottom = training_df['cen_lat'].min() - 0.1
        top    = training_df['cen_lat'].max() + 0.1

        # Spatial + temporal slice from in-memory array (pure numpy, no S3)
        lat_s    = slice(bottom, top) if lat_asc else slice(top, bottom)
        da_slice = da_loaded.sel(
            time=slice(obs_start, obs_end),
            latitude=lat_s,
            longitude=slice(left, right),
        )

        meta = training_df[['cell_id', 'cen_lat', 'cen_lon']]
        lats = xr.DataArray(meta['cen_lat'].values, dims='points')
        lons = xr.DataArray(meta['cen_lon'].values, dims='points')

        prcp_all = da_slice.sel(latitude=lats, longitude=lons, method='nearest')
        raw_vals = prcp_all.values          # shape (time, npoints)
        season_precip_mm = np.round(raw_vals.sum(axis=0), 2)

        precip_df = pd.DataFrame({
            'cell_id':          meta['cell_id'].values,
            'cen_lat':          meta['cen_lat'].values,
            'cen_lon':          meta['cen_lon'].values,
            'precip':           list(raw_vals.T),
            'season_precip_mm': season_precip_mm,
        })

        table = pa.Table.from_pandas(precip_df)
        pq.write_table(table, out_path, compression='BROTLI')

    del da_loaded


def add_aorc_df(WY, output_res, threshold):
    """
    Add season-to-date AORC precipitation, snow, and rain to each training DF
    using the daily zarr store produced by get_aorc_wy_daily().

    Reads from:
        {HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/
            NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/
            {threshold}_fSCA_Thresh/

    Writes to:
        {HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/
            AORC_NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/
            {threshold}_fSCA_Thresh/

    New columns added to each DF
    -----------------------------
    AORC_precip       : season-to-date total precipitation (mm)
    AORC_snow_reg     : season-to-date snow via Jennings et al. (2018) logistic regression (mm)
    AORC_rain_reg     : season-to-date rain via Jennings et al. (2018) logistic regression (mm)
    AORC_snow_wb      : season-to-date snow via Stull (2011) wet-bulb linear ramp (mm)
    AORC_rain_wb      : season-to-date rain via Stull (2011) wet-bulb linear ramp (mm)
    AORC_APDD         : accumulated positive degree-days since Oct 1 (°C·days); sum of max(Tmean, 0)
    AORC_T7d          : 7-day mean air temperature ending on obs date (°C)
    AORC_T14d         : 14-day mean air temperature ending on obs date (°C)
    AORC_SW7d         : 7-day mean downward shortwave radiation ending on obs date (W/m²)
    AORC_SW14d        : 14-day mean downward shortwave radiation ending on obs date (W/m²)

    Parameters
    ----------
    WY : int
        Water year.
    output_res : int
        Spatial resolution in metres.
    threshold : int or str
        fSCA threshold used in training DF path.
    """
    training_df_path = (
        f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/"
        f"NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/"
        f"{threshold}_fSCA_Thresh"
    )
    df_path = (
        f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/"
        f"AORC_NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/"
        f"{threshold}_fSCA_Thresh"
    )
    os.makedirs(df_path, exist_ok=True)

    zarr_path = os.path.join(HOME, "data", "Precipitation", str(WY), "AORC", f"AORC_daily_WY{WY}.zarr")
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"AORC daily zarr not found at {zarr_path}. "
            f"Run get_aorc_wy_daily({WY}, {output_res}, {threshold}) first."
        )

    print(f"Opening AORC daily zarr for WY{WY} from {zarr_path}")
    ds_aorc = xr.open_zarr(zarr_path)
    lat_asc  = ds_aorc.latitude.values[0] < ds_aorc.latitude.values[-1]

    WY_start = f"{WY-1}-10-01"

    training_dfs = [f for f in os.listdir(training_df_path) if f.endswith('.parquet')]

    for geofile in tqdm(training_dfs, desc="Adding AORC met to training DFs"):
        date   = geofile.split('_')[-1].split('.parquet')[0]
        region = geofile.split('_')[-2]
        year, mon, day = date[:4], date[4:6], date[6:]
        strdate = f"{year}-{mon}-{day}"
        print(f"Adding AORC season-to-date met to {region} on {strdate}")

        GDF  = pd.read_parquet(os.path.join(training_df_path, geofile))
        meta = GDF[['cell_id', 'cen_lat', 'cen_lon']]
        GDF.set_index('cell_id', inplace=True)

        # Spatial bbox with padding
        left   = meta['cen_lon'].min() - 0.1
        right  = meta['cen_lon'].max() + 0.1
        bottom = meta['cen_lat'].min() - 0.1
        top    = meta['cen_lat'].max() + 0.1

        aorc_vars = ['APCP_surface', 'SNOW_regression', 'RAIN_regression', 'SNOW_wetbulb', 'RAIN_wetbulb',
                    'TMP_2maboveground', 'DSWRF_surface']
        lat_s    = slice(bottom, top) if lat_asc else slice(top, bottom)
        ds_slice = ds_aorc[aorc_vars].sel(
            time=slice(WY_start, strdate),
            latitude=lat_s,
            longitude=slice(left, right),
        )

        lats = xr.DataArray(meta['cen_lat'].values, dims='points')
        lons = xr.DataArray(meta['cen_lon'].values, dims='points')

        pts = ds_slice.sel(latitude=lats, longitude=lons, method='nearest')

        GDF['AORC_precip']   = np.round(pts['APCP_surface'].values.sum(axis=0), 2)
        GDF['AORC_snow_reg'] = np.round(pts['SNOW_regression'].values.sum(axis=0), 2)
        GDF['AORC_rain_reg'] = np.round(pts['RAIN_regression'].values.sum(axis=0), 2)
        GDF['AORC_snow_wb']  = np.round(pts['SNOW_wetbulb'].values.sum(axis=0), 2)
        GDF['AORC_rain_wb']  = np.round(pts['RAIN_wetbulb'].values.sum(axis=0), 2)
        
        T_C  = pts['TMP_2maboveground'].values - 273.15   # (n_days, n_points), °C
        SW   = pts['DSWRF_surface'].values                 # (n_days, n_points), W/m²

        GDF['AORC_APDD']  = np.round(np.maximum(T_C, 0).sum(axis=0), 2)
        GDF['AORC_T7d']   = np.round(T_C[-7:].mean(axis=0), 2)
        GDF['AORC_T14d']  = np.round(T_C[-14:].mean(axis=0), 2)
        GDF['AORC_SW7d']  = np.round(SW[-7:].mean(axis=0), 2)
        GDF['AORC_SW14d'] = np.round(SW[-14:].mean(axis=0), 2)

        GDF.reset_index(inplace=True)

        table = pa.Table.from_pandas(GDF)
        pq.write_table(table, f"{df_path}/AORC_{geofile}", compression='BROTLI')

    ds_aorc.close()