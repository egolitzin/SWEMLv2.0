"""
get_PRISM.py
Download multiple PRISM 800m daily variables, clip to a bounding box, and
concatenate into a single multi-band xarray Dataset per water year.

Available PRISM variables-
ppt     – precipitation (mm)
tmean   – mean 2-m air temperature (°C)
tmax    – maximum 2-m air temperature (°C)
tmin    – minimum 2-m air temperature (°C)
tdmean  – mean dew-point temperature (°C)
vpdmin  – minimum vapor-pressure deficit (hPa)
vpdmax  – maximum vapor-pressure deficit (hPa)

Output
data/Precipitation/{WY}/prism_800m/{WY}.zarr
  Dimensions : (time, lat, lon)
  Variables  : one per requested PRISM variable (renamed from Band1)

Usage
    from utils.get_PRISM import get_prism, add_prism_df

    # download ppt + tmean clipped to the western US
    bbox = (-125.0, 30.0, -100.0, 50.0)   # (lon_min, lat_min, lon_max, lat_max)
    get_prism(WY=2020, vars=['ppt', 'tmean'], bbox=bbox)
"""

import os
import re
import shutil
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
import zarr
from tqdm.auto import tqdm

# Default set of variables to download
PRISM_VARS = ['ppt', 'tmean', 'tmax', 'tmin', 'tdmean', 'vpdmin', 'vpdmax']
PRISM_BASE_URL = "https://services.nacse.org/prism/data/get/us/800m"
bbox_default = (-125.0, 30.0, -100.0, 50.0) #WUS

HOME = Path(__file__).resolve().parents[1]


def _progress_hook(block_num, block_size, total_size, t):
    downloaded = block_num * block_size
    if total_size > 0:
        t.update(min(block_size, total_size - t.n))
    else:
        t.update(downloaded - t.n)

def _date_from_filename(nc_path: str) -> str | None:
    """Extract YYYYMMDD from a PRISM filename like prism_ppt_us_4kmD2_20121001_bil.nc"""
    m = re.search(r'(\d{8})', Path(nc_path).name)
    return m.group(1) if m else None

def _download_var(start: datetime, stop: datetime, zipped_dir: Path, var: str) -> None:
    """Download daily PRISM zip files for one variable into zipped_dir."""
    zipped_dir.mkdir(parents=True, exist_ok=True)
    day = start
    while day <= stop:
        day_str = day.strftime('%Y%m%d')
        out_file = zipped_dir / day_str
        if not out_file.exists():
            url = f"{PRISM_BASE_URL}/{var}/{day_str}?format=nc"
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024,
                      miniters=1, desc=f'  {var} {day_str}', leave=False) as t:
                urllib.request.urlretrieve(
                    url, out_file,
                    reporthook=lambda bn, bs, ts: _progress_hook(bn, bs, ts, t),
                )
        day += timedelta(days=1)


def _unzip_var(start: datetime, stop: datetime,
               zipped_dir: Path, unzipped_dir: Path) -> None:
    """Unzip daily PRISM files and remove the zip archives."""
    unzipped_dir.mkdir(parents=True, exist_ok=True)
    day = start
    while day <= stop:
        day_str = day.strftime('%Y%m%d')
        zip_path = zipped_dir / day_str
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(unzipped_dir)
            zip_path.unlink()
        day += timedelta(days=1)


def _rh_from_dew(T_C, Td_C):
    """Relative humidity (%) from temperature and dew-point (Magnus formula)."""
    es = 6.112 * np.exp(17.67 * T_C  / (T_C  + 243.5))
    ed = 6.112 * np.exp(17.67 * Td_C / (Td_C + 243.5))
    return np.clip(ed / es * 100.0, 0.0, 100.0)


def _precip_phase(precip, T_C, RH):
    """Jennings et al. (2018) binary phase partition. Returns (snow, rain)."""
    P_snow = 1.0 / (1.0 + np.exp(-10.04 + 1.41 * T_C + 0.09 * RH))
    snow = np.where(P_snow >= 0.5, precip, 0.0).astype(np.float32)
    rain = np.where(P_snow < 0.5,  precip, 0.0).astype(np.float32)
    return snow, rain


# Load one variable as a time-indexed DataArray, clipped to bbox

def _load_var(unzipped_dir: Path, var: str,
              start: datetime, stop: datetime,
              bbox: tuple | None) -> xr.DataArray:
    """
    Open all daily NetCDF files for 'var', sort by date, clip to bbox, and
    return as a DataArray with a proper 'time' dimension.

    bbox : (lon_min, lat_min, lon_max, lat_max) or None for no clipping.
    """
    nc_files = sorted(unzipped_dir.glob('*.nc'))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {unzipped_dir}")

    # Build a mapping date_str -> file, then filter to WY range
    date_map = {}
    for f in nc_files:
        d = _date_from_filename(str(f))
        if d is not None:
            date_map[d] = f

    expected_dates = pd.date_range(start, stop).strftime('%Y%m%d').tolist()
    missing = [d for d in expected_dates if d not in date_map]
    if missing:
        raise FileNotFoundError(
            f"PRISM '{var}': {len(missing)} daily files missing "
            f"(first missing: {missing[0]}). "
            "Re-run download or set on_missing='interpolate'."
        )

    ordered_files = [str(date_map[d]) for d in expected_dates]
    time_index = pd.date_range(start, stop, freq='D')

    # Open with mfdataset, selecting only Band1 to avoid 'crs' dimension issues
    ds = xr.open_mfdataset(
        ordered_files,
        combine='nested',
        concat_dim='time',
        data_vars=['Band1'],
    ).assign_coords(time=time_index)

    da = ds['Band1'].rename(var)

    # Clip to bounding box (lat/lon are 1-D and ascending)
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        da = da.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )

    return da


def get_prism(WY: int,
              vars: list = PRISM_VARS,
              bbox: tuple | None = bbox_default,
              home: Path | str = HOME,
              overwrite: bool = False) -> Path:
    """
    Download PRISM daily data for water year WY, clip to bbox, and save
    a single multi-variable Zarr store (one variable written at a time).

    Parameters
    WY       : Water year (e.g. 2020 → Oct 2019 – Sep 2020).
    vars     : List of PRISM variable names to download.
               Defaults to all seven: ppt tmean tmax tmin tdmean vpdmin vpdmax.
    bbox     : (lon_min, lat_min, lon_max, lat_max) in WGS-84.
               Pass None to keep the full CONUS domain.
    home     : Project root directory (default: parent of this file).
    overwrite: Re-download and overwrite existing output if True.

    Returns: Path to the output Zarr store.
    """
    home = Path(home)
    # prism_path = Path('/uufs/chpc.utah.edu/common/home/johnsonrc-group1/PRISM/800m')
    prism_path = home / 'data' / 'Precipitation' / str(WY) / 'prism_800m'
    out_store  = prism_path / f'{WY}.zarr'

    prism_path.mkdir(parents=True, exist_ok=True)

    if overwrite and out_store.exists():
        shutil.rmtree(out_store)

    start = datetime(WY-1,10,1)
    stop  = datetime(WY,9,30)

    store_attrs = {
        'description': (f'PRISM 800-m daily data for water year {WY} '
                        f'(Oct {WY-1} – Sep {WY})'),
        'variables':   str(vars),
        'bbox':        str(bbox),
        'source':      PRISM_BASE_URL,
        'created':     datetime.now().isoformat(),
    }

    for var in vars:
        # skip if variable already written and not overwriting
        if not overwrite and out_store.exists():
            existing = zarr.open_group(str(out_store), mode='r')
            if var in existing:
                print(f'[PRISM WY{WY}] {var} already in store, skipping')
                continue

        print(f'[PRISM WY{WY}] Variable: {var}')
        zipped_dir   = prism_path / str(WY) /'zipped'   / var
        unzipped_dir = prism_path / str(WY) /'unzipped' / var

        # check again to see if files are downloaded and unzipped
        n_expected = len(pd.date_range(start, stop))
        if len(list(unzipped_dir.glob('*.nc'))) < n_expected:
            _download_var(start, stop, zipped_dir, var)
            _unzip_var(start, stop, zipped_dir, unzipped_dir)

        da = _load_var(unzipped_dir, var, start, stop, bbox)
        ds_var = da.to_dataset()

        # first var creates the store; subsequent vars append
        mode = 'w' if not out_store.exists() else 'a'
        ds_var.to_zarr(out_store, mode=mode)
        print(f'[PRISM WY{WY}] {var} written to store')

    # write global attrs
    z = zarr.open_group(str(out_store), mode='a')
    z.attrs.update(store_attrs)

    print(f'[PRISM WY{WY}] Saved → {out_store}')
    return out_store


def add_prism_df(WY: int,
                 output_res: int,
                 threshold: int,
                 home: Path | str = HOME) -> None:
    """
    Join PRISM precipitation, phase-partitioned snow/rain, and ablation
    features to training DataFrames by nearest-neighbor lookup on lat/lon.

    Requires ppt, tmean, and tdmean in the PRISM Zarr store.

    New columns:
        PRISM_precip  - season-to-date total precipitation (mm)
        PRISM_snow    - season-to-date snow via Jennings (2018) (mm)
        PRISM_rain    - season-to-date rain via Jennings (2018) (mm)
        PRISM_APDD    - accumulated positive degree-days since Oct 1 (deg C days)
        PRISM_T7d     - 7-day mean temperature ending on obs date (deg C)
        PRISM_T14d    - 14-day mean temperature ending on obs date (deg C)
    """
    home = Path(home)
    training_df_path = (home / 'data' / 'TrainingDFs' / str(WY)
                        / f'{output_res}M_Resolution'
                        / 'AORC_NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs'
                        / f'{threshold}_fSCA_Thresh')
    df_path = (home / 'data' / 'TrainingDFs' / str(WY)
               / f'{output_res}M_Resolution'
               / 'PRISM_AORC_NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs'
               / f'{threshold}_fSCA_Thresh')
    df_path.mkdir(parents=True, exist_ok=True)

    training_dfs = [f for f in os.listdir(training_df_path) if f.endswith('.parquet')]

    # prism_store = home / 'data' / 'Precipitation' / str(WY) / 'prism_800m' / f'{WY}.zarr'
    prism_store = Path(f'/uufs/chpc.utah.edu/common/home/johnsonrc-group1/PRISM/800m/{WY}.zarr')

    obs_precip = xr.open_zarr(prism_store)
    missing_vars = [v for v in ('ppt', 'tmean', 'tdmean') if v not in obs_precip]
    if missing_vars:
        obs_precip.close()
        raise KeyError(
            f"PRISM Zarr store missing required variables: {missing_vars}. "
            "Run get_prism(WY, vars=['ppt','tmean','tdmean']) first."
        )
    WY_start = f'{WY-1}-10-01'
    lat_asc  = obs_precip.lat.values[0] < obs_precip.lat.values[-1]

    def _extract(var, strdate, lat_s, left, right, lats, lons):
        return obs_precip[var].sel(
            time=slice(WY_start, strdate),
            lat=lat_s,
            lon=slice(left, right),
        ).sel(lat=lats, lon=lons, method='nearest').values

    try:
        for geofile in tqdm(training_dfs, desc='Processing PRISM files'):
            date   = geofile.split('_')[-1].split('.parquet')[0]
            region = geofile.split('_')[-2]
            strdate = f'{date[:4]}-{date[4:6]}-{date[6:]}'
            print(f'  Connecting PRISM to ASO observations on {strdate} at {region}')

            GDF  = pd.read_parquet(training_df_path / geofile)
            meta = GDF[['cell_id', 'cen_lat', 'cen_lon']]
            GDF.set_index('cell_id', inplace=True)

            left, right  = meta['cen_lon'].min() - 0.1, meta['cen_lon'].max() + 0.1
            bottom, top  = meta['cen_lat'].min() - 0.1, meta['cen_lat'].max() + 0.1
            lat_s        = slice(bottom, top) if lat_asc else slice(top, bottom)

            lats = xr.DataArray(meta['cen_lat'].values, dims='points')
            lons = xr.DataArray(meta['cen_lon'].values, dims='points')

            ppt_vals    = _extract('ppt',    strdate, lat_s, left, right, lats, lons)
            tmean_vals  = _extract('tmean',  strdate, lat_s, left, right, lats, lons)
            tdmean_vals = _extract('tdmean', strdate, lat_s, left, right, lats, lons)

            RH = _rh_from_dew(tmean_vals, tdmean_vals)
            snow_arr, rain_arr = _precip_phase(ppt_vals, tmean_vals, RH)

            GDF['PRISM_precip'] = np.round(ppt_vals.sum(axis=0),                    2)
            GDF['PRISM_snow']   = np.round(snow_arr.sum(axis=0),                    2)
            GDF['PRISM_rain']   = np.round(rain_arr.sum(axis=0),                    2)
            GDF['PRISM_APDD']   = np.round(np.maximum(tmean_vals, 0).sum(axis=0),   2)
            GDF['PRISM_T7d']    = np.round(tmean_vals[-7:].mean(axis=0),            2)
            GDF['PRISM_T14d']   = np.round(tmean_vals[-14:].mean(axis=0),           2)

            GDF.reset_index(inplace=True)

            table = pa.Table.from_pandas(GDF)
            pq.write_table(table, df_path / f'PRISM_{geofile}', compression='BROTLI')
    finally:
        obs_precip.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Download and clip PRISM daily data into per-water-year Zarr stores.'
    )
    parser.add_argument('--wy', type=int, nargs='+', required=True,
                        help='Water year(s) to process, e.g., --wy 2019 2020 2021')
    parser.add_argument('--vars', nargs='+', default=PRISM_VARS,
                        help='PRISM variable(s) to download. Default: all seven.')
    parser.add_argument('--bbox', type=float, nargs=4,
                        default=list(bbox_default),
                        help='Bounding box: lon_min lat_min lon_max lat_max. Default: western US.')
    parser.add_argument('--full_domain', action='store_true',
                        help='Disable bbox clipping and keep the full CONUS domain.')
    parser.add_argument('--home', default=str(HOME),
                        help=f'Project root directory (default: {HOME})')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-download and overwrite existing output files.')
    args = parser.parse_args()

    bbox = None if args.full_domain else tuple(args.bbox)

    for wy in sorted(args.wy):
        get_prism(
            WY=wy,
            vars=args.vars,
            bbox=bbox,
            home=args.home,
            overwrite=args.overwrite,
        )
