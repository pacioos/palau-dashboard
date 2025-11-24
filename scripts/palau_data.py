import numpy as np
import requests
import xarray as xr
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import os
import calendar
from netCDF4 import num2date
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dateutil.relativedelta import relativedelta
from datetime import date
import cftime

# Palau lat/lon
lat = 7.5150
lon = 134.5825

min_lat_ssh, max_lat_ssh = 7.0, 8.0
min_lon_ssh, max_lon_ssh = 134.0, 135.0

os.makedirs("./data", exist_ok=True)
os.makedirs("./data_files", exist_ok=True)

def download_file(url, local_path, retries=5):
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        raise_on_status=False
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded: {url}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {url}\n{e}")


columns = [
    "LastMonth", "LastMonthDate",
    "Forecast", "ForecastDate",
    "Outlook", "OutlookDate"
]

df = pd.DataFrame(columns=columns)
HST = ZoneInfo("Pacific/Honolulu")

now_hst = datetime.now(HST)
today_str = now_hst.strftime("%Y%m%d")
yest_str  = (now_hst - timedelta(days=1)).strftime("%Y%m%d")

# --- Last month/year (robust across year boundaries) ---
prev_month_dt = (
    now_hst.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    - timedelta(days=1)
)

last_month = prev_month_dt.month
last_year  = prev_month_dt.year

months_since_1960 = (last_year - 1960) * 12 + (last_month - 1)
t_value = months_since_1960 + 0.5

url = "https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.CAMS_OPI/.v0208/.anomaly_9120/.prcp/T/%28days%20since%201960-01-01%29streamgridunitconvert/T/differential_mul/T/%28months%20since%201960-01-01%29streamgridunitconvert//units/%28mm/month%29def//long_name/%28Precipitation%20Anomaly%29def/DATA/-500/-450/-400/-350/-300/-250/-200/-150/-100/-50/-25/25/50/100/150/200/250/300/350/400/450/500/VALUES/prcp_anomaly_max500_colors2/dods"
ds = xr.open_dataset(url, decode_times=False)

time_vals = ds["T"].values
units = ds["T"].attrs.get("units", "months since 1960-01-01")
calendar = ds["T"].attrs.get("calendar", "360_day")

if calendar == "360":
    calendar = "360_day"

decoded_time = cftime.num2date(time_vals, units=units, calendar=calendar)

ds = ds.assign_coords(T=("T", decoded_time))

lat_sel = 7.5
lon_sel = 134.5

point = ds.sel(Y=lat_sel, X=lon_sel, method="nearest")
last_time = ds["T"].values[-1]
print("Last time:", last_time)

last_month_ds = ds.sel(T=last_time)

# Extract value for last month at the nearest grid cell to your point
rf_last = point.sel(T=last_time)["aprod"].values.item()
rf_last_in = rf_last / 25.4
month = last_time.strftime("%B %Y")
print(f"Rainfall anomaly for {month}: {rf_last_in:.2f} in")
df.loc["Rain", "LastMonth"] = float(rf_last_in)
df.loc["Rain", "LastMonthDate"] = month

#Rain forecast
url = f'https://www.cpc.ncep.noaa.gov/products/people/mchen/CFSv2FCST/weekly/data/CFSv2.prec.{yest_str}.wkly.anom.nc'
filename = "./data_files/rf.forecast.nc"

response = requests.get(url)
if response.status_code == 200:
    with open(filename, 'wb') as f:
        f.write(response.content)
else:
    print(f"Failed to download file. Status code: {response.status_code}")
    
rf_forecast_dataset = xr.open_dataset(filename)
rf_forecast_palau = rf_forecast_dataset['anom'].sel(lat=slice(min_lat_ssh,max_lat_ssh),lon=slice(min_lon_ssh,max_lon_ssh))
rf_forecast_palau_df = rf_forecast_palau.to_dataframe().reset_index()
rf_forecast_value = rf_forecast_palau_df['anom'].iloc[1]
rf_forecast_value_in = rf_forecast_value/25.4
rf_forecast_date = rf_forecast_palau_df['time'].iloc[1]

df.loc["Rain", "Forecast"] = rf_forecast_value_in
df.loc["Rain", "ForecastDate"] = rf_forecast_date.strftime("%B %d, %Y")

#Rain outlook
url = "https://access-s.clide.cloud/files/global/monthly/data/rain.forecast.anom.monthly.nc"
filename = "./data_files/rf.outlook.nc"

response = requests.get(url)
if response.status_code == 200:
    with open(filename, 'wb') as f:
        f.write(response.content)
else:
    print(f"Failed to download file. Status code: {response.status_code}")

rf_outlook_dataset = xr.open_dataset(filename)
rf_outlook_palau = rf_outlook_dataset['rain'].sel(lat=slice(min_lat_ssh,max_lat_ssh),lon=slice(min_lon_ssh,max_lon_ssh))

rf_outlook_palau_df = rf_outlook_palau.to_dataframe().reset_index()
rf_outlook_value = rf_outlook_palau_df['rain'].iloc[0]
rf_outlook_value_in = rf_outlook_value/25.4
rf_outlook_time = rf_outlook_palau_df['time'].iloc[0]

df.loc["Rain", "Outlook"] = rf_outlook_value_in
df.loc["Rain", "OutlookDate"] = rf_outlook_time.strftime("%B %Y")

#Temp last month
url = "https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.CAMS/.anomaly/.temp_9120/dods"
ds = xr.open_dataset(url, decode_times=False)

time_vals = ds["T"].values
units = ds["T"].attrs.get("units", "months since 1960-01-01")
calendar = ds["T"].attrs.get("calendar", "360_day")

if calendar == "360":
    calendar = "360_day"

decoded_time = cftime.num2date(time_vals, units=units, calendar=calendar)

ds = ds.assign_coords(T=("T", decoded_time))

lat_sel = 7.5
lon_sel = 134.5

point = ds.sel(Y=lat_sel, X=lon_sel, method="nearest")
last_time = ds["T"].values[-1]

last_month_ds = ds.sel(T=last_time)

# Extract value for last month at the nearest grid cell to your point
tanom_last = point.sel(T=last_time)["temp_9120"].values.item()
tanom_last_f = tanom_last * 9/5
month_str = last_time.strftime("%B %Y")

df.loc["TMean", "LastMonth"] = float(tanom_last_f)
df.loc["TMean", "LastMonthDate"] = month_str

#Temp forecast
url = f'https://www.cpc.ncep.noaa.gov/products/people/mchen/CFSv2FCST/weekly/data/CFSv2.tmpsfc.{yest_str}.wkly.anom.nc'
filename = "./data_files/tmean.forecast.nc"

response = requests.get(url)
if response.status_code == 200:
    with open(filename, 'wb') as f:
        f.write(response.content)
else:
    print(f"Failed to download file. Status code: {response.status_code}")
    
tmean_forecast_dataset = xr.open_dataset(filename)

tmean_forecast_dataset_palau = tmean_forecast_dataset['anom'].sel(lat=lat, lon=lon, method='nearest')
tmean_forecast_palau_df = tmean_forecast_dataset_palau.to_dataframe().reset_index()

tmean_forecast_value_c = tmean_forecast_palau_df['anom'].iloc[1]
tmean_forecast_value_f = tmean_forecast_value_c * 9/5
tmean_forecast_date = tmean_forecast_palau_df['time'].iloc[1]
df.loc["TMean", "Forecast"] = tmean_forecast_value_f
df.loc["TMean", "ForecastDate"] = tmean_forecast_date.strftime("%B %d, %Y")

#Tmean outlook
url = "https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd1/glbSSTMon.nc"
filename = "./data_files/tmean.outlook.nc"

response = requests.get(url)
if response.status_code == 200:
    with open(filename, 'wb') as f:
        f.write(response.content)
else:
    print(f"Failed to download file. Status code: {response.status_code}")

tmean_outlook_dataset = xr.open_dataset(filename)

tmean_outlook_dataset_palau = tmean_outlook_dataset['anom'].sel(lat=lat, lon=lon, method='nearest')
tmean_outlook_palau_df = tmean_outlook_dataset_palau.to_dataframe().reset_index()
tmean_outlook_value_c = tmean_outlook_palau_df['anom'].iloc[0]
tmean_outlook_value_f = tmean_outlook_value_c * 9/5
tmean_outlook_date = tmean_outlook_palau_df['time'].iloc[0]

df.loc["TMean", "Outlook"] = tmean_outlook_value_f
df.loc["TMean", "OutlookDate"] = tmean_outlook_date.strftime("%B %Y")

cycle="12" 
grib_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs.{today_str}/{cycle}/time_grib_01/wnd10m.01.{today_str}{cycle}.daily.grb2" 
idx_url = grib_url + ".idx" 
grib_file = "../data/wnd10m.cfs.daily.grb2" 
idx_file = grib_file + ".idx" 

download_file(grib_url, grib_file) 
download_file(idx_url, idx_file) 

ds = xr.open_dataset(grib_file, engine="cfgrib") 
palau = ds.sel(latitude=lat,longitude=lon,method='nearest') 
uv_palau_df = palau[['u10', 'v10']].to_dataframe().reset_index() 
palau_tz = ZoneInfo("Pacific/Palau") 

now_palau = datetime.now(palau_tz) 
uv_palau_df['valid_time'] = pd.to_datetime(uv_palau_df['valid_time']).dt.tz_localize('UTC').dt.tz_convert(palau_tz) 

start_date = (now_palau).replace(hour=0, minute=0, second=0, microsecond=0) 
end_date = start_date + timedelta(days=6) - timedelta(seconds=1) 

uv_palau_3m_df = uv_palau_df[ (uv_palau_df['valid_time'] >= start_date) & (uv_palau_df['valid_time'] <= end_date) ] 
uv_palau_3m_df = uv_palau_3m_df.copy() 
uv_palau_3m_df['wind_speed'] = np.sqrt(uv_palau_3m_df['u10']**2 + uv_palau_3m_df['v10']**2) 
uv_palau_3m_df['Date'] = uv_palau_3m_df['valid_time'].dt.date 

wind_speed_df = uv_palau_3m_df.groupby('Date')[['wind_speed']].max() 
try:
    os.remove(grib_file)
except FileNotFoundError:
    pass

try:
    os.remove(idx_file)
except FileNotFoundError:
    pass

result = wind_speed_df[["wind_speed"]].reset_index()

result.to_json("./data/wind_speed.json",orient="records", date_format="iso")

df.reset_index(inplace=True)
df.rename(columns={"index": "Type"}, inplace=True)

df.to_json("./data/palau_rf_temp.json", orient="records", date_format="iso")
