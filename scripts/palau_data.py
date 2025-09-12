import numpy as np
import requests
import xarray as xr
from datetime import datetime, timedelta
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

today_hst = datetime.now(ZoneInfo("Pacific/Honolulu"))
today_str = today_hst.strftime("%Y%m%d")
yest_hst = today_hst - timedelta(days=1)
yest_str = yest_hst.strftime("%Y%m%d")


today = datetime.utcnow()
if today.month == 1:
    last_month = 12
    last_year = today.year - 1
else:
    last_month = today.month - 1
    last_year = today.year


months_since_1960 = (last_year - 1960) * 12 + (last_month - 1)
t_value = months_since_1960 + 0.5 

filename = "../data/rf_lastMonth.nc"

# Build list of (year, month) to try: last_month, last_month-1, -2, -3
start = date(last_year, last_month, 15)
months_to_try = [( (start - relativedelta(months=k)).year,
                   (start - relativedelta(months=k)).month ) for k in range(0, 4)]

base_url = ("https://iridl.ldeo.columbia.edu/"
            "SOURCES/.NOAA/.NCEP/.CPC/.CAMS_OPI/.v0208/.anomaly_9120/.prcp")

success = False
for y, m in months_to_try:
    try:
        # e.g., "Aug 2025"
        month_str = f"{calendar.month_abbr[m]} {y}"

        # Recompute T index for this (y, m)
        t_value = ((y - 1960) * 12 + (m - 1)) + 0.5

        url = (
            f"{base_url}"
            f"/T/%28days%20since%201960-01-01%29streamgridunitconvert"
            f"/T/differential_mul/T/%28months%20since%201960-01-01%29streamgridunitconvert"
            f"//units/%28mm/month%29def//long_name/%28Precipitation%20Anomaly%29def"
            f"/DATA/-500/-450/-400/-350/-300/-250/-200/-150/-100/-50/-25/25/50/100/150/200/250/300/350/400/450/500/VALUES/prcp_anomaly_max500_colors2"
            f"/Y/%285N%29%2810N%29RANGEEDGES/X/%28130E%29%28140E%29RANGEEDGES"
            f"/T/%28{month_str}%29%28{month_str}%29RANGEEDGES/data.nc"
        )

        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)

        ds = xr.open_dataset(filename, decode_times=False, engine="netcdf4")

        # Pull value at nearest (lat, lon, T)
        val = ds["aprod"].sel(Y=lat, X=lon, T=t_value, method="nearest").item()

        df.loc["Rain", "LastMonth"] = float(val)
        df.loc["Rain", "LastMonthDate"] = month_str
        success = True
        break  
    except Exception:
        # try the next earlier month
        continue

if not success:
    # Optional: mark as missing so downstream logic is explicit
    df.loc["Rain", "LastMonth"] = np.nan
    df.loc["Rain", "LastMonthDate"] = "Missing"

#Rain forecast
url = "https://access-s.clide.cloud/files/global/weekly/data/rain.forecast.anom.weekly.nc"
filename = "./data_files/rf.forecast.nc"

response = requests.get(url)

download_file(url, filename)

rf_forecast_dataset = xr.open_dataset(filename)
rf_forecast_palau = rf_forecast_dataset['rain'].sel(lat=slice(min_lat_ssh,max_lat_ssh),lon=slice(min_lon_ssh,max_lon_ssh))
rf_forecast_palau_df = rf_forecast_palau.to_dataframe().reset_index()
rf_forecast_value = rf_forecast_palau_df['rain'].iloc[1]
rf_forecast_date = rf_forecast_palau_df['time'].iloc[1]
df.loc["Rain", "Forecast"] = rf_forecast_value
df.loc["Rain", "ForecastDate"] = rf_forecast_date

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
rf_outlook_time = rf_outlook_palau_df['time'].iloc[0]

df.loc["Rain", "Outlook"] = rf_outlook_value
df.loc["Rain", "OutlookDate"] = rf_outlook_time

#Temp last month
today_hst = datetime.now(ZoneInfo("Pacific/Honolulu"))
first_of_this_month = today_hst.replace(day=1)
last_month_date = first_of_this_month - timedelta(days=1)
last_month_str = last_month_date.strftime("%b %Y")  # e.g. "Apr 2025"
filename = "../data/tmean_lastMonth.nc"

# Build list of (year, month) to try: last_month, last_month-1, -2, -3
start = date(last_year, last_month, 15)
months_to_try = [( (start - relativedelta(months=k)).year,
                   (start - relativedelta(months=k)).month ) for k in range(0, 4)]

base_url = "https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.CAMS/.anomaly/.temp_9120"

success = False
for y, m in months_to_try:
    try:
        # e.g., "Aug 2025"
        month_str = f"{calendar.month_abbr[m]} {y}"

        # Recompute T index for this (y, m)
        t_value = ((y - 1960) * 12 + (m - 1)) + 0.5
        url = (
            f"{base_url}"
            f"/Y/%285N%29%2810N%29RANGEEDGES/X/%28130E%29%28140E%29RANGEEDGES"
            f"/T/%28{month_str}%29%28{month_str}%29RANGEEDGES/data.nc"
        )

        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)

        ds = xr.open_dataset(filename, decode_times=False, engine="netcdf4")

        # Pull value at nearest (lat, lon, T)
        val = ds["aprod"].sel(Y=lat, X=lon, T=t_value, method="nearest").item()

        df.loc["TMean", "LastMonth"] = float(val)
        df.loc["TMean", "Month"] = month_str
        print(f"Saved month: {month_str}")
        success = True
        break  
    except Exception:
        # try the next earlier month
        continue

if not success:
    # Optional: mark as missing so downstream logic is explicit
    df.loc["TMean", "LastMonth"] = np.nan
    df.loc["TMean", "LastMonthName"] = np.nan

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
df.loc["TMean", "ForecastDate"] = tmean_forecast_date

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
df.loc["TMean", "OutlookDate"] = tmean_outlook_date

# grib_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs.{today_str}/{cycle}/time_grib_01/wnd10m.01.{today_str}{cycle}.daily.grb2"
# idx_url = grib_url + ".idx"

# grib_file = "./data_files/wnd10m.cfs.daily.grb2"

# idx_file = grib_file + ".idx"

# download_file(grib_url, grib_file)
# download_file(idx_url, idx_file)

# ds = xr.open_dataset(grib_file, engine="cfgrib")
# palau = ds.sel(latitude=lat,longitude=lon,method='nearest')
# uv_palau_df = palau[['u10', 'v10']].to_dataframe().reset_index()

# palau_tz = ZoneInfo("Pacific/Palau")
# now_palau = datetime.now(palau_tz)
# uv_palau_df['valid_time'] = pd.to_datetime(uv_palau_df['valid_time']).dt.tz_localize('UTC').dt.tz_convert(palau_tz)

# start_date = (now_palau + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
# end_date = start_date + timedelta(days=7) - timedelta(seconds=1)

# uv_palau_3m_df = uv_palau_df[
#     (uv_palau_df['valid_time'] >= start_date) &
#     (uv_palau_df['valid_time'] <= end_date)
# ]
# uv_palau_3m_df = uv_palau_3m_df.copy()
# uv_palau_3m_df['wind_speed'] = np.sqrt(uv_palau_3m_df['u10']**2 + uv_palau_3m_df['v10']**2)
# uv_palau_3m_df['Date'] = uv_palau_3m_df['valid_time'].dt.date


# wind_speed_df = uv_palau_3m_df.groupby('Date')[['wind_speed']].max()


# result = wind_speed_df[["wind_speed"]].reset_index()

# result.to_json("./data/wind_speed.json",orient="records", date_format="iso")
# source_df.loc['Wind','Forecast'] = 'https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system'


df.reset_index(inplace=True)
df.rename(columns={"index": "Type"}, inplace=True)

df.to_json("./data/palau_rf_temp.json", orient="records", date_format="iso")
