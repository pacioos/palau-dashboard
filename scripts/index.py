import requests
import json
import cdsapi
import xarray as xr
from collections import defaultdict
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import os

url = "https://kukau.org/avg_air_temp_solar_rain.json"
resp = requests.get(url)
data = resp.json()
stations = defaultdict(list)
for row in data:
    stations[row["station_no"]].append(row)

station = "PW74561"

total_rain_in_last3 = sum(d["rain_in"] for d in data if d["station_no"] == station)


dataset = "cams-global-atmospheric-composition-forecasts"
request = {
    "variable": [
        "2m_temperature",
        "surface_solar_radiation_downwards",
        "total_precipitation"
    ],
    "date": ["2025-11-21/2025-11-21"],
    "time": ["00:00"],
    "leadtime_hour": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "88",
        "89",
        "90",
        "91",
        "92",
        "93",
        "94",
        "95",
        "96",
        "97",
        "98",
        "99",
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110"
    ],
    "type": ["forecast"],
    "data_format": "grib",
    "area": [8, 134, 7, 135]
}

client = cdsapi.Client()
result = client.retrieve(dataset, request)
grib_path = result.download()  


ds = xr.open_dataset(grib_path, engine="cfgrib")
ds = ds.assign_coords(valid_time = ds.time + ds.step)

palau = ds.sel(latitude=7.5150, longitude=134.5825, method="nearest")

valid_local = (
    pd.to_datetime(palau.valid_time.values)
      .tz_localize("UTC")
      .tz_convert("Pacific/Palau")
      .tz_localize(None)
)

palau = palau.assign_coords(valid_time=("step", valid_local))
dates = palau.valid_time.dt.floor("D")

today = pd.Timestamp.now(tz="Pacific/Palau").date()
today_str = today.strftime("%Y-%m-%d")

today = pd.Timestamp(today_str).date()
next_3_days = [(today + pd.Timedelta(days=i)) for i in range(1, 4)]
next_3_str = [d.strftime("%Y-%m-%d") for d in next_3_days]

#Temp 
t2m = (palau['t2m'] - 273.15) * 9/5 + 32
daily_avg_t = t2m.groupby(dates).max("step")
daily_avg_t_series = daily_avg_t.to_series()

today_t = float(daily_avg_t_series.get(today_str, np.nan))
next_3_t_values = [
    float(daily_avg_t_series.get(day, np.nan))
    for day in next_3_str
]
next_3_t_max = np.nanmax(next_3_t_values)

def accumulated_to_incremental(data_arr):
    vals = data_arr.values
    inc = np.empty_like(vals)
    inc[0] = 0
    inc[1:] = vals[1:] - vals[:-1]
    return xr.DataArray(
        inc,
        coords={"step": data_arr.step},
        dims=["step"],
        name=data_arr.name + "_inc"
    )
        

ssrd_acc = palau["ssrd"]
ssrd_inc = accumulated_to_incremental(ssrd_acc)
    
ssrd_MJ = ssrd_inc / 1e6
ssrd_MJ = ssrd_MJ.assign_coords(valid_time=("step", valid_local))
ssrd_MJ = ssrd_MJ.swap_dims({"step": "valid_time"})
    
ssrd_series = ssrd_MJ.to_pandas()
daily_solar = ssrd_series.resample("D").sum()
    
today_solar = float(daily_solar.get(today_str, np.nan))
        

tp_acc = palau["tp"]                 # m accumulated
tp_inc = accumulated_to_incremental(tp_acc)
    
tp_mm = tp_inc * 1000                # convert m → mm
tp_mm = tp_mm.assign_coords(valid_time=("step", valid_local))
tp_mm = tp_mm.swap_dims({"step": "valid_time"})
    
tp_series = tp_mm.to_pandas()
daily_precip_mm = tp_series.resample("D").sum()
    
today_precip_mm = float(daily_precip_mm.get(today_str, np.nan))
    


    
next_3_solar_values = [
    float(daily_solar.get(day, np.nan)) for day in next_3_str
]
next_3_precip_values = [
    float(daily_precip_mm.get(day, np.nan)) for day in next_3_str
]
    
next_3_solar_max = float(np.nanmax(next_3_solar_values))
next_3_precip_sum = float(np.nansum(next_3_precip_values))

if today_t > 87 and today_solar > 20:
    if  next_3_t_max > 87 and next_3_solar_max > 20:
        index = "HOT/HOT"
    elif next_3_precip_sum > 1.5:
        index = "HOT/WET"
    elif total_rain_in_last3 > 0.5:
        index = "WET/HOT"
else:
    index = "None"

import json

data = {
    "Index": index,
    "date": today_str,
    "temp_today": today_t,
    "temp_next3days": next_3_t_max,
    "rain_last3days": total_rain_in_last3,
    "Solar_rad_next3days": next_3_solar_max
}

# Save to file
with open("./data/index.json", "w") as f:
    json.dump(data, f, indent=2)

# Convert to DataFrame row
row = {
    "date": data.get("today_str"),
    "Index": data.get("Index"),
    "temp_today": data.get("temp_today"),
    "rain_last3days": data.get("rain_last3days"),
    "Solar_rad_next3days": data.get("Solar_rad_next3days")
}

csv_path = "data/history.csv"

# Append or create
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
else:
    df = pd.DataFrame([row])

df.to_csv(csv_path, index=False)

try:
    os.remove(grib_path)
except FileNotFoundError:
    pass
