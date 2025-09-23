import requests
import json
from collections import defaultdict

# thresholds
TEMP_THRESHOLD = 28.0    # °C
RAIN_THRESHOLD = 10.0    # mm over 3 days


url = "https://kukau.org/avg_air_temp_solar_rain.json"
resp = requests.get(url)
data = resp.json()

stations = defaultdict(list)
for row in data:
    stations[row["station_no"]].append(row)

result = []
for sn, records in stations.items():
    # sort by date (lsd)
    records.sort(key=lambda x: x["lsd"])
    
    # last 3 days (from data tail)
    last3 = records[-3:]
    avg_temp_last3 = sum(r["avg_air_temp"] for r in last3) / len(last3)
    total_rain_last3 = sum(r["rain_mm"] for r in last3)
    
    # for now: mimic next3days with last 3 days too (placeholder)
    next3 = last3  
    avg_temp_next3 = avg_temp_last3
    total_rain_next3 = total_rain_last3
    
    # classify
    last3_meta = {
        "temp": "HOT" if avg_temp_last3 >= TEMP_THRESHOLD else "COOL",
        "rain": "WET" if total_rain_last3 >= RAIN_THRESHOLD else "DRY"
    }
    next3_meta = {
        "temp": "HOT" if avg_temp_next3 >= TEMP_THRESHOLD else "COOL",
        "rain": "WET" if total_rain_next3 >= RAIN_THRESHOLD else "DRY"
    }

    result.append({
        "station_no": sn,
        "station_name": records[0]["station_name"],
        "meta": {
            "last3days": last3_meta,
            "next3days": next3_meta
        }
    })

# --- save file ---
with open("station_conditions.json", "w") as f:
    json.dump(result, f, indent=2)

