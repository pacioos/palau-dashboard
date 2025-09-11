import json
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

url = "https://kukau.org/avg_air_temp_solar_rain.json" 

response = requests.get(url)
response.raise_for_status()
data = response.json()

for d in data:
    d["day"] = datetime.strptime(d["day"], "%Y-%m-%d").date()

#In Palau timezone
palau_tz = ZoneInfo("Pacific/Palau")
today = datetime.now(palau_tz).date()
yesterday = today - timedelta(days=1)

# Filter records for yesterday
yesterday_data = [
    {
        "station_no": d["station_no"],
        "station_name": d["station_name"],
        "date": d["day"].isoformat(),
        "avg_air_temp": d["avg_air_temp"],
        "rain_mm": d["rain_mm"],
        "rain_in": d["rain_in"]
    }
    for d in data if d["day"] == yesterday
]

# Save to JSON file
with open("./data/latest_weather.json", "w") as f:
    json.dump(yesterday_data, f, indent=2)
