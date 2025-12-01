import pandas as pd
import os
import numpy as np

########################################
# 1. Helper functions: weekend flag, short-duration flag, distance band
########################################

def isWeekend(day):  # whether this weekday is weekend
    if day < 5:
        return 0
    else:
        return 1

def isStart(time):
    # In the original code, time < 8 is treated as a short-duration trip; adjust the threshold as needed.
    if time < 8:
        return 1
    else:
        return 0

def isLong(mile):  # distance band: very short, short, medium-short, medium, medium-long, long
    if mile <= 3:
        return 0
    elif mile <= 12:
        return 1
    elif mile <= 24:
        return 2
    else:
        return 3

data_path = os.path.join("data", "raw_taxi_orders_with_labels.csv")
out_csv = os.path.join("data", "processed", "taxi_orders_with_labels.csv")

cols = ['DEP_LON', 'DEP_LAT', 'DEP_TIME', 'DEST_LON', 'DEST_LAT', 'DEST_TIME', 'DRIVE_MILE', 'PRICE', 'IS_ANOMALY']
trips = pd.read_csv(data_path, usecols=cols)

trips = trips.drop_duplicates(subset=cols)   # remove duplicate rows
trips = trips.dropna().reset_index(drop=True)  # drop rows with missing values

# type conversion
trips.DRIVE_MILE = trips.DRIVE_MILE.astype(float)
trips.PRICE = trips.PRICE.astype(float)
print(f"Number of Samples: {len(trips)}")

# time-related features
trips['DEP_TIME'] = pd.to_datetime(trips.DEP_TIME.astype(str), format='mixed')   # departure time
trips['DEST_TIME'] = pd.to_datetime(trips.DEST_TIME.astype(str), format='mixed') # arrival time
trips["DRIVE_TIME"] = (trips.DEST_TIME - trips.DEP_TIME).dt.total_seconds() / 60  # trip duration (minutes)
trips["hr"] = trips.DEP_TIME.dt.hour                 # hour of day
trips["WEEKEND"] = trips.DEP_TIME.dt.weekday         # day of week (0=Monday, ..., 6=Sunday)
trips["isWeekend"] = trips.WEEKEND.apply(isWeekend)  # whether it is weekend
trips["isLong"] = trips.DRIVE_MILE.apply(isLong)     # distance band
trips["isTime"] = trips.DRIVE_TIME.apply(isStart)    # whether this is a "short-duration" trip (following original rule)

trips.DRIVE_MILE = trips.DRIVE_MILE.astype(float)  # (optional) filter by trip distance in km
# trips = trips[trips["DRIVE_MILE"] > 3]

trips.PRICE = trips.PRICE.astype(float)  # (optional) filter by fare price
# trips = trips[trips["PRICE"] > 8].reset_index(drop=True)

trips['DEP_LON'] = trips['DEP_LON'] / 10**6
trips['DEP_LAT'] = trips['DEP_LAT'] / 10**6
trips['DEST_LON'] = trips['DEST_LON'] / 10**6
trips['DEST_LAT'] = trips['DEST_LAT'] / 10**6

# geometric distance related features
trips["Manhattan"] = abs(trips.DEP_LON - trips.DEST_LON) + abs(trips.DEP_LAT - trips.DEST_LAT)  # Manhattan distance

trips['DEP_TIME'] = pd.to_datetime(
    trips['DEP_TIME'],           # target column
    format='%Y%m%d%H%M%S',       # input format: YYYYMMDDhhmmss
    errors='coerce'              # unparseable strings become NaT
)

# similarly process arrival time column
trips['DEST_TIME'] = pd.to_datetime(
    trips['DEST_TIME'],
    format='%Y%m%d%H%M%S',
    errors='coerce'
)

drive_mile = trips["DRIVE_MILE"].astype(float)
drive_time = trips["DRIVE_TIME"].astype(float)

trips["operateSpeed"] = np.where(
    drive_time > 0,
    drive_mile / drive_time,
    np.nan
)

hr = trips["hr"].astype(int)
trips["isPeakTime"] = np.where(
    ((hr >= 7) & (hr <= 10)) | ((hr >= 16) & (hr <= 19)),
    1,
    0
)

trips.to_csv(out_csv, index=False)
