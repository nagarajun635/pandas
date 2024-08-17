import pandas as pd


air_quality_no2 = pd.read_csv('feed/air_quality_no2_long.csv',parse_dates=True)
air_quality_no2 = air_quality_no2[["date.utc", "location","parameter", "value"]]
print(air_quality_no2.head())

air_quality_pm25 = pd.read_csv('feed/air_quality_pm25_long.csv',parse_dates=True)
air_quality_pm25 = air_quality_pm25[["date.utc", "location","parameter", "value"]]
print(air_quality_pm25.head())

air_quality = pd.concat([air_quality_no2, air_quality_pm25],keys=['no2','pm25'])
print(air_quality.head())
print(air_quality_no2.shape)
print(air_quality_pm25.shape)
print(air_quality.shape)


air_quality = air_quality.sort_values(by='date.utc')
print(air_quality.head(10))

stations_coord = pd.read_csv('feed/air_quality_stations.csv')
print(stations_coord.head())
#  here we've common column
air_quality1 = pd.merge(air_quality, stations_coord, how='left', on='location')
print(air_quality1.head())
print(air_quality1.shape)
print(air_quality1.columns)


air_quality2 = pd.concat([air_quality_no2, air_quality_pm25],keys=['no2','pm25'])
stations_coord2 = pd.read_csv('feed/air_quality_stations2.csv')
#  here we don't have common column
air_quality2 = pd.merge(air_quality2, stations_coord2, how='left', left_on='location',right_on='area')
print(air_quality2.head())
print(air_quality2.shape)
print(air_quality2.columns)