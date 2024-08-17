import pandas as pd
import matplotlib.pyplot as plt

air_quality = pd.read_csv('feed/air_quality_no2_long.csv',parse_dates=['date.utc'])
print(air_quality.dtypes)
air_quality = air_quality.rename(columns={'date.utc':'datetime'})
print(air_quality.head())
print(air_quality.city.unique())

air_quality['datetime'] = pd.to_datetime(air_quality['datetime'])
print(air_quality['datetime'])
print(air_quality.dtypes)

print(air_quality['datetime'].max(), air_quality['datetime'].min())
print(air_quality['datetime'].max() - air_quality['datetime'].min())
air_quality['month'] = air_quality['datetime'].dt.month
print(air_quality.head())
print(air_quality.groupby([air_quality['datetime'].dt.weekday,'location'])['value'].mean())


fig, axs = plt.subplots(figsize=(4,4))
air_quality.groupby(air_quality['datetime'].dt.hour)['value'].mean().plot(kind='bar', rot=0, ax=axs)
plt.xlabel('hour of the day')
plt.ylabel("$NO_2 (µg/m^3)$")
# plt.show()


no_2 = air_quality.pivot(index="datetime", columns="location", values="value")
print(no_2.head())
print(no_2.index.year, no_2.index.weekday)

print(no_2['2019-05-20':'2019-05-21'].plot())
# plt.show()
print(no_2.head())

monthly_max = no_2.resample('ME').max()
print(monthly_max)

no_2.resample('D').mean().plot(style='-o',figsize=(4,4))
plt.show()