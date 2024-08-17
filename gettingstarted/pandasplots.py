import pandas as pd
import matplotlib.pyplot as plt

air_quality = pd.read_csv('feed/air_quality_no2.csv', index_col=0, parse_dates=True)
print(air_quality.head())
print(air_quality.shape)
print(air_quality.dtypes)

# air_quality.plot()
# plt.show()


# air_quality["station_paris"].plot()
# plt.show()

# air_quality.plot.scatter(x="station_london", y='station_paris')
# plt.show()

print([method_name for method_name in dir(air_quality.plot) if not method_name.startswith('_')])

# air_quality.plot.box()
# plt.show()


# axs = air_quality.plot.area(figsize=(4, 4), subplots=True)
# plt.show()

fig, axs = plt.subplots(figsize=(4,4))
air_quality.plot.area(ax=axs)
axs.set_ylabel("NO$_2$ concentration")
fig.savefig('feed/no2_concentrations.png')
plt.show()