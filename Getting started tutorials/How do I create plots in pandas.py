import matplotlib.pyplot as plt
import pandas as pd


air_quality = pd.read_csv('/home/ubuntu/PycharmProjects/pandas/feed/air_quality_no2.csv')
print(air_quality)

air_quality.plot()
plt.show()


air_quality["station_paris"].plot()
plt.show()

air_quality.plot.scatter(x="station_london", y="station_paris", alpha=0.5)
plt.show()


print(dir(air_quality.plot))


air_quality.plot.box()
plt.show()

axs = air_quality.plot.area(figsize=(12,4), subplots = True)
plt.show()

fig, axse = plt.subplots(figsize=(5,5))
air_quality.plot.area(ax=axse)
axse.set_ylabel("NO$_2$ concentration")
fig.savefig("no2_concentrations.png")
plt.show()
