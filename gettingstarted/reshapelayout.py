import pandas as pd
import matplotlib.pyplot as plt

titanic = pd.read_csv('feed/titanic.csv')
print(titanic.head())

air_quality = pd.read_csv('feed/air_quality_long.csv', index_col='date.utc', parse_dates=True)
print(air_quality.head())
print(air_quality.shape)

print(titanic.sort_values(by='Age').head())
print(titanic.sort_values(by=['Pclass','Age'], ascending=False).head())

no2 = air_quality[air_quality['parameter'] == 'no2']
print(no2.head())

no2_subset = no2.sort_index().groupby(['location']).head(2)
print(no2_subset.shape)

print(no2_subset.pivot(columns='location', values='value'))

print(no2.head(2))

# no2.pivot(columns='location', values='value').plot()
# plt.show()

print(air_quality.pivot_table(columns='parameter',index='location', values='value', aggfunc='mean',margins=True))
print(air_quality.groupby(['parameter', 'location'])[['value']].mean())
print(air_quality.groupby(['location', 'parameter'])[['value']].mean())

no2_pivoted = no2.pivot(columns='location', values='value').reset_index()
print(no2_pivoted.head())
print(no2_pivoted.melt(id_vars='date.utc').head())
no_2 = no2_pivoted.melt(id_vars='date.utc', var_name='id_location', value_vars=['BETR801','FR04014','London Westminster'], value_name='NO_2')
print(no_2.head())



