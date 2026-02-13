import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 10000000)
air_quality = pd.read_csv('../feed/air_quality_long.csv', index_col='date.utc', parse_dates=True)
print('---------------------------------------------------------------------------------------------------------------')
print(air_quality.head())

titanic = pd.read_csv('../feed/titanic.csv')
print('---------------------------------------------------------------------------------------------------------------')
print(titanic.head())
print('---------------------------------------------------------------------------------------------------------------')
print(titanic.sort_values(by='Age').head())
print('---------------------------------------------------------------------------------------------------------------')
print(titanic.sort_values(by=['Age', 'Pclass'], ascending=False).head())
print('---------------------------------------------------------------------------------------------------------------')
no2 = air_quality[air_quality['parameter'] == 'no2']
no2_subset = no2.sort_index().groupby(["location"]).head(2)
print(no2_subset.head())
print('---------------------------------------------------------------------------------------------------------------')
print(no2_subset.pivot(columns="location", values="value").head())
print('---------------------------------------------------------------------------------------------------------------')

print(no2.head())
print('---------------------------------------------------------------------------------------------------------------')
# no2.pivot(columns="location", values='value').plot()
# plt.show()
print('---------------------------------------------------------------------------------------------------------------')
print(air_quality.pivot_table(index='location', values='value', columns='parameter'))
print('---------------------------------------------------------------------------------------------------------------')
no2_pivoted = no2.pivot(columns='location', values='value').reset_index()
print(no2_pivoted.head())
print('---------------------------------------------------------------------------------------------------------------')
no_2 = no2_pivoted.melt(id_vars="date.utc")
print(no_2.head())
print('---------------------------------------------------------------------------------------------------------------')
no_2 = no2_pivoted.melt(id_vars="date.utc",var_name='id_location', value_vars=['BETR801', 'FR04014', 'London Westminster'], value_name='NO_2')
print(no_2.head())


