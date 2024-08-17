import pandas as pd


titanic = pd.read_csv('feed/titanic.csv')
print(titanic)
print(titanic.head(8))
print(titanic.tail(7))
print(titanic.dtypes)
print(titanic.columns)
print(titanic.describe())
titanic.to_excel('feed/titanic.xlsx',sheet_name='passengers',index=False)
print('files saved successfully')

passengers = pd.read_excel('feed/titanic.xlsx',sheet_name='passengers')
print(passengers)
print(passengers.head())
passengers.info()