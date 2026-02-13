import pandas as pd


titanic = pd.read_csv('../feed/titanic.csv')
print(titanic)
print(titanic.head(10))
print(titanic.tail(10))
print(titanic.dtypes)
titanic.to_excel('../feed/titanic.xlsx', sheet_name='tanic', index=False)
titanic = pd.read_excel('../feed/titanic.xlsx')
print(titanic.head(1))
print(titanic.info())
