import pandas as pd

df = pd.DataFrame({
    'Name': ['naga', 'raju','nagaraj'],
    'Age': [22,23,24],
    'Sex':['male','male','male']
})
print(df)

print(df['Age'])

ages = pd.Series([22,23,24],name='Age')
print(ages)

print(df['Age'].max())
print(ages.min())

print(df.describe())