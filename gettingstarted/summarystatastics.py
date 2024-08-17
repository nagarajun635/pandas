import pandas as pd


titanic = pd.read_csv('feed/titanic.csv')
print(titanic.head())

print(titanic['Age'].mean())

print(titanic[['Age','Fare']].mean())

print(titanic[['Age','Fare']].describe())

print(titanic.agg({
    'Age': ['min', 'max', 'median', 'skew'],
    'Fare': ['min', 'max', 'median', 'mean']
}))

print(titanic[['Age','Sex']].groupby('Sex').mean())
print(titanic.groupby('Sex')['Age'].mean())
print(titanic.groupby('Sex').mean(numeric_only=True))

print(titanic.groupby(['Sex','Pclass'])['Age'].mean())

#  both are same
print(titanic.groupby('Pclass')['Pclass'].count())
print(titanic['Pclass'].value_counts())
