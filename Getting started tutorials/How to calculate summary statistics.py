import pandas as pd


titanic = pd.read_csv('../feed/titanic.csv')
print(titanic.head(1))

print(titanic['Age'].mean())
print(titanic[['Age', "Fare"]].mean())

print(titanic[['Age','Fare']].describe())

print(titanic.agg({
    "Age": ['min', 'max', 'median', 'skew'],
    "Fare": ['min', 'max', 'mean', 'median']
}))

print(titanic[['Sex', 'Age']].groupby('Sex').mean())
print(titanic[['Sex', 'Age']].groupby('Sex').median())

print(titanic.groupby('Sex').mean(numeric_only=True))

print(titanic.groupby('Sex')['Age'].mean())
print(titanic.groupby(['Sex', 'Pclass'])['Fare'].mean())

print(titanic['Pclass'].value_counts())
print(titanic.groupby('Pclass')['Pclass'].count())
