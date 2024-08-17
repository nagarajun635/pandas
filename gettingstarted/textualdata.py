import pandas as pd


titanic = pd.read_csv('feed/titanic.csv')
titanic['Name']=titanic['Name'].str.lower()
print(titanic['Name'].str.lower())
print(titanic['Name'].str.split(','))

titanic['Surname'] = titanic['Name'].str.split(',').str.get(0).astype('str')
print(titanic['Surname'].astype(str))
print(titanic[titanic['Name'].str.contains('countess')])

print(titanic['Name'].str.len().idxmax())
print(titanic.loc[titanic['Name'].str.len().idxmax(),'Name'])

# titanic['sex_short'] = titanic['Sex'].replace({'male':'M','female':'F'})
# print(titanic['sex_short'].head())
titanic['sex_short'] = titanic['Sex'].str.replace('female','F').str.replace('male','M')
# titanic['sex_short'] = titanic['Sex'].
print(titanic['sex_short'].head(200))