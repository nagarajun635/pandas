import pandas as pd


titanic = pd.read_csv('../feed/titanic.csv')
print(titanic.columns)
print(titanic.head(1))
print(titanic.shape)

ages = titanic['Age']
print(ages.head(1))
print(ages.shape)
print(type(ages))

age_sex = titanic[['Age', 'Sex']]
print(age_sex.head(1))
print(age_sex.shape)
print(type(age_sex))

above_35 = titanic[titanic['Age']>35]
print("This is above 35\n", above_35.head(1))

class_23 = titanic[titanic['Pclass'].isin([2,3])]
print("This is class 23 one\n", class_23.head(1))

class_23 = titanic[(titanic['Pclass'] == 2) | (titanic['Pclass'] == 3)]
print("This is class 23 two\n", class_23.head(1))

age_no_na = titanic[titanic['Age'].notna()]
print("This is age not NA\n", age_no_na.head(1))

adult_names = titanic.loc[titanic['Age']>35, 'Name']
print("This is selecting both row and columns\n", adult_names.head(1))
print(adult_names)

print(titanic.iloc[9:25, 2:5])

titanic.iloc[:4, 0] = 0
print('This is passanger ID\n',titanic['PassengerId'])
