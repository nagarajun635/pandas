import pandas as pd


titanic = pd.read_csv('feed/titanic.csv')
print(titanic.head())
ages = titanic['Age']
print(ages.head())
print(type(titanic['Age']))
print(type(ages))
print(titanic.shape)
print(ages.shape)
age_sex = titanic[['Age','Sex']]
print(age_sex.head())
print(type(titanic[['Age','Sex']]))
print(age_sex.shape)

above_35 = titanic[titanic['Age']>35]
print(above_35.head())
print(titanic['Age']>35)
print(above_35.shape)

class_23 = titanic[titanic['Pclass'].isin([2,3])]
print(class_23.head())
print(class_23.shape)

class_23 = titanic[(titanic['Pclass']==2) | (titanic['Pclass']==3)]
print(class_23.shape,class_23.head())

age_no_na = titanic[titanic['Age'].notna()]
print(age_no_na.head())

adult_name = titanic.loc[titanic['Age']>35,'Name']
print(adult_name.head(100))
print(adult_name.shape)

adult_name_iloc = titanic.iloc[9:25,2:7]
print(adult_name_iloc.shape)
print(adult_name_iloc.head(100))
print(titanic.dtypes)
titanic['Survived'] = titanic['Survived'].astype('str')
titanic.iloc[0:3,1]='anonymous'
print(titanic.dtypes)
print(titanic.head())

