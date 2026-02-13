import pandas as pd


df = pd.DataFrame({
    'Names': ["Mr. Owen Harris",
            "Mr. William Henry",
            "Miss Elizabeth"],
    'Age': [20, 30, 40],
    "Sex": ["Male", "Female", "Male"],
})

print(df)
print(df['Age'])
print(df['Age'].max())
print(df.describe())

ages = pd.Series([12,13,14],name='Age')
print(ages)
print(ages.max())
