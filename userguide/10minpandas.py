import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# dates = pd.date_range(start='20240101', periods=5, freq='3ME')
dates = pd.date_range(start='20240101', periods=5)
print(dates)

df1 = pd.DataFrame(np.random.randn(5, 4), index=dates, columns=list('ABCD'))
print(df1)

# df2 = pd.DataFrame({
#     "A": 1.0,
#     "B": pd.Timestamp("20130102"),
#     "C": pd.Series(1, index=list(range(4)), dtype="float32"),
#     "D": np.array([3] * 4, dtype="int32"),
#     "E": pd.Categorical(["test", "train", "test", "train"]),
#     "F": "foo",
# }, index=dates[:4])
df2 = pd.DataFrame({
    "A": 1.0,
    "B": pd.Timestamp("20130102"),
    "C": pd.Series(1, index=list(range(4)), dtype="float32"),
    "D": np.array([3] * 4, dtype="int32"),
    "E": pd.Categorical(["test", "train", "test", "train"]),
    "F": "foo",
})
print(df2)
print(df2.dtypes)

print(df1.head())
print(df2.tail())
print(df1.index)
print(df2.columns)

print(df2.to_numpy())

print(df1.describe())

print(df1.T)

print(df1.sort_index(axis=1, ascending=False))
print(df1.sort_index(axis=0, ascending=False))

print(df1.sort_values(by='B'))

print(df1['A'])
print(df1[:3])
print(df1['20240102':'20240104'])

print(df1.loc[dates[0]])
print(df1.loc[:, ['A', 'B']])
print(df1.loc['20240102':'20240103', ['A', 'B']])
print(df1.loc[dates[1], 'A'])
print(df1.at[dates[0], 'A'])
print(df1.iloc[0:2, 0:4])
print(df1.iloc[3, 2])
print(df1.iat[1, 1])
print(df1.iloc[[0, 2], [1, 2, 3]])
print(df1.iloc[1:3, :2])
# print(df1)

print(df1[df1['A'] > 0])
print(df1[df1 > 0])
# df2['E']
print(df1)
print(df2)

df3 = df1.copy()
df1['E'] = ["one", "one", "two", "four", "three"]
print(df1)
print(df1[df1['E'].isin(['two', 'four'])])

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20240102', periods=6))
print(s1)

df3['F'] = s1
print(df3)
df3.loc[:, 'D'] = np.array(range(5))
print(df3)

df4 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})
print(df4)
df5 = df4.copy()
print(df5)
df5[df5['A'] > 1] = -df4
print(df5)

df6 = df1.reindex(index=dates[0:4],columns=list(df1.columns))
df6.loc[dates[0]:dates[1],'E'] = 1
df6['F'] = s1
df6.loc[:, 'D'] = np.array([5,5,5,5])
df6.iloc[2:, 4] = np.nan
print(df6)
print(pd.isna(df6))
print(df6.fillna(value='5'))
print(df6.dtypes)
print(df6.dropna(how='any'))
print(df6.dtypes)

print(df4.mean(axis=1))

df1 = pd.DataFrame(np.random.randn(5, 4), index=dates, columns=list("ABCD"))
df1.loc[:, "D"] = np.array([5] * len(df1))
df1['F']= [1,2,3,4,5]
print(df1)

print(df1.mean())
print(df1.mean(axis=1))
dates = pd.date_range(start='20240101', periods=6)
s= pd.Series([1, 3, 5, np.nan, 6, 8],index=dates).shift(2)
print(s)
print(type(s))
print(df1)
print(df1.agg(lambda x: sum(x)))
print(df1.transform(lambda x: x*0))

s = pd.Series(np.random.randint(0,7,size=10))
print(type(s))
print(s)
print(s.value_counts())

s = pd.Series(['NagS', 'RajU'])
print(s.str.lower())
print(s)

df1 = pd.DataFrame(np.random.randn(10,4))
print(df1)
pieces = [df1[:3], df1[3:7], df1[7:]]
print(df1[:3])
print(df1[3:7])
print(pd.concat(pieces))

left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
print(pd.merge(left, right, on='key'))

left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})
print(pd.merge(left, right, on='key'))

df1 = pd.DataFrame({
    'A': ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
    'B': ["one", "one", "two", "three", "two", "two", "one", "three"],
    'C': np.random.randn(8),
    'D': np.random.randn(8)
})
print(df1)
print(df1.groupby('A')[['C', 'D']].sum())
print(df1.groupby(['A','B'])[['C','D']].sum())

arrays = [
    ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
    ["one", "two", "one", "two", "one", "two", "one", "two"]
]

index = pd.MultiIndex.from_arrays(arrays, names=['First', 'Second'])
print(index)
df = pd.DataFrame(np.random.randn(8,2), index=index, columns=['A', 'B'])
print(df)
df2 = df[:4]
print(df2)
stacked = df2.stack(future_stack=False)
print(stacked)
print(stacked.unstack())
print(stacked.unstack(0))
print(stacked.unstack(1))



df = pd.DataFrame({
    "A": ["one", "one", "two", "three"] * 3,
    "B": ["A", "B", "C"] * 4,
    "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
    "D": np.random.randn(12),
    "E": np.random.randn(12),
})
print(df)

print(pd.pivot_table(df, index=['A','B'], values='D', columns=['C']))

rng = pd.date_range('20240101', periods=100, freq='s')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(rng)
print(ts)
print(ts.resample('5Min').sum())

rng = pd.date_range('20240101', periods=5, freq='D')
print(rng)
ts = pd.Series(np.random.randn(len(rng)),index=rng)
print(ts)

ts_utc = ts.tz_localize('Asia/Kolkata')
print(ts_utc)
print(ts_utc.tz_convert('US/Eastern'))

rng = pd.date_range('20240101',periods=50, freq='D')
print(rng)
print(rng + pd.offsets.BusinessDay(5))

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "raw_grade": ["a", "b", "b", "a", "a", "e"]
})
print(df)
df['grade'] = df['raw_grade'].astype('category')
print(df['grade'])
print(df)
print(df.dtypes)
new_categories = ['very_good', 'good', 'very_bad']
df['grade'] = df['grade'].cat.rename_categories(new_categories)
print(df.dtypes)
print(df)

df['grade'] = df['grade'].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
print(df['grade'])
print(df.sort_values(by='grade'))
print(df.groupby('grade',observed=True)[['raw_grade']].count())
print(df.groupby('grade',observed=False)[['raw_grade']].count())

ts = pd.Series(np.random.randn(1000), index=pd.date_range('20240101', periods=1000))
print(ts)
ts = ts.cumsum()
# ts.plot()
print(ts)
# print(plt.show())
# plt.close('all')

df = pd.DataFrame(np.random.randn(1000,4), index=ts.index, columns=['A', 'B', 'C', 'D'])
print(df)
df = df.cumsum()
print(df)
# plt.figure()
# df.plot()
# print(plt.show())

df = pd.DataFrame(np.random.randint(111,999,(10,5)))
print(df)
print(df.cumsum())
df.to_excel('df.xlsx',sheet_name='new',index=False, na_rep='NA')
df1 = pd.read_excel('df.xlsx',sheet_name='new')
print(df1)
print(np.random.randn(2,4))
print(np.random.random(10))
print(np.random.randint(111,999,(4,5)))