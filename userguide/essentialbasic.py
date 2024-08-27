import pandas as pd
import numpy as np
index = pd.date_range('20240101', periods=8)
s = pd.Series(np.random.randn(5), index=['A', 'B', 'C', 'D', 'E'])
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))
print(s)
print(df)


long_series = pd.Series(np.random.randn(5))
print(long_series.head())
print(long_series.tail())


print(df[:2])
df.columns = [x.lower() for x in df.columns]
print(df)

print(s.array)
print(s.index.array)
print(s.to_numpy())
print(np.asarray(s))


ser = pd.Series(pd.date_range('2000', periods=2, tz='EST'))
print(ser.to_numpy(dtype=object))
print(ser.to_numpy(dtype='datetime64[ns]'))

print(df.to_numpy())
print(df.values)
# x= df.array
# print(x)

#  By Default tese are enabled
#  pd.set_option("compute.use_bottleneck", False)
#  pd.set_option("compute.use_numexpr", False)

df = pd.DataFrame({
    'one': pd.Series(np.random.randn(3), index=list('abc')),
    'two': pd.Series(np.random.randn(4), index=list('abcd')),
    'three': pd.Series(np.random.randn(3), index=list('bcd'))

})
print(df)
row = df.iloc[1]
print(row)
column = df['two']
print(column)

print(df.sub(row, axis='columns'))
print(df.sub(row, axis='index'))
print(df.sub(row, axis=0))
print(df.sub(row, axis=1))

dfmi = df.copy()
dfmi.index = pd.MultiIndex.from_tuples([(1, "a"), (1, "b"), (1, "c"), (2, "a")], names=["first", "second"])
print(dfmi)

print(dfmi.sub(column, axis=0, level='second'))

s = pd.Series(np.arange(10))
print(s)
div, rem = divmod(s, 3)
print(div, rem)

idx = pd.Index(np.arange(10))
print(idx)
div, rem = divmod(idx, 4)
print(div, rem)


div, rem = divmod(s, [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
print(div, rem)
print(type(pd.Series(div).to_numpy()))
print(type(np.asarray(rem)))

df2 = df.copy()

df2.loc['a', 'three'] = 1.0
print(df)
print(df2)
print(df + df2)
print(df.add(df2, fill_value=0))
# print(df.add(df2, fill_value=0).fillna(value=15))

print(df.eq(df2))

print((df > 0).any())
print((df > 0).all())

print(df.empty)
print(pd.DataFrame(columns=list('ABC')).empty)

print(df + df == df * 2)
print((df + df == df * 2).all())

#  NaNs do not compare as equals

print((df + df).equals(df * 2))


df1 = pd.DataFrame(['foo', 0, np.nan])
df2 = pd.DataFrame([0, np.nan, 'foo'], index=[1, 2, 0])
print(df1.equals(df2))
print(df1.equals(df2.sort_index()))

#  if lenght are not matching then value error will be raised
print(pd.Series(list('ABC')) == 'A')
print(pd.Index(list('ABC')) == 'A')
print(pd.Series(list('ABCD')) == pd.Index(list('BCAD')))
print(pd.Series(list('ACBD')) == np.array(list('ABCD')))

df1 = pd.DataFrame({
    "A": [1.0, np.nan, 3.0, 5.0, np.nan],
    "B": [np.nan, 2.0, 3.0, np.nan, 6.0]
})

df2 = pd.DataFrame({
    "A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
    "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
})

print(df1.combine_first(df2))
print(df2.combine_first(df1))


def combiner(x, y):
    return np.where(pd.isna(x), y, x)


print(df1.combine(df2, combiner))
print(df2.combine(df1, combiner))


print(df)
print(df.mean())
print(df.mean(0))
#  above 2 are same
print(df.mean(1))

print(df.sum(0, skipna=True))
print(df.sum(0, skipna=False))

ts_stand = (df - df.mean())/df.std()
print(ts_stand.std())

xs_stand = df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)
print(xs_stand.std(1))

print(df.cumsum())

print(np.mean(df['one']))
print(np.mean(df['one'].to_numpy()))

series = pd.Series(np.random.randn(500))
print(series)
series[20:500] = np.nan
series[10:20] = 5
print(series.head(50))
print(series.nunique())

series = pd.Series(np.random.randn(1000))
series[::2] = np.nan
print(series.describe())

frame = pd.DataFrame(np.random.randn(1000, 5), columns=list('abcde'))
frame.iloc[::2] = np.nan
print(frame.describe())
print(frame.describe(percentiles=[0.10, 0.50, 0.35, 0.99]))

s = pd.Series(["a", "a", "b", "b", "a", "a", np.nan, "c", "d", "a"])
print(s.describe())
frame = pd.DataFrame({"a": ["Yes", "Yes", "No", "No"], "b": range(4)})
print(frame.describe())
print(frame.describe(include=['object']))
print(frame.describe(include=['number']))
print(frame.describe(include='all'))

s1 = pd.Series(np.random.randn(5))
print(s1.idxmin(), s1.idxmax())
df1 = pd.DataFrame(np.random.randn(10, 3), columns=list('abc'))
print(df1.idxmin(axis=0), df1.idxmax(1))
df3 = pd.DataFrame([2, 1, 1, 3, np.nan], columns=["A"], index=list("edcba"))
print(df3.idxmax(), df3.idxmin())


data = np.random.randint(0, 7, size=50)
s = pd.Series(data)
print(s.value_counts().sort_index())
data = {"a": [1, 2, 3, 4], "b": ["x", "x", "y", "y"]}
frame = pd.DataFrame(data)
print(frame.value_counts())

s5 = pd.Series([1, 1, 3, 3, 3, 5, 5, 7, 7, 7])
print(s5.mode())
df5 = pd.DataFrame({
    "A": np.random.randint(0, 7, size=50),
    "B": np.random.randint(-10, 15, size=50),
})
print(df5.mode())


arr = np.random.randn(20)
print(arr)
factor = pd.cut(arr, 4)
print(factor)
factor = pd.cut(arr, [-5, -1, 0, 1, 5])
print(factor)
factor = pd.qcut(arr, [0, 0.25, 0.5, 0.75, 1])
print(factor)
factor = pd.cut(arr, [-np.inf, 0, np.inf])
print(arr)

def extract_city_name(df):
    df['city_name'] = df['city_and_code'].str.split(',').str.get(0)
    return df


def add_country_name(df, country_name=None):
    col = 'city_name'
    df['city_and_country'] = df[col] +' ' + country_name
    return df


df_p = pd.DataFrame({'city_and_code': ['Chicago, IL']})
print(add_country_name(extract_city_name(df_p), country_name='US'))
print(df_p.pipe(extract_city_name).pipe(add_country_name, country_name='IN'))

print(df.apply(lambda x: np.mean(x)))
print(df.apply(lambda x: np.mean(x), axis=1))
print(df.apply(lambda x: x.max()-x.min()))
print(df.apply(np.cumsum))
print(df.apply(np.exp))
print(df.apply('mean'))
print(df.apply('mean', axis=1))

tsdf = pd.DataFrame(np.random.randn(10, 3), columns=["A", "B", "C"], index=pd.date_range("1/1/2000", periods=10))
print(tsdf)
print(tsdf.apply(lambda x: x.idxmax()))
print(tsdf['A'].max(), tsdf['B'].max(), tsdf['C'].max())

def subtract_and_divide(x, sub, divide=1):
    return (x - sub) / divide


df_udf = pd.DataFrame(np.ones(shape=(2, 2), dtype='int64', ))
print(df_udf)
print(df_udf.apply(subtract_and_divide, args=(5,), divide=3))

tsdf = pd.DataFrame(np.random.randn(10, 3), columns=list('abc'), index=pd.date_range('20240101', periods=10))
print(tsdf)
tsdf.iloc[3:7] = np.nan
print(tsdf)
print(tsdf.apply(pd.Series.interpolate))

tsdf = pd.DataFrame(np.random.randn(10, 3), index=pd.date_range('20240101', periods=10), columns=list('ABC'))
print(tsdf)
tsdf[3:7] = np.nan
print(tsdf)
print(tsdf.agg(lambda x: np.sum(x)))
print(tsdf.agg('sum', axis=1))
print(tsdf.agg(['sum', 'mean'], axis=1))
print(tsdf.sum())
def mymean(x):
    return x.mean()


print(tsdf.agg(['sum', mymean]))

print(tsdf.agg({'A': ['sum', 'mean'], 'B': ['std', 'min', 'max']}))
print(tsdf)
print(tsdf.transform('abs'))
print(tsdf.transform(np.abs))
print(tsdf.transform(lambda x: x.abs()))
print(tsdf['A'].transform([np.abs, lambda x: x + 1]))
print(tsdf.transform({'A': np.abs, 'B': lambda x: x + 2}))
# print(tsdf.transform({'A': np.abs, 'B': [lambda x: x + 2, 'sqrt']}))
#  above statement is turned off due to runtime warning
df4 = df.copy()
print(df4)

def f(x):
    return len(str(x))


print(df4.map(f))
s = pd.Series(["six", "seven", "six", "seven", "two"], index=["a", "b", "c", "d", "e"])
t = pd.Series({"six": 6.0, "two": 2.0, "seven": 7.0})
print(s.map(t))

s = pd.Series(np.random.randn(5), index=list('abcde'))
print(s)
print(s.reindex(index=list('ebfd')))

print(df.reindex(index=list('cfb'), columns=('three', 'two', 'one')))
rs = s.reindex(df.index)
print(rs)
print(rs.index is df.index)
print(df.reindex(['two', 'three', 'one'], axis='columns'))
print(df.reindex(['two', 'three', 'one'], axis=1))
print(df.reindex(['c', 'f', 'b'], axis='index'))
print(df.reindex(['c', 'f', 'b'], axis=0))

df2 = df.reindex(list('abc'), columns=['one', 'two'])
print(df2)
df3 = df2 - df2.mean()
print(df3)
print(df.reindex_like(df2))

s = pd.Series(np.random.randn(5), index=list('abcde'))
s1 = s[:4]
s2 = s[1:]
print(s1, s2)
print(s1.align(s2))
print(s1.align(s2, join='inner'))
print(s1.align(s2, join='left'))
print(s1.align(s2, join='right'))
print(s1.align(s2, join='outer', axis=0))

rng = pd.date_range('20240101', periods=8, freq='2D')
ts = pd.Series(np.random.randn(8), index=rng)
print(s)
ts2 = ts.iloc[[0, 3, 6]]
print(ts2.reindex(ts.index))
print(ts2.reindex(ts.index, method='ffill'))
print(ts2.reindex(ts.index, method='bfill'))
print(ts2.reindex(ts.index, method='nearest'))
print(ts2.reindex(ts.index).ffill())
print(ts2.reindex(ts.index, method='ffill', limit=1))
print(ts2.reindex(ts.index, method='ffill', tolerance='1 day'))

print(df)
print(df.drop(['a', 'd']))
print(df.drop(['two'], axis=1))
print(df.drop(['one'], axis='columns'))
print(df.drop(df.index.difference(['a', 'd'])))
print(df.reindex(df.index.difference(['a', 'd'])))

print(s)
print(s.rename(str.upper))
print(df.rename(index={'a': 'apple', 'b': 'banana', 'd': 'durian'}, columns={'one': 'foo', 'two': 'bar'}))
print(df.rename({"one": "foo", "two": "bar"}, axis="columns"))
print(df.rename({"a": "apple", "b": "banana", "d": "durian"}, axis="index"))
print(s.rename('scalar-name'))

df = pd.DataFrame({'A': [1, 2, 3, 4], 'B':[6, 7, 8, 9]}, index=pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=['let', 'num']))
print(df)
print(df.rename_axis(index={'let': 'nag'}))
print(df.rename_axis(index=str.upper))
print(df.rename(columns={'A': 'C', 'B': 'D'}))
print(df.rename(columns=str.lower))

df = pd.DataFrame({
    'col1': np.random.randn(3),
    'col2': np.random.randn(3)
}, index=list('abc'))
print(df)
for col in df:
    print(col)

for index, row in df.iterrows():
    print(index, row)

for label, ser in df.items():
    print(label, ser, sep='-')

for row in df.itertuples():
    print(row)
print(df)
#  iterrows may convert type of value accordingly
row = next(df.iterrows())[1]
print(row)
print(row['col1'].dtype)
print(df['col1'].dtype)

df2 = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
print(df2.T)
df_t = pd.DataFrame({
    idx: value for idx, value in df2.iterrows()
})
print(df_t)

s = pd.Series(pd.date_range('20240101', periods=10, freq='20D 50Min 50s'))
print(s)
print(s.dt.hour)
print(s.dt.second)
print(s.dt.day)
print(s.dt.year)
print(s.dt.month)
print(s[s.dt.day == 1])
stz = s.dt.tz_localize('Asia/Kolkata')
print(stz)
print(stz.dt.tz_convert('EST'))
print(stz.dt.strftime('%Y-%m-%d'))


s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"], dtype="string")
print(s.str.lower())


df = pd.DataFrame({
    "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
    "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
    "three": pd.Series(np.random.randn(3), index=["b", "c", "d"])
})
unsorted_df = df.reindex(index=['a', 'd', 'c', 'b'], columns=['three', 'two', 'one'])
print(unsorted_df)
print(unsorted_df.sort_index())
print(unsorted_df.sort_index(ascending=False))
print(unsorted_df['three'].sort_index())
print(unsorted_df.sort_index(axis=1))


s1 = pd.DataFrame({
    "a": ["B", "a", "C"],
    "b": [1, 2, 3],
    "c": [2, 3, 4]
}).set_index(list("ab"))

print(s1)
print(s1.sort_index(level='a'))
print(s1.sort_index(level='a', key= lambda x: x.str.lower()))

df1 = pd.DataFrame({
    "one": [2, 1, 1, 1],
    "two": [1, 3, 2, 4],
    "three": [5, 4, 3, 2]
})

print(df1.sort_values(by='two'))
print(df1.sort_values(by=['one', 'two']))
s[2] = np.nan
print(s)
print(s.sort_values())
print(s)
print(s.sort_values(na_position='first'))
print(s)

s1 = pd.Series(["B", "a", "C"])
print(s1)
print(s1.str.lower())
print(s1.str.lower().sort_values(key= lambda x: x.str.lower()))

df = pd.DataFrame({"a": ["B", "a", "C"], "b": [1, 2, 3]})
print(df.sort_values(by='a'))
print(df.sort_values(by='a', key=lambda col: col.str.lower()))
print(df.sort_values(by='b'))

idx = pd.MultiIndex.from_tuples([('a',1), ('a',2), ('a',2), ('b',2), ('b',1), ('b',1)])
idx.names = ['one', 'two']
df_multi = pd.DataFrame({'a': np.arange(6, 0, -1)}, index=idx)
print(df_multi)
print(df_multi.sort_values(by=['two', 'a']))

df = pd.DataFrame({
    "a": [-2, -1, 1, 10, 8, 11, -1],
    "b": list("abdceff"),
    "c": [1.0, 2.0, 4.0, 3.2, np.nan, 3.0, 4.0],
})

print(df.nlargest(1, ['a', 'c']))
print(df.nsmallest(1, ['a', 'c']))
print(df.nlargest(1, 'c'))
print(df.nsmallest(1, 'a'))

df1.columns = pd.MultiIndex.from_tuples([("a", "one"), ("a", "two"), ("b", "three")])
print(df1.sort_values(by=("a", "two")))


dft = pd.DataFrame({
    "A": np.random.rand(3),
    "B": 1,
    "C": "foo",
    "D": pd.Timestamp("20010102"),
    "E": pd.Series([1.0] * 3).astype("float32"),
    "F": False,
    "G": pd.Series([1] * 3, dtype="int8"),
})
print(dft.dtypes)
print(pd.Series([1, 2, 3, 4, 5, 6.0]).dtype)
print(pd.Series([1, 2, 3, 6.0, "foo"]).dtype)
print(dft.dtypes.value_counts())
df1 = pd.DataFrame(np.random.randn(8, 1), columns=["A"], dtype="float32")
print(df1.dtypes)

df2 = pd.DataFrame({
    "A": pd.Series(np.random.randn(8), dtype="float16"),
    "B": pd.Series(np.random.randn(8)),
    "C": pd.Series(np.random.randint(0, 255, size=8), dtype="uint8"),  # [0,255] (range of uint8)
})

print(df2.dtypes)

print(pd.DataFrame([1, 2], columns=["a"]).dtypes)
frame = pd.DataFrame(np.array([1, 2]))
print(frame)
df3 = (df1.reindex_like(df2).fillna(value=0.0) + df2).astype('float32')
print(df1)
print(df2)
print(df3)
print(df3.dtypes)
print(df3.astype('float64').dtypes)

dft1 = pd.DataFrame({"a": [1, 0, 1], "b": [4, 5, 6], "c": [7, 8, 9]})
print(dft1.dtypes)
dft1 = dft1.astype({"a": np.bool, "c": np.float64})
print(dft1.dtypes)
