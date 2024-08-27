import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pandas Series
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(s)
print(s.index)

print(pd.Series(np.random.randn(5)))

d = {'b': 1, 'a': 0, 'c': 2}
print(pd.Series(d))

d = {"a": 0.0, "b": 1.0, "c": 2.0}
print(pd.Series(d))
print(pd.Series(d, index=["b", "c", "a", "d"]))

print(pd.Series(5.0, index=["b", "c", "a", "d"]))

print(s.iloc[0])
print(s.iloc[:2])
print(s[s > s.median()])
print(s.iloc[[4, 2, 1]])
print(np.exp(s))

print(s.dtypes)
print(s.dtype)

print(s.to_numpy())
print(s.array)

print(s['e'])
print('e' in s)
s['e'] = 13
print(s)
print(s.get('f'))

print(s + s)
print(s * 2)
print(np.exp(s))
print(s.iloc[1:] + s.iloc[:-1])
print((s.iloc[1:] + s.iloc[:-1]).dropna())
print(s.dropna())
print(s.name)
s2 = s.rename('different')
print(s2.name)

#  Pandas DataFrame
d = {
    'one': pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c']),
    'two': pd.Series([1.0, 2.0, 3.0, 4.0], index=['a', 'b', 'c', 'd'])
}
df = pd.DataFrame(d)
print(df)

print(pd.DataFrame(d, index=['d', 'b', 'c']))
print(pd.DataFrame(d, index=['d', 'b', 'c'], columns=['two', 'three']))
print(df.index)
print(df.columns)

d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}
print(pd.DataFrame(d))
print(pd.DataFrame(d, index=['a', 'b', 'c', 'd']))

data = np.zeros((2,), dtype=[("A", "i4"), ("B", "f4"), ("C", "S10")])
print(data)
data[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]
pd.DataFrame(data)
print(pd.DataFrame(data, index=['first', 'second']))
print(pd.DataFrame(data, columns=['C', 'A', 'B']))

data2 = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]
print(pd.DataFrame(data2, index=['first', 'second']))
print(pd.DataFrame(data2, columns=['a', 'b']))

print(pd.DataFrame({
    ("a", "b"): {("A", "B"): 1, ("A", "C"): 2},
    ("a", "a"): {("A", "C"): 3, ("A", "B"): 4},
    ("a", "c"): {("A", "B"): 5, ("A", "C"): 6},
    ("b", "a"): {("A", "C"): 7, ("A", "B"): 8},
    ("b", "b"): {("A", "D"): 9, ("A", "B"): 10},
    }
))

ser = pd.Series(range(3), index=list('abc'), name='ser')
print(pd.DataFrame(ser))

print(pd.DataFrame.from_dict(dict([("A", [1, 2, 3]), ("B", [4, 5, 6])])))
print(pd.DataFrame.from_dict(dict([("A", [1, 2, 3]), ("B", [4, 5, 6])]), orient='index', columns=list('xyz')))

print(data)
print(pd.DataFrame.from_records(data, index='B'))
print(pd.DataFrame.from_records(data, index='A'))
print(df)

df['three'] = df['one'] * df['two']
df['flg'] = df['one'] > 2
print(df)
del df['two']
three = df.pop('three')
print(three)
print(df)
df['foo'] = 'bar'
print(df)
df['one_trunc'] = df['one'][:2]
print(df)
df.insert(4, "bar", df["one"])
print(df)

iris = pd.read_csv('/home/bigdata/PycharmProjects/pandas/feed/iris.data')
# iris = pd.read_csv('./feed/iris.data')
print(iris.head())
print(iris.assign(sepal_ratio=iris['SepalWidth']/iris['SepalLength']).head())
print(iris.assign(sepal_ratio=lambda x: x['SepalWidth']/x['SepalLength']).head())

print(iris.query('SepalLength > 5').assign(SepalRatio=lambda x: x.SepalWidth/x.SepalLength,
                                           PetalRatio=lambda y: y.PetalWidth/y.PetalLength
                                           ).plot(kind='scatter', x='SepalRatio', y='PetalRatio'))
plt.show()
print(iris)

dfa = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
print(dfa)
print(dfa.assign(C=lambda x: x['A']+x['B'], D=lambda y: y.C+y.A))
print(df.loc['b'])
print(df.iloc[1])

df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
df2 = pd.DataFrame(np.random.randn(7, 3), columns=list('ABC'))
print(df)
print(df2)
print(df + df2)
print(df - df.iloc[0])
print(df2 * 5 + 2)

df1 = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]}, dtype=bool)
df2 = pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]}, dtype=bool)
print(df1 & df2)
print(df1 | df2)
print(df1 ^ df2)
print(-df1)
print(df)
print(df.T)

print(np.exp(df))
print(np.asarray(df))

ser = pd.Series([1, 2, 3, 4])
print(ser)
print(np.exp(ser))

ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
ser2 = pd.Series([1, 3, 5], index=["b", "a", "c"])
print(np.remainder(ser1, ser2))
ser3 = pd.Series([2, 4, 6], index=["b", "c", "d"])
print(np.remainder(ser1, ser3))


ser = pd.Series([1, 2, 3])
idx = pd.Index([4, 5, 6])
np.maximum(ser, idx)

baseball = pd.read_csv('/home/bigdata/PycharmProjects/pandas/feed/baseball.csv')
print(baseball.info())

print(baseball.to_string())

print(pd.DataFrame(np.random.randn(3, 5), columns=list('abcde')))
print(pd.DataFrame(np.random.randn(3, 5), columns=list('abcde')).sort_index(ascending=False, axis=0))
print(pd.DataFrame(np.random.randn(3, 5), columns=list('abcde')).sort_index(ascending=False, axis=1))

pd.set_option("display.width", 40)

df = pd.DataFrame({"foo1": np.random.randn(5), "foo2": np.random.randn(5)})
print(df)
