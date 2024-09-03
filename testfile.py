import numpy as np
import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession


# df = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [np.nan, 5, 6],
#     'C': [7, np.nan, 9]
# })
#
# print(df)
# print(df.dropna().dtypes)
# print(df[df.notna()])
# # print(df.removena())
# print(df.where(df != np.nan))
# print(df['B'].dropna())
# print(df.dtypes)
# df1 = df.astype({'A': 'str', 'B': 'float', 'C': 'float'}, errors='raise').dropna()
# print(df1.dtypes)
# # df1.astype({'A': 'str', 'B': 'float', 'C': 'float'}, errors='raise')
# # df['B'] = df['B'].astype('int')
# # df['B'] = pd.to_numeric(df['B'])
# print(df1.dtypes)
# print(df1)

#  ---------------------------------------------------------------------------------------------------------------------

# import numpy as np
# import pandas as pd
#
# # Sample DataFrame
# data = {
#     'A': ['1', '2', '3'],
#     'B': ['4.0', '5.1', '6.2'],
#     'C': [np.nan, 8, 9]
# }
#
# df = pd.DataFrame(data)
# print(df.dtypes)
# # Convert multiple columns at once
# df = df.astype({'A': 'float', 'B': 'float', 'C': 'float'})
#
# print(df.dtypes)

#  ---------------------------------------------------------------------------------------------------------------------

spark = SparkSession.builder.appName('pandasInterviwe').getOrCreate()


def multiply_func(a, b):
    return a + b


multiply = pandas_udf(multiply_func, returnType=IntegerType())
x = pd.Series([1, 2, 3])
print('this is first', multiply_func(x, x))
df = spark.createDataFrame(pd.DataFrame(x, columns=['x']))
df.select(multiply(col('x'), col('x'))).show()
