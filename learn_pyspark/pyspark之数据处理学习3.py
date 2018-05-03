from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("dataDeal").getOrCreate()
df_outliers = spark.createDataFrame([
    (1, 143.5, 5.3, 28),
    (2, 154.2, 5.5, 45),
    (3, 342.3, 5.1, 99),
    (4, 144.5, 5.5, 33),
    (5, 133.2, 5.4, 54),
    (6, 124.1, 5.1, 21),
    (7, 129.2, 5.3, 42),
    ], ['id', 'weight', 'height', 'age'])
cols = ['weight','height','age']
bounds = {}
for col in cols:
    quantiles = df_outliers.approxQuantile(col,[0.25,0.75],0.05)
    IQR = quantiles[1] - quantiles[0]
    bounds[col] = [
        quantiles[0] - 1.5*IQR,
        quantiles[1] + 1.5*IQR
    ]
print(bounds)
#标记离群值
outliers = df_outliers.select(*['id']+[(
    (df_outliers[c] < bounds[c][0]) |
    (df_outliers[c]>bounds[c][1])
).alias(c+'_o') for c in cols
])
outliers.show()
#列出和其他数据分布明显不同的值
df_outliers = df_outliers.join(outliers,on='id')
df_outliers.filter('weight_o').select('id','weight').show()
df_outliers.filter('age_o').select('id','age').show()
