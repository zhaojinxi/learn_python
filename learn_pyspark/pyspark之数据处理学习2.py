from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("dataDeal").getOrCreate()
df_miss = spark.createDataFrame([
    (1, 143.5, 5.6, 28,   'M',  100000),
    (2, 167.2, 5.4, 45,   'M',  None),
    (3, None , 5.2, None, None, None),
    (4, 144.5, 5.9, 33,   'M',  None),
    (5, 133.2, 5.7, 54,   'F',  None),
    (6, 124.1, 5.2, None, 'F',  None),
    (7, 129.2, 5.3, 42,   'M',  76000)
    ], ['id', 'weight', 'height', 'age', 'gender', 'income'])
#查看每行记录的缺失值
print(df_miss.rdd.map(lambda row:(row['id'],sum([c==None for c in row]))).collect())
#查看id为3的记录
df_miss.where('id==3').show()
#检查每一列中缺失数据的百分比
import pyspark.sql.functions as fn
df_miss.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_miss.columns]).show()
#移除‘income'属性
df_miss_no_income = df_miss.select([c for c in df_miss.columns if c != 'income'])
df_miss_no_income.show()
#移除id为3的行
df_miss_no_income.dropna(thresh=3).show()
#计算均值
means = df_miss_no_income.agg(*[fn.mean(c).alias(c) for c in df_miss_no_income.columns if c != 'gender']).toPandas().to_dict('records')[0]
means['gender']='missing'
#填充
df_miss_no_income.fillna(means).show()