import pyspark
spark = pyspark.sql.SparkSession.builder.appName("dataDeal").getOrCreate()
df = spark.createDataFrame([
    (1, 144.5, 5.9, 33, 'M'),
    (2, 167.2, 5.4, 45, 'M'),
    (3, 124.1, 5.2, 23, 'F'),
    (4, 144.5, 5.9, 33, 'M'),
    (5, 133.2, 5.7, 54, 'F'),
    (3, 124.1, 5.2, 23, 'F'),
    (5, 129.2, 5.3, 42, 'M'),
    ], ['id', 'weight', 'height', 'age', 'gender'])
print('Count of rows:{0}'.format(df.count()))
print('Count of distinct rows:{0}'.format(df.distinct().count()))
#移除重复的数据
df = df.dropDuplicates()
#查看去重后的数据
df.show()
#对除id以外的列进行对比
print("Count of ids:{0}".format(df.count()))
print("Count of distinct ids:{0}".format(df.select([c for c in df.columns if c != 'id']).distinct().count()))
#去掉除id以外其他属性相同的数据
df = df.dropDuplicates(subset=[c for c in df.columns if c != 'id'])
df.show()
#计算id的总数和id的唯一数
import pyspark.sql.functions as fn
df.agg(fn.count('id').alias('count'), fn.countDistinct('id').alias('distinct')).show()
#重新给每行分配id
df.withColumn('new_id',fn.monotonically_increasing_id()).show()