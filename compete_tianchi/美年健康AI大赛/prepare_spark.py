import pyspark

spark=pyspark.sql.SparkSession.builder.getOrCreate()

data1=spark.read.text('file:///home/zhao/Documents/learn_python/compete_tianchi/美年健康AI大赛/data/meinian_round1_data_part1_20180408.txt')
data2=spark.read.text('file:///home/zhao/Documents/learn_python/compete_tianchi/美年健康AI大赛/data/meinian_round1_data_part2_20180408.txt')
test=spark.read.csv('file:///home/zhao/Documents/learn_python/compete_tianchi/美年健康AI大赛/data/meinian_round1_test_a_20180409.csv', encoding='GBK')
train=spark.read.csv('file:///home/zhao/Documents/learn_python/compete_tianchi/美年健康AI大赛/data/meinian_round1_train_20180408.csv', encoding='GBK')

x=data1.collect()
print(x[0])
data1.show(truncate=False)
data1=data1.withColumnRenamed('value','haha')