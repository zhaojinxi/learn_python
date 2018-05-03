import pyspark
spark= pyspark.sql.SparkSession.builder.appName("dataFrame").getOrCreate()
# 导入类型
from pyspark.sql.types import *
#生成以逗号分隔的数据
stringCSVRDD = spark.sparkContext.parallelize([
    (123,"Katie",19,"brown"),
    (234,"Michael",22,"green"),
    (345,"Simone",23,"blue")])
#指定模式,StructField(name,dataType,nullable)其中name:该字段的名字，dataType：该字段的数据类型，nullable:指示该字段的值是否为空
schema = StructType([
    StructField("id",LongType(),True),
    StructField("name",StringType(),True),
    StructField("age",LongType(),True),
    StructField("eyeColor",StringType(),True)])
#schema可以是一个StructType类，也可以是各列名称所构成的列表如下：
schema=['id','name','age','eyeColor']
#对RDD应用该模式并且创建DataFrame
swimmers = spark.createDataFrame(stringCSVRDD,schema)
#也可以采用spark.createDataFrame(rdd, schema, sampleRatio)的缩写方式：
swimmers = stringCSVRDD.toDF(schema)
#利用DataFrame创建一个临时视图
swimmers.createOrReplaceGlobalTempView("swimmers")
print(swimmers.printSchema())