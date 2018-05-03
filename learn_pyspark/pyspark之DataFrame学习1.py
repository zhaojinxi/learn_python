import pyspark
spark= pyspark.sql.SparkSession.builder.appName("dataFrame").getOrCreate()
#1生成JSON数据
stringJSONRDD = spark.sparkContext.parallelize((
    '''{
    "id":"123",
    "name":"Katie",
    "age":19,
    "eyeColor":"brown"
    }''',
    '''{
    "id":"234",
    "name":"Michael",
    "age":22,
    "eyeColor":"green"
    }''',
    '''{
    "id":"345",
    "name":"Simone",
    "age":23,
    "eyeColor":"blue"
    }'''))
#2创建DataFrame
swimmersJSON = spark.read.json(stringJSONRDD)
#3创建一个临时表
swimmersJSON.createOrReplaceTempView("swimmersJSON")
#4查看前几行
print(swimmersJSON.show())#默认是10行
#5可以通过编写sql语句查询
print(spark.sql("select * from swimmersJSON").collect())
#6打印模式
print(swimmersJSON.printSchema())