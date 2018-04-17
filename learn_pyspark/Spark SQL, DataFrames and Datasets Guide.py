import pyspark
import os
import pyspark.sql.functions
import numpy
import pandas

SparkSession=pyspark.sql.SparkSession.builder.getOrCreate()

df = SparkSession.read.json("file:///usr/local/spark/examples/src/main/resources/people.json")
df.show()
df.printSchema()
df.select("name").show()
df.select(df['name'], df['age'] + 1).show()
df.filter(df['age'] > 21).show()
df.groupBy("age").count().show()

df.createOrReplaceTempView("people")
sqlDF = SparkSession.sql("SELECT * FROM people")
sqlDF.show()

df.createGlobalTempView("people")
SparkSession.sql("SELECT * FROM global_temp.people").show()
SparkSession.newSession().sql("SELECT * FROM global_temp.people").show()

lines = SparkSession.sparkContext.textFile("file:///usr/local/spark/examples/src/main/resources/people.txt")
parts = lines.map(lambda l: l.split(","))
people = parts.map(lambda p: pyspark.sql.Row(name=p[0], age=int(p[1])))
schemaPeople = SparkSession.createDataFrame(people)
schemaPeople.createOrReplaceTempView("people")
teenagers = SparkSession.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")
teenNames = teenagers.rdd.map(lambda p: "Name: " + p.name).collect()
for name in teenNames:
    print(name)

lines = SparkSession.sparkContext.textFile("file:///usr/local/spark/examples/src/main/resources/people.txt")
parts = lines.map(lambda l: l.split(","))
people = parts.map(lambda p: (p[0], p[1].strip()))
schemaString = "name age"
fields = [pyspark.sql.types.StructField(field_name, pyspark.sql.types.StringType(), True) for field_name in schemaString.split()]
schema = pyspark.sql.types.StructType(fields)
schemaPeople = SparkSession.createDataFrame(people, schema)
schemaPeople.createOrReplaceTempView("people")
results = SparkSession.sql("SELECT name FROM people")
results.show()

df = SparkSession.read.load("file:///usr/local/spark/examples/src/main/resources/users.parquet")
df.select("name", "favorite_color").write.save("file:///home/zhao/文档/namesAndFavColors.parquet")

df = SparkSession.read.load("file:///usr/local/spark/examples/src/main/resources/people.json", format="json")
df.select("name", "age").write.save("file:///home/zhao/文档/namesAndAges.parquet", format="parquet")

df = SparkSession.read.load("file:///usr/local/spark/examples/src/main/resources/people.csv", format="csv", sep=":", inferSchema="true", header="true")

df = SparkSession.sql("SELECT * FROM parquet.`file:///usr/local/spark/examples/src/main/resources/users.parquet`")

df = SparkSession.read.json("file:///usr/local/spark/examples/src/main/resources/people.json")
df.write.bucketBy(42, "name").sortBy("age").saveAsTable("people_bucketed")

df = SparkSession.read.load("file:///usr/local/spark/examples/src/main/resources/users.parquet")
df.write.partitionBy("favorite_color").format("parquet").save("file:///home/zhao/文档/namesPartByColor.parquet")

df = SparkSession.read.parquet("file:///usr/local/spark/examples/src/main/resources/users.parquet")
df.write.partitionBy("favorite_color").bucketBy(42, "name").saveAsTable("people_partitioned_bucketed")

peopleDF = SparkSession.read.json("file:///usr/local/spark/examples/src/main/resources/people.json")
peopleDF.write.parquet("file:///home/zhao/文档/people.parquet")
parquetFile = SparkSession.read.parquet("file:///home/zhao/文档/people.parquet")
parquetFile.createOrReplaceTempView("parquetFile")
teenagers = SparkSession.sql("SELECT name FROM parquetFile WHERE age >= 13 AND age <= 19")
teenagers.show()

squaresDF = SparkSession.createDataFrame(SparkSession.sparkContext.parallelize(range(1, 6)).map(lambda i: pyspark.sql.Row(single=i, double=i ** 2)))
squaresDF.write.parquet("file:///home/zhao/文档/data/test_table/key=1")
cubesDF = SparkSession.createDataFrame(SparkSession.sparkContext.parallelize(range(6, 11)).map(lambda i: pyspark.sql.Row(single=i, triple=i ** 3)))
cubesDF.write.parquet("file:///home/zhao/文档/data/test_table/key=2")
mergedDF = SparkSession.read.option("mergeSchema", "true").parquet("file:///home/zhao/文档/data/test_table")
mergedDF.printSchema()

SparkSession.catalog.refreshTable("my_table")

peopleDF = SparkSession.read.json("file:///usr/local/spark/examples/src/main/resources/people.json")
peopleDF.printSchema()
peopleDF.createOrReplaceTempView("people")
teenagerNamesDF = SparkSession.sql("SELECT name FROM people WHERE age BETWEEN 13 AND 19")
teenagerNamesDF.show()
jsonStrings = ['{"name":"Yin","address":{"city":"Columbus","state":"Ohio"}}']
otherPeopleRDD = SparkSession.sparkContext.parallelize(jsonStrings)
otherPeople = SparkSession.read.json(otherPeopleRDD)
otherPeople.show()

warehouse_location = os.path.abspath('spark-warehouse')

SparkSession = pyspark.sql.SparkSession.builder.appName("Python Spark SQL Hive integration example").config("spark.sql.warehouse.dir", warehouse_location).enableHiveSupport().getOrCreate()
SparkSession.sql("CREATE TABLE IF NOT EXISTS src (key INT, value STRING) USING hive")
SparkSession.sql("LOAD DATA LOCAL INPATH 'examples/src/main/resources/kv1.txt' INTO TABLE src")
SparkSession.sql("SELECT * FROM src").show()
SparkSession.sql("SELECT COUNT(*) FROM src").show()
sqlDF = SparkSession.sql("SELECT key, value FROM src WHERE key < 10 ORDER BY key")
stringsDS = sqlDF.rdd.map(lambda row: "Key: %d, Value: %s" % (row.key, row.value))
for record in stringsDS.collect():
    print(record)
Record = pyspark.sql.Row("key", "value")
recordsDF = SparkSession.createDataFrame([Record(i, "val_" + str(i)) for i in range(1, 101)])
recordsDF.createOrReplaceTempView("records")
SparkSession.sql("SELECT * FROM records r JOIN src s ON r.key = s.key").show()

jdbcDF = SparkSession.read.format("jdbc").option("url", "jdbc:postgresql:dbserver").option("dbtable", "schema.tablename").option("user", "username").option("password", "password").load()
jdbcDF2 = SparkSession.read.jdbc("jdbc:postgresql:dbserver", "schema.tablename", properties={"user": "username", "password": "password"})
jdbcDF3 = SparkSession.read.format("jdbc").option("url", "jdbc:postgresql:dbserver").option("dbtable", "schema.tablename").option("user", "username").option("password", "password").option("customSchema", "id DECIMAL(38, 0), name STRING").load()
jdbcDF.write.format("jdbc").option("url", "jdbc:postgresql:dbserver").option("dbtable", "schema.tablename").option("user", "username").option("password", "password").save()
jdbcDF2.write.jdbc("jdbc:postgresql:dbserver", "schema.tablename", properties={"user": "username", "password": "password"})
jdbcDF.write.option("createTableColumnTypes", "name CHAR(64), comments VARCHAR(1024)").jdbc("jdbc:postgresql:dbserver", "schema.tablename", properties={"user": "username", "password": "password"})

pyspark.sql.functions.broadcast(SparkSession.table("src")).join(SparkSession.table("records"), "key").show()

SparkSession.conf.set("spark.sql.execution.arrow.enabled", "true")
pdf = pandas.DataFrame(numpy.random.rand(100, 3))
df = SparkSession.createDataFrame(pdf)
result_pdf = df.select("*").toPandas()

def multiply_func(a, b):
    return a * b
multiply = pyspark.sql.functions.pandas_udf(multiply_func, returnType=pyspark.sql.types.LongType())
x = pandas.Series([1, 2, 3])
print(multiply_func(x, x))
df = SparkSession.createDataFrame(pandas.DataFrame(x, columns=["x"]))
df.select(multiply(pyspark.sql.functions.col("x"), pyspark.sql.functions.col("x"))).show()

df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))
@pyspark.sql.functions.pandas_udf("id long, v double", pyspark.sql.functions.PandasUDFType.GROUPED_MAP)
def substract_mean(pdf):
    # pdf is a pandas.DataFrame
    v = pdf.v
    return pdf.assign(v=v - v.mean())
df.groupby("id").apply(substract_mean).show()