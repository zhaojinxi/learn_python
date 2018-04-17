import pyspark
import pyspark.sql.functions

SparkSession=pyspark.sql.SparkSession.builder.getOrCreate()

textFile = SparkSession.read.text("file:///usr/local/spark/README.md")
textFile.count()  # Number of rows in this DataFrame
textFile.first()  # First row in this DataFrame
linesWithSpark = textFile.filter(textFile.value.contains("Spark"))
textFile.filter(textFile.value.contains("Spark")).count()  # How many lines contain "Spark"?

textFile.select(pyspark.sql.functions.size(pyspark.sql.functions.split(textFile.value, "\s+")).name("numWords")).agg(pyspark.sql.functions.max(pyspark.sql.functions.col("numWords"))).collect()
wordCounts = textFile.select(pyspark.sql.functions.explode(pyspark.sql.functions.split(textFile.value, "\s+")).alias("word")).groupBy("word").count()
wordCounts.collect()

linesWithSpark.cache()
linesWithSpark.count()
linesWithSpark.count()

"""SimpleApp.py"""
from pyspark.sql import SparkSession
logFile = "YOUR_SPARK_HOME/README.md"  # Should be some file on your system
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
logData = spark.read.text(logFile).cache()
numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()
print("Lines with a: %i, lines with b: %i" % (numAs, numBs))
spark.stop()