import pyspark
import pyspark.sql.functions

SparkSession = pyspark.sql.SparkSession.builder.appName("StructuredNetworkWordCount").getOrCreate()

lines = SparkSession.readStream.format("socket").option("host", "localhost").option("port", 9999).load()
# Split the lines into words
words = lines.select(pyspark.sql.functions.explode(pyspark.sql.functions.split(lines.value, " ")).alias("word"))
# Generate running word count
wordCounts = words.groupBy("word").count()

query = wordCounts.writeStream.outputMode("complete").format("console").start()
query.awaitTermination()