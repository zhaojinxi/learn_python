import pyspark
import pyspark.streaming

# Create a local StreamingContext with two working thread and batch interval of 1 second
SparkContext = pyspark.SparkContext("local[2]", "NetworkWordCount")
StreamingContext = pyspark.streaming.StreamingContext(SparkContext, 1)

# Create a DStream that will connect to hostname:port, like localhost:9999
lines = StreamingContext.socketTextStream("localhost", 9999)

# Split each line into words
words = lines.flatMap(lambda line: line.split(" "))

# Count each word in each batch
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)
# Print the first ten elements of each RDD generated in this DStream to the console
wordCounts.pprint()

StreamingContext.start()             # Start the computation
StreamingContext.awaitTermination()  # Wait for the computation to terminate