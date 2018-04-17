import pyspark

conf = pyspark.SparkConf().setAppName('test').setMaster('local[*]')
SparkContext = pyspark.SparkContext(conf=conf)

data = [1, 2, 3, 4, 5]
distData = SparkContext.parallelize(data)

distFile = SparkContext.textFile("file:///usr/local/spark/README.md")

rdd = SparkContext.parallelize(range(1, 4)).map(lambda x: (x, "a" * x))
rdd.saveAsSequenceFile("file:///home/zhao/文档/hahaha")
sorted(SparkContext.sequenceFile("file:///home/zhao/文档/hahaha").collect())

conf = {"es.resource" : "index/type"}  # assume Elasticsearch is running on localhost defaults
rdd = SparkContext.newAPIHadoopRDD("org.elasticsearch.hadoop.mr.EsInputFormat", "org.apache.hadoop.io.NullWritable", "org.elasticsearch.hadoop.mr.LinkedMapWritable", conf=conf)
rdd.first() 

lines = SparkContext.textFile("file:///usr/local/spark/README.md")
lineLengths = lines.map(lambda s: len(s))
totalLength = lineLengths.reduce(lambda a, b: a + b)

lineLengths.persist()

def myFunc(s):
    words = s.split(" ")
    return len(words)
SparkContext.textFile("file.txt").map(myFunc)

counter = 0
rdd = SparkContext.parallelize(data)
# Wrong: Don't do this!!
def increment_counter(x):
    global counter
    counter += x
rdd.foreach(increment_counter)
print("Counter value: ", counter)

lines = SparkContext.textFile("data.txt")
pairs = lines.map(lambda s: (s, 1))
counts = pairs.reduceByKey(lambda a, b: a + b)

accum = SparkContext.accumulator(0)
accum
SparkContext.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
accum.value

class VectorAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return Vector.zeros(initialValue.size)

    def addInPlace(self, v1, v2):
        v1 += v2
        return v1
# Then, create an Accumulator of this type:
vecAccum = SparkContext.accumulator(Vector(...), VectorAccumulatorParam())

accum = SparkContext.accumulator(0)
def g(x):
    accum.add(x)
    return f(x)
data.map(g)

