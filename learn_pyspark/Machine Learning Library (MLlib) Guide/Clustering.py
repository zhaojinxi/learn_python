import pyspark

spark=pyspark.sql.SparkSession.builder.getOrCreate()

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
# Loads data.
dataset = spark.read.format("libsvm").load("file:///usr/local/spark/data/mllib/sample_kmeans_data.txt")
# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)
# Make predictions
predictions = model.transform(dataset)
# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

from pyspark.ml.clustering import LDA
# Loads data.
dataset = spark.read.format("libsvm").load("file:///usr/local/spark/data/mllib/sample_lda_libsvm_data.txt")
# Trains a LDA model.
lda = LDA(k=10, maxIter=10)
model = lda.fit(dataset)
ll = model.logLikelihood(dataset)
lp = model.logPerplexity(dataset)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))
# Describe topics.
topics = model.describeTopics(3)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)
# Shows the result
transformed = model.transform(dataset)
transformed.show(truncate=False)   

from pyspark.ml.clustering import BisectingKMeans
# Loads data.
dataset = spark.read.format("libsvm").load("file:///usr/local/spark/data/mllib/sample_kmeans_data.txt")
# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(2).setSeed(1)
model = bkm.fit(dataset)
# Evaluate clustering.
cost = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(cost))
# Shows the result.
print("Cluster Centers: ")
centers = model.clusterCenters()
for center in centers:
    print(center)

from pyspark.ml.clustering import GaussianMixture
# loads data
dataset = spark.read.format("libsvm").load("file:///usr/local/spark/data/mllib/sample_kmeans_data.txt")
gmm = GaussianMixture().setK(2).setSeed(538009335)
model = gmm.fit(dataset)
print("Gaussians shown as a DataFrame: ")
model.gaussiansDF.show(truncate=False)    