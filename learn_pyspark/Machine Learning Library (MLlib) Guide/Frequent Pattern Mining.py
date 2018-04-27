import pyspark

spark=pyspark.sql.SparkSession.builder.getOrCreate()

from pyspark.ml.fpm import FPGrowth
df = spark.createDataFrame([
    (0, [1, 2, 5]),
    (1, [1, 2, 3, 5]),
    (2, [1, 2])
], ["id", "items"])
fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(df)
# Display frequent itemsets.
model.freqItemsets.show()
# Display generated association rules.
model.associationRules.show()
# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(df).show()