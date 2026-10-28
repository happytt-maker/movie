import time

from pyspark.sql import SparkSession

t = time.time()
spark = SparkSession.builder.appName("smoke").getOrCreate()
print("spark", spark.version, "started", round(time.time() - t, 1), "s", flush=True)
df = spark.read.csv("file:///E:/ml-32m/ml-32m/movies.csv", header=True, inferSchema=True)
print("movies", df.count(), flush=True)
print(df.select("movieId", "title").limit(3).collect(), flush=True)
spark.stop()
print("done", flush=True)