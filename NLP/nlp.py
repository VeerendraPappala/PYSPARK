

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('nlp').getOrCreate()
df = spark.read.csv('SMSSpamCollection', inferSchema = True, sep = '\t')
df.printSchema()

df = df.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1', 'text')
df.show()

from pyspark.sql.functions import length

df = df.withColumn('length', length(df['text']))
df.show(3)

df.groupBy('class').mean().show()

from pyspark.ml.feature import (CountVectorizer, Tokenizer, 
                                StopWordsRemover, IDF, StringIndexer)

tokenizer = Tokenizer(inputCol = 'text', outputCol = 'token_text')
stop_remove = StopWordsRemover(inputCol = 'token_text', outputCol = 'stop_token')
count_vec = CountVectorizer(inputCol = 'stop_token', outputCol = 'c_vec')
idf = IDF(inputCol = 'c_vec', outputCol = 'tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol = 'class', outputCol = 'label')

from pyspark.ml.feature import VectorAssembler

clean_up = VectorAssembler(inputCols = ['tf_idf', 'length'], outputCol = 'features')

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[ham_spam_to_numeric, tokenizer, stop_remove, count_vec, idf, clean_up])

cleaner = pipeline.fit(df)
clean_df = cleaner.transform(df)
clean_df = clean_df.select('label', 'features')
clean_df.show(3)

train, test = clean_df.randomSplit([0.7, 0.3])

df.printSchema()

spam_detector = nb.fit(train)
predictions = spam_detector.transform(test)
predictions.show(3)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator()
print("Test Accuracy: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))