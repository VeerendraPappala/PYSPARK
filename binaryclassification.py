
## Import Libraries
"""

!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://apache.osuosl.org/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz
!tar xf spark-2.3.2-bin-hadoop2.7.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.3.2-bin-hadoop2.7"

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("adult_logreg").getOrCreate()

df = spark.read.csv('adult.csv', inferSchema = True, header=True)
df.show(3)

cols = df.columns

df.printSchema()

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

categoricalColumns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "native-country"]
stages = []

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'income', outputCol = 'label')
stages += [label_stringIdx]

numericCols = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedcols = ["label", "features"] + cols
df = df.select(selectedcols)
df.show(3)

display(df)

train, test = df.randomSplit([0.7, 0.3], seed=100)
print(train.count())
print(test.count())

"""### Logistic Regression"""

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol = 'label', featuresCol = 'features', maxIter=10)
lrModel = lr.fit(train)

predictions = lrModel.transform(test)
predictions.take(1)

predictions.printSchema()

selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

selected.show(4)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print('Area Under ROC', evaluator.evaluate(predictions))

evaluator.getMetricName()

print(lr.explainParams())

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(train)

predictions = cvModel.transform(test)
print('Area Under ROC', evaluator.evaluate(predictions))

"""### Decision Trees"""

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)

print("numNodes = ", dtModel.numNodes)
print("depth = ", dtModel.depth)

predictions = dtModel.transform(test)
predictions.printSchema()

selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [1, 2, 6, 10])
             .addGrid(dt.maxBins, [20, 40, 80])
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(train)

print("numNodes = ", cvModel.bestModel.numNodes)
print("depth = ", cvModel.bestModel.depth)

predictions = cvModel.transform(test)
evaluator.evaluate(predictions)

selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

selected.show(3)

"""### Random Forest"""

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.printSchema()

selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

selected.show(3)

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
cvModel = cv.fit(train)

predictions = cvModel.transform(test)
evaluator.evaluate(predictions)

selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

selected.show(3)

"""### Make Predictions"""

bestModel = cvModel.bestModel
final_predictions = bestModel.transform(df)
evaluator.evaluate(final_predictions)
