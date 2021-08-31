import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.linalg.{Vectors => MLVectors}
import org.apache.spark.ml.feature.{LabeledPoint => MLabeledPoint}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.StandardScaler
import scala.collection.immutable.ListMap
import org.apache.spark.sql.functions.rand
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{FMRegressionModel, FMRegressor}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.classification.{FMClassificationModel, FMClassifier}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.regression.LinearRegression

def getData(s : String )= {
    spark.sparkContext.wholeTextFiles(s,16)
}


def cleanRdd(base : RDD[(String,String)]) = {
 base.flatMap(file=>{
   val lines = file._2.split("\n")
   val label = lines.head.substring(8).toDouble
   val linesc = file._2.split("\n").drop(3).take(2000)
   linesc.map((x=>(label-11.0,MLVectors.dense(x.split(',').map(_.toDouble).take(24)))))
 })
}

val trainRdd = getData("/archive/users/manosanag/data_eeg_age_v1/data2kaggle/train/*.csv") 
val evalRdd = getData("/archive/users/manosanag/data_eeg_age_v1/data2kaggle/eval/*.csv")
//val trainRdd = getData("s3://hy543bucket/train/*.csv") 
//val evalRdd = getData("s3://hy543bucket/eval/*.csv")

val train_cl = (cleanRdd(trainRdd)).groupBy(x=>x._1).map(y=>(y._2.take(2000)))
val eval_cl  = spark.createDataFrame(cleanRdd(evalRdd)).toDF("label", "features")

val train_cl_list = train_cl.collect().toList
var final_list2 : List[(Double, org.apache.spark.ml.linalg.Vector)] = List()
for(i <- 0 to train_cl_list.size - 1){
  val trainDataList2 = train_cl_list(i).toList
  for(j <- 0 to trainDataList2.size - 1){
    final_list2 = trainDataList2(j) :: final_list2
  }
}

val training = sc.parallelize(final_list2.reverse, 50).toDF("label","features").cache

//Code below, checks which model fits best with our prediction problem ,trying different parameters and data transformations.
//One function for each candidate model
//More like Manual gridSearch 
//Gradient Boosted Tree 

//Good RMSE but predictions converge to 40-45(average age of dataset)
def _gbt()={
    
    val featureIndexer = new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4)
        .fit(training.union(eval_cl))
    val gbt = new GBTRegressor()
        .setLabelCol("label")
        .setFeaturesCol("indexedFeatures")
        .setMaxIter(1)
    val pipeline = new Pipeline()
        .setStages(Array( featureIndexer, gbt))

    val paramGrid = new ParamGridBuilder()
        .addGrid(gbt.maxBins,Array(32))
        .addGrid(gbt.maxDepth,Array(0,5))
        .addGrid(gbt.minInstancesPerNode, Array(5,10,50))
        .addGrid(gbt.minWeightFractionPerNode, Array(0.12)) //0.1 0.2 = 17.19420 
        //.addGrid(gbt.minInfoGain, Array(0.000001))
    val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid.build()).setNumFolds(6).setParallelism(2)  
    val cvModel = cv.fit(training)
    val predictions= cvModel.transform(eval_cl)

    predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
}

//Decision Tree Classifier
def _dtc()={
    val dt = new DecisionTreeClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")
    val pipeline = new Pipeline()
    .setStages(Array( dt))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(training)
    // Make predictions.
    //val predictions = model.transform(evaluation)
    val predictions = model.transform(eval_cl)


    // Select example rows to display.
    predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
}

//Decision Tree Regressor
def _dtr()={
    
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(24).fit(training.union(eval_cl))
    val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxBins(32).setMaxDepth(30).setMinInstancesPerNode(16)
    //.setMinWeightFractionPerNode(0.001)
    //.setMinInfoGain(8)
    val pipeline = new Pipeline()
    .setStages(Array( featureIndexer, dt))
    // Train model. This also runs the indexers.
    val model2 = pipeline.fit(training)
    //val model2 = pipeline.fit(train_cl)
    // Make predictions.
    val predictions = model2.transform(eval_cl)
    // Select example rows to display.
    predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val evaluator2 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")    
  val rmse = evaluator.evaluate(predictions)
  val mae = evaluator2.evaluate(predictions)
  println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
  println(s"Root Mean Squared Error (MAE) on test data = $mae")

}
//Random Forest Regressor
def _rfr()={

    val rf = new RandomForestRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")


    val pipeline = new Pipeline()
    .setStages(Array(  rf))

    val model = pipeline.fit(training)

    val predictions = model.transform(eval_cl)

    predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)

    val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("mae")

    val mae = evaluator.evaluate(predictions)

    println(s"mae on test data = $mae")

}
//Random Forest Classifier
def _rfc()={
        
    val rf = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    /*val pipeline = new Pipeline()
    .setStages(Array(  rf))*/
    val model = rf.fit(training)
    //val predictions = model.transform(valDataDF)
    val predictions = model.transform(eval_cl)

    predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

    val evaluator2 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")   

    val rmse = evaluator.evaluate(predictions)
    val mae = evaluator2.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    println(s"Mean Average Error (MAE) on test data = $mae")

}
//Linear Regressor
def _linr()={

    val lr = new LinearRegression()
    .setMaxIter(60)
    .setRegParam(0.01)
    .setElasticNetParam(0.08)

    val model = lr.fit(training)

    val predictions = model.transform(eval_cl)

    predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

    val evaluator2 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")    
    val rmse = evaluator.evaluate(predictions)
    val mae = evaluator2.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    println(s"Root Mean Squared Error (MAE) on test data = $mae")
}
//Logistic Regression
def _logr()={
    /*val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)*/
    val lr = new LogisticRegression()
    .setMaxIter(60)
    .setRegParam(0.01)
    .setElasticNetParam(0.08)
    .setFamily("multinomial")

    val model = lr.fit(training)

    val predictions = model.transform(eval_cl)

    predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

    val evaluator2 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")    
    val rmse = evaluator.evaluate(predictions)
    val mae = evaluator2.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    println(s"Root Mean Squared Error (MAE) on test data = $mae")
}


def _fmr()={
    val featureScaler = new MinMaxScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .fit(training.union(eval_cl))

    val fm = new FMRegressor()
    .setLabelCol("label")
    .setFeaturesCol("scaledFeatures")
    .setStepSize(0.04)

    // Create a Pipeline.
    val pipeline = new Pipeline()
    .setStages(Array(featureScaler, fm))

    // Train model.
    val model = pipeline.fit(training)

    // Make predictions.
    val predictions = model.transform(eval_cl)

    predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)

    val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

    val mae = evaluator.evaluate(predictions)

    println(s"mae on test data = $mae")
}
