import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.{Vectors => MLVectors}
import org.apache.spark.ml.feature.{LabeledPoint => MLabeledPoint}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import scala.collection.immutable.ListMap
import org.apache.spark.sql.functions.rand
import org.apache.spark.ml.Pipeline

//getdata and set partitions
def getData(s : String )= {
    spark.sparkContext.wholeTextFiles(s,16)
}

//data pre-processing clear.
def cleanRdd(base : RDD[(String,String)]) = {
 base.flatMap(file=>{
   val lines = file._2.split("\n")
   val label = lines.head.substring(8).toDouble
   val linesc = file._2.split("\n").drop(3).take(100000)
   linesc.map((x=>(label,MLVectors.dense(x.split(',').map(_.toDouble).take(24)))))
 })
}



val trainRdd = getData("s3://hy543bucket/train/*.csv") 
val evalRdd = getData("s3://hy543bucket/eval/*.csv")

//balance dataset
val train_cl = (cleanRdd(trainRdd)).groupBy(x=>x._1).map(y=>(y._2.take(100000)))
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


val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(24).fit(training.union(eval_cl))

val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxBins(32).setMaxDepth(30).setMinInstancesPerNode(16)

val pipeline = new Pipeline()
  .setStages(Array( featureIndexer, dt))

val model2 = pipeline.fit(training)

val predictions = model2.transform(eval_cl)

predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)

val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

val evaluator2 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")    
  val rmse = evaluator.evaluate(predictions)
  val mae = evaluator2.evaluate(predictions)
  println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
  println(s"Root Mean Squared Error (MAE) on test data = $mae")

  println(s"Multinomial coefficients: ${model2.coefficientMatrix}")


