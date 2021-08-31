import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.{Vectors => MLVectors}
import org.apache.spark.ml.feature.{LabeledPoint => MLabeledPoint}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.RegressionEvaluator
import scala.collection.immutable.ListMap
import org.apache.spark.sql.functions.rand
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.StandardScaler


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
   linesc.map((x=>(label.toDouble-11.0,MLVectors.dense(x.split(',').map(_.toDouble).take(24)))))
 })
}

val trainRdd = getData("s3://hy543bucket/train/*.csv") 
val evalRdd = getData("s3://hy543bucket/eval/*.csv")

//balance dataset : take same even number of features from each label
val train_cl = (cleanRdd(trainRdd)).groupBy(x=>x._1).map(y=>(y._2.take(100000)))

val eval_cl  =spark.createDataFrame(cleanRdd(evalRdd)).toDF("label", "features")

val train_cl_list = train_cl.collect().toList
var final_list2 : List[(Double, org.apache.spark.ml.linalg.Vector)] = List()
for(i <- 0 to train_cl_list.size - 1){
  val trainDataList2 = train_cl_list(i).toList
  for(j <- 0 to trainDataList2.size - 1){
    final_list2 = trainDataList2(j) :: final_list2
  }
}

val finalTrainDataTuple = sc.parallelize(final_list2.reverse, 50).toDF("label","features")

//TrainData Scaler did not achieve (much) higher accuracy accordingly to its cost.
//val scaler = new StandardScaler(withMean = true, withStd = true).fit(finalTrainDataTuple.map(x=>Vectors.dense(x._2.toArray)))


//linear regression
val lr = new LogisticRegression()
  .setMaxIter(60)
  .setRegParam(0.01)
  .setElasticNetParam(0.08)
  .setFamily("multinomial")

val model = lr.fit(finalTrainDataTuple)

val predictions = model.transform(eval_cl)

predictions.select("prediction", "label", "features").orderBy(rand()).limit(100).show(100)

val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

val evaluator2 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")    
  val rmse = evaluator.evaluate(predictions)
  val mae = evaluator2.evaluate(predictions)
  println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
  println(s"Root Mean Squared Error (MAE) on test data = $mae")

 println(s"Multinomial coefficients: ${model.coefficientMatrix}")


