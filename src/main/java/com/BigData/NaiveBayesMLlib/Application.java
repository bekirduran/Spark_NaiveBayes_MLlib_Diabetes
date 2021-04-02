package com.BigData.NaiveBayesMLlib;

import org.apache.spark.ml.classification.MultiClassSummarizer;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.xerces.xs.StringList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class Application {
    public static void main(String[] args) {

        System.setProperty("hadoop.home.dir", "C:\\hadoop-common-2.2.0-bin-master");
        SparkSession sparkSession = SparkSession.builder().appName("Spark NaiveBayes MLlib Example").master("local").getOrCreate();

        Dataset<Row> rowDataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/java/Dataset/diabetes.csv");

        rowDataset.show();

        String[] headers = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"};
        List<String> headerList = Arrays.asList(headers);
        List<String> forVectorHeaders = new ArrayList<String>();

        //Strings to Indexing Number
        for (String h : headerList) {
            if (h.equals("Outcome")) {
                StringIndexer indexer = new StringIndexer().setInputCol(h).setOutputCol("label");
                rowDataset = indexer.fit(rowDataset).transform(rowDataset);
                forVectorHeaders.add("label");
            } else {
                StringIndexer indexer = new StringIndexer().setInputCol(h).setOutputCol(h.toLowerCase() + "_cast");
                rowDataset = indexer.fit(rowDataset).transform(rowDataset);
                forVectorHeaders.add(h.toLowerCase() + "_cast");
            }

        }

        String[] toArray = forVectorHeaders.toArray(new String[forVectorHeaders.size()]);
        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(toArray).setOutputCol("features");

        Dataset<Row> rowDataset1 = vectorAssembler.transform(rowDataset);

        Dataset<Row> selectedDataSet = rowDataset1.select("features", "label");

        Dataset<Row>[] randomSplit = selectedDataSet.randomSplit(new double[]{0.75, 0.25});
        Dataset<Row> trainingDataSet = randomSplit[0];
        Dataset<Row> testDataSet = randomSplit[1];

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.setSmoothing(1);
        NaiveBayesModel naiveBayesModel = naiveBayes.fit(trainingDataSet);
        Dataset<Row> testDataSetResult = naiveBayesModel.transform(testDataSet);
        testDataSetResult.show();
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracyResult = evaluator.evaluate(testDataSetResult);
        System.out.println("Accuracy Result is: "+accuracyResult);


    }
}
