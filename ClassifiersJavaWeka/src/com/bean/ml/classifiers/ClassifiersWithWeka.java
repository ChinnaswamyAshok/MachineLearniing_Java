package com.bean.ml.classifiers;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;

/**
 * @author Ashok Chinnaswamy
 */
public class ClassifiersWithWeka {

  /**
   * @param args
   * @throws Exception
   */
  public static void main(final String[] args) throws Exception {

    /*
     * Loading .arff file from ArffLoader -Training Data setting Classification Class index
     */

    ArffLoader arffloader = new ArffLoader();
    arffloader.setSource(new File("C:\\AshokC\\ML\\irisDataset.arff"));
    Instances ins = arffloader.getDataSet();
    ins.setClassIndex(ins.numAttributes() - 1);


    /*
     * IBkClassifier(K-Nearest Neighbor) to build classifier NaiveBayes to build classifier
     */

    // PolyKernel ClassifierObj = new PolyKernel();
    // ClassifierObj.buildKernel(ins);

    DecisionTable ClassifierObj = new DecisionTable();
    // RandomForest ClassifierObj = new RandomForest();
    // RandomTree ClassifierObj = new RandomTree();
    // NaiveBayes ClassifierObj = new NaiveBayes();
    // IBk ClassifierObj = new IBk();
    // J48 ClassifierObj = new J48();
    ClassifierObj.buildClassifier(ins);

    /*
     * Building the model and writing the model file in the local workspace
     */

    String outputPath = "C:\\AshokC\\ML\\Model.arff";
    SerializationHelper.write(new FileOutputStream(outputPath), ClassifierObj);

    /*
     * Evalution results with traindataset
     */

    Evaluation eval = new Evaluation(ins);
    eval.evaluateModel(ClassifierObj, ins);
    /** Print the algorithm summary */
    System.out.println(" ---------------------------------");
    System.out.println("**Classifier Evaluation with Datasets **");
    System.out.println(eval.toSummaryString());
    System.out.println(" ---------------------------------");
    System.out.print(" The expression for the input data as per alogorithm is ");
    System.out.println(ClassifierObj);
    System.out.println(" ---------------------------------");
    System.out.println(eval.toClassDetailsString());
    System.out.println(" ---------------------------------");


    /*
     * Loading .arff file from ArffLoader -Test Data setting Classification Class index Pass path example
     * C:\\AshokChinnaswamy\\irisDatasetTest.arff
     */

    ArffLoader arffloadertest = new ArffLoader();
    arffloadertest.setSource(new File("C:\\AshokC\\ML\\irisDatasetTest.arff"));
    Instances insTest = arffloadertest.getDataSet();
    insTest.setClassIndex(insTest.numAttributes() - 1);

    /*
     * reading the model from the localpath
     */

    Classifier smo = (Classifier) SerializationHelper.read(outputPath);

    /*
     * Sending all the instance with ? to the model for classify the instance and adding the predicted values in the
     * ListPredictValues
     */

    List<String> ListPredictValues = new ArrayList<String>();
    for (int i = 0; i < insTest.numInstances(); i++) {
      Instance inds = insTest.instance(i);
      double predic = smo.classifyInstance(inds);
      String predicString = insTest.classAttribute().value((int) predic);
      ListPredictValues.add(predicString);
      System.out.println((i + 1) + ". " + predicString);

    }


  }

}
