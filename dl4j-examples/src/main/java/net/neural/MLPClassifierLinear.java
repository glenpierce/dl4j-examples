package net.neural;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class MLPClassifierLinear {
    public static void main(String[] args) {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int numberOfEpochs = 3;
        int numberOfInputs = 2;
        int numberOfOutputs = 2;
        int numberOfHiddenNodes = 20;

        //load training data
        RecordReader recordReader = new CSVRecordReader();
        File testFile = new File("C:\\Users\\glenp\\Documents\\Code\\dl4j-examples\\dl4j-examples\\src\\main\\resources\\classification\\linear_data_train.csv");
        try {
        recordReader.initialize(new FileSplit(testFile));
        } catch (Exception excecption) {
            excecption.printStackTrace();
        }
        DataSetIterator trainingDataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 0, 2);

        //load evaluation data
        RecordReader recordReaderEvaluation = new CSVRecordReader();
        File evaluationFile = new File("C:\\Users\\glenp\\Documents\\Code\\dl4j-examples\\dl4j-examples\\src\\main\\resources\\classification\\linear_data_eval.csv");
        try {
            recordReaderEvaluation.initialize(new FileSplit(evaluationFile));
        } catch (Exception excecption) {
            excecption.printStackTrace();
        }
        DataSetIterator evaluationDataSetIterator = new RecordReaderDataSetIterator(recordReaderEvaluation, batchSize, 0, 2);

        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .maxNumLineSearchIterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(learningRate,0.5))
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(numberOfInputs)
                .nOut(numberOfHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .build()
            )
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(numberOfHiddenNodes)
                .nOut(numberOfOutputs)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SOFTMAX)
                .build()
            )
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
        multiLayerNetwork.init();
        multiLayerNetwork.setListeners(new ScoreIterationListener(10));

        for(int n = 0; n < numberOfEpochs; n++){
            multiLayerNetwork.fit(trainingDataSetIterator);
        }

        Evaluation evaluation = new Evaluation(numberOfOutputs);
        while(evaluationDataSetIterator.hasNext()){
            DataSet dataSet = evaluationDataSetIterator.next();
            INDArray features = dataSet.getFeatureMatrix();
            INDArray labels = dataSet.getLabels();
            INDArray predicted = multiLayerNetwork.output(features, false);
            evaluation.eval(labels, predicted);
        }

        System.out.println(evaluation.stats());
    }
}
