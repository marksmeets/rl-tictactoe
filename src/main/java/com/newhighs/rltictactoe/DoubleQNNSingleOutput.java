package com.newhighs.rltictactoe;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import tutil.json.JsonConverter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Created by mark on 27-10-16.
 * <p>
 * implements a Q Matrix with a neural network as an approximator
 * Same as QNNMatrix, but now the output is not in the range of [0,DIM*DIM] but instead it has DIM*DIM outputs
 * Each output gives the score with which a cell should be chosen, and we can use one hot encoding to select the
 * optimal move
 */
public class DoubleQNNSingleOutput implements QNN
{
  transient public static final Logger _log = Logger.getLogger(DoubleQNNSingleOutput.class);


  public static final int seed = 12345;
  transient private Random _random;

  //Number of iterations per minibatch
  public static final int _iterations = 1;
  public static int _batchSize = 10;
  //Network learning rate
  public static final double _learningRate = 0.01;
  public int _numInputs;
  public int _numOutputs;

  final int _numFitsSwap = 1000;
  int _fits = 0;
  transient MultiLayerNetwork _net1; // we read from net1, and update net2. After _numFitsSwap we swap them
  transient MultiLayerNetwork _net2;

  List<DataSet> _replayMemory;
  List<DataSet> _trainingData;

  private int _replayMemorySize;
  protected boolean _keepTrainingData = false;

  public DoubleQNNSingleOutput(int dimension_, int replayMemorySize_)
  {
    _replayMemorySize = replayMemorySize_;

    // the network gets as input a board (size: dim*dim) plus a move (one-hot encoding of the cells in the board)
    _numInputs = dimension_ * dimension_ * 2;
    _numOutputs = 1;
    final MultiLayerConfiguration conf = getDeepDenseLayerNetworkConfiguration();
    //Create the network
    _net1 = new MultiLayerNetwork(conf);
    _net1.init();
    _net2 = new MultiLayerNetwork(conf);
    _net1.init();

    init();
  }

  public void init()
  {
    _random = new Random(seed);
    _net2.setListeners(new ScoreIterationListener(1000));
//    _net.setListeners(new HistogramIterationListener(1));
    _replayMemory = new LinkedList<>();

    // make sure to periodically truncate the trainingData list; otherwise you'll get out-of-memory errors
    // AsyncQLearningActor needs this
    _trainingData = new ArrayList<>();
    _batchSize = Math.min(_batchSize, _replayMemorySize);
  }

  public void update(double alpha_, double delta_, ElligibilityTraces et_)
  {
    for (Iterator<Pair<State,Action>> i = et_.getIterator(); i.hasNext(); )
    {
      Pair<State,Action> stateActionPair = i.next();
      State s = stateActionPair.getLeft();
      Action a = stateActionPair.getRight();
      double qValue = get(s,a) + alpha_ * delta_ * et_.get(s,a);
      set(s,a, qValue);
    }
  }

  @Override
  public List<Action> argMax(Environment env_, State S_)
  {
    List<Action> best = new ArrayList<>();
    double bestScore = Double.NEGATIVE_INFINITY;
    for (Action a: env_.possibleActions(S_))
    {
      double score = get(S_, a);
//      _log.info("In " + S_.encode() + " action " + a.encode() + " gives " + score );
      if (score > bestScore)
      {
        best.clear();
        best.add(a);
        bestScore = score;
      } else if (Math.abs(score-bestScore) < Double.MIN_VALUE)
      {
        best.add(a);
      }
    }
//    _log.info("Best: " + best.get(0).encode());
    return best;
  }

  public DoubleQNNSingleOutput load(String filename_) throws IOException
  {
    _net1 = ModelSerializer.restoreMultiLayerNetwork(filename_);
    return this;
  }

  public void save(String filename_) throws IOException
  {
    ModelSerializer.writeModel(_net1, filename_, true);
  }

  public List<DataSet> getTrainingData()
  {
    List<DataSet> list = _trainingData;
    _trainingData = new ArrayList<>();
    return list;
  }


  private MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration()
  {
    final int numHiddenNodes1 = _numInputs * 4;
    final int numHiddenNodes2 = _numInputs / 2;
    final int numHiddenNodes3 = _numInputs;
    return new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(_iterations)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(_learningRate)
        .weightInit(WeightInit.XAVIER)
//        .regularization(true).l2(1e-3).dropOut(0.5)
        .updater(Updater.NESTEROVS).momentum(0.9)
        .list()
        .layer(0, new DenseLayer.Builder().nIn(_numInputs).nOut(numHiddenNodes1)  // hidden layer #1
            .activation("relu")
            .build())
        .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes1).nOut(numHiddenNodes2) // hidden layer #2
            .activation("relu")
            .build())
//        .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes2).nOut(numHiddenNodes3) // hidden layer #3
//            .activation("relu")
//            .build())
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // output layer
            .activation("identity")
            .nIn(numHiddenNodes2).nOut(_numOutputs).build())

        .pretrain(false).backprop(true).build();
  }

  int _counter = 0;
  // set is like fitting the network
  // input is from_ (board)
  // output consists of qValue for all moves. Here only the qValue for move_ is updated, previous values otherwise
  public void set(State state_, Action action_, double qValue_)
  {
    INDArray input = encode(state_, action_);
    INDArray output = Nd4j.scalar(qValue_);
    DataSet newData = new DataSet(input, output);

//    _log.info(state_.encode() + " , " + action_.encode() + " -> " + qValue_ + "  ---  " + input + " -> " + output);
    _replayMemory.add(newData);
    if (_replayMemory.size() > _replayMemorySize)
    {
      _replayMemory.remove(0);
    }
    if (_replayMemory.size() < _batchSize)
    {
      return;
    }

    // get a minibatch from the replaymemory
    List<DataSet> replayCopy = new LinkedList<>(_replayMemory);
    Collections.shuffle(replayCopy);
    Set<DataSet> randomBatch = new HashSet<>(replayCopy.subList(0, Math.min(replayCopy.size(), _batchSize)));

    final List<DataSet> list = new ArrayList<>(randomBatch);
    update(list);
  }

  public void update(List<DataSet> trainingData_)
  {
    Collections.shuffle(trainingData_, _random);
    DataSetIterator iterator = new ListDataSetIterator(trainingData_, _batchSize);

//    long start = System.currentTimeMillis();
    _fits ++;
    if (_fits > _numFitsSwap)
    {
      _fits = 0;
      MultiLayerNetwork tmp = _net1;
      _net1 = _net2;
      _net2 = tmp;
      _log.info("Swap!");
    }
    _net2.fit(iterator);
//    _log.info("It took " + (System.currentTimeMillis() - start) + " ms to fit()");
  }

  public DoubleQNNSingleOutput copy()
  {
    // we can't do deepcopy on the neural net; that's why it's transient
    DoubleQNNSingleOutput copy = JsonConverter.deepCopy(this, DoubleQNNSingleOutput.class);
    copy._net1 = _net1.clone();
    copy._net1.init();
    copy._random = new Random(seed);
    return copy;
  }

  public void keepTrainingData(boolean b_)
  {
    _keepTrainingData = b_;
  }

  private INDArray encode(State s_, Action a_)
  {
    INDArray array = Nd4j.create(_numInputs);
    INDArray s = s_.toNd4jArray();

    int N = _numInputs / 2;
    for (int z = 0; z < N; z++)
    {
      array.putScalar(z, s.getDouble(z));
    }
    for (int z = 0; z < N; z++)
    {
      if (a_.toNd4jArrayIndex() == z)
      {
        array.putScalar(N + z, 1);
      } else
      {
        array.putScalar(N + z, 0);
      }
    }
    return array;
  }

  @Override
  public double get(State s_, Action a_)
  {
    INDArray array = encode(s_, a_);
    return _net1.output(array, false).getDouble(0);
  }
}


