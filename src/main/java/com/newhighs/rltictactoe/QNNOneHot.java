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
import org.nd4j.linalg.lossfunctions.LossFunctions;

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
public class QNNOneHot implements QNN
{
  transient public static final Logger _log = Logger.getLogger(QNNOneHot.class);


  public static final int seed = 12345;
  transient private Random _random;

  //Number of iterations per minibatch
  public static final int _iterations = 1;
  public static int _batchSize = 100;
  //Network learning rate
  public static final double _learningRate = 0.01;
  public int _numInputs;
  public int _numOutputs;

  transient MultiLayerNetwork _net;

  List<DataSet> _replayMemory;
  List<DataSet> _trainingData;

  private int _replayMemorySize;
  protected boolean _keepTrainingData = false;

  public QNNOneHot(int dimension_, int replayMemorySize_)
  {
    _replayMemorySize = replayMemorySize_;

    // the network gets as input a board (size: dim*dim) plus a move (one-hot encoding of the cells in the board)
    _numInputs = dimension_ * dimension_;
    _numOutputs = dimension_ * dimension_;
    final MultiLayerConfiguration conf = getDeepDenseLayerNetworkConfiguration();
    //Create the network
    _net = new MultiLayerNetwork(conf);
    _net.init();

    init();
  }

  public void init()
  {
    _random = new Random(seed);
    _net.setListeners(new ScoreIterationListener(1000));
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
    double qValue = Double.NEGATIVE_INFINITY;
    INDArray input = encode(S_);
    INDArray output = _net.output(input, false);
    List<Action> best = new ArrayList<>();
    for (Action a : env_.possibleActions(S_))
    {
      if ( qValue < output.getDouble(a.toNd4jArrayIndex()))
      {
        best.clear();
        best.add(a);
        qValue = output.getDouble(a.toNd4jArrayIndex());
      } else if ( Math.abs(qValue - output.getDouble(a.toNd4jArrayIndex())) < 0.0000001)
      {
        best.add(a);
      }
    }
    return best;
  }

  public QNNOneHot load(String filename_) throws IOException
  {
    _net = ModelSerializer.restoreMultiLayerNetwork(filename_);
    return this;
  }

  public void save(String filename_) throws IOException
  {
    ModelSerializer.writeModel(_net, filename_, true);
  }

  public List<DataSet> getTrainingData()
  {
    List<DataSet> list = _trainingData;
    _trainingData = new ArrayList<>();
    return list;
  }


  private MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration()
  {
    final int numHiddenNodes1 = _numInputs * 1;
    final int numHiddenNodes2 = _numInputs * 1;
    final int numHiddenNodes3 = _numInputs;
    return new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(_iterations)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(_learningRate)
        .weightInit(WeightInit.XAVIER)
//        .regularization(true).l2(1e-3).dropOut(0.5)
        .updater(Updater.NESTEROVS).momentum(0.9)
        .biasInit(0)
        .list()
        .layer(0, new DenseLayer.Builder().nIn(_numInputs).nOut(numHiddenNodes1)  // hidden layer #1
            .activation("tanh")
            .build())
        .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes1).nOut(numHiddenNodes2) // hidden layer #2
            .activation("tanh")
            .build())
//        .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes2).nOut(numHiddenNodes3) // hidden layer #3
//            .activation("relu")
//            .build())
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // output layer
            .activation("identity")
            .nIn(numHiddenNodes2).nOut(_numOutputs).build())
        .backprop(true)
        .pretrain(false).backprop(true).build();
  }

  int _counter = 0;
  // set is like fitting the network
  // input is from_ (board)
  // output consists of qValue for all moves. Here only the qValue for move_ is updated, previous values otherwise
  public void set(State state_, Action action_, double qValue_)
  {
    INDArray input = encode(state_);
    INDArray output = _net.output(input, false);
//    _log.info("Before: " + state_.encode() + " , " + action_.encode() + " -> " + qValue_ + "  ---  " + input + " -> " + output);
    output.putScalar(action_.toNd4jArrayIndex(), qValue_);
//    _log.info("target: " + output);
    DataSet newData = new DataSet(input, output);

    // feed only a single point of data (don't use replay)
    List<DataSet> single = new LinkedList<>();
    single.add(newData);
    DataSetIterator iterator = new ListDataSetIterator(single, 1);
    _net.fit(iterator);
    output = _net.output(input, false);
//    _log.info("After : " + state_.encode() + " , " + action_.encode() + " -> " + qValue_ + "  ---  " + input + " -> " + output);
    if (true)
      return;

    if (_keepTrainingData)
    {
      _trainingData.add(newData);
    }
    _replayMemory.add(newData);
    if (_replayMemory.size() > _replayMemorySize)
    {
      _replayMemory.remove(0);
    }
    if (_replayMemory.size() < _batchSize)
    {
      return;
    }
    _counter++;
    int threshold = _batchSize/2;
    if (_counter < threshold)
    {
      return;
    }
    _counter = 0;
    // add the last unseen trainingexamples to the minibatch
    List<DataSet> unseen = _replayMemory.subList(_replayMemory.size() - threshold, _replayMemory.size()-1);
    // get a minibatch from the replaymemory
    List<DataSet> replayCopy = new LinkedList<>(_replayMemory);
    Collections.shuffle(replayCopy);
    Set<DataSet> randomBatch = new HashSet<>(replayCopy.subList(0, Math.min(replayCopy.size(), _batchSize - threshold)));
    randomBatch.addAll(unseen);

    final List<DataSet> list = new ArrayList<>(randomBatch);
    update(list);
  }

  public void update(List<DataSet> trainingData_)
  {
    Collections.shuffle(trainingData_, _random);
    DataSetIterator iterator = new ListDataSetIterator(trainingData_, _batchSize);

//    long start = System.currentTimeMillis();
    _net.fit(iterator);
//    _log.info("It took " + (System.currentTimeMillis() - start) + " ms to fit()");
  }

//  public QNNOneHot copy()
//  {
//    // we can't do deepcopy on the neural net; that's why it's transient
//    QNNOneHot copy = JsonConverter.deepCopy(this, QNNOneHot.class);
//    copy._net = _net.clone();
//    copy._net.init();
//    copy._random = new Random(seed);
//    return copy;
//  }

  public void keepTrainingData(boolean b_)
  {
    _keepTrainingData = b_;
  }

  private INDArray encode(State s_)
  {
    INDArray array = s_.toNd4jArray();
    return array;
  }

  // here we see the benefit of one-hot encoding the outputs: we only need one forward pass for the MLP
  // further optimization: to save on one MLP pass we compute the qValue and bestMove here in one go and return
  // them as an array with 2 cells: bestMove and qValue
  private Object[] argMaxQ(State from_)
  {
    double qValue = Double.NEGATIVE_INFINITY;
    INDArray input = encode(from_);
    INDArray output = _net.output(input, false);
    List<Integer> bestMoves = new ArrayList<>();
    for (int z = 0; z < output.length(); z++)
    {
      double q = output.getDouble(z);
      if (q > qValue)
      {
        bestMoves.clear();
        bestMoves.add(z);
        qValue = q;
      } else if (Math.abs(q - qValue) < 0.0000001)
      {
        bestMoves.add(z);
      }
    }
    int bestMove = bestMoves.get(_random.nextInt(bestMoves.size()));
    return new Object[]{bestMove, qValue};
  }

  public double getMaxQValue(Board from_)
  {
    return (double) argMaxQ(from_)[1];
  }

  public int getQMaximizer(Board from_)
  {
    return (int) argMaxQ(from_)[0];
  }

  public double getQValue(Board from_, int move_)
  {
    INDArray array = encode(from_);
    return _net.output(array, false).getDouble(move_);
  }

  @Override
  public double get(State s_, Action a_)
  {
    INDArray array = s_.toNd4jArray();
    return _net.output(array, false).getDouble(a_.toNd4jArrayIndex());
  }
}


