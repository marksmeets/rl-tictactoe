package com.newhighs.rltictactoe;

import com.newhighs.rltictactoe.Board.Cell;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

/**
 * Created by mark on 25-10-16.
 *
 * From David Silvers RL lectures
 */
public class QLambdaNN extends AbstractLearner
{
  transient public static final Logger _log = Logger.getLogger(QLambdaNN.class);

  final QFunction _QNN;
  EligibilityTraces _ET = new EligibilityTraces();
  double _gamma = 0.9;
  double _alpha = 0.5;
  double _lambda = 0.5;

  public QLambdaNN(QFunction QNN_)
  {
    super (0);
    _QNN = QNN_;

  }

  public QFunction getQFunction()
  {
    return _QNN;
  }

  public void episode(Environment env_)
  {
    _ET.clear();
    env_.new_episode();
    State S = env_.getState();
    Action A = env_.epsilonGreedyPolicy(_QNN, S, _epsilon);
    do
    {

      // take action A, observe reward R, next state S'
      Object[] rewardNextState = env_.apply(S,A);
      double R = (Double)rewardNextState[0];
      State SPrime = (State)rewardNextState[1];

      // choose A' from S' using policy derived from Q (eg, epsilon-greedy)
      Action APrime = env_.epsilonGreedyPolicy(_QNN, SPrime, _epsilon);
      // if APrime was the result of an exploratory move, it will be different from AStar
      Action AStar = env_.greedyPolicy(_QNN, SPrime, APrime);

      double delta = R + _gamma * _QNN.get(SPrime, AStar) - _QNN.get(S, A);
      _ET.inc(S,A);

      // for all s in S, a in A(s): update Q(s,a) := Q(s,a) + alpha*delta*E(s,a)
      _QNN.update(_alpha,delta,_ET);
      // if APrime == AStar:
      // -> for all s in S, a in A(s): update E(s,a) := gamma * lambda * E(s,a)
      // otherwise, E(s,a) = 0 for all s in S, a in A(s)
      if (APrime == AStar)
      {
        _ET.update(_gamma, _lambda);
      } else
      {
        _ET.clear();
      }

      S = SPrime;
      A = APrime;


    } while ( ! S.isTerminal() );
  }

  public static void main(String[] args)
  {
    PropertyConfigurator.configure(QLambdaNN.class.getClassLoader().getResource("resources/log4j.properties"));
    TicTacToe game = new TicTacToe( Cell.X, new RandomPlayer(Cell.O));
    AbstractLearner qLambda = new QLambdaNN(new QNNOneHot(Board.DIM, 500));
    for (int z = 0; z < 1000000; z++)
    {
      game.resetStats();
      for (int i = 0; i < 10; i++)
      {
        qLambda.episode(game);
      }
      qLambda.decrEpsilon();
      _log.info("Episode #" + z + " Won: " + game.won() + "\t Lost: " + game.lost() + "\t Draw: " + game.draws() + "\t illegal moves: " + game.illegals() + "\t epsilon: " + qLambda.getEpsilon());
//      _log.info(sarsaLambda._QTable);
    }
  }
}
