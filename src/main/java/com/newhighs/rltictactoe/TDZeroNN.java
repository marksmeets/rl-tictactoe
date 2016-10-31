package com.newhighs.rltictactoe;

import com.newhighs.rltictactoe.Board.Cell;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

/**
 * Created by mark on 25-10-16.
 *
 * From David Silvers RL lectures
 */
public class TDZeroNN extends AbstractLearner
{
  transient public static final Logger _log = Logger.getLogger(TDZeroNN.class);

  final QFunction _QNN;
  double _gamma = 0.9;
  double _alpha = 0.5;

  public TDZeroNN(QFunction QNN_)
  {
    super (0.0);
    _QNN = QNN_;
  }

  public QFunction getQFunction()
  {
    return _QNN;
  }

  public void episode(Environment env_)
  {
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

      if (SPrime.isTerminal())
      {
        _QNN.set(S, A, R );
      } else
      {
        // when using function approximators the TD target is R + gamma * Q(S',A')
        _QNN.set(S, A, R + _gamma * _QNN.get(SPrime, APrime));
//        double oldQValue = _QNN.get(S,A);
//        _QNN.set(S,A, oldQValue + _alpha * ( R + _gamma * _QNN.get(SPrime, APrime) - oldQValue ));
      }

      S = SPrime;
      A = APrime;


    } while ( ! S.isTerminal() );
//    System.out.println(S);
//    System.out.println(_QTable);
  }

  public static void main(String[] args)
  {
    PropertyConfigurator.configure(TDZeroNN.class.getClassLoader().getResource("resources/log4j.properties"));

    TicTacToe game = new TicTacToe( Cell.X, new RandomPlayer(Cell.O));
    AbstractLearner qLambda = new TDZeroNN(new QNNOneHot(Board.DIM));
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
