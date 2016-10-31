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

  final QNN _QNN;
  double _gamma = 0.9;
  double _alpha = 0.5;

  public TDZeroNN(QNN QNN_)
  {
    super (0.0);
    _QNN = QNN_;
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

//      double oldQValue = _QNN.get(S,A);
//      _QNN.set(S,A, oldQValue + _alpha * ( R + _gamma * _QNN.get(SPrime, APrime) - oldQValue ));
//      if (R < 0)
//      {
//        _log.info("Reward " + R + " Q(S,A) before: " + oldQValue + " new: " + _QNN.get(S,A));
//      }

//      _log.info("REeward " + R + " Sprime terminal? " + SPrime.isTerminal() + " QNN(S',A') = " + _QNN.get(SPrime,APrime));
      if (SPrime.isTerminal())
      {
        _QNN.set(S, A, R );
      } else
      {
        _QNN.set(S, A, R + _gamma * _QNN.get(SPrime, APrime));
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
    AbstractLearner qLambda = new TDZeroNN(new QNNOneHot(Board.DIM, 500));
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
