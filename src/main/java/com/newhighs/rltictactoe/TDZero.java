package com.newhighs.rltictactoe;

import com.newhighs.misc.TicTacToe2.Board.Cell;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

/**
 * Created by mark on 25-10-16.
 *
 * From David Silvers RL lectures
 */
public class TDZero extends AbstractLearner
{
  transient public static final Logger _log = Logger.getLogger(TDZero.class);

  QTable _QTable = new QTable();
  double _gamma = 0.9;
  double _alpha = 0.5;

  public TDZero()
  {
    super (0.0);
    _QTable = new QTable();

  }

  public void episode(Environment env_)
  {
    env_.new_episode();
    State S = env_.getState();
    Action A = env_.epsilonGreedyPolicy(_QTable, S, _epsilon);
    do
    {
      _QTable.createIfNotExist(S,A);

      // take action A, observe reward R, next state S'
      Object[] rewardNextState = env_.apply(S,A);
      double R = (Double)rewardNextState[0];
      State SPrime = (State)rewardNextState[1];

      // choose A' from S' using policy derived from Q (eg, epsilon-greedy)
      Action APrime = env_.epsilonGreedyPolicy(_QTable, SPrime, _epsilon);

      double oldQValue = _QTable.get(S,A);
//      _QTable.set(S,A, oldQValue + _alpha * ( R + _gamma * _QTable.get(SPrime, APrime) - oldQValue ));
      _QTable.set(S,A, oldQValue + _alpha * ( R + _gamma * _QTable.get(SPrime, APrime) - oldQValue ));

      S = SPrime;
      A = APrime;


    } while ( ! S.isTerminal() );
//    System.out.println(S);
//    System.out.println(_QTable);
  }

  public static void main(String[] args)
  {
    PropertyConfigurator.configure(TDZero.class.getClassLoader().getResource("resources/log4j.properties"));
    TicTacToe game = new TicTacToe( Cell.X, new RandomPlayer(Cell.O));
    AbstractLearner qLambda = new TDZero();
    for (int z = 0; z < 1000000; z++)
    {
      game.resetStats();
      for (int i = 0; i < 1000; i++)
      {
        qLambda.episode(game);
      }
      qLambda.decrEpsilon();
      _log.info("Episode #" + z + " Won: " + game.won() + "\t Lost: " + game.lost() + "\t Draw: " + game.draws() + "\t illegal moves: " + game.illegals() + "\t epsilon: " + qLambda.getEpsilon());
//      _log.info(sarsaLambda._QTable);
    }
  }
}
