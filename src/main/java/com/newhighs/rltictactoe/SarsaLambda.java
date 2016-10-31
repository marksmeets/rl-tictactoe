package com.newhighs.rltictactoe;

import com.newhighs.rltictactoe.Board.Cell;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

/**
 * Created by mark on 25-10-16.
 *
 * From David Silvers RL lectures
 */
public class SarsaLambda extends AbstractLearner
{
  transient public static final Logger _log = Logger.getLogger(SarsaLambda.class);

  QTable _QTable = new QTable();
  ElligibilityTraces _ET = new ElligibilityTraces();
  double _gamma = 0.9;
  double _alpha = 0.5;
  double _lambda = 0.5;

  public SarsaLambda()
  {
    super (0.9999);
    _QTable = new QTable();

  }

  public void episode(Environment env_)
  {
    _ET.clear();
    env_.new_episode();
    State S = env_.getState();
    Action A = env_.epsilonGreedyPolicy(_QTable, S, 0.0);
    do
    {
      _QTable.createIfNotExist(S,A);

      // take action A, observe reward R, next state S'
      Object[] rewardNextState = env_.apply(S,A);
      double R = (Double)rewardNextState[0];
      State SPrime = (State)rewardNextState[1];

      // choose A' from S' using policy derived from Q (eg, epsilon-greedy)
      Action APrime = env_.epsilonGreedyPolicy(_QTable, SPrime, _epsilon);

      _QTable.createIfNotExist(SPrime,APrime);

      double delta = R + _gamma * _QTable.get(SPrime, APrime) - _QTable.get(S, A);
//      _ET.set1(S,A);
      _ET.inc(S,A);

      // for all s in S, a in A(s): update Q(s,a) := Q(s,a) + alpha*delta*E(s,a)
      _QTable.update(_alpha,delta,_ET);
      // for all s in S, a in A(s): update E(s,a) := gamma * lambda * E(s,a)
      _ET.update(_gamma, _lambda);

      S = SPrime;
      A = APrime;


    } while ( ! S.isTerminal() );
//    System.out.println(S);
//    System.out.println(_QTable);
  }

  public static void main(String[] args)
  {
    PropertyConfigurator.configure(SarsaLambda.class.getClassLoader().getResource("resources/log4j.properties"));
    TicTacToe game = new TicTacToe( Cell.X, new RandomPlayer(Cell.O));
    SarsaLambda sarsaLambda = new SarsaLambda();
    for (int z = 0; z < 10000000; z++)
    {
      game.resetStats();
      for (int i = 0; i < 1000; i++)
      {
        sarsaLambda.episode(game);
      }
      sarsaLambda._epsilon = sarsaLambda._epsilon * sarsaLambda._epsilon;
      _log.info("Episode #" + z + " Won: " + game.won() + "\t Lost: " + game.lost() + "\t Draw: " + game.draws() + "\t illegal moves: " + game.illegals() + "\t epsilon: " + sarsaLambda._epsilon);
//      _log.info(sarsaLambda._QTable);
    }
  }
}
