package com.newhighs.rltictactoe;

import org.apache.log4j.Logger;

import java.util.List;
import java.util.Random;

/**
 * Created by mark on 25-10-16.
 */
public abstract class Environment
{
  public static final Logger _log = Logger.getLogger(Environment.class);

  protected Random _random = new Random(1);

  public abstract void new_episode();

  public abstract State getState();

  public abstract Action offPolicyAction(State s_);

  /**
   *
   * @param a_
   * @return an array of length 2 consisting of a reward (double) and next state (State), as a result of applying
   * action a_
   */
  public abstract Object[] apply(State s_, Action a_);

  public Action epsilonGreedyPolicy(QFunction qTable_, State S_, double epsilon_)
  {
    if (_random.nextDouble() < epsilon_)
    {
      // pick a random action;
      List<? extends Action> actions = possibleActions(S_);
      return actions.get(_random.nextInt( actions.size()) );
    } else
    {
      // pick the best action from S_, according to the policy:
      List<Action> best = qTable_.argMax(this, S_);

      if (best.size() == 0)
      {
        // pick a random one anyway (could be the QTable hasn't been initialized yet)
        List<? extends Action> actions = possibleActions(S_);
        return actions.get(_random.nextInt( actions.size()) );
      }
      return best.get(_random.nextInt( best.size()) );
    }
  }

  // return the action based upon a greedy policy. If there are multiple actions optimal and APrime is one of them,
  // return APrime
  public Action greedyPolicy(QFunction qTable_, State S_, Action APrime)
  {
    // pick the best action from S_, according to the policy:
    List<Action> best = qTable_.argMax(this, S_);

    if (best.contains(APrime))
    {
      return APrime;
    }
    if (best.size() == 0)
    {
      // pick a random one anyway (could be the QTable hasn't been initialized yet)
      List<? extends Action> actions = possibleActions(S_);
      return actions.get(_random.nextInt( actions.size()) );
    }
    return best.get(_random.nextInt( best.size()) );
  }

  protected abstract List<? extends Action> possibleActions(State s_);

}


