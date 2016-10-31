package com.newhighs.rltictactoe;

import org.apache.log4j.Logger;

/**
 * Created by mark on 27-10-16.
 */
public abstract class AbstractLearner implements Learner
{
  transient public static final Logger _log = Logger.getLogger(AbstractLearner.class);

  double _epsilon = 0.999;

  public AbstractLearner( double epsilon_)
  {
    _epsilon = epsilon_;
  }

  @Override
  public void decrEpsilon()
  {
    _epsilon = _epsilon * _epsilon;
  }

  @Override
  public double getEpsilon()
  {
    return _epsilon;
  }

  public void emptyBoardStats(Environment env_)
  {
    _log.info("Values for Q(S,A) with S is empty board: ");
    env_.new_episode();
    State S = env_.getState();
    for (Action a: env_.possibleActions(S))
    {
      _log.info("Action " + a.encode() + " : " + getQFunction().get(S,a));
    }
  }

}
