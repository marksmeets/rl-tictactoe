package com.newhighs.rltictactoe;

/**
 * Created by mark on 27-10-16.
 */
public interface Learner
{
  void episode(Environment game_);

  void decrEpsilon();

  double getEpsilon();

  QFunction getQFunction();
}
