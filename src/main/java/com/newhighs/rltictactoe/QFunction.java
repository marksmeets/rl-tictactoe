package com.newhighs.rltictactoe;

import java.util.List;

/**
 * Created by mark on 27-10-16.
 */
public interface QFunction
{
  double get(State s_, Action a_);

  void update(double alpha_, double delta_, ElligibilityTraces et_);

  List<Action> argMax(Environment env_, State S_);
}
