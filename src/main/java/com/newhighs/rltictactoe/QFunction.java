package com.newhighs.rltictactoe;

import java.util.List;

/**
 * Created by mark on 27-10-16.
 *
 * interface for accessing Q(S,A)
 */
public interface QFunction
{
  double get(State s_, Action a_);

  void set(State s_, Action a_, double v_);

  void update(double alpha_, double delta_, EligibilityTraces et_);

  List<Action> argMax(Environment env_, State S_);
}
