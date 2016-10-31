package com.newhighs.rltictactoe;

/**
 * Created by mark on 27-10-16.
 */
public interface QNN extends QFunction
{
  void set(State s_, Action a_, double v_);
}
