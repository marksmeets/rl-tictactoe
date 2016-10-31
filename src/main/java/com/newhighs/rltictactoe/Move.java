package com.newhighs.rltictactoe;

import org.apache.log4j.Logger;

/**
 * Created by mark on 25-10-16.
 */
public class Move implements Action
{
  transient public static final Logger _log = Logger.getLogger(Move.class);

  int _cell = -1;

  public Move(int value)
  {
    _cell = value;
  }

  @Override
  public String encode()
  {
    return Integer.toString(_cell);
  }

  @Override
  public int toNd4jArrayIndex()
  {
    return _cell;
  }

  public Move copy()
  {
    return new Move(_cell);
  }
}
