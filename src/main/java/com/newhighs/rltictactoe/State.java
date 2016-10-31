package com.newhighs.rltictactoe;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by mark on 25-10-16.
 */
public interface State
{
  public static final Logger _log = Logger.getLogger(State.class);

  boolean isTerminal();

  String encode();

  INDArray toNd4jArray();
}
