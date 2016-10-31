package com.newhighs.rltictactoe;

/**
 * Created by mark on 25-10-16.
 */
public interface Action
{
  String encode();

  int toNd4jArrayIndex();
}

