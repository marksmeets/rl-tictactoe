package com.newhighs.rltictactoe;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Created by mark on 25-10-16.
 */
public class ElligibilityTraces
{
  transient public static final Logger _log = Logger.getLogger(ElligibilityTraces.class);

  // map from State x Action -> Double. State and Action are encoded as Strings
  private Map<String, Map<String, Double>> _eMap = new HashMap<>();

  private List<Pair<State, Action>> _traces = new ArrayList<>();

  public Iterator<Pair<State, Action>> getIterator()
  {
    return _traces.iterator();
  }

  public void set1(State s_, Action a_)
  {
    String s = s_.encode();
    String a = a_.encode();
    Map<String, Double> map = _eMap.get(s);
    if (map == null)
    {
      map = new HashMap<>();
      _eMap.put(s, map);
    }
    map.put(a, 1.0);
  }

  public double get(String s_, String a_)
  {
    Map<String, Double> map = _eMap.get(s_);
    if (map == null)
    {
      return 0;
    }
    Double d = map.get(a_);
    if (d == null)
    {
      return 0;
    }
    return d;
  }

  public double get(State s_, Action a_)
  {
    return get(s_.encode(), a_.encode());
  }

  public void update(double gamma_, double lambda_)
  {
    for (String s : _eMap.keySet())
    {
      Map<String, Double> map = _eMap.get(s);
      for (String a : map.keySet())
      {
        map.put(a, gamma_ * lambda_ * map.get(a));
      }
    }
  }

  public void clear()
  {
    _eMap.clear();
    _traces.clear();
  }

  // either use inc or set1 to update the elligibility traces.
  public void inc(State s_, Action a_)
  {
    _traces.add(new ImmutablePair<State, Action>(s_, a_));

    String s = s_.encode();
    String a = a_.encode();
    Map<String, Double> map = _eMap.get(s);
    if (map == null)
    {
      map = new HashMap<>();
      _eMap.put(s, map);
    }
    Double d = map.get(a);
    if (d == null)
    {
      d = 0.0;
    }
    map.put(a, 1.0 + d);
  }
}
