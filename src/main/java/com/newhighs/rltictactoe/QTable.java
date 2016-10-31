package com.newhighs.rltictactoe;

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
public class QTable implements QFunction
{
  transient public static final Logger _log = Logger.getLogger(QTable.class);

  // map from State x Action -> Double. State and Action are encoded as Strings
  private Map<String, Map<String, Double>> _qMap = new HashMap<>();

  public double get(State s_, Action a_)
  {
    Map<String, Double> map = _qMap.get(s_.encode());
    if (map == null)
    {
      return 0;
    }
    Double d = map.get(a_.encode());
    if (d == null)
    {
      return 0;
    }
    return d;
  }

  public void update(double alpha_, double delta_, EligibilityTraces et_)
  {
    for (Iterator<Pair<State,Action>> i = et_.getIterator(); i.hasNext(); )
    {
      Pair<State,Action> stateActionPair = i.next();
      String s = stateActionPair.getLeft().encode();
      String a = stateActionPair.getRight().encode();
      Map<String, Double> map = _qMap.get(s);
      map.put(a, map.get(a) + alpha_ * delta_ * et_.get(s,a));
    }
//    for (String s: _qMap.keySet())
//    {
//      Map<String, Double> map = _qMap.get(s);
//      for (String a: map.keySet())
//      {
//        map.put(a, map.get(a) + alpha_ * delta_ * et_.get(s,a));
//      }
//    }
  }

  @Override
  public List<Action> argMax(Environment env_, State S_)
  {
    List<Action> best = new ArrayList<>();
    double bestScore = Double.NEGATIVE_INFINITY;
    for (Action a: env_.possibleActions(S_))
    {
      double score = get(S_, a);
      if (score > bestScore)
      {
        best.clear();
        best.add(a);
        bestScore = score;
      } else if (Math.abs(score-bestScore) < Double.MIN_VALUE)
      {
        best.add(a);
      }
    }
    return best;
  }

  public String toString()
  {
    String r = "";
    for (String s: _qMap.keySet())
    {
      r += s + " : ";
      Map<String, Double> map = _qMap.get(s);
      for (String a: map.keySet())
      {
        r += a+"(" + map.get(a) + ") ";
      }
      r += "\n";
    }
    return r;
  }

  public void createIfNotExist(State s_, Action a_)
  {
    String s = s_.encode();
    String a = a_.encode();
    Map<String, Double> map = _qMap.get(s);
    if (map == null)
    {
      map = new HashMap<>();
      _qMap.put(s, map);
    }
    Double d = map.get(a);
    if (d == null)
    {
      map.put(a,0.0);
    }
  }

  public void set(State s_, Action a_, double v_)
  {
//    _log.info(s_.encode() + " , " + a_.encode() + " -> " + v_);
    // precondition: Q(s,a) exists
    _qMap.get(s_.encode()).put(a_.encode(), v_);
  }
}
