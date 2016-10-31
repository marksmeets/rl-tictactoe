package com.newhighs.rltictactoe;

import com.newhighs.rltictactoe.Board.Cell;
import org.apache.log4j.Logger;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * Created by mark on 25-10-16.
 */
public class RandomPlayer extends Player
{
  transient public static final Logger _log = Logger.getLogger(RandomPlayer.class);

  Random _random = new Random(1);

  public RandomPlayer(Cell myStone_)
  {
    super(myStone_);
  }

  @Override
  public Move play(Board board_)
  {
    List<Move> moves = new ArrayList<> (board_.possibleMoves());
    // remove all non-empty places
    for (Iterator<Move> iterator = moves.iterator(); iterator.hasNext(); )
    {
      Move move = iterator.next();
      if (!board_.isLegal(move))
      {
        iterator.remove();
      }
    }
    if (moves.size() > 0)
    {
      // take a random, allowed action
      return moves.get(_random.nextInt(moves.size()));
    }
    return new Move(-1);
  }
}
