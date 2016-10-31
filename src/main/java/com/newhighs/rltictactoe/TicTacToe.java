package com.newhighs.rltictactoe;

import com.newhighs.rltictactoe.Board.Cell;
import org.apache.log4j.Logger;

import java.util.List;

/**
 * Created by mark on 25-10-16.
 */
public class TicTacToe extends Environment
{
  transient public static final Logger _log = Logger.getLogger(TicTacToe.class);

  public static final int DIM = 3;
  private final Player _otherPlayer;
  private final Cell _myStone;


  Board _board = new Board();

  int _won = 0;
  int _lost = 0;
  int _draws = 0;
  int _illegals = 0;

  public TicTacToe(Cell myStone_ , Player otherPlayer )
  {
    _otherPlayer = otherPlayer;
    _myStone = myStone_;
  }

  public int lost()
  {
    return _lost;
  }

  public int won()
  {
    return _won;
  }

  public int draws()
  {
    return _draws;
  }

  public int illegals()
  {
    return _illegals;
  }

  public void resetStats()
  {
    _won = 0;
    _lost = 0;
    _draws = 0;
    _illegals = 0;
  }

  @Override
  public void new_episode()
  {
    _board.clear();
  }

  @Override
  public State getState()
  {
    // DO NOT return a reference here!
    return _board.copy();
  }

  @Override
  public Action offPolicyAction(State s_)
  {
    return possibleActions(s_).get(_random.nextInt( possibleActions(s_).size() ));
  }


  @Override
  public Object[] apply(State s_, Action a_)
  {
    double reward = 0.0;
    Move move = ((Move)a_).copy();
    Board board = ((Board)s_).copy();
    if (!board.isLegal( move))
    {
      _illegals ++;
      reward = -1.0; // illegal move!
      return new Object[] { reward, board};
    }
    // update the board by us
    board.apply(move, _myStone);
    if (board.hasWinner() == _myStone)
    {
      reward = 1.0;
      _won++;
    } else
    {
      Move otherMove = _otherPlayer.play(board);
      if (board.isLegal(otherMove))
      {
        board.apply(otherMove, _myStone.otherStone());
        if (board.hasWinner() == _myStone.otherStone())
        {
          reward = -1.0;
          _lost++;
        }
      }
    }
    if (board.isTerminal() && (Math.abs(reward) < 0.01))
    {
      _draws++;
    }
    return new Object[] { reward, board};
  }

  @Override
  protected List<? extends Action> possibleActions(State s_)
  {
    List<Move> list = _board.possibleMoves();
    // filter any illegal moves (not necessary, algorithm should learn what is illegal!)

//    for (Iterator<Move> iterator = list.iterator(); iterator.hasNext(); )
//    {
//      Move move = iterator.next();
//      if (!_board.isLegal(move))
//      {
//        iterator.remove();
//      }
//    }

    return list;
  }

}
