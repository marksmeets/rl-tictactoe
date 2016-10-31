package com.newhighs.rltictactoe;

import com.newhighs.rltictactoe.Board.Cell;

/**
 * Created by mark on 25-10-16.
 */
public abstract class Player
{

  protected Cell _myStone;

  public Player(Cell myStone_)
  {
    _myStone = myStone_;
  }

  public abstract Move play(Board board_);
}
