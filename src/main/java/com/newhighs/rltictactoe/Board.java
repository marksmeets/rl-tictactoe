package com.newhighs.rltictactoe;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by mark on 25-10-16.
 */
public class Board implements State
{
  transient public static final Logger _log = Logger.getLogger(Board.class);

  public static final int DIM = 3;
  private Cell[] _cells = new Cell[DIM * DIM];

  public void clear()
  {
    for (int i = 0; i < DIM*DIM; i++)
    {
      _cells[i] = Cell.EMPTY;
    }
  }

  public boolean isLegal(Move move_)
  {
    if (move_._cell < 0)
    {
      return false;
    }
    return _cells[move_._cell ] == Cell.EMPTY;
  }

  public void apply(Move move_, Cell myStone_)
  {
    _cells[move_._cell] = myStone_;
  }

  // possible actions is always every cell (we let the learner figure out itself if it did an illegal move)
  private List<Move> _possiblemoves = null;

  public List<Move> possibleMoves()
  {
    if (_possiblemoves == null)
    {
      _possiblemoves = new ArrayList<>();
      for (int i = 0; i < Board.DIM * Board.DIM; i++)
      {
        Move move = new Move(i);
        _possiblemoves.add(move);
      }
    }
    return _possiblemoves;
  }

  @Override
  public String toString()
  {
    String s = "";
    String indent = "   ";
    for (int y = 0; y < DIM; y++)
    {
      s += indent;
      for (int x = 0; x < DIM; x++)
      {
        s += ("+----");
      }
      s += ("+\n");

      s += (indent);
      for (int x = 0; x < DIM; x++)
      {
        s += ("| " + _cells[y * DIM + x].toString(y * DIM + x) + " ");
      }
      s += ("|\n");
    }

    s += (indent);
    for (int x = 0; x < DIM; x++)
    {
      s += ("+----");
    }
    s += ("+\n");
    return s;
  }

  public Board copy()
  {
    Board board = new Board();
    for (int i = 0; i < DIM*DIM; i++)
    {
      board._cells[i] = _cells[i];
    }
    return board;
  }

  public enum Cell
  {
    EMPTY, X, O;

    public String toString(int i)
    {
      if (this == EMPTY)
      {
        return String.format("%02d", i);
      }
      if (this == X)
      {
        return "x ";
      }
      if (this == O)
      {
        return "o ";
      }
      return null;
    }

    public Cell otherStone()
    {
      if (this == X)
      {
        return O;
      }
      if (this == O)
      {
        return X;
      }
      return Cell.EMPTY; // should not happen
    }

    @Override
    public String toString()
    {
      if (this == EMPTY)
      {
        return " ";
      }
      if (this == X)
      {
        return "x";
      }
      if (this == O)
      {
        return "o";
      }
      return null;
    }
  }

  @Override
  public boolean isTerminal()
  {
    if (hasWinner() != Cell.EMPTY)
    {
      return true;
    }
    for (int i = 0; i < DIM *DIM; i++)
    {
      if (_cells[i] == Cell.EMPTY)
      {
        return false;
      }
    }
    return true;
  }

  @Override
  public String encode()
  {
    String s = "";
    for (int i = 0; i < DIM*DIM; i++)
    {
      s += _cells[i].toString();
    }
    return s;
  }

  @Override
  public INDArray toNd4jArray()
  {
    INDArray array = Nd4j.create(DIM*DIM);
    for (int i = 0; i < DIM*DIM; i++)
    {
      if (_cells[i] == Cell.X)
      {
        array.putScalar(i, 1);
      } else if (_cells[i] == Cell.O)
      {
        array.putScalar(i, -1);
      } else if (_cells[i] == Cell.EMPTY)
      {
        array.putScalar(i, 0);
      }
    }
    return array;
  }

  public Cell hasWinner()
  {
    for (int y = 0; y < DIM; y++)
    {
      boolean allSame = true;
      for (int x = 0; x < DIM; x++)
      {
        if (_cells[y * DIM + x] == Cell.EMPTY)
        {
          allSame = false;
          break;
        }
        if (_cells[y * DIM + x] != _cells[y * DIM + 0])
        {
          allSame = false;
          break;
        }
      }
      if (allSame)
      {
//        System.out.println("row " +y);
        return _cells[y * DIM + 0];
      }
    }

    for (int x = 0; x < DIM; x++)
    {
      boolean allSame = true;
      for (int y = 0; y < DIM; y++)
      {
        if (_cells[y * DIM + x] == Cell.EMPTY)
        {
          allSame = false;
          break;
        }
        if (_cells[y * DIM + x] != _cells[0 * DIM + x])
        {
          allSame = false;
          break;
        }
      }
      if (allSame)
      {
//        System.out.println("column " +x);
        return _cells[0 * DIM + x];
      }
    }

    // diagonal: \
    boolean allSame = true;
    for (int x = 0; x < DIM; x++)
    {
      int y = x;
      if (_cells[y * DIM + x] == Cell.EMPTY)
      {
        allSame = false;
        break;
      }
      if (_cells[y * DIM + x] != _cells[0 * DIM])
      {
        allSame = false;
        break;
      }
    }
    if (allSame)
    {
//      System.out.println("diagonal \\");
      return _cells[0 * DIM + 0];
    }

    // diagonal: /
    allSame = true;
    for (int y = 0; y < DIM; y++)
    {
      int x = DIM-y-1;
      if (_cells[y * DIM + x] == Cell.EMPTY)
      {
        allSame = false;
        break;
      }
      if (_cells[y * DIM + x] != _cells[ DIM -1])
      {
        allSame = false;
        break;
      }
    }
    if (allSame)
    {
//      System.out.println("diagonal /");
      return _cells[0 * DIM + DIM-1];
    }

    return Cell.EMPTY;
  }

}
