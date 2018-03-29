/* -*- Mode: java; c-basic-offset: 2; -*- */
package fr.imag.clips.papillon.business.pivax;

import java.sql.*;
import java.io.*;


/**
 *
 */
public class DBConnection {

  /**
   *
   */
  public DBConnection(String host,
                      int port,
                      String database,
                      String user,
                      String password)
    throws SQLException {

    try {
      out = new PrintStream(java.lang.System.out, true, "UTF-8");
    } catch (Exception ex) {
      ex.printStackTrace();
      java.lang.System.exit(-1);
    }

    try {
     Class.forName("org.postgresql.Driver").newInstance();	 
    } catch (Exception ex) {
      ex.printStackTrace();
      java.lang.System.exit(-1);
    }

    this.databaseName = database;
    connectURL = "jdbc:postgresql://" + host + ":" + port + "/" + database
      + "?user=" + user + "&password=" + password + "&autoReconnect=true"
      + "&characterEncoding=utf8";
    connect();
  }
    

  /**
   *
   */
  private void connect()
    throws SQLException {
    db = DriverManager.getConnection(connectURL);
  }

  /**
   *
   */
  public ResultSet executeQuery(String query)
    throws SQLException {
    // out.println("[" + query + "]");
    ResultSet res = null;
    Statement stmt = db.createStatement();
    try {
      res = stmt.executeQuery(query);
    } catch (Exception ex) {
      try {
        connect();
        stmt = db.createStatement();
        res = stmt.executeQuery(query);
      } catch (Exception ex2) {
        throw new SQLException(ex.getMessage());
      }
    }
    return res;
  }

  /**
   *
   */
  public void executeUpdate(String query)
    throws SQLException {
    out.println("[" + query + "]");
    Statement stmt = db.createStatement();
    try {
      stmt.executeUpdate(query);
    } catch (Exception ex) {
      try {
        connect();
        stmt = db.createStatement();
        stmt.executeUpdate(query);
      } catch (Exception ex2) {
        throw new SQLException(ex.getMessage());
      }
    }
  }

  /**
   *
   */
  public int getNextId(String table)
    throws SQLException {
    ResultSet res = executeQuery("SELECT max(id) + 1 FROM " + table);

    int nextId = 0;
    if (res.next()) {
      nextId = res.getInt(1);
    }

    if (nextId == 0) {
      nextId = 1;
    }

    return nextId;
  }

  /**
   *
   */
  public int getId(String table,
                   String where)
    throws SQLException {
    ResultSet res = executeQuery("SELECT id " + "FROM " + table + " WHERE "
      + where);

    if (res.next()) {
      return res.getInt(1);
    }

    return -1;
  }

  /**
   *
   */
  public static String quote(String s) {
    StringBuffer out = new StringBuffer();
    out.append("_utf8'");

    if (s != null) {
      int lastIndex;
      int index = -1;
      do {
        lastIndex = index + 1;
        index = s.indexOf('\'', lastIndex);
        int tmpIndex = s.indexOf('\\', lastIndex);
        if (index == -1 || (tmpIndex != -1 && tmpIndex < index)) {
          index = tmpIndex;
        }

        if (index != -1) {
          out.append(s.substring(lastIndex, index));
          out.append('\\');
          out.append(s.charAt(index));
        }

      } while (index != -1);

      out.append(s.substring(lastIndex));
    }

    out.append('\'');

    return out.toString();
  }

  /**
   * TODO: why doesn't res.getString() work? What do we need to set and where?
   */
  public static String getString(ResultSet res,
                                 String id)
    throws SQLException {
    try {
      return new String((byte[]) res.getObject(id), "UTF-8");
    } catch (SQLException ex) {
      throw ex;
    } catch (ClassCastException ex) {
      try {
        return res.getString(id);
      } catch (SQLException ex2) {
        throw ex2;
      }
    } catch (Exception ex) {
      ex.printStackTrace();
    }
    return null;
  }

  /**
   * TODO: why doesn't res.getString() work?? What do we need to set and where?
   */
  public static String getString(ResultSet res,
                                 int index)
    throws SQLException {
    try {
      return new String((byte[]) res.getObject(index), "UTF-8");
    } catch (SQLException ex) {
      throw ex;
    } catch (ClassCastException ex) {
      try {
        return res.getString(index);
      } catch (SQLException ex2) {
        throw ex2;
      }
    } catch (Exception ex) {
    }
    return res.getString(index);
  }

  private Connection db;

  private String connectURL;

  private String databaseName;

  private PrintStream out;
}
