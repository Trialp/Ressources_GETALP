/*-----------------------------------------------------------------------------
 * $Id$
 *-----------------------------------------------------------------------------
 * $Log$
 * 
 *Revision 1.20 2012/03/26 17:20:00 zhang
 *Added a new table link for separating index & link from table index
 *
 * Revision 1.12  2007/04/05 12:55:54  serasset
 * Added a DBLayer Version management with an auto-update of db layer.
 *
 * Revision 1.11  2007/01/15 17:12:18  serasset
 * Several notes added, suppressed the HTMLDOM_CACHE stuff.
 *
 * Revision 1.10  2007/01/05 13:57:26  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.9  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 *-----------------------------------------------------------------------------
*/

package fr.imag.clips.papillon.papillon_data;

import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;

import java.sql.*;

import javax.swing.JOptionPane;

import com.lutris.appserver.server.*;
import com.lutris.appserver.server.sql.*;
import com.lutris.dods.builder.generator.query.DataObjectException;
import com.lutris.dods.builder.generator.query.RDBTable;
import com.lutris.dods.builder.generator.query.QueryBuilder;
import com.lutris.dods.builder.generator.query.RDBColumn;

import org.enhydra.dods.DODS;

/* For debug messages */
//import fr.imag.clips.papillon.business.PapillonLogger;

public class ManageDatabase implements Query {


    protected static final String createTableSql = "CREATE TABLE ";
    protected static final String createIndexSql = "CREATE INDEX ";
 
 //   protected static final String truncateTableSql = "TRUNCATE TABLE ";
    protected static final String truncateTableSql = "DELETE FROM ";
    protected static final String dropTableSql = "DROP TABLE ";
    protected static final String dropTableIfExistsSql = "DROP TABLE IF EXISTS ";
   
	
	protected static final String DatabaseUserString = "DatabaseManager.DB.papillon.Connection.User";
	
	protected static java.util.regex.Pattern quotePattern = java.util.regex.Pattern.compile("'");
    
    protected String currentSQL = "";
    protected DBTransaction transaction;
	

    /**
     *
     * @param trans
     * @param sql
     */
    public ManageDatabase(DBTransaction trans, String sql) {
        this.transaction = trans;
        this.currentSQL = sql;
    }
    
   /**
        * WARNING!	 This method is	disabled.
     * The reason is that this special set of Queries do not return any result.
     * Moreover, it is also disabled in Quries implementations.
    * @deprecated use getNextDO() instead.
	 */
	public Object next(ResultSet rs) throws SQLException, ObjectIdException {
        // TODO: It	would be nice to throw an unchecked	exception here
        // (an exception that extends RuntimeException)
        // that	would be guaranteed	to appear during application testing.
        throw new ObjectIdException("next()	should not be used.	 Use getNextDO() instead." );
        //return null;
    }
    
    /*<table id="fr.imag.clips.papillon.data.VolumeEntry" dbTableName="volume">
        <column id="headword" isIndex="true" usedForQuery="true">
        <javadoc>
        </javadoc>
        <type canBeNull="true" dbType="LONGVARCHAR" javaType="String"/>
        </column>
        <column id="id" usedForQuery="true">
        <javadoc>
        </javadoc>
        <type canBeNull="true" dbType="VARCHAR" javaType="String" size="255"/>
        </column>
        <column id="pos" usedForQuery="true">
        <javadoc>
        </javadoc>
        <type canBeNull="true" dbType="VARCHAR" javaType="String" size="255"/>
        </column>
        <column id="pronunciation" isIndex="true" usedForQuery="true">
        <javadoc>
        </javadoc>
        <type canBeNull="true" dbType="VARCHAR" javaType="String" size="255"/>
        </column>
        <column id="translation" isIndex="true" usedForQuery="true">
        <javadoc>
        </javadoc>
        <type canBeNull="true" dbType="LONGVARCHAR" javaType="String"/>
        </column>
        <column id="key1" usedForQuery="true">
        <javadoc>
        </javadoc>
        <type canBeNull="true" dbType="VARCHAR" javaType="String" size="255"/>
        </column>
        <column id="key2" usedForQuery="true">
        <javadoc>
         </javadoc>
        <type canBeNull="true" dbType="VARCHAR" javaType="String" size="255"/>
        </column>
        <column id="xmlCode" usedForQuery="true">
        <javadoc>
         </javadoc>
        <type dbType="LONGVARCHAR" javaType="String"/>
        </column>
        </table>*/ 
    
    
    protected static final String createVolumeTableParamsSql = " (" +
        "headword TEXT DEFAULT '\'''\''    ," +
		//"dom BYTEA NOT NULL   ," +
		//"htmldom BYTEA NOT NULL   ," +
        "xmlCode TEXT DEFAULT '\'''\'' NOT NULL   ," +
        
        "ObjectId DECIMAL(19,0) NOT NULL PRIMARY KEY," +
        "ObjectVersion INTEGER NOT NULL);";
    
    protected static final String createIndexTableParamsSql = " (" +
		"key VARCHAR(255) DEFAULT '\'''\''    ," +
		"lang VARCHAR(3) DEFAULT '\'''\''    ," +
		"value VARCHAR(255) DEFAULT '\'''\''    ," +
        "entryid DECIMAL(19,0) NOT NULL    ," +
        "msort VARCHAR(255) DEFAULT '\'''\''    ," +
        
        "ObjectId DECIMAL(19,0) NOT NULL PRIMARY KEY," +
        "ObjectVersion INTEGER NOT NULL);";
    
    protected static final String createLinkTableParamsSql = "(" +
		"targetId VARCHAR(255) DEFAULT '\'''\''    ,"+
		"elementId VARCHAR(255) DEFAULT '\'''\''    ,"+
		"name VARCHAR(255) DEFAULT '\'''\''    ," +
    	"lang VARCHAR(3) DEFAULT '\'''\''    ," +
    	"volumeTarget VARCHAR(255) DEFAULT '\'''\''    ," +
   		"type VARCHAR(255) DEFAULT '\'''\''    ," +
   		"label VARCHAR(255) DEFAULT '\'''\''    ," +
   		"entryId DECIMAL(19,0) NOT NULL    ," +
  		"weight DECIMAL(3,2) NOT NULL    ," +  
   		
        "ObjectId DECIMAL(19,0) NOT NULL PRIMARY KEY," +
        "ObjectVersion INTEGER NOT NULL);";
    
    private static String createVolumeTableSql(String table, String lang) throws  PapillonBusinessException {
        String createVolumeQuery = createTableSql + table + createVolumeTableParamsSql;
        String sortIndexQuery = createIndexSql + table + "_msort_idx" + " ON " + table + " (multilingual_sort( '" + lang + "',headword ));";
            return  createVolumeQuery + sortIndexQuery;
        }
    
    private static String createIndexTableSql(String indexTable) throws  PapillonBusinessException {
        String createIndexTable = createTableSql + indexTable + createIndexTableParamsSql;
        String createKeyIndex = createIndexSql + indexTable + "_key_idx" + " ON " + indexTable + " ( key,lang,value );";
        String createEntryidIndex = createIndexSql + indexTable + "_entryid_idx" + " ON " + indexTable + " ( entryid );";
        String createMsortIndex = createIndexSql + indexTable + "_msort_idx" + " ON " + indexTable + " ( msort );";
            return  createIndexTable + createKeyIndex + createEntryidIndex + createMsortIndex;
        }
    
    private static String createLinkTableSql(String linkTable) throws  PapillonBusinessException {
       String createLinkTable = createTableSql + linkTable + createLinkTableParamsSql;
        String createEntryidIndex = createIndexSql + linkTable + "_entryid_idx" + " ON " + linkTable + " ( entryid );";
        String createNameIndex = createIndexSql + linkTable + "_name_idx" + " ON " + linkTable + " ( name );";
        String createLangIndex = createIndexSql + linkTable + "_lang_idx" + " ON " + linkTable + " ( lang );";
        String createVolumeIndex = createIndexSql + linkTable + "_volumetarget_idx" + " ON " + linkTable + " ( volumetarget );";
        String createTypeIndex = createIndexSql + linkTable + "_type_idx" + " ON " + linkTable + " ( type );";
        String createWeightIndex = createIndexSql + linkTable + "_weight_idx" + " ON " + linkTable + " ( weight );";
        String createLabelIndex = createIndexSql + linkTable + "_label_idx" + " ON " + linkTable + " ( label );";

        return  createLinkTable + createEntryidIndex + createNameIndex + createLangIndex + createVolumeIndex + createTypeIndex + createWeightIndex + createLabelIndex;
    }
   
    public static void createVolumeTables(String volumeTable, String lang, String indexTable, String linkTable)  throws  PapillonBusinessException {
        java.util.Vector TableNames = ManageDatabase.getTableNames();
        String createTablesSql = "";
        if (!TableNames.contains(volumeTable)) {
            createTablesSql += createVolumeTableSql(volumeTable, lang);
        }
        if (!TableNames.contains(indexTable)) {
            createTablesSql += createIndexTableSql(indexTable);
        }
        if (!TableNames.contains(linkTable)) {
            createTablesSql += createLinkTableSql(linkTable);
        }
        executeSql(createTablesSql);
    }
    

    
    
    public static void truncateIndexForTable(String table, String name) throws  PapillonBusinessException {
        String query = truncateTableSql + table + "_" + name + "_idx";
        executeSql(query);
    }
    
    public static void truncateTable(String table) throws  PapillonBusinessException {
        try {    
            executeSql(truncateTableSql + table);
            //((DBTransaction) CurrentDBTransaction.get()).commit();
            //fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Table: " + table + " truncated");
            
             } catch (Exception e) {
                 throw new PapillonBusinessException("ManageDatabase.truncateTable: " + truncateTableSql + table);
             }
        }
    
    public static void dropVolumeTables(String volumeTable, String indexTable, String linkTable) throws  PapillonBusinessException {
        
        executeSql(dropTableIfExistsSql + volumeTable + ";" + dropTableIfExistsSql + indexTable + ";" +dropTableIfExistsSql + linkTable + ";");
        
        //fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Table: " + table + " dropped");
        
    }
    
    public static String multilingual_sort(String lang, String value) throws  PapillonBusinessException {
		//FIXME: should be called getSortKey
 		String result = lang + value;
		
		java.util.regex.Matcher quoteMatcher = quotePattern.matcher(value);
		String newValue = quoteMatcher.replaceAll("''");
		
		String sql = "SELECT multilingual_sort('"+ lang + "','" + newValue + "')";
		java.util.Vector myResultVector = executeSqlQuery(sql,"multilingual_sort");
		
		if (myResultVector != null && myResultVector.size()>0) {
			result = (String) myResultVector.elementAt(0);
		}
		if (result.length()>255) {
			result = result.substring(0,254);
		}
		return result;
    }

    public static void executeSql (String sql) throws PapillonBusinessException {
		DBTransaction transaction = CurrentDBTransaction.get();
        ManageDatabase req = new ManageDatabase(transaction, sql);
        
        //Flush the current transaction (?)
        // Is this really usefull ?
        if ((transaction!=null) &&
            (transaction instanceof com.lutris.appserver.server.sql.CachedDBTransaction)) {
            
            if(((com.lutris.appserver.server.sql.CachedDBTransaction)transaction).getAutoWrite()) try {
                transaction.write();
            } catch (SQLException sqle) {
                sqle.printStackTrace();
                throw new PapillonBusinessException("Couldn't write transaction: "+sqle);
            }
        }
        
        // Create the DB Query Object
        DBQuery	dbQuery = null;
        
        try {
            if (transaction == null) {
                dbQuery	= DODS.getDatabaseManager().createQuery();
            } else {
                dbQuery	= transaction.createQuery();
            }
			
            dbQuery.query( req	);	  // invokes executeQuery
            
        } catch ( DatabaseManagerException e ) { 
            String err = "ERROR SpecialDatabaseRequest: Could not create a DBQuery.  ";
            throw new PapillonBusinessException( err, e );
        } catch ( SQLException e ) { 
            String err = "ERROR SpecialDatabaseRequest: Exception while running the query: " + sql;
            throw new PapillonBusinessException( err, e );
        } finally {
            if ( null != dbQuery ) {
                dbQuery.release();
            }
        }
    }
    
    
    public ResultSet executeQuery(DBConnection conn) throws SQLException {
        conn.execute(currentSQL);
        return null;
    }
    
    public static java.util.Vector getTableNames ()
        throws PapillonBusinessException {
			//String sql = "SELECT tablename FROM pg_tables where tableowner='papillon'";
			
			java.util.Vector TableNames = new java.util.Vector();
            DBConnection myDbConnection = null;
            try {
                myDbConnection = Enhydra.getDatabaseManager().allocateConnection();
                String dbUserStringConfig = "DatabaseManager.DB." + Enhydra.getDatabaseManager().getDefaultDB() + ".Connection.User";
                String databaseUser = Enhydra.getApplication().getConfig().getString(dbUserStringConfig);

				com.lutris.dods.builder.generator.query.QueryBuilder myQueryBuilder = new com.lutris.dods.builder.generator.query.QueryBuilder("pg_tables");
				com.lutris.dods.builder.generator.query.RDBTable pg_tables = new com.lutris.dods.builder.generator.query.RDBTable("pg_tables");
				com.lutris.dods.builder.generator.query.RDBColumn tableOwnerColumn = new com.lutris.dods.builder.generator.query.RDBColumn(pg_tables,"tableowner");
				myQueryBuilder.addWhere(tableOwnerColumn, databaseUser);
				java.sql.ResultSet myResultSet = myQueryBuilder.executeQuery(myDbConnection);
               while (myResultSet.next()) {
				   TableNames.addElement(myResultSet.getString("tablename"));
			   }
            }  catch(SQLException se) {
                //very important to throw out bad connections
                PapillonLogger.writeErrorMsg("SQL exception: ");
				// System.out.println("SQL exception: ");
				se.printStackTrace();
                if(myDbConnection != null && myDbConnection.handleException(se)) myDbConnection=null;
            } catch(Exception e) {
				String err = "ERROR DatabaseConnexion: Exception while querying the table names";
                PapillonLogger.writeErrorMsg(err);
                // System.out.println(err);
				e.printStackTrace();
				//throw new PapillonBusinessException( err, e );
            } finally {
                if(myDbConnection!=null) {
					try {
						myDbConnection.reset();
						myDbConnection.release();
					}
					catch ( SQLException e ) { 
						String err = "ERROR DatabaseConnexion2: Exception while releasing transaction after querying the table names";
                        PapillonLogger.writeErrorMsg(err);
						// System.out.println(err);
						e.printStackTrace();
//						throw new PapillonBusinessException( err, e );
					}
                }
            }
			return TableNames;
        }

        public static java.util.Vector getColumnNames(String table)
        throws PapillonBusinessException {
			//select attname from pg_attribute, pg_class where pg_class.relname='lexalpfra'
            //   and pg_class.oid=pg_attribute.attrelid and pg_attribute.attnum > 0;

			java.util.Vector columnNames = new java.util.Vector();
            DBConnection myDbConnection = null;
            try {
                myDbConnection = Enhydra.getDatabaseManager().allocateConnection();

				QueryBuilder myQueryBuilder = new QueryBuilder("pg_attribute", "attname");
				RDBTable pg_attribute = new RDBTable("pg_attribute");
                RDBTable pg_class = new RDBTable("pg_class");
                RDBColumn relname = new RDBColumn(pg_class,"relname");
                RDBColumn classoid = new RDBColumn(pg_class,"oid");
                RDBColumn attrelid = new RDBColumn(pg_attribute,"attrelid");
                RDBColumn attnum = new RDBColumn(pg_attribute,"attnum");
				myQueryBuilder.addWhere(relname, table);
                myQueryBuilder.addWhere(classoid,attrelid);
                myQueryBuilder.addWhere(attnum,0,QueryBuilder.GREATER_THAN);
                java.sql.ResultSet myResultSet = myQueryBuilder.executeQuery(myDbConnection);
               while (myResultSet.next()) {
				   columnNames.addElement(myResultSet.getString("attname"));
			   }
            }  catch(SQLException se) {
                //very important to throw out bad connections
                PapillonLogger.writeErrorMsg("SQL exception: ");
				// System.out.println("SQL exception: ");
				se.printStackTrace();
                if(myDbConnection != null && myDbConnection.handleException(se)) myDbConnection=null;
            } catch(Exception e) {
				String err = "ERROR DatabaseConnexion: Exception while querying the table names";
                PapillonLogger.writeErrorMsg(err);
                // System.out.println(err);
				e.printStackTrace();
				//throw new PapillonBusinessException( err, e );
            } finally {
                if(myDbConnection!=null) {
					try {
						myDbConnection.reset();
						myDbConnection.release();
					}
					catch ( SQLException e ) {
						String err = "ERROR DatabaseConnexion2: Exception while releasing transaction after querying the table names";
                        PapillonLogger.writeErrorMsg(err);
						// System.out.println(err);
						e.printStackTrace();
//						throw new PapillonBusinessException( err, e );
					}
                }
            }
			return columnNames;
        }
        /*
		I keep this function because the other one (executeSql) use a transaction and thus, I cannot retrieve the results because
		 the resultSet is closed.
		*/
	    protected static java.util.Vector executeSqlQuery (String sql, String columnName) {
	            DBConnection myDbConnection = null;
	            java.util.Vector resVector = new java.util.Vector();
	            try {
	                myDbConnection = Enhydra.getDatabaseManager().allocateConnection();
	                
	                ResultSet myResultSet = myDbConnection.executeQuery(sql);
	                
	                if(myResultSet != null) {
						while (myResultSet.next() && columnName != null) {
							resVector.addElement(myResultSet.getString(columnName));			
						}
	                    myResultSet.close();
	                }
	            }  catch(SQLException se) {
	                se.printStackTrace();
	                //very important to throw out bad connections
	                
	                if(myDbConnection.handleException(se)) myDbConnection=null;
	            } catch(Exception e) {
	                e.printStackTrace();
	            } finally {
	                if(myDbConnection!=null) {
						try {
							myDbConnection.reset();
							myDbConnection.release();
						}  catch(SQLException se) {
							se.printStackTrace();
							//very important to throw out bad connections
						if(myDbConnection.handleException(se)) myDbConnection=null;
						}
	                }
	            }
				return resVector;
	        }
	
	public static String getColumnType(String table, String column) {
		String result = "";
		
		String sql = "select typname from pg_type where typelem in " +
		"(select atttypid from pg_attribute where attname='" + column + "' "+
		 "and attrelid in (select oid from pg_class where relname='" + table + "'));";
		java.util.Vector resultVector = executeSqlQuery(sql, "typname");
		
		if (resultVector != null && resultVector.size()>0) {
			result = (String) resultVector.elementAt(0);
		}
		return result;
	}

/*
    protected static void simpleExecuteSql (String sql)
        throws PapillonBusinessException {
            DBConnection myDbConnection = null;
            try {
                myDbConnection = Enhydra.getDatabaseManager().allocateConnection();
                
                myDbConnection.execute(sql);
                
            }  catch(SQLException se) {
                //very important to throw out bad connections
                
                if(myDbConnection.handleException(se)) myDbConnection=null;
            } catch(Exception e) {
				String err = "ERROR DatabaseConnexion: Exception while running the query: " + sql;
				throw new PapillonBusinessException( err, e );
            } finally {
                if(myDbConnection!=null) {
					try {
						myDbConnection.reset();
						myDbConnection.release();
					}
					catch ( SQLException e ) { 
						String err = "ERROR DatabaseConnexion: Exception while running the query: " + sql;
						throw new PapillonBusinessException( err, e );
					}
                }
            }
        } */
//    
//    protected static void executeSqlQuery (String sql)
//        throws java.sql.SQLException {
//            DBConnection myDbConnection = null;
//            ResultSet myResultSet = null;
//            try {
//                myDbConnection = Enhydra.getDatabaseManager().allocateConnection();
//                
//                myResultSet = myDbConnection.executeQuery(sql);
//                
//                if(myResultSet != null) {
//                    myResultSet.close();
//                }
//            }  catch(SQLException se) {
//                se.printStackTrace();
//                //very important to throw out bad connections
//                
//                if(myDbConnection.handleException(se)) myDbConnection=null;
//            } catch(Exception e) {
//                e.printStackTrace();
//            } finally {
//                if(myDbConnection!=null) {
//                    myDbConnection.reset();
//                    myDbConnection.release();
//                }
//            }
//        }
}
