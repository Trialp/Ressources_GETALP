<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
        <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="fileHeader">
<xsl:value-of select="dodst:initColumn()"/>
/*-----------------------------------------------------------------------------
 * Enhydra Java Application Server
 * Copyright 1997-2000 Lutris Technologies, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes Enhydra software developed by Lutris
 *    Technologies, Inc. and its contributors.
 * 4. Neither the name of Lutris Technologies nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LUTRIS TECHNOLOGIES AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL LUTRIS TECHNOLOGIES OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * contains WebDocWf extensions
 *
 *-----------------------------------------------------------------------------
 * <xsl:value-of select="PACKAGE"/>/<xsl:value-of select="CLASS_NAME"/>DO.java
 *-----------------------------------------------------------------------------
 */

<xsl:value-of select="dodst:resetCounter()"/>
package <xsl:value-of select="translate(PACKAGE,'/','.')"/>;

import java.io.*;
import java.sql.*;
import java.math.*;
import java.util.Hashtable;
import java.util.Collection;
import java.util.Properties;
import java.io.FileInputStream;
import java.util.Vector;
import java.util.Map;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Iterator;
import java.lang.reflect.Method;

import com.lutris.logging.LogChannel;
import com.lutris.logging.Logger;
import org.enhydra.dods.DODS;
import com.lutris.util.Config;
import com.lutris.util.ConfigException;
import com.lutris.appserver.server.sql.*;
import com.lutris.appserver.server.sql.standard.*;
import com.lutris.dods.builder.generator.dataobject.*;
import com.lutris.dods.builder.generator.query.*;
import org.enhydra.dods.cache.Condition;
import org.enhydra.dods.cache.DataStructCache;
import org.enhydra.dods.cache.QueryCache;
import org.enhydra.dods.cache.QueryCacheImpl;
import org.enhydra.dods.cache.QueryResult;
import org.enhydra.dods.cache.ConfigurationAdministration;
import org.enhydra.dods.statistics.Statistics;
import com.lutris.classloader.MultiClassLoader;
import org.enhydra.xml.XMLConfig;
import org.enhydra.dods.Common;
import org.enhydra.dods.cache.CacheConstants;
import org.enhydra.dods.exceptions.AssertionDataObjectException;
import org.enhydra.dods.exceptions.CacheObjectException;


<xsl:if test="DO_IS_OID_BASED='true'">
// WebDocWf extension for DODS row instance security and genric store
import org.webdocwf.dods.access.*;
// end of WebDocWf extension for DODS row instance security and generic store
</xsl:if>
/**
 * Data core class, used to set and retrieve the <xsl:value-of select="CLASS_NAME"/>DO information.
 *
 * @version $Revision$
 * @author  <xsl:value-of select="AUTHOR"/>
 * @since   <xsl:value-of select="PROJECT_NAME"/>
 */
<xsl:if test="IS_ABSTRACT='true'">abstract</xsl:if> public class <xsl:value-of select="CLASS_NAME"/>DO extends <xsl:if test="(GENERATE_SECURE='true' or IS_ANY_COLUMN_SECURE='true') and DO_IS_OID_BASED='true'">org.webdocwf.dods.access.SecureDO</xsl:if><xsl:if test="not((GENERATE_SECURE='true' or IS_ANY_COLUMN_SECURE='true') and DO_IS_OID_BASED='true')">com.lutris.dods.builder.generator.dataobject.GenericDO</xsl:if> implements <xsl:value-of select="CLASS_NAME"/>DOI, java.io.Serializable {
    /**
     * Static final data members name the table and columns for this DO.
     * By using these members with an instance of the QueryBuilder class,
     * an application can perform straight SQL queries while retaining
     * compile-time checking of table and column usage.
     *
     * Example:  List the Cities containing Persons named Bob:
     *
     *    Using straight SQL with QueryBuilder:
     *      Pro: code runs faster because you create fewer objects
     *      Con: code is less clear
     *
     *         Vector fields = new Vector();
     *         fields.addElement( AddressDO.City );
     *         QueryBuilder qb = new QueryBuilder( fields );
     *         qb.addWhere( PersonDO.FirstName, "Bob" );
     *         qb.addWhere( PersonDO.PrimaryKey, AddressDO.Person );
     *         RDBRow row;
     *         while ( null != ( row = qb.getNextRow() ) ) {
     *             String city = row.get( AddressDO.City ).getString();
     *         }
     *
     *    Using Query/DO classes:
     *      Pro: code is (often) clearer
     *      Con: code runs slower because you create more objects
     *
     *         PersonQuery pq = new PersonQuery();
     *         pq.setQueryFirstName( "Bob" );
     *         PersonDO[] bobs = pq.getDOArray();
     *         for ( int i = 0; i &lt; bobs.length; i++ ) {
     *             AddressQuery aq = new AddressQuery();
     *             aq.setQueryPerson( bobs[i] );
     *             AddressDO addr = aq.getNextDO();
     *             String city = addr.getCity();
     *         }
     */
	 
	 // GS: added the dbtablename + made table a non static attribute (will be initialized at construction time)
     protected String dbtablename;

    /**
     * List of reference objects
     */
    private HashMap refs = null;


    /**
    * null : use old style init query
    * true : use columns names with table prefix
    * false: use columns names withouth table prefix
    */
    private static Boolean useOrderedWithTable = null ;


    /**
    * String with column names
    */
	// static String columnsNameString = null;
    public  String getColumnsNameString() {
        return getColumnsNameString(this.dbtablename);
    }
    
    /**
    * Is DO first node in DELETE CASCADE sekvence
    */
    private boolean isRootDeleteNode = true;


    /**
     * Name of the logical database for which DO object was created
     */
    protected String originDatabase = null;


    /**
     * Return the name of the logical database for which DO object was created.
     *
     * @deprecated Use get_OriginDatabase()
     * @return origin logical database name.
     *
     */
    public String getOriginDatabase() {
        return get_OriginDatabase();
    }

    /**
     * Return the name of the logical database for which DO object was created.
     *
     * @return origin logical database name.
     *
     */
    public String get_OriginDatabase() {
        return get_DataStruct().get_Database();
    }

	// GS: we now return the value of dbtablename (value initialized at construction time)
    /**
     * Return the value of dbtablename as the name of the table in the database
     * which contains <xsl:value-of select="CLASS_NAME"/>DO objects.
     * This method overrides CoreDO.getTableName()
     * and is used by CoreDO.executeUpdate() during error handling.
     *
     * @return The database table name.
     *
     * @see com.lutris.appserver.server.sql.CoreDO CoreDO
     * author Jay Gunter
     */
    public String getTableName() {
        return this.dbtablename;
	    //return &quot;<xsl:value-of select="TABLE_NAME"/>&quot;;
    }
<xsl:if test="DO_IS_OID_BASED='true'">
    /**
     * final RDBColumn PrimaryKey for use with QueryBuilder.
     * See example above.
     */
	 // GS: The PrimaryKey is not static anymore. It will be set at construction time;
     public static RDBColumn getPrimaryKey(String tablename) {
         RDBTable  table = new RDBTable(tablename);
         return new RDBColumn( table,
                               getPrimaryKeyName() );

     }
    /* RDBColumns for <xsl:value-of select="CLASS_NAME"/>DO attributes are defined below. */
</xsl:if>

    /* Using a DO (and its Query class) to access a VIEW instead of a TABLE:
     *
     * A DO (and its Query class) can be used to access a VIEW
     * instead of a TABLE.  The Data Object is created as usual in DODS,
     * but the "create table" SQL command for that DO is not used.
     * Instead, you substitute a "create view" command to create a
     * virtual table in the database; this is often done to provide
     * convenient access to a collection of tables joined together.
     *
     * A VIEW usually does not return "oid" and "version" columns;
     * often (but now always) a VIEW is defined to return the "oid" column
     * for one of the tables joined together in the definition of the VIEW.
     *
     * If notUsingOId is true, <xsl:value-of select="CLASS_NAME"/>DO.createExisting(ResultSet)
     * will NOT invoke the GenericDO(ResultSet) constructor
     * so to avoid attempting to extract the "oid" and "version" columns
     * from the ResultSet.
     */
<xsl:if test="DO_IS_OID_BASED='false'">    static protected final boolean notUsingOId = true;
</xsl:if>
<xsl:if test="DO_IS_OID_BASED='true'">    static protected final boolean notUsingOId = false;
</xsl:if>

    /**
     * A DO class contains a reference to a DataStruct class.
     * This reference can be null (when the data for the DO
     * has not yet been retrieved from the database),
     * allowing a DO object to be a lightweight placeholder
     * until its data is needed.
     */
    private <xsl:value-of select="CLASS_NAME"/>DataStruct data = null;

    /**
     * A DO class contains a reference to a DBTransaction class.
     * This reference can be null (when the DO is created without
     * transaction).
     */
    private DBTransaction transaction = null;

    /**
     * Return transaction which DO belongs.
     *
     * @return DBTransaction or null if not specified.
     *
     */
    public DBTransaction get_transaction() {
        return transaction;
    }

    /**
     * Set Transaction to current DO.
     *
     * @param trans The transaction.
     * @return true if the operation was successfully performed, otherwise false.
     */
    protected boolean setTransaction(DBTransaction trans) {
        boolean isOK=false;
        if (get_transaction() == null) {
            transaction = trans;
            isOK=true;
        } else {
        if(get_transaction().equals(trans))
            isOK=true;
        }
        return isOK;
    }


    /**
     * Return information whether the data for this object has been marked read-only.
     *
     * @return True if the data for this object has been marked read-only.
     *
     */
    public boolean isReadOnly() {
      return getConfigurationAdministration().getTableConfiguration().isReadOnly();
     }

    /**
     * Sets DO's data.
     * @param data Data object.
     * @deprecated Use set_Data()
     */
     public void setData (Object data) {
        set_Data(data);
     }

    /**
     * Sets DO's data.
     * @param data Data object.
     */
     public void set_Data (Object data) {
        this.data = (<xsl:value-of select="CLASS_NAME"/>DataStruct)data;
     }

    /**
     * Sets original DO's data.
     * @param data Data object.
     */
     public void originalData_set (Object data) {
        originalData = (<xsl:value-of select="CLASS_NAME"/>DataStruct)data;
     }

    /**
     * Returns DO's data.
     * @return DO's data.
     * @deprecated Use get_Data()
     */
     public Object getData () {
        return get_Data();
     }

    /**
     * Returns DO's data.
     * @return DO's data.
     */
     public Object get_Data () {
        return (null != data)? data : originalData;
     }

    /**
     * Returns dataStruct.
     * @return Data Struct object.
     * @deprecated Use get_DataStruct()
     */
    public <xsl:value-of select="CLASS_NAME"/>DataStruct getDataStruct () {
        return get_DataStruct();
    }

    /**
     * Returns dataStruct.
     * @return Data Struct object.
     */
    public <xsl:value-of select="CLASS_NAME"/>DataStruct get_DataStruct () {
        return (<xsl:value-of select="CLASS_NAME"/>DataStruct) get_Data();
    }

    /**
     * Returns original DO's data.
     * @return Original DO's data.
     */
     public Object originalData_get () {
        return originalData;
     }

     public void checkDup () throws DatabaseManagerException, com.lutris.appserver.server.sql.ObjectIdException {
         if (isDeletedFromDatabase)
             throw new DatabaseManagerException("Object "+get_Handle()+" is deleted");
         if (null == data) {
             data = ((<xsl:value-of select="CLASS_NAME"/>DataStruct)originalData).duplicate();
         data.readOnly = false;
         }
     }

	
    /**
     * Protected constructor. Only derived classes should call it.
     *
     * @param is_view Is this view or not.
     *
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     */
	 // GS: Constructors now have a tablename additional mandatory operator.
    protected <xsl:value-of select="CLASS_NAME"/>DO ( String tablename, boolean is_view )
    throws ObjectIdException, DatabaseManagerException {
        super( is_view );
        this.dbtablename = tablename;

        if(isTransactionCheck()) {
            DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created :"+(is_view?"":" Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_Handle()+", version: "+get_Version())+" \n");
            (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

        }
    }
    /**
     * Protected constructor. Only derived classes should call it.
     *
     * @param is_view Is this view or not.
     * @param dbTrans Database transaction.
     *
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     */
 	 // GS: Constructors now have a tablename additional mandatory operator.
    protected <xsl:value-of select="CLASS_NAME"/>DO ( String tablename, boolean is_view, DBTransaction dbTrans )
    throws ObjectIdException, DatabaseManagerException {
        super( is_view );
        this.dbtablename = tablename;

        setTransaction(dbTrans);
        if(dbTrans!=null) {
           originDatabase = dbTrans.getDatabaseName();
        }
        if(originDatabase==null)
           originDatabase = get_logicalDBName();<xsl:if test="DO_IS_MULTIDB_BASED='true'">
        else {
           try {
            if(useLogicalDatabase(tablename, originDatabase))
               readCacheConfiguration(tablename, originDatabase);
           } catch (Exception ex) {
             DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default cache configuration for '"+originDatabase+"' database");
           }
         }</xsl:if>
        get_DataStruct().set_Database(originDatabase);
        addToTransactionCache();

    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'">
	/**
     * GenericDO constructor. Extended with the tablename.
     *
     * @param is_view Is this view or not.
     *
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     */
    protected <xsl:value-of select="CLASS_NAME"/>DO ( String tablename, String dbName, ResultSet rs )
    throws java.sql.SQLException, ObjectIdException, DatabaseManagerException {
        super( dbName, rs );
        this.dbtablename = tablename;
    }

    /**
     * Protected constructor. Only derived classes should call it.
     *
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param is_view
     *
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     */
 	 // GS: Constructors now have a tablename additional mandatory operator.
    protected <xsl:value-of select="CLASS_NAME"/>DO ( String tablename, String dbName,  boolean is_view )
    throws ObjectIdException, DatabaseManagerException {
        super(dbName,is_view);
        this.dbtablename = tablename;

        if (dbName == null)
            dbName = get_logicalDBName();
        if(useLogicalDatabase(tablename, dbName)) {
          try {
            readCacheConfiguration(tablename, dbName);
          } catch (Exception ex) {
             DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default cache configuration for '"+dbName+"' database");
          }
        }
        get_DataStruct().set_Database(dbName);
        if(isTransactionCheck()) {
        DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_Handle()+", version: "+get_Version()+" \n");
        (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

        }
    }

</xsl:if>
    /**
     * Protected constructor.  Only derived classes should call it.
     *
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     */
 	 // GS: Constructors now have a tablename additional mandatory operator.
    protected <xsl:value-of select="CLASS_NAME"/>DO ( String tablename )
    throws ObjectIdException, DatabaseManagerException {
        super( notUsingOId );
        this.dbtablename = tablename;

        originDatabase = get_logicalDBName();
        get_DataStruct().set_Database(originDatabase);
        if(isTransactionCheck()) {
            DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_Handle()+", version: "+get_Version()+" \n");
            (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

        }
        if (autoSaveAllowed&amp;&amp;isAutoSaveCreateVirgin()&amp;&amp;null != transaction) {
            try {
                save(transaction,false);
            } catch (Exception ex) {
                        DODS.getLogChannel().write(Logger.DEBUG,"Faild to AutoSave virgin DO: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class\n");
            }
        }
    }

    /**
     * Protected constructor.  Only derived classes should call it.
     * @param dbTrans The current database transaction.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     */
 	 // GS: Constructors now have a tablename additional mandatory operator.
    protected <xsl:value-of select="CLASS_NAME"/>DO ( String tablename, DBTransaction dbTrans )
    throws ObjectIdException, DatabaseManagerException {
        super( notUsingOId );
        this.dbtablename = tablename;

        setTransaction(dbTrans);
        if(dbTrans!=null) {
           originDatabase = dbTrans.getDatabaseName();
        }
        if(originDatabase==null)
           originDatabase = get_logicalDBName();<xsl:if test="DO_IS_MULTIDB_BASED='true'">
        else {
            if(useLogicalDatabase(tablename, originDatabase)) {
             try {
               readCacheConfiguration(tablename, originDatabase);
            } catch (Exception ex) {
              DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default cache configuration for '"+originDatabase+"' database");
          }
         }
        }    </xsl:if>
        get_DataStruct().set_Database(originDatabase);
<xsl:if test="DO_IS_OID_BASED='true'">
</xsl:if>
        addToTransactionCache();
        if (autoSaveAllowed&amp;&amp;isAutoSaveCreateVirgin()&amp;&amp;null != transaction) {
            try {
                save(transaction,false);
            } catch (Exception ex) {
                        DODS.getLogChannel().write(Logger.DEBUG,"Faild to AutoSave virgin DO: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class\n");
            }
        }
    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'">
    /**
     * Protected constructor. Only derived classes should call it.
     *
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     *
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     */
 	 // GS: Constructors now have a tablename additional mandatory operator.
    protected <xsl:value-of select="CLASS_NAME"/>DO ( String tablename, String dbName)
    throws ObjectIdException, DatabaseManagerException {
        super(dbName,notUsingOId);
        this.dbtablename = tablename;

        if (dbName == null)
            dbName = get_OriginDatabase();
        if(useLogicalDatabase(tablename, dbName)) {
          try {
            readCacheConfiguration(tablename, dbName);
         } catch (Exception ex) {
             DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default cache configuration for '"+dbName+"' database");
         }
        }
        get_DataStruct().set_Database(dbName);
<xsl:if test="DO_IS_OID_BASED='true'">
</xsl:if>
        if(isTransactionCheck()) {
        DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_Handle()+", version: "+get_Version()+" \n");
        (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

        }
        if (autoSaveAllowed&amp;&amp;isAutoSaveCreateVirgin()&amp;&amp;null != transaction) {
            try {
                save(transaction,false);
            } catch (Exception ex) {
                        DODS.getLogChannel().write(Logger.DEBUG,"Faild to AutoSave virgin DO: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class\n");
            }
        }
    }
</xsl:if>

    /**
     * isLoaded()
     * Returns information whether object's data is loaded from database.
     * @return true if the data for this object has been retrieved
     * from the database.
     */
    public boolean isLoaded() {
        return (null != originalData)&amp;&amp;(!get_DataStruct().isEmpty);
    }

<xsl:if test="DO_IS_OID_BASED='true'">
    /**
     * Load the fields for the DO from the database.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    public void loadData()
    throws SQLException, ObjectIdException, DataObjectException {
        if (null == originalData&amp;&amp;!get_DataStruct().isEmpty) {
            originalData = new <xsl:value-of select="CLASS_NAME"/>DataStruct ();
        }
<xsl:if test="IS_ABSTRACT='false'">
        ObjectId id = get_OId();
        if ( null == id )
            return;
        if ( ! isPersistent() )   // DO from createVirgin
            return;
        // DO from createExisting.  Complain if no record in database.
        <xsl:value-of select="CLASS_NAME"/>Query query;
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        try {</xsl:if>
            if(get_transaction()!=null)
                query = new <xsl:value-of select="CLASS_NAME"/>Query (dbtablename, get_transaction());
            else {<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
              if (get_OriginDatabase() != null)
                query = new <xsl:value-of select="CLASS_NAME"/>Query (dbtablename, get_OriginDatabase());
              else</xsl:if></xsl:if>
               query = new <xsl:value-of select="CLASS_NAME"/>Query (dbtablename, get_transaction());
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
          }<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        } catch (DatabaseManagerException e) {
            throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException", e);
        }
</xsl:if></xsl:if>
//        if(get_refs()!=null)
//        {
        query.setRefs(get_refs());
//        }
        query.setQueryOId( id );
        query.requireUniqueInstance();
        <xsl:value-of select="CLASS_NAME"/>DO obj;
        try {
           query.setLoadData(true);
           obj = query.getNextDO();
            if ( null == obj )
                throw new DataObjectException(&quot;<xsl:value-of select="CLASS_NAME"/>DO DO not found for id=&quot; + id );
            makeIdentical(obj);
            set_Version(    obj.get_Version() );
            get_DataStruct().isEmpty = false;
        } catch ( NonUniqueQueryException e ) {
            throw new ObjectIdException( "Duplicate ObjectId" );
        }
</xsl:if>
    }
</xsl:if>

<xsl:if test="DO_IS_OID_BASED='true'">
    /**
     * Load the actual DO data if necessary.
     * Called by get/set methods.
     *
     * @exception DataObjectException If a data access error occurs.
     */
    protected void checkLoad()
    throws DataObjectException {
        if (null == originalData||get_DataStruct().isEmpty) {
            try {
                loadData();
            } catch ( Exception e ) {
                throw new DataObjectException(&quot;Unable to load data for <xsl:value-of select="CLASS_NAME"/>DO id=&quot; + get_OId() +
                                              &quot;, error = &quot;, e);
            }
    }
    }
</xsl:if>

    /**
     * Protected constructor used by createExisting(ObjectId) above.
     *
     * @param id The ObjectId for the object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   Should never see this exception since GenericDO.ctor(ObjectId)
     *   never accesses the database.
     */
    protected <xsl:value-of select="CLASS_NAME"/>DO(String tablename, ObjectId id )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        super( id );
        this.dbtablename = tablename;
        originDatabase = get_logicalDBName();
        get_DataStruct().set_Database(originDatabase);
        if(isTransactionCheck()) {
            DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_Handle()+", version: "+get_Version()+" \n");
           (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

        }

    }

    /**
     * Protected constructor used by createExisting(ObjectId, DBTransaction) above.
     *
     * @param id The ObjectId for the object.
     * @param dbTrans The current database transaction
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   Should never see this exception since GenericDO.ctor(ObjectId)
     *   never accesses the database.
     */
    protected <xsl:value-of select="CLASS_NAME"/>DO(String tablename, ObjectId id , DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        super( id );
        this.dbtablename = tablename;
        setTransaction(dbTrans);
        if(dbTrans!=null) {
           originDatabase = dbTrans.getDatabaseName();
        }
        if(originDatabase==null)
           originDatabase = get_logicalDBName();<xsl:if test="DO_IS_MULTIDB_BASED='true'">
        else {
           if(useLogicalDatabase(tablename, originDatabase)) {
             try {
              readCacheConfiguration(tablename, originDatabase);
            } catch (Exception ex) {
                DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default cache configuration for '"+originDatabase+"' database");
            }
         }
        } </xsl:if>
        get_DataStruct().set_Database(originDatabase);
        addToTransactionCache();
    }


<xsl:if test="DO_IS_MULTIDB_BASED='true'">
    /**
     * Protected constructor used by createExisting(dbName,ObjectId) above.
     *
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param id The ObjectId for the object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   Should never see this exception since GenericDO.ctor(ObjectId)
     *   never accesses the database.
     */
    protected <xsl:value-of select="CLASS_NAME"/>DO(String tablename, String dbName, ObjectId id )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        super(dbName,id);
        this.dbtablename = tablename;
        if (dbName == null)
            dbName = get_OriginDatabase();
        if(useLogicalDatabase(tablename, dbName)) {
          try {
            readCacheConfiguration(tablename, dbName);
          } catch (Exception ex) {
                DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default cache configuration for '"+dbName+"' database");
          }
         }
        get_DataStruct().set_Database(dbName);
        if(isTransactionCheck()) {
            DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_Handle()+", version: "+get_Version()+" \n");
            (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

        }
    }

</xsl:if>
<xsl:if test="IS_ABSTRACT='false'">

    /**
     * Represents table and cache (if there is caching) statistics.
     */
    // protected static Statistics statistics;

    /**
     * Get table statistics.
     *
     * @return Table statistics.
     */
    public static Statistics get_statistics(String tablename) {
        return getCache(tablename).getStatistics();
    }

    /**
     * Refresh table statistics.
     */
    public static void refreshStatistics(String tablename) {
       getCache(tablename).refreshStatistics();
    }
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
    /**
     * This set contains list of logical databases used by this <xsl:value-of select="CLASS_NAME"/>DO.
     */
    protected static HashSet usedDatabase;

    /**
     * Add logical database in set of used logical databases. Set of this databases is important for
     * refreshing cache in full caching mode. Tables from this databases will be refreshed.
     * This method returns information whether database already exists in set usedDatabase.
     *
     * @return True if database does not already exist in database set, otherwise false.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static synchronized boolean useLogicalDatabase(String tablename, String database)
    throws DatabaseManagerException, ObjectIdException {
        if (database == null)
            database = get_logicalDBName();
        if (usedDatabase == null)
            usedDatabase = new HashSet();
        boolean result = usedDatabase.add(database);
        if (result &amp;&amp; cache != null &amp;&amp; getCache(tablename).isFull() &amp;&amp; isFullCacheNeeded) {
            try {<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
                <xsl:value-of select="CLASS_NAME"/>Query query = new <xsl:value-of select="CLASS_NAME"/>Query (tablename, database);</xsl:if>
<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
                DBTransaction tmpTrans = DODS.getDatabaseManager().createTransaction(database);
                <xsl:value-of select="CLASS_NAME"/>Query query = new <xsl:value-of select="CLASS_NAME"/>Query (tablename, tmpTrans);</xsl:if>
                query.hitDatabase();
                <xsl:value-of select="CLASS_NAME"/>DO obj;
                <xsl:value-of select="CLASS_NAME"/>DataStruct objData;
                while ( null != ( obj = query.getNextDO() ) ) {
                    objData = (<xsl:value-of select="CLASS_NAME"/>DataStruct)obj.originalData_get();
                    if (objData != null) {
                       objData = addToCache( objData );
                    }
                } <xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
                tmpTrans.release();</xsl:if>
              } catch (DataObjectException e) {
                throw new DatabaseManagerException("INTERNAL ERROR: unexpected DataObjectException", e);<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
             } catch (SQLException e) {
                throw new DatabaseManagerException("INTERNAL ERROR: error in creating transaction", e); </xsl:if>
            } catch ( NonUniqueQueryException ex ) {
              throw new ObjectIdException( "Duplicate ObjectId" );
            }
        }
        return result;
    }

    /**
     * Get all used logical databases.
     *
     * @deprecated Use get_UsedLogicalDatabases()
     * @return Array that contains names of all used logical databases.
     *
     */
    public static String[] getUsedLogicalDatabases() {
        return get_UsedLogicalDatabases();
    }

    /**
     * Get all used logical databases.
     *
     * @return Array that contains names of all used logical databases.
     *
     */
    public static String[] get_UsedLogicalDatabases() {
        if (usedDatabase != null)
            return (String[])usedDatabase.toArray();
        return null;
    }

</xsl:if>

<xsl:if test="DO_IS_MULTIDB_BASED='false'">

    /**
     * Get all used logical databases.
     *
     * @deprecated Use get_UsedLogicalDatabases()
     * @return Array that contains names of all used logical databases.
     *
     */
    public static String[] getUsedLogicalDatabases() {
        return get_UsedLogicalDatabases();
    }

    /**
     * Get all used logical databases.
     *
     * @return Array that contains names of all used logical databases.
     *
     */
    public static String[] get_UsedLogicalDatabases() {
        String[] str = { get_logicalDBName() };
        return str;
    }</xsl:if>

    //GS: We have as many caches as we have accessed tables
    //protected static DataStructCache cache; // cache for <xsl:value-of select="CLASS_NAME"/>DO
	protected static java.util.Hashtable cachesTable = new Hashtable();
    protected static DataStructCache getCache(String tablename) {
		DataStructCache resultCache = (DataStructCache) cachesTable.get(tablename);
		if (resultCache==null) {
			resultCache = initCache(tablename);
		}
		return resultCache;
	}
    protected DataStructCache getCache() {
		DataStructCache resultCache = (DataStructCache) cachesTable.get(dbtablename);
		if (resultCache==null) {
			resultCache = initCache(this.dbtablename);
		}
		return resultCache;
	}
    protected static void setCache(String tablename, DataStructCache newCache) {
		if (newCache!=null) {
            // GS: Maybe we should test if the cache exists and sync it...
			cachesTable.put(tablename, newCache);
		}
	}


    protected static boolean isFullCacheNeeded = false; // it depends of CacheFullCacheCountLimit parameter and number of data in database.
    


    /**
     * Read cache configuration from application configuration file:
     * cache size for <xsl:value-of select="translate(PACKAGE,'/','.')"/>.<xsl:value-of select="TABLE_NAME"/> table or default cache size.
     * @param database DO's database.
     * @exception CacheObjectException
     */
    public static void readCacheConfiguration(String tablename, String database) throws CacheObjectException {
        if (getConfigurationAdministration(tablename).isDisabled()) {
            throw new CacheObjectException("Caching is disabled");
        }
        Config tableConfig = null;
        Config cacheConfig = null;
        try {
            tableConfig = (Config)DODS.getDatabaseManager().getConfig().getSection(&quot;DB.&quot;+database+&quot;.<xsl:value-of select="TABLE_NAME"/>&quot;);
        } catch (Exception ex) {
                    DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default configuration for '<xsl:value-of select="TABLE_NAME"/>' table");
        }
        try {
            cacheConfig = (Config)DODS.getDatabaseManager().getConfig().getSection(&quot;DB.&quot;+database+&quot;.<xsl:value-of select="TABLE_NAME"/>.cache&quot;);
        } catch ( Exception e ) {
            DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default cache configuration for '<xsl:value-of select="TABLE_NAME"/>' table");
        }
        <xsl:if test="DO_IS_MULTIDB_BASED='true'"></xsl:if>
        getCache(tablename).readConfiguration(tableConfig,cacheConfig, database);
    }


    /**
     * Get name of the table that is cached.
     *
     * @return Name of the table that is cached.
     */
    public static String getCacheDodsTableName() {
        return &quot;<xsl:value-of select="TABLE_NAME"/>&quot;;
    }


    /**
     * Returns <xsl:value-of select="TABLE_NAME"/> table cache.
     *
     * @return <xsl:value-of select="TABLE_NAME"/> table cache.
     */
    public ConfigurationAdministration getConfigurationAdministration() {
        return getCache();
    }
    public static ConfigurationAdministration getConfigurationAdministration(String tablename) {
        return getCache(tablename);
    }

    /**
     * Queries all rows in table, and for each row
     * creates a DO instance in the cache.
     * For these DOs, data.readOnly = true,
     * which causes set methods to throw an exception.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    public static void refreshCache(String tablename)
    throws java.sql.SQLException, DatabaseManagerException, ObjectIdException, DataObjectException {
       getConfigurationAdministration(tablename).getCacheAdministration(CacheConstants.DATA_CACHE).refresh();
       DataStructCache cache = getCache(tablename);
       String querySnt = cache.getInitialQueryCache();
       int maxSize = cache.getCacheAdministration(CacheConstants.DATA_CACHE).getMaxCacheSize();
        	 if (maxSize != 0) {
	        cache.checkFull();
           if(cache.isFull()) {
           if(cache.getTableConfiguration().getFullCacheCountLimit() == 0) {
               isFullCacheNeeded=false;
           } else if(cache.getTableConfiguration().getFullCacheCountLimit() &gt; 0) {
            int numOfRows = 0;
  	            <xsl:value-of select="CLASS_NAME"/>Query query;
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
	            if (usedDatabase != null) {
	                try {
	                    String database;
	                    for (Iterator i = usedDatabase.iterator(); i.hasNext();) {
	                        database = (String)i.next();<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
	                        query = new <xsl:value-of select="CLASS_NAME"/>Query (tablename, database);</xsl:if><xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
	                        DBTransaction tmpTrans = DODS.getDatabaseManager().createTransaction(database);
	                        query = new <xsl:value-of select="CLASS_NAME"/>Query (tablename, tmpTrans); </xsl:if>
	                        <xsl:value-of select="CLASS_NAME"/>DO obj;
	                        numOfRows = query.getCount();<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
	                        tmpTrans.release();</xsl:if>
	                    }
	                } catch (DatabaseManagerException e) {
	                    throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException", e);
	                } catch ( NonUniqueQueryException ex ) {
	                    // Since we do not call query.requireUniqueInstance()
	                    // this should never happen.
	                    throw new ObjectIdException( "Duplicate ObjectId" );
	                }
	            } else {
</xsl:if>
	                DBTransaction tmpTransaction = DODS.getDatabaseManager().createTransaction(get_logicalDBName());
	                query = new <xsl:value-of select="CLASS_NAME"/>Query (tablename, tmpTransaction);
	                try {
	                    numOfRows = query.getCount();
	                    tmpTransaction.release();
	                } catch ( NonUniqueQueryException ex ) {
	                    // Since we do not call query.requireUniqueInstance()
	                    // this should never happen.
	                    throw new ObjectIdException( "Duplicate ObjectId" );
	                }
	<xsl:if test="DO_IS_MULTIDB_BASED='true'">
	            }</xsl:if>

               if (numOfRows &lt; cache.getTableConfiguration().getFullCacheCountLimit())
                  isFullCacheNeeded = true;
               else
                  isFullCacheNeeded = false;      
               
           } else 
               isFullCacheNeeded=true;  
	        }
	        if (querySnt != null) {
	            <xsl:value-of select="CLASS_NAME"/>Query query;
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
	            if (usedDatabase != null) {
	                try {
	                    String database;
	                    for (Iterator i = usedDatabase.iterator(); i.hasNext();) {
	                        database = (String)i.next();<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
	                        query = new <xsl:value-of select="CLASS_NAME"/>Query (tablename, database);</xsl:if><xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
	                        DBTransaction tmpTrans = DODS.getDatabaseManager().createTransaction(database);
	                        query = new <xsl:value-of select="CLASS_NAME"/>Query (tablename, tmpTrans); </xsl:if>
	                        query.hitDatabase();
// tj 02.04.2004	                        maxSize = cache.getCacheAdministration(CacheConstants.DATA_CACHE).getMaxCacheSize();
                            	int initMaxSize = 0;
                            	if(cache.getInitialDSCacheSize()&gt; 0){
                                    initMaxSize=cache.getInitialDSCacheSize();
                	             }
                	            if((initMaxSize &gt; 0) &amp;&amp; ((initMaxSize &lt; maxSize)||(maxSize &lt; 0))) {
                	               isFullCacheNeeded = false;   
                              maxSize=initMaxSize;
                           }

	                        if (maxSize > 0)
	                            query.setMaxRows(maxSize);
	                        if (!querySnt.equals("*")) {
	                            QueryBuilder builder = query.getQueryBuilder();
	                            builder.addWhere(querySnt);
	                        }
	                        <xsl:value-of select="CLASS_NAME"/>DO obj;
      	                	if (cache.getInitialCacheFetchSize()>0){
	                              query.set_FetchSize(cache.getInitialCacheFetchSize());
	                      	}
	                      	
	                      	   query.set_CursorType(cache.getTableConfiguration().getInitCachesResultSetType(),cache.getTableConfiguration().getInitCachesResultSetConcurrency());                     	                            
	                            query.setLoadData(true);
	                            query.getNextDO();<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
	                        tmpTrans.release();</xsl:if>
	                    }
	                } catch (DatabaseManagerException e) {
	                    throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException", e);
	                } catch ( NonUniqueQueryException ex ) {
	                    // Since we do not call query.requireUniqueInstance()
	                    // this should never happen.
	                    throw new ObjectIdException( "Duplicate ObjectId" );
	                }
	            } else {
</xsl:if>
	                DBTransaction tmpTransaction = DODS.getDatabaseManager().createTransaction(get_logicalDBName());
	                query = new <xsl:value-of select="CLASS_NAME"/>Query (tablename, tmpTransaction);
	                query.hitDatabase();
// tj 02.04.2004	                maxSize = cache.getCacheAdministration(CacheConstants.DATA_CACHE).getMaxCacheSize();
                	int initMaxSize = 0;
                	if(cache.getInitialDSCacheSize()&gt; 0){
                        initMaxSize=cache.getInitialDSCacheSize();
                	}
                	if((initMaxSize &gt; 0) &amp;&amp; ((initMaxSize &lt; maxSize)||(maxSize &lt; 0))) {
                	   isFullCacheNeeded = false;   
                  maxSize=initMaxSize;
               }
	                if (maxSize > 0)
	                    try {
	                        query.setMaxRows(maxSize);
	                    }
	                    catch (NonUniqueQueryException nuEx){
	                        System.err.println ("NonUniqueQueryException in refreshCache() method : too many rows were      found.");
	                    }
	                if (!querySnt.equals("*")) {
	                    QueryBuilder builder = query.getQueryBuilder();
	                    builder.addWhere(querySnt);
	                }
	                <xsl:value-of select="CLASS_NAME"/>DO obj;
	                try {
	                	if (cache.getInitialCacheFetchSize()>0){
	                        query.set_FetchSize(cache.getInitialCacheFetchSize());
	                	}
	                	query.set_CursorType(cache.getTableConfiguration().getInitCachesResultSetType(),cache.getTableConfiguration().getInitCachesResultSetConcurrency());                     	                            
	                    query.setLoadData(true);
	                    query.getNextDO();
	                    tmpTransaction.release();
	                } catch ( NonUniqueQueryException ex ) {
	                    // Since we do not call query.requireUniqueInstance()
	                    // this should never happen.
	                    throw new ObjectIdException( "Duplicate ObjectId" );
	                }
	<xsl:if test="DO_IS_MULTIDB_BASED='true'">
	            }</xsl:if>
	        }
	     }
	     cache.refreshStatistics();
    }

    private static boolean isDisabledCaching = false;

    /**
     * Disable caching.
     * @exception java.sql.SQLException
     * @exception DatabaseManagerException
     * @exception ObjectIdException
     * @exception DataObjectException
     */
    public void disableCaching()
    throws java.sql.SQLException, DatabaseManagerException, ObjectIdException, DataObjectException {
        isDisabledCaching = true;
        getConfigurationAdministration().getCacheAdministration(CacheConstants.DATA_CACHE).disable();
    }

    /**
     * Enable caching.
     * @exception java.sql.SQLException
     * @exception DatabaseManagerException
     * @exception ObjectIdException
     * @exception DataObjectException
     */
    public void enableCaching()
    throws java.sql.SQLException, DatabaseManagerException, ObjectIdException, DataObjectException {
        if (isDisabledCaching){
            getConfigurationAdministration().getCacheAdministration(CacheConstants.DATA_CACHE).enable();
            DataStructCache cache = getCache();
            int maxSize = cache.getCacheAdministration(CacheConstants.DATA_CACHE).getMaxCacheSize();
            if (maxSize != 0){
           if(cache.isFull()) {
           if(cache.getTableConfiguration().getFullCacheCountLimit() == 0) {
               isFullCacheNeeded=false;
           } else if(cache.getTableConfiguration().getFullCacheCountLimit() &gt; 0) {
            int numOfRows = 0;
  	            <xsl:value-of select="CLASS_NAME"/>Query query;
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
	            if (usedDatabase != null) {
	                try {
	                    String database;
	                    for (Iterator i = usedDatabase.iterator(); i.hasNext();) {
	                        database = (String)i.next();<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
	                        query = new <xsl:value-of select="CLASS_NAME"/>Query (dbtablename, database);</xsl:if><xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
	                        DBTransaction tmpTrans = DODS.getDatabaseManager().createTransaction(database);
	                        query = new <xsl:value-of select="CLASS_NAME"/>Query (dbtablename, tmpTrans); </xsl:if>
	                        <xsl:value-of select="CLASS_NAME"/>DO obj;
	                        numOfRows = query.getCount();<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
	                        tmpTrans.release();</xsl:if>
	                    }
	                } catch (DatabaseManagerException e) {
	                    throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException", e);
	                } catch ( NonUniqueQueryException ex ) {
	                    // Since we do not call query.requireUniqueInstance()
	                    // this should never happen.
	                    throw new ObjectIdException( "Duplicate ObjectId" );
	                }
	            } else {
</xsl:if>
	                DBTransaction tmpTransaction = DODS.getDatabaseManager().createTransaction(get_logicalDBName());
	                query = new <xsl:value-of select="CLASS_NAME"/>Query (dbtablename, tmpTransaction);
	                try {
	                    numOfRows = query.getCount();
	                    tmpTransaction.release();
	                } catch ( NonUniqueQueryException ex ) {
	                    // Since we do not call query.requireUniqueInstance()
	                    // this should never happen.
	                    throw new ObjectIdException( "Duplicate ObjectId" );
	                }
	<xsl:if test="DO_IS_MULTIDB_BASED='true'">
	            }</xsl:if>

               if (numOfRows &lt; cache.getTableConfiguration().getFullCacheCountLimit())
                  isFullCacheNeeded = true;
               else
                  isFullCacheNeeded = false;      
               
           } else 
               isFullCacheNeeded=true;  
	        }
            
            String querySnt = cache.getInitialQueryCache();
            if (querySnt != null) {
	                <xsl:value-of select="CLASS_NAME"/>Query query;
	    <xsl:if test="DO_IS_MULTIDB_BASED='true'">
	                if (usedDatabase != null) {
	                    try {
	                        String database;
	                        for (Iterator i = usedDatabase.iterator(); i.hasNext();) {
	                            database = (String)i.next();<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
	                            query = new <xsl:value-of select="CLASS_NAME"/>Query (dbtablename, database);</xsl:if><xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
	                            DBTransaction tmpTrans = DODS.getDatabaseManager().createTransaction(database);
	                            query = new <xsl:value-of select="CLASS_NAME"/>Query (dbtablename, tmpTrans); </xsl:if>
	                            query.hitDatabase();
                                 	 int initMaxSize = 0;
                	                if(cache.getInitialDSCacheSize()&gt; 0){
                                          initMaxSize=cache.getInitialDSCacheSize();
                	                }
                	                if((initMaxSize &gt; 0) &amp;&amp; ((initMaxSize &lt; maxSize)||(maxSize&lt;0))){
                	                  isFullCacheNeeded = false;   
                                 maxSize=initMaxSize;
                               }

                               if (maxSize > 0)
	                                 query.setMaxRows(maxSize);
	                       	   if (cache.getInitialCacheFetchSize()>0){
                                         query.set_FetchSize(cache.getInitialCacheFetchSize());
                                       } 
                                       query.set_CursorType(cache.getTableConfiguration().getInitCachesResultSetType(),cache.getTableConfiguration().getInitCachesResultSetConcurrency());                     	                            
	                            query.setLoadData(true);
	                            if (!querySnt.equals("*")) {
	                                QueryBuilder builder = query.getQueryBuilder();
	                                builder.addWhere(querySnt);
	                            }
	                            <xsl:value-of select="CLASS_NAME"/>DO obj;
	                            query.getNextDO();<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
	                            tmpTrans.release();</xsl:if>
	                        }
	                    } catch (DatabaseManagerException e) {
	                        throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException", e);
	                    } catch ( NonUniqueQueryException ex ) {
	                        // Since we do not call query.requireUniqueInstance()
	                        // this should never happen.
	                        throw new ObjectIdException( "Duplicate ObjectId" );
	                    }
	                } else {
	    </xsl:if>
	                    DBTransaction tmpTransaction = DODS.getDatabaseManager().createTransaction(get_logicalDBName());
	                    query = new <xsl:value-of select="CLASS_NAME"/>Query (dbtablename, tmpTransaction);
	                    query.hitDatabase();
                           	int initMaxSize = 0;
                         	if(cache.getInitialDSCacheSize()&gt; 0){
        	                 	initMaxSize=cache.getInitialDSCacheSize();
                	         }
                	         if((initMaxSize &gt; 0) &amp;&amp; ((initMaxSize &lt; maxSize)||(maxSize &lt; 0))){
                	            isFullCacheNeeded = false;
                           maxSize=initMaxSize;
                        }

     	                 if (maxSize > 0)
	                     try {
	                        query.setMaxRows(maxSize);
	                     }
	                     catch (NonUniqueQueryException nuEx){
	                        System.out.println ("NonUniqueQueryException in enableCache() method : too many rows were found.");
	                     }

	                    if (!querySnt.equals("*")) {
	                        QueryBuilder builder = query.getQueryBuilder();
	                        builder.addWhere(querySnt);
	                    }
	                    <xsl:value-of select="CLASS_NAME"/>DO obj;
	                    try {
	                       	if (cache.getInitialCacheFetchSize()>0){
                                query.set_FetchSize(cache.getInitialCacheFetchSize());
                        	}
                        	query.set_CursorType(cache.getTableConfiguration().getInitCachesResultSetType(),cache.getTableConfiguration().getInitCachesResultSetConcurrency());                     	                            
	                        query.setLoadData(true);
	                        query.getNextDO();
	                        tmpTransaction.release();
	                    } catch ( NonUniqueQueryException ex ) {
	                        // Since we do not call query.requireUniqueInstance()
	                        // this should never happen.
	                        throw new ObjectIdException( "Duplicate ObjectId" );
	                    }
	    <xsl:if test="DO_IS_MULTIDB_BASED='true'">
	                }</xsl:if>
	            }
	         }
        }
    }
    private static CachedDBTransaction _tr_(DBTransaction dbt) {
        return (CachedDBTransaction)dbt;
    }

    /**
     * Add DO to cache.
     * If DO already exists in cache, just the data member is replaced,
     * so that application references to the DO remain valid.
     * If there  is no caching newDO object is returned.
     *
     * @param newDO Data object that will be added to cache.
     * @return Data object added to cache.
     */
    private <xsl:value-of select="CLASS_NAME"/>DO addToTransactionCache( <xsl:value-of select="CLASS_NAME"/>DO newDO ) {
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        if(get_transaction()!=null &amp;&amp; _tr_(get_transaction()).getTransactionCache()!=null) {
            ret = (<xsl:value-of select="CLASS_NAME"/>DO)_tr_(get_transaction()).getTransactionCache().addDO(newDO);
        }
        if (ret == null)
            return newDO;
        return ret;
    }

    /**
     * Add DataStruct object to cache.
     * If there is no caching newDO object is returned.
     *
     * @param newDS DataStruct object that will be added to cache.
     * @return DataStruct object added to cache.
     */
    public static synchronized <xsl:value-of select="CLASS_NAME"/>DataStruct addToCache( String tablename, <xsl:value-of select="CLASS_NAME"/>DataStruct newDS ) {
        <xsl:value-of select="CLASS_NAME"/>DataStruct ret = (<xsl:value-of select="CLASS_NAME"/>DataStruct)getCache(tablename).addDataStruct(newDS);
        if (ret == null)
            return newDS;
        return ret;
    }

    /**
     * Add DataStruct object to cache.
     * If there is no caching newDO object is returned.
     *
     * @param newDS DataStruct object that will be added to cache.
     * @return DataStruct object added to cache.
     */
    public synchronized <xsl:value-of select="CLASS_NAME"/>DataStruct addToCache( <xsl:value-of select="CLASS_NAME"/>DataStruct newDS ) {
        <xsl:value-of select="CLASS_NAME"/>DataStruct ret = (<xsl:value-of select="CLASS_NAME"/>DataStruct)getCache().addDataStruct(newDS);
        if (ret == null)
            return newDS;
        return ret;
    }

    /**
     * Add DO's original data object to cache.
     */
    public void addToCache() {
        addToCache((<xsl:value-of select="CLASS_NAME"/>DataStruct)this.originalData_get());
    }

    /**
     * Add DO to cache.
     * If DO already exists in cache, just the data member is replaced,
     * so that application references to the DO remain valid.
     * This method overides method addToCache of the class CoreDO.
     */
    private void addToTransactionCache() {
        addToTransactionCache(this);
    }


     /**
     * UpdateCache for given DataStruct object.
     *
     * @param updDS DataStruct object
     * @return Updated or inserted DataStruct object.
     */
    public synchronized <xsl:value-of select="CLASS_NAME"/>DataStruct updateCache( <xsl:value-of select="CLASS_NAME"/>DataStruct updDS) {
        <xsl:value-of select="CLASS_NAME"/>DataStruct ret = (<xsl:value-of select="CLASS_NAME"/>DataStruct)getCache().updateDataStruct(updDS);
        if (ret == null)
            return updDS;
        return ret;
    }

    /**
     * Update Cache.
     */
    public void updateCache() {
        updateCache((<xsl:value-of select="CLASS_NAME"/>DataStruct)this.originalData_get());
    }

    /**
     * Delete DataStruct object from cache
     *
     * @param data DataStruct object for deleting
     *
     * @return Deleted DataStruct object
     */
    public static synchronized <xsl:value-of select="CLASS_NAME"/>DataStruct deleteFromCache(String tablename, <xsl:value-of select="CLASS_NAME"/>DataStruct data ) {
        getCache(tablename).deleteDataStruct(data);
        return data;
    }


    /**
     * Remove DataStruct object from cache.
     *
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DataStruct object will be removed.
     * @param handle Handle of DataStruct object which will be re moved.
     */
    public static synchronized void removeFromCache(String tablename, String dbName, String handle) {
        String cacheHandle = dbName+"."+handle;
        getCache(tablename).removeDataStruct( cacheHandle );
    }

    /**
     * Delete object from cache
     */
    public void deleteFromCache() {
        deleteFromCache(this.dbtablename, (<xsl:value-of select="CLASS_NAME"/>DataStruct)this.originalData_get());
    }


    /**
     * Remove DataStruct from cache.
     *
     * @param data DataStruct object which will be removed.
     *
     * @return Removed DataStruct object or null if DataStruct object doesn't
     * exist in the cache.
     */
    public static synchronized <xsl:value-of select="CLASS_NAME"/>DataStruct removeFromCache( String tablename, <xsl:value-of select="CLASS_NAME"/>DataStruct data ) {
        return (<xsl:value-of select="CLASS_NAME"/>DataStruct)getCache(tablename).removeDataStruct(data);
    }

    /**
     * Remove DataStruct from cache.
     */
    public void evict() {
        if (!isPersistent())
            removeFromCache(this.dbtablename, (<xsl:value-of select="CLASS_NAME"/>DataStruct)this.originalData_get());
    }

    /**
     * Remove DataStruct objects from cache.
     *
     * @param DSs Array of DataStruct objects which will be removed from cache.
     */
    public static void evict(String tablename, <xsl:value-of select="CLASS_NAME"/>DataStruct[] DSs) {
        for (int i=0; i&lt;DSs.length; i++)
            removeFromCache(tablename, (<xsl:value-of select="CLASS_NAME"/>DataStruct) DSs[i]);
    }

    /**
     * Remove DataStruct objects from cache.
     *
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DataStruct object will be removed.
     * @param handles array of DataStruct object handles that will be removed
     * from cache.
     */
    public static void evict(String tablename, String dbName, String[] handles) {
        if (handles!=null) {
            for (int i=0; i&lt;handles.length; i++)
                removeFromCache(tablename, dbName, handles[i]);
        }
    }

    /**
     * The cache is used automatically by the class.
     * Callers of the create() methods do not know whether
     * the returned DO instance is shared or not.
     * This prevents the existence of multiple instances in memory
     * of the same database object.
     * The cache is instantiated when the first DO instance is created.
     */

    static {
            String dbName = get_logicalDBName();
        	useOrderedWithTable = null;
        	try {
                String orderedResultSetStr = ((StandardLogicalDatabase)DODS.getDatabaseManager()
                        	                  .findLogicalDatabase(dbName)).getDriverProperty(Common.VENDOR_ORDERED_RESULT_SET);
                if (orderedResultSetStr!=null){
                	if (orderedResultSetStr.equalsIgnoreCase(&quot;oldStyle&quot;)){
                	}else if (orderedResultSetStr.equalsIgnoreCase(&quot;withPrefix&quot;)){
                        useOrderedWithTable = new Boolean(true);
                	}else if (orderedResultSetStr.equalsIgnoreCase(&quot;noPrefix&quot;)){
                        useOrderedWithTable = new Boolean(false);
                	}else{
                        DODS.getLogChannel().write(Logger.DEBUG,&quot;<xsl:value-of select="CLASS_NAME"/>DO : Invalid value for OrderedResultSet parameter. Using default. &quot;);
                	}
                }
        	} catch (DatabaseManagerException e){
                DODS.getLogChannel().write(Logger.DEBUG,&quot;<xsl:value-of select="CLASS_NAME"/>DO : Unable to read configuration for OrderedResultSet. Using default. &quot;);
        	}
    }
    /**
     * Class that contains unchanging (static) data from the database
     * will have a cache of DOs representing the entire contents of the table.
     */
    // GS: this is not a static initialization anymore: static {
    protected static DataStructCache initCache(String tablename) {
 	   DataStructCache cache = null;
       try {
        	String dbName = get_logicalDBName();

            XMLConfig dodsConf = Common.getDodsConf();
            String cacheClassPath = null;
            String cacheClassName = null;
              
        	String queryCacheImplClass = 
                ((StandardLogicalDatabase)DODS.getDatabaseManager().findLogicalDatabase(dbName)).getDatabaseConfiguration().getQueryCacheImplClass();              
            try {
            	if (queryCacheImplClass==null){
                	cacheClassPath = dodsConf.getText("CacheJar");
                	cacheClassName = dodsConf.getText("CacheClassName");
                	if (cacheClassPath != null &amp;&amp; cacheClassName != null) {
                        MultiClassLoader loader = new MultiClassLoader(null);
                        loader.setClassPath(cacheClassPath);
                        Class cacheClass = loader.loadClass(cacheClassName);
						cache = (DataStructCache)cacheClass.newInstance();
						cache = cache.newInstance();
                	} else  {
						cache = new QueryCacheImpl();
                	}
            	}else{
                	Class cacheClass = Class.forName(queryCacheImplClass);
					cache = (DataStructCache)cacheClass.newInstance();
					cache = cache.newInstance();
            	}
            } catch ( Exception e ) {}
                        
           if (cache == null) {
                cache = new QueryCacheImpl();
            }
            
            setCache(tablename, cache);

            readCacheConfiguration(tablename, get_logicalDBName());
            get_statistics(tablename); // set statistics
            refreshCache(tablename);
        } catch ( Exception e ) {
            // cannot throw from static block
        }
		return cache;
    }

    /**
     * This method is invoked whenever object needs to be loaded from database.
     *
     * @exception DataObjectException If a data access error occurs.
     */
    public void refresh() throws DataObjectException {
        try {
            loadData();
        } catch ( Exception e ) {
            throw new DataObjectException(&quot;Unable to load data for <xsl:value-of select="CLASS_NAME"/>DO id=&quot; + get_OId() +
                                          &quot;, error = &quot;, e);
        }
    }

    /**
     * This method is invoked whenever objects needs to be loaded from database.
     *
     * @param DOs Array of DOs which will be red from database.
     *
     * @exception DataObjectException If a data access error occurs.
     */
    public static void refresh(<xsl:value-of select="CLASS_NAME"/>DO[] DOs) throws DataObjectException {
        for (int i=0; i&lt;DOs.length; i++)
            DOs[i].refresh();
    }


    /**
     * Refresh cache by removing from the cache results of the query
     * <i>querySnt</i>
     *
     * @param querySnt query used in this
     * @exception QueryException If a data access error occurs.
     */
         public void refresh(String querySnt)  throws QueryException {
                 try {
                   QueryBuilder qb = new QueryBuilder();
                   qb.select(this.getPrimaryKey(this.dbtablename));
                   qb.addWhere(querySnt);
         BigDecimal objId;
         String handle;
         String database = get_logicalDBName();
                   RDBRow row;
                   try {
                           while ( null != ( row = qb.getNextRow() ) ) {
                                objId = row.get( this.getPrimaryKey(this.dbtablename) ).getBigDecimal();
                                   handle = objId.toString();
                       removeFromCache(this.dbtablename, database, handle);
                           }
                   }catch ( Exception e ) {
                           throw new QueryException(&quot; Query Exception occured,&quot; );
                   }
         }catch (Exception ex){
                 System.out.println("Error in refresh(String) of DO object.");
            }
         }

<xsl:if test="/TABLE/GENERATE_INSECURE='true' and not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * createVirgin(String tablename)
     * Creates a DO that has no ObjectId or data.
     * Such a DO is used to insert a new database entry
     * after its data has been set.
     * @param tablename The name of the table in which the DO is to be created
     *
     * @return Created data object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     *
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createVirgin(String, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createVirgin(String tablename)
    throws DatabaseManagerException, ObjectIdException {
        return new <xsl:value-of select="CLASS_NAME"/>DO (tablename);
    }
</xsl:if>

<xsl:if test="/TABLE/GENERATE_INSECURE='true'">
    /**
     * createVirgin(String , DBTransaction)
     * @param tablename The name of the table in which the DO is to be created
     * @param dbTrans The current database transaction
     * @return Created data object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createVirgin(String tablename, DBTransaction dbTrans)
    throws DatabaseManagerException, ObjectIdException {
        return new <xsl:value-of select="CLASS_NAME"/>DO (tablename, dbTrans);
    }
</xsl:if>


<xsl:if test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true' and not(/TABLE/GENERATE_DIRTY='Omit')">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * createVirgin(String , org.webdocwf.dods.access.User)
     * Creates a DO that has no ObjectId or data.
     * Such a DO is used to insert a new database entry
     * after its data has been set.
     *
     * @param tablename The name of the table in which the DO is to be created
     * @param usr The user for security check.
     *
     * @return Created data object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception AccessException
     *   The user is not allowed to create new instances
     *
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createVirgin(String, User, DBTransaction) instead.
</xsl:if>
     *
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createVirgin(String tablename, org.webdocwf.dods.access.User usr )
    throws DatabaseManagerException, ObjectIdException, AccessException {
        assertDOCreateAccess( usr );
<xsl:if test=" /TABLE/GENERATE_INSECURE='true'">
        return createVirgin(tablename);
</xsl:if>
<xsl:if test=" /TABLE/GENERATE_INSECURE='false'">
        return new <xsl:value-of select="CLASS_NAME"/>DO (tablename);
</xsl:if>
    }
</xsl:if>

<xsl:if test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * createVirgin(String , DBTransaction, org.webdocwf.dods.access.User)
     *
     * @param tablename The name of the table in which the DO is to be created
     * @param usr The user for security check.
     * @param dbTrans The current database transaction
     * @return Created data object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception AccessException
     *   The user is not allowed to create new instances
     *
     * WebDocWf extension
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createVirgin(String tablename, DBTransaction dbTrans, org.webdocwf.dods.access.User usr)
    throws DatabaseManagerException, ObjectIdException, AccessException {
        assertDOCreateAccess( usr );
<xsl:if test=" /TABLE/GENERATE_INSECURE='true'">
        return createVirgin(tablename dbTrans);
</xsl:if>
<xsl:if test=" /TABLE/GENERATE_INSECURE='false'">
        return new <xsl:value-of select="CLASS_NAME"/>DO (tablename, dbTrans);
</xsl:if>
    }
</xsl:if>


</xsl:if>
<xsl:if test="DO_IS_MULTIDB_BASED='true' and /TABLE/GENERATE_INSECURE='true' and not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * createVirgin(String, String)
     * Creates a DO that has no ObjectId or data.
     * Such a DO is used to insert a new database entry
     * after its data has been set.
     *
     * @param tablename The name of the table in which the DO is to be created
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     *
     * @return Created data object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createVirgin(String,String,DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createVirgin(String tablename, String dbName)
    throws DatabaseManagerException, ObjectIdException {
        if (dbName == null)
            dbName = get_logicalDBName();
        return new <xsl:value-of select="CLASS_NAME"/>DO (tablename, dbName);
    }
</xsl:if>
<xsl:if test="DO_IS_MULTIDB_BASED='true' and DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true' and not(/TABLE/GENERATE_DIRTY='Omit')">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * createVirgin(String, String, org.webdocwf.dods.access.User)
     * Creates a DO that has no ObjectId or data.
     * Such a DO is used to insert a new database entry
     * after its data has been set.
     *
     * @param tablename The name of the table in which the DO is to be created
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param usr The user for security check.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception AccessException
     *   The user is not allowed to create new instances
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createVirgin(String, String, User, DBTransaction) instead.
</xsl:if>
     *
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createVirgin( String tablename, String dbName, org.webdocwf.dods.access.User usr )
    throws DatabaseManagerException, ObjectIdException, AccessException {
        if (dbName == null)
            dbName = get_logicalDBName();
        assertDOCreateAccess( usr );
<xsl:if test=" /TABLE/GENERATE_INSECURE='true'">
        return createVirgin(tablename, dbName);
</xsl:if>
<xsl:if test=" /TABLE/GENERATE_INSECURE='false'">
        return new <xsl:value-of select="CLASS_NAME"/>DO (tablename, dbName);
</xsl:if>
    }
</xsl:if>

<xsl:if test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
    /**
     * Ensure that the given user is allowed to create new instances.
     *
     * @param usr The user for security check.
     *
     * @exception AccessException If user is not allowed to create new instances.
     *
     * WebDocWf extension
     *
     */
    public static void assertDOCreateAccess( org.webdocwf.dods.access.User usr )
    throws AccessException {
        if (! hasDOCreateAccess( usr ))
            throw new AccessRightException("No access !", usr, "Get", null, &quot;<xsl:value-of select="translate(PACKAGE,'/','.')"/>.<xsl:value-of select="CLASS_NAME"/>DO&quot;, null, null, null, null, null);
    }

    /**
     * Check whether the given user is allowed to create new instances.
     *
     * @param usr The user for security check.
     *
     * @return Whether the given user is allowed to create new instances.
     *
     * WebDocWf extension
     *
     */
    public static boolean hasDOCreateAccess( org.webdocwf.dods.access.User usr )
    throws AccessEvalException {
        return usr.hasDOCreateAccess( &quot;<xsl:value-of select="translate(PACKAGE,'/','.')"/>.<xsl:value-of select="CLASS_NAME"/>DO&quot; );
    }
    // end of WebDocWf extension for DODS row instance security

</xsl:if>
<xsl:if test="DO_IS_OID_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * createExisting( String, BigDecimal )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed BigDecimal value as the primary key.
     *
     * Creates a DO that represents an existing entry in the database.
     * Such a DO is used to examine and possibly update such an entry.
     * createExisting() is called only from the code that retrieves
     * an ObjectId from a ResultSet (database query result).
     * createExisting() is protected because no other DO or BO should ever
     * need to call it.
     * FIX unfortunately the createExisting(BigDecimal) form *does* need
     * to be public because it is called by the public ctors of other DOs.
     * For example:
     * AaaDO contains a ref to a BbbDO,
     * so there is a method AaaDO.setBbb(BbbDO).
     * In the ctor AaaDO(ResultSet), we have the call
     * setBbb( BbbDO.createExisting( rs.getBigDecimal( &quot;bbb&quot;)));
     * Since AaaDO is not in the same package as BbbDO,
     * BbbDO.createExisting(BigDecimal) must be public, not protected.
     * Java needs the C++ 'friend' idea.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createExistring(String, BigDecimal, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, BigDecimal bd)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        return publicCreateExisting(tablename, null, bd, null, null);
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     *
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, BigDecimal bd)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (null == bd)
            return null;
        return ceInternal(tablename, new ObjectId(bd));
    }
</xsl:if>

    /**
     * createExisting( String, BigDecimal, DBTransaction )
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param dbTrans The current database transaction.
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, BigDecimal bd, DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        return publicCreateExisting(tablename, null, bd, null, dbTrans);
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param dbTrans The current database transaction.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     *
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, BigDecimal bd, DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (null == bd)
            return null;
        return ceInternal(tablename, new ObjectId(bd), dbTrans);
    }

    /**
     * createExisting(String, BigDecimal, HashMap, DBTransaction )
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param queryRefs HashMap Created referenced DO's (key datamaseName.Oid)
     * @param dbTrans The current database transaction.
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, BigDecimal bd, HashMap queryRefs, DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        return publicCreateExisting(tablename, null, bd, queryRefs, dbTrans);
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param queryRefs HashMap Created referenced DO's (key datamaseName.Oid)
     * @param dbTrans The current database transaction.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     *
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, BigDecimal bd, HashMap queryRefs, DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (null == bd)
            return null;
        return ceInternal(tablename, new ObjectId(bd), queryRefs, dbTrans);
    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * createExisting( String, String, BigDecimal )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed BigDecimal value as the primary key.
     *
     * Creates a DO that represents an existing entry in the database.
     * Such a DO is used to examine and possibly update such an entry.
     * createExisting() is called only from the code that retrieves
     * an ObjectId from a ResultSet (database query result).
     * createExisting() is protected because no other DO or BO should ever
     * need to call it.
     * FIX unfortunately the createExisting(BigDecimal) form *does* need
     * to be public because it is called by the public ctors of other DOs.
     * For example:
     * AaaDO contains a ref to a BbbDO,
     * so there is a method AaaDO.setBbb(BbbDO).
     * In the ctor AaaDO(ResultSet), we have the call
     * setBbb( BbbDO.createExisting( rs.getBigDecimal( &quot;bbb&quot;)));
     * Since AaaDO is not in the same package as BbbDO,
     * BbbDO.createExisting(BigDecimal) must be public, not protected.
     * Java needs the C++ 'friend' idea.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createExisting(BigDecimal, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, String dbName, BigDecimal bd)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        return publicCreateExisting(tablename, dbName, bd, null, null);
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param dbName Logical name of the database
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     *
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, BigDecimal bd)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (null == bd)
            return null;
        if (dbName == null)
            dbName = get_logicalDBName();
        return ceInternal(tablename, dbName,new ObjectId(bd));
    }

    /**
     * createExisting( String, String, BigDecimal, HashMap queryRefs )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed BigDecimal value as the primary key.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param dbTrans The current database transaction
     * @param queryRefs HashMap
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createExisting(String, String, BigDecimal, HashMap, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, String dbName, BigDecimal bd, HashMap queryRefs)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        return publicCreateExisting(tablename, dbName, bd, queryRefs, null);
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param dbTrans The current database transaction.
     * @param queryRefs HashMap
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     *
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, BigDecimal bd, HashMap queryRefs)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (null == bd)
            return null;
        return ceInternal(tablename, dbName, new ObjectId(bd), queryRefs);
    }

</xsl:if>
</xsl:if></xsl:if>

<xsl:if test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * createExisting(String, BigDecimal, org.webdocwf.dods.access.User )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed BigDecimal value as the primary key.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param usr The user for security check.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception AccessException
     *   The user is not allowed to read the instance existance
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createExisting(BigDecimal, User, DBTransaction) instead.
</xsl:if>
     *
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, BigDecimal bd, org.webdocwf.dods.access.User usr)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException, AccessException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = publicCreateExisting(tablename, null, bd, null, null);
        ret.assertDOGetAccess(usr);
        return ret;
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param usr The user for security check.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     *
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, BigDecimal bd, org.webdocwf.dods.access.User usr)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException, AccessException {
        if (null == bd)
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = ceInternal(tablename, new ObjectId(bd));
        ret.assertDOGetAccess(usr);
        return ret;
    }
</xsl:if>
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * createExisting(String, BigDecimal, org.webdocwf.dods.access.User , DBTransaction)
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param usr The user for security check.
     * @param dbTrans The current database transaction.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception AccessException
     *   The user is not allowed to read the instance existance
     *
     * WebDocWf extension
     *
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, BigDecimal bd, DBTransaction dbTrans, org.webdocwf.dods.access.User usr)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException, AccessException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = publicCreateExisting(tablename, null, bd, null, dbTrans);
        ret.assertDOGetAccess(usr);
        return ret;
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param dbTrans The current database transaction.
     * @param usr The user for security check.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     *
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, BigDecimal bd, DBTransaction dbTrans, org.webdocwf.dods.access.User usr)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException, AccessException {
        if (null == bd)
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = ceInternal(tablename, new ObjectId(bd), dbTrans);
        ret.assertDOGetAccess(usr);
        return ret;
    }
</xsl:if>
<xsl:if test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * createExisting( String, String, BigDecimal, org.webdocwf.dods.access.User )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed BigDecimal value as the primary key.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param usr The user for security check.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception AccessException
     *   The user is not allowed to read the instance existance
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createExisting(String, String, BigDecimal, User, DBTransaction) instead.
</xsl:if>
     *
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, String dbName, BigDecimal bd, org.webdocwf.dods.access.User usr)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException, AccessException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = publicCreateExisting(tablename, dbName, bd, null, null);
        ret.assertDOGetAccess(usr);
        return ret;
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param dbName Logical name of the database
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param usr The user for security check.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     *
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     * @exception ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, BigDecimal bd, org.webdocwf.dods.access.User usr)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException, AccessException {
        if (null == bd)
            return null;
        if (dbName == null)
            dbName = get_logicalDBName();
        <xsl:value-of select="CLASS_NAME"/>DO ret = ceInternal(tablename, dbName, new ObjectId(bd));
        ret.assertDOGetAccess(usr);
        return ret;
    }

</xsl:if></xsl:if>
<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Check whether the given user is allowed to read the DO existance.
     * Static function to be called without having the object.
     *
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param usr The user for security check.
     *
     * @return Whether the given user is allowed to read the DO existance
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use hasDOGetAccess( BigDecimal, User, DBTransaction) instead.
</xsl:if>
     * WebDocWf extension
     *
     */
    public static boolean hasDOGetAccess(String tablename, BigDecimal bd, org.webdocwf.dods.access.User usr)
    throws AccessEvalException {
        <xsl:value-of select="CLASS_NAME"/>DO newDO;
        try {
            newDO = ceInternal(tablename, new ObjectId(bd));
        } catch (Exception e) {
            throw new AccessEvalException("Error in hasDoGetAccess/createExisting !", e, usr, "Get", null,
                                          &quot;<xsl:value-of select="translate(PACKAGE,'/','.')"/>.<xsl:value-of select="CLASS_NAME"/>DO&quot;, null,null, null, null, null);
        }
        return usr.hasDOGetAccess(newDO);
    }
</xsl:if>

    /**
     * Check whether the given user is allowed to read the DO existance.
     * Static function to be called without having the object.
     *
     * @param bd The BigDecimal representation of the ObjectId for the object.
     * @param usr The user for security check.
     * @param dbTrans.
     *
     * @return Whether the given user is allowed to read the DO existance
     *
     * WebDocWf extension
     *
     */
    public static boolean hasDOGetAccess(String tablename, BigDecimal bd, DBTransaction dbTrans, org.webdocwf.dods.access.User usr)
    throws AccessEvalException {
        <xsl:value-of select="CLASS_NAME"/>DO newDO;
        try {
            newDO = ceInternal(tablename, new ObjectId( bd ) , dbTrans);
        } catch (Exception e) {
            throw new AccessEvalException("Error in hasDoGetAccess/createExisting !", e, usr, "Get", null,
                                          &quot;<xsl:value-of select="translate(PACKAGE,'/','.')"/>.<xsl:value-of select="CLASS_NAME"/>DO&quot;, null,null, null, null, null);
        }
        return usr.hasDOGetAccess(newDO);
    }
    // end of WebDocWf extension for DODS row instance security
</xsl:if>

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * createExisting(String, String )
     *
     * The createExisting method is used to create a &lt;CODE&gt;<xsl:value-of select="CLASS_NAME"/>DO&lt;/CODE&gt;
     * from a string handle.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param handle String representation of the ObjectId for the object.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createExisting(String, String, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, String handle) {
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        try {
            BigDecimal bd = new BigDecimal(handle);
            ret = publicCreateExisting(tablename, null, bd, null, null);
        } catch (Exception e) {
            DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n : Create existing failed");
        }
        return ret;
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param handle String representation of the ObjectId for the object.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String handle) {
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        try {
            BigDecimal bd = new BigDecimal(handle);
            ret = ceInternal(tablename, bd);
        } catch (Exception e) {
            DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n : Create existing failed");
        }
        return ret;
    }
</xsl:if>

    /**
     * createExisting(String, String, DBTransaction )
     *
     * The createExisting method is used to create a &lt;CODE&gt;<xsl:value-of select="CLASS_NAME"/>DO&lt;/CODE&gt;
     * from a string handle.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param handle String representation of the ObjectId for the object.
     * @param dbTrans The current database transaction.
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, String handle, DBTransaction dbTrans) {
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        try {
            BigDecimal bd = new BigDecimal(handle);
            ret = publicCreateExisting(tablename, null, bd, null, dbTrans);
        } catch (Exception e) {
            DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n : Create existing failed");
        }
        return ret;
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param handle String representation of the ObjectId for the object.
     * @param dbTrans The current database transaction.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String handle, DBTransaction dbTrans) {
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        try {
            BigDecimal bd = new BigDecimal(handle);
            ret = ceInternal(tablename, bd, dbTrans);
        } catch (Exception e) {
            DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n : Create existing failed");
        }
        return ret;
    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * createExisting(String, String, String )
     *
     * The createExisting method is used to create a &lt;CODE&gt;<xsl:value-of select="CLASS_NAME"/>DO&lt;/CODE&gt;
     * from a string handle.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param handle String representation of the ObjectId for the object.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createExisting(String, String, String, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, String dbName, String handle) {
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        try {
            BigDecimal bd = new BigDecimal(handle);
            ret = publicCreateExisting(tablename, dbName, bd, null, null);
        } catch (Exception e) {
            DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n : Create existing failed");
        }
        return ret;
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param dbName Logical name of the database
     * @param handle String representation of the ObjectId for the object.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, String handle) {
        if (dbName == null)
            dbName = get_logicalDBName();
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        try {
            BigDecimal bd = new BigDecimal(handle);
            ret = ceInternal(tablename, dbName,bd);
        } catch (Exception e) {
            DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n : Create existing failed");
        }
        return ret;
    }
</xsl:if></xsl:if>


<xsl:if test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * createExisting(String, String, org.webdocwf.dods.access.User )
     *
     * The createExisting method is used to create a &lt;CODE&gt;<xsl:value-of select="CLASS_NAME"/>DO&lt;/CODE&gt;
     * from a string handle.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param handle String representation of the ObjectId for the object.
     * @param usr The user for security check.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * WebDocWf extension
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createExisting(String, String, User, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, String handle,org.webdocwf.dods.access.User usr)
    throws AccessException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = createExisting(tablename, handle);
        if (null != ret) {
            ret.assertDOGetAccess(usr);
        }
        return ret;
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param handle String representation of the ObjectId for the object.
     * @param usr The user for security check.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String handle,org.webdocwf.dods.access.User usr)
    throws AccessException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = ceInternal(tablename, handle);
        if (null != ret) {
            ret.assertDOGetAccess(usr);
        }
        return ret;
    }

</xsl:if>
    // WebDocWf extension for DODS row instance security

    /**
     * createExisting(String, String, org.webdocwf.dods.access.User, DBTransaction )
     *
     * The createExisting method is used to create a &lt;CODE&gt;<xsl:value-of select="CLASS_NAME"/>DO&lt;/CODE&gt;
     * from a string handle.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param handle String representation of the ObjectId for the object.
     * @param usr The user for security check.
     * @param dbTrans The current database transaction
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * WebDocWf extension
     *
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, String handle, DBTransaction dbTrans, org.webdocwf.dods.access.User usr)
    throws AccessException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = createExisting(tablename, handle, dbTrans);
        if (null != ret) {
            ret.assertDOGetAccess(usr);
        }
        return ret;
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param handle String representation of the ObjectId for the object.
     * @param dbTrans The current database transaction
     * @param usr The user for security check.
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String handle, DBTransaction dbTrans, org.webdocwf.dods.access.User usr)
    throws AccessException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = ceInternal(tablename, handle, dbTrans);
        if (null != ret) {
            ret.assertDOGetAccess(usr);
        }
        return ret;
    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * createExisting(String,  String, String, org.webdocwf.dods.access.User )
     *
     * The createExisting method is used to create a &lt;CODE&gt;<xsl:value-of select="CLASS_NAME"/>DO&lt;/CODE&gt;
     * from a string handle.
     *
     * @param tablename, the table in which the objectis to be queried.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param handle String representation of the ObjectId for the object.
     * @param usr The user for security check.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * WebDocWf extension
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createExisting(String, String, String, User, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createExisting(String tablename, String dbName, String handle,org.webdocwf.dods.access.User usr)
    throws AccessException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = createExisting(tablename, dbName, handle);
        if (ret!=null) {
            ret.assertDOGetAccess(usr);
        }
        return ret;
    }

    /**
     * Method ceInternal is public, only to allow generated classes
     * to instantiate one another. Public modifier doesn't mean user
     * application could use it - it mustn't.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param dbName Logical name of the database
     * @param handle String representation of the ObjectId for the object.
     * @param usr The user for security check.
     *
     * @return instance of <xsl:value-of select="CLASS_NAME"/>DO or null
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, String handle, org.webdocwf.dods.access.User usr)
    throws AccessException {
        if (dbName == null)
            dbName = get_logicalDBName();
        <xsl:value-of select="CLASS_NAME"/>DO ret = ceInternal(tablename, dbName, handle);
        if (ret!=null) {
            ret.assertDOGetAccess(usr);
        }
        return ret;
    }
</xsl:if></xsl:if>

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Check whether the given user is allowed to read the DO existance
     * Static function to be called without having the object
     *
     * @param handle The string representation of the ObjectId for the object.
     * @param usr The user for security check.
     *
     * @return Whether the given user is allowed to read the DO existance
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use hasDOGetAccess( String, User, DBTransaction) instead.
</xsl:if>
     * WebDocWf extension
     *
     */
    public static boolean hasDOGetAccess(String tablename, String handle, org.webdocwf.dods.access.User usr)
    throws AccessEvalException {
        <xsl:value-of select="CLASS_NAME"/>DO newDO;
        try {
            newDO = ceInternal(tablename, handle);
        } catch (Exception e) {
            throw new AccessEvalException("Error in hasDoGetAccess/createExisting !", e, usr, "Get",null,
                                          &quot;<xsl:value-of select="translate(PACKAGE,'/','.')"/>.<xsl:value-of select="CLASS_NAME"/>DO&quot;, null, null, null, null, null);
        }
        return usr.hasDOGetAccess( newDO );
    }
</xsl:if>

    /**
     * Check whether the given user is allowed to read the DO existance
     * Static function to be called without having the object
     *
     * @param handle The string representation of the ObjectId for the object.
     * @param usr The user for security check.
     * @param dbTrans The current database transaction
     * @return Whether the given user is allowed to read the DO existance
     *
     * WebDocWf extension
     *
     */
    public static boolean hasDOGetAccess(String tablename, String handle, DBTransaction dbTrans, org.webdocwf.dods.access.User usr)
    throws AccessEvalException {
        <xsl:value-of select="CLASS_NAME"/>DO newDO;
        try {
            newDO = ceInternal(tablename, handle, dbTrans);
        } catch (Exception e) {
            throw new AccessEvalException("Error in hasDoGetAccess/createExisting !", e, usr, "Get",null,
                                          &quot;<xsl:value-of select="translate(PACKAGE,'/','.')"/>.<xsl:value-of select="CLASS_NAME"/>DO&quot;, null, null, null, null, null);
        }
        return usr.hasDOGetAccess( newDO );
    }
    // end of WebDocWf extension for DODS row instance security

</xsl:if>

    /**
     * createExisting( ObjectId , DBTransaction)
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed ObjectID value as the primary key.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param id The ObjectId for the object.
     * @param dbTrans The current database transaction.
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, ObjectId id, DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (null == id)
            return null;
        return ceInternal(tablename, id, null, dbTrans);
    }

    /**
     * ceInternal( ObjectId , HashMap queryRefs, DBTransaction)
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed ObjectID value as the primary key.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param id The ObjectId for the object.
     * @param queryRefs HashMap of available references.
     * @param dbTrans The current database transaction.
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, ObjectId id, HashMap queryRefs, DBTransaction dbTrans )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        
        boolean isNewDO = true;
        
        if (null == id)
            return null;
        String cacheHandle = get_logicalDBName()+"."+id.toString();
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        <xsl:value-of select="CLASS_NAME"/>DataStruct data = null;
        if (null == queryRefs) {
            queryRefs = new HashMap();
        }
        if (null!= dbTrans &amp;&amp; null!= _tr_(dbTrans).getTransactionCache()) {
            ret = (<xsl:value-of select="CLASS_NAME"/>DO)_tr_(dbTrans).getTransactionCache().getDOByHandle(cacheHandle);
            isNewDO=false;
        } else if (queryRefs.containsKey(cacheHandle)) {
            ret = (<xsl:value-of select="CLASS_NAME"/>DO)queryRefs.get(cacheHandle);
            isNewDO=false;
        }
        if (ret==null) {
            ret = (<xsl:value-of select="CLASS_NAME"/>DO)createDO (tablename, id, dbTrans);
        }
        if (ret.get_DataStruct().isEmpty) {
            data = findCachedObjectByHandle(tablename, cacheHandle);
            if (data != null)
               ret.originalData_set(data);
        }
        ret.setPersistent(true);  // mark DO as persistent (preexisting)

        if (queryRefs!=null) {
            ret.set_refs(queryRefs);
            ret.addRefs(cacheHandle, ret);
        }
        if (!getCache(tablename).getTableConfiguration().isLazyLoading() ) { // If not lazy-loading, fetch DO data now.
           if(ret.get_DataStruct().isEmpty)
            ret.loadData();
        } else {
            get_statistics(tablename).incrementLazyLoadingNum();
        }
        // unset the GenericDO.dirty flag.
        if(isNewDO)
         ret.markClean();

        return ret;
    }

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal(String, ObjectId )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed ObjectID value as the primary key.
     * @param tablename the name of the table.
     * @param id The ObjectId for the object.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(String, ObjectId, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, ObjectId id )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == id )
            return null;
        return ceInternal(tablename, id, null);
    }
</xsl:if>

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal(String, String, ObjectId , HashMap)
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed ObjectID value as the primary key.
     *
     * @param tablename the name of the table.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param id The ObjectId for the object.
     * @param queryRefs HashMap
     * @param dbTrans The current database transaction.
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(String, ObjectId, HashMap, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, ObjectId id, HashMap queryRefs)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (dbName == null)
            dbName = get_logicalDBName();
        if ( null == id )
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        <xsl:value-of select="CLASS_NAME"/>DataStruct data = null;
        String cacheHandle = dbName+"."+id.toString();
        if(queryRefs==null)
            queryRefs = new HashMap();
        if(queryRefs.containsKey(cacheHandle)) {
            ret = (<xsl:value-of select="CLASS_NAME"/>DO)queryRefs.get(cacheHandle);
            return ret;
        }
        data = (<xsl:value-of select="CLASS_NAME"/>DataStruct)findCachedObjectByHandle(tablename, cacheHandle );
        if ( data != null ){
            ret = (<xsl:value-of select="CLASS_NAME"/>DO)createDO (tablename, data.get_OId());
            if(ret!=null) {
                ret.originalData_set(data);
                ret.setPersistent(true);
            }
            return ret;
        }
        ret = new <xsl:value-of select="CLASS_NAME"/>DO( tablename, dbName,id );
        ret.setPersistent( true );  // mark DO as persistent (preexisting)
        if(queryRefs!=null) {
            ret.set_refs(queryRefs);
            ret.addRefs(cacheHandle, ret);
        }
        // changed by tj 15.06.2003.
        // before: if ( ! <xsl:value-of select="IS_LAZY_LOADING"/> ) { // If not lazy-loading, fetch DO data now.
        if (!getCache(tablename).getTableConfiguration().isLazyLoading()) { // If not lazy-loading, fetch DO data now.
            ret.loadData();
        } else {
            get_statistics(tablename).incrementLazyLoadingNum();
        }
        // unset the GenericDO.dirty flag.
        ret.markClean();
        return ret;
    }
</xsl:if>

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
   /**
     * ceInternal( String, ObjectId )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed ObjectID value as the primary key.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param id The ObjectId for the object.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(ObjectId, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, ObjectId id )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == id )
            return null;
        return ceInternal(tablename, dbName, id, null);
      }
</xsl:if></xsl:if>

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal( ResultSet )
     *
     * Factory method used to create an instance of this class to
     * represent a Data Object already existing in the database.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param rs The ResultSet returned by the Query class for
     * an existing Data Object stored in the database.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(ResultSet, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, ResultSet rs )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == rs )
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        if ( notUsingOId ) {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename);
            ret.initFromResultSet( rs );
        } else {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename, rs );
        }
        return ret;
    }
</xsl:if>

    /**
     * ceInternal( ResultSet, DBTransaction )
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param rs The ResultSet returned by the Query class for
     * an existing Data Object stored in the database.
     * @param dbTrans The current database transaction
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, ResultSet rs , DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == rs )
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        if ( notUsingOId ) {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename, dbTrans);
            ret.initFromResultSet( rs );
        } else {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO ( tablename, rs, dbTrans );
        }
        return ret;
    }

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal( ResultSet , HashMap)
     *
     * Factory method used to create an instance of this class to
     * represent a Data Object already existing in the database.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param rs The ResultSet returned by the Query class for
     * an existing Data Object stored in the database.
     * @param queryRefs list of created refernce objects.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(ResultSet, HashMap, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, ResultSet rs , HashMap queryRefs)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == rs )
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        if ( notUsingOId ) {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename);
            ret.set_refs(queryRefs);
            ret.initFromResultSet( rs );
        } else {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO ( tablename, rs, queryRefs );
        }
        return ret;
    }
</xsl:if>

    /**
     * ceInternal( ResultSet , HashMap, DBTransaction)
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param rs The ResultSet returned by the Query class for
     * an existing Data Object stored in the database.
     * @param queryRefs list of created refernce objects.
     * @param dbTrans The current database transaction
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, ResultSet rs , HashMap queryRefs, DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == rs )
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        if ( notUsingOId ) {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename, dbTrans);
            ret.set_refs(queryRefs);
            ret.initFromResultSet( rs );
        } else {
            if(queryRefs==null)
                queryRefs = new HashMap();
            BigDecimal tmpOid = rs.getBigDecimal(get_OIdColumnName());
            String cacheHandle = get_logicalDBName()+"."+tmpOid;
            if(queryRefs.containsKey(cacheHandle)) {
                ret = (<xsl:value-of select="CLASS_NAME"/>DO)queryRefs.get(cacheHandle);
                if (!ret.isLoaded()) {
                    ret.set_refs(queryRefs);
                    ret.initFromResultSet(rs);
                }
                return ret;
            }
            if (useOrderedWithTable != null){
            	ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename, new ObjectId(tmpOid),dbTrans);
            	ret.set_refs(queryRefs);
            	ret.initFromResultSet(rs);
            	//if(dbTrans!=null) dbTrans.lockDO(this);
            }else{
                ret = new <xsl:value-of select="CLASS_NAME"/>DO ( tablename, rs, queryRefs, dbTrans );
            }
        }
        return ret;
    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal( String, ResultSet )
     *
     * Factory method used to create an instance of this class to
     * represent a Data Object already existing in the database.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param rs The ResultSet returned by the Query class for
     * an existing Data Object stored in the database.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(ResultSet, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, ResultSet rs )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (dbName == null)
            dbName = get_logicalDBName();
        if ( null == rs )
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        if ( notUsingOId ) {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename, dbName);
            ret.initFromResultSet( rs );
        } else {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO ( tablename, dbName, rs );
        }
        return ret;
    }

</xsl:if>

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal( String, ResultSet , HashMap)
     *
     * Factory method used to create an instance of this class to
     * represent a Data Object already existing in the database.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param rs The ResultSet returned by the Query class for
     * an existing Data Object stored in the database.
     * @param queryRefs list of created refernce objects.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(ResultSet, HashMap, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, ResultSet rs , HashMap queryRefs)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == rs )
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        if ( notUsingOId ) {
            ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename, dbName);
            ret.set_refs(queryRefs);
            ret.initFromResultSet( rs );
        } else {
           if(queryRefs==null)
                queryRefs = new HashMap();
            String cacheHandle = get_logicalDBName()+"."+rs.getBigDecimal(get_OIdColumnName());
            if(queryRefs.containsKey(cacheHandle)) {
                ret = (<xsl:value-of select="CLASS_NAME"/>DO)queryRefs.get(cacheHandle);
                if (!ret.isLoaded()) {
                    ret.set_refs(queryRefs);
                    ret.initFromResultSet(rs);
                }
                return ret;
            }
            ret = new <xsl:value-of select="CLASS_NAME"/>DO ( tablename, dbName, rs );
        }
        return ret;
    }
</xsl:if>
</xsl:if>

<xsl:if test="DO_IS_OID_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal( RDBRow )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey value
     * in the passed RDBRow.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param row A row returned by QueryBuilder.getNextRow().
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the RDBRow does not contain a <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(RDBRow, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, RDBRow row )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == row )
            return null;
        RDBColumnValue pk = null;
        try {
            pk = row.get( <xsl:value-of select="CLASS_NAME"/>DO.getPrimaryKey(tablename) );
            return ceInternal( tablename, pk );
        } catch ( Exception e ) {
            throw new DataObjectException(&quot;Cannot create <xsl:value-of select="CLASS_NAME"/>DO, row does not &quot; +
                                          &quot;contain <xsl:value-of select="CLASS_NAME"/>DO primary key.&quot; );
        }
    }
</xsl:if>

    /**
     * ceInternal( RDBRow , DBTransaction)
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey value
     * in the passed RDBRow.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param row A row returned by QueryBuilder.getNextRow().
     * @param dbTrans The current database transaction.
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the RDBRow does not contain a <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, RDBRow row, DBTransaction dbTrans )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == row )
            return null;
        RDBColumnValue pk = null;
        try {
            pk = row.get( <xsl:value-of select="CLASS_NAME"/>DO.getPrimaryKey(tablename) );
            return ceInternal( tablename, pk, dbTrans );
        } catch ( Exception e ) {
            throw new DataObjectException(&quot;Cannot create <xsl:value-of select="CLASS_NAME"/>DO, row does not &quot; +
                                          &quot;contain <xsl:value-of select="CLASS_NAME"/>DO primary key.&quot; );
        }
    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal( String, RDBRow )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey value
     * in the passed RDBRow.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param RDBRow A row returned by QueryBuilder.getNextRow().
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the RDBRow does not contain a <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(RDBRow, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, RDBRow row )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (dbName == null)
            dbName = get_logicalDBName();
        if ( null == row )
            return null;
        RDBColumnValue pk = null;
        try {
            pk = row.get( <xsl:value-of select="CLASS_NAME"/>DO.getPrimaryKey(tablename) );
            return ceInternal( tablename, dbName,pk );
        } catch ( Exception e ) {
            throw new DataObjectException(&quot;Cannot create <xsl:value-of select="CLASS_NAME"/>DO, row does not &quot; +
                                          &quot;contain <xsl:value-of select="CLASS_NAME"/>DO primary key.&quot; );
        }
    }
</xsl:if></xsl:if>

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal( RDBColumnValue )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param pk a PrimaryKey column value from a row that was returned by
     * QueryBuilder.getNextRow().
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the RDBColumnValue does not contain a <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(RDBColumnValue, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, RDBColumnValue pk )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == pk )
            return null;
        if ( ! pk.equals( <xsl:value-of select="CLASS_NAME"/>DO.getPrimaryKey(tablename) ) )
            throw new DataObjectException(&quot;Cannot create <xsl:value-of select="CLASS_NAME"/>DO, &quot; +
                                          &quot;RDBColumnValue is not <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.&quot; );
        BigDecimal bd = null;
        try {
            bd = pk.getBigDecimal();
        } catch ( Exception e ) {
            throw new DataObjectException(&quot;Cannot create <xsl:value-of select="CLASS_NAME"/>DO, bad primary key.&quot; );
        }
        if ( null == bd )
            return null;
        return ceInternal( tablename, bd );
    }
</xsl:if>


    /**
     * ceInternal( RDBColumnValue, DBTransaction )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param pk a PrimaryKey column value from a row that was returned by
     * QueryBuilder.getNextRow().
     * @param dbTrans The current database transaction.
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the RDBColumnValue does not contain a <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, RDBColumnValue pk, DBTransaction dbTrans )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if ( null == pk )
            return null;
        if ( ! pk.equals( <xsl:value-of select="CLASS_NAME"/>DO.getPrimaryKey(tablename) ) )
            throw new DataObjectException(&quot;Cannot create <xsl:value-of select="CLASS_NAME"/>DO, &quot; +
                                          &quot;RDBColumnValue is not <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.&quot; );
        BigDecimal bd = null;
        try {
            bd = pk.getBigDecimal();
        } catch ( Exception e ) {
            throw new DataObjectException(&quot;Cannot create <xsl:value-of select="CLASS_NAME"/>DO, bad primary key.&quot; );
        }
        if ( null == bd )
            return null;
        return ceInternal( tablename, bd , dbTrans);
    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * ceInternal( String, RDBColumnValue )
     *
     * Factory method creates a <xsl:value-of select="CLASS_NAME"/>DO object by searching for it
     * in the database using the passed <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.
     *
     * @param tablename The name of the table in which the object is to be created.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param RDBColumnValue a PrimaryKey column value from a row
     * that was returned by QueryBuilder.getNextRow().
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception DataObjectException
     *   If the RDBColumnValue does not contain a <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.
     *   If the object is not found in the database.
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use ceInternal(RDBColumnValue, DBTransaction) instead.
</xsl:if>
     */
    protected static <xsl:value-of select="CLASS_NAME"/>DO ceInternal(String tablename, String dbName, RDBColumnValue pk )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (dbName == null)
            dbName = get_logicalDBName();
        if ( null == pk )
            return null;
        if ( ! pk.equals( <xsl:value-of select="CLASS_NAME"/>DO.getPrimaryKey(tablename) ) )
            throw new DataObjectException(&quot;Cannot create <xsl:value-of select="CLASS_NAME"/>DO, &quot; +
                                          &quot;RDBColumnValue is not <xsl:value-of select="CLASS_NAME"/>DO.PrimaryKey.&quot; );
        BigDecimal bd = null;
        try {
            bd = pk.getBigDecimal();
        } catch ( Exception e ) {
            throw new DataObjectException(&quot;Cannot create <xsl:value-of select="CLASS_NAME"/>DO, bad primary key.&quot; );
        }
        if ( null == bd )
            return null;
        return ceInternal(tablename, dbName,bd);
    }
</xsl:if></xsl:if></xsl:if>

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Creates a DO that has no ObjectId
     * but has a copy of an existing DO's data.
     * Such a DO is used to insert a new database entry
     * that is largely similar to an existing entry.
     *
     * @param tablename Name of the table in which the DO is to be created.
     * @param data The data struct to copy values from.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createCopy(String, <xsl:value-of select="CLASS_NAME"/>DataStruct, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createCopy(String tablename,  <xsl:value-of select="CLASS_NAME"/>DataStruct data )
    throws DatabaseManagerException, ObjectIdException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename, true);
        ret.originalData_set(data);
        return ret;
    }
</xsl:if>
    /**
     * Creates a DO that has no ObjectId
     * but has a copy of an existing DO's data.
     * Such a DO is used to insert a new database entry
     * that is largely similar to an existing entry.
     *
     * @param tablename Name of the table in which the DO is to be created.
     * @param data The data struct to copy values from.
     * @param dbTrans The current database transaction
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createCopy( String tablename, <xsl:value-of select="CLASS_NAME"/>DataStruct data, DBTransaction dbTrans )
    throws DatabaseManagerException, ObjectIdException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename, true, dbTrans);
        ret.originalData_set(data);
        ret.markClean();
        return ret;
    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Creates a DO that has no ObjectId
     * but has a copy of an existing DO's data.
     * Such a DO is used to insert a new database entry
     * that is largely similar to an existing entry.
     *
     * @param tablename Name of the table in which the DO is to be created.
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param data The data struct to copy values from.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createCopy(String, <xsl:value-of select="CLASS_NAME"/>DataStruct, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createCopy(String tablename, String dbName, <xsl:value-of select="CLASS_NAME"/>DataStruct data )
    throws DatabaseManagerException, ObjectIdException {
        if (dbName == null)
            dbName = get_logicalDBName();
        <xsl:value-of select="CLASS_NAME"/>DO ret = new <xsl:value-of select="CLASS_NAME"/>DO (tablename, dbName);
        ret.originalData_set(data);
        ret.markClean();
        return ret;
    }

</xsl:if>
</xsl:if>

    /**
     * Creates a DO that has no ObjectId
     * but has a copy of an existing DO's data.
     * Such a DO is used to insert a new database entry
     * that is largely similar to an existing entry.
     *
     * @param orig The original DO to copy.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createCopy( <xsl:value-of select="CLASS_NAME"/>DO orig )
    throws DatabaseManagerException, ObjectIdException {
        if (null == orig)
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = new <xsl:value-of select="CLASS_NAME"/>DO (orig.dbtablename, true);
        if (null != orig.originalData_get()) {
            ret.originalData_set(orig.originalData_get());
            ret.markClean();
            ret.transaction = orig.transaction;
            ret.setPersistent(orig.isPersistent());
        }
        return ret;
    }

   /**
     *
     * @param orig The original DO to copy.
     * @param dbTrans The current database transaction
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createCopy( <xsl:value-of select="CLASS_NAME"/>DO orig, DBTransaction dbTrans )
    throws DatabaseManagerException, ObjectIdException {
        if (null == orig)
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = new <xsl:value-of select="CLASS_NAME"/>DO (orig.dbtablename, true, dbTrans);
        if (null != orig.originalData_get()) {
            ret.originalData_set(orig.originalData_get());
            ret.markClean();
            ret.setPersistent(orig.isPersistent());
        }
        return ret;

    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Creates a DO that has no ObjectId
     * but has a copy of an existing DO's data.
     * Such a DO is used to insert a new database entry
     * that is largely similar to an existing entry.
     *
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param orig The original DO to copy.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createCopy(<xsl:value-of select="CLASS_NAME"/>DO, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createCopy( String dbName, <xsl:value-of select="CLASS_NAME"/>DO orig )
    throws DatabaseManagerException, ObjectIdException {
        if (dbName == null)
            dbName = get_logicalDBName();
        if (null == orig)
            return null;
        <xsl:value-of select="CLASS_NAME"/>DO ret = new <xsl:value-of select="CLASS_NAME"/>DO (orig.dbtablename, dbName);
        if (orig.isLoaded()) {
            ret.originalData_set(orig.get_DataStruct().duplicate());
            ret.markClean();
            ret.transaction = orig.transaction;
            ret.setPersistent(orig.isPersistent());
        }
        return ret;
    }

</xsl:if>
</xsl:if>

<xsl:if test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * Creates a DO that has no ObjectId
     * but has a copy of an existing DO's data.
     * Such a DO is used to insert a new database entry
     * that is largely similar to an existing entry.
     *
     * @param orig The original DO to copy.
     * @param usr The user for security check.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception AccessException
     *   If the user is not allowed to make a copy.
     *
     * WebDocWf extension
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createCopy( <xsl:value-of select="CLASS_NAME"/>DO orig, org.webdocwf.dods.access.User usr )
    throws DatabaseManagerException, ObjectIdException, AccessException {
        orig.assertDOCopyAccess ( usr );
        <xsl:value-of select="CLASS_NAME"/>DO ret = createCopy ( orig );
        return ret;
    }

    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**

     * @param orig The original DO to copy.
     * @param usr The user for security check.
     * @param dbTrans The current database transaction
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception AccessException
     *   If the user is not allowed to make a copy.
     *
     * WebDocWf extension
     *
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createCopy( <xsl:value-of select="CLASS_NAME"/>DO orig, DBTransaction dbTrans, org.webdocwf.dods.access.User usr )
    throws DatabaseManagerException, ObjectIdException, AccessException {
        orig.assertDOCopyAccess ( usr );
        <xsl:value-of select="CLASS_NAME"/>DO ret = createCopy ( orig, dbTrans );
        return ret;
    }

<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Creates a DO that has no ObjectId
     * but has a copy of an existing DO's data.
     * Such a DO is used to insert a new database entry
     * that is largely similar to an existing entry.
     *
     * @param dbName Logical name of the database from which
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     * @param orig The original DO to copy.
     * @param usr The user for security check.
     *
     * @return Created <xsl:value-of select="CLASS_NAME"/>DO object.
     *
     * @exception com.lutris.appserver.server.sql.ObjectIdException
     *   If an object's id can't be allocated for this object.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception AccessException
     *   If the user is not allowed to make a copy
     *
     * WebDocWf extension
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createCopy(String, <xsl:value-of select="CLASS_NAME"/>DO, User, DBTransaction) instead.
</xsl:if>
     */
    public static <xsl:value-of select="CLASS_NAME"/>DO createCopy( String dbName, <xsl:value-of select="CLASS_NAME"/>DO orig, org.webdocwf.dods.access.User usr )
    throws DatabaseManagerException, ObjectIdException, AccessException {
        if (dbName == null)
            dbName = get_logicalDBName();
        orig.assertDOCopyAccess ( usr );
        <xsl:value-of select="CLASS_NAME"/>DO ret = createCopy ( dbName,orig );
        return ret;
    }

</xsl:if>
    </xsl:if>
    // end of WebDocWf extension for DODS row instance security
</xsl:if>

<xsl:if test="DO_IS_OID_BASED='true'">
    /**
     * Causes the DO to refresh itself from the database
     * the next time a set or get method is called.
     */
    public void reload() {
        originalData = data = null;
    }
</xsl:if>

<xsl:if test="DO_IS_OID_BASED='true'">
    /**
     * The methods &lt;CODE&gt;
     *     getHandle
     *     hasMatchingHandle
<xsl:if test="DO_IS_OID_BASED='true'">
     *     findCachedObjectByHandle
</xsl:if>
     * &lt;/CODE&gt; are used by Presentation Objects that need to populate
     * HTML select lists with Data Objects as options.
     * The &lt;CODE&gt;getHandle()&lt;/CODE&gt; method is used
     * to set the value for each option,
     * and the &lt;CODE&gt;hasMatchingHandle()&lt;CODE&gt;
<xsl:if test="DO_IS_OID_BASED='false'">
     * and &lt;CODE&gt;findCachedObjectByHandle()&lt;CODE&gt;
</xsl:if>
     * methods are used to lookup the Data Object when the selection has
     * been made.
     *
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     *
     * @return id of this DO as a string
     *   If an object's id can't be allocated for this object.
     * @deprecated Use get_Handle() instead.
     */
    public String  getHandle()
    throws DatabaseManagerException {
        return get_Handle();
    }


    /**
     * The methods &lt;CODE&gt;
     *     get_Handle
     *     hasMatchingHandle
<xsl:if test="DO_IS_OID_BASED='true'">
     *     findCachedObjectByHandle
</xsl:if>
     * &lt;/CODE&gt; are used by Presentation Objects that need to populate
     * HTML select lists with Data Objects as options.
     * The &lt;CODE&gt;get_Handle()&lt;/CODE&gt; method is used
     * to set the value for each option,
     * and the &lt;CODE&gt;hasMatchingHandle()&lt;CODE&gt;
<xsl:if test="DO_IS_OID_BASED='false'">
     * and &lt;CODE&gt;findCachedObjectByHandle()&lt;CODE&gt;
</xsl:if>
     * methods are used to lookup the Data Object when the selection has
     * been made.
     *
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     *
     * @return id of this DO as a string
     *   If an object's id can't be allocated for this object.
     */
    public String  get_Handle()
    throws DatabaseManagerException {
        /*
        String ret = null;
        if ( null == get_OId() )
               throw new DatabaseManagerException( "ID not set " );
        ret = get_OId().toString();
        return ret;
        */
        if (null == __the_handle)
            throw new DatabaseManagerException( "ID not set " );
        return __the_handle;
    }

    /**
     * Returns cache handle.
     *
     * @return cache handle.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public String  get_CacheHandle()
    throws DatabaseManagerException {
        String ret = get_OriginDatabase() + "." + get_Handle();
        return ret;
    }
</xsl:if>

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Created DO with specified OID.
     *
     * @param tablename, DO's table.
     * @param oid DO's oid.
     * @return copy of DO (with the same id).
     *
     * @exception SQLException
     * @exception ObjectIdException
     * @exception DataObjectException
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createDO(String, ObjectId, DBTransaction) instead.
</xsl:if>
     */
    public static GenericDO createDO(String tablename, ObjectId oid) throws java.sql.SQLException, com.lutris.appserver.server.sql.ObjectIdException, com.lutris.dods.builder.generator.query.DataObjectException, com.lutris.appserver.server.sql.DatabaseManagerException{
        return new <xsl:value-of select="CLASS_NAME"/>DO(tablename, oid);
    }
</xsl:if>

    /**
     * Created DO with specified OID.
     *
     * @param tablename, DO's table.
     * @param oid DO's oid.
     * @param dbTrans The current database transaction.
     * @return copy of DO (with the same id).
     *
     * @exception SQLException
     * @exception ObjectIdException
     * @exception DataObjectException
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public static GenericDO createDO(String tablename, ObjectId oid, DBTransaction dbTrans) throws java.sql.SQLException, com.lutris.appserver.server.sql.ObjectIdException, com.lutris.dods.builder.generator.query.DataObjectException, com.lutris.appserver.server.sql.DatabaseManagerException{
        return new <xsl:value-of select="CLASS_NAME"/>DO(tablename, oid, dbTrans);
    }

<xsl:if test="DO_IS_OID_BASED='true'">
   /**
     * Compare string version of the id of this DO and handle.
     *
     * @param handle
     *   &lt;CODE&gt;String&lt;/CODE&gt; version of DO id.
     *
     * @return boolean
     *   True if the string version of the id of this DO matches passed handle.
     *
     * @see <xsl:value-of select="CLASS_NAME"/>DO#get_Handle() get_Handle
     */
    public boolean hasMatchingHandle( String handle ) {
        boolean ret = false;
        if (null == __the_handle) {
           return false;
        } else {
           //String thisHnadle = get_OId().toString();
           ret = __the_handle.equals( handle );
        }
        return ret;
    }
</xsl:if>

   /**
     * Get data object with key cacheHandle from the cache.
     *
     * @param cacheHandle
     *   &lt;CODE&gt;String&lt;/CODE&gt; version of concatenation of:
     * name of the data object's database, followed by '.', followed by
     * data object's id.
     *
     * @return &lt;CODE&gt;<xsl:value-of select="CLASS_NAME"/>DO&lt;/CODE&gt;
     *   Object if one is found in cache, otherwise null.
     *
     * @see <xsl:value-of select="CLASS_NAME"/>DO#get_Handle() get_Handle
     */
    public <xsl:value-of select="CLASS_NAME"/>DO findTransactionCachedObjectByHandle( String cacheHandle ){
        if ( null == cacheHandle )
            return null;
        if(get_transaction()!=null &amp;&amp; _tr_(get_transaction()).getTransactionCache()!= null)
            return (<xsl:value-of select="CLASS_NAME"/>DO)_tr_(get_transaction()).getTransactionCache().getDOByHandle( cacheHandle );
        else
            return null;
    }

     /**
     * Get DataStruct object with key cacheHandle from the cache.
     *
     * @param tablename
     * @param cacheHandle
     *   &lt;CODE&gt;String&lt;/CODE&gt; version of concatenation of:
     * name of the data object's database, followed by '.', followed by
     * data object's id.
     *
     * @return &lt;CODE&gt;<xsl:value-of select="CLASS_NAME"/>DataStruct&lt;/CODE&gt;
     *   Object if one is found in cache, otherwise null.
     *
     * @see <xsl:value-of select="CLASS_NAME"/>DO#get_Handle() get_Handle
     */
    public static <xsl:value-of select="CLASS_NAME"/>DataStruct findCachedObjectByHandle( String tablename, String cacheHandle ){
        if ( null == cacheHandle )
            return null;
        return ( <xsl:value-of select="CLASS_NAME"/>DataStruct ) getCache(tablename).getDataStructByHandle( cacheHandle );
    }

<xsl:if test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
     /**
     * Get data object with key cacheHandle from the cache.
     *
     * @param tablename
     * @param cacheHandle
     *   <CODE>String</CODE> version of concatenation of:
     * name of the data object's database, followed by '.', followed by
     * data object's id.
     * @param usr The user for security check
     *
     * @return <CODE><xsl:value-of select="CLASS_NAME"/>DO</CODE>
     *   Object if one is found in cache, otherwise null.
     *
     * WebDocWf extension
     */
    public <xsl:value-of select="CLASS_NAME"/>DO findTransactionCachedObjectByHandle( String cacheHandle,org.webdocwf.dods.access.User usr )
    throws AccessException {
        <xsl:value-of select="CLASS_NAME"/>DO ret = findTransactionCachedObjectByHandle(cacheHandle );
        if (ret != null)
            ret.assertDOGetAccess( usr );
        return ret;
    }

    /**
     * Get DataStruct object with key cacheHandle from the cache.
     *
     * @param tablename
     * @param cacheHandle
     *   <CODE>String</CODE> version of concatenation of:
     * name of the data object's database, followed by '.', followed by
     * data object's id.
     * @param usr The user for security check
     *
     * @return <CODE><xsl:value-of select="CLASS_NAME"/>DataStruct</CODE>
     *   Object if one is found in cache, otherwise null.
     *
     * WebDocWf extension
     */
    public static <xsl:value-of select="CLASS_NAME"/>DataStruct findCachedObjectByHandle(String tablename, String cacheHandle,org.webdocwf.dods.access.User usr )
    throws AccessException {
        <xsl:value-of select="CLASS_NAME"/>DataStruct obj = findCachedObjectByHandle(tablename, cacheHandle );
        <xsl:value-of select="CLASS_NAME"/>DO ret = null;
        if (obj != null) {
         try {
            DBTransaction tmpTransaction = DODS.getDatabaseManager().createTransaction(get_logicalDBName());
            ret = (<xsl:value-of select="CLASS_NAME"/>DO)createDO (tablename, obj.get_OId(), tmpTransaction);
                        ret.originalData_set(obj);
                        ret.setPersistent(true);
                        tmpTransaction.release();
         } catch (Exception ex) {}
            if(ret!=null)
                ret.assertDOGetAccess( usr );
         }
        return obj;
    }
    // end of WebDocWf extension for DODS row instance security
</xsl:if>

    /**
     * Assigns the DataStruct of an existing DO to this DO.
     * Does not duplicate data. Just assigns the reference.
     *
     * @param orig The original DO.
     *
     */
    protected void makeIdentical( <xsl:value-of select="CLASS_NAME"/>DO orig ) {
        super.makeIdentical(orig);
        originalData = orig.originalData;
        data = orig.data;
    }

    /**
     * Return Data object's version.
     * @return Data object's version.
     */
    public int getVersion() {
        return get_Version();
    }

    /**
     * get_Version makes the protected method public in CoreDO.
     *
     * @return Data object's version.
     */
    public int get_Version() {
        return (null != data)?data.get_Version():super.get_Version();
    }

    /**
     * getNewVersion overloaded
     *
     * @return Data object's version.
     * @deprecated get_NewVersion()
     */
    public int getNewVersion() {
        if (null != data)
            return data.get_Version();
        else
            return super.get_Version();
    }

    /**
     * get_NewVersion overloaded
     *
     * @return Data object's version.
     */
    public int get_NewVersion() {
        if (null != data)
            return data.get_Version();
        else
            return super.get_Version();
    }

    /**
     * setVersion overloaded.
     * @param _ver DO's version.
     */
    public void setVersion(int _ver) {
         set_Version(_ver);
    }

    /**
     * set_Version overloaded.
     *
     * @param _ver DO's version.
     */
    public void set_Version(int _ver) {
        if(_ver &lt; get_Version()) {
            new Throwable("WOW, ("+get_OId()+") oldVer:"+get_Version()+", new one is "+_ver).printStackTrace();
        } else if (null != data)
            data.set_Version(_ver);
        else
            super.set_Version(_ver);
    }

    /**
     * setNewVersion overloaded.
     *
     * @param _ver New DO's version.
     * @deprecated Use set_NewVersion
     */
    public void setNewVersion(int _ver){
        set_NewVersion(_ver);
    }

    /**
     * set_NewVersion overloaded.
     *
     * @param _ver Ignored.
     */
    public void set_NewVersion(int _ver){
        try {
            doTouch();
        } catch (Exception e) {
                DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" can't set new version ");
        }
    }

    // WebDocWf extension for writeable fully cached objects
    /**
     * Mark the object as read-only.
     *
     * WebDocWf extension
     *
     */
    public void makeReadOnly() {
        if (null != data) {
          try{
            checkDup();
          }catch(Exception ex) {
            DODS.getLogChannel().write(Logger.DEBUG, " MakeReadOnly failed: Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_OId()+", version: "+get_Version()+" \n");
            (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));
          }
        }
         data.readOnly = true;
    }

    /**
     * Mark the object as read-write.
     *
     * WebDocWf extension
     */
    public void makeReadWrite() {
        if (null != data) {
          try{
            checkDup();
          }catch(Exception ex) {
            DODS.getLogChannel().write(Logger.DEBUG, " MakeReadWrite failed: Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_OId()+", version: "+get_Version()+" \n");
            (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

          }
        }
         data.readOnly = false;
    }
    // end of WebDocWf extension for writeable fully cached objects

    // WebDocWf extension for writeable fully cached objects

    /**
     * Set reference objects.
     *
     * @param queryRefs Reference objects.
     */
    private void set_refs(HashMap queryRefs) {
        refs = queryRefs;
    }
    /**
     * Return Object with the handle <i>key</i> from reference objects HashMap.
     *
     * @param key DO's handle.
     * @return Object with the handle <i>key</i> from reference objects HashMap.
     */
    private Object getRefs(String key) {
        if(get_transaction()!=null &amp;&amp; _tr_(get_transaction()).getTransactionCache()!= null) {
            return _tr_(get_transaction()).getTransactionCache().getDOByHandle(key);
        } else if (null == refs) {
            refs = new HashMap();
            return null;
        } else {
            return refs.get(key);
        }
    }
    /**
     * Add Object <i>newRefs</i> with the handle <i>key</i> to
     * reference objects HashMap.
     *
     * @param key DO's handle.
     * @param newRefs Object to be added.
     */
    private void addRefs(String key, Object newRefs) {
        if (null == refs) {
            refs = new HashMap();
        }
        refs.put(key, newRefs);
    }

    /**
     * Get reference objects.
     * @return HashMap with reference objects.
     */
    private HashMap get_refs() {
        return refs;
    }

    /**
     * Returns this object's identifier.
     * @return this object's identifier.
     */
    public ObjectId getOId() {
        return get_OId();
    }

    /**
     * Returns this object's identifier.
     * @return this object's identifier.
     */
    public ObjectId get_OId() {
        return get_DataStruct().get_OId();
    }

    /**
     * Sets this object's identifier.
     * @deprecated Use set_OId()
     * @param _oId this object's identifier.
     */
    protected void setOId(ObjectId _oId) {
        set_OId(_oId);
    }

    private String __the_handle;

    /**
     * Sets this object's identifier.
     * @param _oId this object's identifier.
     */
    protected void set_OId(ObjectId _oId) {
        if (get_DataStruct() == null)
            originalData = new <xsl:value-of select="CLASS_NAME"/>DataStruct();
             get_DataStruct().set_OId(_oId);
        __the_handle = _oId.toString();
    }

    /**
     * Creates a clone of the object, but ensures that
     * a new and unique object id is created for the object
     * and that the version number is set to zero.
     *
     * @return Cloned object.
     * @exception DatabaseManagerException if an error occurs while
     * allocation a new object id from the default logical database.
     * @exception ObjectIdException if a new object id could not be
     * allocated.
     */
    public synchronized Object cloneUnique()
        throws DatabaseManagerException, ObjectIdException {

        <xsl:value-of select="CLASS_NAME"/>DO _clone = new <xsl:value-of select="CLASS_NAME"/>DO (dbtablename, get_transaction());

        try {
            checkLoad();
            <xsl:value-of select="CLASS_NAME"/>DataStruct toClone = (null != get_Data())
                     ?(<xsl:value-of select="CLASS_NAME"/>DataStruct)get_Data()
                     :(<xsl:value-of select="CLASS_NAME"/>DataStruct)originalData_get();
            if (null != toClone) {
                _clone.set_Data(toClone.duplicate());
                ((<xsl:value-of select="CLASS_NAME"/>DataStruct)_clone.get_Data()).set_OId(((<xsl:value-of select="CLASS_NAME"/>DataStruct)_clone.originalData_get()).get_OId());
                ((<xsl:value-of select="CLASS_NAME"/>DataStruct)_clone.get_Data()).set_Version(0);
                changedFlags_set(true);
            }
        } catch (Exception e) {
            DODS.getLogChannel().write(Logger.DEBUG," cloneUnique failed: Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_Handle()+", version: "+get_Version()+" \n");
           (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

        }
        return _clone;
    }

    protected boolean deleted;

    /**
     * Returns the value of delete tag.
     * @return true if DO has been deleted, but not commited yet
     */
    public boolean isDeleted() {
        return deleted;
    }

    /**
     * Sets the value of delete tag.
     * @param flag true if DO has been deleted, but not commited yet.
     */
    public void setDeleted(boolean flag) {
        deleted = flag;
    }


    /**
     * If transaction succeeded marks this object as clean.
     * @param success true if the transaction succeeded
     *   and this object was successfully inserted into the database.
     */
    public void finalizeInsert(boolean success) {
        // on rollback reject inserted (createdVirgin) DO
        if (!success &amp;&amp; !isPersistent())
            setDeleted(true);
        super.finalizeInsert(success);
        if (success)
            syncStructs(true);
    }

    /**
     * If transaction succeeded marks this object as clean.
     * @param success true if the transaction succeeded
     *   and this object was successfully updated in the database.
     */
    public void finalizeUpdate(boolean success) {
        super.finalizeUpdate(success);
        if (success)
            syncStructs(true);
    }

    /**
     * Currently does nothing.
     *
     * @param success true if the transaction succeeded
     *   and this object was successfully deleted from the
     *   database.
     */
    public void finalizeDelete(boolean success) {
        super.finalizeDelete(success);
        if (success) {
            deleteFromCache();
        }
    }

    /**
     * Method SyncStructs(boolean _updateCache).
     *
     * @param _updateCache boolean value.
     */
    private synchronized void syncStructs(boolean _updateCache) {
        if (null != data)
            originalData = data;
        data = null;
        changedFlags_set(false);
        ((<xsl:value-of select="CLASS_NAME"/>DataStruct)originalData).readOnly = true;
        if (_updateCache)
            updateCache();
    }

    /**
     * Returns information whether the DO is created virgin and hasn't been
     * commited yet.
     * @return true for DO that's created virgin and hasn't been commited yet.
     */
    public boolean isVirgin() {
        return !isPersistent();
    }

    /**
     * Make DO's data from cache visible.
     */
    public void makeVisible () {
        try {
            ((QueryCache)getCache()).makeVisible(get_CacheHandle());
        } catch (DatabaseManagerException dme) {
            System.err.println("makeVisible for "+super.toString()+"failed");
        }
    }

    /**
     * Make DO's  data from cache Invisible.
     */
    public void makeInvisible () {
        try {
            ((QueryCache)getCache()).makeInvisible(get_CacheHandle());
        } catch (DatabaseManagerException dme) {
            System.err.println("makeInvisible for "+super.toString()+"failed");
        }
    }

    /**
     * Inserts this object into the database.
     *
     * @param conn the database connection.
     * @exception java.sql.SQLException if a database access error occurs.
     * @exception DBRowUpdateException If a version error occurs.
     */
    public synchronized void executeInsert(DBConnection conn)
    throws SQLException, DBRowUpdateException {
        if (dirty) {
            super.executeInsert(conn);
            changedFlags_set(false);
        }
    }


    /**
     * Returns value for AutoSave.
     * @return true if AutoSave is on, otherwise false.
     */
    protected boolean isAutoSave()
    {
      boolean flag = false;
      try {
        flag = ((StandardLogicalDatabase)(DODS.getDatabaseManager().findLogicalDatabase(get_OriginDatabase()))).getDatabaseConfiguration().getAutoSave();
      } catch (Exception ex) {}
      return flag;
    }

    /**
     * Returns value for AutoSaveCreateVirgin.
     * @return true if AutoSaveCreateVirgin is on, otherwise false.
     */
    protected boolean isAutoSaveCreateVirgin()
    {
      boolean flag = false;
      try {
        flag = ((StandardLogicalDatabase)(DODS.getDatabaseManager().findLogicalDatabase(get_OriginDatabase()))).getDatabaseConfiguration().getAutoSaveCreateVirgin();
      } catch (Exception ex) {}
      return flag;
    }

    /**
     * Returns value for TransactionCheck.
     * @return true if TransactionCheck is on, otherwise false.
     */
    protected boolean isTransactionCheck()
    {
      boolean flag = false;
      try {
        flag = ((StandardLogicalDatabase)(DODS.getDatabaseManager().findLogicalDatabase(get_OriginDatabase()))).getDatabaseConfiguration().getTransactionCheck();
      } catch (Exception ex) {}
      return flag;
    }

    /**
     * Returns value for TransactionCaches.
     * @return true if TransactionCaches are on, otherwise false.
     */
    protected boolean isTransactionCaches()
    {
      boolean flag = false;
      try {
        flag = ((StandardLogicalDatabase)(DODS.getDatabaseManager().findLogicalDatabase(get_OriginDatabase()))).getDatabaseConfiguration().getTransactionCaches();
      } catch (Exception ex) {}
      return flag;
    }

    /**
     * Returns value for DeleteCheckVersion.
     * @return true if DeleteCheckVersion is on, otherwise false.
     */
    protected boolean isDeleteCheckVersion()
    {
      boolean flag = false;
      try {
        flag = ((StandardLogicalDatabase)(DODS.getDatabaseManager().findLogicalDatabase(this.get_OriginDatabase()))).getDatabaseConfiguration().getDeleteCheckVersion();
      } catch (Exception ex) {}
      return flag;
    }


    /**
     * Returns value for AllReadOnly.
     * @return true if AllReadOnly is on, otherwise false.
     */
    protected static boolean isAllReadOnly()
    {
      boolean flag = false;
      try {
        flag = ((StandardLogicalDatabase)(DODS.getDatabaseManager().findLogicalDatabase(get_logicalDBName()))).getDatabaseConfiguration().isAllReadOnly();
      } catch (Exception ex) {}
      return flag;
    }



     /**
     * Undo action.
     *
     * @exception DataObjectException
     */
    public void undo()throws com.lutris.dods.builder.generator.query.DataObjectException
    {
      try{
        if(null != transaction){
                if((data!=null) || (data==null &amp;&amp; isDeleted())){
                        int tempVersion=get_Version();
                                if(isDeleted() &amp;&amp; !isDeletedFromDatabase){
                                                unDelete(transaction);
                                }else if (isDeleted() &amp;&amp; isDeletedFromDatabase){
                                        data =((<xsl:value-of select="CLASS_NAME"/>DataStruct)originalData).duplicate();
                                        set_Version(tempVersion);
                                        persistent=false;
                                        deleted=false;
                                        isDeletedFromDatabase=false;
                                        if (isAutoSave()) {
                            save(transaction,false);
                        }
                                }else{
                                        data =((<xsl:value-of select="CLASS_NAME"/>DataStruct)originalData).duplicate();
                                        set_Version(tempVersion);
                                        if ( isAutoSave()) {
                            save(transaction,false);
                        }
                                }
                }
        }else{
                        throw new DataObjectException("Error during Undo operation");
                }
      }catch(Exception ex){
                throw new DataObjectException("Error during Undo operation");
      }

    }

    /**
     * dumpData action.
     *
     * @param incrementVersion Increment version.
     */
    public void dumpData(boolean incrementVersion) {
        data = null;
        ObjectId _oid = get_OId();
        int _version = (incrementVersion?1:0)+ get_Version();
        originalData_set(new <xsl:value-of select="CLASS_NAME"/>DataStruct());
        set_OId(_oid);
        set_Version(_version);
    }
    /**
     * publicCreateExisting action.
     *
     * @param _tablename Table name.
     * @param _dbName Database name.
     * @param _oid DO's oid.
     * @param _refs Reference objects.
     * @param dbt Current transaction.
     * @return Created DO or null if doesn't exist.
     *
     * @exception SQLException
     * @exception ObjectIdException
     * @exception DataObjectException
     * @exception DatabaseManagerException
     */
    private static <xsl:value-of select="CLASS_NAME"/>DO publicCreateExisting(String _tablename, String _dbName, BigDecimal _oid, HashMap _refs, DBTransaction dbt)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        if (null == _oid)
            return null;
        if (null == _dbName)
            _dbName = get_logicalDBName();
        <xsl:value-of select="CLASS_NAME"/>Query qry = new <xsl:value-of select="CLASS_NAME"/>Query(_tablename, dbt);
        qry.setLogicalDatabase(_dbName);
        if (null != _refs) {
            qry.setRefs(_refs);
        }
        qry.setQueryOId(new ObjectId(_oid));
        qry.requireUniqueInstance();
        try {
            return qry.getNextDO();
        } catch (com.lutris.dods.builder.generator.query.NonUniqueQueryException nuqe) {
            throw new DataObjectException("Query didn't give unique result.");
        }
    }
</xsl:template>
</xsl:stylesheet>
