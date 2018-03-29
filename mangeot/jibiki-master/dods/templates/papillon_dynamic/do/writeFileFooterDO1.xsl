<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="fileFooter1">
<xsl:value-of select="FK_GET_SET_METHODS"/>
    /**
     * logicalDbName is logical database name
     * set by setLogicalDBName()
     * and retrieved by get_logicalDBName().
     */
    static private String logicalDbName = null;

    /**
     * setLogicalDBName sets the logical database name that will be used
     * to create DBTransaction and DBQuery objects used by
     * <xsl:value-of select="CLASS_NAME"/>DO and the corresponding Query class.
     * 
     * @param logicalDbNameInConfFile The logical database specified in the
     * application's .conf file.
     *
     * @deprecated It is dangeruous to use this method in multiuser environment because, 
     * this setings are applied to all users (sets logical database to all users)
     */
    static public synchronized void setLogicalDBName( String logicalDbNameInConfFile ) {
        if ( null != logicalDbNameInConfFile &amp;&amp; 0    != logicalDbNameInConfFile.length() )
            logicalDbName = logicalDbNameInConfFile;
        else 
            logicalDbName = DODS.getDatabaseManager().getDefaultDB();
    }
    
    /**
     * get_logicalDBName retrieves the logical database name 
     * set by setLogicalDBName().
     *
     * @return the logical database name that was set by method setLogicalDBName()
     *
     */
    static public synchronized String get_logicalDBName() {
        if (logicalDbName == null)
            logicalDbName = DODS.getDatabaseManager().getDefaultDB();                    
        return logicalDbName;
    }

    /**
     * createTransaction() creates a new DBTransaction.
     * This method uses the logical database name set by method setLogicalDBName().
     *
     * If setLogicalDBName() was used to set the logical database name
     * to something other than the value of DatabaseManager.DefaultDatabase
     * in the application's .conf file, then any DBTransaction passed to 
     * save(DBTransaction) or delete(DBTransaction) should be created using 
     * <xsl:value-of select="CLASS_NAME"/>DO.createTransaction().
     *
     * The <xsl:value-of select="CLASS_NAME"/>DO save() and delete() methods use this method.
     *
     * @return A DBTransaction object to use with the <xsl:value-of select="CLASS_NAME"/>DO class.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    static public DBTransaction createTransaction() 
    throws DatabaseManagerException, SQLException {
        DBTransaction ret;
        try {
            ret = DODS.getDatabaseManager().createTransaction(get_logicalDBName());
            ret.setDatabaseName(get_logicalDBName());
            return ret;
        } catch ( DatabaseManagerException e ) { 
            String err = "";
            if ( null != get_logicalDBName() )
                err = &quot;ERROR: Could not create a DBTransaction.  &quot; +
                      &quot;<xsl:value-of select="CLASS_NAME"/>DO.logicalDbName='&quot; + get_logicalDBName() + &quot;'.  &quot;+
                      &quot;The application .conf file must list this name in &quot; +
                      &quot;DatabaseManager.Databases[], and there must be &quot; +
                      &quot;DatabaseManager.DB.&quot; + get_logicalDBName() + &quot; settings.&quot;;
            throw new DatabaseManagerException( err, e );
        }
    }
    
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
    /**
     * createTransaction() creates a new DBTransaction.
     * This method uses the logical database name set by method setLogicalDBName().
     *
     * If setLogicalDBName() was used to set the logical database name
     * to something other than the value of DatabaseManager.DefaultDatabase
     * in the application's .conf file, then any DBTransaction passed to 
     * save(DBTransaction) or delete(DBTransaction) should be created using 
     * <xsl:value-of select="CLASS_NAME"/>DO.createTransaction().
     *
     * The <xsl:value-of select="CLASS_NAME"/>DO save() and delete() methods use this method.
     *
     * @param dbName Logical name of the database from which 
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     
     * @return a DBTransaction object to use with the <xsl:value-of select="CLASS_NAME"/>DO class.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    static public DBTransaction createTransaction(String dbName) 
    throws DatabaseManagerException, SQLException {
        if (dbName == null)
            dbName = get_logicalDBName();
        DBTransaction ret;
        try {
            ret = DODS.getDatabaseManager().createTransaction(dbName);
            ret.setDatabaseName(dbName);
            return ret;
        } catch ( DatabaseManagerException e ) { 
            String err = "";
            if ( null != dbName )
                err = &quot;ERROR: Could not create a DBTransaction(dbName).  &quot; +
                      &quot;used database name ='&quot; + dbName + &quot;'.  &quot;+
                      &quot;The application .conf file must list this name in &quot; +
                      &quot;DatabaseManager.Databases[], and there must be &quot; +
                      &quot;DatabaseManager.DB.&quot; + dbName + &quot; settings.&quot;;
            throw new DatabaseManagerException( err, e );
        }
    }
</xsl:if>

<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * createQuery() creates a new DBQuery.
     * This method uses the logical database name set by method setLogicalDBName().
     *
     * If setLogicalDBName() was used to set the logical database name
     * to something other than the value of DatabaseManager.DefaultDatabase
     * in the application's .conf file, then any DBQuery object used to 
     * access the '<xsl:value-of select="TABLE_NAME"/>' table should be created using 
     * <xsl:value-of select="CLASS_NAME"/>DO.createQuery().
     *
     * The Query class corresponding to <xsl:value-of select="CLASS_NAME"/>DO uses this method.
     *
     * @return A DBQuery object to use in accessing the '<xsl:value-of select="TABLE_NAME"/>' table.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createQuery(DBTransaction) instead.</xsl:if>          
     */
    static public DBQuery createQuery() 
    throws DatabaseManagerException, SQLException {
        try {
	 	return DODS.getDatabaseManager().createQuery(get_logicalDBName());
        } catch ( DatabaseManagerException e ) { 
            String err = "";
            if ( null != get_logicalDBName() )
                err = &quot;ERROR: Could not create a DBQuery.  &quot; +
                      &quot;<xsl:value-of select="CLASS_NAME"/>DO.logicalDBName='&quot; + get_logicalDBName() + &quot;'.  &quot;+
                      &quot;The application .conf file must list this name in &quot; +
                      &quot;DatabaseManager.Databases[], and there must be &quot; +
                      &quot;DatabaseManager.DB.&quot; + get_logicalDBName() + &quot; settings.&quot;;
            throw new DatabaseManagerException( err, e );
        }
    }
</xsl:if>

    /**
     * createQuery() creates a new DBQuery.
     * This method uses the logical database name set by method setLogicalDBName().
     *
     * If setLogicalDBName() was used to set the logical database name
     * to something other than the value of DatabaseManager.DefaultDatabase
     * in the application's .conf file, then any DBQuery object used to 
     * access the '<xsl:value-of select="TABLE_NAME"/>' table should be created using 
     * <xsl:value-of select="CLASS_NAME"/>DO.createQuery().
     *
     * The Query class corresponding to <xsl:value-of select="CLASS_NAME"/>DO uses this method.
     *
     * @param trans DBTransaction
     * @return A DBQuery object to use in accessing the '<xsl:value-of select="TABLE_NAME"/>' table.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
     */
    static public DBQuery createQuery(DBTransaction trans) 
    throws DatabaseManagerException, SQLException {
<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        try {
	 	return (null==trans)?createQuery():trans.createQuery();</xsl:if><xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">
	 	return trans.createQuery();</xsl:if>
<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        } catch ( DatabaseManagerException e ) { 
            String err = "";
            if ( null != get_logicalDBName() )
                err = &quot;ERROR: Could not create a DBQuery.  &quot; +
                      &quot;<xsl:value-of select="CLASS_NAME"/>DO.logicalDBName='&quot; + get_logicalDBName() + &quot;'.  &quot;+
                      &quot;The application .conf file must list this name in &quot; +
                      &quot;DatabaseManager.Databases[], and there must be &quot; +
                      &quot;DatabaseManager.DB.&quot; + get_logicalDBName() + &quot; settings.&quot;;
            throw new DatabaseManagerException( err, e );
        }</xsl:if>
    }
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * createQuery() creates a new DBQuery.
     * This method uses the logical database name set by method setLogicalDBName().
     *
     * If setLogicalDBName() was used to set the logical database name
     * to something other than the value of DatabaseManager.DefaultDatabase
     * in the application's .conf file, then any DBQuery object used to 
     * access the '<xsl:value-of select="TABLE_NAME"/>' table should be created using 
     * <xsl:value-of select="CLASS_NAME"/>DO.createQuery().
     *
     * The Query class corresponding to <xsl:value-of select="CLASS_NAME"/>DO uses this method.
     *
     * @param dbName Logical name of the database from which 
     * <xsl:value-of select="CLASS_NAME"/>DO object will be created.
     
     * @return A DBQuery object to use in accessing the '<xsl:value-of select="TABLE_NAME"/>' table.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     * @exception SQLException
     *   If the database rejects the SQL generated to retrieve data
     *   for this object, or if the table contains a bad foreign key, etc.
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use createQuery(DBTransaction) instead.</xsl:if>     
     */
    static public DBQuery createQuery(String dbName) 
    throws DatabaseManagerException, SQLException {
        if (dbName == null)
            dbName = get_logicalDBName();
        try {
            return DODS.getDatabaseManager().createQuery(dbName);
        } catch ( DatabaseManagerException e ) { 
            String err = "";
            if ( null != dbName )
                err = &quot;ERROR: Could not create a DBQuery(dbName).  &quot; +
                      &quot;used database name='&quot; + dbName + &quot;'.  &quot;+
                      &quot;The application .conf file must list this name in &quot; +
                      &quot;DatabaseManager.Databases[], and there must be &quot; +
                      &quot;DatabaseManager.DB.&quot; + dbName + &quot; settings.&quot;;
            throw new DatabaseManagerException( err, e );
        }
    }
</xsl:if></xsl:if>

    /**
     * Protected constructor.
     *
     * @param rs Result set from which to obtain product data.
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
    protected <xsl:value-of select="CLASS_NAME"/>DO(String tablename, ResultSet rs)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        super(rs);
		this.dbtablename = tablename;
        initFromResultSet( rs );
        originDatabase = get_logicalDBName();
        get_DataStruct().set_Database(originDatabase);
        set_OId(new ObjectId(rs.getBigDecimal(get_OIdColumnName())));
    	if ( versioning )
       	    set_Version(rs.getInt(get_versionColumnName()));
        if(isTransactionCheck()) {
            DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_Handle()+", version: "+get_Version()+" \n");
            (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));
            
        }    
    }

    /**
     * Protected constructor.
     *
     * @param rs Result set from which to obtain product data.
     * @param queryRefs Reference objects.
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
    protected <xsl:value-of select="CLASS_NAME"/>DO(String tablename, ResultSet rs, HashMap queryRefs)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        this(tablename, rs,queryRefs,null);
    }

    /**
     * Protected constructor.
     *
     * @param rs Result set from which to obtain product data.
     * @param queryRefs Reference objects.
     * @param dbTrans DBTransaction object.
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
    protected <xsl:value-of select="CLASS_NAME"/>DO(String tablename, ResultSet rs, HashMap queryRefs, DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        super(rs);
		this.dbtablename = tablename;
        set_refs(queryRefs);
	setTransaction(dbTrans);
        initFromResultSet( rs );
        if(dbTrans!=null)
            originDatabase = dbTrans.getDatabaseName();
        if(originDatabase==null)    
           originDatabase = get_logicalDBName();<xsl:if test="DO_IS_MULTIDB_BASED='true'">
        else {    
           if(useLogicalDatabase(originDatabase)) {
             try {
               readCacheConfiguration(originDatabase);
          } catch (Exception ex) {
             DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default cache configuration for '"+originDatabase+"' database");  
          }  
         }     
        }    </xsl:if>
        get_DataStruct().set_Database(originDatabase);
        addToTransactionCache();
        if(dbTrans!=null)
            dbTrans.lockDO(this);
    }    

    /**
     * Protected constructor.
     *
     * @param rs Result set from which to obtain product data.
     * @param dbTrans The current database transaction
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
    protected <xsl:value-of select="CLASS_NAME"/>DO(String tablename, ResultSet rs, DBTransaction dbTrans)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        this(tablename, rs, null, dbTrans);
    }
    
    
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
    /**
     * Protected constructor.
     *
     * @param dbName Logical name of the database from which to obtain a object.
     * @param rs Result set from which to obtain product data.
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
    protected <xsl:value-of select="CLASS_NAME"/>DO(String tablename, String dbName, ResultSet rs)
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        super(dbName,rs);
		this.dbtablename = tablename;
        if (dbName == null)
            dbName = get_OriginDatabase();       
        initFromResultSet( rs );
        originDatabase = dbName;
        if(useLogicalDatabase(dbName)) {
          try {  
            readCacheConfiguration(dbName);
          } catch (Exception ex) {
             DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>DO class\n :"+" Using default cache configuration for '"+dbName+"' database"); 
          }  
        }   
        get_DataStruct().set_Database(dbName);
        set_OId(new ObjectId(rs.getBigDecimal(get_OIdColumnName())));
        if ( versioning )
       	    set_Version(rs.getInt(get_versionColumnName()));
      if(isTransactionCheck()){
            DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+get_Handle()+", version: "+get_Version()+" \n");
           (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));
            
        }     
    }
</xsl:if>


    /**
     * while in initFromResultSet, auto save can't be allowed
     */
    private boolean autoSaveAllowed = true;

    /**
     * initFromResultSet initializes the data members of <xsl:value-of select="TABLE_NAME"/>.
     * This code was separated from the ResultSet constructor
     * so that createExisting(ResultSet) could handle VIEWs.
     *
     * @param rs ResultSet from which data members are initialized.
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
    private void initFromResultSet( ResultSet rs )
    throws SQLException, ObjectIdException, DataObjectException, DatabaseManagerException {
        autoSaveAllowed = false;
        // Constructing a DO from a ResultSet means we definitely need the 
        // DataStruct ready for the setXxx methods invoked below.
        if (null == get_DataStruct())
            originalData = new <xsl:value-of select="CLASS_NAME"/>DataStruct ();
	get_DataStruct().isEmpty = false; 
        // writeMemberStuff uses the ResultSetExtraction.template
        // to build up the value for this tag:
        // the value is a series of calls to the DO set methods.
        
        if ( versioning )set_Version(rs.getInt(get_versionColumnName()));
<xsl:value-of select="RESULT_SET_EXTRACTION_FOR_LONG_CALLS"/>
<xsl:value-of select="RESULT_SET_EXTRACTION_CALLS"/>
</xsl:template>
</xsl:stylesheet>
