<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                version="1.0">
<xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="manyToOneReferrersDO">
<xsl:if test="dodst:compareReferrerDoName(/TABLE/PACKAGE,/TABLE/CLASS_NAME,@do_name) = 'true'">
    /**
     * Get array of <xsl:value-of select="../@name"/>DO objects that refer to this DO.
     *
     * @return Array of <xsl:value-of select="../@name"/>DO objects.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception QueryException
     *   If an error occured while building the query before execution.
     */
    public <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO[] <xsl:value-of select="dodst:getArrayDO(../@name,@name)"/> () 
    throws DataObjectException, QueryException {
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO[] ret = null;
        try {
            <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query q;
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
            if (originDatabase != null)
                q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(originDatabase);
            else</xsl:if></xsl:if>
                 q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(get_transaction());
<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">
            q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( this, QueryBuilder.EQUAL );</xsl:if>
<xsl:if test="not(/TABLE/DO_IS_OID_BASED='true')">
            q.setQuery<xsl:value-of select="../@name"/>DO( this, QueryBuilder.EQUAL);</xsl:if>
            ret = q.getDOArray();
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
        } catch (DatabaseManagerException e) {
            throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException" );</xsl:if>
        } catch ( NonUniqueQueryException e ) { 
            throw new DataObjectException("INTERNAL ERROR: unexpected NonUniqueQueryException" );
        } finally {
            if ( null == ret )
            ret = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO[ 0 ];
        }
        return ret;
    }

    /**
     * Get the single <xsl:value-of select="../@name"/>DO object
     * that refers to this DO.
     *
     * @return <xsl:value-of select="../@name"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception QueryException
     *   If an error occured while building the query before execution.
     * @exception NonUniqueQueryException
     *   If more than one <xsl:value-of select="../@name"/>DO object was found.
     */
    public <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO <xsl:value-of select="dodst:getSingleDO(../@name,@name)"/> () 
    throws DataObjectException, QueryException, NonUniqueQueryException {
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query q;
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        try {
            if (originDatabase != null)
                q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(originDatabase);
            else</xsl:if></xsl:if>
               q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(get_transaction());
<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">
            q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( this, QueryBuilder.EQUAL );</xsl:if>
<xsl:if test="not(/TABLE/DO_IS_OID_BASED='true')">
            q.setQuery<xsl:value-of select="../@name"/>DO( this, QueryBuilder.EQUAL );</xsl:if>
            q.requireUniqueInstance();
            return q.getNextDO();
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        } catch (DatabaseManagerException e) {
            throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException" );
        }</xsl:if></xsl:if>
    }
    
    
<xsl:if test="/TABLE/DO_IS_OID_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Add (set &amp; commit) a <xsl:value-of select="../@name"/>DO object that refers to this DO.
     *
     * @param referrer <xsl:value-of select="../@name"/>DO to be set to point to this DO and committed.
     *
     * @exception DatabaseManagerException if could not create a transaction.
     * @exception java.sql.SQLException if any SQL errors occur.
     * @exception DataObjectException If object is not found in the database.
     * @exception RefAssertionException Thrown by okTo method.
     * @exception DBRowUpdateException
     * @exception QueryException
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use <xsl:value-of select="dodst:addSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO , DBTransaction )
</xsl:if>     
     */
    public void <xsl:value-of select="dodst:addSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO referrer )
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException {
        // WebDocWf extension for DODS row instance security
        // The following line has been changed:
        <xsl:value-of select="dodst:addSingleDO(../@name,@name)"/>( referrer, (DBTransaction)null );
        // before: <xsl:value-of select="dodst:addSingleDO(../@name,@name)"/>( referrer, null );
        // end of WebDocWf extension for DODS row instance security
    }
</xsl:if>


    /**
     * Add (set &amp; commit) a <xsl:value-of select="../@name"/>DO object that refers to this DO.
     *
     * @param referrer <xsl:value-of select="../@name"/>DO to be set to point to this DO and committed.
     * @param tran The transaction to be used for the commit.
     * If null, a new transaction is created.
     *
     * @exception DatabaseManagerException if could not create a transaction.
     * @exception java.sql.SQLException if any SQL errors occur.
     * @exception DataObjectException If object is not found in the database.
     * @exception RefAssertionException Thrown by okTo method.
     * @exception DBRowUpdateException
     * @exception QueryException
     */
    public void <xsl:value-of select="dodst:addSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO referrer, DBTransaction tran )
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException {
        referrer.set<xsl:value-of select="common:capitalizeName(@name)"/>( this );
        referrer.save( tran );
    }
    
    
<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Remove (delete) a <xsl:value-of select="../@name"/>DO object that refers to this DO.
     *
     * @param referrer <xsl:value-of select="../@name"/>DO to be deleted.
     *
     * @exception DatabaseManagerException if could not create a transaction.
     * @exception java.sql.SQLException if any SQL errors occur.
     * @exception DataObjectException If object is not found in the database.
     * @exception RefAssertionException Thrown by okTo method.
     * @exception DBRowUpdateException
     * @exception QueryException
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use <xsl:value-of select="dodst:removeSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO, DBTransaction )
</xsl:if>     
     */
    public void <xsl:value-of select="dodst:removeSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO referrer )
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException {
        // WebDocWf extension for DODS row instance security
        // The following line has been changed:
        <xsl:value-of select="dodst:removeSingleDO(../@name,@name)"/>( referrer, (DBTransaction)null );
        // before: <xsl:value-of select="dodst:removeSingleDO(../@name,@name)"/>( referrer, null );
        // end of WebDocWf extension for DODS row instance security
    }
</xsl:if> 


    /**
     * Remove (delete) a <xsl:value-of select="../@name"/>DO object that refers to this DO.
     *
     * @param referrer <xsl:value-of select="../@name"/>DO to be deleted.
     * @param tran The transaction to be used for the commit.
     * If null, a new transaction is created.
     *
     * @exception DatabaseManagerException if could not create a transaction.
     * @exception java.sql.SQLException if any SQL errors occur.
     * @exception DataObjectException If object is not found in the database.
     * @exception RefAssertionException Thrown by okTo method.
     * @exception DBRowUpdateException
     * @exception QueryException
     */
    public void <xsl:value-of select="dodst:removeSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO referrer, DBTransaction tran )
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException {
        <xsl:value-of select="/TABLE/CLASS_NAME"/>DO referred = referrer.get<xsl:value-of select="common:capitalizeName(@name)"/>();
        String referredHandle = referred.get_Handle();
        String mydoHandle = this.get_Handle();
        if ( null == referredHandle || null == mydoHandle || ( ! referredHandle.equals( mydoHandle ) ) ) {
            throw new DataObjectException( "Object " + referrer +
            " does not refer to object " + this + ", cannot be removed this way." );
        }
        referrer.delete( tran );
    }
</xsl:if>

<xsl:if test="/TABLE/DO_IS_OID_BASED='true' and ../@generateSecure='true'">
    // WebDocWF extension for DODS row instance security
    // The following lines have been added:
    /**
     * Get array of <xsl:value-of select="../@name"/>DO objects that refer to this DO.
     *
     * @param usr The user for security checks.
     *
     * @return Array of <xsl:value-of select="../@name"/>DO objects.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception QueryException
     *   If an error occured while building the query before execution.
     * @exception AccessException
     *   The user is not allowed to create a query for 
     *   <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/> or 
     *   to access the foreign key attribute <xsl:value-of select="common:capitalizeName(@name)"/>.
     *
     * This is a WebDocWf extension for DODS row instance security
     *
     */
    public <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO[] <xsl:value-of select="dodst:getArrayDO(../@name,@name)"/> ( org.webdocwf.dods.access.User usr ) 
    throws DataObjectException, QueryException, AccessException {
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO[] ret = null;
        try {
            <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query q;
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
            if (originDatabase != null)
                q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query( originDatabase,usr );
            else</xsl:if></xsl:if>
                    q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(get_transaction(), usr);
<xsl:if test="@generateSecure='true'">                
         q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( this, usr );
</xsl:if>
<xsl:if test="@generateSecure='false'">                
         q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( this );
</xsl:if>
            
            ret = q.getDOArray();
        } catch ( NonUniqueQueryException e ) { 
            throw new DataObjectException( "INTERNAL ERROR: unexpected NonUniqueQueryException" );
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
        } catch ( DatabaseManagerException e ) { 
            throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException" );</xsl:if>
        } finally {
            if ( null == ret )
                ret = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO[ 0 ];
        }
        return ret;
    }

    /**
     * Get the single <xsl:value-of select="../@name"/>DO object
     * that refers to this DO.
     *
     * @param usr The user for security checks.
     *
     * @return <xsl:value-of select="../@name"/>DO object.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception QueryException
     *   If an error occured while building the query before execution.
     * @exception NonUniqueQueryException
     *   If more than one <xsl:value-of select="../@name"/>DO object was found.
     * @exception AccessException
     *   The user is not allowed to create a query for 
     *   <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/> or 
     *   to access the foreign key attribute <xsl:value-of select="common:capitalizeName(@name)"/>.
     *
     * This is a WebDocWf extension for DODS row instance security
     *
     */
    public <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO <xsl:value-of select="dodst:getSingleDO(../@name,@name)"/> ( org.webdocwf.dods.access.User usr ) 
    throws DataObjectException, QueryException, NonUniqueQueryException, AccessException {
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query q;
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        try {
            if (originDatabase != null)
                q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query( originDatabase,usr );
            else</xsl:if></xsl:if>
                    q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(get_transaction(), usr);
<xsl:if test="@generateSecure='true'">                
            q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( this, usr );
</xsl:if>            
<xsl:if test="@generateSecure='false'">                            
            q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( this );
</xsl:if>
            q.requireUniqueInstance();
            return q.getNextDO();
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        } catch (DatabaseManagerException e) {
            throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException" );
        }</xsl:if></xsl:if>
    }
    
<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
    /**
     * Add (set &amp; commit) a <xsl:value-of select="../@name"/>DO object that refers to this DO.
     *
     * @param referrer <xsl:value-of select="../@name"/>DO to be set to point to this DO and committed.
     * @param usr The user for security checks.
     *
     * @exception DatabaseManagerException if could not create a transaction.
     * @exception java.sql.SQLException if any SQL errors occur.
     * @exception DataObjectException If object is not found in the database.
     *
     * WebDocWf extension
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use <xsl:value-of select="dodst:addSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO, DBTransaction, org.webdocwf.dods.access.User)
</xsl:if>     
     *
     */
    public void <xsl:value-of select="dodst:addSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO referrer, org.webdocwf.dods.access.User usr )
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException, AccessException {
        <xsl:value-of select="dodst:addSingleDO(../@name,@name)"/>( referrer, null, usr );
    }
</xsl:if> 

    /**
     * Add (set &amp; commit) a <xsl:value-of select="../@name"/>DO object that refers to this DO.
     *
     * @param referrer <xsl:value-of select="../@name"/>DO to be set to point to this DO and committed.
     * @param tran The transaction to be used for the commit.
     * @param usr The user for security checks.
     * If null, a new transaction is created.
     *
     * @exception DatabaseManagerException if could not create a transaction.
     * @exception java.sql.SQLException if any SQL errors occur.
     * @exception DataObjectException If object is not found in the database.
     *
     * WebDocWf extension
     *
     */
    public void <xsl:value-of select="dodst:addSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO referrer, DBTransaction tran, org.webdocwf.dods.access.User usr )
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException, AccessException {
<xsl:if test="@generateSecure='true'">
        referrer.set<xsl:value-of select="common:capitalizeName(@name)"/>( this, usr );
</xsl:if>
<xsl:if test="@generateSecure='false'">
        referrer.set<xsl:value-of select="common:capitalizeName(@name)"/>( this );
</xsl:if>
        referrer.save( tran );
    }
<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')"> 
    /**
     * Remove (delete) a <xsl:value-of select="../@name"/>DO object that refers to this DO.
     *
     * @param referrer <xsl:value-of select="../@name"/>DO to be deleted.
     * @param usr The user for security checks.
     *
     * @exception DatabaseManagerException if could not create a transaction.
     * @exception java.sql.SQLException if any SQL errors occur.
     * @exception DataObjectException If object is not found in the database.
     *
     * WebDocWf extension
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use <xsl:value-of select="dodst:removeSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO, DBTransaction, org.webdocwf.dods.access.User)
</xsl:if>       
     *
     */
    public void <xsl:value-of select="dodst:removeSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO referrer, org.webdocwf.dods.access.User usr )
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException, AccessException {
        <xsl:value-of select="dodst:removeSingleDO(../@name,@name)"/>( referrer, null, usr );
    }
</xsl:if> 

 
    /**
     * Remove (delete) a <xsl:value-of select="../@name"/>DO object that refers to this DO.
     *
     * @param referrer <xsl:value-of select="../@name"/>DO to be deleted.
     * @param tran The transaction to be used for the commit.
     * If null, a new transaction is created.
     * @param usr The user for security checksa
     *
     * @exception DatabaseManagerException if could not create a transaction.
     * @exception java.sql.SQLException if any SQL errors occur.
     * @exception DataObjectException If object is not found in the database.
     *
     * WebDocWf extension
     *
     */
    public void <xsl:value-of select="dodst:removeSingleDO(../@name,@name)"/>( <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO referrer, DBTransaction tran, org.webdocwf.dods.access.User usr )
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException, AccessException {
<xsl:if test="@generateSecure='true'">
        <xsl:value-of select="/TABLE/CLASS_NAME"/>DO referred = referrer.get<xsl:value-of select="common:capitalizeName(@name)"/>( usr );
</xsl:if>
<xsl:if test="@generateSecure='false'">
        <xsl:value-of select="/TABLE/CLASS_NAME"/>DO referred = referrer.get<xsl:value-of select="common:capitalizeName(@name)"/>();
</xsl:if>
        String referredHandle = referred.get_Handle();
        String mydoHandle = this.get_Handle();
        if ( null == referredHandle || null == mydoHandle || ( ! referredHandle.equals( mydoHandle ) ) ) {
            throw new DataObjectException( "Object " + referrer + " does not refer to object " + this + ", cannot be removed this way." );
        }
        referrer.delete( tran, usr );
    }
    // end of WebDocWf extension for DODS row instance security
</xsl:if>

    // WebDocWF extension for DODS row counter
    // The following lines have been added:
    /**
     * Get the number of <xsl:value-of select="../@name"/>DOs that refer to this DO.
     * via <xsl:value-of select="common:capitalizeName(@name)"/>
     *
     * @return The number of objects that refer to this DO.
     *
     * @exception DataObjectException If object is not found in the database.
     * @exception QueryException
     * @exception NonUniqueQueryException
     * @exception SQLException If any SQL errors occur.
     * @exception DatabaseManagerException If a Transaction can not be created.
     *
     * This is a WebDocWf extension for DODS row instance security.
     *
     */
    public int <xsl:value-of select="dodst:getArrayDO(../@name,@name)"/>Count () 
    throws DataObjectException, QueryException, NonUniqueQueryException, SQLException, DatabaseManagerException {
        int ret = 0;
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query q;
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        try {
            if (originDatabase != null)
                q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(originDatabase);
            else</xsl:if></xsl:if>
                q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(get_transaction());
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        } catch (DatabaseManagerException e) {
            throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException" );
        }</xsl:if></xsl:if>
        q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( this, QueryBuilder.EQUAL );
        ret = q.getCount();
        return ret;
    }
    
<xsl:if test="/TABLE/DO_IS_OID_BASED='true' and ../@generateSecure='true'">
    /**
     * Get the number of <xsl:value-of select="../@name"/>DOs that refer to this DO.
     * via <xsl:value-of select="common:capitalizeName(@name)"/>
     *
     * @param usr The user for security checks.
     *
     * @return The number of objects that refer to this DO.
     *
     * @exception DataObjectException If object is not found in the database.
     * @exception AccessException
     *   The user is not allowed to create a query for 
     *   <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/> or 
     *   to access the foreign key attribute <xsl:value-of select="common:capitalizeName(@name)"/>.
     *
     * This is a WebDocWf extension for DODS row instance security
     *
     */
    public int <xsl:value-of select="dodst:getArrayDO(../@name,@name)"/>Count ( org.webdocwf.dods.access.User usr ) 
    throws DataObjectException, QueryException, NonUniqueQueryException, SQLException, AccessException, DatabaseManagerException {
        int ret = 0;
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query q;
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        try {
            if (originDatabase != null)
                q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(originDatabase,usr);
            else</xsl:if></xsl:if>
                q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(get_transaction(), usr);
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        } catch ( DatabaseManagerException e ) { 
            throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException" );
        }</xsl:if></xsl:if>
<xsl:if test="@generateSecure='true'">
        q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( this, usr );
</xsl:if>
<xsl:if test="@generateSecure='true'">
        q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( this );
</xsl:if>
        ret = q.getCount();
        return ret;
    }
    // end of WebDocWF extension for DODS row counter
</xsl:if></xsl:if>
</xsl:template>
</xsl:stylesheet>
 

