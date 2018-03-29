<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                version="1.0">
<xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="manyToManyReferrersDO">
<xsl:if test="@n2n='true'">
<xsl:if test="/TABLE/GENERATE_INSECURE='true'">    
    /**
     * From the many-to-many relationship expressed by <xsl:value-of select="../@name"/>DO,
     * get array of <xsl:value-of select="dodst:getDoName(@do_name)"/>DO objects that indirectly refer
     * to this DO.
     *
     * @return Array of <xsl:value-of select="dodst:getDoName(@do_name)"/>DO objects.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public <xsl:value-of select="@do_name"/>DO[] <xsl:value-of select="dodst:getArrayDON2N(dodst:getDoName(@do_name),../@name)"/> ()
    throws DataObjectException {
        <xsl:value-of select="@do_name"/>DO[] ret = null;
        try {
            <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO[] arr = get<xsl:value-of select="../@name"/>DOArray();
            ret = new <xsl:value-of select="@do_name"/>DO[ arr.length ];
            for ( int i = 0; i &lt; arr.length; i++ ) {
                ret[ i ] = arr[ i ].get<xsl:value-of select="common:capitalizeName(@name)"/>();
            }
        } catch ( Exception e ) {
            throw new DataObjectException(
            "INTERNAL ERROR: ", e );
        } finally {
            if ( null == ret )
            ret = new <xsl:value-of select="@do_name"/>DO[ 0 ];
        }
        return ret;
    }

    /**
     * To the many-to-many relationship expressed by <xsl:value-of select="../@name"/>DO,
     * add a <xsl:value-of select="dodst:getDoName(@do_name)"/>DO object that indirectly refers
     * to this DO.
     *
     * @param d The <xsl:value-of select="dodst:getDoName(@do_name)"/>DO to add to the <xsl:value-of select="../@name"/>DO mapping
     * for this DO.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void map<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( <xsl:value-of select="@do_name"/>DO d )
    throws DataObjectException, DatabaseManagerException, RefAssertionException, SQLException, DBRowUpdateException, QueryException {
        // WebDocWf fix, was ambiguous, the following line was changed:
        map<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( d, get_transaction() );
        // before: map<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( d, null );
        // end of WebDocWf fix
    }

    /**
     * To the many-to-many relationship expressed by <xsl:value-of select="../@name"/>DO,
     * add a <xsl:value-of select="dodst:getDoName(@do_name)"/>DO object that indirectly refers to this DO.
     *
     * @param d The <xsl:value-of select="dodst:getDoName(@do_name)"/>DO to add to the <xsl:value-of select="../@name"/>DO mapping for this DO.
     * @param tran The transaction to be used. If null, a new transaction is created.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void map<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( <xsl:value-of select="@do_name"/>DO d, DBTransaction tran )
    throws DataObjectException, DatabaseManagerException, RefAssertionException, SQLException, DBRowUpdateException, QueryException {
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO m = null;
        try {
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
            if (originDatabase != null)
                m = <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin(tran);
            else {</xsl:if>
                if(get_transaction()!=null &amp;&amp; tran!=null) {
                    if(!get_transaction().equals(tran))
                        throw new DatabaseManagerException ("This DO doesn't belong this transaction ");    
                }
                if(tran != null)
                   m = <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin(tran);
                else if(get_transaction()!=null)
                   m = <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin(get_transaction()); 
                else
                   m = <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin((DBTransaction)null);    
<xsl:if test="DO_IS_MULTIDB_BASED='true'">}</xsl:if>
        } catch ( Exception e ) {
            throw new DataObjectException(&quot;<xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin failed&quot;, e );
        }
        m.set<xsl:value-of select="common:capitalizeName(@name)"/>( d );
        m.set<xsl:value-of select="common:capitalizeName(@another)"/>( this );
        m.save( tran );
    }

    /**
     * From the many-to-many relationship expressed by <xsl:value-of select="../@name"/>DO,
     * remove (delete) the <xsl:value-of select="dodst:getDoName(@do_name)"/>DO object that indirectly refers
     * to this DO.
     *
     * @param d The <xsl:value-of select="dodst:getDoName(@do_name)"/>DO to remove from the <xsl:value-of select="../@name"/>DO mapping
     * for this DO.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception QueryException
     *   If an error occured while building the query before execution.
     */
    public void unmap<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( <xsl:value-of select="@do_name"/>DO d )
    throws DataObjectException, DatabaseManagerException, RefAssertionException, SQLException, DBRowUpdateException, QueryException {
        // WebDocWf fix, was ambiguous, the following line was changed:
        unmap<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( d, get_transaction());
        // before:  unmap<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( d, null );
        //end of WebDocWf fix
    }

    /**
     * From the many-to-many relationship expressed by <xsl:value-of select="../@name"/>DO,
     * remove (delete) the <xsl:value-of select="dodst:getDoName(@do_name)"/>DO object that indirectly refers
     * to this DO.
     *
     * @param d The <xsl:value-of select="dodst:getDoName(@do_name)"/>DO to remove from the <xsl:value-of select="../@name"/>DO mapping
     * for this DO.
     * @param tran The transaction to be used. If null, a new transaction is created.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception QueryException
     *   If an error occured while building the query before execution.
     */
    public void unmap<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( <xsl:value-of select="@do_name"/>DO d, DBTransaction tran )
    throws DataObjectException, DatabaseManagerException, RefAssertionException, SQLException, DBRowUpdateException, QueryException {
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query q;
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
        if (originDatabase != null)
            q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(tran);
        else {</xsl:if></xsl:if>
          if(get_transaction()!=null &amp;&amp; tran!=null) {
                    if(!get_transaction().equals(tran))
                        throw new DatabaseManagerException ("This DO doesn't belong this transaction ");    
          }
          if(tran != null)
            q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(tran);
          else
             q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(get_transaction());    
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">}</xsl:if></xsl:if>             
        q.setQuery<xsl:value-of select="common:capitalizeName(@another)"/>( this, QueryBuilder.EQUAL );
        q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( d, QueryBuilder.EQUAL );
        q.requireUniqueInstance();
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO m = null;
        try {
            m = q.getNextDO();
        } catch ( NonUniqueQueryException e ) {
            throw new DataObjectException( &quot;Multiple mappings for &quot; +
            this + &quot; and &quot; + d + &quot; in <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/> table.&quot; );
        }
        m.delete( tran );
    }
 
</xsl:if>     
<xsl:if test="/TABLE/DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">

    /**
     * To the many-to-many relationship expressed by <xsl:value-of select="../@name"/>DO,
     * add a <xsl:value-of select="dodst:getDoName(@do_name)"/>DO object that indirectly refers
     * to this DO.
     *
     * @param d The <xsl:value-of select="dodst:getDoName(@do_name)"/>DO to add to the <xsl:value-of select="../@name"/>DO mapping
     * for this DO.
     * @param usr The user for security checks
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     *
     * WebDocWf extension
     *
     */
    public void map<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( <xsl:value-of select="@do_name"/>DO d, org.webdocwf.dods.access.User usr )
    throws DataObjectException, DatabaseManagerException, RefAssertionException, SQLException, DBRowUpdateException, QueryException, AccessException {
        map<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( d, null, usr );
    }

    /**
     * To the many-to-many relationship expressed by <xsl:value-of select="../@name"/>DO,
     * add a <xsl:value-of select="dodst:getDoName(@do_name)"/>DO object that indirectly refers to this DO.
     *
     * @param d The <xsl:value-of select="dodst:getDoName(@do_name)"/>DO to add to the <xsl:value-of select="../@name"/>DO mapping for this DO.
     * @param tran The transaction to be used. If null, a new transaction is created.
     * @param usr The user for security checks
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     *
     * WebDocWf extension
     *
     */
    public void map<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( <xsl:value-of select="@do_name"/>DO d, DBTransaction tran, org.webdocwf.dods.access.User usr )
    throws DataObjectException, DatabaseManagerException, RefAssertionException, SQLException, DBRowUpdateException, QueryException, AccessException {
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO m = null;
        try {
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
            if (originDatabase != null)
                m = <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin( tran,usr );
            else {</xsl:if>
               if(get_transaction()!=null &amp;&amp; tran!=null) {
                  if(!get_transaction().equals(tran))
                     throw new DatabaseManagerException ("This DO doesn't belong this transaction ");    
               }
               if(tran != null)
                    m = <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin( tran, usr);
                else if(get_transaction()!=null)                    
                    m = <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin( get_transaction(), usr);
                else    
                    m = <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin( (DBTransaction)null, usr );
<xsl:if test="DO_IS_MULTIDB_BASED='true'">}</xsl:if>                    
        } catch ( ObjectIdException e ) { 
            throw new DataObjectException("<xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO.createVirgin failed", e );
        }
        m.set<xsl:value-of select="common:capitalizeName(@name)"/>( d, usr );
        m.set<xsl:value-of select="common:capitalizeName(@another)"/>( this, usr );
        m.save( tran );
    }
    
    /**
     * From the many-to-many relationship expressed by <xsl:value-of select="../@name"/>DO,
     * remove (delete) the <xsl:value-of select="dodst:getDoName(@do_name)"/>DO object that indirectly refers
     * to this DO.
     *
     * @param d The <xsl:value-of select="dodst:getDoName(@do_name)"/>DO to remove from the <xsl:value-of select="../@name"/>DO mapping
     * for this DO.
     * @param usr The user for security checks
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception QueryException
     *   If an error occured while building the query before execution.
     *
     * WebDocWf extension
     *
     */
    public void unmap<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( <xsl:value-of select="@do_name"/>DO d, org.webdocwf.dods.access.User usr )
    throws DataObjectException, DatabaseManagerException, RefAssertionException, SQLException, DBRowUpdateException, QueryException, AccessException {
        unmap<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( d, null, usr );
    }

    /**
     * From the many-to-many relationship expressed by <xsl:value-of select="../@name"/>DO,
     * remove (delete) the <xsl:value-of select="dodst:getDoName(@do_name)"/>DO object that indirectly refers
     * to this DO.
     *
     * @param d The <xsl:value-of select="dodst:getDoName(@do_name)"/>DO to remove from the <xsl:value-of select="../@name"/>DO mapping
     * for this DO.
     * @param tran The transaction to be used. If null, a new transaction is created.
     * @param usr The user for security checks
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception QueryException
     *   If an error occured while building the query before execution.
     *
     * WebDocWf extension
     *
     */
    public void unmap<xsl:value-of select="dodst:getDoName(@do_name)"/>_via_<xsl:value-of select="../@name"/>DO( <xsl:value-of select="@do_name"/>DO d, DBTransaction tran, org.webdocwf.dods.access.User usr )
    throws DataObjectException, DatabaseManagerException, RefAssertionException, SQLException, DBRowUpdateException, QueryException, AccessException {
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query q;
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
        try {
            if (originDatabase != null)
                q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(tran,usr);
            else {</xsl:if>
               if(get_transaction()!=null &amp;&amp; tran!=null) {
                  if(!get_transaction().equals(tran))
                     throw new DatabaseManagerException ("This DO doesn't belong this transaction ");    
               }
               if(tran != null)
                 q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(tran, usr);
               else if(get_transaction()!=null) 
                 q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query(get_transaction(), usr );
               else  
                 q = new <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>Query((DBTransaction)null, usr );
<xsl:if test="DO_IS_MULTIDB_BASED='true'"> }
        } catch ( DatabaseManagerException e ) { 
            throw new DataObjectException("INTERNAL ERROR: unexpected DatabaseManagerException" );
        }
</xsl:if>
        q.setQuery<xsl:value-of select="common:capitalizeName(@another)"/>( this , usr);
        q.setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( d , usr);
        q.requireUniqueInstance();
        <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/>DO m = null;
        try {
            m = q.getNextDO();
        } catch ( NonUniqueQueryException e ) { 
            throw new DataObjectException( "Multiple mappings for " + this + " and " + d + " in <xsl:value-of select="miu:getAdjustedPackageName(../@package)"/>.<xsl:value-of select="../@name"/> table." );
        }
        m.delete( tran, usr );
    }
    // end of WebDocWf extension for DODS row instance security
</xsl:if></xsl:if>
</xsl:template>
</xsl:stylesheet>
