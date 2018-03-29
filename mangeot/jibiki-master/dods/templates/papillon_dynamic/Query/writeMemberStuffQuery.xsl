<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
    <xsl:template name="memberStuff"><xsl:if test="(USED_FOR_QUERY='true' and GENERATE_INSECURE='true')">
    /**
     * Set the <xsl:value-of select="@name"/> to query, with a QueryBuilder comparison operator.
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     * @param cmp_op QueryBuilder comparison operator to use.
     *
     * @exception DataObjectException If a database access error occurs.
     * @exception QueryException If comparison operator is inappropriate
     * (e.g. CASE_SENSITIVE_CONTAINS on an integer field).
     */

    
    <xsl:if test="(USED_FOR_QUERY='true' and GENERATE_INSECURE='true')">public</xsl:if><xsl:if test="(USED_FOR_QUERY='false' or GENERATE_INSECURE='false')">private</xsl:if> void setQuery<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, String cmp_op )
    throws DataObjectException, QueryException {
    
    String cachePrefix = getLogicalDatabase()+".";
<xsl:if test="(REFERENCE_OBJECT/IS_FOREIGN_KEY='false')">
        if(transaction!=null &amp;&amp; x!=null &amp;&amp; x.get_transaction()!=null) {
            if(!transaction.equals(x.get_transaction()))
                 throw new DataObjectException("Referenced DO doesn't belong the same transaction.");
        }
        if(refs==null)
            refs = new HashMap();
        if(x!=null)
            refs.put(cachePrefix+x.get_OId(),x);
</xsl:if>
        hasNonOidCond = true;
        cmp_op = fixCaseSensitiveCondition(cmp_op);
        Condition cond = null;
<xsl:if test="(REFERENCE_OBJECT/IS_FOREIGN_KEY='false')">        
         if((x!=null)&amp;&amp; (x instanceof CoreDO)) {
           CoreDataStruct xDataStruct = (CoreDataStruct)x.get_DataStruct(); 
           cond = new Condition(<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct.COLUMN_<xsl:value-of select="common:upperCaseName(@name)"/>,xDataStruct,cmp_op);    
         } else  
</xsl:if>      cond = new Condition(<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct.COLUMN_<xsl:value-of select="common:upperCaseName(@name)"/>,x,cmp_op);
        queryItem.addCond(cond);
        // WebDocWf extension for extended wildcard support
        // The following lines have been added:
        if (cmp_op.equals(QueryBuilder.CASE_INSENSITIVE_MATCH) || cmp_op.equals(QueryBuilder.CASE_SENSITIVE_MATCH) || cmp_op.equals(QueryBuilder.USER_CASE_SENSITIVE_MATCH) || cmp_op.equals(QueryBuilder.USER_CASE_INSENSITIVE_MATCH)) {
            hitDb = true;
        } else {
            // end of WebDocWf extension for extended wildcard support
            
            if (<xsl:value-of select="/TABLE/CLASS_NAME"/>DO.getCache(this.dbtablename).isFull() &amp;&amp; (<xsl:value-of select="/TABLE/CLASS_NAME"/>DO.isFullCacheNeeded)) {
                // Remove from cacheHits any DOs that do not meet this
                // setQuery requirement.
                <xsl:value-of select="/TABLE/CLASS_NAME"/>DO DO = null;
                <xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct DS = null;
// 12.04.2004 tj
//                for ( Iterator iter = (new HashSet(cacheHits.values())).iterator(); iter.hasNext();) 
                for ( int i = 0; i &lt; cacheHits.size(); i++ ) {
                      try {
                        boolean findInTransactionCache = false;
//                        DS = (<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct)iter.next();
                        DS = (<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct)cacheHits.elementAt( i );
                        if(transaction!=null &amp;&amp; _tr_(transaction).getTransactionCache()!=null) {
                            DO = (<xsl:value-of select="/TABLE/CLASS_NAME"/>DO)_tr_(transaction).getTransactionCache().getDOByHandle(cachePrefix+DS.get_Handle());
                            if(DO != null)
                                findInTransactionCache = true;
                        }
                        if(!findInTransactionCache){
                            DO = (<xsl:value-of select="/TABLE/CLASS_NAME"/>DO)<xsl:value-of select="/TABLE/CLASS_NAME"/>DO.ceInternal(this.dbtablename, DS.get_OId(), transaction);
                        }
                      }catch (Exception ex) {
                      System.out.println("Error in query member stuff");
                    }            
                    <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> m = DO.get<xsl:value-of select="common:capitalizeName(@name)"/>();
                    if(!QueryBuilder.compare( m, x, cmp_op )) {
                        try {
                            String cacheHandle = DO.get_CacheHandle();
//                            cacheHits.remove(cacheHandle);
                            cacheHits.removeElementAt( i-- );
                        } catch (DatabaseManagerException e) {
                            throw new DataObjectException("Error in loading data object's handle.");
                        }
                    }
                } // for
            }
        }
        // Also prepares the SQL needed to query the database
        // in case there is no cache, or the query involves other tables.
        // WebDocWf patch for correct queries in fully cached objects
        // The following line has been put under comments:
        // if ( partOrLru || hitDb )
        // end of WebDocWf patch for correct queries in fully cached objects
        builder.addWhere( <xsl:value-of select="/TABLE/CLASS_NAME"/>DO.get<xsl:value-of select="common:capitalizeName(@name)"/>Column(this.dbtablename), x, cmp_op );
    }

    /**
     * Set the <xsl:value-of select="@name"/> to query, with a QueryBuilder comparison operator.
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     *
     * @exception DataObjectException If a database access error occurs.
     * @exception QueryException If comparison operator is inappropriate
     * (e.g. CASE_SENSITIVE_CONTAINS on an integer field).
     */
    <xsl:if test="(USED_FOR_QUERY='true' and GENERATE_INSECURE='true')">public</xsl:if><xsl:if test="(USED_FOR_QUERY='false' or GENERATE_INSECURE='false')">private</xsl:if> void setQuery<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x)
    throws DataObjectException, QueryException {
         setQuery<xsl:value-of select="common:capitalizeName(@name)"/>(x, QueryBuilder.EQUAL);
    }

    /**
     * Add <xsl:value-of select="@name"/> to the ORDER BY clause.
     * NOTE: The DO cache does not yet support the Order By operation.
     * Using the addOrderBy method forces the query to hit the database.
     *
     * @param direction_flag  True for ascending order, false for descending
     */
    public void addOrderBy<xsl:value-of select="common:capitalizeName(@name)"/>(boolean direction_flag) {
        hitDb = true;
        builder.addOrderByColumn(&quot;<xsl:value-of select="@name"/>&quot;, (direction_flag) ? "ASC" : "DESC");
    }


    /**
     * Add <xsl:value-of select="@name"/> to the ORDER BY clause.  This convenience
     * method assumes ascending order.
     * NOTE: The DO cache does not yet support the Order By operation.
     * Using the addOrderBy method forces the query to hit the database.
     */
    public void addOrderBy<xsl:value-of select="common:capitalizeName(@name)"/>() {
        hitDb = true;
        builder.addOrderByColumn(&quot;<xsl:value-of select="@name"/>&quot;,"ASC");
    }

<xsl:if test="/TABLE/DO_IS_OID_BASED='true' and GENERATE_SECURE='true'">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added:
    /**
     * Set the <xsl:value-of select="@name"/> to query.
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     * @param exact to use matches or not
     * @param usr The user for security check
     *
     * @exception DataObjectException If a database access error occurs.
     * @exception AccessException The user is not allowed to query this attribute
     * @exception QueryException
     *
     * @deprecated Instead use setQuery<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x,String cmp_op, org.webdocwf.dods.access.User usr)
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void setQuery<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, boolean exact, org.webdocwf.dods.access.User usr )
    throws DataObjectException, QueryException, AccessException  {
        String cmp_op = QueryBuilder.EQUAL;
        if ( ! exact )
            cmp_op = QueryBuilder.CASE_INSENSITIVE_CONTAINS;
        assertQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access( x, cmp_op, usr );
        setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( x, cmp_op );
    }

    /**
     * Set the <xsl:value-of select="@name"/> to query.
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     * @param usr The user for security check
     *
     * @exception DataObjectException If a database access error occurs.
     * @exception AccessException The user is not allowed to query this attribute
     * @exception QueryException
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void setQuery<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, org.webdocwf.dods.access.User usr )
    throws DataObjectException, QueryException, AccessException {
        assertQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access( x, QueryBuilder.EQUAL, usr );
        setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( x, QueryBuilder.EQUAL );
    }

    /**
     * Set the <xsl:value-of select="@name"/> to query, with a QueryBuilder comparison operator.
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     * @param cmp_op QueryBuilder comparison operator to use.
     * @param usr The user for security check
     *
     * @exception DataObjectException If a database access error occurs.
     * @exception QueryException If comparison operator is inappropriate
     * (e.g. CASE_SENSITIVE_CONTAINS on an integer field).
     * @exception AccessException
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void setQuery<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, String cmp_op, org.webdocwf.dods.access.User usr )
    throws DataObjectException, QueryException, AccessException {
        assertQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access( x, cmp_op, usr );
        setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( x, cmp_op );
    }

    /**
     * Ensure the given user is allowed to query the attribute
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     * @param exact to use matches or not
     * @param usr The user for security check
     *
     * @exception AccessException The user is not allowed to query this attribute
     *
     * @deprecated Instead use assertQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, String cmp_op, org.webdocwf.dods.access.User usr )
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void assertQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, boolean exact, org.webdocwf.dods.access.User usr )
    throws AccessException {
        String cmp_op = QueryBuilder.EQUAL;
        if ( ! exact )
            cmp_op = QueryBuilder.CASE_INSENSITIVE_CONTAINS;
        if (!hasQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access( x, cmp_op, usr ))
            throw new AccessRightException("No access !", usr, "QueryAttr", null, &quot;<xsl:value-of select="/TABLE/CLASS_NAME"/>DO&quot;, &quot;<xsl:value-of select="@name"/>&quot;, null, null, null, cmp_op);
    }

    /**
    * Ensure the given user is allowed to query the attribute
    *
    * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
    * @param cmp_op QueryBuilder comparison operator to use.
    * @param usr The user for security check
    *
    * @exception AccessException The user is not allowed to query this attribute
    *
    * WebDocWf extension
    *
    */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void assertQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, String cmp_op, org.webdocwf.dods.access.User usr )
    throws AccessException {
        if (!hasQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access( x, cmp_op, usr ))
            throw new AccessRightException("No access !", usr, "QueryAttr", null, &quot;<xsl:value-of select="/TABLE/CLASS_NAME"/>DO&quot;, &quot;<xsl:value-of select="@name"/>&quot;, null, null, null, cmp_op);
    }

    /**
     * Ensure the given user is allowed to query the attribute
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     * @param usr The user for security check
     *
     * @exception AccessException The user is not allowed to query this attribute
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void assertQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, org.webdocwf.dods.access.User usr )
    throws AccessException {
        if (!hasQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access( x, QueryBuilder.EQUAL, usr ))
            throw new AccessRightException("No access !", usr, "QueryAttr", null, &quot;<xsl:value-of select="/TABLE/CLASS_NAME"/>DO&quot;, &quot;<xsl:value-of select="@name"/>&quot;, null, null, null, QueryBuilder.EQUAL);
    }

    /**
     * Check whether the given user is allowed to query the attribute
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     * @param exact to use matches or not
     * @param usr The user for security check
     *
     * @return Whether the given user is allowed to query the attribute
     * @exception AccessEvalException
     *
     * @deprecated Instead use hasQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, String cmp_op, org.webdocwf.dods.access.User usr )
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> boolean hasQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, boolean exact, org.webdocwf.dods.access.User usr )
    throws AccessEvalException  {
        String cmp_op = QueryBuilder.EQUAL;
        if ( ! exact )
            cmp_op = QueryBuilder.CASE_INSENSITIVE_CONTAINS;
        return usr.hasQueryAttrAccess(&quot;<xsl:value-of select="translate(/TABLE/PACKAGE,'/','.')"/>.<xsl:value-of select="/TABLE/CLASS_NAME"/>DO&quot;, &quot;<xsl:value-of select="translate(/TABLE/PACKAGE,'/','.')"/>.<xsl:value-of select="/TABLE/CLASS_NAME"/>DO:<xsl:value-of select="@name"/>&quot;, x, cmp_op);
    }

    /**
     * Check whether the given user is allowed to query the attribute
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     * @param cmp_op QueryBuilder comparison operator to use.
     * @param usr The user for security check
     *
     * @return Whether the given user is allowed to query the attribute
     * @exception AccessEvalException
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> boolean hasQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, String cmp_op, org.webdocwf.dods.access.User usr )
    throws AccessEvalException {
        return usr.hasQueryAttrAccess(&quot;<xsl:value-of select="translate(/TABLE/PACKAGE,'/','.')"/>.<xsl:value-of select="/TABLE/CLASS_NAME"/>DO&quot;, &quot;<xsl:value-of select="translate(/TABLE/PACKAGE,'/','.')"/>.<xsl:value-of select="/TABLE/CLASS_NAME"/>DO:<xsl:value-of select="@name"/>&quot;, x, cmp_op);
    }

    /**
     * Check whether the given user is allowed to query the attribute
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query.
     * @param usr The user for security check
     *
     * @return Whether the given user is allowed to query the attribute
     * @exception AccessEvalException
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> boolean hasQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, org.webdocwf.dods.access.User usr )
    throws AccessEvalException {
        return usr.hasQueryAttrAccess(&quot;<xsl:value-of select="translate(/TABLE/PACKAGE,'/','.')"/>.<xsl:value-of select="/TABLE/CLASS_NAME"/>DO&quot;, &quot;<xsl:value-of select="translate(/TABLE/PACKAGE,'/','.')"/>.<xsl:value-of select="/TABLE/CLASS_NAME"/>DO:<xsl:value-of select="@name"/>&quot;, x, QueryBuilder.EQUAL);
    }
    // end of WebDocWf extension for DODS row instance security
</xsl:if>

    // WebDocWf extension for extended wildcard support
    // The following lines have been added:
<xsl:if test="JAVA_TYPE='String'">
    /**
     * Set the <xsl:value-of select="@name"/> to query with a user wildcard string
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query with user wildcards
     *
     * @exception DataObjectException If a database access error occurs.
     * @exception QueryException If a query error occurs.
     *
     * @deprecated Use comparison operators instead
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void setUserMatch<xsl:value-of select="common:capitalizeName(@name)"/>( <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x )
    throws DataObjectException, QueryException {
        <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> y = convertUserSearchValue( x );
        setDBMatch<xsl:value-of select="common:capitalizeName(@name)"/>( y );
    }

<xsl:if test="/TABLE/DO_IS_OID_BASED='true' and GENERATE_SECURE='true'">
    /**
     * Set the <xsl:value-of select="@name"/> to query with a user wildcard string
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query with user wildcards
     * @param usr The user for security check
     *
     * @exception DataObjectException If a database access error occurs.
     * @exception QueryException If a query error occurs.
     * @exception AccessException The user is not allowed to query this attribute
     *
     * @deprecated Use comparison operators instead
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void setUserMatch<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x , org.webdocwf.dods.access.User usr)
    throws DataObjectException, QueryException, AccessException {
        <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> y = convertUserSearchValue( x );
        setDBMatch<xsl:value-of select="common:capitalizeName(@name)"/>( y, usr );
    }
</xsl:if>

    /**
     * Set the <xsl:value-of select="@name"/> to query with a DB wildcard string
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query with DB wildcards
     *
     * @exception DataObjectException If a database access error occurs.
     * @exception QueryException If a query error occurs.
     *
     * @deprecated Use comparison operators instead
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void setDBMatch<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x )
    throws DataObjectException, QueryException {
        if (containsWildcards(x) || builder.getUserStringAppendWildcard()) {
            builder.addMatchClause( <xsl:value-of select="/TABLE/CLASS_NAME"/>DO.get<xsl:value-of select="common:capitalizeName(@name)"/>Column(this.dbtablename), x );
            hitDb = true;
        } else
            setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( x, QueryBuilder.EQUAL );
    }

<xsl:if test="/TABLE/DO_IS_OID_BASED='true' and GENERATE_SECURE='true'">
    /**
     * Set the <xsl:value-of select="@name"/> to query with a DB wildcard string
     *
     * @param x The <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> to query with DB wildcards
     * @param usr The user for security check
     *
     * @exception DataObjectException If a database access error occurs.
     * @exception AccessException The user is not allowed to query this attribute
     * @exception QueryException If a query error occurs.
     *
     * @deprecated Use comparison operators instead
     *
     * WebDocWf extension
     *
     */
    <xsl:if test="USED_FOR_QUERY='true'">public</xsl:if><xsl:if test="USED_FOR_QUERY='false'">private</xsl:if> void setDBMatch<xsl:value-of select="common:capitalizeName(@name)"/>( <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x, org.webdocwf.dods.access.User usr )
    throws DataObjectException, QueryException, AccessException {
        assertQueryAttr<xsl:value-of select="common:capitalizeName(@name)"/>Access( x, containsWildcards(x) || builder.getUserStringAppendWildcard(), usr );
        setDBMatch<xsl:value-of select="common:capitalizeName(@name)"/>( x );
    }
</xsl:if>
</xsl:if>
    // end of WebDocWf extension for extended wildcard support
</xsl:if><xsl:if test="(USED_FOR_QUERY='false' or GENERATE_INSECURE='false')">
    /* column <xsl:value-of select="@name"/> is not used for query */
</xsl:if>
</xsl:template>
</xsl:stylesheet>

