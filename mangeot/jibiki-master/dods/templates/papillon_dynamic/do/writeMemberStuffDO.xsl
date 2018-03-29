<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                version="1.0">
                
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="memberStuff">

    ////////////////////////// data member <xsl:value-of select="common:capitalizeName(@name)"/>
   <xsl:if test="not(IS_CONSTANT='true')">
    <xsl:if test="REFERENCE_OBJECT"><xsl:value-of select="dodst:addColumnName(@name,IS_PRIMARY_KEY,'true',USED_FOR_QUERY,REFERENCE_OBJECT/IS_FOREIGN_KEY,REFERENCE_OBJECT/IS_ABSTRACT)"/></xsl:if><xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="dodst:addColumnName(@name,IS_PRIMARY_KEY,'false',USED_FOR_QUERY,REFERENCE_OBJECT/IS_FOREIGN_KEY,REFERENCE_OBJECT/IS_ABSTRACT)"/></xsl:if>
    <xsl:value-of select="dodst:addFK(@name,REFERENCE_OBJECT/CONSTRAINT,REFERENCE_OBJECT/@name,REFERENCE_OBJECT/PACKAGE,REFERENCE_OBJECT/FOREIGN_KEY_COLUMN_NAME,REFERENCE_OBJECT/FOREIGN_KEY_GROUP)"/>

    /**
     * static final RDBColumn <xsl:value-of select="common:capitalizeName(@name)"/> for use with QueryBuilder.
     * See RDBColumn PrimaryKey at the top of this file for usage example.
     */
    //static public final RDBColumn <xsl:value-of select="common:capitalizeName(@name)"/> = new RDBColumn( table, &quot;<xsl:value-of select="@name"/>&quot;, <xsl:if test="CAN_BE_NULL='false'">true</xsl:if><xsl:if test="CAN_BE_NULL='true'">false</xsl:if>);
    public static RDBColumn get<xsl:value-of select="common:capitalizeName(@name)"/>Column(String dbtable) {
         RDBTable  table = new RDBTable(dbtable);
		 return new RDBColumn( table, &quot;<xsl:value-of select="@name"/>&quot;, false );
	 }

    private boolean changed<xsl:value-of select="common:capitalizeName(@name)"/> = false;

    /**
     * Use for query caching.
     */
    static public final int COLUMN_<xsl:value-of select="common:upperCaseName(@name)"/> = <xsl:value-of select="dodst:getCounter()"/>;
    <xsl:value-of select="dodst:incCounter()"/>
    

<xsl:if test="IS_PRIMARY_KEY='true'">
    /*
     * <xsl:value-of select="/TABLE/CLASS_NAME"/>DO does not use OID as the Primary Key.
     * The data member <xsl:value-of select="@name"/> is a PK for <xsl:value-of select="/TABLE/CLASS_NAME"/>DO.
     * The "get" method(s) for the column(s) that comprise the PK
     * test isSet_ booleans to determine if the value has been set.
     * If not, an exception is thrown.
     * Since the getInsertStatement() and getUpdateStatement() methods
     * call the "get" methods, if this DO does not have its PK column(s) set,
     * it will not be written to the database.
     * (This test is not necessary for oid-based DOs because the CoreDO
     * constructor allocates a PK (oid) value.)
     */
    private boolean isSet_<xsl:value-of select="@name"/> = false;
</xsl:if>
    <xsl:value-of select="dodst:declMaxLength(@name,SIZE,DB_TYPE,JAVA_TYPE)"/>
</xsl:if>

    /**
     * Get <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @return <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> get<xsl:value-of select="common:capitalizeName(@name)"/> ()
    throws DataObjectException {<xsl:if test="not(IS_CONSTANT='true')"><xsl:if test="/TABLE/DO_IS_OID_BASED='true'">
        checkLoad();
        </xsl:if><xsl:if test="IS_PRIMARY_KEY='true'">
        if ( ! isSet_<xsl:value-of select="@name"/> )
          throw new DataObjectException( &quot;Primary Key '<xsl:value-of select="@name"/>' unset.&quot; );</xsl:if></xsl:if><xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
        if (null == get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>()) {
            return null;
        }
        String qKey = get_logicalDBName()+"."+get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>().toString();
        <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> ret = (<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)getRefs(qKey);
        if(transaction!=null &amp;&amp; ret!=null &amp;&amp; ret.get_transaction()!=null) {
            if(!(ret.get_transaction()).equals(transaction))
                throw new DataObjectException ("Referenced DO doesn't belong this transaction ");
        } else if (null == ret) {
            try {
                ret = <xsl:value-of select="JAVA_TYPE"/>.ceInternal(get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>(), get_refs(), get_transaction());
                addRefs(qKey, ret);
            } catch (Exception e) {
                throw new DataObjectException("FIXME: should make proper log entry - didn't created ref", e); //FIXME: should make proper log entry
            }
        }</xsl:if></xsl:if>
        return <xsl:if test="not(REFERENCE_OBJECT)">get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>()</xsl:if><xsl:if test="REFERENCE_OBJECT">ret</xsl:if>;
    }
<xsl:if test="/TABLE/DO_IS_OID_BASED='true'"><xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">

    /**
     * Get BigDecimal value of <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @return <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public BigDecimal oid_get<xsl:value-of select="common:capitalizeName(@name)"/> ()
    throws DataObjectException {<xsl:if test="not(IS_CONSTANT='true')">
        checkLoad();
        <xsl:if test="IS_PRIMARY_KEY='true'">
        if ( ! isSet_<xsl:value-of select="@name"/> )
          throw new DataObjectException( &quot;Primary Key '<xsl:value-of select="@name"/>' unset.&quot; );</xsl:if></xsl:if>
        return get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>();
    }</xsl:if></xsl:if></xsl:if>

    /**
     * Get original <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @return <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> originalData_get<xsl:value-of select="common:capitalizeName(@name)"/> ()
    throws DataObjectException {<xsl:if test="not(IS_CONSTANT='true')"><xsl:if test="/TABLE/DO_IS_OID_BASED='true'">
        checkLoad();</xsl:if><xsl:if test="IS_PRIMARY_KEY='true'">
        if ( ! isSet_<xsl:value-of select="@name"/> )
            throw new DataObjectException( &quot;Primary Key '<xsl:value-of select="@name"/>' unset.&quot; );</xsl:if></xsl:if><xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
        if (null ==((<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct)originalData_get()).get<xsl:value-of select="common:capitalizeName(@name)"/>()) {
            return null;
        }
        String qKey = get_logicalDBName()+"."+((<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct)originalData_get()).get<xsl:value-of select="common:capitalizeName(@name)"/>().toString();
        <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> ret = (<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)getRefs(qKey);
        if(transaction!=null &amp;&amp; ret!=null &amp;&amp; ret.get_transaction()!=null) {
            if(!(ret.get_transaction()).equals(transaction))
                throw new DataObjectException ("Referenced DO doesn't belong this transaction ");
        } else if (null == ret) {
          try {
            ret = <xsl:value-of select="JAVA_TYPE"/>.ceInternal(((<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct)originalData_get()).get<xsl:value-of select="common:capitalizeName(@name)"/>(), get_refs(), get_transaction());
            addRefs(qKey, ret);
            } catch (Exception e) {
                throw new DataObjectException("FIXME: should make proper log entry - didn't created ref", e); //FIXME: should make proper log entry
            }
        }</xsl:if></xsl:if>
        return <xsl:if test="not(REFERENCE_OBJECT)">((<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct)originalData_get()).get<xsl:value-of select="common:capitalizeName(@name)"/>()</xsl:if><xsl:if test="REFERENCE_OBJECT">ret</xsl:if>;
    }

<xsl:if test="/TABLE/DO_IS_OID_BASED='true' and GENERATE_SECURE='true'">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added
    /**
     * Get <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @param usr The user for security check.
     *
     * @return <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception AccessException
     *   If the user is not allowed to read the attribute(value).
     *
     * WebDocWf extension
     *
     */
    public <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> get<xsl:value-of select="common:capitalizeName(@name)"/> ( org.webdocwf.dods.access.User usr )
    throws DataObjectException, AccessException {
<xsl:if test="GENERATE_INSECURE='true'">
        <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> ret = get<xsl:value-of select="common:capitalizeName(@name)"/>();
        return ret;
</xsl:if>
<xsl:if test="GENERATE_INSECURE='false'">
        <xsl:if test="not(IS_CONSTANT='true')"><xsl:if test="/TABLE/DO_IS_OID_BASED='true'">
        checkLoad();</xsl:if><xsl:if test="IS_PRIMARY_KEY='true'">
        if ( ! isSet_<xsl:value-of select="@name"/> )
          throw new DataObjectException( &quot;Primary Key '<xsl:value-of select="@name"/>' unset.&quot; );</xsl:if></xsl:if><xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
        if (null == get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>()) {
            return null;
        }
        String qKey = get_logicalDBName()+"."+get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>().toString();
        <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> ret = (<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)getRefs(qKey);
        if(transaction!=null &amp;&amp; ret!=null &amp;&amp; ret.get_transaction()!=null) {
            if(!(ret.get_transaction()).equals(transaction))
                throw new DataObjectException ("Referenced DO doesn't belong this transaction ");
        } else if (null == ret) {
          try {
            ret = <xsl:value-of select="JAVA_TYPE"/>.ceInternal(get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>(), get_refs(), get_transaction());
            addRefs(qKey, ret);
            } catch (Exception e) {
                throw new DataObjectException("FIXME: should make proper log entry - didn't created ref", e); //FIXME: should make proper log entry
            }
        }</xsl:if></xsl:if>
        return <xsl:if test="not(REFERENCE_OBJECT)">get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>()</xsl:if><xsl:if test="REFERENCE_OBJECT">ret</xsl:if>;
</xsl:if>
    }

    // WebDocWf extension for DODS row instance security
    // The following lines have been added
    /**
     * Get original <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @param usr The user for security check.
     *
     * @return <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception AccessException
     *   If the user is not allowed to read the attribute(value).
     *
     * WebDocWf extension
     *
     */
    public <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> originalData_get<xsl:value-of select="common:capitalizeName(@name)"/> ( org.webdocwf.dods.access.User usr )
    throws DataObjectException, AccessException {
<xsl:if test="GENERATE_INSECURE='true'">
        <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> ret = originalData_get<xsl:value-of select="common:capitalizeName(@name)"/>();
        return ret;
</xsl:if>
<xsl:if test="GENERATE_INSECURE='false'">
        <xsl:if test="not(IS_CONSTANT='true')"><xsl:if test="/TABLE/DO_IS_OID_BASED='true'">
        checkLoad();</xsl:if><xsl:if test="IS_PRIMARY_KEY='true'">
        if ( ! isSet_<xsl:value-of select="@name"/> )
          throw new DataObjectException( &quot;Primary Key '<xsl:value-of select="@name"/>' unset.&quot; );</xsl:if></xsl:if><xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
        if (null == get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>()) {
            return null;
        }
        String qKey = get_logicalDBName()+"."+get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>().toString();
        <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> ret = (<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)getRefs(qKey);
        if(transaction!=null &amp;&amp; ret!=null &amp;&amp; ret.get_transaction()!=null) {
            if(!(ret.get_transaction()).equals(transaction))
                throw new DataObjectException ("Referenced DO doesn't belong this transaction ");
        } else if (null == ret) {
          try {
            ret = <xsl:value-of select="JAVA_TYPE"/>.ceInternal(((<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct)originalData_get()).get<xsl:value-of select="common:capitalizeName(@name)"/>(), get_refs(), get_transaction());
            addRefs(qKey, ret);
            } catch (Exception e) {
                throw new DataObjectException("FIXME: should make proper log entry - didn't created ref", e); //FIXME: should make proper log entry
            }
          }</xsl:if></xsl:if>
        return <xsl:if test="not(REFERENCE_OBJECT)">get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>()</xsl:if><xsl:if test="REFERENCE_OBJECT">ret</xsl:if>;
</xsl:if>
    }
    // end of WebDocWf extension for DODS row instance security
</xsl:if>

   <xsl:if test="not(IS_CONSTANT='true')">
   <xsl:if test="IS_PRIMARY_KEY='true'">
    /**
     * Original value of <xsl:value-of select="@name"/> first retrieved from database.
     * This original value is set by ceInternal(ResultSet).
     * It is used to construct the WHERE clause for UPDATEs,
     * in case a user-defined (non-oid) Primary Key value has changed.
     */
    private <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> orig_<xsl:value-of select="@name"/> = <xsl:value-of select="miu:fixDefault(JAVA_TYPE,JAVA_DEFAULT_VALUE)"/>;
</xsl:if>

    /**
     * Set <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @param <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void set<xsl:value-of select="common:capitalizeName(@name)"/> ( <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/><xsl:text> </xsl:text><xsl:value-of select="@name"/> )
    throws DataObjectException {
<xsl:choose><xsl:when test="REFERENCE_OBJECT/IS_FOREIGN_KEY='true'">
      throw new DataObjectException( &quot;\nCalls to <xsl:value-of select="/TABLE/CLASS_NAME"/>DO.set<xsl:value-of select="common:capitalizeName(@name)"/>() are disallowed. &quot;+
                                     &quot;\n<xsl:value-of select="/TABLE/CLASS_NAME"/>DO.<xsl:value-of select="common:capitalizeName(@name)"/> is part of a &quot; +
                                     &quot;multicolumn foreign key. &quot; + &quot;\nTo ensure the foreign key is set completely, call &quot; +
                                     &quot;\n   <xsl:value-of select="/TABLE/CLASS_NAME"/>DO.<xsl:value-of select="SET_COMBO_FK_USING_DO_METHOD"/>()\n&quot; );
    </xsl:when><xsl:otherwise>        _set<xsl:value-of select="common:capitalizeName(@name)"/> ( <xsl:value-of select="@name"/> );</xsl:otherwise>
</xsl:choose>
    }
    
    /**
     * _set<xsl:value-of select="common:capitalizeName(@name)"/> is a protected method that is called by
     * set<xsl:value-of select="common:capitalizeName(@name)"/> if <xsl:value-of select="@name"/> is not part of
     * a multicolumn foreign key.
     *
     * @param <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    protected void _set<xsl:value-of select="common:capitalizeName(@name)"/> ( <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/><xsl:text> </xsl:text><xsl:value-of select="@name"/> )
    throws DataObjectException {<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">
<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
        if(transaction!=null &amp;&amp; <xsl:value-of select="@name"/>!=null &amp;&amp; <xsl:value-of select="@name"/>.get_transaction()!=null) {
            if(!<xsl:value-of select="@name"/>.get_transaction().equals(this.transaction))
                throw new DataObjectException ("Referenced DO doesn't belong this transaction ");
         }</xsl:if></xsl:if>
        checkLoad();
        <xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
        if (!originDatabase.equals(<xsl:value-of select="@name"/>.get_OriginDatabase()))
            throw new DataObjectException("Logical database of referenced object is different from DO's.");</xsl:if></xsl:if></xsl:if></xsl:if>
        try {
            checkDup();
        } catch (Exception e) {
            throw new DataObjectException ("Coudn't duplicate DataStruct:", e);
        }<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
        //
        BigDecimal bdNewOId = null;
        if (null != <xsl:value-of select="@name"/>) {
            bdNewOId = <xsl:value-of select="@name"/>.get_OId().toBigDecimal();
            String qKey = get_logicalDBName()+"."+ bdNewOId.toString();
            addRefs(qKey, <xsl:value-of select="@name"/>);
        }
        //</xsl:if></xsl:if>
        if (data.isEmpty)
            data.isEmpty = false;<!--
        colChanged = changed<xsl:value-of select="common:capitalizeName(@name)"/> ;-->
        data.set<xsl:value-of select="common:capitalizeName(@name)"/>(markNewValue(get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>(),  <xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="@name"/></xsl:if><xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">bdNewOId</xsl:if></xsl:if><xsl:value-of select="dodst:markNewValueExtras(@name,DB_TYPE,JAVA_TYPE,CAN_BE_NULL)"/>));
        changed<xsl:value-of select="common:capitalizeName(@name)"/> |= colChanged;
        if (changed<xsl:value-of select="common:capitalizeName(@name)"/>) {
            if (autoSaveAllowed&amp;&amp;isAutoSave()&amp;&amp;null != transaction) {
                try {
                    save(transaction,false);
                } catch (Exception ex) {
                    throw new DataObjectException("Error during transaction's writting data into database",ex);
                }
            }
        }<xsl:if test="IS_PRIMARY_KEY='true'">
        if ( ! isSet_<xsl:value-of select="@name"/> )
            orig_<xsl:value-of select="@name"/> = <xsl:value-of select="@name"/>;
        isSet_<xsl:value-of select="@name"/> = true;</xsl:if>
    }

<!-- -->
<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
    /**
     * Set <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     * 
     * @param <xsl:value-of select="@name"/> value of <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> as a BigDecimal value.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void oid_set<xsl:value-of select="common:capitalizeName(@name)"/>(BigDecimal<xsl:text> </xsl:text><xsl:value-of select="@name"/>)
    throws DataObjectException {<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">
        checkLoad();
        <xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
        if (!originDatabase.equals(<xsl:value-of select="@name"/>.get_OriginDatabase()))
            throw new DataObjectException("Logical database of referenced object is different from DO's.");</xsl:if></xsl:if></xsl:if></xsl:if>
        try {
            checkDup();
        } catch (Exception e) {
            throw new DataObjectException ("Coudn't duplicate DataStruct:", e);
        }
        if (data.isEmpty)
            data.isEmpty = false;<!--
        colChanged = changed<xsl:value-of select="common:capitalizeName(@name)"/> ;  -->
        data.set<xsl:value-of select="common:capitalizeName(@name)"/>(markNewValue(get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>(), <xsl:value-of select="@name"/>));
        changed<xsl:value-of select="common:capitalizeName(@name)"/> |= colChanged;
        if (changed<xsl:value-of select="common:capitalizeName(@name)"/>) {
            if (autoSaveAllowed&amp;&amp;isAutoSave()&amp;&amp;null != transaction) {
                try {
                    save(transaction,false);
                } catch (Exception ex) {
                    throw new DataObjectException("Error during transaction's writting data into database",ex);
                }
            }
        }<xsl:if test="IS_PRIMARY_KEY='true'">
        if ( ! isSet_<xsl:value-of select="@name"/> )
            orig_<xsl:value-of select="@name"/> = <xsl:value-of select="@name"/>;
        isSet_<xsl:value-of select="@name"/> = true;</xsl:if>
    }
<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">    
    /**
     * Set <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     * 
     * @param <xsl:value-of select="@name"/> value of <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> as a ObjectId value.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void oid_set<xsl:value-of select="common:capitalizeName(@name)"/>(ObjectId<xsl:text> </xsl:text><xsl:value-of select="@name"/>)
    throws DataObjectException {
        if(<xsl:value-of select="@name"/> != null)
            oid_set<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="@name"/>.toBigDecimal());
        else
            oid_set<xsl:value-of select="common:capitalizeName(@name)"/>((BigDecimal)null);    
    }
</xsl:if><xsl:if test="/TABLE/DO_IS_OID_BASED='true'">
    /**
     * Set <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     * 
     * @param <xsl:value-of select="@name"/> value of <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/> as a String value.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void oid_set<xsl:value-of select="common:capitalizeName(@name)"/>(String<xsl:text> </xsl:text><xsl:value-of select="@name"/>)
    throws DataObjectException {
        if(<xsl:value-of select="@name"/> != null)
            oid_set<xsl:value-of select="common:capitalizeName(@name)"/>(new BigDecimal(<xsl:value-of select="@name"/>));
        else
            oid_set<xsl:value-of select="common:capitalizeName(@name)"/>((BigDecimal)null);    
    }
</xsl:if>    
</xsl:if></xsl:if>
<!-- -->
<xsl:if test="/TABLE/DO_IS_OID_BASED='true' and GENERATE_SECURE='true'">
    // WebDocWf extension for DODS row instance security
    // The following lines have been added
    /**
     * Set <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     *
     * @param <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>.
     * @param usr The user for security check.
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     * @exception AccessException
     *   If the user is not allowed to read the attribute(value)
     *
     * WebDocWf extension
     *
     */
    public void set<xsl:value-of select="common:capitalizeName(@name)"/> ( <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/><xsl:text> </xsl:text><xsl:value-of select="@name"/>, org.webdocwf.dods.access.User usr )
    throws DataObjectException, AccessException {
<xsl:if test="GENERATE_INSECURE='true'">
        set<xsl:value-of select="common:capitalizeName(@name)"/>( <xsl:value-of select="@name"/> );
</xsl:if>
<xsl:if test="GENERATE_INSECURE='false'">
<xsl:choose><xsl:when test="REFERENCE_OBJECT/IS_FOREIGN_KEY='true'">
      throw new DataObjectException( &quot;\nCalls to <xsl:value-of select="/TABLE/CLASS_NAME"/>DO.set<xsl:value-of select="common:capitalizeName(@name)"/>() are disallowed. &quot;+
                                     &quot;\n<xsl:value-of select="/TABLE/CLASS_NAME"/>DO.<xsl:value-of select="common:capitalizeName(@name)"/> is part of a &quot; +
                                     &quot;multicolumn foreign key. &quot; + &quot;\nTo ensure the foreign key is set completely, call &quot; +
                                     &quot;\n   <xsl:value-of select="/TABLE/CLASS_NAME"/>DO.<xsl:value-of select="SET_COMBO_FK_USING_DO_METHOD"/>()\n&quot; );
    </xsl:when><xsl:otherwise>        _set<xsl:value-of select="common:capitalizeName(@name)"/> ( <xsl:value-of select="@name"/> );</xsl:otherwise>
</xsl:choose>
</xsl:if>
        
    }
    
</xsl:if>
</xsl:if>
</xsl:template>
</xsl:stylesheet>

