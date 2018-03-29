<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="fileFooter6"><xsl:if test="not(IS_CONSTANT='true')"><xsl:if test="DO_IS_OID_BASED='true'">            setPrepStmtParam_BigDecimal( stmt, param, get_OId().toBigDecimal() );
            setPrepStmtParam_int(stmt, param, get_Version() - 1);</xsl:if></xsl:if>
<xsl:if test="not(IS_CONSTANT='true')">
        } catch ( Exception e ) {
            e.printStackTrace();
            throw new SQLException( "Data Object error: " + e.getMessage() );
        }
        get_statistics(this.dbtablename).incrementUpdateNum();
        return stmt;
    }

    /**
     * Prepares the statement used to delete this object
     * from the database.
     *
     * @param conn The database connection
     *
     * @return The delete statement.
     *
     * @exception java.sql.SQLException if an error occurs.
     */
    public PreparedStatement getDeleteStatement(DBConnection conn)
    throws SQLException {
        String sql="";
        if(isDeleteCheckVersion())
            sql =<xsl:if test="DO_IS_OID_BASED='true'">
            &quot;delete from &quot; + this.dbtablename + &quot; \n&quot; +
            &quot;where &quot; + get_OIdColumnName() + &quot; = ? and &quot; + get_versionColumnName() + &quot; = ?&quot;;</xsl:if>
<xsl:if test="DO_IS_OID_BASED='false'">
            &quot;delete from &quot; + this.dbtablename + &quot; where <xsl:value-of select="dodst:updateWhereClause()"/>&quot;;</xsl:if>
        else    
            sql =<xsl:if test="DO_IS_OID_BASED='true'">
            &quot;delete from &quot; + this.dbtablename + &quot;\n&quot; +
            &quot;where &quot; + get_OIdColumnName() + &quot; = ?&quot;;</xsl:if>
<xsl:if test="DO_IS_OID_BASED='false'">
            &quot;delete from &quot; + this.dbtablename + &quot; where <xsl:value-of select="dodst:updateWhereClause()"/>&quot;;</xsl:if>
        PreparedStatement stmt = conn.prepareStatement(sql);<xsl:if test="DO_IS_OID_BASED='true'">
        stmt.setBigDecimal(1, get_OId().toBigDecimal());</xsl:if>
        if(isDeleteCheckVersion()) {
            stmt.setInt(2, get_Version());
        }
        get_statistics(this.dbtablename).incrementDeleteNum();
        isDeletedFromDatabase = true;
        return stmt;
    }
    
<xsl:if test="DO_IS_OID_BASED='false'">
    // for non-oid-based DOs, no fancy analysis on update failure.
    public synchronized void executeUpdate(DBConnection conn)
    throws SQLException, DBRowUpdateException {
        PreparedStatement stmt = getUpdateStatement(conn);
        if (stmt != null) {
            if ( 0 == conn.executeUpdate(stmt, "execute update") )
                    throw new DBRowUpdateException( "Zero rows were affected by the update.\n" + "The Primary Key was not found.\n" +
            this.toString() );
        }
    }</xsl:if></xsl:if>

    /*
     * toString - for debugging
     *
     * @return String for debugging.
     *
     */
    public String toString(){
        return toString( 1 );
    }

    public String toString( int indentCount ){
        String indent = "";
        for ( int i = 0; i &lt; indentCount; i++ ) {
            indent += ". ";
        }
        StringBuffer sb = new StringBuffer();
        sb.append(indent + &quot;<xsl:value-of select="CLASS_NAME"/>DO:&quot;);
<xsl:if test="DO_IS_OID_BASED='true'">
        ObjectId oid = get_OId();
        String id = "virgin";
        if ( null != oid )
            id = oid.toString();
        sb.append(" OID=" + id + ",VERSION=" + get_Version());</xsl:if>
        if (isLoaded()) {</xsl:template> 
</xsl:stylesheet>
