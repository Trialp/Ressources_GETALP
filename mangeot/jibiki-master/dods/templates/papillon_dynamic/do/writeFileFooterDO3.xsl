<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="fileFooter3">
<xsl:if test="not(IS_CONSTANT='true')">

    private int[] param = null;
    private boolean isDeletedFromDatabase = false;
    /**
     * Prepares the statement used to insert this object
     * into the database.
     *
     * @param conn The database connection.
     *
     * @return The insert statement.
     *
     * @exception java.sql.SQLException if an error occurs.
     */
    public PreparedStatement getInsertStatement(DBConnection conn)
    throws SQLException {
        ObjectId oid;
    
	if (isDeletedFromDatabase) 
            throw new SQLException("Object "+get_OId()+" is already deleted");

        PreparedStatement stmt = conn.prepareStatement(  
            &quot;insert into &quot; + this.dbtablename + &quot; ( <xsl:value-of select="dodst:insertColumnNames()"/> )&quot; +
            &quot;values ( <xsl:value-of select="dodst:insertQuestionMarks()"/> )&quot; 
        );

        param = new int[1]; param[0] = 1;
        // writeMemberStuff uses the JDBCsetCalls.template
        // to build up the value for this tag:
        // the value is a series of calls to setPrepStmtParam_TYPE methods.
        // Those methods are defined in GenericDO.
        try {
</xsl:if>
</xsl:template>
</xsl:stylesheet>
