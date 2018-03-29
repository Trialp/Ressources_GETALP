<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                version="1.0">
	<xsl:output method="text" encoding="iso-8859-1"/>
	<xsl:template name="fK_methods">
<xsl:if test="dodst:hasMoreElements()='true'">

    /**
     * Get referenced DO by doing a query using foreign key fields in this DO.
     *
     * @return Referenced DO by doing a query using foreign key fields.
     * 
     * @exception DataObjectException If a database or data layer error occurs.
     */
    public <xsl:value-of select="dodst:getFKRef()"/>DO get<xsl:value-of select="dodst:getMethodName()"/> () 
    throws DataObjectException {
        try {
	    <xsl:value-of select="dodst:getFKRef()"/>Query q = new <xsl:value-of select="dodst:getFKRef()"/>Query();
<xsl:value-of select="dodst:getFKsetQueryCalls()"/>
	    q.requireUniqueInstance();
	    return q.getNextDO();
        } catch ( Exception e ) { 
	    throw new DataObjectException( "Query Failed: ", e );
        }
    }

    /**
     * Set foreign key fields from primary key fields in referenced DO.
     * 
     * @param x the DO which is the referenced object.
     * The primary key fields from that objects are used
     * to set foreign key values in this DO.
     *
     * @exception DataObjectException If a database or data layer error occurs.
     */
    public void set<xsl:value-of select="dodst:setMethodName()"/>( <xsl:value-of select="dodst:getFKRef()"/>DO x ) 
    throws DataObjectException {
	if ( null == x ) 
	    throw new DataObjectException( 
		&quot;Argument for set<xsl:value-of select="dodst:setMethodName()"/>() cannot be null.  &quot; +
		&quot;This method sets foreign key values in <xsl:value-of select="/TABLE/CLASS_NAME"/>DO &quot; +
		&quot;using primary key values from a <xsl:value-of select="dodst:getFKRef()"/>DO object.&quot; );
    <xsl:value-of select="dodst:getFKsetCalls()"/>
    }
</xsl:if>
	</xsl:template>
</xsl:stylesheet>
	
