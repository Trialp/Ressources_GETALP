<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
	<xsl:output method="text" encoding="iso-8859-1"/>
	<xsl:template name="setQueryMethods">
<xsl:if test="miu:hasMoreElements()='true'">
    /**
     * Set the <xsl:value-of select="dodst:getFKGroup()"/> to query.
     *
     * @param x The <xsl:value-of select="/TABLE/CLASS_NAME"/>DO of the <xsl:value-of select="/TABLE/TABLE_NAME"/> table to query.
     * @exception DataObjectException If a database access error occurs.
     * @exception QueryException If comparison operator is inappropriate
     * (e.g. CASE_SENSITIVE_CONTAINS on an integer field).
     */
    public void setQuery<xsl:value-of select="common:capitalizeName(dodst:getFKGroup())"/>( <xsl:value-of select="dodst:getFKRef()"/>DO x )
    	throws DataObjectException, QueryException
    {
			openParen();
		<xsl:value-of select="dodst:setQueryCalls()"/>
			closeParen();
    }
    </xsl:if>
	</xsl:template>
</xsl:stylesheet>
