<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
	<xsl:output method="text" encoding="iso-8859-1"/>
	<xsl:template name="setQueryMethodsHeader">
    /**
     * Set the <xsl:value-of select="@name"/> to query.
     *
     * @param x The <xsl:value-of select="../@do_name"/> of the <xsl:value-of select="../@table_name"/> table to query.
     * @exception DataObjectException If a database access error occurs.
     * @exception QueryException If comparison operator is inappropriate
     * (e.g. CASE_SENSITIVE_CONTAINS on an integer field).
     */
    public void setQuery<xsl:value-of select="common:capitalizeName(@name)"/>( <xsl:value-of select="../@do_name"/> x )
    	throws DataObjectException, QueryException
    {
			openParen();
	</xsl:template>
</xsl:stylesheet>
