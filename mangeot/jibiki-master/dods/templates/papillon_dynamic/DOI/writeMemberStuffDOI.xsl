<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
                
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="middle">
<xsl:if test="GENERATE_INSECURE='true'">
    ////////////////////////// data member <xsl:value-of select="common:capitalizeName(@name)"/>

    /**
     * Get <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>
     *
     * @return <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public <xsl:value-of select="JAVA_TYPE"/> get<xsl:value-of select="common:capitalizeName(@name)"/> () 
    throws DataObjectException;
    
<xsl:if test="IS_CONSTANT='false'">
    /**
     * Set <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>
     *
     * @param <xsl:value-of select="@name"/> of the <xsl:value-of select="/TABLE/TABLE_NAME"/>
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void set<xsl:value-of select="common:capitalizeName(@name)"/> ( <xsl:value-of select="JAVA_TYPE"/> <xsl:text> </xsl:text><xsl:value-of select="@name"/> )
    throws DataObjectException;
</xsl:if>
</xsl:if>
</xsl:template>
</xsl:stylesheet>

