<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="stubForCommitAssertionDO">  
<xsl:if test="REFERENCE_OBJECT">
<xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
    /**
     * A stub method for implementing pre-commit assertions 
     * for the <xsl:value-of select="@name"/> data member.
     * Implement this stub to throw an RefAssertionException for cases
     * where <xsl:value-of select="@name"/> is not valid for writing to the database.
     *
     * @param member <xsl:value-of select="@name"/> data member. 
     * @exception RefAssertionException
     *
     */
    protected void okToCommit<xsl:value-of select="common:capitalizeName(@name)"/>( <xsl:value-of select="REFERENCE_OBJECT/PACKAGE"/>.<xsl:value-of select="REFERENCE_OBJECT/@name"/>DO member ) 
    throws RefAssertionException { }

    /**
     * A stub method for implementing pre-delete assertions 
     * for the <xsl:value-of select="@name"/> data member.
     * Implement this stub to throw an RefAssertionException for cases
     * where <xsl:value-of select="@name"/> is not valid for deletion from the database.
     *
     * @param member <xsl:value-of select="@name"/> data member
     * @exception RefAssertionException
     *
     */
    protected void okToDelete<xsl:value-of select="common:capitalizeName(@name)"/>( <xsl:value-of select="REFERENCE_OBJECT/PACKAGE"/>.<xsl:value-of select="REFERENCE_OBJECT/@name"/>DO member ) 
    throws RefAssertionException { }
</xsl:if></xsl:if>
</xsl:template>
</xsl:stylesheet>

