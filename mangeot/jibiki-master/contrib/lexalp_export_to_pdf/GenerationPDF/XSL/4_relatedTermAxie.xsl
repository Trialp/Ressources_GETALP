<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
    
    <!-- Copy (deep) node -->
    <xsl:template match="node()|@*">
        <!-- Copy the current node -->
        <xsl:copy>
            <!-- Including any attributes it has and any child nodes -->
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>
    
    <!-- Improve information of related terms-->
    <xsl:template match="relatedTerm" xmlns:d="http://www-clips.imag.fr/geta/services/dml"> 
        <xsl:variable
            name="termreference"
            select="@termref">
        </xsl:variable>
        
        <xsl:copy>
            <xsl:attribute name="isHarmonised"><xsl:value-of select="@isHarmonised"/></xsl:attribute>
            <xsl:attribute name="relationToTerm"><xsl:value-of select="@relationToTerm"/></xsl:attribute>
            <xsl:attribute name="termref"><xsl:value-of select="@termref"/></xsl:attribute>
            <xsl:for-each select="//entry[@id=$termreference]">
                <xsl:value-of select="term"/>
            </xsl:for-each>
        </xsl:copy>
    </xsl:template>
    
    <!-- Add terms in axie -->
    <xsl:template match="termref" xmlns:d="http://www-clips.imag.fr/geta/services/dml"> 
        <xsl:variable
            name="termreference"
            select="@idref">
        </xsl:variable>
                <xsl:for-each select="//entry[@id=$termreference]">
                    <xsl:apply-templates select="."/>
                </xsl:for-each>
    </xsl:template>
    
</xsl:stylesheet>
