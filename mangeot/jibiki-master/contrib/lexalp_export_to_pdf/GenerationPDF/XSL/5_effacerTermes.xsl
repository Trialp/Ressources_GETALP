<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

    <!-- Copy (deep) node -->
    <xsl:template match="node()|@*">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>

	<!-- Remove terms in axie part and term part -->
    <xsl:template match="//d:contribution[not(@alone='yes')]" xmlns:d="http://www-clips.imag.fr/geta/services/dml"> 
        <xsl:variable
            name="termreference"
            select="d:data/entry/@id">
        </xsl:variable>
        
        <xsl:choose>
            <xsl:when test="//d:data/axie/entry[@id=$termreference]" xmlns:d="http://www-clips.imag.fr/geta/services/dml">
            </xsl:when>
            <xsl:otherwise>
                <d:contribution>
                        <xsl:attribute name="alone">yes</xsl:attribute>
                        <xsl:apply-templates select="@*|node()"/>
                </d:contribution>
            </xsl:otherwise>
        </xsl:choose>

        
    </xsl:template>

</xsl:stylesheet>
