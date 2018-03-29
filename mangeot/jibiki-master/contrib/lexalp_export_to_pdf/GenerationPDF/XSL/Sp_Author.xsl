<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

    <!-- Copy (deep) node -->
    <xsl:template match="node()|@*">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>

	<!-- Remove axies without term which author is IPrzysucha -->
    <xsl:template match="//d:contribution[not(@author='IPrzysucha')]" xmlns:d="http://www-clips.imag.fr/geta/services/dml"> 
        
        <xsl:choose>
            <xsl:when test="d:data/axie/entry[@author='IPrzysucha' and @lang='deu']" xmlns:d="http://www-clips.imag.fr/geta/services/dml">
              <d:contribution>
                        <xsl:attribute name="author">IPrzysucha</xsl:attribute>
                        <xsl:apply-templates select="@*|node()"/>
                </d:contribution>
            </xsl:when>
            <xsl:otherwise>
            </xsl:otherwise>
        </xsl:choose>
        
    </xsl:template>


</xsl:stylesheet>
