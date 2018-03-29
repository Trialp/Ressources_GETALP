<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

    <!-- Copy (deep) node -->
    <xsl:template match="node()|@*">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>

    <!-- Remove d:metadata node -->
    <xsl:template match="d:metadata" xmlns:d="http://www-clips.imag.fr/geta/services/dml"> 
    </xsl:template>

</xsl:stylesheet>
