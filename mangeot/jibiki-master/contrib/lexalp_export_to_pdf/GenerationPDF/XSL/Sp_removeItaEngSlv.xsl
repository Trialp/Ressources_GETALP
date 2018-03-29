<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

    <!-- Copy (deep) node -->
    <xsl:template match="node()|@*">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>

	<!-- Remove term ita slv eng -->
   	<xsl:template match="//fo:block[contains(@id,'ita.') or contains(@id,'slv.') or contains(@id,'eng.')]" xmlns:fo="http://www.w3.org/1999/XSL/Format">
    </xsl:template>


</xsl:stylesheet>
