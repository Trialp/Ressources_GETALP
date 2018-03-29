<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="columnsNameString">
			columnsNameString += tableName + &quot;<xsl:value-of select="@name"/>, &quot;;
</xsl:template>
</xsl:stylesheet>