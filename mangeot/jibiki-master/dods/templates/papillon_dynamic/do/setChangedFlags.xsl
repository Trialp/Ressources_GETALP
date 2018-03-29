<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="setChangedFlags">        changed<xsl:value-of select="common:capitalizeName(@name)"/> = value;
</xsl:template>
</xsl:stylesheet>
