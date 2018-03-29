<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:stu="org.ejen.ext.StringUtil"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                version="1.0">
	<xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="cloning">

<xsl:if test="not(IS_CONSTANT='true')">            ret.<xsl:value-of select="@name"/> = <xsl:if test="REFERENCE_OBJECT">GenericDO.copyBigDecimal(<xsl:value-of select="@name"/>)</xsl:if><xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="miu:cloneValue(@name,JAVA_TYPE,'false')"/></xsl:if>;
</xsl:if>

</xsl:template>
</xsl:stylesheet>
