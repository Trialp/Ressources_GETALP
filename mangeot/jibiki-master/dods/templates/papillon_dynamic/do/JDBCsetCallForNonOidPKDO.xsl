<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="jdbcSetCallForNonOidPKDO">            if (changed<xsl:value-of select="common:capitalizeName(@name)"/>)
                updateStmt.append(&quot;, <xsl:value-of select="common:capitalizeName(@name)"/> = ? &quot;);
</xsl:template>
<!--
<xsl:template name="jdbcSetCallForNonOidPKDO">            if (changed<xsl:value-of select="common:capitalizeName(@name)"/>)
            setPrepStmtParam_<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">DO</xsl:if></xsl:if><xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="miu:javaType(JAVA_TYPE)"/></xsl:if><xsl:if test="REFERENCE_OBJECT"><xsl:if test="not(REFERENCE_OBJECT/IS_FOREIGN_KEY='false')"><xsl:value-of select="miu:javaType(JAVA_TYPE)"/></xsl:if></xsl:if>( stmt, param, get<xsl:value-of select="common:capitalizeName(@name)"/>() );
</xsl:template>
-->
</xsl:stylesheet>
