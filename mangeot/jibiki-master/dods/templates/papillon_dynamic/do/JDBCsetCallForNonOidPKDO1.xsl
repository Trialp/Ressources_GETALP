<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="jdbcSetCallForNonOidPKDO1">            if (changed<xsl:value-of select="common:capitalizeName(@name)"/>) {
                setPrepStmtParam_<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">BigDecimal</xsl:if></xsl:if><xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="miu:javaType(JAVA_TYPE)"/></xsl:if><xsl:if test="REFERENCE_OBJECT"><xsl:if test="not(REFERENCE_OBJECT/IS_FOREIGN_KEY='false')"><xsl:value-of select="miu:javaType(JAVA_TYPE)"/></xsl:if></xsl:if>( stmt, param, <xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">oid_</xsl:if></xsl:if>get<xsl:value-of select="common:capitalizeName(@name)"/>());
                changed<xsl:value-of select="common:capitalizeName(@name)"/> = false;
            }
<xsl:if test="DO_IS_OID_BASED='false'">            setPrepStmtParam_<xsl:if test="REFERENCE_OBJECT">DO</xsl:if><xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="miu:javaType(JAVA_TYPE)"/></xsl:if>( stmt, param, orig_<xsl:value-of select="@name"/> );</xsl:if>
</xsl:template>
</xsl:stylesheet>
