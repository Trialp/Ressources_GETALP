<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
        <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="compareCond">              case COLUMN_<xsl:value-of select="common:upperCaseName(@name)"/>:
<!-- <xsl:if test="REFERENCE_OBJECT">                   return QueryBuilder.compare((null != <xsl:value-of select="@name"/>)?(<xsl:value-of select="JAVA_TYPE"/>.createExisting(<xsl:value-of select="@name"/>,(DBTransaction)null)):null,cond.getValue(),cond.getOperator());
</xsl:if>
<xsl:if test="not(REFERENCE_OBJECT)"> -->                   return QueryBuilder.compare(get<xsl:value-of select="common:capitalizeName(@name)"/>(),cond.getValue(),cond.getOperator());
<!-- </xsl:if> -->
</xsl:template>
</xsl:stylesheet>
