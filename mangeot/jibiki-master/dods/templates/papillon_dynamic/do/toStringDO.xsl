<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="toStringDO">    
                sb.append(&quot;\n&quot; + indent + &quot;<xsl:value-of select="@name"/><xsl:value-of select="dodst:memberRemarks(REFERENCE_OBJECT/IS_FOREIGN_KEY,IS_PRIMARY_KEY)"/>=&quot; + get_DataStruct().get<xsl:value-of select="common:capitalizeName(@name)"/>());<!--                sb.append(&quot;\n&quot; + indent + &quot;<xsl:value-of select="@name"/><xsl:value-of select="dodst:memberRemarks(REFERENCE_OBJECT/IS_FOREIGN_KEY,IS_PRIMARY_KEY)"/>=&quot; + <xsl:if test="REFERENCE_OBJECT"><xsl:value-of select="dodst:dumpMethod1(@name,'true',REFERENCE_OBJECT/IS_FOREIGN_KEY)"/></xsl:if><xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="dodst:dumpMethod1(@name,'false',REFERENCE_OBJECT/IS_FOREIGN_KEY)"/></xsl:if>); -->
</xsl:template>
</xsl:stylesheet>
