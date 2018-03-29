<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
        <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="compareCond">        case COLUMN_<xsl:value-of select="common:upperCaseName(@name)"/>:
<xsl:if test="(REFERENCE_OBJECT/IS_FOREIGN_KEY='false')">                    
                    if ((get<xsl:value-of select="common:capitalizeName(@name)"/>()!=null)&amp;&amp; (get<xsl:value-of select="common:capitalizeName(@name)"/>() instanceof CoreDO)) {
                     CoreDataStruct xDataStruct = (CoreDataStruct)get<xsl:value-of select="common:capitalizeName(@name)"/>().get_DataStruct(); 
                     return QueryBuilder.compare(xDataStruct,cond.getValue(),cond.getOperator());
                    } 
                    else
</xsl:if>                    
                     return QueryBuilder.compare(get<xsl:value-of select="common:capitalizeName(@name)"/>(),cond.getValue(),cond.getOperator());
</xsl:template>
</xsl:stylesheet>
