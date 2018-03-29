<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                version="1.0">
<xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="resultSetExstractionDO">
<xsl:if test="not(IS_CONSTANT='true')">
<xsl:if test="not(DB_TYPE='LONGVARCHAR')">
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
        if (originDatabase != null) {
<xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
            if(get_refs()!=null &amp;&amp; get_refs().containsKey(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;)</xsl:if>)){
                set<xsl:value-of select="common:capitalizeName(@name)"/>(originDatabase, (<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)get_refs().get(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;) </xsl:if>));
            } else {
              if(get_refs()==null)
                refs = new HashMap();</xsl:if>
              set<xsl:value-of select="common:capitalizeName(@name)"/>(  <xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'"> <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>.ceInternal( rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;)</xsl:if>, get_refs(), get_transaction())</xsl:if>
                <xsl:if test="not(REFERENCE_OBJECT/IS_FOREIGN_KEY='false')"> rs.get<xsl:value-of select="miu:jdbcType(JAVA_TYPE)"/>( &quot;<xsl:value-of select="@name"/>&quot;  )</xsl:if> );
        }
        else {</xsl:if><!-- 14.XI <xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
          if(get_refs()!=null &amp;&amp; get_refs().containsKey(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;)</xsl:if>)){
             set<xsl:value-of select="common:capitalizeName(@name)"/>((<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)get_refs().get(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;) </xsl:if>));
          } else {
             if(get_refs()==null)
                refs = new HashMap();</xsl:if>
             set<xsl:value-of select="common:capitalizeName(@name)"/>( <xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'"> <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>.ceInternal( rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;)</xsl:if>, get_refs(), get_transaction() )</xsl:if> 14.XI -->
<xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">          oid_set<xsl:value-of select="common:capitalizeName(@name)"/>( rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal(</xsl:if> &quot;<xsl:value-of select="@name"/>&quot;)</xsl:if>
<xsl:if test="not(REFERENCE_OBJECT/IS_FOREIGN_KEY='false')">         set<xsl:value-of select="common:capitalizeName(@name)"/>( rs.get<xsl:value-of select="miu:jdbcType(JAVA_TYPE)"/>( &quot;<xsl:value-of select="@name"/>&quot;  )</xsl:if> );<!--
<xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
            }
</xsl:if>
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
            }
</xsl:if>-->
</xsl:if>
</xsl:if>
</xsl:template>
</xsl:stylesheet>
