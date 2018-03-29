<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
<xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="resultSetExstractionForLONGDO">
<xsl:if test="not(IS_CONSTANT='true')"> <xsl:if test="DB_TYPE='LONGVARCHAR'"> {
        java.io.Reader ir = rs.getCharacterStream( &quot;<xsl:value-of select="@name"/>&quot; );
        if (ir != null)  {
            int bufSize = 2000;
            char[] buf = new char[bufSize];
            StringBuffer sb = new StringBuffer();
            int n = 0;
            try {
                while ( -1 != ( n = ir.read( buf ) ) ) {
                    sb.append( buf, 0, n );
                }
            } catch ( Exception e ) {
                throw new DataObjectException("IO error processing input stream for column <xsl:value-of select="@name"/>: ", e );
            }
<xsl:if test="DO_IS_MULTIDB_BASED='true'">
            if (originDatabase != null) {
<xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
            if(get_refs()!=null &amp;&amp; get_refs().containsKey(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;)</xsl:if>)){
                set<xsl:value-of select="common:capitalizeName(@name)"/>(originDatabase, (<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)get_refs().get(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;) </xsl:if>));
            } else {
              if(get_refs()==null)
                refs = new HashMap();</xsl:if>
                set<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'"><xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>.ceInternal( originDatabase,new BigDecimal(</xsl:if></xsl:if>sb.toString()<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">),get_refs(), get_transaction())</xsl:if></xsl:if> );
            } else {</xsl:if>
<xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
              if(get_refs()!=null &amp;&amp; get_refs().containsKey(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;)</xsl:if>))
                set<xsl:value-of select="common:capitalizeName(@name)"/>((<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)get_refs().get(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;) </xsl:if>));
            else</xsl:if>
               set<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'"><xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>.ceInternal(new BigDecimal(</xsl:if></xsl:if>sb.toString()<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">),get_refs(), get_transaction())</xsl:if></xsl:if> );
           <xsl:if test="DO_IS_MULTIDB_BASED='true'">
            }
         </xsl:if>   
        } else {
            if (originDatabase != null) {
<xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
              if(get_refs()!=null &amp;&amp; get_refs().containsKey(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;)</xsl:if>)){
                 set<xsl:value-of select="common:capitalizeName(@name)"/>(originDatabase, (<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)get_refs().get(&quot;<xsl:value-of select="@name"/>&quot;) );
               }else {
               if(get_refs()==null)
                refs = new HashMap();</xsl:if>
                 set<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'"><xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>.ceInternal(originDatabase,</xsl:if></xsl:if> null<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'"> )</xsl:if></xsl:if> );
            } else {
<xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
            if(get_refs()!=null &amp;&amp; get_refs().containsKey(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;)</xsl:if>))
                set<xsl:value-of select="common:capitalizeName(@name)"/>((<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>)get_refs().get(get_logicalDBName()+"."+ rs<xsl:if test="/TABLE/DO_IS_OID_BASED='true'">.getBigDecimal( &quot;<xsl:value-of select="@name"/>&quot;) </xsl:if>));
            else {</xsl:if>
               set<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'"><xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/>.ceInternal(</xsl:if></xsl:if> null<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">, get_refs(), get_transaction())</xsl:if></xsl:if> );
            }    
        }
    }</xsl:if>  </xsl:if>
</xsl:template>
</xsl:stylesheet>
