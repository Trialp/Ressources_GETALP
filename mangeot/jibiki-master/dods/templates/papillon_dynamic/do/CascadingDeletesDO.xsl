<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="cascadingDeletesDO">
<xsl:if test="dodst:compareReferrerDoName(/TABLE/PACKAGE,/TABLE/CLASS_NAME,@do_name) = 'true'">
<xsl:if test="/TABLE/DELETE_CASCADES='false'">            _performCascadingDelete_<xsl:value-of select="dodst:getArrayDO(../@name,@name)"/>(dbt);<!--
                // perform cascading delete on referring table
                if ((Common.getDodsConfProperty(&quot;DeleteCascade&quot;,DODS.getDatabaseManager().getType(get_OriginDatabase())) != null) &amp;&amp;
                    (!((Common.getDodsConfProperty(&quot;DeleteCascade&quot;,DODS.getDatabaseManager().getType(get_OriginDatabase()))).equalsIgnoreCase("true")))) {
                        <xsl:value-of select="../@package"/>.<xsl:value-of select="../@name"/>DO[] a = <xsl:value-of select="dodst:getArrayDO(../@name,@name)"/>();

                        for ( int i = 0; i &lt; a.length; i++ ) {
                             if(!_tr_(dbt).getDeletedDOs().contains(get_logicalDBName()+"."+a[ i ].get_Handle()) &amp;&amp; !a[i].isDeleted()) {
                                a[ i ].delete( dbt, false);
                             }   
                        }
                } else {
                    <xsl:value-of select="../@package"/>.<xsl:value-of select="../@name"/>DO[] a = <xsl:value-of select="dodst:getArrayDO(../@name,@name)"/>();
                    for ( int i = 0; i &lt; a.length; i++ ) 
                         a[ i ].setDeleted(true);
                }    -->
</xsl:if></xsl:if>
</xsl:template>
</xsl:stylesheet>
