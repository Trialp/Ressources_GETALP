<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="commitDeleteOtherDO">
<xsl:if test="REFERENCE_OBJECT">
<xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">
<xsl:text>                    </xsl:text><xsl:value-of select="REFERENCE_OBJECT/PACKAGE"/>.<xsl:value-of select="REFERENCE_OBJECT/@name"/>DO <xsl:value-of select="@name"/>_DO = get<xsl:value-of select="common:capitalizeName(@name)"/>();
                    if ( null != <xsl:value-of select="@name"/>_DO &amp;&amp; !this.equals(<xsl:value-of select="@name"/>_DO)) {
                        if ( <xsl:value-of select="@name"/>_DO.isDirty() ) {
                            boolean tmpDirty = true;
                            if(isDirty()) {
                                markClean();
                                tmpDirty = false;
                            }    
                            okToCommit<xsl:value-of select="common:capitalizeName(@name)"/>( <xsl:value-of select="@name"/>_DO );
                            if ( !<xsl:value-of select="@name"/>_DO.getConfigurationAdministration().getTableConfiguration().isReadOnly() ) {
                                <xsl:value-of select="@name"/>_DO.save( dbt );
                                if(!tmpDirty)
                                   markNewValue(); 
                            }    
                        } else {
                            // since the referenced DO is not loaded,
                            // it cannot be dirty, so there is no need to commit it.
                        }
                    } else {
                        if ( ! <xsl:value-of select="CAN_BE_NULL"/> )
                            throw new RefAssertionException( &quot;Cannot commit <xsl:value-of select="/TABLE/CLASS_NAME"/>DO ( &quot; + toString() +
                                                             &quot; ) because <xsl:value-of select="@name"/> is not allowed to be null.&quot; );
                    }
</xsl:if>
</xsl:if>
</xsl:template>
</xsl:stylesheet>
