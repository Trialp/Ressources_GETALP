<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                version="1.0">
<xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="handleArrayDO">
<xsl:if test="IS_ARRAY='true'">
    /* The array data is stored in a separate database table.
     * A special &lt;I&gt;element&lt;/I&gt;DO class has been automatically created to 
     * represent the data in this database table.
     * An &lt;I&gt;element&lt;/I&gt;Query class has been automatically created to 
     * retrieve instances of the &lt;I&gt;element&lt;/I&gt;DO class.
     * Each &lt;I&gt;element&lt;/I&gt;DO returned represents
     * an individual array element and its index.
     * After getting an array of the &lt;I&gt;element&lt;/I&gt;DO objects,
     * we order them by their index, and construct 
     * the array of data elements that we return.
     *
     * @return Array of the <xsl:value-of select="JAVA_TYPE"/> objects.
     *
     */
    <xsl:value-of select="JAVA_TYPE"/>[] get<xsl:value-of select="common:capitalizeName(@name)"/>() {
        <xsl:value-of select="JAVA_TYPE"/>[] ret;
        try {
            <xsl:value-of select="dodst:arrayQueryName(/TABLE/CLASS_NAME,translate(/TABLE/PACKAGE,'/','.'),@name,IS_ARRAY)"/> q = new <xsl:value-of select="dodst:arrayQueryName(/TABLE/CLASS_NAME,translate(/TABLE/PACKAGE,'/','.'),@name,IS_ARRAY)"/> ();
            <xsl:value-of select="dodst:arrayDOName(/TABLE/CLASS_NAME,@name,IS_ARRAY)"/>[] a = q.getDOArray();
            int len = 0;
            for ( int i = 0; i &lt; a.length; i++ )
            if ( len &lt; a[i].getIndex() )
                 len = a[i].getIndex();
            ret = new <xsl:value-of select="JAVA_TYPE"/> [ len ];
            for ( int i = 0; i &lt; a.length; i++ ) {
                ret[ a[i].getIndex() ] =    a[i].getValue();
    
                /* If Attribute.isObjectRef()==true :
                 *    the the MEMBER is an DO
                 *    and the value in the table is an ObjectId,
                 *    so we invoke the createExisting method
                 *    for that DO class (JAVA_TYPE).
                 * Otherwise, we just assign the primitive value.
                 */
            }
        } catch ( Exception e ) { 
            ret = null;
        }
        return ret;
    }
</xsl:if>
</xsl:template>
</xsl:stylesheet>
