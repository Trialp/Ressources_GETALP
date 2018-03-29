<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
        <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="beginCompareCond">
    /**
     * Used for query caching.
     */
    static public final int COLUMN_OID = <xsl:value-of select="dodst:getCounter()"/>;

    /**
     * Compares whether this DataStruct object satisfies condition cond.
     *
     * @param cond Condition of the query.
     *
     * @return true if this DataStruct object satisfies condition of this query,
     * otherwise false.
     */
    public boolean compareCond(Condition cond) {
        try {
            switch (cond.getColumnIndex()) {
</xsl:template>
</xsl:stylesheet>
