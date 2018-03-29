<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
        <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="preCompareCond">
    /**
     * Compares whether this DO satisfies condition cond.
     *
     * @param cond condition.
     * 
     * @return <code>true</code> if DO satisfies condition cond, otherwise <code>false</code>.
     */
    public boolean compareCond(Condition cond) {
        try {
            switch(cond.getColumnIndex()) {
</xsl:template>
</xsl:stylesheet>
