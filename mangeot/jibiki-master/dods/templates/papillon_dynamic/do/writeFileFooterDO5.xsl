<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="fileFooter5">            updateStmt.append(&quot; where <xsl:value-of select="dodst:updateWhereClause()"/>&quot;);

            stmt = conn.prepareStatement(updateStmt.toString());
<xsl:if test="DO_IS_OID_BASED='true'">            setPrepStmtParam_int(stmt, param, get_Version());

</xsl:if>
</xsl:template>
</xsl:stylesheet>
