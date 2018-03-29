<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:common="org.enhydra.dods.Common"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="memberStuff">
<xsl:if test="JAVADOC">
<xsl:text>
</xsl:text>
</xsl:if>
<xsl:text>    </xsl:text><xsl:value-of select="JAVADOC"/>
    private <xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="JAVA_TYPE"/></xsl:if><xsl:if test="REFERENCE_OBJECT">BigDecimal</xsl:if><xsl:text> </xsl:text> <xsl:value-of select="@name"/> = <xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="miu:fixDefault(JAVA_TYPE,JAVA_DEFAULT_VALUE)"/></xsl:if><xsl:if test="REFERENCE_OBJECT"> null </xsl:if>;

    /**
     * Used for query caching.
     */
    static public final int COLUMN_<xsl:value-of select="common:upperCaseName(@name)"/> = <xsl:value-of select="dodst:getCounter()"/>;
    <xsl:value-of select="dodst:incCounter()"/>

    /**
     * Sets <xsl:value-of select="common:capitalizeName(@name)"/> column.
     * @param _<xsl:value-of select="@name"/> new column value.
     */
    public void set<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="JAVA_TYPE"/></xsl:if><xsl:if test="REFERENCE_OBJECT">BigDecimal</xsl:if> _<xsl:value-of select="@name"/>) {
        if (readOnly)
            throw new Error("This should never happen! set<xsl:value-of select="common:capitalizeName(@name)"/> on "
			    + this +" is being called although readOnly is true");
//        boolean bDiff = GenericDO.isNewDataDifferent_<xsl:if test="REFERENCE_OBJECT"><xsl:if test="REFERENCE_OBJECT/IS_FOREIGN_KEY='false'">BigDecimal</xsl:if></xsl:if><xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="miu:javaType(JAVA_TYPE)"/></xsl:if><xsl:if test="REFERENCE_OBJECT"><xsl:if test="not(REFERENCE_OBJECT/IS_FOREIGN_KEY='false')"><xsl:value-of select="miu:javaType(JAVA_TYPE)"/></xsl:if></xsl:if>(<xsl:value-of select="@name"/>, _<xsl:value-of select="@name"/>);
        <xsl:value-of select="@name"/> = _<xsl:value-of select="@name"/>;
//        return bDiff;
    }

    /**
     * Return value of <xsl:value-of select="common:capitalizeName(@name)"/> column.
     * @return value of <xsl:value-of select="common:capitalizeName(@name)"/> column.
     */
    public <xsl:if test="not(REFERENCE_OBJECT)"><xsl:value-of select="JAVA_TYPE"/></xsl:if><xsl:if test="REFERENCE_OBJECT">BigDecimal</xsl:if> get<xsl:value-of select="common:capitalizeName(@name)"/>() {
        return <xsl:value-of select="@name"/>;
    }
</xsl:template>
</xsl:stylesheet>
