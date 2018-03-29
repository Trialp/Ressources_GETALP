<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:gvs="org.ejen.ext.GlobalVariables" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0" extension-element-prefixes="gvs" exclude-result-prefixes="gvs">

<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/Query/writeFileHeaderQuery.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/Query/writeMemberStuffQuery.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/Query/FK_setQuery_methodsHeaderQuery.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/Query/FK_setQuery_callsQuery.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/Query/FK_setQuery_methodsFooterQuery.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/Query/writeFileFooterQuery.xsl"/>
                    
<xsl:output omit-xml-declaration="yes"/>

<xsl:template match="/">
    <xsl:apply-templates/>
</xsl:template>


<xsl:template match="TABLE">
    <xsl:call-template name="fileHeader"/>
    <xsl:for-each select="//COLUMN">
        <xsl:call-template name="memberStuff"/>
    </xsl:for-each>
<!-- // for future release
    <xsl:for-each select="//FK_REF">
        <xsl:for-each select="FK_GROUP">
            <xsl:call-template name="setQueryMethodsHeader"/>
                <xsl:for-each select="PAIR">
                    <xsl:call-template name="setQueryCalls"/>
                </xsl:for-each>
            <xsl:call-template name="setQueryMethodsFooter"/>
        </xsl:for-each>
    </xsl:for-each>
-->
    <xsl:call-template name="fileFooter"/>
</xsl:template>
</xsl:stylesheet>
