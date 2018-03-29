<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:gvs="org.ejen.ext.GlobalVariables" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0" extension-element-prefixes="gvs" exclude-result-prefixes="gvs">

<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DataStruct/writeFileHeaderDataStruct.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DataStruct/writeMemberStuffDataStruct.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DataStruct/writeBeginCompareCondDS.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DataStruct/writeCompareCondDS.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DataStruct/writeEndCompareCondDS.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DataStruct/writeFileFooterDataStructBegin.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DataStruct/cloningDataStruct.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DataStruct/writeFileFooterDataStructEnd.xsl"/> 
                    
<xsl:output omit-xml-declaration="yes"/>

<xsl:template match="/">
    <xsl:apply-templates/>
</xsl:template>

<xsl:template match="TABLE">
    <xsl:call-template name="fileHeader"/>
    <xsl:for-each select="//COLUMN">
        <xsl:call-template name="memberStuff"/>
    </xsl:for-each>
    <xsl:call-template name="beginCompareCond"/>
    <xsl:for-each select="//COLUMN">
  	   <xsl:call-template name="compareCond"/>
    </xsl:for-each>
    <xsl:call-template name="endCompareCond"/>
    <xsl:call-template name="fileFooterBegin"/>
    <xsl:for-each select="//COLUMN">
        <xsl:call-template name="cloning"/>
    </xsl:for-each>
    <xsl:call-template name="fileFooterEnd"/>
</xsl:template>
</xsl:stylesheet>
