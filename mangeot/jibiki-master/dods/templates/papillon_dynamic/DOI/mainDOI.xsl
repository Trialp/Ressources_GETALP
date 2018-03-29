<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:gvs="org.ejen.ext.GlobalVariables" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0" extension-element-prefixes="gvs" exclude-result-prefixes="gvs">
	<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DOI/writeFileHeaderDOI.xsl"/>
	<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DOI/writeMemberStuffDOI.xsl"/>
	<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/DOI/writeFileFooterDOI.xsl"/> 
	                    
	<xsl:output omit-xml-declaration="yes"/>

	<xsl:template match="/">
		    <xsl:apply-templates/>
	</xsl:template>

	<xsl:template match="TABLE">
        <xsl:call-template name="begining"/>
	<xsl:for-each select="//COLUMN">
            <xsl:call-template name="middle"/>
	 </xsl:for-each>
	 <xsl:call-template name="end"/>
	 </xsl:template>

</xsl:stylesheet>
