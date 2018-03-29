<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:gvs="org.ejen.ext.GlobalVariables" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0" extension-element-prefixes="gvs" exclude-result-prefixes="gvs">
					
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileHeaderDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeMemberStuffDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writePreCompareCondDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeCompareCondDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO0.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/FK_get_set_methodsDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO1.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/ResultSetExtractionForLONG.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/ResultSetExtractionDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO2.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/HandleArrayDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO3.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/JDBCsetCallDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO4.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/JDBCsetCallForNonOidPKDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO5.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/JDBCsetCallForNonOidPKDO1.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO6.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/toStringDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO7.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/manyToOneReferrersDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/addReferrersDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/referrersDOInit.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/manyToManyReferrersDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO8.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/StubForCommitAssertionDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO9.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/CascadingDeletesDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO10.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/commitDeleteOtherDO.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO11.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/setChangedFlags.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/performCascadingDeletes.xsl"/>
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeFileFooterDO12.xsl"/>	
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeColumnsNameString.xsl"/>			
<xsl:import href="/Projects/jibiki/branches/LINKS_1_0/dods/templates/papillon_dynamic/do/writeColumnsNameStringEnd.xsl"/>		
					
<xsl:output omit-xml-declaration="yes"/>


<xsl:template match="/">
  <xsl:apply-templates/>
</xsl:template>




<xsl:template match="TABLE">
  <xsl:call-template name="fileHeader"/>

  <xsl:for-each select="//COLUMN">
  	<xsl:call-template name="memberStuff"/>
  </xsl:for-each>

  <xsl:call-template name="preCompareCond"/>

  <xsl:for-each select="//COLUMN">
  	<xsl:call-template name="compareCond"/>
  </xsl:for-each>

  <xsl:call-template name="fileFooter0"/>
<!--
  <xsl:for-each select="//COLUMN">
	  <xsl:call-template name="fK_methods"/>
  </xsl:for-each>
-->
  <xsl:call-template name="fileFooter1"/>

  <xsl:for-each select="//COLUMN">
        <xsl:if test="DB_TYPE='LONGVARCHAR'">
                <xsl:call-template name="resultSetExstractionForLONGDO"/>
        </xsl:if>
        <xsl:if test="not(DB_TYPE='LONGVARCHAR')">
                <xsl:call-template name="resultSetExstractionDO"/>
        </xsl:if>
  </xsl:for-each>

  <xsl:call-template name="fileFooter2"/>
  
  <xsl:for-each select="//COLUMN">
                <xsl:call-template name="columnsNameString"/>
  </xsl:for-each>
 
  <xsl:call-template name="columnsNameStringEnd"/>  

  <xsl:for-each select="//COLUMN">
	  <xsl:call-template name="handleArrayDO"/>
  </xsl:for-each>

  <xsl:call-template name="fileFooter3"/>

  <xsl:for-each select="//COLUMN">
	  <xsl:call-template name="jdbcSetCallDO"/>
  </xsl:for-each>

  <xsl:call-template name="fileFooter4"/>

  <xsl:for-each select="//COLUMN">
	  <xsl:call-template name="jdbcSetCallForNonOidPKDO"/>
  </xsl:for-each>

  <xsl:call-template name="fileFooter5"/>

  <xsl:for-each select="//COLUMN">
	  <xsl:call-template name="jdbcSetCallForNonOidPKDO1"/>
  </xsl:for-each>

  <xsl:call-template name="fileFooter6"/>

  <xsl:for-each select="//COLUMN">
	  <xsl:call-template name="toStringDO"/>
  </xsl:for-each>

  <xsl:call-template name="fileFooter7"/>

  <xsl:for-each select="//REFERRER">
 		<xsl:call-template name="referrersDOInit"/>
	  <xsl:for-each select="REFATTR">
  		<xsl:call-template name="addReferrersDO"/>
	  </xsl:for-each>
	  <xsl:for-each select="REFATTR">
		  <xsl:call-template name="manyToOneReferrersDO"/>
	  </xsl:for-each>
  </xsl:for-each>

  <xsl:for-each select="//REFERRER">
 		<xsl:call-template name="referrersDOInit"/>
	  <xsl:for-each select="REFATTR">
  		<xsl:call-template name="addReferrersDO"/>
	  </xsl:for-each>
	  <xsl:for-each select="REFATTR">
		  <xsl:call-template name="manyToManyReferrersDO"/>
	  </xsl:for-each>
  </xsl:for-each>

  <xsl:call-template name="fileFooter8"/>

  <xsl:for-each select="//COLUMN">
	  <xsl:call-template name="stubForCommitAssertionDO"/>
  </xsl:for-each>

  <xsl:call-template name="fileFooter9"/>

  <xsl:for-each select="//REFERRER">
 		<xsl:call-template name="referrersDOInit"/>
	  <xsl:for-each select="REFATTR">
  		<xsl:call-template name="addReferrersDO"/>
	  </xsl:for-each>
	  <xsl:for-each select="REFATTR">
		  <xsl:call-template name="cascadingDeletesDO"/>
	  </xsl:for-each>
  </xsl:for-each>

  <xsl:call-template name="fileFooter10"/>

  <xsl:for-each select="//COLUMN">
	  <xsl:call-template name="commitDeleteOtherDO"/>
  </xsl:for-each>

  <xsl:call-template name="fileFooter11"/>

  <xsl:for-each select="//COLUMN">
	  <xsl:call-template name="setChangedFlags"/>
  </xsl:for-each>

  <xsl:for-each select="//REFERRER">
      <xsl:call-template name="referrersDOInit"/>
      <xsl:for-each select="REFATTR">
	  <xsl:call-template name="addReferrersDO"/>
      </xsl:for-each>
      <xsl:for-each select="REFATTR">
	  <xsl:call-template name="performCascadingDeletes"/>
      </xsl:for-each>
  </xsl:for-each>

  <xsl:call-template name="fileFooter12"/>

</xsl:template>

</xsl:stylesheet>
