<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!--
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *
 *  
 *  $Id$
 *  
 *  $Log$
 *  Revision 1.6  2005/08/02 14:41:49  mangeot
 *  Work on stylesheets and
 *  added a reset button for Review and AdminContrib forms
 *
 *  Revision 1.5  2005/07/16 13:42:08  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.4  2005/06/15 16:48:28  mangeot
 *  Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *  Revision 1.3.6.1  2005/06/10 09:38:50  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.3  2005/01/15 17:43:19  mangeot
 *  Added cvs Id and Log on the files
 *
 *
 *  
-->
<!DOCTYPE xsl:stylesheet [
	<!ENTITY nbsp " ">
]>
<xsl:stylesheet version="1.0"
	xmlns:d='http://www-clips.imag.fr/geta/services/dml' 
	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
	xmlns:xlink="http://www.w3.org/1999/xlink"
	xmlns="http://www.w3.org/1999/xhtml"
	exclude-result-prefixes="d xlink xsl xsi">

<xsl:output method="html" encoding="utf-8" indent="yes"/> 

<!-- Copy through any element that doesn't have a template.-->
<!-- Include attributes and content. -->
<xsl:template match="*|@*" priority="-2">
<xsl:copy><xsl:apply-templates select="@*|*|text()"/></xsl:copy>
</xsl:template>

<xsl:template match="comment()">
  <xsl:comment><xsl:value-of select="."/></xsl:comment>
</xsl:template>

<!-- transform any dml element into a span -->

<xsl:template match="d:*" priority="-1">
<span>
<xsl:attribute name="class"><xsl:value-of select="local-name()" /></xsl:attribute>
  <xsl:apply-templates/>
</span>
</xsl:template>

<!-- formating dml elements -->

<xsl:template match="d:br"><br /></xsl:template>

<xsl:template match="d:comment"><span class="comment"><xsl:apply-templates/></span><br /></xsl:template>

<xsl:template match="d:space">&nbsp;</xsl:template>


<!-- content dml elements -->

<xsl:template match="d:examples">		
<blockquote><span class="examples"><xsl:apply-templates/></span></blockquote>
</xsl:template>

<xsl:template match="d:idioms">		
<blockquote><span class="idioms"><xsl:apply-templates/></span></blockquote>
</xsl:template>

<xsl:template match="d:reflexie">		
<a class="reflexie">
	<xsl:attribute name="href">
		<xsl:value-of select="@xlink:href"/>
	</xsl:attribute>
<xsl:apply-templates/>
</a>
</xsl:template>

<xsl:template match="d:pronunciation">
<xsl:if test="text()">		
<xsl:text>/</xsl:text><span class="pronunciation"><xsl:apply-templates/></span><xsl:text>/</xsl:text>
</xsl:if>
</xsl:template>

<xsl:template match="d:pos">
<xsl:if test="text()"><span class="pos"><xsl:apply-templates/></span></xsl:if>
</xsl:template>

<xsl:template match="d:headword">		
<xsl:if test="text()"><span class="headword"><xsl:apply-templates/></span></xsl:if>
</xsl:template>

<xsl:template match="d:reading">		
<xsl:if test="text()"><span class="reading">[<xsl:apply-templates/>]</span></xsl:if>
</xsl:template>

<xsl:template match="d:sense">		
<blockquote><span class="sense"><xsl:apply-templates/></span></blockquote>
</xsl:template>

<xsl:template match="d:translations">		
<blockquote><xsl:apply-templates/></blockquote>
</xsl:template>

<xsl:template match="d:writing">		
<xsl:if test="text()">
<xsl:text>【</xsl:text><span class="writing"><xsl:apply-templates/></span><xsl:text>】</xsl:text>
</xsl:if>
</xsl:template>


 <xsl:template name="alternated-row">
        <xsl:attribute name="class">
            <xsl:if test="position() mod 2 = 1">oddrow</xsl:if>
            <xsl:if test="position() mod 2 = 0">evenrow</xsl:if>
        </xsl:attribute>
</xsl:template>

</xsl:stylesheet> 
