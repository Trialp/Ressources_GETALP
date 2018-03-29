<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!--
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *
 *  
 *  $Id$
 *  
 *  $Log$
 *  Revision 1.2.8.1  2007/10/29 15:11:03  serasset
 *  NEW: lexalp css now defines different forms for HARMONISED/REJECTED entries
 *  NEW: added new db url/user/password configuration keys in papillon.properties file
 *  BUG158: headwords are now harmonised at edition and search time, added a "normalise headword" admin action
 *
 *  Revision 1.2  2005/08/02 14:41:49  mangeot
 *  Work on stylesheets and
 *  added a reset button for Review and AdminContrib forms
 *
 *  Revision 1.1  2005/07/22 08:54:32  mangeot
 *  *** empty log message ***
 *
 *
 *  
-->
<xsl:stylesheet version="1.0"
	xmlns:d='http://www-clips.imag.fr/geta/services/dml' 
    xmlns:fo="http://www.w3.org/1999/XSL/Format"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
	xmlns:xhtml="http://www.w3.org/1999/xhtml"
	xmlns="http://www.w3.org/1999/xhtml">

<xsl:strip-space elements="*" />

<xsl:output method="xml" encoding="utf-8" indent="no"/> 

<!-- general templates please do not modify -->

<!-- here I do not use the xsl:copy because it copies also xmlns: attributes -->
<xsl:template match="d:*" priority="-1"><xsl:apply-templates /></xsl:template>
<xsl:template match="*" priority="-1"><xsl:apply-templates /></xsl:template>

<xsl:template match="comment()"></xsl:template>

<xsl:template match="/"><fo:block><xsl:apply-templates/></fo:block></xsl:template>

<!-- xhtml templates -->
<xsl:template match="xhtml:br"><xsl:text>
</xsl:text></xsl:template>

<xsl:template match="xhtml:b">
<fo:inline font-weight="bold"><xsl:apply-templates /></fo:inline>
</xsl:template>

<xsl:template match="xhtml:i">
<fo:inline font-style="italic"><xsl:apply-templates /></fo:inline>
</xsl:template>

<xsl:template match="xhtml:sup">
<fo:inline  baseline-shift="super" font-size="smaller"><xsl:apply-templates /></fo:inline>
</xsl:template>

<xsl:template match="xhtml:sub">
<fo:inline  baseline-shift="sub" font-size="smaller"><xsl:apply-templates /></fo:inline>
</xsl:template>


<!-- dml specific templates -->
<xsl:template match="d:br"><xsl:text>
</xsl:text></xsl:template>

<xsl:template match="d:comment">
<fo:block color="gray"><xsl:apply-templates /></fo:block></xsl:template>

<xsl:template match="d:space"><xsl:text> </xsl:text></xsl:template>

<xsl:template match="d:est">
<fo:inline  color="#FF9911" font-family="sans-serif"><xsl:apply-templates /></fo:inline>
</xsl:template>

<xsl:template match="d:fra">
<fo:inline  color="#3366FF" font-family="Helvetica"><xsl:apply-templates /></fo:inline>
</xsl:template>


<!-- volume specific templates -->

<xsl:template match="d:contribution">
<fo:block>
Contribution: 
</fo:block><xsl:apply-templates select="*" />
</xsl:template>

<xsl:template match="d:entry"><xsl:apply-templates select="*" /></xsl:template>

<xsl:template match="d:headword"><xsl:apply-templates /></xsl:template>

<xsl:template match="d:pronunciation">
<xsl:if test="text()!=''"> /<xsl:apply-templates />/</xsl:if>
</xsl:template>

<xsl:template match="d:pos">
<xsl:if test="text()!=''">, <xsl:apply-templates /></xsl:if>
</xsl:template>

<xsl:template match="d:particule">
   <xsl:if test="text()!=''">
      <xsl:apply-templates/><xsl:text> </xsl:text>
    </xsl:if>
</xsl:template>

<xsl:template match="d:flexion">
  <xsl:text>[</xsl:text>
	<xsl:apply-templates/>
  <xsl:text>]</xsl:text>
</xsl:template>

<xsl:template match="d:formes-de-base">
	<xsl:apply-templates/>
</xsl:template>

<xsl:template match="d:inf-da">
	<xsl:apply-templates/>
</xsl:template>

<xsl:template match="d:pres-indicatif">
	<xsl:apply-templates/>
</xsl:template>

<xsl:template match="d:sense">
<fo:block>
	<xsl:apply-templates select="*"/>
</fo:block>
</xsl:template>

<xsl:template match="d:cat-gram"><xsl:apply-templates/></xsl:template>

<xsl:template match="registre-sens-vedette">
   <xsl:if test="text()!=''">
		<xsl:apply-templates/>
   </xsl:if>
</xsl:template>

<xsl:template match="d:indication-sem1">
   <xsl:if test="text()!=''">
  		<d:label>
			<xsl:apply-templates/>
  		</d:label>
   </xsl:if>
</xsl:template>

<xsl:template match="d:sous-bloc-semantique">
  <d:sense>
	<xsl:apply-templates select="d:indication-sem2"/>
	<xsl:apply-templates select="d:bloc-contextuel"/>
  </d:sense>
</xsl:template>

<xsl:template match="d:indication-sem2">
	<xsl:if test="text()!=''"><xsl:apply-templates/></xsl:if>
</xsl:template>

<xsl:template match="d:bloc-contextuel">
  <d:sense>
	<xsl:apply-templates/>
  </d:sense>
</xsl:template>

<xsl:template match="d:indication-cont">
  <d:label>
	<xsl:apply-templates/>
  </d:label>
</xsl:template>

<xsl:template match="d:bloc-equivalent">
  <d:sense>
	<xsl:apply-templates/>
  </d:sense>
</xsl:template>

<xsl:template match="d:avant">
  <d:fra>
	<xsl:apply-templates/>
  </d:fra>
</xsl:template>

<xsl:template match="d:mot-princ">
  <d:fra>
  <d:translation>
	<xsl:apply-templates/>
  </d:translation>
  </d:fra>
</xsl:template>

<xsl:template match="d:apres">
	<xsl:apply-templates/>
</xsl:template>

<xsl:template match="d:registre-equiv">
	<xsl:apply-templates select="*"/>
</xsl:template>

<xsl:template match="d:rection-equiv-est">
	<xsl:apply-templates/>
</xsl:template>

<xsl:template match="d:rection-equiv-fra">
	<xsl:apply-templates/>
</xsl:template>

<xsl:template match="d:exemples">
<fo:block>
Exemples : </fo:block>
<xsl:apply-templates select="*"/>
</xsl:template>

<xsl:template match="d:exemple">
<fo:block font-style="italic"><xsl:apply-templates /></fo:block>
</xsl:template>

<xsl:template match="d:exemple-est">
 <d:est>
  <d:example>
	<xsl:apply-templates/>
  </d:example>
  </d:est>
</xsl:template>

<xsl:template match="d:traduction-exe">
 <d:fra>
  <d:example>
	<xsl:apply-templates/>
  </d:example>
  </d:fra>
</xsl:template>

<xsl:template match="d:idioms">
<fo:block>
Phraséologie : </fo:block>
<xsl:apply-templates select="*"/>
</xsl:template>

<xsl:template match="d:unite-phraseol">
	<xsl:apply-templates/>
</xsl:template>

<xsl:template match="d:expression-est">
 <xsl:if test="text()!=''">
 <d:est>
  <d:idiom>
	<xsl:apply-templates/>
  </d:idiom>
  </d:est>
  <br />
  </xsl:if>
</xsl:template>

<xsl:template match="d:expression-fra">
 <xsl:if test="text()!=''">
 <d:fra>
  <d:idiom>
	<xsl:apply-templates/>
  </d:idiom>
  </d:fra>
  <br />
  </xsl:if>
</xsl:template>

<xsl:template match="d:traduction-expr">
 <xsl:if test="text()!=''">
 <d:fra>
  <d:idiom>
	<xsl:apply-templates/>
  </d:idiom>
  </d:fra>
  <br />
  </xsl:if>
</xsl:template>

<xsl:template match="d:bloc-renvois">
  <d:sense><xsl:text>Renvois :</xsl:text><d:space/>
  	<xsl:for-each select="d:renvoi-phraseol">
	<xsl:apply-templates/>
   <xsl:if test="position()!=last()">
      <xsl:text>, </xsl:text>
    </xsl:if>
	</xsl:for-each>
  </d:sense>
</xsl:template>

<xsl:template match="d:renvoi-phraseol">
	<xsl:apply-templates/>
  <xsl:if test="position()!=last()">
      <xsl:text>, </xsl:text>
    </xsl:if>
</xsl:template>



</xsl:stylesheet>
