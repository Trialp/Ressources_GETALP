<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!--
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *
 *  
 *  $Id: TEXT-view.xsl 712 2007-10-29 15:11:03Z serasset $
 *  
 *  $Log$
 *  Revision 1.4.8.1  2007/10/29 15:11:03  serasset
 *  NEW: lexalp css now defines different forms for HARMONISED/REJECTED entries
 *  NEW: added new db url/user/password configuration keys in papillon.properties file
 *  BUG158: headwords are now harmonised at edition and search time, added a "normalise headword" admin action
 *
 *  Revision 1.4  2005/08/02 14:41:49  mangeot
 *  Work on stylesheets and
 *  added a reset button for Review and AdminContrib forms
 *
 *  Revision 1.3  2005/07/22 08:54:32  mangeot
 *  *** empty log message ***
 *
 *
 *  
-->
<xsl:stylesheet version="1.0"
	xmlns:d='http://www-clips.imag.fr/geta/services/dml' 
	xmlns:p='http://atoum.imag.fr/geta/User/viacheslav.dikonov/corrections/pivax_local.xsd' 
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
	xmlns="http://www.w3.org/1999/xhtml">

<xsl:strip-space elements="*" />

<xsl:output method="text" encoding="utf-8" indent="no"/> 

<!-- general templates please do not modify -->

<!-- here I do not use the xsl:copy because it copies also xmlns: attributes -->
<xsl:template match="d:*" priority="-1"><xsl:apply-templates /></xsl:template>
<xsl:template match="*" priority="-1"><xsl:apply-templates /></xsl:template>

<xsl:template match="comment()"></xsl:template>


<xsl:template match="/"><xsl:apply-templates/></xsl:template>

<!-- xhtml templates -->
<xsl:template match="br"><xsl:text>
</xsl:text></xsl:template>


<!-- dml specific templates -->
<xsl:template match="d:br"><xsl:text>
</xsl:text></xsl:template>

<xsl:template match="d:comment"><xsl:apply-templates /><xsl:text>
</xsl:text></xsl:template>

<xsl:template match="d:space"><xsl:text> </xsl:text></xsl:template>


<!-- volume specific templates -->

<xsl:template match="d:contribution"><xsl:text>
Contribution: 
</xsl:text><xsl:apply-templates select="*" /><xsl:text>
</xsl:text></xsl:template>

<xsl:template match="d:author">Author: <xsl:apply-templates />, </xsl:template>
<xsl:template match="d:creation-date">creation-date: <xsl:apply-templates />, </xsl:template>
<xsl:template match="d:status">status: <xsl:apply-templates />.</xsl:template>


<xsl:template match="d:data"><xsl:apply-templates /></xsl:template>


<xsl:template match="p:lexie"><xsl:apply-templates select="*" /></xsl:template>

<xsl:template match="p:lemma">Lemma:<xsl:apply-templates /></xsl:template>

<xsl:template match="p:pos">
<xsl:if test="text()!=''"> /<xsl:apply-templates />/ </xsl:if>
</xsl:template>

<xsl:template match="p:gender">
   <xsl:if test="text()!=''"><xsl:apply-templates/></xsl:if>
</xsl:template>

<xsl:template match="p:processors">
<xsl:text>
Processors :</xsl:text>
<xsl:apply-templates/>
</xsl:template>

<xsl:template match="p:processor">
  Processor: <xsl:apply-templates select="@p:name"/>, data: <xsl:apply-templates select="@p:data"/>, datatype: <xsl:apply-templates select="@p:datatype"/>, access: <xsl:apply-templates select="@p:access"/>
</xsl:template>



</xsl:stylesheet>
