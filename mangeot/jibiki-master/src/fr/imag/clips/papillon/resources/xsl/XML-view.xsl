<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!--
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *
 *  $Id$
 *  $Log$
 *  Revision 1.12  2006/07/10 10:58:20  serasset
 *  *** empty log message ***
 *
 *  Revision 1.10  2006/03/29 12:44:55  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.9  2006/03/29 12:43:29  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.8  2006/03/29 12:42:00  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.7  2006/03/29 12:40:05  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.6  2006/03/05 17:15:11  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.5  2006/03/05 16:55:31  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.4  2005/10/14 19:02:17  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.3  2005/05/24 12:51:22  serasset
 *  Updated many aspect of the Papillon project to handle lexalp project.
 *  1. Layout is now parametrable in the application configuration file.
 *  2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 *  3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 *  4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 *  5. It is now possible to give a name to the cookie key in the app conf file
 *  6. Several bug fixes.
 *
 *  Revision 1.2  2005/01/15 17:43:19  mangeot
 *  Added cvs Id and Log on the files
 *
-->
<!DOCTYPE xsl:stylesheet [
	<!ENTITY nbsp "  ">
]>
<xsl:stylesheet  version="1.0"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:output method="html" 
	encoding="UTF-8"
	standalone="yes"
	indent="no"
	/>

<!--

  Feuille de style pour transformer les documents
xml en html avec des couleurs

For HTML spaces, I did not find a good solution.
I can't use the feature disable-output-escaping="yes"
so I cheated and added a dummy css style all white for the <span class="space"> class

I would like to find a solution in genreal for HTML entities...

-->

<!-- root element -->
<!--xsl:template match="/">	
<div><xsl:apply-templates/></div>
</xsl:template-->


<!-- root element -->
<xsl:template match="/">	
<xsl:variable name="actualnamespaces"><xsl:for-each select="namespace::*"><xsl:value-of select="name()"/>,</xsl:for-each></xsl:variable>

<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="fr">
<head>
	<meta http-equiv="content-type" content="text/html; charset=utf-8" />
</head>
<body>
	<xsl:apply-templates select="*"/>
</body>
</html>

</xsl:template>



<!-- comments -->
<xsl:template match="comment()">
<br /><br />
<span class="xmlcomment">&lt;!-- <xsl:value-of select="normalize-space()" /> --&gt;</span>
</xsl:template>


<!-- elements -->
<xsl:template name="generic" match="*">
<xsl:param name="namespaces"></xsl:param>

<xsl:variable name="actualnamespaces"><xsl:for-each select="namespace::*"><xsl:value-of select="name()"/>,</xsl:for-each></xsl:variable>

<xsl:variable name="newnamespaces">
<xsl:if test="$namespaces=''"><xsl:value-of select="$actualnamespaces"/></xsl:if>
<xsl:if test="$namespaces!=''"><xsl:value-of select="substring($actualnamespaces,string-length($namespaces))"/></xsl:if>
</xsl:variable>

<br /><xsl:for-each select="ancestor::*">&nbsp;</xsl:for-each>
<span class="xmlcar">&lt;</span>

<xsl:if test="substring-before(name(),concat(':',local-name()))!=''">
<span class="xmlnsprefix"><xsl:value-of select="substring-before(name(),concat(':',local-name()))"/>
</span>	
<xsl:if test="substring-before(name(),concat(':',local-name()))!=''">
<span class="xmlcar">:</span>
</xsl:if>
</xsl:if>
<span class="xmlelement"><xsl:value-of select="local-name()"/></span>

<xsl:if test="count(ancestor::*)=0"><xsl:call-template name="print-default-namespace"/></xsl:if>
	<xsl:call-template name="print-namespaces">
    	<xsl:with-param name="namespaces" select="$newnamespaces"/>
    	<xsl:with-param name="ancestors" select="ancestor::*"/>
	</xsl:call-template>


<xsl:apply-templates select="@*"/>
<xsl:if test="*|text()">
<span class="xmlcar">&gt;</span>
</xsl:if>
<xsl:if test="not(*|text())">
<span class="xmlcar">/&gt;</span>
</xsl:if>

<xsl:apply-templates select="*|text()|comment()" >
      <xsl:with-param name="namespaces" select="$actualnamespaces"/>
</xsl:apply-templates>


<xsl:if test="*|text()">
<xsl:if test="*">
<br/>
<xsl:for-each select="ancestor::*">&nbsp;</xsl:for-each>
</xsl:if>
<span class="xmlcar">&lt;/</span>
<xsl:if test="substring-before(name(),concat(':',local-name()))!=''">
<span class="xmlnsprefix"><xsl:value-of select="substring-before(name(),concat(':',local-name()))"/>
</span>	
<xsl:if test="substring-before(name(),concat(':',local-name()))!=''">
<span class="xmlcar">:</span>
</xsl:if>
</xsl:if>
<span class="xmlelement"><xsl:value-of select="local-name()"/></span>
<span class="xmlcar">&gt;</span>
</xsl:if>
</xsl:template>


<!-- attributes -->
<xsl:template match="@*" >
&nbsp;
<!-- xmlnsprefix -->
<span class="xmlnsprefix">
	<xsl:value-of select="substring-before(name(),concat(':',local-name()))"/>
</span>
<!--les : eventuels -->			
<xsl:if test="substring-before(name(),concat(':',local-name()))!=''">
	<span class="xmlcar">:</span>
</xsl:if>
<span class="xmlattribute">
	<xsl:value-of select="local-name(current())"/>
	<span class="xmlcar">="</span>
	<span class="xmlvalue">
		<xsl:value-of select="current()"/>
	</span>
	<span class="xmlcar">"</span>
</span>
</xsl:template>

<!-- text -->
<xsl:template match="text()" >
	<xsl:if test="normalize-space()!=''">
		<span class="xmltext"><xsl:value-of  select="normalize-space()" /></span>
	</xsl:if>
	<xsl:apply-templates />
</xsl:template>

<!-- namespaces for the root element -->
<xsl:template name="print-namespaces" >
<xsl:param name="namespaces">xml,</xsl:param>
<xsl:param name="ancestors"></xsl:param>
	<xsl:for-each select="namespace::*">
		<xsl:variable name="namespace"><xsl:value-of select="name()"/>,</xsl:variable>
	   <xsl:if test="contains($namespaces,$namespace)">
			<xsl:if test="name()!=''">
	   			<br /><xsl:for-each select="ancestor::*">&nbsp;</xsl:for-each>
				<span class="xmlnsprefix">xmlns</span>
				<span class="xmlcar">:</span>
				<span class="xmlnsprefix">
					<xsl:value-of select="name()"/>						
				</span>
				<span class="xmlcar">="</span>					
				<span class="xmlnsuri">
					<xsl:value-of select="current()"/>
				</span>
				<span class="xmlcar">"</span>					
	        </xsl:if>
	   </xsl:if>
	</xsl:for-each>
</xsl:template>

<!-- namespaces for the root element -->
<xsl:template name="print-default-namespace">
	<br />&nbsp;
	<xsl:for-each select="namespace::*">
			<xsl:if test="name()=''">
				<span class="xmlnsprefix">xmlns</span>
				<span class="xmlcar">="</span>
				<span class="xmlnsuri">
					<xsl:value-of select="current()"/>
				</span>
				<span class="xmlcar">"</span>
			</xsl:if>			
	</xsl:for-each>
</xsl:template>


</xsl:stylesheet>
