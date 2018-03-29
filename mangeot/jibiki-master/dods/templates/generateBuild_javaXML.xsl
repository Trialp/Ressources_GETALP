<?xml version="1.0" encoding="iso-8859-1"?>

<xsl:stylesheet version="1.0"
				xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:xalan-ext="http://xml.apache.org/xalan"
				xmlns:gvs="org.ejen.ext.GlobalVariables"
				xmlns:msg="org.ejen.ext.Messenger"
				xmlns:stu="org.ejen.ext.StringUtil"
				xmlns:xmi="org.ejen.ext.XMLInclude"
				xmlns:ver="org.ejen.ext.Version"
				xmlns:mta="org.ejen.ext.db.BasicMetaDataConnection"
				xmlns:common="org.enhydra.dods.Common"
                extension-element-prefixes="msg gvs mta xmi common"
                exclude-result-prefixes="xalan xalan-ext gvs msg mta stu ver xmi">

<xsl:output method="text" encoding="UTF-8"/>
<!-- edited by Dragan Radeka 24.09.2002. -->
	
<xsl:template match ="/">
  <xsl:apply-templates />
</xsl:template>



<!-- DATABASE -->

<xsl:template match="DATABASE">
	<gvs:put name="targetName" select="''"/>

&lt;project name="generateJava" default="generate_all"&gt;
&lt;property name="doml.file" value="<xsl:value-of select="common:getDomlFile()"/>"/&gt;
&lt;property name="project.root" value="<xsl:value-of select="common:getProjectRoot()"/>"/&gt;
&lt;property name="template.dir" value="<xsl:value-of select="common:getTemplateDir()"/>"/&gt;
&lt;property name="template.set" value="<xsl:value-of select="common:getTemplateSet()"/>"/&gt;

	&lt;taskdef name="ejen" classname="org.ejen.EjenTask"/&gt;

	<xsl:for-each select="TABLE">
		&lt;target name="<xsl:value-of select="@id"/>"&gt;
		<xsl:if test="position()!=last()"><gvs:put name="targetName" select="concat(gvs:get('targetName'), @id, ', ')"/></xsl:if>
		<xsl:if test="position()=last()"><gvs:put name="targetName" select="concat(gvs:get('targetName'), @id)"/></xsl:if>
			&lt;ejen&gt;
				&lt;source file="<xsl:value-of select="common:getProjectRoot()"/>/<xsl:value-of select="@path"/>/<xsl:value-of select="@class"/>.xml"/&gt;
				&lt;template file="${template.dir}/${template.set}/Query/mainQUERY.xsl" outdated="true"
						  filepattern="<xsl:value-of select="common:getProjectRoot()"/>/<xsl:value-of select="@path"/>/<xsl:value-of select="@class"/>Query.java"&gt;
				&lt;/template&gt;
				
				&lt;template file="${template.dir}/${template.set}/DOI/mainDOI.xsl" outdated="true"
						  filepattern="<xsl:value-of select="common:getProjectRoot()"/>/<xsl:value-of select="@path"/>/<xsl:value-of select="@class"/>DOI.java"&gt;
				&lt;/template&gt;
				
				&lt;template file="${template.dir}/${template.set}/DataStruct/mainDataStruct.xsl" outdated="true"
						  filepattern="<xsl:value-of select="common:getProjectRoot()"/>/<xsl:value-of select="@path"/>/<xsl:value-of select="@class"/>DataStruct.java"&gt;
				&lt;/template&gt;
				
				&lt;template file="${template.dir}/${template.set}/do/mainDO.xsl" outdated="true"
						  filepattern="<xsl:value-of select="common:getProjectRoot()"/>/<xsl:value-of select="@path"/>/<xsl:value-of select="@class"/>DO.java"&gt;
		  		&lt;/template&gt;
				<xsl:if test="@massUpdates='true'">
				&lt;template file="${template.dir}/${template.set}/Update/mainUpdate.xsl" outdated="true"
						  filepattern="<xsl:value-of select="common:getProjectRoot()"/>/<xsl:value-of select="@path"/>/<xsl:value-of select="@class"/>Update.java"&gt;
		  		&lt;/template&gt;
		  		</xsl:if><xsl:if test="@massDeletes='true'">
				&lt;template file="${template.dir}/${template.set}/Delete/mainDelete.xsl" outdated="true"
						  filepattern="<xsl:value-of select="common:getProjectRoot()"/>/<xsl:value-of select="@path"/>/<xsl:value-of select="@class"/>Delete.java"&gt;
		  		&lt;/template&gt;
		  		</xsl:if>
			&lt;/ejen&gt;
		&lt;/target&gt;
		</xsl:for-each>
		
		&lt;target name="generate_all" depends="<xsl:value-of select="gvs:get('targetName')"/>"&gt;
		&lt;/target&gt;

&lt;/project&gt;

</xsl:template>

</xsl:stylesheet>
