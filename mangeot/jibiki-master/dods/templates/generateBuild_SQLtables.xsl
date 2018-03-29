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
				xmlns:miu="org.enhydra.dods.Common"
            extension-element-prefixes="msg gvs mta xmi miu"
            exclude-result-prefixes="xalan xalan-ext gvs msg mta stu ver xmi">

<xsl:output method="text" encoding="UTF-8"/>

<!-- edited by Dragan Radeka 24.09.2002. -->



<xsl:template match ="/">
  <xsl:apply-templates />
</xsl:template>



<!-- DATABASE -->

<xsl:template match="DATABASE">
	<gvs:put name="targetName" select="''"/>

&lt;project name="generateSQL" default="generateSql"&gt;
&lt;property name="doml.file" value="<xsl:value-of select="miu:getDomlFile()"/>"/&gt;
&lt;property name="project.root" value="<xsl:value-of select="miu:getProjectRoot()"/>"/&gt;
&lt;property name="template.dir" value="<xsl:value-of select="miu:getTemplateDir()"/>"/&gt;
&lt;property name="template.set" value="<xsl:value-of select="miu:getTemplateSet()"/>"/&gt;

	&lt;taskdef name="ejen" classname="org.ejen.EjenTask"/&gt;

	<xsl:for-each select="TABLE">
		&lt;target name="<xsl:value-of select="@id"/>"&gt;
		<xsl:if test="position()!=last()"><gvs:put name="targetName" select="concat(gvs:get('targetName'), @id, ', ')"/></xsl:if>
		<xsl:if test="position()=last()"><gvs:put name="targetName" select="concat(gvs:get('targetName'), @id)"/></xsl:if>
			&lt;ejen&gt;
				&lt;source file="${project.root}/<xsl:value-of select="@path"/>/<xsl:value-of select="@class"/>.xml"/&gt;
				&lt;template file="${template.dir}/${template.set}/sql/SQLcreateTables.xsl" outdated="true"
						  filepattern="${project.root}/<xsl:value-of select="@path"/>/<xsl:value-of select="@class"/>.sql"/&gt;
				
				&lt;template file="${template.dir}/${template.set}/sql/SQLdropTables.xsl"/&gt;
				
			&lt;/ejen&gt;
		&lt;/target&gt;
		</xsl:for-each>
		
		&lt;target name="generateSql" depends="<xsl:value-of select="gvs:get('targetName')"/>"&gt;
		&lt;/target&gt;

&lt;/project&gt;

</xsl:template>

</xsl:stylesheet>
