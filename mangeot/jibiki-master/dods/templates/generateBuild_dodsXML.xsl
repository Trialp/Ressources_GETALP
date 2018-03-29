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
  <xsl:call-template name="build_dods"/>
</xsl:template>



<!-- DATABASE -->

<xsl:template name="build_dods">&lt;?xml version="1.0" encoding="UTF-8"?&gt;

&lt;project name="generateDods" default="compileJava"&gt;
<xsl:value-of select="common:getInstallPropertiesParam()"/>
&lt;property name="project.name" value="<xsl:value-of select="common:getFileName()"/>"/&gt;
&lt;property name="doml.filename" value="<xsl:value-of select="common:getDomlFileName()"/>"/&gt;
&lt;property name="project.root" value="<xsl:value-of select="common:getProjectRoot()"/>"/&gt;
&lt;property name="database.vendor" value="<xsl:value-of select="common:getDatabaseVendor()"/>"/&gt;
&lt;property name="database.splitsqlprimary" value="<xsl:value-of select="common:getSplitSQLPrimary()"/>"/&gt;
&lt;property name="temp.dir" value="ejen_temp"/&gt;
&lt;property name="template.dir" value="<xsl:value-of select="common:getTemplateDir()"/>"/&gt;
&lt;property name="template.set" value="<xsl:value-of select="common:getTemplateSet()"/>"/&gt;

    
	&lt;!-- ==================================================================================== --&gt;

	&lt;taskdef name="ejen" classname="org.ejen.EjenTask"/&gt;
	&lt;taskdef name="DODS_XMLBuilder" classname="org.enhydra.ant.taskdefs.DODS_XMLBuilder"/&gt;
	&lt;taskdef name="sqlsplitter" classname="org.enhydra.ant.taskdefs.SQLSplitterTask"/&gt;

	&lt;!-- ==================================================================================== --&gt;


	
	&lt;target name="validate" &gt;
		&lt;xmlvalidate file="${project.root}/${doml.filename}" warn="yes" 
	         classname="org.apache.xerces.parsers.SAXParser" 
	         classpath="${enhydra.root}/lib/xercesImpl.jar"&gt;
   	&lt;/xmlvalidate&gt;
	&lt;/target&gt;

	&lt;target name="dods" depends="validate"&gt;
		&lt;DODS_XMLBuilder doml="${project.root}/${doml.filename}"
	   			outdir="${project.root}"
	   			database="${database.vendor}"
	   			/&gt;
	&lt;/target&gt;
	
	
	&lt;target name="prepareSQL"&gt;
		&lt;ejen stacktrace="true"&gt;
			&lt;source file="${project.root}/${doml.filename}"/&gt;
			&lt;template file="${template.dir}/${template.set}/sql/prepareSQL.xsl"/&gt;
		&lt;/ejen&gt;
	&lt;/target&gt;
	


	&lt;target name="createSQLs"&gt;
		&lt;ejen stacktrace="true"&gt;
			&lt;source file="${project.root}/${doml.filename}"/&gt;
			&lt;template file="${template.dir}/${template.set}/sql/createSQLs.xsl"/&gt;
		&lt;/ejen&gt;
	&lt;/target&gt;

	&lt;target name="sqlsplitter" &gt;
		&lt;sqlsplitter dir="${project.root}"
	   			resultsql="${project.root}/${project.name}.sql"
	   			splitsqlprimary="${database.splitsqlprimary}"
	   			/&gt;
	&lt;/target&gt;

	&lt;target name="generateAll" &gt;
  		&lt;ant antfile="${project.root}/build_dods.xml" target="generateSql"/&gt;
      &lt;ant antfile="${project.root}/build_java.xml" target="generate_all"/&gt;
      &lt;ant antfile="${project.root}/build_dods.xml" target="compileJava"/&gt;
	&lt;/target&gt;

	&lt;target name="generateAllSplit" &gt;
  		&lt;ant antfile="${project.root}/build_dods.xml" target="generateSqlSplit"/&gt;
      &lt;ant antfile="${project.root}/build_java.xml" target="generate_all"/&gt;
      &lt;ant antfile="${project.root}/build_dods.xml" target="compileJava"/&gt;
	&lt;/target&gt;

	&lt;target name="generateSql" depends="dods,prepareSQL"&gt;
      &lt;ant antfile="${project.root}/build_sql.xml" target="generateSql"/&gt;
      &lt;ant antfile="${project.root}/build_dods.xml" target="createSQLs"/&gt;
	&lt;/target&gt;

	&lt;target name="generateSqlSplit" depends="dods,prepareSQL"&gt;
      &lt;ant antfile="${project.root}/build_sql.xml" target="generateSql"/&gt;
      &lt;ant antfile="${project.root}/build_dods.xml" target="sqlsplitter"/&gt;
	&lt;/target&gt;

	&lt;target name="generateJava" depends="dods"&gt;
      &lt;ant antfile="${project.root}/build_java.xml" target="generate_all"/&gt;
      &lt;ant antfile="${project.root}/build_dods.xml" target="compileJava"/&gt;
	&lt;/target&gt;
	
	&lt;target name="generateJavaNoCompile" depends="dods"&gt;
      &lt;ant antfile="${project.root}/build_java.xml" target="generate_all"/&gt;
	&lt;/target&gt;
	
	&lt;target name="generateNoCompile" depends="generateSql"&gt;
      &lt;ant antfile="${project.root}/build_java.xml" target="generate_all"/&gt;
	&lt;/target&gt;
	
	&lt;target name="generateNoCompileSplit" depends="generateSqlSplit"&gt;
      &lt;ant antfile="${project.root}/build_java.xml" target="generate_all"/&gt;
	&lt;/target&gt;
	
	
	&lt;target name="compileJava"&gt;
     &lt;mkdir dir="${project.root}/classes"/&gt;
     &lt;javac srcdir="${project.root}" destdir="${project.root}/classes" deprecation="off" fork="yes" memoryInitialSize="80m" memoryMaximumSize="800m" nowarn="yes"/&gt;
	 &lt;copy todir="${project.root}/classes/org/enhydra/dods" includeEmptyDirs="no"&gt;
	     &lt;fileset dir="${project.root}/org/enhydra/dods"&gt;
	          &lt;include name="DODSClassList.xml"/&gt;
	     &lt;/fileset&gt;
	 &lt;/copy&gt;
	
	&lt;/target&gt;

	

	&lt;!-- ================================================================================ --&gt;

&lt;/project&gt;



</xsl:template>

</xsl:stylesheet>
