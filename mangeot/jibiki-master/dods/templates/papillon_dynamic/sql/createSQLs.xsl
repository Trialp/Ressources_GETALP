<?xml version="1.0" encoding="UTF-8"?>

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
				xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                extension-element-prefixes="msg gvs mta xmi miu"
                exclude-result-prefixes="xalan xalan-ext gvs msg mta stu ver xmi">
                
<xsl:output omit-xml-declaration="yes"/>
	
<xsl:template match ="/">
  <miu:createSQLcomplete/>
  <gvs:clear/>
  <miu:createSQLdrop/>
</xsl:template>

</xsl:stylesheet>
