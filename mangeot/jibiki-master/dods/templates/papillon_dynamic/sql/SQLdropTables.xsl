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
  <gvs:clear/>
  <xsl:apply-templates />
</xsl:template>

<!-- TABLE -->

<xsl:template match="TABLE">
<!--
  <gvs:put name="columnLoop" select="'one'"/>
-->
  <xsl:call-template name="columns"/>

<!--
  <gvs:put name="columnLoop" select="'two'"/>
-->
  <xsl:call-template name="indexes"/>
  
  <miu:fillVecDropTable tableName="TABLE_NAME"/>

</xsl:template>



<!-- columns -->

<xsl:template name="columns">

<!--
  <xsl:if test="gvs:get('columnLoop')='one'">
-->
    <xsl:for-each select="COLUMN">
    <xsl:choose>
      <xsl:when test="IS_CONSTANT">
        <xsl:if test="IS_CONSTANT!='true'">
          <xsl:call-template name="column1"/>
        </xsl:if>
      </xsl:when>
      <xsl:otherwise>
      <xsl:call-template name="column1"/>
      </xsl:otherwise>
    </xsl:choose>
    </xsl:for-each>
    
    <xsl:if test="COLUMN[IS_PRIMARY_KEY='true']">
        <miu:fillVecDropPrimaryKey tableName="TABLE_NAME" columnName="@name" vendorName="DB_VENDOR"/>

    </xsl:if>

    <xsl:if test="COLUMN[IS_UNIQUE='true']">
        <miu:fillVecDropUnique tableName="TABLE_NAME" columnName="@name" vendorName="DB_VENDOR"/>

    </xsl:if>
    


</xsl:template>



<!-- indexes -->

<xsl:template name="indexes">



  <xsl:for-each select="INDEX">
      <miu:fillVecDropIndex tableName="./../TABLE_NAME" indexName="@id" vendorName="./../DB_VENDOR"/>
      
  </xsl:for-each><xsl:text>
</xsl:text>


</xsl:template>



<!-- column1 -->

<xsl:template name="column1">

    <xsl:if test="REFERENCE_OBJECT[CONSTRAINT='true']">
<!--    
    <miu:fillVector attribut0="./../TABLE_NAME" attribut1="@name" attribut2="REFERENCE_OBJECT/TABLE_NAME"
attribut3="REFERENCE_OBJECT/FOREIGN_KEY_COLUMN" attribut4="REFERENCE_OBJECT/FOREIGN_KEY_GROUP"/>
-->
    </xsl:if>
</xsl:template>



</xsl:stylesheet>
