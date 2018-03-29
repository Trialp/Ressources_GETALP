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
  <miu:resetCreateVectors/>

  <xsl:call-template name="columns"/>
  
  <xsl:call-template name="indexes"/>

</xsl:template>



<!-- columns -->

<xsl:template name="columns">

create table <xsl:value-of select="TABLE_NAME"/> 
(
<xsl:if test="miu:getDodsProperty('CommentStart', /TABLE/DB_VENDOR)!='NONE'"><xsl:if
test="miu:getDodsProperty('CommentStart', /TABLE/DB_VENDOR)!='none'"><xsl:value-of
select="miu:getDodsProperty('CommentStart', /TABLE/DB_VENDOR)"/> class <xsl:value-of select="CLASS_NAME"/><xsl:text></xsl:text>
<xsl:value-of select="miu:getDodsProperty('CommentEnd', /TABLE/DB_VENDOR)"/><xsl:text>
</xsl:text></xsl:if></xsl:if>
<xsl:text>    </xsl:text>
    <xsl:choose>
      <xsl:when test="DO_IS_OID_BASED='false'"></xsl:when>
      <xsl:otherwise>
        <xsl:choose>
          <xsl:when test="COLUMN/IS_PRIMARY_KEY='true'"></xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="miu:getDodsProperty('OidDbColumnName', /TABLE/DB_VENDOR)"/><xsl:text> </xsl:text>
            <xsl:value-of select="miu:getDodsProperty('OidDbType', /TABLE/DB_VENDOR)"/> NOT NULL PRIMARY KEY ,
    <xsl:value-of select="miu:getDodsProperty('VersionDbColumnName', /TABLE/DB_VENDOR)"/><xsl:text> </xsl:text>
    <xsl:value-of select="miu:getDataType('INTEGER', /TABLE/DB_VENDOR)"/>  NOT NULL ,<xsl:text>
    </xsl:text>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:otherwise>
    </xsl:choose>
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
    
    <xsl:if test="COLUMN[IS_PRIMARY_KEY='true']"> ,    
    CONSTRAINT <xsl:value-of select="miu:returnFixedConstraintName(/TABLE/TABLE_NAME,COLUMN[@name],'CREATE','PK',/TABLE/TABLE_NAME,COLUMN[@name],/TABLE/DB_VENDOR)"/> PRIMARY KEY (<xsl:for-each
select="COLUMN[IS_PRIMARY_KEY='true']"><xsl:value-of select="@name"/><xsl:if
test="position()!=last()">,</xsl:if></xsl:for-each>)</xsl:if>

    <xsl:if test="COLUMN[IS_UNIQUE='true']"> ,    
    CONSTRAINT <xsl:value-of select="miu:returnFixedConstraintName(/TABLE/TABLE_NAME,COLUMN[@name],'CREATE','U',/TABLE/TABLE_NAME,COLUMN[@name],/TABLE/DB_VENDOR)"/> UNIQUE (<xsl:for-each
select="COLUMN[IS_UNIQUE='true']"><xsl:value-of
select="@name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>)</xsl:if>
    
    <xsl:if test="COLUMN/REFERENCE_OBJECT"> ,    
<xsl:value-of select="miu:returnForeignKeyConstraints(/TABLE/DB_VENDOR)"/>
    </xsl:if>
);


</xsl:template>



<!-- column1 -->

<xsl:template name="column1">
    <xsl:value-of select="@name"/>
    <xsl:text> </xsl:text>
    <xsl:choose>
      <xsl:when test="DB_TYPE='none'"><xsl:value-of select="miu:getDodsProperty('OidDbType', /TABLE/DB_VENDOR)"/></xsl:when>
      <xsl:otherwise><xsl:value-of select="miu:getDataType(DB_TYPE, /TABLE/DB_VENDOR)"/></xsl:otherwise>
    </xsl:choose>
    
    <xsl:choose>
      <xsl:when test="SIZE"><xsl:if test="DB_TYPE!='none'"><xsl:if test="DB_TYPE!='LONGVARBINARY'"> (<xsl:value-of
select="SIZE"/>)</xsl:if></xsl:if></xsl:when>
      <xsl:otherwise/>
    </xsl:choose>
    
    <xsl:choose>
      <xsl:when test="CAN_BE_NULL">
        <xsl:if test="CAN_BE_NULL='true'"/>
        <xsl:if test="CAN_BE_NULL='false'"> NOT NULL</xsl:if>
      </xsl:when>
      <xsl:otherwise> NOT NULL</xsl:otherwise>
    </xsl:choose>
    <xsl:if test="position()!=last()"> ,<xsl:text>
    </xsl:text></xsl:if><!--
    --><xsl:if test="REFERENCE_OBJECT/CONSTRAINT='true'">
    <miu:fillVector attribut0="./../TABLE_NAME" attribut1="@name" attribut2="REFERENCE_OBJECT/TABLE_NAME"
attribut3="REFERENCE_OBJECT/FOREIGN_KEY_COLUMN" attribut4="REFERENCE_OBJECT/FOREIGN_KEY_GROUP" attribut5="./../DB_VENDOR"/>
    </xsl:if>
</xsl:template>



<!-- indexes -->

<xsl:template name="indexes">
  <xsl:for-each select="INDEX">CREATE <xsl:if test="@unique='true'">UNIQUE </xsl:if><xsl:if
test="@clustered='true'">CLUSTERED </xsl:if>INDEX <xsl:value-of select="miu:returnFixedConstraintName(./../TABLE_NAME,@id,'CREATE','I',./../TABLE_NAME,@id,/TABLE/DB_VENDOR)"/> ON <xsl:value-of 
select="./../TABLE_NAME"/> (<xsl:for-each select="INDEX_COLUMN"><xsl:value-of
select="@id"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
</xsl:for-each><xsl:text>

</xsl:text>
</xsl:template>



</xsl:stylesheet>
