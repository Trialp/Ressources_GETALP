<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="fileFooter2">
        get_DataStruct().isEmpty = false;
        setPersistent(true);
        markClean();
        syncStructs(false);
//        refs = null;
        autoSaveAllowed = true;
    }

    	public static String getColumnsNameString(String dbtblname){
        String columnsNameString = null;
        
    	if(useOrderedWithTable==null){
    		columnsNameString=dbtblname+&quot;.*&quot;;
    	}else{
			columnsNameString = &quot;&quot;;
   		String tableName;
			if (useOrderedWithTable.booleanValue()){
				tableName=&quot;@T@_&quot;+dbtblname+&quot;_@@.&quot;;
			}else{
				tableName=&quot;@F@_&quot;+dbtblname+&quot;_@@.&quot;;
			}
			String oidStr = <xsl:value-of select="/TABLE/CLASS_NAME"/>DO.get_OIdColumnName();
			String verStr = <xsl:value-of select="/TABLE/CLASS_NAME"/>DO.get_versionColumnName();
			<xsl:if test="DO_IS_OID_BASED='true'">columnsNameString = tableName+oidStr+&quot;, &quot; ;</xsl:if>
			if ( versioning ) {
				columnsNameString +=tableName+verStr+&quot;, &quot;;
			}
</xsl:template>
</xsl:stylesheet>
