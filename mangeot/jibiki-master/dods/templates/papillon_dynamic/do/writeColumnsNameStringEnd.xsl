<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="columnsNameStringEnd">
			if ( versioning ) {
				columnsNameString += tableName + verStr ;
			}else{
				columnsNameString = columnsNameString.substring(0,columnsNameString.length()-2);
			}
		}
        return columnsNameString;
	}
</xsl:template>
</xsl:stylesheet>