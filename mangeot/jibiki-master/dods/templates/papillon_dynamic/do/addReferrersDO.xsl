<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                version="1.0">
	<xsl:output method="text" encoding="iso-8859-1"/>
	<xsl:template name="addReferrersDO">
	<xsl:value-of select="dodst:addRefAttr(@name,@do_name,/TABLE/PACKAGE,/TABLE/CLASS_NAME)"/>
	</xsl:template>
</xsl:stylesheet>
 

