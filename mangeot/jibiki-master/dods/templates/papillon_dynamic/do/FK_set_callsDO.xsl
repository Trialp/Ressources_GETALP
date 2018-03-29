<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
	<xsl:output method="text" encoding="iso-8859-1"/>
	<xsl:template name="fK_Sset_callsDO">_set<xsl:value-of select="common:capitalizeName(@FK_NAME)"/>( x.get<xsl:value-of select="common:capitalizeName(@PK_NAME)"/>() );
	</xsl:template>
</xsl:stylesheet>
