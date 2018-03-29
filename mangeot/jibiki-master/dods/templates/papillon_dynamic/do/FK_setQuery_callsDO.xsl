<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:common="org.enhydra.dods.Common"
                version="1.0">
<xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="fK_set_query_calls">	    q.setQuery<xsl:value-of select="common:capitalizeName(@FK_NAME)"/>( get<xsl:value-of select="common:capitalizeName(@PK_NAME)"/>() );
	</xsl:template>
</xsl:stylesheet>
