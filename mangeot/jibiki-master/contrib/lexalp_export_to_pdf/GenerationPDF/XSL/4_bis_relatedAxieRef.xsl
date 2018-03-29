<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
    
    <!-- Copy (deep) node -->
    <xsl:template match="node()|@*">
        <!-- Copy the current node -->
        <xsl:copy>
            <!-- Including any attributes it has and any child nodes -->
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>
    
    
    <!-- Add axies related in axie  -->
     <xsl:template match="axieref" xmlns:d="http://www-clips.imag.fr/geta/services/dml"> 
        <xsl:variable
            name="axiereference"
            select="@idref">
        </xsl:variable>
        
                <xsl:for-each select="//axie[@id=$axiereference]">
                    <axieref2>
                    	<xsl:attribute name="idref"><xsl:value-of select="@id"/></xsl:attribute>
                     	<xsl:for-each select="entry">
                     		<termref2>
                    			<xsl:attribute name="idref"><xsl:value-of select="@id"/></xsl:attribute>
                    			<xsl:attribute name="legalSystem"><xsl:value-of select="@legalSystem"/></xsl:attribute>
                    			<xsl:attribute name="lang"><xsl:value-of select="@lang"/></xsl:attribute>
                    			<xsl:value-of select="term"/>
                    		</termref2>
                	 	</xsl:for-each>
                	</axieref2>
                </xsl:for-each>
                
    </xsl:template>
    
     <!-- Add axies related in axie (inverse link !) -->
     <xsl:template match="axie" xmlns:d="http://www-clips.imag.fr/geta/services/dml"> 
        <xsl:variable
            name="id"
            select="@id">
        </xsl:variable>
        	<xsl:copy>
        		<xsl:apply-templates select="@*|node()"/>
        
                <xsl:for-each select="//axieref[@idref=$id]">
                    <axieref2>
                        <xsl:attribute name="idref"><xsl:value-of select="parent::node()/attribute::id"/></xsl:attribute>
                        <xsl:for-each select="parent::node()/entry">
                     		<termref2>
                    			<xsl:attribute name="idref"><xsl:value-of select="@id"/></xsl:attribute>
                    			<xsl:attribute name="legalSystem"><xsl:value-of select="@legalSystem"/></xsl:attribute>
                    			<xsl:attribute name="lang"><xsl:value-of select="@lang"/></xsl:attribute>
                    			<xsl:value-of select="term"/>
                    		</termref2>
                	 	</xsl:for-each>
                	</axieref2>
                </xsl:for-each>
                
                <xsl:for-each select="//axieref2[@idref=$id]">
                    <axieref2>
                        <xsl:attribute name="idref"><xsl:value-of select="parent::node()/attribute::id"/></xsl:attribute>
                        <xsl:for-each select="parent::node()/entry">
                     		<termref2>
                    			<xsl:attribute name="idref"><xsl:value-of select="@id"/></xsl:attribute>
                    			<xsl:attribute name="legalSystem"><xsl:value-of select="@legalSystem"/></xsl:attribute>
                    			<xsl:attribute name="lang"><xsl:value-of select="@lang"/></xsl:attribute>
                    			<xsl:value-of select="term"/>
                    		</termref2>
                	 	</xsl:for-each>
                	</axieref2>
                </xsl:for-each>
                
            </xsl:copy>
    </xsl:template>
    
</xsl:stylesheet>
