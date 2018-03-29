<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:fo="http://www.w3.org/1999/XSL/Format" version="1.0">
    <xsl:output method="xml" version="1.0" encoding="UTF-8" indent="yes"/>

    <xsl:template match="/volume">

		<!-- ITALIAN INDEX
       <fo:block break-before="page" space-before="10mm" space-after="10mm" font-size="18pt"
              display-align="center" text-align="center" text-align-last="center"
              start-indent="18mm" end-indent="18mm" width="130mm" height="20mm"
              background-color="#EEEEEE" border-style="outset" border-color="#888888"
              padding-top="5pt" padding-bottom="5pt">
                        <fo:inline>Italian index</fo:inline></fo:block>
              
        <xsl:if test="&SUP_SUP_TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &SUP_SUP_DOMAIN_NUMBER : &SUP_SUP_DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:if test="&SUP_TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &SUP_DOMAIN_NUMBER : &SUP_DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:if test="&TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &DOMAIN_NUMBER : &DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:apply-templates select="d:contribution/d:data/entry[@lang='ita' and domain='&DOMAIN_NUMBER']"
            xmlns:d="http://www-clips.imag.fr/geta/services/dml"/>
            -->
   
   		<!-- FRENCH INDEX-->
		<fo:block break-before="page" space-before="10mm" space-after="10mm" font-size="18pt"
              display-align="center" text-align="center" text-align-last="center"
              start-indent="18mm" end-indent="18mm" width="130mm" height="20mm"
              background-color="#EEEEEE" border-style="outset" border-color="#888888"
              padding-top="5pt" padding-bottom="5pt">
                        <fo:inline>French index</fo:inline></fo:block>
                  
          <xsl:if test="&SUP_SUP_TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &SUP_SUP_DOMAIN_NUMBER : &SUP_SUP_DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:if test="&SUP_TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &SUP_DOMAIN_NUMBER : &SUP_DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:if test="&TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &DOMAIN_NUMBER : &DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:apply-templates select="d:contribution/d:data/entry[@lang='fra' and domain='&DOMAIN_NUMBER']"
            xmlns:d="http://www-clips.imag.fr/geta/services/dml"/>
             
            
		<!-- GERMAN INDEX-->
        <fo:block space-before="10mm" space-after="10mm" font-size="18pt"
              display-align="center" text-align="center" text-align-last="center"
              start-indent="18mm" end-indent="18mm" width="130mm" height="20mm"
              background-color="#EEEEEE" border-style="outset" border-color="#888888"
              padding-top="5pt" padding-bottom="5pt">
                        <fo:inline>German index</fo:inline></fo:block>
                      
          <xsl:if test="&SUP_SUP_TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &SUP_SUP_DOMAIN_NUMBER : &SUP_SUP_DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:if test="&SUP_TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &SUP_DOMAIN_NUMBER : &SUP_DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:if test="&TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &DOMAIN_NUMBER : &DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:apply-templates select="d:contribution/d:data/entry[@lang='deu' and domain='&DOMAIN_NUMBER']"
            xmlns:d="http://www-clips.imag.fr/geta/services/dml"/>
           
            
       <!-- SLOVENE INDEX
       <fo:block space-before="10mm" space-after="10mm" font-size="18pt"
              display-align="center" text-align="center" text-align-last="center"
              start-indent="18mm" end-indent="18mm" width="130mm" height="20mm"
              background-color="#EEEEEE" border-style="outset" border-color="#888888"
              padding-top="5pt" padding-bottom="5pt">
                        <fo:inline>Slovene index</fo:inline></fo:block>
                      
         <xsl:if test="&SUP_SUP_TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &SUP_SUP_DOMAIN_NUMBER : &SUP_SUP_DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:if test="&SUP_TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &SUP_DOMAIN_NUMBER : &SUP_DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:if test="&TEST()">
        	<fo:block text-align="justify" font-size="14pt" space-after="5mm">
            	<fo:inline>Domain &DOMAIN_NUMBER : &DOMAIN_NAME</fo:inline>
        	</fo:block>
        </xsl:if>
        
        <xsl:apply-templates select="d:contribution/d:data/entry[@lang='slv' and domain='&DOMAIN_NUMBER']"
            xmlns:d="http://www-clips.imag.fr/geta/services/dml"/>
               -->
    </xsl:template>


    <xsl:template match="entry">
        <fo:block margin-left="0.25em">

            <fo:block text-align="justify" text-align-last="justify" hyphenate="false">
                <xsl:value-of select="term"/>
                <fo:leader leader-pattern="dots" margin-left="0.5em" leader-length.optimum="0%"/>
                <fo:inline keep-together.within-line="always">
                    <fo:leader leader-pattern="dots" leader-length.optimum="0em"
                        margin-right="0.5em"/>
                    <fo:basic-link>
                        <xsl:attribute name="internal-destination">
                            <xsl:value-of select="@id"/>
                        </xsl:attribute>
                        <fo:page-number-citation>
                            <xsl:attribute name="ref-id">
                                <xsl:value-of select="@id"/>
                            </xsl:attribute>
                        </fo:page-number-citation>
                    </fo:basic-link>
                </fo:inline>
            </fo:block>
        </fo:block>
    </xsl:template>


</xsl:stylesheet>
