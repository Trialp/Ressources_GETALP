<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<xsl:stylesheet version="2.0" xmlns:d="http://www-clips.imag.fr/geta/services/dml"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:fo="http://www.w3.org/1999/XSL/Format"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.w3.org/1999/xhtml">
    <!-- FIXME: The date/time functions require xslt 2.0 processing, using Saxon 8 -->
    <xsl:output method="xml" encoding="utf-8" indent="yes"/>
    <!-- general templates please do not modify -->
    <xsl:template match="/volume">
        <fo:root>
            <fo:layout-master-set>
                <fo:simple-page-master size="A4" margin="25mm 25mm 25mm 25mm"
                    master-name="PageMaster-Cover">
                    <fo:region-body margin="0mm 0mm 0mm 0mm"/>
                </fo:simple-page-master>
                <fo:simple-page-master margin="10mm 10mm 10mm 10mm" master-name="PageMaster"
                    size="A4">
                    <fo:region-body margin="10mm 10mm 10mm 10mm"/>
                    <fo:region-before region-name="header" extent="10mm" display-align="after"/>
                    <fo:region-after region-name="footer" extent="10mm" display-align="before"/>
                </fo:simple-page-master>
            </fo:layout-master-set>
            <fo:page-sequence master-reference="PageMaster-Cover" force-page-count="no-force">
                <fo:flow flow-name="xsl-region-body" font-size="18pt" font-family="Arial">
                    <fo:block space-before="90mm" space-before.conditionality="retain"
                        space-after="60mm" font-size="24pt" display-align="center"
                        text-align="center" text-align-last="center" start-indent="18mm"
                        end-indent="18mm" width="100mm" height="20mm" background-color="#EEEEEE"
                        border-style="outset" border-color="#888888" padding-top="5pt"
                        padding-bottom="5pt">
                        <fo:block>LexALP Dictionary Export</fo:block>
                    </fo:block>
                    <fo:block font-size="14pt" text-align="center" text-align-last="center"
                        space-after="5mm">Domain &DOMAIN_NUMBER : &DOMAIN_NAME</fo:block>
                    <fo:block font-size="12pt" text-align="center" text-align-last="center"
                        space-after="5mm">Date: <fo:inline font-style="italic">&DATE
                        </fo:inline></fo:block>
                </fo:flow>
            </fo:page-sequence>
            <fo:page-sequence master-reference="PageMaster" initial-page-number="1">
                <fo:static-content flow-name="header">
                    <fo:block font-size="10pt" font-family="Arial" text-align="center" border-after-width="thin"
                        border-after-style="solid">Domain &DOMAIN_NUMBER : &DOMAIN_NAME <fo:inline font-style="italic">&DATE
                        </fo:inline></fo:block>
                </fo:static-content>
                <fo:static-content flow-name="footer">
                    <fo:block font-size="10pt" text-align="center">
                        <fo:inline>
                            <fo:page-number/>
                        </fo:inline>
                    </fo:block>
                </fo:static-content>
                <fo:flow flow-name="xsl-region-body" font-size="12pt" font-family="Arial">

                    <fo:block break-before="page" space-before="90mm"
                        space-before.conditionality="retain" space-after="60mm" font-size="24pt"
                        display-align="center" text-align="center" text-align-last="center"
                        start-indent="18mm" end-indent="18mm" width="130mm" height="20mm"
                        background-color="#EEEEEE" border-style="outset" border-color="#888888"
                        padding-top="5pt" padding-bottom="5pt">
                        <fo:block>Part I : axies with related terms</fo:block>
                    </fo:block>

                    <xsl:apply-templates select="d:contribution/d:data/axie[entry/domain[1]='&DOMAIN_NUMBER']"/>

                    <fo:block break-before="page" space-before="90mm"
                        space-before.conditionality="retain" space-after="60mm" font-size="24pt"
                        display-align="center" text-align="center" text-align-last="center"
                        start-indent="18mm" end-indent="18mm" width="130mm" height="20mm"
                        background-color="#EEEEEE" border-style="outset" border-color="#888888"
                        padding-top="5pt" padding-bottom="5pt">
                        <fo:block>Part II : terms <fo:inline font-size="12pt">(not related to
                            axies)</fo:inline></fo:block>
                    </fo:block>

                    <fo:block break-before="page">
                        <xsl:apply-templates select="d:contribution/d:data/entry[domain[1]='&DOMAIN_NUMBER']"
                        />
                    </fo:block>

                    <fo:block break-before="page" space-before="90mm"
                        space-before.conditionality="retain" space-after="60mm" font-size="24pt"
                        display-align="center" text-align="center" text-align-last="center"
                        start-indent="18mm" end-indent="18mm" width="130mm" height="20mm"
                        background-color="#EEEEEE" border-style="outset" border-color="#888888"
                        padding-top="5pt" padding-bottom="5pt">Index</fo:block>
		        </fo:flow>
            </fo:page-sequence>
        </fo:root>
    </xsl:template>


    <xsl:template match="//axie">
        <fo:block break-before="page" font-size="10pt">
            <xsl:attribute name="id">
                <xsl:value-of select="@id"/>
            </xsl:attribute>

            <fo:inline font-weight="bold" font-size="12pt">Concept (id <fo:inline
                    font-style="italic">
                    <xsl:value-of select="@id"/>
                </fo:inline>) : </fo:inline>

            <fo:block space-before="3mm" space-after="3mm" start-indent="18mm" end-indent="18mm"
                width="130mm" height="20mm" background-color="#EEEEEE" border-style="outset"
                border-color="#888888" padding-top="5pt" padding-bottom="5pt" padding-right="10pt"
                padding-left="10pt">

                <fo:block>
                    <xsl:for-each select="entry[@legalSystem='AC' and @lang='ita']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <fo:basic-link>
                                    <xsl:attribute name="internal-destination">
                                        <xsl:value-of select="@id"/>
                                    </xsl:attribute>
                                    <xsl:value-of select="term"/>
                                </fo:basic-link>
                            </fo:basic-link> (ita, AC) </fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='AC' and @lang='fra']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (fra, AC)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='AC' and @lang='deu']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (deu, AC)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='AC' and @lang='slv']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (slv, AC)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='AC' and @lang='eng']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (eng, AC)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='EU' and @lang='ita']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (ita, EU)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='EU' and @lang='fra']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (fra, EU)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='EU' and @lang='deu']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (deu, EU)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='EU' and @lang='slv']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (slv, EU)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='EU' and @lang='eng']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (eng, EU)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='INT' and @lang='ita']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (ita, INT)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='INT' and @lang='fra']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (fra, INT)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='INT' and @lang='deu']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (deu, INT)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='INT' and @lang='slv']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (slv, INT)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='INT' and @lang='eng']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (eng, INT)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='IT']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (<xsl:value-of select="@lang"/>, IT)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='CH' and @lang='ita']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (ita, CH)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='FR']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (<xsl:value-of select="@lang"/>, FR)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='CH' and @lang='fra']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (fra, CH)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='DE']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (<xsl:value-of select="@lang"/>, DE)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='AT']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (<xsl:value-of select="@lang"/>, AT)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='CH' and @lang='deu']">
                        <fo:block>
                            <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (deu, CH)</fo:block>
                    </xsl:for-each>
                    <xsl:for-each select="entry[@legalSystem='SL']">
                        <fo:block><fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@id"/>
                                </xsl:attribute>
                                <xsl:value-of select="term"/>
                            </fo:basic-link> (<xsl:value-of select="@lang"/>, SL)</fo:block>
                    </xsl:for-each>
                </fo:block>

                <xsl:for-each select="axieref2[@idref!='']">
                    <fo:block space-before="3mm">
                        <fo:block>related to concept <fo:basic-link>
                                <xsl:attribute name="internal-destination">
                                    <xsl:value-of select="@idref"/>
                                </xsl:attribute>
                                <fo:inline font-style="italic">
                                    <xsl:value-of select="@idref"/>
                                </fo:inline> (page <fo:page-number-citation>
                                    <xsl:attribute name="ref-id">
                                        <xsl:value-of select="@idref"/>
                                    </xsl:attribute>
                                </fo:page-number-citation>) </fo:basic-link></fo:block>

                        <xsl:for-each select="termref2" xmlns:fo="http://www.w3.org/1999/XSL/Format">
                            <fo:block start-indent="24mm" end-indent="21mm">
                                <xsl:for-each select=".[@legalSystem='AC' and @lang='ita']">
                                    <fo:block>
                                  
                                            <fo:basic-link>
                                                <xsl:attribute name="internal-destination">
                                                  <xsl:value-of select="@idref"/>
                                                </xsl:attribute>
                                                <xsl:value-of select="."/>
                                            </fo:basic-link> (ita, AC) </fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='AC' and @lang='fra']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (fra, AC)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='AC' and @lang='deu']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (deu, AC)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='AC' and @lang='slv']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (slv, AC)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='AC' and @lang='eng']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (eng, AC)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='EU' and @lang='ita']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (ita, EU)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='EU' and @lang='fra']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (fra, EU)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='EU' and @lang='deu']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (deu, EU)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='EU' and @lang='slv']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (slv, EU)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='EU' and @lang='eng']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (eng, EU)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='INT' and @lang='ita']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (ita, INT)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='INT' and @lang='fra']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (fra, INT)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='INT' and @lang='deu']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (deu, INT)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='INT' and @lang='slv']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (slv, INT)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='INT' and @lang='eng']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (eng, INT)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='IT']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (<xsl:value-of select="@lang"/>,
                                    IT)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='CH' and @lang='ita']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (ita, CH)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='FR']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (<xsl:value-of select="@lang"/>,
                                    FR)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='CH' and @lang='fra']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (fra, CH)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='DE']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (<xsl:value-of select="@lang"/>,
                                    DE)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='AT']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (<xsl:value-of select="@lang"/>,
                                    AT)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='CH' and @lang='deu']">
                                    <fo:block>
                                        <fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (deu, CH)</fo:block>
                                </xsl:for-each>
                                <xsl:for-each select=".[@legalSystem='SL']">
                                    <fo:block><fo:basic-link>
                                            <xsl:attribute name="internal-destination">
                                                <xsl:value-of select="@idref"/>
                                            </xsl:attribute>
                                            <xsl:value-of select="."/>
                                        </fo:basic-link> (<xsl:value-of select="@lang"/>,
                                    SL)</fo:block>
                                </xsl:for-each>


                            </fo:block>
                        </xsl:for-each>
                    </fo:block>
                </xsl:for-each>
            </fo:block>
            <xsl:apply-templates select="entry[@legalSystem='AC' and @lang='ita']"/>
            <xsl:apply-templates select="entry[@legalSystem='AC' and @lang='fra']"/>
            <xsl:apply-templates select="entry[@legalSystem='AC' and @lang='deu']"/>
            <xsl:apply-templates select="entry[@legalSystem='AC' and @lang='slv']"/>
            <xsl:apply-templates select="entry[@legalSystem='AC' and @lang='eng']"/>
            <xsl:apply-templates select="entry[@legalSystem='EU' and @lang='ita']"/>
            <xsl:apply-templates select="entry[@legalSystem='EU' and @lang='fra']"/>
            <xsl:apply-templates select="entry[@legalSystem='EU' and @lang='deu']"/>
            <xsl:apply-templates select="entry[@legalSystem='EU' and @lang='slv']"/>
            <xsl:apply-templates select="entry[@legalSystem='EU' and @lang='eng']"/>
            <xsl:apply-templates select="entry[@legalSystem='INT' and @lang='ita']"/>
            <xsl:apply-templates select="entry[@legalSystem='INT' and @lang='fra']"/>
            <xsl:apply-templates select="entry[@legalSystem='INT' and @lang='deu']"/>
            <xsl:apply-templates select="entry[@legalSystem='INT' and @lang='slv']"/>
            <xsl:apply-templates select="entry[@legalSystem='INT' and @lang='eng']"/>
            <xsl:apply-templates select="entry[@legalSystem='IT']"/>
            <xsl:apply-templates select="entry[@legalSystem='CH' and @lang='ita']"/>
            <xsl:apply-templates select="entry[@legalSystem='FR']"/>
            <xsl:apply-templates select="entry[@legalSystem='CH' and @lang='fra']"/>
            <xsl:apply-templates select="entry[@legalSystem='DE']"/>
            <xsl:apply-templates select="entry[@legalSystem='AT']"/>
            <xsl:apply-templates select="entry[@legalSystem='CH' and @lang='deu']"/>
            <xsl:apply-templates select="entry[@legalSystem='SL']"/>
        </fo:block>
    </xsl:template>

    <!-- dictionary specific templates -->
    <!--SELECTION DES PARTIES A AFFICHER-->
    <xsl:template match="entry">
        <fo:block space-before="10pt" font-size="10pt">
            <xsl:attribute name="id">
                <xsl:value-of select="@id"/>
            </xsl:attribute>
            <fo:block>
                <fo:inline font-weight="bold" font-size="12pt">
                    <xsl:value-of select="term/text()"/>
                </fo:inline>
                <fo:inline font-size="10pt"> , <xsl:value-of select="grammar/text()"/>
                </fo:inline>
                <fo:inline font-size="10pt"> [<xsl:value-of select="@status"/>/<xsl:value-of
                        select="@process_status"/>] </fo:inline>
            </fo:block>

            <!-- CREATION, MODIFICATION 
            <fo:block text-align="justify" margin-left="1em" font-size="10pt" space-before="1pt"> ID
                term <fo:inline font-style="italic">
                    <xsl:value-of select="@id"/>
                </fo:inline>
                , created by <xsl:value-of select="//d:metadata/d:author/text()"/>
                    , last modified by <xsl:value-of select="//d:modification/d:author/text()"/>
            </fo:block>-->
				
			<!-- phraseological unit -->
			<xsl:if test="phraseological-unit/text()='true'">
            	<fo:block text-align="justify" margin-left="1em" space-before="1pt">
                	<fo:inline font-variant="small-caps">Phraseological unit</fo:inline>
            	</fo:block>
            </xsl:if>
            
            <!-- LANGUAGES & LEGAL SYSTEM -->
            <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                <fo:inline font-variant="small-caps">Language: </fo:inline>
                <xsl:value-of select="@lang"/>
                <fo:inline font-variant="small-caps">, Legal system: </fo:inline>
                <xsl:value-of select="@legalSystem"/>
            </fo:block>

            <!-- DOMAINS -->
            <xsl:apply-templates select="domain"/>

            <!-- Related terms and forms -->
            <!-- We first give harmonized elements, then non harmonised ones... -->
            <xsl:apply-templates select="relatedTerm[@isHarmonised='true']"/>
            <xsl:apply-templates select="relatedForm[@isHarmonised='true']"/>
            <!-- We should distinguish forms of harmonized and non harmonized related terms -->
            <xsl:apply-templates
                select="relatedTerm[@isHarmonised!='true']|relatedTerm[not(@isHarmonised)]"/>
            <xsl:apply-templates
                select="relatedForm[@isHarmonised!='true']|relatedForm[not(@isHarmonised)]"/>

            <!--DEFINITION-->
            <xsl:if test="string(definition/text/text()) ">
                <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                    <fo:inline font-variant="small-caps">Definition: </fo:inline>
                    <xsl:value-of select="definition/text/text()"/>
                    <fo:inline font-size="10pt">
                        <xsl:apply-templates select="definition/source"/>
                    </fo:inline>
                </fo:block>
            </xsl:if>

            <!--CONTEXT-->
            <xsl:if test="string(context/text/text()) ">
                <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                    <fo:inline font-variant="small-caps">Context: </fo:inline>
                    <xsl:value-of select="context/text/text()"/>
                    <fo:inline font-size="10pt">
                        <xsl:apply-templates select="context/source"/>
                    </fo:inline>
                </fo:block>
            </xsl:if>

            <!--NOTE-->
            <xsl:if test="string(note/text/text()) ">
                <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                    <fo:inline font-variant="small-caps">Note: </fo:inline>
                    <xsl:value-of select="note/text/text()"/>
                    <fo:inline font-size="10pt">
                        <xsl:apply-templates select="note/source"/>
                    </fo:inline>
                </fo:block>
            </xsl:if>

            <!--HARMONISATION NOTE-->
            <xsl:if test="string(harmonisation-note/text/text()) ">
                <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                    <fo:inline font-variant="small-caps">Harmonisation Note: </fo:inline>
                    <xsl:value-of select="harmonisation-note/text/text()"/>
                    <fo:inline font-size="10pt">
                        <xsl:apply-templates select="harmonisation-note/source"/>
                    </fo:inline>
                </fo:block>
            </xsl:if>
            <!-- Eurovoc 
                <xsl:apply-templates select="eurovoc"/>-->

        </fo:block>

    </xsl:template>



    <!-- Source template -->
    <xsl:template match="source">
        <xsl:if test="string(text())"> [ <xsl:if test="@sourceType='Url'"> Url: <fo:basic-link>
                    <xsl:if test="substring-after(text(),'http://')!=''">
                        <xsl:attribute name="external-destination">
                            <xsl:copy-of select="text()"/>
                        </xsl:attribute>
                        <xsl:if test="substring-before(substring-after(text(),'http://'), '/')!=''">
                            <xsl:value-of
                                select="substring-before(substring-after(text(),'http://'), '/')"/>
                        </xsl:if>
                        <xsl:if test="substring-before(substring-after(text(),'http://'), '/')=''">
                            <xsl:value-of select="substring-after(text(),'http://')"/>
                        </xsl:if>
                    </xsl:if>
                    <xsl:if test="substring-after(text(),'ftp://')!=''">
                        <xsl:attribute name="external-destination">
                            <xsl:copy-of select="text()"/>
                        </xsl:attribute>
                        <xsl:if test="substring-before(substring-after(text(),'ftp://'), '/')!=''">
                            <xsl:value-of
                                select="substring-before(substring-after(text(),'ftp://'), '/')"/>
                        </xsl:if>
                        <xsl:if test="substring-before(substring-after(text(),'ftp://'), '/')=''">
                            <xsl:value-of select="substring-after(text(),'ftp://')"/>
                        </xsl:if>
                    </xsl:if>
                    <xsl:if
                        test="substring-after(text(),'http://')='' and substring-after(text(),'ftp://')=''">
                        <xsl:attribute name="external-destination">http://<xsl:copy-of
                                select="text()"/></xsl:attribute>
                        <xsl:if test="substring-before(text(), '/')!=''">
                            <xsl:value-of select="substring-before(text(), '/')"/>
                        </xsl:if>
                        <xsl:if test="substring-before(text(), '/')=''">
                            <xsl:value-of select="text()"/>
                        </xsl:if>
                    </xsl:if>
                </fo:basic-link>
                <xsl:if test="@date!=''"> (<xsl:value-of select="@date"/>) </xsl:if>
            </xsl:if>
            <xsl:if test="@sourceType='Document'"> Document: <xsl:copy-of select="text()"/>
                <xsl:if test="@date!=''"> (<xsl:value-of select="@date"/>) </xsl:if>
            </xsl:if>
            <xsl:if test="@sourceType='Author'"> Author: <xsl:copy-of select="text()"/>
                <xsl:if test="@date!=''"> (<xsl:value-of select="@date"/>) </xsl:if>
            </xsl:if> ] </xsl:if>
    </xsl:template>

    <!-- Domain template -->
    <xsl:template match="domain">
        <xsl:if test="text()!=''">
            <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                <fo:inline font-variant="small-caps">Domain: </fo:inline>
                <xsl:choose>
                    <xsl:when test="../@lang='deu'">
                        <xsl:choose>
                            <xsl:when test="text()='1.1'">1.1 Naturschutz</xsl:when>
                            <xsl:when test="text()='1.2'">1.2 Landschaftspflege</xsl:when>
                            <xsl:when test="text()='1.3'">1.3 Schutzgebiete</xsl:when>
                            <xsl:when test="text()='1.4'">1.4 Schutz der Fauna und der Flora</xsl:when>
                            <xsl:when test="text()='1.5.0'">1.5.0 Umweltpolitik</xsl:when>
                            <xsl:when test="text()='1.5.1'">1.5.1 Bodenschutz</xsl:when>
                            <xsl:when test="text()='1.5.2'">1.5.2 Abfälle</xsl:when>
                            <xsl:when test="text()='1.5.3'">1.5.3 Umweltverschmutzung</xsl:when>
                            <xsl:when test="text()='1.6'">1.6 Wasserschutz</xsl:when>
                            <xsl:when test="text()='1.7'">1.7 Umweltverträglichkeitsprüfung</xsl:when>
                            <xsl:when test="text()='1.8'">1.8 Naturkatastrophen</xsl:when>
                            <xsl:when test="text()='2.0'">2.0 Verkehrspolitik</xsl:when>
                            <xsl:when test="text()='2.1'">2.1 Eisenbahnverkehr</xsl:when>
                            <xsl:when test="text()='2.2'">2.2 Straßenverkehr</xsl:when>
                            <xsl:when test="text()='2.3'">2.3 Luftverkehr</xsl:when>
                            <xsl:when test="text()='2.4'">2.4 Seeverkehr</xsl:when>
                            <xsl:when test="text()='2.5'">2.5 Binnenschifffahrt</xsl:when>
                            <xsl:when test="text()='2.6'">2.6 Kombinierter Verkehr</xsl:when>
                            <xsl:when test="text()='2.7'">2.7 Personenverkehr</xsl:when>
                            <xsl:when test="text()='2.8'">2.8 Gütertransport</xsl:when>
                            <xsl:when test="text()='2.9'">2.9 Transportnetzwerke</xsl:when>
                            <xsl:when test="text()='2.10'">2.10 Verkehrssicherheit</xsl:when>
                            <xsl:when test="text()='2.11'">2.11 Beförderungsvertrag</xsl:when>
                            <xsl:when test="text()='3.1'">3.1 Regionalpolitik</xsl:when>
                            <xsl:when test="text()='3.2'">3.2 Industrie</xsl:when>
                            <xsl:when test="text()='3.3'">3.3 Handel</xsl:when>
                            <xsl:when test="text()='3.4'">3.4 Handwerk</xsl:when>
                            <xsl:when test="text()='3.5'">3.5 Tourismus</xsl:when>
                            <xsl:when test="text()='3.6'">3.6 Arbeit</xsl:when>
                            <xsl:when test="text()='3.7'">3.7 Genossenschaften</xsl:when>
                            <xsl:when test="text()='3.8'">3.8 Energie</xsl:when>
                            <xsl:when test="text()='4.1'">4.1 Landwirtschaft</xsl:when>
                            <xsl:when test="text()='4.2'">4.2 Wälder</xsl:when>
                            <xsl:when test="text()='4.3'">4.3 Viehhaltung</xsl:when>
                            <xsl:when test="text()='4.4'">4.4 Jagd, Fischerei und Fischzucht</xsl:when>
                            <xsl:when test="text()='4.5'">4.5 Naturgefahren</xsl:when>
                            <xsl:when test="text()='5.1'">5.1 Städteplanung</xsl:when>
                            <xsl:when test="text()='5.2'">5.2 Stadtgebiete</xsl:when>
                            <xsl:when test="text()='5.3'">5.3 Bauwesen</xsl:when>
                            <xsl:when test="text()='6'">6 Allgemeine Termini</xsl:when>
                            <xsl:otherwise>domain error</xsl:otherwise>
                        </xsl:choose>
                    </xsl:when>
                    <xsl:when test="../@lang='fra'">
                        <xsl:choose>
                            <xsl:when test="text()='1.1'">1.1 Protection de la nature</xsl:when>
                            <xsl:when test="text()='1.2'">1.2 Entretien des paysages</xsl:when>
                            <xsl:when test="text()='1.3'">1.3 Espaces protégés</xsl:when>
                            <xsl:when test="text()='1.4'">1.4 Protection de la faune et de la flore</xsl:when>
                            <xsl:when test="text()='1.5.0'">1.5.0 Politique environnementale</xsl:when>
                            <xsl:when test="text()='1.5.1'">1.5.1 Protection des sols</xsl:when>
                            <xsl:when test="text()='1.5.2'">1.5.2 Déchets</xsl:when>
                            <xsl:when test="text()='1.5.3'">1.5.3 Pollution</xsl:when>
                            <xsl:when test="text()='1.6'">1.6 Protection de l’eau</xsl:when>
                            <xsl:when test="text()='1.7'">1.7 Etudes d’impact sur l’environnement</xsl:when>
                            <xsl:when test="text()='1.8'">1.8 Catastrophes naturelles</xsl:when>
                            <xsl:when test="text()='2.0'">2.0 Politique des transports</xsl:when>
                            <xsl:when test="text()='2.1'">2.1 Transport ferroviaire</xsl:when>
                            <xsl:when test="text()='2.2'">2.2 Transport routier</xsl:when>
                            <xsl:when test="text()='2.3'">2.3 Transport aérien</xsl:when>
                            <xsl:when test="text()='2.4'">2.4 Transport maritime</xsl:when>
                            <xsl:when test="text()='2.5'">2.5 Navigation intérieure</xsl:when>
                            <xsl:when test="text()='2.6'">2.6 Transport combiné</xsl:when>
                            <xsl:when test="text()='2.7'">2.7 Transport de personnes</xsl:when>
                            <xsl:when test="text()='2.8'">2.8 Transport de marchandises</xsl:when>
                            <xsl:when test="text()='2.9'">2.9 Réseaux de transport</xsl:when>
                            <xsl:when test="text()='2.10'">2.10 Sécurité des transports</xsl:when>
                            <xsl:when test="text()='2.11'">2.11 Contrat de transport</xsl:when>
                            <xsl:when test="text()='3.1'">3.1 Politique régionale</xsl:when>
                            <xsl:when test="text()='3.2'">3.2 Industrie</xsl:when>
                            <xsl:when test="text()='3.3'">3.3 Commerce</xsl:when>
                            <xsl:when test="text()='3.4'">3.4 Artisanat</xsl:when>
                            <xsl:when test="text()='3.5'">3.5 Tourisme</xsl:when>
                            <xsl:when test="text()='3.6'">3.6 Emploi</xsl:when>
                            <xsl:when test="text()='3.7'">3.7 Sociétés coopératives</xsl:when>
                            <xsl:when test="text()='3.8'">3.8 Energie</xsl:when>
                            <xsl:when test="text()='4.1'">4.1 Agriculture</xsl:when>
                            <xsl:when test="text()='4.2'">4.2 Forêts</xsl:when>
                            <xsl:when test="text()='4.3'">4.3 Zootechnie</xsl:when>
                            <xsl:when test="text()='4.4'">4.4 Chasse, pêche et élevage aquacole</xsl:when>
                            <xsl:when test="text()='4.5'">4.5 Risques naturels</xsl:when>
                            <xsl:when test="text()='5.1'">5.1 Urbanisme</xsl:when>
                            <xsl:when test="text()='5.2'">5.2 Zones urbaines</xsl:when>
                            <xsl:when test="text()='5.3'">5.3 Bâtiment</xsl:when>
                            <xsl:when test="text()='6'">6 Termes généraux</xsl:when>
                            <xsl:otherwise>domain error</xsl:otherwise>
                        </xsl:choose>
                    </xsl:when>
                    <xsl:when test="../@lang='ita'">
                        <xsl:choose>
                            <xsl:when test="text()='1.1'">1.1 Protezione della natura</xsl:when>
                            <xsl:when test="text()='1.2'">1.2 Tutela del paesaggio</xsl:when>
                            <xsl:when test="text()='1.3'">1.3 Aree protette</xsl:when>
                            <xsl:when test="text()='1.4'">1.4 Tutela della flora e della fauna</xsl:when>
                            <xsl:when test="text()='1.5.0'">1.5.0 Politica ambientale</xsl:when>
                            <xsl:when test="text()='1.5.1'">1.5.1 Difesa del suolo</xsl:when>
                            <xsl:when test="text()='1.5.2'">1.5.2 Rifiuti</xsl:when>
                            <xsl:when test="text()='1.5.3'">1.5.3 Inquinamento</xsl:when>
                            <xsl:when test="text()='1.6'">1.6 Tutela dell’acqua</xsl:when>
                            <xsl:when test="text()='1.7'">1.7 Valutazione di impatto ambientale</xsl:when>
                            <xsl:when test="text()='1.8'">1.8 Calamità naturali</xsl:when>
                            <xsl:when test="text()='2.0'">2.0 Politica dei trasporti</xsl:when>
                            <xsl:when test="text()='2.1'">2.1 Trasporto ferroviario</xsl:when>
                            <xsl:when test="text()='2.2'">2.2 Trasporto stradale</xsl:when>
                            <xsl:when test="text()='2.3'">2.3 Trasporto aereo</xsl:when>
                            <xsl:when test="text()='2.4'">2.4 Trasporto marittimo</xsl:when>
                            <xsl:when test="text()='2.5'">2.5 Navigazione interna</xsl:when>
                            <xsl:when test="text()='2.6'">2.6 Trasporto combinato</xsl:when>
                            <xsl:when test="text()='2.7'">2.7 Trasporto di persone</xsl:when>
                            <xsl:when test="text()='2.8'">2.8 Trasporto di merci</xsl:when>
                            <xsl:when test="text()='2.9'">2.9 Reti di trasporto</xsl:when>
                            <xsl:when test="text()='2.10'">2.10 Sicurezza nei trasporti</xsl:when>
                            <xsl:when test="text()='2.11'">2.11 Contratto di trasporto</xsl:when>
                            <xsl:when test="text()='3.1'">3.1 Politica regionale</xsl:when>
                            <xsl:when test="text()='3.2'">3.2 Industria</xsl:when>
                            <xsl:when test="text()='3.3'">3.3 Commercio</xsl:when>
                            <xsl:when test="text()='3.4'">3.4 Artigianato</xsl:when>
                            <xsl:when test="text()='3.5'">3.5 Turismo</xsl:when>
                            <xsl:when test="text()='3.6'">3.6 Lavoro</xsl:when>
                            <xsl:when test="text()='3.7'">3.7 Cooperazione</xsl:when>
                            <xsl:when test="text()='3.8'">3.8 Energia</xsl:when>
                            <xsl:when test="text()='4.1'">4.1 Agricoltura</xsl:when>
                            <xsl:when test="text()='4.2'">4.2 Foreste</xsl:when>
                            <xsl:when test="text()='4.3'">4.3 Zootecnia</xsl:when>
                            <xsl:when test="text()='4.4'">4.4 Caccia, pesca e acquacoltura</xsl:when>
                            <xsl:when test="text()='4.5'">4.5 Rischi naturali</xsl:when>
                            <xsl:when test="text()='5.1'">5.1 Pianificazione urbanistica</xsl:when>
                            <xsl:when test="text()='5.2'">5.2 Aree urbanistiche</xsl:when>
                            <xsl:when test="text()='5.3'">5.3 Edilizia</xsl:when>
                            <xsl:when test="text()='6'">6 Termini generici</xsl:when>
                            <xsl:otherwise>domain error</xsl:otherwise>
                        </xsl:choose>
                    </xsl:when>
                    <xsl:when test="../@lang='slv'">
                        <xsl:choose>
                            <xsl:when test="text()='1.1'">1.1 Varstvo narave</xsl:when>
                            <xsl:when test="text()='1.2'">1.2 Ohranjanje krajine</xsl:when>
                            <xsl:when test="text()='1.3'">1.3 Zavarovana območja</xsl:when>
                            <xsl:when test="text()='1.4'">1.4 Zavarovane rastlinske in živalske
                                vrste</xsl:when>
                            <xsl:when test="text()='1.5.0'">1.5.0 Okoljska politika</xsl:when>
                            <xsl:when test="text()='1.5.1'">1.5.1 Varstvo tal</xsl:when>
                            <xsl:when test="text()='1.5.2'">1.5.2 Odpadki</xsl:when>
                            <xsl:when test="text()='1.5.3'">1.5.3 Onesnaževanje</xsl:when>
                            <xsl:when test="text()='1.6'">1.6 Varstvo voda</xsl:when>
                            <xsl:when test="text()='1.7'">1.7 Presoje vplivov na okolje</xsl:when>
                            <xsl:when test="text()='1.8'">1.8 Elementarne nesreče</xsl:when>
                            <xsl:when test="text()='2.0'">2.0 Prometna politika</xsl:when>
                            <xsl:when test="text()='2.1'">2.1 Železniški promet</xsl:when>
                            <xsl:when test="text()='2.2'">2.2 Cestni promet</xsl:when>
                            <xsl:when test="text()='2.3'">2.3 Letalski promet</xsl:when>
                            <xsl:when test="text()='2.4'">2.4 Morski promet</xsl:when>
                            <xsl:when test="text()='2.5'">2.5 Celinski rečni promet</xsl:when>
                            <xsl:when test="text()='2.6'">2.6 Kombiniran promet</xsl:when>
                            <xsl:when test="text()='2.7'">2.7 Prevoz potnikov</xsl:when>
                            <xsl:when test="text()='2.8'">2.8 Prevoz blaga</xsl:when>
                            <xsl:when test="text()='2.9'">2.9 Prometno omrežje</xsl:when>
                            <xsl:when test="text()='2.10'">2.10 Prometna varnost</xsl:when>
                            <xsl:when test="text()='2.11'">2.11 Prevozne pogodbe</xsl:when>
                            <xsl:when test="text()='3.1'">3.1 Regionalna politika</xsl:when>
                            <xsl:when test="text()='3.2'">3.2 Industrija</xsl:when>
                            <xsl:when test="text()='3.3'">3.3 Trgovina</xsl:when>
                            <xsl:when test="text()='3.4'">3.4 Obrt</xsl:when>
                            <xsl:when test="text()='3.5'">3.5 Turizem</xsl:when>
                            <xsl:when test="text()='3.6'">3.6 Zaposlovanje</xsl:when>
                            <xsl:when test="text()='3.7'">3.7 Sodelovanje (=zadruge)</xsl:when>
                            <xsl:when test="text()='3.8'">3.8 Energetika</xsl:when>
                            <xsl:when test="text()='4.1'">4.1 Kmetijstvo</xsl:when>
                            <xsl:when test="text()='4.2'">4.2 Gozdovi</xsl:when>
                            <xsl:when test="text()='4.3'">4.3 Zootehnika</xsl:when>
                            <xsl:when test="text()='4.4'">4.4 Lov, ribištvo in gojenje rib</xsl:when>
                            <xsl:when test="text()='4.5'">4.5 Preprečevanje naravnih nesreč</xsl:when>
                            <xsl:when test="text()='5.1'">5.1 Načrtovanje mest</xsl:when>
                            <xsl:when test="text()='5.2'">5.2 Poselitveni prostor</xsl:when>
                            <xsl:when test="text()='5.3'">5.3 Gradnja</xsl:when>
                            <xsl:when test="text()='6'">6 Splošni termini</xsl:when>
                            <xsl:otherwise>domain error</xsl:otherwise>
                        </xsl:choose>
                    </xsl:when>
                    <xsl:when test="../@lang='eng'">
                        <xsl:choose>
                            <xsl:when test="text()='1.1'">1.1 Conservation of nature</xsl:when>
                            <xsl:when test="text()='1.2'">1.2 Landscape protection</xsl:when>
                            <xsl:when test="text()='1.3'">1.3 Protected areas</xsl:when>
                            <xsl:when test="text()='1.4'">1.4 Protection of fauna and flora</xsl:when>
                            <xsl:when test="text()='1.5.0'">1.5.0 Environmental policy</xsl:when>
                            <xsl:when test="text()='1.5.1'">1.5.1 Soil protection</xsl:when>
                            <xsl:when test="text()='1.5.2'">1.5.2 Waste</xsl:when>
                            <xsl:when test="text()='1.5.3'">1.5.3 Pollution</xsl:when>
                            <xsl:when test="text()='1.6'">1.6 Water protection</xsl:when>
                            <xsl:when test="text()='1.7'">1.7 Environmental impact assessment</xsl:when>
                            <xsl:when test="text()='1.8'">1.8 Natural disasters</xsl:when>
                            <xsl:when test="text()='2.0'">2.0 Transport policy</xsl:when>
                            <xsl:when test="text()='2.1'">2.1 Rail transport</xsl:when>
                            <xsl:when test="text()='2.2'">2.2 Road transport</xsl:when>
                            <xsl:when test="text()='2.3'">2.3 Air transport</xsl:when>
                            <xsl:when test="text()='2.4'">2.4 Sea transport</xsl:when>
                            <xsl:when test="text()='2.5'">2.5 Inland navigation</xsl:when>
                            <xsl:when test="text()='2.6'">2.6 Combined transport</xsl:when>
                            <xsl:when test="text()='2.7'">2.7 Passenger transport</xsl:when>
                            <xsl:when test="text()='2.8'">2.8 Transport of goods</xsl:when>
                            <xsl:when test="text()='2.9'">2.9 Transport networks</xsl:when>
                            <xsl:when test="text()='2.10'">2.10 Transport safety</xsl:when>
                            <xsl:when test="text()='2.11'">2.11 Contract of carriage</xsl:when>
                            <xsl:when test="text()='3.1'">3.1 Regional policy</xsl:when>
                            <xsl:when test="text()='3.2'">3.2 Industry</xsl:when>
                            <xsl:when test="text()='3.3'">3.3 Trade</xsl:when>
                            <xsl:when test="text()='3.4'">3.4 Handicraft</xsl:when>
                            <xsl:when test="text()='3.5'">3.5 Tourism</xsl:when>
                            <xsl:when test="text()='3.6'">3.6 Employment</xsl:when>
                            <xsl:when test="text()='3.7'">3.7 Co-operation</xsl:when>
                            <xsl:when test="text()='3.8'">3.8 Energy</xsl:when>
                            <xsl:when test="text()='4.1'">4.1 Agriculture</xsl:when>
                            <xsl:when test="text()='4.2'">4.2 Forests</xsl:when>
                            <xsl:when test="text()='4.3'">4.3 Zootechnics</xsl:when>
                            <xsl:when test="text()='4.4'">4.4 Hunting, fisheries and fish farming</xsl:when>
                            <xsl:when test="text()='4.5'">4.5 Natural hazards</xsl:when>
                            <xsl:when test="text()='5.1'">5.1 Town planning</xsl:when>
                            <xsl:when test="text()='5.2'">5.2 Urban areas</xsl:when>
                            <xsl:when test="text()='5.3'">5.3 Building</xsl:when>
                            <xsl:when test="text()='6'">6 General terms</xsl:when>
                            <xsl:otherwise>domain error</xsl:otherwise>
                        </xsl:choose>
                    </xsl:when>
                    <xsl:otherwise>domain lang error</xsl:otherwise>
                </xsl:choose>
            </fo:block>
        </xsl:if>
    </xsl:template>


    <!-- Related term template -->
    <xsl:template match="relatedTerm">
        <xsl:if test="@termref!='' and text()!=''">
            <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                <fo:inline font-variant="small-caps"><xsl:value-of select="@relationToTerm"/>: </fo:inline>
                <fo:inline>
                    <fo:basic-link>
                        <xsl:attribute name="internal-destination">
                            <xsl:value-of select="@termref"/>
                        </xsl:attribute>
                        <xsl:value-of select="text()"/> (see page <fo:page-number-citation>
                            <xsl:attribute name="ref-id">
                                <xsl:value-of select="@termref"/>
                            </xsl:attribute>
                        </fo:page-number-citation>) </fo:basic-link>
                </fo:inline>
            </fo:block>
        </xsl:if>
        <xsl:if test="@termref='' and  text()!=''">
            <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                <fo:inline font-variant="small-caps"><xsl:value-of select="@relationToTerm"/>: </fo:inline>
                <xsl:value-of select="text()"/>
            </fo:block>
        </xsl:if>
    </xsl:template>

    <!-- Related form template -->
    <xsl:template match="relatedForm">
        <xsl:if test="string(text())">
            <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                <fo:inline font-variant="small-caps"><xsl:value-of select="@relationToForm"/>: </fo:inline>
                <xsl:value-of select="text()"/>
            </fo:block>
        </xsl:if>
    </xsl:template>

    <!-- Eurovoc template -->
    <xsl:template match="eurovoc">
        <xsl:if test="@id!=''">
            <fo:block text-align="justify" margin-left="1em" space-before="1pt">
                <fo:inline font-variant="small-caps">Eurovoc: </fo:inline> thesaurus ID = <fo:inline
                    font-style="italic">
                    <xsl:value-of select="@id"/>
                </fo:inline>
                <xsl:if test="string(text())"> , descriptor ID = <fo:inline font-style="italic">
                        <xsl:copy-of select="text()"/>
                    </fo:inline>
                </xsl:if>
                <xsl:value-of select="text()"/>
            </fo:block>
        </xsl:if>
    </xsl:template>

</xsl:stylesheet>
