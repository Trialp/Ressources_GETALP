<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="endCompareCond">
                case COLUMN_OID:
                    return QueryBuilder.compare(get_CacheHandle(),cond.getValue(),cond.getOperator());
            }
        } catch (Exception e) {
          System.out.println("**************************  compareCond catck blok");
        }
        return false;
    }
</xsl:template>
</xsl:stylesheet>
