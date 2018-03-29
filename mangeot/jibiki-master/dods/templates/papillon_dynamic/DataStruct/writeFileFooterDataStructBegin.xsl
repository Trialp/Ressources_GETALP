<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
	<xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="fileFooterBegin">
    /**
     * Create a copy of the guts of a DO.
     *
     * @return Copied DataStruct object.
     *
     * @exception DatabaseManagerException 
     *       if createExisting() fails for a contained DO
     * @exception ObjectIdException 
     *       if GenericDO has trouble obtaining a valid id.
     */
    public <xsl:value-of select="CLASS_NAME"/>DataStruct duplicate() 
    throws DatabaseManagerException, ObjectIdException {
        <xsl:value-of select="CLASS_NAME"/>DataStruct ret = new <xsl:value-of select="CLASS_NAME"/>DataStruct ();
        if (!isEmpty) {
</xsl:template>
</xsl:stylesheet>
