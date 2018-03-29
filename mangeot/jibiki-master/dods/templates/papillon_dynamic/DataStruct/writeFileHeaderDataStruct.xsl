<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:dodst="org.enhydra.dods.xslt.DODSTag"
                     xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
<xsl:template name="fileHeader">
/*-----------------------------------------------------------------------------
 * Enhydra Java Application Server
 * Copyright 1997-2000 Lutris Technologies, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes Enhydra software developed by Lutris
 *    Technologies, Inc. and its contributors.
 * 4. Neither the name of Lutris Technologies nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LUTRIS TECHNOLOGIES AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL LUTRIS TECHNOLOGIES OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *-----------------------------------------------------------------------------
 * <xsl:value-of select="PACKAGE"/>/<xsl:value-of select="CLASS_NAME"/>DataStruct.java
 *-----------------------------------------------------------------------------
 */

<xsl:value-of select="dodst:resetCounter()"/>
package <xsl:value-of select="translate(PACKAGE,'/','.')"/>;

import com.lutris.appserver.server.sql.*;
import java.sql.*;
import java.math.*;
import java.io.Serializable;
import com.lutris.dods.builder.generator.dataobject.*;
import com.lutris.dods.builder.generator.query.*;
import org.enhydra.dods.cache.Condition;


/**
 * Data structure for DO class.
 * A container for data members of a DO class.
 * A DO class contains a reference to a DataStruct class.  This reference
 * can be null (a DO whose data has not yet been retrieved from the database),
 * allowing a DO object to be a lightweight placeholder until its data is needed.
 *
 * @version $Revision$
 * @author  <xsl:value-of select="AUTHOR"/>
 * @since   <xsl:value-of select="PROJECT_NAME"/>;
 */
public class <xsl:value-of select="CLASS_NAME"/>DataStruct extends CoreDataStruct implements Cloneable, Serializable {

    /**
     * A DO refers to this DataStruct.
     * readOnly is set to true when the DO is stored in its class cache.
     */
    public boolean readOnly = false;

    /**
     * Since originalData is being constructed for every DO, this flag
     * "knows" if DataStruct has any useful content.
     */
    protected boolean isEmpty = true;

    /**
     * String identifying logical database this DataStruct belongs to.
     */
    private String databaseName = null;

    private byte[] copyByteArray( byte[] source ) {
        byte[] dest = new byte[ source.length ];
        System.arraycopy( source, 0, dest, 0, source.length );
        return dest;
    }

    /**
     * Returns DataStruct's version.    
     * @return DataStruct's version.    
     */
    protected int getVersion () {
        return get_Version();
    }

    /**
     * Returns DataStruct's version.    
     * @return DataStruct's version.    
     */
    protected int get_Version () {
        return super.get_Version();
    }

    /**
     * Sets DataStruct's version.
     * @param v new DataStruct's version.
     */
    protected void setVersion (int v) {
        set_Version(v);
    }

    /**
     * Sets DataStruct's version.
     * @param v new DataStruct's version.
     */
    protected void set_Version (int v) {
        super.set_Version(v);
    }

    /**
     * Returns this object's identifier.
     * @return this object's identifier.
     */
    public ObjectId getOId() {
	   return get_OId();
    }

    /**
     * Returns this object's identifier.
     * @return this object's identifier.
     */
    public ObjectId get_OId() {
	return super.get_OId();
    }

    /**
     * @deprecated Use set_OId()
     * @param oId this object's identifier.
     */
    protected void setOId(ObjectId oId) {
    	set_OId(oId);
    }

    /**
     * Sets this object's identifier.
     * @param oId this object's identifier.
     */
    protected void set_OId(ObjectId oId) {
    	super.set_OId(oId);
        __the_handle = oId.toString();
    }
    private String __the_handle;
    
<xsl:if test="DO_IS_OID_BASED='true'">
    /**
     * Returns this object's handle (identifier as a string).
     * @return This object's identifier as a string.
     *
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public String  get_Handle()
    throws DatabaseManagerException {
        String ret = null;
        if (null == __the_handle)
               throw new DatabaseManagerException( "ID not set " );
        //ret = get_OId().toString();
        return __the_handle;
    }


    /**
     * Returns this object's cache handle (String in the form:
     * "&lt;database_name&gt;.&lt;indetifier_as_String&gt;").
     *
     * @return cache handle.
     * @exception DatabaseManagerException
     *   If a connection to the database cannot be established, etc.
     */
    public String  get_CacheHandle()throws DatabaseManagerException {
        String ret = get_Database() + "." + get_Handle();
        return ret;
    }
</xsl:if>


    /**
     * @param dbName - name of the logical database this DataStruct should belong to.
     */
    public void set_Database(String dbName) {
         if (null != databaseName)
             throw new Error("Whatta hack you are doing! Multiple db setting not allowed.");
         databaseName = dbName;
    }


    /**
     * @return name of the logical database this DataStruct belongs to.
     */
    public String get_Database() {
         return databaseName;
    }

    public CoreDataStruct dumpData(boolean incrementVersion) {
        <xsl:value-of select="CLASS_NAME"/>DataStruct ret =
            new <xsl:value-of select="CLASS_NAME"/>DataStruct();
        ret.set_OId(get_OId());
        ret.set_Version((incrementVersion?1:0)+get_Version());
        ret.set_Database(get_Database());
        return ret;
    }
</xsl:template>
</xsl:stylesheet>
