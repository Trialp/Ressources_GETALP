<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                    xmlns:gvs="org.ejen.ext.GlobalVariables"
                    extension-element-prefixes="gvs"
                    exclude-result-prefixes="gvs"
                    xmlns:xalan="http://xml.apache.org/xslt"
                    xmlns:stu="org.ejen.ext.StringUtil"
                    xmlns:common="org.enhydra.dods.Common">

    <xsl:output method="text" encoding="iso-8859-1"/>

<xsl:template name="begining">/*-----------------------------------------------------------------------------
 * Enhydra Java Application Server
 * Copyright Together Teamsolutions.
 * All rights reserved.
 *
 *-----------------------------------------------------------------------------
 * <xsl:value-of select="PACKAGE"/>/<xsl:value-of select="CLASS_NAME"/>Delete.java
 *-----------------------------------------------------------------------------
 */

package <xsl:value-of select="translate(PACKAGE,'/','.')"/>;

import java.io.*;
import java.sql.*;
import java.math.*;
import java.util.*;
import com.lutris.appserver.server.sql.*;
import com.lutris.appserver.server.sql.standard.*;
import com.lutris.dods.builder.generator.query.*;
import com.lutris.logging.*;
import org.enhydra.dods.DODS;
import org.enhydra.dods.cache.*;
import org.enhydra.dods.exceptions.AssertionDataObjectException;

/**
 *  *
 * @version $Revision$
 * @author  <xsl:value-of select="AUTHOR"/>
 * @since   <xsl:value-of select="PROJECT_NAME"/>
 */
public class <xsl:value-of select="CLASS_NAME"/>Delete extends CoreDO {

    private CachedDBTransaction myDBt = null;
    private Vector vecOIdsUpdated = null;
    private <xsl:value-of select="CLASS_NAME"/>Query myQuery;
    private QueryBuilder myQB;
    private ObjectId OId;
    private boolean doSelectOIds;

    /**
     * @param query
     */
    public <xsl:value-of select="CLASS_NAME"/>Delete (<xsl:value-of select="CLASS_NAME"/>Query query)
           throws ObjectIdException, DataObjectException, DatabaseManagerException {
        OId = DODS.getDatabaseManager().allocateObjectId();
        myQuery = query;
        myDBt =(CachedDBTransaction)query.transaction;
        myQB = query.getQueryBuilder();
        doSelectOIds = true;
        setPersistent(true);
        if (myDBt != null &amp;&amp; isAutoSave()) {
            try {
                save(myDBt);
            } catch (Exception ex) {
                throw new DataObjectException
                    ("Error during transaction's writting data into database",ex);
            }
        }
    }

    /**
     * @param dbt
     */
    public void save(DBTransaction dbt) throws
    SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException {
        if (<xsl:value-of select="CLASS_NAME"/>DO.getCache().getTableConfiguration().isReadOnly()) {
            throw new AssertionDataObjectException(&quot;<xsl:value-of select="CLASS_NAME"/>DO's cache is read-only. Therefore, DML opertions are not allowed.&quot;);
        }
            DBTransaction dbtlocal = dbt;
            boolean needToCommit = false;

        try {

            if (dbtlocal == null) {
              if( myDBt==null) {
                dbtlocal = DODS.getDatabaseManager().createTransaction(myQuery.getLogicalDatabase());
                dbtlocal.setDatabaseName(myQuery.getLogicalDatabase());
                needToCommit = true;
              }
             else
                dbtlocal=myDBt;
            } else {<xsl:if test="DO_IS_MULTIDB_BASED='true'">
                if ( !dbtlocal.getDatabaseName().equals(myQuery.getLogicalDatabase()) ) {
                    throw new DatabaseManagerException("Logical database of transaction is different from DO's.");
                }
</xsl:if>
                if(myDBt!=null) {
                    if(!myDBt.equals(dbt))
                          throw new DatabaseManagerException("DO doesn't belong this transaction.");
                }
            }

            dbtlocal.delete(this);
            if (needToCommit) {
                dbtlocal.commit();
                dbtlocal.release();
            }
        } catch (SQLException e) {
            StringBuffer message = new StringBuffer("Failed to delete : ");
            message.append(e.getMessage());
            // rollback, if necessary
            if (needToCommit) {
                try {
                    dbtlocal.rollback();
                    dbtlocal.release();
                } catch (SQLException sqle2) {
                    message.insert(0,"\n");
                    message.insert(0,sqle2.getMessage());
                    message.insert(0,"Rollback failed: ");
                }
            }

            throw new SQLException(message.toString());
        }
    }

    /**
     *
     */
    public void save() throws
    SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException  {
        save(myDBt);
    }

    /**
     * overloaded method from CoreDO
     */
    public void executeDelete(DBConnection conn)
            throws SQLException {
        /*
         * optional OId gathering depends on parameter
         * and existence either of caches
         */
        DODS.getLogChannel().write(Logger.DEBUG, "executing <xsl:value-of select="CLASS_NAME"/>Delete");
        if (beMorePrecise()&amp;&amp;(hasTransactionCache()||hasGlobalCache())) {
            // optional "select oid" + all clauses from query builder
            // which makes vector of oids to update directly in cache
            vecOIdsUpdated = myQuery.collectOIds();
        }

        myQB.setDeleteQuery();
        myQB.executeUpdate(conn);

        if (hasTransactionCache()) {
            // empty all DO objects from xxx table in transaction cache
            //  - crate new empty DataStruct discarding both old ones
            // (or from list acquired in executeUpdate)
            if (beMorePrecise()) {
                ((TransactionQueryCache)myDBt.getTransactionCache()).removeEntries(vecOIdsUpdated);
            } else {
                ((TransactionQueryCache)myDBt.getTransactionCache()).removeEntries(<xsl:value-of select="CLASS_NAME"/>DO.class);
            }
        }
    }

    /**
     * overloaded method from CoreDO
     */
    public void finalizeDelete(boolean success) {
        if (success) {
            try {
                if (beMorePrecise()) {
                    ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache()).removeEntries(vecOIdsUpdated);
                } else {
                    ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache()).removeEntries();
                }
            } catch(Exception e) {}
        }
    }

    /**
     *
     */
    public ObjectId get_OId() {
        return OId;
    }

    private boolean hasTransactionCache() {
        return (null != myDBt)&amp;&amp;(null != myDBt.getTransactionCache());
    }

    private static boolean hasGlobalCache() {
        return !<xsl:value-of select="CLASS_NAME"/>DO.getCache().getCacheType().equals("none");
    }

    /**
     *
     */
    protected boolean isAutoSave() {
        boolean flag = false;
        try {
            String dbName =(myDBt!=null)?
                myDBt.getDatabaseName():
                <xsl:value-of select="CLASS_NAME"/>DO.get_logicalDBName();
            flag = ((StandardLogicalDatabase)(DODS.getDatabaseManager().findLogicalDatabase(dbName))).getDatabaseConfiguration().getAutoSave();
        } catch (Exception ex) {}
        return flag;
    }

    /**
     * @return true for ...
     */
    public boolean isSelectOIds() {
        return doSelectOIds;
    }

    /**
     * @param arg - without explanation, yet
     */
    public void setSelectOIds(boolean arg) {
        doSelectOIds = arg;
    }

    /**
     *
     */
    protected boolean beMorePrecise() {
        try {
            if (doSelectOIds)
                return <xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration().getTableConfiguration().getSelectOids();
        } catch (Exception ex) {}
        return false;
    }
    public void dumpData(boolean incrementVersion) {}
</xsl:template>


<xsl:template name="end">

    public PreparedStatement getInsertStatement(DBConnection conn)
            throws SQLException {
        throw new RuntimeException("NOT IMPLEMENTED");
    }

    public PreparedStatement getUpdateStatement(DBConnection conn)
            throws SQLException {
        throw new RuntimeException("NOT IMPLEMENTED");
    }

    public PreparedStatement getDeleteStatement(DBConnection conn)
            throws SQLException {
        //throw new RuntimeException("NOT IMPLEMENTED");
        myQB.setDeleteQuery();
        return myQB.getStatement(conn);
    }
}
</xsl:template>


<xsl:output method="text" encoding="iso-8859-1"/>

<xsl:template match ="/">
    <xsl:apply-templates />
</xsl:template>

<!-- doml -->
<xsl:template match="TABLE">
    <xsl:call-template name="begining"/>
    <xsl:call-template name="end"/>
</xsl:template>

</xsl:stylesheet>
