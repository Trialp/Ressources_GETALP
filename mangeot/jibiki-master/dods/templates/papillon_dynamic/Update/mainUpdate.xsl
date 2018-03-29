<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                    xmlns:gvs="org.ejen.ext.GlobalVariables"
                    extension-element-prefixes="gvs"
                    exclude-result-prefixes="gvs"
                    xmlns:xalan="http://xml.apache.org/xslt"
                    xmlns:stu="org.ejen.ext.StringUtil"
                    xmlns:miu="org.enhydra.dods.xslt.XSLTUtil"
                    xmlns:dtag="org.enhydra.dods.xslt.DODSTag"
                    xmlns:common="org.enhydra.dods.Common">

    <xsl:output method="text" encoding="iso-8859-1"/>

<xsl:template name="begining">/*-----------------------------------------------------------------------------
 * Enhydra Java Application Server
 * Copyright (c) 2003 Together Teamsolutions.
 * All rights reserved.
 *
 *-----------------------------------------------------------------------------
 * <xsl:value-of select="PACKAGE"/>.<xsl:value-of select="CLASS_NAME"/>Update
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
import org.enhydra.dods.DODS;
import org.enhydra.dods.cache.*;
import com.lutris.logging.*;
import org.enhydra.dods.exceptions.AssertionDataObjectException;

/**
 *  *
 * @version $Revision$
 * @author  <xsl:value-of select="AUTHOR"/>
 * @since   <xsl:value-of select="PROJECT_NAME"/>
 */
public class <xsl:value-of select="CLASS_NAME"/>Update extends CoreDO {

    private CachedDBTransaction myDBt = null;
    private Vector vecOIdsUpdated = null;
    private <xsl:value-of select="CLASS_NAME"/>Query myQuery;
    private QueryBuilder myQB;
    private ObjectId OId;
    private boolean dirty = false;
    private boolean doSelectOIds;

    /**
     * @param query - <xsl:value-of select="CLASS_NAME"/>Query
     */
    public <xsl:value-of select="CLASS_NAME"/>Update(<xsl:value-of select="CLASS_NAME"/>Query query)
           throws ObjectIdException, DatabaseManagerException {
        OId = DODS.getDatabaseManager().allocateObjectId();
        myQuery = query;
        myDBt =(CachedDBTransaction)query.transaction;
        myQB = query.getQueryBuilder();
        doSelectOIds = true;
        setPersistent(true);
        DODS.getLogChannel().write(Logger.DEBUG, "<xsl:value-of select="CLASS_NAME"/>Update constructed");
    }

    public void save(DBTransaction dbt)
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException {
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

            dbtlocal.update(this);
            if (needToCommit) {
                dbtlocal.commit();
                dbtlocal.release();
            }
        } catch (SQLException e) {
            StringBuffer message = new StringBuffer("Failed to update : ");
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

    public void save()
    throws SQLException, DatabaseManagerException, DataObjectException, RefAssertionException, DBRowUpdateException, QueryException {
        save(myDBt);
    }

    /**
     * overloaded method from CoreDO
     * @param conn - DBConnection on which to execute update
     */
    public synchronized void executeUpdate(DBConnection conn)
            throws SQLException, DBRowUpdateException {
        /*
         * optional OId gathering depends on parameter
         * and existence either of caches
         */
        DODS.getLogChannel().write(Logger.DEBUG, "executing <xsl:value-of select="CLASS_NAME"/>Update");
        if (beMorePrecise()&amp;&amp;(hasTransactionCache()||hasGlobalCache())) {
            // optional "select oid" + all clauses from query builder
            // which makes vector of oids to update directly in cache
            vecOIdsUpdated = myQuery.collectOIds();
        }
        updateQueryBuilder();
        myQB.executeUpdate(conn);

        if (hasTransactionCache()) {
            // empty all DO objects from xxx table in transaction cache
            //  - crate new empty DataStruct discarding both old ones
            // (or from list acquired in executeUpdate)
            if (beMorePrecise()) {
                ((TransactionQueryCache)myDBt.getTransactionCache()).emptyEntries(vecOIdsUpdated, isIncrementVersions());
            } else if (isIncrementVersions()) {
                ((TransactionQueryCache)myDBt.getTransactionCache()).removeEntries(<xsl:value-of select="CLASS_NAME"/>DO.class);
            } else {
                ((TransactionQueryCache)myDBt.getTransactionCache()).emptyEntries(<xsl:value-of select="CLASS_NAME"/>DO.class);
            }
        }
    }

    /**
     * overloaded method from CoreDO
     * @param success - true if commit succeeded, false otherwise
     */
    public void finalizeUpdate(boolean success) {
        if (success) {
            try {
                if (beMorePrecise()) {
                    ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache()).emptyEntries(vecOIdsUpdated, isIncrementVersions());
                } else if (isIncrementVersions()) {
                    ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache()).removeEntries();
                } else {
                    ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache()).emptyEntries();
                }
            } catch(Exception e) {}
        }
        /*
         * otherwise (transactions gets rolled back)
         * nothing needs to be done
         */
    }

    /**
     * @return ObjectId for this instance
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
     * @return value of AutoSave parameter for this table/logicalDB/dbManager
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
     * @return flag indicating should we do a select on DB before doing update
     */
    protected boolean beMorePrecise() {
        try {
            if (doSelectOIds)
                return <xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration().getTableConfiguration().getSelectOids();
        } catch (Exception ex) {}
        return false;
    }

    /**
     * @return flag indicating should we do a select on DB before doing update
     */
    private boolean isIncrementVersions() {
        try {
            return <xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration().getTableConfiguration().getIncrementVersions();
        } catch (Exception ex) {}
        return true;
    }

    public PreparedStatement getInsertStatement(DBConnection conn)
            throws SQLException {
        //throw new RuntimeException("NOT IMPLEMENTED");
        updateQueryBuilder();
        return myQB.getStatement(conn);
    }

    public PreparedStatement getUpdateStatement(DBConnection conn)
            throws SQLException {
        //throw new RuntimeException("NOT IMPLEMENTED");
        updateQueryBuilder();
        return myQB.getStatement(conn);
    }

    public PreparedStatement getDeleteStatement(DBConnection conn)
            throws SQLException {
        throw new RuntimeException("NOT IMPLEMENTED");
    }
    public void dumpData(boolean incrementVersion) {}
</xsl:template>

<xsl:template name="updateColumns">

    /*
     * we need here some default value different to null,
     * System.err is good as any
     */
    private Object prevValue<xsl:value-of select="common:capitalizeName(@name)"/> = System.err;
    private String prevExpr<xsl:value-of select="common:capitalizeName(@name)"/> = null;

    /**
     * @param x <xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> to update
     */
    public void set<xsl:value-of select="common:capitalizeName(@name)"/>(<xsl:value-of select="miu:adjustJavaType(JAVA_TYPE)"/> x)
            throws DataObjectException {

        if(myQB != null) {
            Object new<xsl:value-of select="common:capitalizeName(@name)"/> = <xsl:value-of select="dtag:convertJavaTypes(JAVA_TYPE)"/>;
            if ((null == prevValue<xsl:value-of select="common:capitalizeName(@name)"/>)
                ? (null != new<xsl:value-of select="common:capitalizeName(@name)"/>)
                : !prevValue<xsl:value-of select="common:capitalizeName(@name)"/>.equals(new<xsl:value-of select="common:capitalizeName(@name)"/>)) {
                prevValue<xsl:value-of select="common:capitalizeName(@name)"/> = new<xsl:value-of select="common:capitalizeName(@name)"/>;
                prevExpr<xsl:value-of select="common:capitalizeName(@name)"/> = null;
                if (myDBt != null &amp;&amp; isAutoSave()) {
                    try {
                        save(myDBt);
                    } catch (Exception ex) {
                        throw new DataObjectException
                            ("Error during transaction's writting data into database",ex);
                    }
                }
            }
        }
   }

    /**
     * @param x String to update
     */
    public void exprSet<xsl:value-of select="common:capitalizeName(@name)"/>(String x)
            throws DataObjectException {
        if (null == x)
            throw new NullPointerException("Parameter x cannot be null, please.");
        if (myQB != null) {
            if ((null == prevExpr<xsl:value-of select="common:capitalizeName(@name)"/>)
                ? (true)
                : !prevExpr<xsl:value-of select="common:capitalizeName(@name)"/>.equals(x)) {
                prevExpr<xsl:value-of select="common:capitalizeName(@name)"/> = x;
                prevValue<xsl:value-of select="common:capitalizeName(@name)"/> = System.err;
                if (myDBt != null &amp;&amp; isAutoSave()) {
                    try {
                        save(myDBt);
                    } catch (Exception ex) {
                        throw new DataObjectException
                            ("Error during transaction's writting data into database",ex);
                    }
                }
            }
        }
    }
</xsl:template>

<xsl:template name="updateQueryBuilderColumns">        if (null != prevExpr<xsl:value-of select="common:capitalizeName(@name)"/>) {
             myQB.addUpdateSQL(<xsl:value-of select="/TABLE/CLASS_NAME"/>DO.<xsl:value-of select="common:capitalizeName(@name)"/>.getColumnName(), prevExpr<xsl:value-of select="common:capitalizeName(@name)"/>);
        }
        if (System.err != prevValue<xsl:value-of select="common:capitalizeName(@name)"/>) {
             myQB.addUpdateColumn(<xsl:value-of select="/TABLE/CLASS_NAME"/>DO.<xsl:value-of select="common:capitalizeName(@name)"/>.getColumnName(), prevValue<xsl:value-of select="common:capitalizeName(@name)"/>);
        }
</xsl:template>

<xsl:template name="end">        if (isIncrementVersions()) {
            String verColName = <xsl:value-of select="CLASS_NAME"/>DO.get_versionColumnName();
            myQB.addUpdateSQL(verColName, "1 + "+ verColName);
        }
    }
}
</xsl:template>
<xsl:template name="updateQueryBuilder">
    private void updateQueryBuilder() {
</xsl:template>


<xsl:output method="text" encoding="iso-8859-1"/>

<xsl:template match ="/">
    <xsl:apply-templates />
</xsl:template>

<!-- doml -->
<xsl:template match="TABLE">
    <xsl:call-template name="begining"/>
    <xsl:for-each select="//COLUMN">
        <xsl:call-template name="updateColumns"/>
    </xsl:for-each>
    <xsl:call-template name="updateQueryBuilder"/>
    <xsl:for-each select="//COLUMN">
        <xsl:call-template name="updateQueryBuilderColumns"/>
    </xsl:for-each>
    <xsl:call-template name="end"/>
</xsl:template>

</xsl:stylesheet>
