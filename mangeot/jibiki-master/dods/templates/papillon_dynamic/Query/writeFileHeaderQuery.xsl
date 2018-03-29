<?xml version="1.0"	encoding="iso-8859-1"?>
<xsl:stylesheet	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
		xmlns:xalan="http://xml.apache.org/xslt"
		xmlns:stu="org.ejen.ext.StringUtil"
		version="1.0">
<xsl:output	method="text" encoding="iso-8859-1"/>
<xsl:template name="fileHeader">
/*-----------------------------------------------------------------------------
 * Enhydra Java	Application	Server
 * Copyright 1997-2000 Lutris Technologies,	Inc.
 * All rights reserved.
 *
 * Redistribution and use in source	and	binary forms, with or without
 * modification, are permitted provided	that the following conditions
 * are met:
 * 1. Redistributions of source	code must retain the above copyright
 *	  notice, this list	of conditions and the following	disclaimer.
 * 2. Redistributions in binary	form must reproduce	the	above copyright
 *	  notice, this list	of conditions and the following	disclaimer in
 *	  the documentation	and/or other materials provided	with the distribution.
 * 3. All advertising materials	mentioning features	or use of this software
 *	  must display the following acknowledgement:
 *		This product includes Enhydra software developed by	Lutris
 *		Technologies, Inc. and its contributors.
 * 4. Neither the name of Lutris Technologies nor the names	of its contributors
 *	  may be used to endorse or	promote	products derived from this software
 *	  without specific prior written permission.
 *
 * THIS	SOFTWARE IS PROVIDED BY	LUTRIS TECHNOLOGIES AND	CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL LUTRIS TECHNOLOGIES OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,	SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;	OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED	AND ON ANY THEORY OF LIABILITY,	WHETHER	IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR	OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF	THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *-----------------------------------------------------------------------------
 * <xsl:value-of select="PACKAGE"/>/<xsl:value-of select="CLASS_NAME"/>Query.java
 *-----------------------------------------------------------------------------
 */

package	<xsl:value-of select="translate(PACKAGE,'/','.')"/>;

import org.enhydra.dods.DODS;
import com.lutris.dods.builder.generator.query.*;
import com.lutris.appserver.server.sql.*;
//import org.enhydra.dods.cache.LRUCache;
import org.enhydra.dods.cache.DataStructCache;
import org.enhydra.dods.cache.QueryCache;
import org.enhydra.dods.cache.QueryCacheItem;
import org.enhydra.dods.cache.QueryResult;
import org.enhydra.dods.cache.DataStructShell;
import org.enhydra.dods.cache.DOShell;
import org.enhydra.dods.cache.Condition;
import org.enhydra.dods.statistics.Statistics;
import org.enhydra.dods.cache.CacheConstants;
import org.enhydra.dods.statistics.*;
import com.lutris.logging.LogChannel;
import com.lutris.logging.Logger;
import com.lutris.appserver.server.sql.CachedDBTransaction;

//import org.enhydra.dods.util.LRUMap;
import java.sql.*;
import java.util.*;
import java.math.*;
import java.util.Date;	// when	I say Date,	I don't	mean java.sql.Date
<xsl:if	test="DO_IS_OID_BASED='true'">
// WebDocWf	extension for DODS row instance	based security
// The following line has been added
import org.webdocwf.dods.access.*;
// end of WebDocWf extension for DODS row instance based security
</xsl:if>
/**
 * <xsl:value-of select="CLASS_NAME"/>Query	is used	to query the <xsl:value-of select="TABLE_NAME"/> table in the database.&lt;BR&gt;
 * It returns objects of type <xsl:value-of	select="CLASS_NAME"/>DO.
 * &lt;P&gt;
 * General usage:
 * &lt;P&gt;
 *	   In DODS:
 *		  Create a Data	Object named "Dog",
 *		  and create an	Attribute named	"Name",
 *		  and set that Attribute to	"Can be	queried."
 *		  DODS will	then generate the method DogQuery.setQueryName().
 * &lt;P&gt;
 *	   In your Business	Layer, prepare the query:&lt;BR&gt;
 * &lt;P&gt;&lt;PRE&gt;
 *			   DogQuery	dq = new DogQuery();
 *			   dq.setQueryName("Rex")
 *			   if (	Rex	is a reserved name )
 *				   dq.requireUniqueInstance();
 * &lt;/PRE&gt;
 *	   Then, get the query results one of two ways:
 * &lt;P&gt;
 *		   #1:&lt;PRE&gt;
 *			   String names	= "";
 *			   DogDO[] dogs	= dq.getDOArray();
 *			   for ( int i = 0;	i &lt; dogs.length;	i++	) {
 *				   names +=	dogs[i].getName() +	" ";
 *			   }
 * &lt;/PRE&gt;
 *		or #2:&lt;PRE&gt;
 *			   String names	= "";
 *			   DogDO dog;
 *			   while ( null	!= ( dog = dq.getNextDO() )	) {
 *				   names +=	dog.getName() +	" ";
 *			   }
 * &lt;/PRE&gt;
 * &lt;P&gt;
 *	   Note:   If &lt;CODE&gt;requireUniqueInstance()&lt;/CODE&gt; was called,
 *			   then	&lt;CODE&gt;getDOArray()&lt;/CODE&gt; or &lt;CODE&gt;getNextDO()&lt;/CODE&gt;
 *			   will	throw an exception if more than	one	"Rex" was found.
 * &lt;P&gt;
 *	   Note:   Results of the query	will come from the Data	Object cache if:
 *			   -  The cache	is available.
 *			   -  Matches were found in	the	cache.
 *			   -  No other tables (Data	Objects	of other types)	were involved
 *				  in the query.
 *				  This can happen if you extend	the	&lt;CODE&gt;DogQuery&lt;/CODE&gt; class
 *				  and you make calls to	the	&lt;CODE&gt;QueryBuilder&lt;/CODE&gt; object
 *				  to add SQL involving other tables.
 *			   If any of these conditions is not true,
 *			   then	any	results	from the query will	come from the database.
 * &lt;P&gt;
 *	   To reuse	the	query object, call:
 * &lt;P&gt;&lt;PRE&gt;
 *			   dq.reset();
 * &lt;/PRE&gt;
 * @author <xsl:value-of select="AUTHOR"/>
 * @version	$Revision$
 */
public class <xsl:value-of select="CLASS_NAME"/>Query implements Query {

	private	QueryBuilder builder;

	/**
	 * logical name	of the database	for	which <xsl:value-of	select="CLASS_NAME"/>Query object has been created
	 */
	private	String logicalDatabase;

	private	ResultSet resultSet	= null;
	private	boolean	uniqueInstance	= false;
	private	boolean	loadData	= false;
	private	<xsl:value-of select="CLASS_NAME"/>DO[]	DOs	= null;
	private	int	arrayIndex = -1;
	private	boolean	needToRun =	true;
// 12.04.2004 tj
//	private	LRUMap cacheHits	= null;
  private	Vector cacheHits	= null;
	private	boolean	isQueryByOId = false;
	private	boolean	hasNonOidCond	= false;
	private	boolean	hitDb =	false;
	private	boolean	userHitDb =	false;
	private	int	maxDBrows =	0;
	private boolean orderRelevant = true; // true if order of query results is relavant, otherwise false
	private	QueryCacheItem queryItem = null;
	private	String currentHandle = null;
	private HashMap refs = new HashMap();
    private int iCurrentFetchSize = -1;
    private int iCurrentQueryTimeout = 0;
    DBTransaction transaction = null;
	
    // GS: Adaptation for dynamic tables.
    private String dbtablename;
	
<xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
	/**
	 * Public constructor.
         *
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">
     * @deprecated Use <xsl:value-of select="CLASS_NAME"/>Query(DBTransaction) instead.</xsl:if>
	 */
     // GS: Constructor needs a tablename to work for dynamic papillon db.
	public <xsl:value-of select="CLASS_NAME"/>Query( String tablename ) {
        this.dbtablename = tablename;
		builder	= new QueryBuilder(	dbtablename, <xsl:value-of select="CLASS_NAME"/>DO.getColumnsNameString(dbtablename) );

		String dbName =	<xsl:value-of select="CLASS_NAME"/>DO.get_logicalDBName();
		try	{
			String vendor =	DODS.getDatabaseManager().logicalDatabaseType(dbName);
			if (vendor != null)	{
				builder.setDatabaseVendor(vendor);
				logicalDatabase	= dbName;
			} else {
				builder.setDatabaseVendor();
				logicalDatabase	=  DODS.getDatabaseManager().getDefaultDB();
			}
		} catch	(Exception e) {
			builder.setDatabaseVendor();
			logicalDatabase	=  DODS.getDatabaseManager().getDefaultDB();
		}
		builder.setUserStringAppendWildcard( false );
		builder.setUserStringTrim( false );
		reset();
	}
</xsl:if>
	/**
	 * Public constructor.
         *
	 * @param dbTrans current database transaction
	 */
     // GS: Constructor needs a tablename to work for dynamic papillon db.
	public <xsl:value-of select="CLASS_NAME"/>Query( String tablename, DBTransaction dbTrans ) {
        this.dbtablename = tablename;
		builder	= new QueryBuilder(	dbtablename, <xsl:value-of select="CLASS_NAME"/>DO.getColumnsNameString(dbtablename) );

		String dbName = null;
		if(dbTrans!=null)
		    dbName = dbTrans.getDatabaseName();
		else
		    dbName = <xsl:value-of select="CLASS_NAME"/>DO.get_logicalDBName();
		try	{
		    transaction = dbTrans;
			String vendor =	DODS.getDatabaseManager().logicalDatabaseType(dbName);
			if (vendor != null)	{
				builder.setDatabaseVendor(vendor);
				logicalDatabase	= dbName;
			} else {
				builder.setDatabaseVendor();
				logicalDatabase	=  DODS.getDatabaseManager().getDefaultDB();
			}
		} catch	(Exception e) {
			builder.setDatabaseVendor();
			logicalDatabase	=  DODS.getDatabaseManager().getDefaultDB();
		}
		builder.setUserStringAppendWildcard( false );
		builder.setUserStringTrim( false );
		reset();
	}


<xsl:if	test="DO_IS_OID_BASED='true' and GENERATE_SECURE='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
	// WebDocWf	extension for DODS row instance	based security
	// The following lines have	been added:
	/**
	 * Constructor for Query with security
	 *
	 * @param usr The user for security	check
	 *
	 * @exception AccessException
	 *	 The user is not allowed to	create a query
	 *
	 * WebDocWf	extension
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">     * @deprecated Use <xsl:value-of select="CLASS_NAME"/>Query(User, DBTransaction) instead.</xsl:if>
	 */
     // GS: Constructor needs a tablename to work for dynamic papillon db.
	public <xsl:value-of select="CLASS_NAME"/>Query(String tablename, org.webdocwf.dods.access.User usr)
	throws AccessException {
        this.dbtablename = tablename;
		assertQueryAccess( usr );
		builder	= new QueryBuilder( dbtablename, <xsl:value-of select="CLASS_NAME"/>DO.getColumnsNameString(dbtablename));
		String dbName =	<xsl:value-of select="CLASS_NAME"/>DO.get_logicalDBName();
		try	{
			String vendor =	DODS.getDatabaseManager().logicalDatabaseType(dbName);
			if (vendor != null)	{
				builder.setDatabaseVendor(vendor);
				logicalDatabase	= dbName;
			} else {
				builder.setDatabaseVendor();
				logicalDatabase	=  DODS.getDatabaseManager().getDefaultDB();
			}
		} catch	(DatabaseManagerException e) {
			throw new AccessEvalException ("INTERNAL ERROR:	unexpected DatabaseManagerException" );
		} catch	(SQLException e) {
			throw new AccessEvalException ("INTERNAL ERROR:	unexpected SQLException" );
		}
		builder.setUserStringAppendWildcard( false );
		builder.setUserStringTrim( false );
		user = usr;
		reset();
	}

</xsl:if>

	// WebDocWf	extension for DODS row instance	based security
	// The following lines have	been added:
	/**
	 * Constructor for Query with security
	 *
	 * @param usr The user for security	check
	 *
	 * @param dbTrans current database transaction

	 * @exception AccessException
	 *	 The user is not allowed to	create a query
	 *
	 * WebDocWf	extension
	 *
	 */
     // GS: Constructor needs a tablename to work for dynamic papillon db.
	public <xsl:value-of select="CLASS_NAME"/>Query	(String tablename, DBTransaction dbTrans, org.webdocwf.dods.access.User	usr)
	throws AccessException {
        this.dbtablename = tablename;
		assertQueryAccess( usr );
		transaction = dbTrans;
   	 builder	= new QueryBuilder(	dbtablename, <xsl:value-of select="CLASS_NAME"/>DO.getColumnsNameString(dbtablename) );
        String dbName = null;
		if(dbTrans!=null)
		    dbName = dbTrans.getDatabaseName();
		else
		    dbName = <xsl:value-of select="CLASS_NAME"/>DO.get_logicalDBName();
		try	{
			String vendor =	DODS.getDatabaseManager().logicalDatabaseType(dbName);
			transaction = dbTrans;
			if (vendor != null)	{
				builder.setDatabaseVendor(vendor);
				logicalDatabase	= dbName;
			} else {
				builder.setDatabaseVendor();
				logicalDatabase	=  DODS.getDatabaseManager().getDefaultDB();
			}
		} catch	(DatabaseManagerException e) {
			throw new AccessEvalException ("INTERNAL ERROR:	unexpected DatabaseManagerException" );
		} catch	(SQLException e) {
			throw new AccessEvalException ("INTERNAL ERROR:	unexpected SQLException" );
		}
		builder.setUserStringAppendWildcard( false );
		builder.setUserStringTrim( false );
		user = usr;
		reset();
	}

</xsl:if>

<xsl:if	test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
	/**
	 * Public constructor.
	 *
	 * @param dbName
	 *	 Logical name of the database from which <xsl:value-of select="CLASS_NAME"/>DO object will be created.
	 *
	 * @exception AccessException
	 *	 The user is not allowed to	create a query
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">     * @deprecated Use <xsl:value-of select="CLASS_NAME"/>Query(DBTransaction) instead.</xsl:if>
	 */
     // GS: Constructor needs a tablename to work for dynamic papillon db.
	public <xsl:value-of select="CLASS_NAME"/>Query(String tablename, String dbName) throws DatabaseManagerException {
        this.dbtablename = tablename;
        if (dbName == null )
		   dbName = getLogicalDatabase();
	   	builder	= new QueryBuilder(	dbtablename, <xsl:value-of select="CLASS_NAME"/>DO.getColumnsNameString(dbtablename) );
		try	{
			String vendor =	DODS.getDatabaseManager().logicalDatabaseType(dbName);
			if (vendor != null)	{
				builder.setDatabaseVendor(vendor);
				logicalDatabase	= dbName;
			} else {
				builder.setDatabaseVendor();
				logicalDatabase	=  DODS.getDatabaseManager().getDefaultDB();
			}
		} catch	(SQLException e) {
			throw new DatabaseManagerException ("INTERNAL ERROR: unexpected	SQLException" );
		}
		builder.setUserStringAppendWildcard( false );
		builder.setUserStringTrim( false );
		reset();
	}

</xsl:if>

<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">

	//	WebDocWf extension for DODS	row	instance based security
	//	The	following lines	have been added:
	/**
	 * Constructor for Query with security
	 *
	 * @param dbName
	 *	 Logical name of the database from which <xsl:value-of select="CLASS_NAME"/>DO object will be created.
	 * @param usr The user for security	check
	 *
	 * @exception AccessException
	 *	 The user is not allowed to	create a query
<xsl:if test="/TABLE/GENERATE_DIRTY='Deprecate'">     * @deprecated Use <xsl:value-of select="CLASS_NAME"/>Query(User, DBTransaction) instead.</xsl:if>
	 *
	 * WebDocWf	extension
	 */
     // GS: Constructor needs a tablename to work for dynamic papillon db.
	public <xsl:value-of select="CLASS_NAME"/>Query	( String tablename, String dbName, org.webdocwf.dods.access.User usr )
	throws AccessException,	DatabaseManagerException {
        this.dbtablename = tablename;
		assertQueryAccess( usr );
  		builder	= new QueryBuilder(	dbtablename, <xsl:value-of select="CLASS_NAME"/>DO.getColumnsNameString(dbtablename) );
      if (dbName == null )
			dbName = getLogicalDatabase();
		try	{
			String vendor =	DODS.getDatabaseManager().logicalDatabaseType(dbName);
			if (vendor != null)	{
				builder.setDatabaseVendor(vendor);
				logicalDatabase	= dbName;
			} else {
				builder.setDatabaseVendor();
				logicalDatabase	=  DODS.getDatabaseManager().getDefaultDB();
			}
		} catch	(SQLException e) {
			throw new DatabaseManagerException ("INTERNAL ERROR: unexpected	SQLException" );
		}
		builder.setUserStringAppendWildcard( false );
		builder.setUserStringTrim( false );
		user = usr;
		reset();
	}
</xsl:if>
	//	WebDocWf extension for DODS	row	instance based security
	//	The	following lines	have been added:
</xsl:if>
</xsl:if>
	/**
	 * Return logical name of the database that	<xsl:value-of select="CLASS_NAME"/>Query object	uses
	 *
	 * @return param logical database name
	 *
	 */
	public String getLogicalDatabase() {
		return logicalDatabase;
	}

	/**
	 * Change logical database to another logical database (which name is dbName)
	 *
	 * @param dbName the logical name of the database
	 * @exception SQLException
	 * @exception DatabaseManagerException
	 */
	public void	setLogicalDatabase(String dbName) throws SQLException, DatabaseManagerException	{
		String vendor =	DODS.getDatabaseManager().logicalDatabaseType(dbName);
		if (vendor != null)	{
			builder.setDatabaseVendor(vendor);
			logicalDatabase	= dbName;
		} else {
			builder.setDatabaseVendor();
			logicalDatabase	=  DODS.getDatabaseManager().getDefaultDB();
		}
		reset();
	}

<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
	/**
	 * Ensure that the given user is allowed to	create a query
	 *
	 * @param usr The user for security	check
	 *
	 * @exception AccessException
	 *	 The user is not allowed to	create a query
	 *
	 * WebDocWf	extension
	 *
	 */
	static public void assertQueryAccess( org.webdocwf.dods.access.User	usr	)
	throws AccessException {
		if (!hasQueryAccess( usr ))
			throw new AccessRightException("No access !", usr, "Query",	null, dbtablename, null, null, null, null, null);
	}

	/**
	 * Check whether the given user	is allowed to create a query
	 *
	 * @param usr The user for security	check
	 *
	 * @return Whether the given user is allowed to	create a query
	 *
	 * WebDocWf	extension
	 *
	 */
	static public boolean hasQueryAccess( org.webdocwf.dods.access.User	usr	)
	throws AccessEvalException		{
		return usr.hasQueryAccess( &quot;<xsl:value-of select="translate(PACKAGE,'/','.')"/>.<xsl:value-of select="CLASS_NAME"/>DO&quot; );
	}

	// end of WebDocWf extension for DODS row instance based security
</xsl:if>
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
	// WebDocWf	extension for DODS row instance	security
	// The following lines has been	added:
	private	org.webdocwf.dods.access.User user	 = null;

	/**
	 * Get the user	of the query
	 *
	 * @return The user	of the query
	 *
	 * WebDocWf	extension
	 *
	 */
	public org.webdocwf.dods.access.User getUser() { return	user; }
	// end of WebDocWf extension for DODS row instance security
</xsl:if>

	//	WebDocWf extension for unique query	results	without	SQL	DISTINCT
	//	The	following lines	has	been added:
	private	boolean	unique = false;

	/**
	 * Set the unique flag of the query
	 *
	 * @param newUnique	The	unique flag	for	the	query
	 *
	 * WebDocWf	extension
	 *
	 */
	public void	setUnique(boolean newUnique) { unique =	newUnique; }

	/**
	 * Get the unique flag of the query
	 *
	 * @return The unique flag of the query
	 *
	 * WebDocWf	extension
	 *
	 */
	public boolean getUnique() { return	unique;	}

	//	WebDocWf extension for skipping	the	n first	rows of	the	result
	//	The	following lines	has	been added:
	private	int	readSkip = 0;

	/**
	 * Set the readSkip	number of the query
	 *
	 * @param newReadSkip The number of	results	to skip.
	 *
	 * WebDocWf	extension
	 *
	 */
	public void	setReadSkip(int	newReadSkip) {
	  readSkip = newReadSkip;
	}

	/**
	 * Get the readSkip	number of the query
	 *
	 * @return The number of rows which	are	skipped
	 *
	 * WebDocWf	extension
	 *
	 */
	public int getReadSkip() { return readSkip;	}

	// WebDocWf	extension for select rowcount limit
	// The following lines has been	added:
	private	int	databaseLimit =	0;

	/**
	 * Set the database	limit of the query
	 *
	 * @param newLimit The limit for the query
	 *
	 * WebDocWf	extension
	 *
	 */
	public void	setDatabaseLimit(int newLimit) {
	  databaseLimit	= newLimit;
	}

	/**
	 * Get the database	limit of the query
	 *
	 * @return The database	limit of the query
	 *
	 * WebDocWf	extension
	 *
	 */
	public int getDatabaseLimit() {	return databaseLimit; }

	private	boolean	databaseLimitExceeded =	false;

	/**
	 * Get the database	limit exceeded flag	of the query.
	 *
	 * @return The database	limit exceeded flag	of the query
	 *	 True if there would have been more	rows than the limit, otherwise false.
	 *
	 * WebDocWf	extension
	 */
	public boolean getDatabaseLimitExceeded() {	return databaseLimitExceeded; }
	// end of WebDocWf extension for select	rowcount limit

	/**
	 * Set that	all	queries	go to database,	not	to cache.
	 */
	public void	hitDatabase() {	userHitDb =	true; }

	// WebDocWf	extension for extended wildcard	support
	// The following rows have been	added:

	/**
	 * Set user	string wildcard.
	 *
	 * @param newUserStringWildcard	New	user string	wildcard.
	 *
	 * WebDocWf	extension
	 */
	public void	setUserStringWildcard(String newUserStringWildcard)	{
		builder.setUserStringWildcard( newUserStringWildcard );
	}

	/**
	 * Set user	string single wildcard.
	 *
	 * @param newUserStringSingleWildcard New user string single wildcard.
	 *
	 * WebDocWf	extension
	 */
	public void	setUserStringSingleWildcard(String newUserStringSingleWildcard)	{
		builder.setUserStringSingleWildcard( newUserStringSingleWildcard );
	}

	/**
	 * Set user	string single wildcard escape.
	 *
	 * @param newUserStringSingleWildcardEscape	New	user string	single wildcard	escape.
	 *
	 * WebDocWf	extension
	 */
	public void	setUserStringSingleWildcardEscape(String newUserStringSingleWildcardEscape)	{
		builder.setUserStringSingleWildcardEscape( newUserStringSingleWildcardEscape );
	}

	/**
	 * Set user	string wildcard	escape.
	 *
	 * @param newUserStringWildcardEscape New user string wildcard escape.
	 *
	 * WebDocWf	extension
	 */
	public void	setUserStringWildcardEscape(String newUserStringWildcardEscape)	{
		builder.setUserStringWildcardEscape( newUserStringWildcardEscape );
	}

	/**
	 * Set user	string append wildcard.
	 *
	 * @param userStringAppendWildcard New user	string append wildcard.
	 *
	 * WebDocWf	extension
	 */
	public void	setUserStringAppendWildcard(boolean	userStringAppendWildcard ) {
		builder.setUserStringAppendWildcard( userStringAppendWildcard );
	}

	 /**
	 * Set user	string trim.
	 *
	 * @param userStringTrim New user string trim.
	 *
	 * WebDocWf	extension
	 */
	public void	setUserStringTrim(boolean userStringTrim ) {
		builder.setUserStringTrim( userStringTrim );
	}

	/**
	 * Get user	string wildcard.
	 *
	 * @return User	string wildcard.
	 *
	 * WebDocWf	extension
	 */
	public String getUserStringWildcard() {
		return builder.getUserStringWildcard();
	}

	/**
	 * Get user	string single wildcard.
	 *
	 * @return User	string single wildcard.
	 *
	 * WebDocWf	extension
	 */
	public String getUserStringSingleWildcard()	{
		return builder.getUserStringSingleWildcard();
	}

	/**
	 * Get user	string single wildcard escape.
	 *
	 * @return User	string single wildcard escape.
	 *
	 * WebDocWf	extension
	 */
	public String getUserStringSingleWildcardEscape() {
		return builder.getUserStringSingleWildcardEscape();
	}

	/**
	 * Get user	string wildcard	escape.
	 *
	 * @return User	string wildcard	escape.
	 *
	 * WebDocWf	extension
	 */
	public String getUserStringWildcardEscape()	{
		return builder.getUserStringWildcardEscape();
	}

	/**
	 * Get user	string append wildcard.
	 *
	 * @return User	string append wildcard.
	 *
	 * WebDocWf	extension
	 */
	public boolean getUserStringAppendWildcard() {
		return builder.getUserStringAppendWildcard();
	}

	/**
	 * Get user	string trim.
	 *
	 * @return User	string trim.
	 *
	 * WebDocWf	extension
	 */
	public boolean getUserStringTrim() {
		return builder.getUserStringTrim();
	}
	// end of WebDocWf extension for extended wildcard support

	/**
	 * Perform the query on	the	database, and prepare the array of	returned
	 * DO	objects.
	 *
	 * @param DOs Vector of result oids which will be switched with the whole DOs
	 * (from the database).
	 * @exception DataObjectException If a database	access error occurs.
	 * @exception NonUniqueQueryException If too many rows were	found.
	 */
	private	void getQueryByOIds(Vector DOs)
	throws DataObjectException {
		if (DOs.size() == 0)
			return;
		<xsl:value-of select="CLASS_NAME"/>DO DO = null;
		DOShell	shell =	null;
		<xsl:value-of select="CLASS_NAME"/>Query tmpQuery = null;
		Date startQueryTime	= new Date();
		long queryTime = 0;
		for	(int i=0; i&lt;DOs.size(); i++)	{
			shell =	(DOShell)DOs.elementAt(i);
			tmpQuery = new <xsl:value-of select="CLASS_NAME"/>Query(this.dbtablename, transaction);
			try {
			 tmpQuery.setQueryHandle( shell.handle );
             tmpQuery.requireUniqueInstance();

             DO = tmpQuery.getNextDO();
             if ( null == DO )
                    throw new DataObjectException(&quot;<xsl:value-of select="CLASS_NAME"/>DO DO not found for id=&quot; + shell.handle );
            } catch ( Exception e ) {
               throw new DataObjectException( "Duplicate ObjectId" );
            }
            shell.dataObject = DO;
		}
		Date stopQueryTime = new Date();
		queryTime =	stopQueryTime.getTime()	- startQueryTime.getTime();
		<xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).updateQueryByOIdAverageTime((new Long(queryTime)).intValue(),DOs.size());
	}



	/**
	 * Perform the query on	the	database, and prepare the
	 * array of	returned DO	objects.
	 *
	 * @exception DataObjectException If a database	access error occurs.
	 * @exception NonUniqueQueryException If too many rows were	found.
	 */
	private	void runQuery()
	throws DataObjectException,	NonUniqueQueryException	{
	    needToRun =	false;
	    arrayIndex = -1;
	    DBQuery	dbQuery	= null;
	    Date startQueryTime	= new Date();
	    long queryTime = 0;
	    boolean	readDOs	= false;
	    boolean canUseQueryCache = true;
	    CacheStatistics stat = null;
	    boolean resultsFromQCache = false;
	    QueryCacheItem queryCachedItem = null;

	    if(builder.isUnionTableJoin())throw new DataObjectException( "Could not use UNION [ALL] statement in query witch retrieve data object." );

	    if ((transaction!=null) &amp;&amp;
		(transaction instanceof com.lutris.appserver.server.sql.CachedDBTransaction)) {
		if(((com.lutris.appserver.server.sql.CachedDBTransaction)transaction).getAutoWrite()) try {
		    transaction.write();
		} catch (SQLException sqle) {
		    sqle.printStackTrace();
		    throw new DataObjectException("Couldn't write transaction: "+sqle);
		}
		((com.lutris.appserver.server.sql.CachedDBTransaction)transaction).dontAggregateDOModifications();
	    }
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
// check user for Query restriction
		try	{
			if (user !=	null)
				user.restrictQuery(this);
		} catch	( AccessEvalException ae ) {
			throw new DataObjectException("AccessException in restrictQuery", ae);
		}
</xsl:if>
		try	{
			QueryResult	results	= null;
			DOShell	shell =	null;
			if (isQueryByOId &amp;&amp;	!hasNonOidCond) {	// query by	OId
				builder.setCurrentFetchSize(1);
				results	= new QueryResult();
				if (currentHandle != null) {
        		 	if(transaction!=null &amp;&amp; _tr_(transaction).getTransactionCache()!=null &amp;&amp; !loadData) {
        			     <xsl:value-of select="CLASS_NAME"/>DO DO= (<xsl:value-of select="CLASS_NAME"/>DO)_tr_(transaction).getTransactionCache().getDOByHandle(currentHandle);
        			     if(DO!=null){
        			        shell = new DOShell(DO);
        					results.DOs.add(shell);
        					resultsFromQCache = true;
        				 }
// tj 12.04.2004 put under comment next 2 lines
//        				 else
//        				    resultsFromQCache = false;
        			}
        			if(!resultsFromQCache) {  //	DO isn't found in the transaction cache
        			    <xsl:value-of select="CLASS_NAME"/>DataStruct DS = (<xsl:value-of select="CLASS_NAME"/>DataStruct)<xsl:value-of	select="CLASS_NAME"/>DO.getCache(this.dbtablename).getDataStructByHandle(currentHandle);
        			    if (DS != null &amp;&amp; !(DS.isEmpty &amp;&amp; loadData))	{ // DO	is found in	the	cache
        				  <xsl:value-of select="CLASS_NAME"/>DO DO = (<xsl:value-of select="CLASS_NAME"/>DO)<xsl:value-of select="CLASS_NAME"/>DO.ceInternal(this.dbtablename, DS.get_OId(), transaction);
        				  shell =	new	DOShell(DO);
        				  results.DOs.add(shell);
        				  resultsFromQCache = true;
        				 }
// tj 12.04.2004 put under comment next 4 lines
//        				 else{ //	DO isn't found in the cache
//        				  	 if (<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).isFull())
//        						 resultsFromQCache =	false;
//        				 }
        			}
				}//currentHandle != null
			}
			else { //	other queries
                if (	<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).isFull() &amp;&amp; (<xsl:value-of select="CLASS_NAME"/>DO.isFullCacheNeeded)
                                          &amp;&amp;(!hitDb) &amp;&amp; (maxDBrows == 0) &amp;&amp; (databaseLimit == 0)
                                          &amp;&amp; (readSkip == 0) &amp;&amp; !builder.isMultiTableJoin() ) {
                      resultsFromQCache = true;
                }
                else {
                 if (<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getLevelOfCaching()	== CacheConstants.QUERY_CACHING) {
                    if (builder.isMultiTableJoin()) { // statistics about multi join query
                      stat = null;
       				        stat = <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).getCacheStatistics(CacheConstants.MULTI_JOIN_QUERY_CACHE);
       				        if (stat!= null){
       				          stat.incrementCacheAccessNum(1);
       				        }
       				      }
       				      else {
	                    if (hitDb) { // statistics about complex query
	       				        stat = null;
	       				        stat = <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).getCacheStatistics(CacheConstants.COMPLEX_QUERY_CACHE);
	       				        if (stat!= null){
	       				         stat.incrementCacheAccessNum(1);
	       				        }
	       				      }else{ // statistics about simple query
	       				        stat = null;
	       				        stat = <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).getCacheStatistics(CacheConstants.SIMPLE_QUERY_CACHE);
	       				        if (stat!= null){
	       				          stat.incrementCacheAccessNum(1);
	       				        }
	       			       }
	       			     }
                 }
                 if(transaction != null)
                    canUseQueryCache = !transaction.preventCacheQueries();
       			  if ((<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getLevelOfCaching()	== CacheConstants.QUERY_CACHING) &amp;&amp; canUseQueryCache) {
       					String queryID = builder.getSQLwithParms(); //unique representation of	query
       					int resNum = 0;
       					int evaluateNo = 0;
           			if (builder.isMultiTableJoin()) {
           			  queryCachedItem = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getMultiJoinQueryCacheItem(logicalDatabase, queryID);
           			}
           			else{
           			   if (hitDb)
           			      queryCachedItem = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getComplexQueryCacheItem(logicalDatabase, queryID);
           			   else
           			      queryCachedItem = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getSimpleQueryCacheItem(logicalDatabase, queryID);
           			}
           				queryItem.setQueryId(queryID);  // queryItem defined as private template attribute
           		  if (queryCachedItem == null) { // query	doesn't	exist
// tj 03.09.2004        					   if ((!builder.isMultiTableJoin()) || <xsl:value-of select="CLASS_NAME"/>DO.isAllReadOnly())
        					   if (builder.isMultiTableJoin()){
        					     ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).addMultiJoinQuery(queryItem); // register multi join query
        					   }
        					   else {
        			   		   if (hitDb)
        					         ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).addComplexQuery(queryItem); // register complex query
        					      else
        						      ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).addSimpleQuery(queryItem); //	register simple	query
           			   	}
           				}
           				else{ //	query found
           				   if ( !(isOrderRelevant() &amp;&amp; queryCachedItem.isModifiedQuery()) ) {
    				           if (builder.isMultiTableJoin()){ // statistics about multi join cache
    				             stat = null;
           				       stat = <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).getCacheStatistics(CacheConstants.MULTI_JOIN_QUERY_CACHE);
           				       if (stat!= null){
           				           stat.incrementCacheHitsNum(1);
           				       }
    				           }
    				           else {
    				            if (hitDb) { // statistics about complex cache
        					         stat = null;
           				         stat = <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).getCacheStatistics(CacheConstants.COMPLEX_QUERY_CACHE);
           				         if (stat!= null){
           				             stat.incrementCacheHitsNum(1);
           				         }
        				   	    }else{ // statistics about simple cache
        					         stat = null;
           				         stat = <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).getCacheStatistics(CacheConstants.SIMPLE_QUERY_CACHE);
           				         if (stat!= null){
           				            stat.incrementCacheHitsNum(1);
           				         }
        				         }
        				       }
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
                           if (user == null) {
</xsl:if>
                              int limitOfRes;
                              if (databaseLimit == 0)
                                 limitOfRes = 0;
                              else
                                 limitOfRes = readSkip+databaseLimit+1;
                              if (! unique) {
                                 if (builder.isMultiTableJoin()) {
                                   results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getMultiJoinQueryResults(logicalDatabase, queryID, limitOfRes, maxDBrows);
                                 }
                                 else {
	                                 if (hitDb)
	                                   results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getComplexQueryResults(logicalDatabase, queryID, limitOfRes, maxDBrows);
	                                 else
	                                   results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getSimpleQueryResults(logicalDatabase, queryID, limitOfRes, maxDBrows);
	                               }
                              }else{ // (! unique)
                                 if (builder.isMultiTableJoin()) {
                                   results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getMultiJoinQueryResults(logicalDatabase, queryID, limitOfRes, maxDBrows, true);
                                 }
                                 else {
	                                 if (hitDb)
	                                    results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getComplexQueryResults(logicalDatabase, queryID, limitOfRes, maxDBrows, true);
	                                 else
	                                    results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getSimpleQueryResults(logicalDatabase, queryID, limitOfRes, maxDBrows, true);
	                               }
                              } // (! unique)
                              if (results != null) {
                                 resNum = results.DOs.size();
// tj 01.02.2004 remove skipped
                              	 if (readSkip &gt; 0) {
                                     if (results.DOs.size() &gt; readSkip) {
                                        for (int i = 0; i &lt; readSkip; i++) {
                                           results.DOs.remove(0);
                                        }
                                      }
                                      else {
                                         results.DOs.clear();
                                      }
                                 }

 //sinisa 06.08.2003.
                                 results = getCachedResults(results);
                                 if (databaseLimit != 0) { // databaseLimitExceeded begin
                                     if (resNum == readSkip+databaseLimit+1) {
                                         resNum--;
                                         databaseLimitExceeded = true;
                                         results.DOs.remove(databaseLimit);
                                     }else{
                                         if ( (resNum == readSkip+databaseLimit) &amp;&amp; (!queryCachedItem.isCompleteResult()) )
                                              databaseLimitExceeded = true;
                                     }
                                 }  // databaseLimitExceeded end
                                 if ( (databaseLimit!=0 &amp;&amp;(resNum == (readSkip+databaseLimit))) || (maxDBrows!=0 &amp;&amp; (resNum + results.skippedUnique) == maxDBrows) || (queryCachedItem.isCompleteResult()) ) {
     				                      int lazyTime = <xsl:value-of	select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).getQueryByOIdAverageTime()*results.lazy.size();
                 		         		 if (lazyTime &lt;= queryCachedItem.getTime()) {
                 				              resultsFromQCache = true;
                 					           getQueryByOIds(results.lazy);  // gets cached query results from database
                                     }else
                                        databaseLimitExceeded = false;
                 			         }else
                                     databaseLimitExceeded = false;

												//remove skiped
                              } // (results != null)
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
                          }  // if (user == null)
                          else {
                            if (! unique) {
                                if (builder.isMultiTableJoin()) {
                                  results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getMultiJoinQueryResults(logicalDatabase, queryID, 0, maxDBrows);
                                }
                                else {
	                                if (hitDb)
	                                    results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getComplexQueryResults(logicalDatabase, queryID, 0, maxDBrows);
	                                else
	                                    results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getSimpleQueryResults(logicalDatabase, queryID, 0, maxDBrows);
	                              }
                            }
                            else { // (! unique)
                               if (builder.isMultiTableJoin()) {
                                 results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getMultiJoinQueryResults(logicalDatabase, queryID, 0, maxDBrows, true);
                               }
                               else {
	                               if (hitDb)
	                                   results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getComplexQueryResults(logicalDatabase, queryID, 0, maxDBrows, true);
	                               else
	                                   results = ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getSimpleQueryResults(logicalDatabase, queryID, 0, maxDBrows, true);
	                             }
                            } // (! unique)
                            if (results != null) {
                                results = getCachedResults(results);
                                resNum = results.DOs.size();
                                Double d = new Double((readSkip + databaseLimit)* (1+((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).getReserveFactor()));
                                evaluateNo = d.intValue();
                                int i=0;
                                if ((results.DOs.size() &gt;= evaluateNo) || ((evaluateNo == 0) &amp;&amp; (queryCachedItem.isCompleteResult()))){
                                    int noInBase = 0;
                                    int noInCache = 0;
                                    DOShell tmpShell = null;
                                    <xsl:value-of select="CLASS_NAME"/>DO tmpDO = null;
                                    Vector removeVector = null;
                                    if ((evaluateNo == 0) &amp;&amp; (queryCachedItem.isCompleteResult()))
                                       evaluateNo = results.DOs.size();
                                    while (i &lt; evaluateNo) {
                                        tmpShell = (DOShell)results.DOs.elementAt(i);
                                        tmpDO = (<xsl:value-of select="CLASS_NAME"/>DO) tmpShell.dataObject;
            				                if (tmpDO != null){
            				                    try {
                                                if (!tmpDO.hasQueryFindAccess(user)) {
                                                    if (removeVector == null)
                                                        removeVector = new Vector();
                                                    removeVector.add(tmpShell);
                                                }
                                                else { // (!tmpDO.hasQueryFindAccess(user))
                                                    noInCache++;
                                                }
    		        		                       }catch	( AccessEvalException ae ) {
    				        	                        throw new DataObjectException("AccessException in restrictQuery", ae);
            				                    }
                                        } // (tmpDO != null)
                                        else {
                                            noInBase++;
                                        }
                                        i++;
                                    } // while
                                    if (removeVector != null) {
                                        i = 0;
                                        while (i &lt; removeVector.size()) {
                                            results.DOs.remove(removeVector.elementAt(i++));
                                        }
                                        removeVector = null;
                                    }
                                    int lazyTime = <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).getQueryByOIdAverageTime() * noInBase;
                  			  	      if (lazyTime &lt;= queryCachedItem.getTime()) {
                                         i = 0;
                                         Vector tmpVector = new Vector();
                                         int resultCount = 0;
// tj 14.04.2004                                          Date startOidQueryTime = null;
// tj 14.04.2004                                         Date stopOidQueryTime = null;
// tj 14.04.2004                                         queryTime=0;

                                         while (	(i &lt; results.DOs.size()) &amp;&amp;	(databaseLimit==0 || (tmpVector.size()&lt;=databaseLimit)) ) {
                                            tmpShell = (DOShell)results.DOs.elementAt(i);
                                            tmpDO = (<xsl:value-of select="CLASS_NAME"/>DO) tmpShell.dataObject;
                                            if (tmpDO == null) {
// tj 14.04.2004                                                startOidQueryTime	= new Date();
// tj 14.04.2004       		                                    queryTime = 0;
<xsl:if	test="DO_IS_MULTIDB_BASED='true'">
        						                        if (logicalDatabase	!= null) {
        						                           if(transaction!=null)
        							                            tmpDO = <xsl:value-of select="CLASS_NAME"/>DO.ceInternal ( this.dbtablename, tmpShell.handle, transaction);
        							                        else
        							                            tmpDO = <xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename, <xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">logicalDatabase,</xsl:if> tmpShell.handle<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">, (DBTransaction)null</xsl:if>);
        							                     }else
       </xsl:if>
         					                           tmpDO =	<xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename, tmpShell.handle,transaction);
// tj 14.04.2004          					                        Date stopQueryTime = new Date();
// tj 14.04.2004       		                                    queryTime =	stopQueryTime.getTime()	- startQueryTime.getTime();
// tj 14.04.2004         		                                 <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).updateQueryByOIdAverageTime((new Long(queryTime)).intValue(),1);

          					                        tmpShell.dataObject = tmpDO;
                                            }
                                            if (tmpDO != null){
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
    				                                try	{
        							                       if ( user == null || tmpDO.hasQueryFindAccess( user	) )	{
</xsl:if>
                                                        if (resultCount &gt;= readSkip)	{
                                                            tmpVector.add(tmpShell);
                                                        }
                  						                    resultCount++;
                                                   }
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
                    					              }catch( AccessException ae ) {
                  							           throw new DataObjectException("AccessException in runQuery()", ae);
                  						           }
</xsl:if>
                                            }  //(tmpDO != null)
                                            i++;
                                        } // while
                                        if (databaseLimit != 0) { // tanja databaseLimitExceeded begin
                                            if (tmpVector.size() == databaseLimit+1) {
                                               tmpVector.remove(databaseLimit);
                                               databaseLimitExceeded = true;
                                            }else{
                                                if ((tmpVector.size() == databaseLimit) &amp;&amp; (!queryCachedItem.isCompleteResult()) )
                                                databaseLimitExceeded = true;
                                            }
                                        }   // databaseLimitExceeded end
                                        if ((queryCachedItem.isCompleteResult()) ||(results.DOs.size() == databaseLimit)|| ((maxDBrows ==  resNum+results.skippedUnique) &amp;&amp; (maxDBrows != 0))) {
                                           resultsFromQCache = true;
                                           results.lazy = null;
                                           results.DOs = tmpVector;
                                        }else
                                            databaseLimitExceeded = false;
                                    } // (lazyTime	&lt; queryCachedItem.getTime())
                                }// (results.DOs.size() &gt; evaluateNo)
                            } // results != null)
                          }  //  (user == null)
</xsl:if>
                        } // !(isOrderRelevant() &amp;&amp; queryCachedItem.isModifiedQuery())
                    } // query found
                  } // if QUERY_CACHING
                } // full caching
            } // other queries
            if (( userHitDb) || (!resultsFromQCache)) { // go to	database
<xsl:if test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
                if ((transaction==null)&amp;&amp;(logicalDatabase	!= null))
 			           dbQuery	= <xsl:value-of select="CLASS_NAME"/>DO.createQuery(logicalDatabase);
   			    else
</xsl:if></xsl:if>
				        dbQuery	= <xsl:value-of	select="CLASS_NAME"/>DO.createQuery(transaction);
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
				if ((user == null) &amp;&amp; (databaseLimit !=	0) &amp;&amp; (! unique))
					setMaxRows(databaseLimit + readSkip + 1);
</xsl:if>
			   if(uniqueInstance)
			     builder.setCurrentFetchSize(1);

		   	results	= new QueryResult();
			   	int	resultCount=0;
				   boolean	bHasMoreResults	= false;
          if (( (<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching()) &amp;&amp; (!builder.getPreventPrimaryKeySelect())  &amp;&amp; !loadData) {
		         builder.resetSelectedFields();
        		   builder.setSelectClause(dbtablename+<xsl:value-of select="CLASS_NAME"/>DO.get_OIdColumnName()+&quot;, &quot;+this.dbtablename+<xsl:value-of select="CLASS_NAME"/>DO.get_versionColumnName());
		       }
		       else
		        builder.setSelectClause(<xsl:value-of select="CLASS_NAME"/>DO.getColumnsNameString(dbtablename));
				   dbQuery.query( this	);	  // invokes executeQuery


				   if (! unique) {
    	   			while (	(bHasMoreResults = resultSet.next()) &amp;&amp;	(databaseLimit==0 || (results.DOs.size()&lt;databaseLimit)) )	{
    					   <xsl:value-of select="CLASS_NAME"/>DO newDO;
    					   <xsl:value-of select="CLASS_NAME"/>DataStruct newDS;
<xsl:if	test="DO_IS_MULTIDB_BASED='true'">
     					   if (logicalDatabase	!= null) {
		                   if(transaction!=null) {
			                 if (( (<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching()) &amp;&amp; (!builder.getPreventPrimaryKeySelect())  &amp;&amp; !loadData) {
                              newDO = <xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename,  new ObjectId(resultSet.getBigDecimal(CoreDO.get_OIdColumnName())) , refs, transaction);
                              newDO.set_Version(resultSet.getInt(<xsl:value-of select="CLASS_NAME"/>DO.get_versionColumnName()));
                          } else
  			                    newDO = <xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename,  resultSet , refs, transaction);
			               } else {
			                    if (( (<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching()) &amp;&amp; (!builder.getPreventPrimaryKeySelect())  &amp;&amp; !loadData) {
			                        newDO =<xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename, <xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">logicalDatabase,</xsl:if> new ObjectId(resultSet.getBigDecimal(1)) , refs<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">, (DBTransaction)null</xsl:if>);
                                                newDO.set_Version(resultSet.getInt(<xsl:value-of select="CLASS_NAME"/>DO.get_versionColumnName()));
                             } else
			                        newDO =<xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename, <xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">logicalDatabase,</xsl:if> resultSet , refs<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">, (DBTransaction)null</xsl:if>);
      					   }
      					    if(transaction ==null) {
                             if(newDO!=null &amp;&amp; newDO.isTransactionCheck()) {
                                DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+newDO.get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+newDO.get_Handle()+", version: "+newDO.get_Version()+" \n");
                                (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

                             }
                         }

    					   }
    					   else { // logicalDatabase != null
</xsl:if>
                    if (( (<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching()) &amp;&amp; (!builder.getPreventPrimaryKeySelect())  &amp;&amp; !loadData) {
   						   newDO =	<xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename,  new ObjectId(resultSet.getBigDecimal(CoreDO.get_OIdColumnName())) , refs, transaction);
                                                   newDO.set_Version(resultSet.getInt(<xsl:value-of select="CLASS_NAME"/>DO.get_versionColumnName()));
   						} else
   						    newDO =	<xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename,  resultSet , refs, transaction);
       					    if(transaction==null) {
                             if(newDO!=null &amp;&amp; newDO.isTransactionCheck()) {
                                DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+newDO.get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+newDO.get_Handle()+", version: "+newDO.get_Version()+" \n");
                                (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

                             }
                          }
    <xsl:if	test="DO_IS_MULTIDB_BASED='true'">
    					    } // logicalDatabase != null
    </xsl:if>
                        if (queryItem != null) {
                            queryItem.add((<xsl:value-of select="CLASS_NAME"/>DataStruct)newDO.originalData_get());
                        }
                        if (( (<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching()) &amp;&amp; (!builder.getPreventPrimaryKeySelect())  &amp;&amp; !loadData)
                         {}
                        else
                           newDS =<xsl:value-of select="CLASS_NAME"/>DO.addToCache(this.dbtablename, (<xsl:value-of select="CLASS_NAME"/>DataStruct)newDO.originalData_get());
    <xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
    					try	{
    					    if ( user == null || newDO.hasQueryFindAccess( user	) )	{
    </xsl:if>
        					    if (resultCount	&gt;= readSkip)	{
                                    shell =	new	DOShell(newDO);
                                    results.DOs.add(shell);
    							}
    						    resultCount++;
    <xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
    						}
    					}
    					catch	( AccessException ae ) {
    					    throw new DataObjectException("AccessException in runQuery()", ae);
    					}
    </xsl:if>
    				} // while
 			    } // (!unique)
<xsl:if test="DO_IS_OID_BASED='true'">
                else { // (! unique)
 			        HashSet	hsResult = new HashSet(readSkip+databaseLimit);
				    while((bHasMoreResults = resultSet.next()) &amp;&amp; (databaseLimit==0 || (results.DOs.size()&lt;databaseLimit)) ) {
 				        <xsl:value-of select="CLASS_NAME"/>DO newDO;
 				        <xsl:value-of select="CLASS_NAME"/>DataStruct newDS;
<xsl:if	test="DO_IS_MULTIDB_BASED='true'">
						if (logicalDatabase	!= null){
		                   if(transaction!=null) {
    			                 if (( (<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching()) &amp;&amp; (!builder.getPreventPrimaryKeySelect())  &amp;&amp; !loadData) {
                                 newDO = <xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename,  new ObjectId(resultSet.getBigDecimal(CoreDO.get_OIdColumnName())) , refs, transaction);
                                 newDO.set_Version(resultSet.getInt(<xsl:value-of select="CLASS_NAME"/>DO.get_versionColumnName()));
                          } else
  			                  newDO = <xsl:value-of select="CLASS_NAME"/>DO.ceInternal ( this.dbtablename, resultSet , refs, transaction);
			                } else {
		                    if (( (<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching()) &amp;&amp; (!builder.getPreventPrimaryKeySelect())  &amp;&amp; !loadData) {
		                        newDO =<xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename, <xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">logicalDatabase,</xsl:if> new ObjectId(resultSet.getBigDecimal(1)) , refs<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">, (DBTransaction)null</xsl:if>);
                                        newDO.set_Version(resultSet.getInt(<xsl:value-of select="CLASS_NAME"/>DO.get_versionColumnName()));
                          } else
   			                 newDO = <xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename, <xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">logicalDatabase,</xsl:if> resultSet , refs<xsl:if test="/TABLE/GENERATE_DIRTY='Omit'">, (DBTransaction)null</xsl:if>);
			              }
							if(transaction==null) {
                                if(newDO!=null &amp;&amp; newDO.isTransactionCheck()) {
                                   DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+newDO.get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+newDO.get_Handle()+", version: "+newDO.get_Version()+" \n");
                                   (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

                                }
                             }
						}
						else {
</xsl:if>
                    if (( (<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching()) &amp;&amp; (!builder.getPreventPrimaryKeySelect())  &amp;&amp; !loadData) {
   						   newDO =	<xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename,  new ObjectId(resultSet.getBigDecimal(CoreDO.get_OIdColumnName())) , refs, transaction);
                                                   newDO.set_Version(resultSet.getInt(<xsl:value-of select="CLASS_NAME"/>DO.get_versionColumnName()));
   						} else
    					      newDO =	<xsl:value-of select="CLASS_NAME"/>DO.ceInternal (this.dbtablename,  resultSet , refs , transaction);
							if(transaction==null) {
                                if(newDO!=null &amp;&amp; newDO.isTransactionCheck()) {
                                   DODS.getLogChannel().write(Logger.WARNING, "DO without transaction context is created : Database: "+newDO.get_OriginDatabase()+" <xsl:value-of select="CLASS_NAME"/>DO class, oid: "+newDO.get_Handle()+", version: "+newDO.get_Version()+" \n");
                                   (new Throwable()).printStackTrace(DODS.getLogChannel().getLogWriter(Logger.WARNING));

                                }
                              }
<xsl:if	test="DO_IS_MULTIDB_BASED='true'">
						}
</xsl:if>
                        if (queryItem != null) {
                            queryItem.add((<xsl:value-of select="CLASS_NAME"/>DataStruct)newDO.originalData_get());
                        }
                        if (( (<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching()) &amp;&amp; (!builder.getPreventPrimaryKeySelect())  &amp;&amp; !loadData)
                         {
                        } else
                           newDS = <xsl:value-of select="CLASS_NAME"/>DO.addToCache(this.dbtablename, (<xsl:value-of select="CLASS_NAME"/>DataStruct)newDO.originalData_get());
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
						try	{
							if ( user == null || newDO.hasQueryFindAccess( user	) )	{
</xsl:if>
								if (!hsResult.contains(newDO.get_Handle())) {
									hsResult.add(newDO.get_Handle());
    								if (resultCount	&gt;= readSkip)	{

                                        shell =	new	DOShell(newDO);
                                        results.DOs.add(shell);
                                    }
    								resultCount++;
    							}
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
							}
 				        }
 				        catch	( AccessException ae ) {
 							throw new DataObjectException("AccessException in runQuery()", ae);
						}
</xsl:if>
 				    } // while
 			    } // else (! unique)
</xsl:if>
				if ((results.DOs.size()==databaseLimit)&amp;&amp; bHasMoreResults) {
					resultSet.close();
					databaseLimitExceeded =	true;
				}
        if (maxDBrows > 0) {
					if (!bHasMoreResults) {
					  if ((databaseLimit &gt; 0) &amp;&amp; databaseLimit &lt; maxDBrows )
					      queryItem.setCompleteResult(true);
          }
        }
				else {
				  if (!bHasMoreResults)
				      queryItem.setCompleteResult(true);
				}
				Date stopQueryTime = new Date();
				queryTime =	stopQueryTime.getTime()	- startQueryTime.getTime();
 			    if (queryItem != null) {
				    queryItem.setTime((new Long(queryTime)).intValue());
     				    if (queryCachedItem != null) {
     				      if ( queryItem.isCompleteResult() || (queryCachedItem.isModifiedQuery() &amp;&amp; isOrderRelevant()) || (queryCachedItem.getResultNum() &lt; queryItem.getResultNum()) ) {
// tj 03.09.2004                    if ((!builder.isMultiTableJoin()) || <xsl:value-of select="CLASS_NAME"/>DO.isAllReadOnly() )
                    if (builder.isMultiTableJoin()){
                      ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).addMultiJoinQuery(queryItem);
                    }
                    else {
	   				            if (hitDb) {
	   				              ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).addComplexQuery(queryItem);
	   				            }
	   				            else {
	   				              ((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).addSimpleQuery(queryItem);
   				              }
                   	} // else from if (builder.isMultiTableJoin())
                  }
           			  else {
           			    if ( (queryCachedItem.getResultNum() &lt; (readSkip + databaseLimit) ) &amp;&amp; (queryCachedItem.getResultNum() &lt; maxDBrows) ) {
     				          queryCachedItem.setCompleteResult(true);
 				            }
     				      }
     				      if ( (queryItem.getResultNum() &lt; (readSkip + databaseLimit) ) &amp;&amp; (queryItem.getResultNum() &lt; maxDBrows) )
     				            queryItem.setCompleteResult(true);
     				    } // (queryCachedItem != null)
     			} // (queryItem != null)
				    int maxExecuteTime = <xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getTableConfiguration().getMaxExecuteTime();
				    if (maxExecuteTime &gt; 0 &amp;&amp; queryTime &gt; maxExecuteTime)
				        DODS.getLogChannel().write(Logger.WARNING, "sql = " + builder.getSQLwithParms()+" execute time = "+queryTime + "max table execute time = "+maxExecuteTime);
       	    }
       	    else { //	( userHitDb) || (!resultsFromQCache)
			    if (	<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).isFull() &amp;&amp; (<xsl:value-of select="CLASS_NAME"/>DO.isFullCacheNeeded)
			                     &amp;&amp; (!hitDb) &amp;&amp; (maxDBrows == 0) &amp;&amp; (databaseLimit == 0)
                              &amp;&amp; (readSkip == 0) &amp;&amp; !builder.isMultiTableJoin()) {
                 results = new QueryResult();
  			      if (readSkip&lt;cacheHits.size()) {
// 12.04.2004 tj					      Vector vect	= new Vector(cacheHits.values());
   					    results.DOs = new Vector();
   					    <xsl:value-of select="/TABLE/CLASS_NAME"/>DO DO = null;
   					    <xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct DS = null;
						String cachePrefix = getLogicalDatabase()+".";
            int i = 0;
            int resNumber = 0;
            Vector uniqueResults = new Vector();
						while (i &lt; cacheHits.size() ) { //  &amp;&amp; ((databaseLimit==0) || (results.DOs.size()&lt;=databaseLimit))
                                               //	 &amp;&amp; ((maxDBrows==0) || (i &lt; maxDBrows))
						    boolean findInTransactionCache = false;
						    DS = (<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct)cacheHits.get(i);
						    if(transaction!=null &amp;&amp; _tr_(transaction).getTransactionCache()!=null &amp;&amp; !loadData) {
                                                       DO = (<xsl:value-of select="/TABLE/CLASS_NAME"/>DO)_tr_(transaction).getTransactionCache().getDOByHandle(cachePrefix+DS.get_Handle());
                               if(DO != null)
                                findInTransactionCache = true;
                            }
                            if(!findInTransactionCache){
                                DO = (<xsl:value-of select="/TABLE/CLASS_NAME"/>DO)<xsl:value-of select="/TABLE/CLASS_NAME"/>DO.ceInternal(this.dbtablename, DS.get_OId(), transaction);
   					        }
   					        <xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
                            try	{
				                if ( user == null || DO.hasQueryFindAccess( user	) )	{
                            </xsl:if>
                             if (unique) {
                                if (!uniqueResults.contains(cachePrefix+DS.get_Handle())) {
                                  uniqueResults.add(cachePrefix+DS.get_Handle());
                                  //if (resNumber	&gt;= readSkip ){
                                  results.DOs.add(DO);
                                  //}
                                  resNumber++;
                                }
                             }
                             else {
                                //if (resNumber	&gt;= readSkip ){
                                results.DOs.add(DO);
                                //}
                                resNumber++;
                             }
<xsl:if	test="DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'">
                			    }
                			}
					        catch	( AccessException ae ) {
						      throw new DataObjectException("AccessException in runQuery()", ae);
					        }
       </xsl:if>
   					    i++;
   					    }
						readDOs	= true;
					}

 /*         if ((databaseLimit != 0) &amp;&amp; (results.DOs.size() == databaseLimit+1)) {
            databaseLimitExceeded = true;
            results.DOs.remove(databaseLimit);
          }
*/
				}  //if full

   	        } //	( userHitDb) || (!resultsFromQCache)
// end of WebDocWf extension
			if (results != null) {  // tj 01.02.2004
			if ( results.DOs.size()	&gt; 1 &amp;&amp; uniqueInstance )
				throw new NonUniqueQueryException("Too many	rows returned from database" );
			DOs	= new <xsl:value-of	select="CLASS_NAME"/>DO	[ results.DOs.size() ];
			<xsl:value-of select="CLASS_NAME"/>DataStruct orig;
			if (readDOs) {
				for	( int i	= 0; i &lt;	results.DOs.size();	i++	) {
					DOs[i] = (<xsl:value-of	select="CLASS_NAME"/>DO)results.DOs.elementAt(i);
   		        }
   		    }
			else {
			    for	( int i	= 0; i &lt;	results.DOs.size();	i++	) {
				    DOs[i] = (<xsl:value-of	select="CLASS_NAME"/>DO)((DOShell)results.DOs.elementAt(i)).dataObject;
                }
            }
			arrayIndex = 0;
			}
			else {
				DOs	= new <xsl:value-of	select="CLASS_NAME"/>DO	[0];
			}
			if (isQueryByOId &amp;&amp;	!hasNonOidCond) {
			   <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).incrementQueryByOIdNum();
			   <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).updateQueryByOIdAverageTime((new Long(queryTime)).intValue(),1);
			}
			else {
			   <xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).incrementQueryNum();
				<xsl:value-of select="CLASS_NAME"/>DO.get_statistics(this.dbtablename).updateQueryAverageTime((new Long(queryTime)).intValue());
			}
		} catch	( SQLException se )	{
			if (null ==	se.getSQLState() ) {
				throw new DataObjectException("Unknown SQLException", se );
			}
			if ( se.getSQLState().startsWith("02") &amp;&amp; se.getErrorCode()	== 100 ) {
				throw new DataObjectException("Update or delete	DO is out of synch", se	);
			} else if (		se.getSQLState().equals("S1000") &amp;&amp;		se.getErrorCode() == -268) {
				throw new DataObjectException("Integrity constraint	violation",	se );
			} else {
				throw new DataObjectException( "Data Object	Error",	se );
			}
		} catch	( ObjectIdException	oe ) {
			throw new DataObjectException( "Object ID Error", oe );
		} catch	( DatabaseManagerException de )	{
			throw new DataObjectException( "Database connection	Error",	de );
		}
		finally {
                        if ( null != dbQuery )dbQuery.release();
		}
	}

	/**
	 * Limit the number	of rows	(DOs) returned.
	 * NOTE: When setting a	limit on rows returned by a	query,
	 * you usually want	to use a call to an	addOrderBy method
	 * to cause	the	most interesting rows to be	returned first.
<xsl:if	test="DO_IS_OID_BASED='true'">
	 * However,	the	DO cache does not yet support the Order	By operation.
	 * Using the addOrderBy	method forces the query	to hit the database.
	 * So, setMaxRows also forces the query	to hit the database.
</xsl:if>
	 *
	 * @param maxRows Max number of	rows (DOs) returned.
	 *
	 * @exception DataObjectException If a database	access error occurs.
	 * @exception NonUniqueQueryException If too many rows were	found.
	 */
	public void	setMaxRows(	int	maxRows	)
	throws DataObjectException,	NonUniqueQueryException	{

    maxDBrows = maxRows;
		builder.setMaxRows(	maxRows	);
	}

	/**
	 * Return limit of rows	(DOs) returned.
	 * @return Max number of	rows (DOs) returned.
	 *
	 */
	public int	getMaxRows() {
    return maxDBrows;
	}
	/**
	 * Returns attribute orderRelevant.
	 *
	 * @return true if order of query results is relavant, otherwise false.
	 */
	public boolean	isOrderRelevant() {
    return orderRelevant;
	}

	/**
	 * Sets attribute orderRelevant.
	 *
	 * @param newOrderRelevant new value of attribute orderRelavant.
	 */
	public void setOrderRelevant(boolean newOrderRelevant) {
    orderRelevant = newOrderRelevant;
	}


	/**
	 * Return QueryResult with read DOs or DataStructs from caches.
	 *
	 * @param result QueryResult object with result oids.
	 * @return QueryResult object with filled DOs or DataStructs that are found
	 * in the cache.
	 *
	 * @exception DataObjectException If a database	access error occurs.
	 */
	public QueryResult getCachedResults(QueryResult result) throws DataObjectException	{
	    Vector tempVec = result.DOs;
	    if (tempVec == null)
	        return null;
	    result.DOs = new Vector();
	    result.lazy = new Vector();
	    DOShell shell = null;
	    <xsl:value-of select="CLASS_NAME"/>DO cacheDO = null;
	    <xsl:value-of select="CLASS_NAME"/>DataStruct cacheDS = null;
	    String handle = "";
	    String cachePrefix=getLogicalDatabase()+".";

        for(int i=0; i &lt; tempVec.size(); i++) {

            if(tempVec.get(i)!=null) {
                cacheDO = null;
                cacheDS = null;
                handle=(String)tempVec.get(i);
                shell = new DOShell(handle);
                if(transaction!=null &amp;&amp; _tr_(transaction).getTransactionCache()!=null &amp;&amp; !loadData) {
                    try {
                        cacheDO = (<xsl:value-of select="CLASS_NAME"/>DO)_tr_(transaction).getTransactionCache().getDOByHandle(cachePrefix+handle);
                    } catch (Exception e) {
                    }
                }
                if (cacheDO == null){
                    cacheDS = (<xsl:value-of select="CLASS_NAME"/>DataStruct)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getDataStructByHandle(cachePrefix+handle);
                    if(cacheDS!=null) {
                        try {
                            cacheDO = (<xsl:value-of select="CLASS_NAME"/>DO)<xsl:value-of select="CLASS_NAME"/>DO.ceInternal(this.dbtablename, cacheDS.get_OId(), transaction);
       		        } catch (Exception e) {
                    }
                }
               }
                if (cacheDO == null){
                    result.lazy.add(shell);
                }
                else {
                    shell.dataObject = cacheDO;
                }
                result.DOs.add(shell);
          }
        } //for
		return result;
	}



	/**
	 * Return array	of DOs constructed from	ResultSet returned by query.
	 *
	 * @return Array of	DOs	constructed	from ResultSet returned	by query.
	 *
	 * @exception DataObjectException If a database	access error occurs.
	 * @exception NonUniqueQueryException If too many rows were	found.
	 */
	public <xsl:value-of select="CLASS_NAME"/>DO[] getDOArray()
	throws DataObjectException,	NonUniqueQueryException	{
		if ( needToRun )
			runQuery();
		return DOs;
	}

	/**
	 * Return successive DOs from array	built from ResultSet returned by query.
	 *
	 * @return DOs from	array built	from ResultSet returned	by query.
	 *
	 * @exception DataObjectException If a database	access error occurs.
	 * @exception NonUniqueQueryException If too many rows were	found.
	 */
	public <xsl:value-of select="CLASS_NAME"/>DO getNextDO()
	throws DataObjectException,	NonUniqueQueryException	{
		if ( needToRun )
			runQuery();
		if ( null == DOs ) {
			/* This	should never happen.
			 * runQuery() should either	throw an exception
			 * or create an	array of DOs, possibly of zero length.
			 */
			return null;
		}
		if ( arrayIndex	&lt; DOs.length	)
			return DOs[	arrayIndex++ ];
		return null;
	}

<xsl:if	test="DO_IS_OID_BASED='true'">
	/**
	 * Set the OID to query.
	 * WARNING!	 This method assumes that table	&lt;CODE&gt;<xsl:value-of select="TABLE_NAME"/>&lt;/CODE&gt;
	 * has a column	named &lt;CODE&gt;"oid"&lt;/CODE&gt;.
	 * This	method is called from the DO classes to	retrieve an	object by id.
	 *
	 * @param oid The object id	to query.
	 */
	public void	setQueryOId(ObjectId oid) {
		// Remove from cacheHits any DO	that does not meet this
		// setQuery	requirement.
		String handle =	getLogicalDatabase()+ "." +oid.toString();
		if (<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).isFull() &amp;&amp; (<xsl:value-of select="CLASS_NAME"/>DO.isFullCacheNeeded)) {
// 12.04.2004 tj new
      <xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct DS = null;
      for ( int i = 0; i &lt; cacheHits.size(); i++ ) {
	      DS = (<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct)cacheHits.elementAt( i );
	      String cacheHandle = null;
	      try{
	        cacheHandle = DS.get_CacheHandle();
	      }
	      catch (Exception e){}
	      if (cacheHandle != null) {
  	      if ( ! cacheHandle.equals( handle ) )
  		      cacheHits.removeElementAt( i-- );
  		  }
      } // for
    } // full
			if (isQueryByOId) {	// query by	OId	already	has	been invoked

			  hasNonOidCond = true; // this is not simple query by oid: has at least two conditions for oids


			} else { // (isQueryByOid)
				currentHandle =	 handle;
		}
				isQueryByOId = true;
		try {
			Condition cond = new Condition(<xsl:value-of select="/TABLE/CLASS_NAME"/>DataStruct.COLUMN_OID,handle,"=");
			queryItem.addCond(cond);
		}
		catch (Exception e){
			DODS.getLogChannel().write(Logger.DEBUG," <xsl:value-of select="CLASS_NAME"/>Query class\n :"+" condition are not added");
		}
		// Also	prepare	the	SQL	needed to query	the	database
		// in case there is	no cache, or the query involves	other tables.
		builder.addWhere( <xsl:value-of	select="CLASS_NAME"/>DO.getPrimaryKey(this.dbtablename),	oid.toBigDecimal(),	QueryBuilder.EQUAL );
	}

	/**
	 * Set the object handle to	query.
	 * This	is a variant of	setQueryOId().
	 *
	 * @param handle The string	version	of the id to query.
	 * @exception ObjectIdException
	 */
	public void	setQueryHandle(String handle)
	throws ObjectIdException {
		ObjectId oid = new ObjectId( handle	);
		setQueryOId( oid );
	}
</xsl:if>

	/**
	 * Set "unique instance" assertion bit.
	 * The first call to the next()	method will	throw an exception
	 * if more than	one	object was found.
	 */
	public void	requireUniqueInstance()	{
		uniqueInstance = true;
	}

	/**
	 * Set loadData parameter. if parameter is set to true, Query select t.* is performed.
	 * @param newValue boolean (true/false)
	 */
	public void	setLoadData(boolean newValue)	{
		loadData = newValue;
	}

	/**
	 * Return true if Query is prepared for select t1.* statement. Otherwise return false.
	 * @return boolean (true/false)
	 */
	public boolean	getLoadData()	{
		if(loadData)
		   return true;
		else
		   return ((<xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isLazyLoading()) || isCaching());
	}

	/**
	 * Reset the query parameters.
	 */
	public void	reset()	{
      if(<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).isFull() &amp;&amp; (<xsl:value-of select="CLASS_NAME"/>DO.isFullCacheNeeded)) {
      if(<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getTableConfiguration().getFullCacheCountLimit() &gt; 0){
		     if(<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getCacheAdministration(CacheConstants.DATA_CACHE).getCacheSize() &gt; <xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getTableConfiguration().getFullCacheCountLimit())
		         <xsl:value-of select="CLASS_NAME"/>DO.isFullCacheNeeded = false;
	   }
	  }
		if(<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).isFull() &amp;&amp; (<xsl:value-of select="CLASS_NAME"/>DO.isFullCacheNeeded)) {

		  Map m = null;
		  synchronized (<xsl:value-of	select="CLASS_NAME"/>DO.getCache(this.dbtablename)) {
			  m = <xsl:value-of	select="CLASS_NAME"/>DO.getCache(this.dbtablename).getCacheContent();
			  if(m!=null)
		         cacheHits =	new Vector(m.values());
		     else
		         cacheHits = new Vector();
		  }
		}
		DOs	= null;
		uniqueInstance	= false;
		needToRun		= true;
		isQueryByOId = false;
		hasNonOidCond	= false;
		loadData=false;
		builder.reset();
		if (<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getLevelOfCaching()	== CacheConstants.QUERY_CACHING) {
			queryItem =	((QueryCache)<xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename)).newQueryCacheItemInstance(logicalDatabase);
		}
	}

	/**
	 * Return the appropriate QueryBuilder flag	for	selecting
	 * exact matches (SQL '=') or inexact matches (SQL 'matches').
	 *
	 * @param exact Flag that indicates whether	it is exact match (SQL '=') or
	 * inexact matches (SQL 'matches').
	 * @return boolean True	if it is exact match, otherwise	false.
	 *
	 */
	private	boolean	exactFlag( boolean exact ) {
		return exact ? QueryBuilder.EXACT_MATCH	: QueryBuilder.NOT_EXACT;
	}

	//
	// Implementation of Query interface
	//

	/**
	 * Method to query objects from	the	database.
	 * The following call in runQuery()
	 *			dbQuery.query( this	);
	 * causes the dbQuery object to	invoke
	 *		executeQuery()
	 *
	 * @param conn Handle to database connection.
	 *
	 * @return ResultSet with the results of the query.
	 *
	 * @exception java.sql.SQLException	If a database access error occurs.
	 */
	public ResultSet executeQuery(DBConnection conn)
	throws SQLException		{
		builder.setCurrentFetchSize(iCurrentFetchSize);
		builder.setCurrentQueryTimeout(iCurrentQueryTimeout);
		resultSet =	builder.executeQuery( conn );
		return resultSet;
	}

	/**
	 * WARNING!	 This method is	disabled.
	 * It's	implementation is forced by	the	Query interface.
	 * This	method is disabled for 2 reasons:
	 * 1)  the getDOArray()	and	getNextDO()	methods	are	better
	 *	   because they	return DOs instead of JDBC objects.
	 * 2)  the ceInternal()	method throws an exception
	 *	   that	we cannot reasonably handle	here,
	 *	   and that	we cannot throw	from here.
	 *
	 * @param rs JDBC result set from which	the	next object
	 *	 will be instantiated.
	 * @return Next result.
	 *
	 * @exception java.sql.SQLException
	 *	 If	a database access error	occurs.
	 * @exception com.lutris.appserver.server.sql.ObjectIdException
	 *	 If	an invalid object id was queried from the database.
	 */
	public Object next(ResultSet rs)
	throws SQLException, ObjectIdException {
		// TODO: It	would be nice to throw an unchecked	exception here
		// (an exception that extends RuntimeException)
		// that	would be guaranteed	to appear during application testing.
		throw new ObjectIdException("next()	should not be used.	 Use getNextDO() instead." );
		//return null;
	}


	// WebDocWf	extension for extended wildcard	support
	// The following lines have	been added:
	/**
	 * Convert a String	with user wildcards	into a string with DB wildcards
	 *
	 * @param userSearchValue The string with user wildcards
	 *
	 * @return The string with DB wildcards
	 *
	 * WebDocWf	extension
	 *
	 */
	public String convertUserSearchValue( String userSearchValue ) {
		return builder.convertUserSearchValue( userSearchValue );
	}

	/**
	 * Check whether a string contains DB wildcards
	 *
	 * @param dbSearchValue	The	string with	possible DB	wildcards
	 *
	 * @return Whether a string	contains DB	wildcards
	 *
	 * WebDocWf	extension
	 *
	 */
	public boolean containsWildcards( String dbSearchValue ) {
		return builder.containsWildcards( dbSearchValue	);
	}
	// end of WebDocWf extension for extended wildcard support

	// WebDocWf	extension for query	row	counter
	// The following lines have	been added:
	/**
	 * Get the rowcount	of the query
	 * If possible,	do it without reading all rows
	 *
	 * @return The row count
	 * @exception NonUniqueQueryException
	 * @exception DataObjectException
	 * @exception SQLException
	 * @exception DatabaseManagerException
	 *
	 * WebDocWf	extension
	 *
	 */
	public int getCount()
	throws NonUniqueQueryException,	DataObjectException, SQLException, DatabaseManagerException		{
		int	rowCount=0;
		if (needToRun &amp;&amp; databaseLimit==0<xsl:if test="TEMPLATE_SET='webdocwf' and	DO_IS_OID_BASED='true' and /TABLE/GENERATE_SECURE='true'"> &amp;&amp;	user==null</xsl:if>) {
			rowCount = selectCount();
		} else {
			if (needToRun) runQuery();
				rowCount = DOs.length;
		}
		return rowCount;
	}

    /**
     * Set reference objects.
     * @param queryRefs Reference objects.
     */
    protected void setRefs(HashMap queryRefs) {
        refs = queryRefs;
    }

    /**
     * set the current cursor type - overrides default value from dbVendorConf file.
     * @param resultSetType a result set type; one of ResultSet.TYPE_FORWARD_ONLY, ResultSet.TYPE_SCROLL_INSENSITIVE, or ResultSet.TYPE_SCROLL_SENSITIVE.
     * @param resultSetConcurrency a concurrency type; one of ResultSet.CONCUR_READ_ONLY or ResultSet.CONCUR_UPDATABLE.
     * @return the current queryTimeout;
     */
    public void set_CursorType(int resultSetType, int resultSetConcurrency) {
         builder.setCursorType(resultSetType,resultSetConcurrency);
     }

    /**
     * Set fetch size for this query
     * @param iCurrentFetchSizeIn Query fetch size.
     */
    public void set_FetchSize (int iCurrentFetchSizeIn) {
         iCurrentFetchSize = iCurrentFetchSizeIn;
     }

   /**
    * reads the current fetchsize for this query
    * @return the current fetchsize; if -1 the no fetchsize is defined, defaultFetchSize will be use if defined
    */
   public int get_FetchSize() {
       return (iCurrentFetchSize &lt; 0)? builder.getDefaultFetchSize() : iCurrentFetchSize;
   }

   /**
    * Reads the current queryTimeout for this query.
    * @return the current queryTimeout;
    */
   public int get_QueryTimeout() {
       return iCurrentQueryTimeout;
   }

   /**
    * Sets the current queryTimeout for this query.
    * @param iQueryTimeoutIn current queryTimeout.
    */
   public void set_QueryTimeout(int iQueryTimeoutIn) {
       iCurrentQueryTimeout =  (iCurrentQueryTimeout &lt; 0)? builder.getDefaultQueryTimeout() : iCurrentQueryTimeout;
   }


	/**
	 * Get the rowcount	of the query by	using count(*) in the DB
	 *
	 * @return The row count.
	 * @exception SQLException
	 * @exception DatabaseManagerException
	 *
	 * WebDocWf	extension
	 *
	 */
	public int selectCount()
	throws SQLException, DatabaseManagerException {
		int	rowCount=0;
		String tempClause =	builder.getSelectClause();
		builder.setSelectClause(" count(*) as \"counter\" ");
		DBQuery	dbQuery;
<xsl:if	test="DO_IS_MULTIDB_BASED='true'"><xsl:if test="not(/TABLE/GENERATE_DIRTY='Omit')">
		if ((transaction == null)&amp;&amp;(logicalDatabase	!= null))
			dbQuery	= <xsl:value-of	select="CLASS_NAME"/>DO.createQuery(logicalDatabase);
		else</xsl:if></xsl:if>
			dbQuery	= <xsl:value-of	select="CLASS_NAME"/>DO.createQuery(transaction);
		dbQuery.query(this);
		resultSet.next();
		rowCount=resultSet.getInt("counter");
		resultSet.close();
      if (transaction == null)
		    dbQuery.release();
		builder.close();
		resultSet =	null;
		builder.setSelectClause(tempClause);
		return rowCount;
	}
	// end of WebDocWf extension for query row counter

	/**
	 * Return true if some caching for this table is enabled.
	 * @return (true/false)
	 */

   private boolean isCaching() {
      double cachePercentage = <xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getCachePercentage();
      double usedPercentage = 0;
      if(cachePercentage == -1)
         return false;
      else if(cachePercentage == 0)
         return true;
      else {
       try {
          usedPercentage = <xsl:value-of select="CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getStatistics().getCacheStatistics(CacheConstants.DATA_CACHE).getUsedPercents();
        } catch (Exception ex) {
               return false;
        }
         if(usedPercentage > <xsl:value-of select="CLASS_NAME"/>DO.getCache(this.dbtablename).getCachePercentage()*100)
            return true;
         else
            return false;
      }
   }

   private static CachedDBTransaction _tr_(DBTransaction dbt) {
      return (CachedDBTransaction)dbt;
   }



    private String fixCaseSensitiveCondition(String cmp_op){
        if(<xsl:value-of select="/TABLE/CLASS_NAME"/>DO.getConfigurationAdministration(this.dbtablename).getTableConfiguration().isCaseSensitive()){
			if( cmp_op.equals(builder.CASE_INSENSITIVE_CONTAINS) ){
				return builder.CASE_SENSITIVE_CONTAINS;
			}else if( cmp_op.equals( builder.CASE_INSENSITIVE_STARTS_WITH ) ){
				return builder.CASE_SENSITIVE_STARTS_WITH;
			}else if( cmp_op.equals(builder.CASE_INSENSITIVE_ENDS_WITH) ){
				return builder.CASE_SENSITIVE_ENDS_WITH;
			}else if( cmp_op.equals(builder.CASE_INSENSITIVE_EQUAL) ){
				return builder.EQUAL;
			}else if( cmp_op.equals(builder.CASE_INSENSITIVE_MATCH) ){
				return builder.CASE_SENSITIVE_MATCH;
			}else if( cmp_op.equals(builder.USER_CASE_INSENSITIVE_MATCH) ){
				return builder.USER_CASE_SENSITIVE_MATCH;
			}
        }
        return cmp_op;
    }



<xsl:if test="/TABLE/MASS_UPDATES='true' or /TABLE/MASS_DELETES='true'">
    /**
     *
     */
    Vector collectOIds() throws SQLException {
        Vector res = new Vector();
        String tempClause = builder.getSelectClause();
        builder.setSelectClause(" <xsl:value-of select="TABLE_NAME"/>."
                                + <xsl:value-of select="CLASS_NAME"/>DO.get_OIdColumnName()+" ");
        DBQuery dbQuery = null;

        try {
            dbQuery = <xsl:value-of select="CLASS_NAME"/>DO.createQuery(transaction);
            dbQuery.query(this);
            while (resultSet.next()) {
                String _OId = logicalDatabase +"."
                    + resultSet.getBigDecimal
                      (<xsl:value-of select="CLASS_NAME"/>DO.get_OIdColumnName()).toString();
                res.add(_OId);
            }
        } catch (Exception dme) {
            dme.printStackTrace();
            throw new SQLException("collect OId failed");//.initCause(dme);
        } finally {
            if (null != resultSet)
                resultSet.close();
            if (transaction == null)
                dbQuery.release();
            builder.close();
            resultSet = null;
            builder.setSelectClause(tempClause);
        }
        return res;
    }
</xsl:if>
</xsl:template>
</xsl:stylesheet>
