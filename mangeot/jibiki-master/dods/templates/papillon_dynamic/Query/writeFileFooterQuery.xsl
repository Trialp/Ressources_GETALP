<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xalan="http://xml.apache.org/xslt"
                xmlns:stu="org.ejen.ext.StringUtil"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
    <xsl:template name="fileFooter">
    <xsl:value-of select="FK_SETQUERY_METHODS"/>
    /**
     * Returns the &lt;code&gt;QueryBuilder&lt;/code&gt; that this &lt;code&gt;<xsl:value-of select="CLASS_NAME"/>Query&lt;/code&gt;
     * uses to construct and execute database queries.
     * &lt;code&gt;<xsl:value-of select="CLASS_NAME"/>Query.setQueryXXX&lt;/code&gt; methods use 
     * the &lt;code&gt;QueryBuilder&lt;/code&gt; to
     * append SQL expressions to the &lt;code&gt;"WHERE"&lt;/code&gt; clause to be executed.
     * The &lt;code&gt;QueryBuilder.addEndClause method.&lt;/code&gt; can be used to
     * append freeform SQL expressions to the &lt;code&gt;WHERE&lt;/code&gt; clause,
     * e.g. "ORDER BY name".
     *
     * &lt;b&gt;Notes regarding cache-enabled DO classes:&lt;/b&gt;
     * DO classes can be cache-enabled.  
     * If when using a &lt;code&gt;<xsl:value-of select="CLASS_NAME"/>Query&lt;/code&gt;, the application developer
     * &lt;b&gt;does not call&lt;/b&gt; &lt;code&gt;getQueryBuilder&lt;/code&gt;,
     * then &lt;code&gt;<xsl:value-of select="CLASS_NAME"/>Query.setQueryXXX&lt;/code&gt; methods 
     * simply prune the DO cache and return the remaining results.
     * However, a &lt;code&gt;QueryBuilder&lt;/code&gt; builds
     * &lt;CODE&gt;SELECT&lt;/CODE&gt; statements for execution by the actual database,
     * and never searches the built-in cache for the DO.
     * So, if the DO class is cache-enabled, and &lt;code&gt;getQueryBuilder&lt;/code&gt;
     * is called, this &lt;CODE&gt;<xsl:value-of select="CLASS_NAME"/>Query&lt;/CODE&gt; object ignores the cache 
     * and hits the actual database.
     * 
     * @return QueryBuilder that is used to construct and execute database queries.   
     */
    public QueryBuilder getQueryBuilder() {
         hitDb = true;
        return builder;
    }

    /**
     * Insert an &lt;CODE&gt;OR&lt;/CODE&gt; between &lt;CODE&gt;WHERE&lt;/CODE&gt; clauses.
     * Example:  find all the persons named Bob or Robert:
     * &lt;CODE&gt;
     *    PersonQuery pq = new PersonQuery();
     *    pq.setQueryFirstName( "Bob" );
     *    pq.or();
     *    pq.setQueryFirstName( "Robert" );
     * &lt;/CODE&gt;
     * 
     * Note:  Calls to &lt;CODE&gt;setQueryXxx&lt;/CODE&gt; methods
     * are implicitly &lt;CODE&gt;AND&lt;/CODE&gt;ed together,
     * so the following example will always return nothing:
     * &lt;CODE&gt;
     *    PersonQuery pq = new PersonQuery();
     *    pq.setQueryFirstName( "Bob" );
     *    // AND automatically inserted here.
     *    pq.setQueryFirstName( "Robert" );
     * &lt;/CODE&gt;
<xsl:if test="DO_IS_OID_BASED='true'">
     * 
     * NOTE: The DO cache does not yet support the OR operator.
     * Using the or() method forces the query to hit the database.
</xsl:if>
     * 
     * @see com.lutris.dods.builder.generator.query.QueryBuilder QueryBuilder to construct more elaborate queries.
     * author Jay Gunter
     */
    public void or() {
        hitDb = true;
        builder.addWhereOr();
    }

    /**
     * Place an open parenthesis in the &lt;CODE&gt;WHERE&lt;/CODE&gt; clause.
     * Example usage:  find all the Bobs and Roberts who are 5 or 50 years old:
     * &lt;CODE&gt;
     *    PersonQuery pq = new PersonQuery();
     *    pq.openParen();
     *       pq.setQueryFirstName( "Bob" );
     *       pq.or();
     *       pq.setQueryFirstName( "Robert" );
     *    pq.closeParen();
     *    // AND automatically inserted here.
     *    pq.openParen();
     *       pq.setQueryAge( 5 );
     *       pq.or();
     *       pq.setQueryAge( 50 );
     *    pq.closeParen();
     * &lt;/CODE&gt;
    <xsl:if test="DO_IS_OID_BASED='true'">
     * 
     * NOTE: The DO cache does not yet support the Open Paren operator.
     * Using the openParen() method forces the query to hit the database.
    </xsl:if>
     * 
     * @see com.lutris.dods.builder.generator.query.QueryBuilder QueryBuilder to construct more elaborate queries.
     * author Jay Gunter
     */
    public void openParen() {
        hitDb = true;
        builder.addWhereOpenParen();
    }

    /**
     * Place a closing parenthesis in the &lt;CODE&gt;WHERE&lt;/CODE&gt; clause.
    <xsl:if test="DO_IS_OID_BASED='true'">
     * 
     * NOTE: The DO cache does not yet support the Close Paren operator.
     * Using the closeParen() method forces the query to hit the database.
    </xsl:if>
     * 
     * @see <xsl:value-of select="CLASS_NAME"/>Query#openParen openParen
     * author Jay Gunter
     */
    public void closeParen() {
        hitDb = true;
        builder.addWhereCloseParen();
    }
}
</xsl:template>
</xsl:stylesheet>
