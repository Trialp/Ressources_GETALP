/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.4.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.4  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.3  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.2.4.1  2005/12/02 10:04:09  fbrunet
 * Add Pre/Post edition processing
 * Add index reconstruction
 * Add new query request
 * Add fuzzy search
 * Add new contribution administration
 * Add xsl transformation volume
 *
 * Revision 1.2  2005/06/17 12:38:56  mangeot
 * Changed lexiesCollection into lexiesHashtable in order to implement the getDirectTranslations
 *
 * Revision 1.1  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 *-----------------------------------------------
 */

package fr.imag.clips.papillon.business.dictionary;

/* standards imports */

import java.util.HashMap;
import java.util.ArrayList;
import java.util.Vector;
import fr.imag.clips.papillon.business.PapillonBusinessException;

/**
 * A QueryResult is a business object return when querying dictionaries.
 * Its form depend on the kind of dictionary that is queried. 
 *    If the dictionary is monolingual, it only contain one entry. 
 *    If the dictionary is bilingual or is a fork dictionary, it only contains one bi/multi lingual entry.
 *    If the dictionary is a pivot dictionary, it contains:
 *       - one entry (the source entry)
 *       - ONE axie (which is connected to the queried entry).
 *       - a collection of all lexies DIRECTLY connected to the axie in requested target languages.
 *    If the result has been obtained by reverse lookup, it states it.
 */
public class QueryResult {
    
    // Kind of result
    public static final int UNKNOWN = 0;
    public static final int UNIQUE_RESULT = 1;
    public static final int REVERSE_UNIQUE_RESULT = 2;
    public static final int AXIE_COLLECTION_RESULT = 3;
	public static final int DIRECT_TRANSLATIONS_RESULT = 4; // FIXME: new Hashtable lexiesDirectHashtable
    
    protected int resultKind;
	//protected VolumeEntry sourceEntry = null;
	protected ArrayList sourceEntries = new ArrayList();
    protected VolumeEntry resultAxie = null;
    protected HashMap lexiesHashMap = new HashMap();
	
    public QueryResult() {
        this.resultKind = UNKNOWN;
    }

    // new QueryResult initialized from a prototype.
    public QueryResult(QueryResult qr) {
        this.resultKind = qr.getResultKind();
        this.sourceEntries = qr.getSourceEntries();
        this.resultAxie = qr.getResultAxie();
        this.lexiesHashMap = qr.getLexiesHashMap();
    }
    
    
    public QueryResult(int kind, VolumeEntry source) {
        this.resultKind = kind;
        this.sourceEntries.add(source);
    }
    
    public QueryResult(int kind, VolumeEntry source, VolumeEntry axie) {
        this(kind, source);
        this.resultAxie = axie;
    }
    
    public QueryResult(int kind, VolumeEntry source, VolumeEntry axie, HashMap lexies) {
        this(kind, source, axie);
        this.lexiesHashMap = lexies;
    }
    
    public void addSourceEntry(VolumeEntry ve) {
        this.sourceEntries.add(ve);
    }
    
    /**
     * returns the unique source Entry of the Query Result.
     * @throws PapillonBusinessException if there are more than one source entry. 
     */
    public VolumeEntry getSourceEntry() throws PapillonBusinessException {
        if (this.sourceEntries.size() != 1) throw new PapillonBusinessException("Multiple source Entry in this query.");
        return (VolumeEntry)this.sourceEntries.get(0);
    }

    public VolumeEntry getFirstSourceEntry() {
        return (VolumeEntry)this.sourceEntries.get(0);
    }

    public ArrayList getSourceEntries() {
        return this.sourceEntries;
    }
    
    public void setResultKind(int rk) {
        this.resultKind = rk;
    }
    
    public int getResultKind() {
        return this.resultKind;
    }

    public void setResultAxie(VolumeEntry axies) {
        this.resultAxie = axies;
    }
    
    public VolumeEntry getResultAxie() {
        return this.resultAxie;
    }

    public void setLexiesHashMap(HashMap lexies) {
        this.lexiesHashMap = lexies;
    }
    
    public HashMap getLexiesHashMap() {
        return this.lexiesHashMap;
    }
        
    public java.util.Collection getLexiesCollection() {
        return this.lexiesHashMap.values();
    }
    
    
    
}
