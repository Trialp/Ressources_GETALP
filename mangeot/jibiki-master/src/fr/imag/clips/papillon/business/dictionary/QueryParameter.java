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
 * Revision 1.5  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.4  2006/08/10 22:17:12  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.3  2006/04/24 13:43:29  fbrunet
 * Add new class ViewQueryResult : allow to use one class to create result display in advancedSearch and EditEntryInit (like advancedQueryForm)
 * Improve result display : view n results per page
 *
 * Revision 1.2  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.1.4.1  2005/08/31 15:01:39  serasset
 * Applied modifications done on the LEXALP_1_0 branch to updated sources of the
 * trunk to create a new updated LEXALP_1_1 branch.
 *
 * Revision 1.1.2.1  2005/07/22 13:28:32  serasset
 * Modified EditEntryInit for Lexalp. It now serves as a main page for db maintenance.
 * Added a function to get url for QueryParameter.
 * Modified the way xslsheets are handled in order to allow several xslsheet with the same name, different dicts.
 *
 * Revision 1.1  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 *-----------------------------------------------
 */

package fr.imag.clips.papillon.business.dictionary;

/* standards imports */

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


/**
* A QueryParameter is a business object passed when querying dictionaries.
*/
public class QueryParameter {
    
    // Parameters
    protected ArrayList dictionaries; // Collection of Dictionary
    protected ArrayList targets;   // Collection of String
    protected ArrayList criteria;
    protected int offset; 
    protected int limit; 
    protected String xsl;
    
    public QueryParameter() {
        
        // Initialize
        dictionaries = new ArrayList();
        targets = new ArrayList();
        criteria = new ArrayList();
        offset = 0; 
        limit = 0; 
    }
    
    public Collection getDictionaries() {
        return dictionaries;
    }
    
    public void setDictionaries(Collection dicts) {
        if (dicts != null) {
            dictionaries = new ArrayList(dicts);
        } else {
            dictionaries = new ArrayList();
        }
    }
    
    public List getCriteria() {
        return criteria;
    }
    
    public void setCriteria(List newCriteria) {
        if (newCriteria != null) {
            criteria = new ArrayList(newCriteria);
        } else {
            criteria = new ArrayList();
        }
    }

    public int getOffset() {
        return offset;
    }
    
    public String getOffsetString() {
        return Integer.toString(offset);
    }
    
    public void setOffset(int newOffset) {
        offset = newOffset;
    }

    public int getLimit() {
        return limit;
    }
    
    public String getLimitString() {
        return Integer.toString(limit);
    }
    
    public void setLimit(int newLimit) {
        limit = newLimit;
    }

    public String getXsl() {
        return xsl;
    }
    
    public void setXsl(String newXsl) {
        xsl = newXsl;
    }

    public Collection getTargets() {
        return targets;
    }
    
    public void setTargets(Collection newTargets) {
        if (newTargets != null) {
            targets = new ArrayList(newTargets);
        } else {
            targets = new ArrayList();
        }
    }
	
    public QueryParameter duplicate() {
            
        //
        QueryParameter duplicate = new QueryParameter();
        
        // NOT clone() ATTENTION NO DEEP COPY
        // JUST Create the SAME queryParamater. It use to create next and previous query request !
        duplicate.setDictionaries(dictionaries);
        duplicate.setCriteria(criteria);
        duplicate.setOffset(offset);
        duplicate.setLimit(limit);
        duplicate.setXsl(xsl);
        duplicate.setTargets(targets);
        
        return duplicate;
    }
}
