/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.4  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.3  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.2.2.1  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.logs;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.CurrentDBTransaction;

import com.lutris.appserver.server.sql.DatabaseManagerException;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.dods.builder.generator.query.DataObjectException;

/**
 * Represents a Query Log. 
 */
public class QueryLog {


    public static final int MAX_LOGGED_RESULTS = 10;

    public static final String STORE_STRING = "Papillon.Logging.StoreQueryLogs";

    /**
     * The DO of the QueryLog.
     */
    protected QueryLogDO myDO = null;

    /**
     * The public constructor.
     */
    public QueryLog() throws PapillonBusinessException {
        try {
            this.myDO = QueryLogDO.createVirgin(CurrentDBTransaction.get());  
        } catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error creating empty QueryLog", ex);
        } catch(ObjectIdException ex) {
            throw new PapillonBusinessException("Error creating empty QueryLog", ex);
        }
    }

    /** The protected constructor
     *
     * @param theDisc. The data object of the QueryLog.
     */
    protected QueryLog(QueryLogDO theQueryLog) 
        throws PapillonBusinessException  {
        this.myDO = theQueryLog;
    }

    public boolean isEmpty() {
        return (this.myDO==null) ;
    }


    /**
     * Gets the table name of the QueryLog
     *
     * @return the table name.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    /* FIXME: the method myDO.getTableName is protected, therefore, I can't
       use it. So I hard code the name of the database table ! 
    If one day, we stop to use dods ...*/
    public String getTableName()
        throws PapillonBusinessException {
	/*   try {
	      return this.myDO.getTableName();
        } catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error getting QueryLogs's table name", ex);
        }*/
            return "querylogs";
    }   


    /**
     * Gets the object id for the QueryLog
     *
     * @return the object id.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getHandle()
        throws PapillonBusinessException {
        try {
            return this.myDO.getHandle();
        } catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error getting QueryLogs's handle", ex);
        }
    }   

    /**
        * Get Date of the log
     *
     * @return Date of the log
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public java.sql.Date getDate () throws PapillonBusinessException {
        try {
            return this.myDO.getDate();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting date", ex);
        }
    }



    /**
     * Set date of the log
     *
     * @param date of the log
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setDate ( java.sql.Date date ) throws PapillonBusinessException {
        try {
            myDO.setDate(date);
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting date", ex);
        }
    }



   /**
        * Set date
     *
     * @param date (as a java.util.Date) of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setDate ( java.util.Date date ) throws PapillonBusinessException {
        try {
            myDO.setDate(new java.sql.Date(date.getTime()));
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's FileType", ex);
        }
    }


    /**
     * Gets the login of the QueryLog
     *
     * @return the login.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getLogin() 
        throws PapillonBusinessException {
        try {
            return myDO.getLogin();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's date", ex);
        }
    }
    
    /**
     * Sets the login of the QueryLog
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	 public void setLogin(String login) 
        throws PapillonBusinessException {
        try {
            myDO.setLogin(login);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's login", ex);
        }
    }


    /**
     * Gets the formName of the QueryLog
     *
     * @return the formName.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getFormName() 
        throws PapillonBusinessException {
        try {
            return myDO.getFormName();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's form name", ex);
        }
    }
    /**
     * Sets the formName of the QueryLog
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    
	 public void setFormName(String formName) 
        throws PapillonBusinessException {
        try {
	    if (formName != null && formName.length()>50) {
		formName = formName.substring(0,49);
	    }
            myDO.setFormName(formName);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's form name", ex);
        }
    }
 


    /**
     * Gets the prefLang of the QueryLog
     *
     * @return the prefferred language.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getPrefLang() 
        throws PapillonBusinessException {
        try {
            return myDO.getPrefLang();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's prefferred language", ex);
        }
    }
    /**
     * Sets the preferred language of the QueryLog
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    
	 public void setPrefLang(String lang) 
        throws PapillonBusinessException {
        try {
	    if (lang != null && lang.length()>3) {
		lang = lang.substring(0,2);
	    }
            myDO.setPrefLang(lang);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's source language", ex);
        }
    }
 


    /**
     * Gets the headword query of the QueryLog
     *
     * @return the headword query.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getQuery() 
        throws PapillonBusinessException {
        try {
            return myDO.getQuery();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's headword query", ex);
        }
    }
    /**
     * Sets the headword query of the QueryLog
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    
	 public void setQuery(String query) 
        throws PapillonBusinessException {
        try {
	    if (query != null && query.length()>255) {
		query = query.substring(0,2);
	    }
            myDO.setQuery(query);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's headword query", ex);
        }
    }
 


    /**
     * Gets the headword results of the QueryLog
     *
     * @return the headword results.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getResults() 
        throws PapillonBusinessException {
        try {
            return myDO.getResults();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's headword results", ex);
        }
    }
    /**
     * Sets the headword results of the QueryLog
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    
	 public void setResults(String results) 
        throws PapillonBusinessException {
        try {
            myDO.setResults(results);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's headword results", ex);
        }
    }
 


    /**
     * Gets the source language of the QueryLog
     *
     * @return the source language of the query
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getSrcLang() 
        throws PapillonBusinessException {
        try {
            return myDO.getSrcLang();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's source language", ex);
        }
    }
    /**
     * Sets the source language of the query user
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    
	 public void setSrcLang(String lang) 
        throws PapillonBusinessException {
        try {
	    if (lang != null && lang.length()>3) {
		lang = lang.substring(0,2);
	    }
            myDO.setSrcLang(lang);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's source language", ex);
        }
    }
 

    /**
     * Gets the target languages of the QueryLog
     *
     * @return the target language of the query
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getTrgLangs() 
        throws PapillonBusinessException {
        try {
            return myDO.getTrgLangs();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's target languages", ex);
        }
    }
    /**
     * Sets the target languages of the query user
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    
	 public void setTrgLangs(String langs) 
        throws PapillonBusinessException {
        try {
	    if (langs != null && langs.length()>512) {
		langs = langs.substring(0,511);
	    }
            myDO.setTrgLangs(langs);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's targets languages", ex);
        }
    }
 

    /**
     * Gets the dictionaries of the QueryLog
     *
     * @return the dictionaries of the query
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getDicts() 
        throws PapillonBusinessException {
        try {
            return myDO.getDicts();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's dictionaries", ex);
        }
    }
    /**
     * Sets the dictionaries of the query user
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    
	 public void setDicts(String dicts) 
        throws PapillonBusinessException {
        try {
	    if (dicts != null && dicts.length()>1024) {
		dicts = dicts.substring(0,1023);
	    }
            myDO.setDicts(dicts);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's dictionaries", ex);
        }
    }
 

    /**
     * Gets the strategy of the QueryLog
     *
     * @return the strategy of the query
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getStrategy() 
        throws PapillonBusinessException {
        try {
            return myDO.getStrategy();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's strategy", ex);
        }
    }
    /**
     * Sets the strategy of the query user
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    
	 public void setStrategy(String strat) 
        throws PapillonBusinessException {
        try {
	    if (strat != null && strat.length()>255) {
		strat = strat.substring(0,254);
	    }
            myDO.setStrategy(strat);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's strategy", ex);
        }
    }
 

    /**
     * Gets the entire query string of the QueryLog
     *
     * @return the query string of the query
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getQueryString() 
        throws PapillonBusinessException {
        try {
            return myDO.getQueryString();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting QueryLog's query string", ex);
        }
    }
    /**
     * Sets the query string of the query user
     *
     * @return void.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    
	 public void setQueryString(String query) 
        throws PapillonBusinessException {
        try {
            myDO.setQueryString(query);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting QueryLog's query string", ex);
        }
    }
 


  

  
   
    public void save() 
        throws PapillonBusinessException {
        try {
            this.myDO.commit();
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error saving QueryLog", ex);
        }
    }
    
    /**
     * Deletes the QueryLog from the database.
     *
     * @exception PapillonBusinessException if an error occurs
     *   deleting data (usually due to an underlying data layer
     *   error).
     */
    public void delete() 
        throws PapillonBusinessException {
        try {
            this.myDO.delete();
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error deleting QueryLog", ex);
        }
    }
}
