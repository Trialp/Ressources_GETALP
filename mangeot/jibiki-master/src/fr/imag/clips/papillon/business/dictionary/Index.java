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
 * Revision 1.6  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.5  2006/02/26 19:21:38  mangeot
 * Work on BrowseVolume
 *
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
 * Revision 1.2.2.2  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.2.2.1  2005/01/25 13:54:54  mangeot
 * changed the volume volumeEntry and index objects. Does not compile but need a backup...
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

package fr.imag.clips.papillon.business.dictionary;

import com.lutris.appserver.server.sql.DatabaseManagerException;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.dods.builder.generator.query.DataObjectException;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.data.IndexDO;

/**
 * Represents a volume index. 
 */
public class Index {

	protected static java.util.regex.Pattern quotePattern = java.util.regex.Pattern.compile("'");
	// Beware, for the text database columns, there is an UTF-8 conversion needed
	// but for the character columns, no need !
    
    /**
     * The DO of the Dictionary.
     */
    
    protected IndexDO myDO = null;

    /**
     * The public constructor.
     * Should find a better method instead of these if elsif elsif
     * How to do it ?
     */

    public Index(String tableName) throws PapillonBusinessException {
        try {
            this.myDO = IndexDO.createVirgin(tableName, CurrentDBTransaction.get());
        }
        catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error creating empty index", ex);
        } catch(ObjectIdException ex) {
            throw new PapillonBusinessException("Error creating empty index", ex);
        }
    }

    /** The protected constructor
     *
     * @param theDisc. The data object of the Volume.
     */
    protected Index(IndexDO theIndex) 
        throws PapillonBusinessException  {
        this.myDO = theIndex;
    }

    public boolean isEmpty() {
        return (this.myDO==null) ;
    }


    /**
     * Gets the object id for the Dictionary
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
        } 
        catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error getting index's handle", ex);
        }
    }

    /**
     * Gets the subject of the Dictionary
     *
     * @return the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getTableName() {
        return this.myDO.getTableName(); 
    }

    /**
		* Gets the key of the index
     *
     * @return the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
						  *   error).
     */
	public String getKey()
        throws PapillonBusinessException {
			try {
				return myDO.getKey();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting index's key", ex);
			}
		}

    public void setKey(String key)
        throws PapillonBusinessException {
            try {
                myDO.setKey(key);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting volume's key", ex);
            }
        }

    /**
		* Gets the language of the key
     *
     * @return the language as an ISO-639/2T 3 letters language code (fra, deu, eng, etc.).
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
						  *   error).
     */
	public String getLang()
        throws PapillonBusinessException {
			try {
				return myDO.getLang();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting index lang", ex);
			}
		}

    /**
		* Sets the language of the key
     *
     * @param the language as an ISO-639/2T 3 letters language code (fra, deu, eng, etc.).
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public void setLang(String lang)
        throws PapillonBusinessException {
            try {
                myDO.setLang(lang);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting index lang", ex);
            }
        }

	/**
	 * Gets the msort of the key
     *
     * @return the msort key
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
	public String getMsort()
	throws PapillonBusinessException {
		try {
			return myDO.getMsort();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting index msort", ex);
		}
	}
	

    /**
		* Gets the value of the key
     *
     * @return the value as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
						  *   error).
     */
	public String getValue()
        throws PapillonBusinessException {
			try {
				return myDO.getValue();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting index's value", ex);
			}
		}

    /**
		* Sets the value of the key
     *
     * @param the value as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public void setValue(String value)
        throws PapillonBusinessException {
            try {
                myDO.setValue(value);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting index value", ex);
            }
        }

    /**
     * Gets the EntryId
	 *
     * @return the entry id.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
	public int getEntryId()
        throws PapillonBusinessException {
			try {
				return myDO.getEntryId();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting index's entry id", ex);
			}
		}

    /**
	 * Sets the entry id
     *
     * @param the entry id.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
     public void setEntryId(int entryId)
        throws PapillonBusinessException {
            try {
                myDO.setEntryId(entryId);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting index's entry id", ex);
            }
        }

    /**
        * Saves the index to the database.
     *
     * @exception PapillonBusinessException if an error occurs
     *   deleting data (usually due to an underlying data layer
                        *   error).
     */
    public void save() 
        throws PapillonBusinessException {
        try {
			String msort = fr.imag.clips.papillon.papillon_data.ManageDatabase.multilingual_sort(this.getLang(), this.getValue());
			myDO.setMsort(msort);
            this.myDO.commit();
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error saving index", ex);
        }
    }
    
    /**
     * Deletes the volume entry from the database.
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
            throw new PapillonBusinessException("Error deleting index", ex);
        }
    }
}
