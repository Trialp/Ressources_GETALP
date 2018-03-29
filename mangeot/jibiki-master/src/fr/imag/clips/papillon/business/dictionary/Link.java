/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 *-----------------------------------------------
 * $Id: Link.java 1017 2012-03-26 21:00:00 Zhang ying $
 *-----------------------------------------------
 * $Log$
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
import fr.imag.clips.papillon.data.LinkDO;

/**
 * Represents a volume index. 
 */
public class Link {
	// Beware, for the text database columns, there is an UTF-8 conversion needed
	// but for the character columns, no need !
    
	
	public static final String FINAL_TYPE = "final";
	public static final String AXIE_TYPE = "axi";
	public static final String INDIRECT_TYPE = "indirect";
	public static final String AXEME_TYPE = "axeme";
	public static final String EQU_TYPE = "equ";
	public static final String CNT_TYPE = "cnt";

	public static final String DIRECTION_UP = "up";
	public static final String DIRECTION_DOWN = "down";

    /**
     * The DO of the Dictionary.
     */
    protected LinkDO myDO = null;
	protected VolumeEntry linkedEntry = null;

    /**
     * The public constructor.
     * Should find a better method instead of these if elsif elsif
     * How to do it ?
     */

    public Link(String tableName) throws PapillonBusinessException {
        try {
            this.myDO = LinkDO.createVirgin(tableName, CurrentDBTransaction.get());
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
    protected Link(LinkDO theIndex) 
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

   
   public String getTargetId()
    throws PapillonBusinessException {
		try {
			return myDO.getTargetId();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting target id", ex);
		}
	}



	
	public void setTargetId(String idTarget)
	    throws PapillonBusinessException {
	        try {
	            myDO.setTargetId(idTarget);
	        } catch(DataObjectException ex) {
	            throw new PapillonBusinessException("Error setting index lang", ex);
	        }
	    }
	
	public String getElementId()
    throws PapillonBusinessException {
		try {
			return myDO.getElementId();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting element id", ex);
		}
	}
	
	
	
	
	public void setElementId(String id)
	throws PapillonBusinessException {
		try {
			myDO.setElementId(id);
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error setting element id", ex);
		}
	}
	
	
	
    public String getVolumeTarget()
    throws PapillonBusinessException {
		try {
			return myDO.getVolumeTarget();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting target volume", ex);
		}
	}

	/**

	 */
	public void setVolumeTarget(String volume)
	    throws PapillonBusinessException {
	        try {
	            myDO.setVolumeTarget(volume);
	        } catch(DataObjectException ex) {
	            throw new PapillonBusinessException("Error setting target volume", ex);
	        }
	    }
	
    public String getType()
    throws PapillonBusinessException {
		try {
			return myDO.getType();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting type", ex);
		}
	}

	/**

	 */
	public void setType(String type)
	    throws PapillonBusinessException {
	        try {
	            myDO.setType(type);
	        } catch(DataObjectException ex) {
	            throw new PapillonBusinessException("Error setting type", ex);
	        }
	    }
	
    public String getLabel()
    throws PapillonBusinessException {
		try {
			return myDO.getLabel();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting label", ex);
		}
	}

	/**

	 */
	public void setLabel(String label)
	    throws PapillonBusinessException {
	        try {
	            myDO.setLabel(label);
	        } catch(DataObjectException ex) {
	            throw new PapillonBusinessException("Error setting label", ex);
	        }
	    }
	
    public double getWeight()
    throws PapillonBusinessException {
		try {
			return myDO.getWeight();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting weight", ex);
		}
	}

	/**

	 */
	public void setWeight(String weight)
	throws PapillonBusinessException {
		setWeight(Double.parseDouble(weight));
	}

	public void setWeight(double weight)
	    throws PapillonBusinessException {
	        try {
	            myDO.setWeight(weight);
	        } catch(DataObjectException ex) {
	            throw new PapillonBusinessException("Error setting weight", ex);
	        }
	    }
	
    public String getName()
    throws PapillonBusinessException {
		try {
			return myDO.getName();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting link name", ex);
		}
	}
	
	/**
	 
	 */
	public void setName(String name)
	throws PapillonBusinessException {
		try {
			myDO.setName(name);
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error setting link name", ex);
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
				throw new PapillonBusinessException("Error getting index's lang", ex);
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
		* Gets the value of the key
     *
     * @return the value as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
						  *   error).
     */

	/**
     * Gets the linkedEntry
	 *
     * @return the entry.
     */
	public VolumeEntry getLinkedEntry() {
		return linkedEntry;
	}
	
    /**
	 * Sets the linkedEntry
     *
     * @param the entry.
     */
	public void setLinkedEntry(VolumeEntry theEntry) {
		linkedEntry = theEntry;
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