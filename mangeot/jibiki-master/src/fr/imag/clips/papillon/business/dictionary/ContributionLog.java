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
 * Revision 1.3  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.2  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.1.2.1  2005/04/29 15:29:09  mangeot
 * Added classes for new version
 *
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
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.data.ContributionLogDO;


/**
 * Represents a Mailing list Dictionary. 
 */
public class ContributionLog {

    /**
     * The DO of the Dictionary.
     */
    protected fr.imag.clips.papillon.data.ContributionLogDO myDO = null;

    /**
     * The public constructor.
     */
    public ContributionLog() throws PapillonBusinessException {
        try {
			this.myDO = ContributionLogDO.createVirgin(CurrentDBTransaction.get());

        }
        catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error creating empty contribution", ex);
        } catch(ObjectIdException ex) {
            throw new PapillonBusinessException("Error creating empty contribution", ex);
        }
    }

    /** The protected constructor
     *
     * @param theDisc. The data object of the ContributionLog.
     */				
    protected ContributionLog(ContributionLogDO theContributionLog) 
        throws PapillonBusinessException  {
			this.myDO = theContributionLog;
		}

    public boolean isEmpty() {
        return (this.myDO==null) ;
    }


    /**
     * Gets the handle of the contribution
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
            throw new PapillonBusinessException("Error getting ContributionLog's handle", ex);
        }
    }


    /**
        * Gets the author of the contribution
     *
     * @return the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
                          *   error).
     */
    public String getAuthor()
        throws PapillonBusinessException {
            try {
                return myDO.getAuthor();
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error getting contribution's author", ex);
            }
        }

    public void setAuthor(String author)
        throws PapillonBusinessException {
            try {
                myDO.setAuthor(author);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting contribution's author", ex);
            }
        }


    /**
        * Gets the groups of the contribution
     *
     * @return the groups.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
                          *   error).
     */
		public String[] getGroupsArray() throws PapillonBusinessException {
			String[] Groups = null;
			String groups = getGroups();
			if (null != groups && !groups.equals("")){
			// delete the first separator in order to avoid an empty group
				if (groups.indexOf(User.GROUPS_SEPARATOR_STRING) ==0) {
					groups = groups.substring(User.GROUPS_SEPARATOR_STRING.length());
				}
				Groups = groups.split(User.GROUPS_SEPARATOR_STRING);
			}
			return Groups;
		}
        
        protected String getGroups()
            throws PapillonBusinessException {
                try {
                   return myDO.getGroups();

                } catch(DataObjectException ex) {
                    throw new PapillonBusinessException("Error getting user's groups", ex);
                }
            }
        protected void setGroups(String groups)
            throws PapillonBusinessException {
                try {
                myDO.setGroups(groups);

                } catch(DataObjectException ex) {
                    throw new PapillonBusinessException("Error setting user's groups", ex);
                }
            }
        public void setGroupsArray(String[] Groups)
            throws PapillonBusinessException {
                String groups = null;
                if (null != groups && Groups.length >0) {
                    for (int i=0; i< Groups.length; i++) {
                        groups = groups + User.GROUPS_SEPARATOR_STRING + Groups[i];
                    }
                }
                groups.trim();
                setGroups(groups);
            }
    /**
        * Gets the volume name of the contribution
     *
     * @return the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
                          *   error).
     */
    public String getVolumeName()
        throws PapillonBusinessException {
            try {
                return myDO.getVolume();
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error getting contribution's volume", ex);
            }
        }

    public void setVolumeName(String volume)
        throws PapillonBusinessException {
            try {
                myDO.setVolume(volume);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting contribution's volume", ex);
            }
        }

    /**
        * Gets the source language of the contribution
     *
     * @return the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
                          *   error).
     */
    public String getSourceLanguage()
        throws PapillonBusinessException {
            try {
                return myDO.getSourceLanguage();
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error getting contribution's source language", ex);
            }
        }

    public void setSourceLanguage(String lang)
        throws PapillonBusinessException {
            try {
                myDO.setSourceLanguage(lang);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting contribution's source language", ex);
            }
        }

    /**
        * Gets the headword of the volume
     *
     * @return the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
                          *   error).
     */
	public String getHeadword () throws PapillonBusinessException {
        try {
            return this.myDO.getHeadword();
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting contribution entry id", ex);
        }
	}

	public String getHeadwords () throws PapillonBusinessException {
		return this.getHeadword();
	}	
		
    public void setHeadword(String headword)
        throws PapillonBusinessException {
            try {
                myDO.setHeadword(headword);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting volume's headword", ex);
            }
        }

    /**
     * Gets the id of the entry created with this contribution
     *
     * @return the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getEntryId()
        throws PapillonBusinessException {
        try {
            return this.myDO.getEntryId();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting contribution entry id", ex);
        }
    }

		public void setEntryId(String entryId)
        throws PapillonBusinessException {
        try {
		  		myDO.setEntryId(entryId);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting contribution entry id", ex);
        }
    }

    /**
     * Gets the id of the contribution
     *
     * @return the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getContributionId()
        throws PapillonBusinessException {
        try {
            return this.myDO.getContributionId();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting contribution id", ex);
        }
    }

		public void setContributionId(String contributionId)
        throws PapillonBusinessException {
        try {
		  		myDO.setContributionId(contributionId);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting contribution id", ex);
        }
    }

    /**
		 * Get CreationDate of the InformationDocument
     *
     * @return CreationDate of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public java.util.Date getDate () throws PapillonBusinessException {
		java.sql.Timestamp myTimestamp = null;
        try {
            myTimestamp = this.myDO.getDate();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting Date", ex);
        }
		return (java.util.Date) myTimestamp;
    }

    /**
        * Set Date of the ContributionLog
     *
     * @param Date (as a java.util.Date) of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setDate ( java.util.Date date ) throws PapillonBusinessException {
        try {
            myDO.setDate(new java.sql.Timestamp(date.getTime()));
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's FileType", ex);
        }
    }

    /**
        * Gets the status of the contribution
     *
     * @return the status.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
                          *   error).
     */
    public String getStatus()
        throws PapillonBusinessException {
            try {
                return myDO.getStatus();
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error getting contribution's status", ex);
            }
        }

    public void setStatus(String status)
        throws PapillonBusinessException {
            try {
                myDO.setStatus(status);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting contribution's status", ex);
            }
        }

	  /**
		 * Deletes the volume from the database.
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
            throw new PapillonBusinessException("Error saving contribution", ex);
        }
    }

    /**
     * Deletes the volume from the database.
     *
     * @exception PapillonBusinessException if an error occurs
     *   deleting data (usually due to an underlying data layer
     *   error).
     */		
		public void delete() throws PapillonBusinessException {
        try {
            this.myDO.delete();
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error deleting contribution", ex);
        }
    }
}
