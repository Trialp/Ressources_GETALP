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
 * Revision 1.7  2005/09/15 13:21:04  mangeot
 * Fixed a bug when non registered users could not change their preferred language permanently
 *
 * Revision 1.6  2005/08/02 14:41:49  mangeot
 * Work on stylesheets and
 * added a reset button for Review and AdminContrib forms
 *
 * Revision 1.5  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.4  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.3.2.1  2005/02/25 10:22:08  mangeot
 * Bug fixes and added the use of referrer when exiting from Reviewcontributions.po
 *
 * Revision 1.3  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *
 *-----------------------------------------------
 * 
 */


package fr.imag.clips.papillon.presentation;

import fr.imag.clips.papillon.business.user.User;
import java.util.ArrayList;
import java.util.Hashtable;

import fr.imag.clips.papillon.business.PapillonBusinessException;


/**
 * Object that will be held in session data.  This should
 * be the only object held there.  Methods should be called
 * on this object to set and get data.
 */
public class PapillonSessionData {
    protected final int INITIAL_MESSAGE_SIZE = 256;
    
    /**
     * Hash key to save session data for the DiscRack app in the Session
     */
    public static final String SESSION_KEY = "PapillonSessionData";
    
    protected User sessionUser = null;
    protected StringBuffer userMessage = null;
    protected ArrayList acceptLanguages = null;
    protected boolean ClientWithLabelDisplayProblems = false;
	protected Hashtable PreferencesTable = null;
    
    public PapillonSessionData() 
        throws PapillonBusinessException {
			this.userMessage = new StringBuffer(INITIAL_MESSAGE_SIZE);            
        }

    public PapillonSessionData(User myUser, ArrayList userAcceptLanguage, String requestHeader)
		throws PapillonBusinessException {
            this();
			setUser(myUser);
			setUserAcceptLanguages(userAcceptLanguage);
			if (myUser != null && !myUser.isEmpty()) {
				setUserPreferredLanguage(myUser.getLang());
			}
			setClientWithLabelDisplayProblems(requestHeader);
    }

    /**
        * Language selection support
     *
     * @param lang the language name (3 characters String).
     */
    public void setUserPreferredLanguage(String lang) {
		if (this.acceptLanguages != null) {
			this.acceptLanguages.set(0, lang);
		}
    }

    public ArrayList getUserAcceptLanguages() {
        return this.acceptLanguages;
    }
    
    public String getUserPreferredLanguage() {
        return (String)this.acceptLanguages.get(0);
    }

    protected void setUserAcceptLanguages(ArrayList accept) {
        this.acceptLanguages = accept;
    }

    public boolean getClientWithLabelDisplayProblems() {
        return this.ClientWithLabelDisplayProblems;
    }

    protected void setClientWithLabelDisplayProblems(String requestHeader) {
        if (requestHeader != null) {
            this.ClientWithLabelDisplayProblems = (requestHeader.indexOf("MSIE") > 0
                                                   || requestHeader.indexOf("iCab") > 0);
        }
    }
   
    /**
        * Sets the person object
     *
     * @param thePerson the person object
     */
    protected void setUser(User theUser) {
        this.sessionUser = theUser;
    }

    /**
     * Gets the person object
     *
     * @return person
     */
    public User getUser() {
        return this.sessionUser;
    }    
    
    /**
     * Method to remove the current user from the session
     */
    public void removeUser() {
        this.sessionUser = null;
    }
    
    public String getUserMessage() {
        return this.userMessage.toString();
    }
    
    /**
     * Appends the given String to the Message that will be provided to the user
     * after the completion of the action.
     */
    public void writeUserMessage(String msg) {
        if  (null != msg && !msg.equals("")) {
            this.userMessage.append(msg);
            if (null != msg && !msg.endsWith("\n")) {
                this.userMessage.append('\n');
            }
        }
    }
    
    public void flushUserMessage() {
        this.userMessage.setLength(0);
    } 

    /**
        *  Description of the Method
     *
     * @return    Description of the Return Value
     */
    public boolean userHasBeenIdentified() {
        return (this.sessionUser != null);
    }
    
    
// management of the preferences
    public String getPreference(String url, String name) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
        String pref = "";
		if (this.PreferencesTable ==null) {
			if (this.sessionUser!=null) {
				this.PreferencesTable = this.sessionUser.getPreferences();
			}
			else {
				this.PreferencesTable = new Hashtable();
			}
		}
		if (this.PreferencesTable !=null) {
			pref = (String) this.PreferencesTable.get(url + User.KEY_SEP + name);
		}
		return pref;
    } 

	
    public void setPreference(String url, String name, String value) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			setPreference(url, name, value, true);
	} 

    public void setPreference(String url, String name, String value, boolean persistent) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			if (value == null) {
				value = "";
			}
			if (this.PreferencesTable ==null) {
				if (this.sessionUser!=null) {
					this.PreferencesTable = this.sessionUser.getPreferences();
				}
				else {
					this.PreferencesTable = new Hashtable();
				}
			}
			String pref = (String) this.PreferencesTable.get(url + User.KEY_SEP + name);
			if (!value.equals(pref)) {
				//fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("AddPref: url: " + url + " name: " + name  + " value: " + value + " Pref: " + pref);
				this.PreferencesTable.put(url + User.KEY_SEP + name, value);
				if (this.sessionUser!=null && persistent) {
					this.sessionUser.setPreferences(PreferencesTable);
				}				
			}
		}
	
	public void resetPreferences(String url) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		for (java.util.Enumeration e = PreferencesTable.elements() ; e.hasMoreElements() ;) {
			String key = (String) e.nextElement();
			if (key.regionMatches(0,url,0,url.length())) {
				this.PreferencesTable.put(key, "");
			}
		}
		if (this.sessionUser!=null) {
			this.sessionUser.setPreferences(PreferencesTable);
		}						
	}
}
