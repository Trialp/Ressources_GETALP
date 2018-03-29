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
 * Revision 1.10  2007/01/05 13:57:26  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.9  2006/05/05 02:08:23  fbrunet
 * bug correction : url utf8 transfert (in createEntryInit)
 *
 * Revision 1.8  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.7.4.1  2006/01/25 15:22:23  fbrunet
 * Improvement of QueryRequest
 * Add new search criteria
 * Add modified status
 *
 * Revision 1.7  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.6  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.5.4.1  2005/06/09 11:07:45  mangeot
 * Deleted the countEntriesCache. entries counts are not cached any more.
 * Fixed a few bugs.
 *
 * Revision 1.5  2005/04/15 11:42:52  mangeot
 * Fixed a bug in isInNormalGroups when groups can be empty
 *
 * Revision 1.4  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.3.2.2  2005/03/30 11:17:07  mangeot
 * Modified table contributions: replaced originalhandle by originalid
 * Corrected a few bugs when validating an already existing entry
 *
 * Revision 1.3.2.1  2005/03/29 09:41:33  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.3  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:32  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.user;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.CurrentDBTransaction;

/* for password encryption */
import java.io.*;
import java.security.*;
import java.util.*;
import java.lang.*;



/* for the ADMIN password in config file */
import com.lutris.appserver.server.Enhydra;
import com.lutris.util.ConfigException;

import com.lutris.appserver.server.sql.DatabaseManagerException;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.dods.builder.generator.query.DataObjectException;

/* for XML management */
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;


public class User implements com.lutris.appserver.server.user.User {
	
    public final static String KEY_SEP = "#";
    public final static String GROUPS_SEPARATOR_STRING = "#";

    protected final static String PASSWORD_STRING = "Papillon.Users.Groups.Password";
    protected final static String PASSWORD_ENCODING = "US-ASCII";
    //protected final static String PASSWORD_ENCODING = "UTF-8";
    protected final static String PASSWORD_DIGEST = "SHA";
    protected final static int PASSWORD_DIGEST_LENGTH = 30;
	
	protected final static String USER_TAG = "user";
	protected final static String PREFERENCES_TAG = "preferences";
	protected final static String PREFERENCE_TAG = "preference";
	protected final static String CREDITS_TAG = "credits";
	protected final static String URL_ATTR = "href";
	protected final static String NAME_ATTR = "name";
	protected final static String VALUE_ATTR = "value";
	
	protected final static String STAR = "★";
	
	protected Hashtable UserPreferencesTable = null;
	
    /**
		* The DO of the Volume.
	 */
	protected UserDO myDO = null;
	
	/**
		* The public constructor.
	 */
	public User() throws PapillonBusinessException {
		try {
			this.myDO = UserDO.createVirgin(CurrentDBTransaction.get());
		} catch(DatabaseManagerException ex) {
			throw new PapillonBusinessException("Error creating empty user", ex);
		} catch(ObjectIdException ex) {
			throw new PapillonBusinessException("Error creating empty user", ex);
		}
		/* FIXME: I changed the papillon.doml with canBeNull = yes for groups
		but the existing databases have to be changed... */
		this.setGroups("");
	}
	
	/** The protected constructor
		*
		* @param theDisc. The data object of the Volume.
		*/
	protected User(UserDO theUser)
		throws PapillonBusinessException  {
			this.myDO = theUser;
		}
	
	public boolean isEmpty() {
		return (this.myDO==null) ;
	}
	
	// access methods
	
	/**
		* Gets the object id for the Volume
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
				throw new PapillonBusinessException("Error getting user's handle", ex);
			}
		}
	
	/**
		* Gets the name of the Volume
	 *
	 * @return the name.
	 * @exception odes not throw any exception for compatability with the interface.
	 */
	public String getName() {
		String result = null;
		try {
			result = myDO.getName();
		}  catch(DataObjectException ex) {
				fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Error getting user's name: " + ex.toString());
			}
		return result;
	}
	public void setName(String name)
		throws PapillonBusinessException {
			try {
				myDO.setName(name);
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting Volume's name", ex);
			}
		}
	
	
	/**
		* Gets the dictname of the Volume
	 *
	 * @return the dictname.
	 * @exception PapillonBusinessException if an error occurs
	 *   retrieving data (usually due to an underlying data layer
						  *   error).
	 */
	public String getLogin()
		throws PapillonBusinessException {
			try {
				return myDO.getLogin();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting user's login", ex);
			}
		}
	public void setLogin(String login)
		throws PapillonBusinessException {
			try {
                myDO.setLogin(login);
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting user's login", ex);
			}
		}
	
	
	/**
		* Gets the password of the user
	 *
	 * @return the password.
	 * @exception PapillonBusinessException if an error occurs
	 *   retrieving data (usually due to an underlying data layer
						  *   error).
	 */
	public String getPassword()
		throws PapillonBusinessException {
			try {
 				return myDO.getPassword();
				
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting users's password", ex);
			}
		}
	public void setPassword(String password)
		throws PapillonBusinessException {
			try {
				String passwordString = makeDigest(this.getLogin(), password);
                myDO.setPassword(passwordString);
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting user's password", ex);
			}
		}
	
	/**
		* Gets the email of the user
	 *
	 * @return the email.
	 * @exception PapillonBusinessException if an error occurs
	 *   retrieving data (usually due to an underlying data layer
						  *   error).
	 */
	public String getEmail()
		throws PapillonBusinessException {
			try {
				return myDO.getEmail();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting user's email", ex);
			}
		}
	public void setEmail(String email)
		throws PapillonBusinessException {
			try {
				myDO.setEmail(email);
				
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting user's email", ex);
			}
		}
	
	/**
		* Gets the preferred language of the user
	 *
	 * @return the preferred language.
	 * @exception PapillonBusinessException if an error occurs
	 *   retrieving data (usually due to an underlying data layer
						  *   error).
	 */
	public String getLang()
		throws PapillonBusinessException {
			try {
				return myDO.getLang();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting user's lang", ex);
			}
		}
	public void setLang(String lang)
		throws PapillonBusinessException {
			try {
				if (lang != null && lang.length() > 3) {
					lang = lang.substring(0,2);
				}
				myDO.setLang(lang);
				
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting user's lang", ex);
			}
		}
	
	/**
		* Gets the groups of the user
	 *
	 * @return the groups.
	 * @exception PapillonBusinessException if an error occurs
	 *   retrieving data (usually due to an underlying data layer
						  *   error).
	 */
	public String[] getGroupsArray()
		throws PapillonBusinessException {
			return Utility.deserializeStringIntoArray(getGroups(),GROUPS_SEPARATOR_STRING);
		}
	
	public String getGroups()
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
			setGroups(Utility.serializeStringArray(Groups,GROUPS_SEPARATOR_STRING));
		}
	
	/**
	 * Gets the number of credits
	 *
	 * @return the dictname.
	 * @exception PapillonBusinessException if an error occurs
	 *   retrieving data (usually due to an underlying data layer
	 *   error).
	 */
	public int getCredits()
	throws PapillonBusinessException {
		try {
			return myDO.getCredits();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting user's credits", ex);
		}
	}
	public void setCredits(int credits)
	throws PapillonBusinessException {
		try {
			myDO.setCredits(credits);
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error setting user's credits", ex);
		}
	}

	/**
		* Gets the xmlcode of the user
	 *
	 * @return the xmlcode as a string.
	 * @exception PapillonBusinessException if an error occurs
	 *   retrieving data (usually due to an underlying data layer
						  *   error).
	 */
    public String getXmlCode()
        throws PapillonBusinessException {
			try {
				return this.myDO.getXmlCode();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting User's xmlcode", ex);
			} 
		}
	public void setXmlCode(String code)
        throws PapillonBusinessException {
			try {
		  		myDO.setXmlCode(code);   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting User's xmlcode", ex);
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
    public java.sql.Date getCreationDate () throws PapillonBusinessException {
        try {
            return this.myDO.getCreationDate();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting User's creationDate", ex);
        }
    }
    
    /**
     * Get ModificationDate of the InformationDocument
     *
     * @return ModificationDate of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public java.sql.Date getModificationDate () throws PapillonBusinessException {
        try {
            return this.myDO.getModificationDate();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting User's modificationDate", ex);
        }
    }
    
	// group convenience methods
	protected void addNewGroup(String group)
		throws PapillonBusinessException {
			String groups = getGroups();
            if (groups == null || groups.equals("")) {
                groups = GROUPS_SEPARATOR_STRING;
            }
			groups += group + GROUPS_SEPARATOR_STRING;
			setGroups(groups);
		}
	
	public void addGroup(String group)
		throws PapillonBusinessException {
			if (!isInGroup(group)) {
				addNewGroup(group);
			}
		}
	
	public void removeGroup(String group)
		throws PapillonBusinessException {
			if (isInGroup(group)) {
				String[] groups = getGroupsArray();
				String newGroups = GROUPS_SEPARATOR_STRING;
				for (int i=0;i< groups.length;i++) {
					String tempGroup = groups[i];
					if (!tempGroup.equals(group)) {
						newGroups += tempGroup + GROUPS_SEPARATOR_STRING;
					}
				}
				setGroups(newGroups);
			}
		}
	
	
	
	public boolean isInGroups(String[] groups) 
		throws PapillonBusinessException {
			boolean answer = false;
			String[] myGroups = this.getGroupsArray();
			if (myGroups != null && myGroups.length>0) {
				if (groups !=null && groups.length>0) {
					int i=0;
					while (!answer && i<groups.length) {
						answer = this.isInGroup(groups[i]);
						i++;
					}
				}
			}
			return answer;
		}
	
	
	public boolean isInNormalGroups(String[] groups) 
		throws PapillonBusinessException {
			boolean answer = false;
			java.util.Vector groupsVector = new java.util.Vector();
			if (groups != null) {
				for (int i=0;i<groups.length;i++) {
					String tmpGroup = groups[i];
					if (tmpGroup!=null &&
						!tmpGroup.equals("") &&
						!tmpGroup.equals(Group.ADMIN_GROUP) &&
						!tmpGroup.equals(Group.SPECIALIST_GROUP) &&
						!tmpGroup.equals(Group.VALIDATOR_GROUP)) {
							groupsVector.add(tmpGroup);
					}
				}
			}
			if (groupsVector.size()==0) {
				answer = true;
			}
			else {
				String[] myGroups = this.getGroupsArray();			
				if (myGroups != null && myGroups.length>0) {
					int i=0;
					while (!answer && i<myGroups.length) {
						answer = (groupsVector.contains(myGroups[i]));
						i++;
					}
				}
			}
			return answer;
		}
	
    public boolean isInGroup()
		throws PapillonBusinessException {
            String[] Groups = getGroupsArray();
            return !(Groups == null);
        }
    
	public boolean isInGroup(String group)
		throws PapillonBusinessException {
            String[] Groups = getGroupsArray();
            boolean is = false;
            if (null != Groups && Groups.length > 0) {
 				int i =0;
                while (!is && i<Groups.length) {
                    is = group.equals(Groups[i]);
					i++;
                }
            }
            return is;
        }
	
	public boolean isAdmin()
		throws PapillonBusinessException {
			return isInGroup(Group.ADMIN_GROUP);
		}
	
	public boolean isSpecialist()
		throws PapillonBusinessException {
			return isInGroup(Group.SPECIALIST_GROUP);
		}
	
	public boolean isValidator()
		throws PapillonBusinessException {
			return isInGroup(Group.VALIDATOR_GROUP);
		}
	
	public String getGroupPassword(String group)
		throws PapillonBusinessException {
            String password = null;
			try {
				password =  Enhydra.getApplication().getConfig().getString(PASSWORD_STRING + "." + group);
			}
            catch (ConfigException e) {
				//     throw new PapillonBusinessException("Password not found. Check the application config file.", e);
            }
            return password;
        }
	
	/**
		* save the object in the database
	 */
	
	public void save()
		throws PapillonBusinessException {
            // on supprime la sérialisation car elle pose des pbs de synchro de BD
            //String xml = this.serializeXml();
			// PapillonLogger.writeDebugMsg("User.save" + xml);
            //this.setXmlCode(this.serializeXml());
			try {
                if (null == this.getCreationDate()) {
                    this.myDO.setCreationDate(new java.sql.Date(new java.util.Date().getTime()));
                }
                this.myDO.setModificationDate(new java.sql.Date(new java.util.Date().getTime()));
				this.myDO.commit();
			} catch(Exception ex) {
				throw new PapillonBusinessException("Error saving user", ex);
			}
		}
	
	/**
		* Deletes the Volume from the database.
	 *
	 * @exception PapillonBusinessException if an error occurs
	 *   deleting data (usually due to an underlying data layer
						*   error).
	 */
	public void delete()
		throws PapillonBusinessException {
            UsersFactory.removeUserFromCache(this);
			try {
				this.myDO.delete();
			} catch(Exception ex) {
				throw new PapillonBusinessException("Error deleting Volume", ex);
			}
		}
	
	
	/**	Uses the given digest algorithm to compute a 20 byte array of the
		*	user name and password.
		*/
	protected static String makeDigest(String user, String password)
		throws PapillonBusinessException {
            String givenPasswordString = "";
			byte[] digestbytes = new byte[PASSWORD_DIGEST_LENGTH];
			if (user == null || user.equals("")) {
				throw new PapillonBusinessException("Error in makeDigestString: user login empty");
			}
			try {
				MessageDigest messagedigest = MessageDigest.getInstance(PASSWORD_DIGEST);
				messagedigest.update(user.getBytes());
				messagedigest.update(password.getBytes());
				ByteArrayOutputStream baos = new ByteArrayOutputStream();
				messagedigest.update(baos.toByteArray());
				digestbytes = messagedigest.digest();
                givenPasswordString = new String(digestbytes, PASSWORD_ENCODING);
                givenPasswordString = givenPasswordString.replaceAll("\\x00", "");
            } catch(NoSuchAlgorithmException nsae) {
                throw new PapillonBusinessException("Error in makeDigest: NoSuchAlgorithmException " + PASSWORD_DIGEST + " : ", nsae);
            }
            catch(UnsupportedEncodingException uee) {
                throw new PapillonBusinessException("Error in makeDigest: UnsupportedEncoding " + PASSWORD_ENCODING + " : ", uee);
            }
            //PapillonLogger.writeDebugMsg("makeDigest: [" + user +"],[" +password+"] = [" + digestbytes.toString()+"]");
			return givenPasswordString;
		}
	
	public boolean HasCorrectPassword(String password)
		throws PapillonBusinessException {
            boolean answer = false;
            String login = this.getLogin();
            if (null != login && !login.equals("")) {
                String givenPasswordString = makeDigest(login,password);
                answer = this.getPassword().equals(givenPasswordString);
            }
            return answer;
        } 
		
	// xmlcode management methods
	
	protected String serializeXml() 
		throws PapillonBusinessException {
			String xmlCode = "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>\n" +
"<" + USER_TAG +
"   xmlns='http://www-clips.imag.fr/geta/services/dml'\n" +
"   xmlns:xlink='http://www.w3.org/1999/xlink'\n" +
"   xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'\n" +
"   xsi:schemaLocation='http://www-clips.imag.fr/geta/services/dml\n" +
"   http://www-clips.imag.fr/geta/services/dml/dml.xsd'\n";
			if (getName()!=null) {
				xmlCode += "name='" + getName() + "'\n";
			}
			if (getLang()!=null) {
				xmlCode += "lang='" + getLang() + "'\n";
			}

	//end of the attributes
			xmlCode += ">\n";
			
	// login
			if (getLogin()!=null) {
				xmlCode += "<login>" + getLogin() + "</login>\n";
			}
 
	// password
			// disabled for security reasons
			/*if (getPassword()!=null) {
				xmlCode += "<password encoding='" + PASSWORD_ENCODING + "' ";
				xmlCode += "digest-length='" + PASSWORD_DIGEST_LENGTH + "' ";
				xmlCode += "digest='" + PASSWORD_DIGEST + "' ";
				xmlCode += ">" + getPassword() + "</password>\n";
			}*/
 
	// email
			if (getEmail()!=null) {
				xmlCode += "<email>" + getEmail() + "</email>\n";
			}

	// groups
			xmlCode += "<groups>\n";
			String[] groupsArray = getGroupsArray();
			if (groupsArray!=null) {
				for (int i=0;i<groupsArray.length;i++) {
					xmlCode += "<group-ref name='" + groupsArray[i] + "' />\n";
				}
			}
			xmlCode += "</groups>\n";
			
			// credits
			xmlCode += "<" + CREDITS_TAG + ">" + getCredits() + "</" + CREDITS_TAG + ">\n";
			
			// preferences
			xmlCode += "<" + PREFERENCES_TAG + ">"+ serializePreferences()+"</" + PREFERENCES_TAG + ">\n";
			
	// end of user
			xmlCode += "</" + USER_TAG + ">\n";
			
			return xmlCode;
		}
			
	public Hashtable getPreferences() 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		if (UserPreferencesTable==null) {
			UserPreferencesTable = getXMLPreferences();
		}
		return UserPreferencesTable;
	}
	
		// preference management methods	
		protected Hashtable getXMLPreferences() 
			throws fr.imag.clips.papillon.business.PapillonBusinessException {
			Hashtable PrefsTable = new Hashtable();
		
			String xmlCode = getXmlCode();
			if (xmlCode!=null && !xmlCode.equals("")) {
				Document myDocDOM = XMLServices.buildDOMTree(xmlCode);
				NodeList myNodeList = myDocDOM.getElementsByTagName(PREFERENCE_TAG);
				for (int i=0;i<myNodeList.getLength ();i++) {
					Element currentElt = (Element) myNodeList.item(i);
					String tmpUrl = currentElt.getAttribute(URL_ATTR);
					String tmpName = currentElt.getAttribute(NAME_ATTR);
					String tmpPref = currentElt.getAttribute(VALUE_ATTR);
					if (tmpUrl != null && tmpName != null && tmpPref != null) {
						PrefsTable.put(tmpUrl + KEY_SEP + tmpName, tmpPref);
					}
				}
			}
			return PrefsTable;
		}
	
	public void setPreferences(Hashtable PreferencesTable) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		UserPreferencesTable = PreferencesTable;
		this.save();
	}
	
		protected String serializePreferences() 
			throws fr.imag.clips.papillon.business.PapillonBusinessException {
				if (UserPreferencesTable==null) {
					UserPreferencesTable = getXMLPreferences();
				}
			String result = "";
			Enumeration e = UserPreferencesTable.keys();
				while(e.hasMoreElements()) {
				String key = (String) e.nextElement();
				String value = (String) UserPreferencesTable.get(key);
				String[] urlName = key.split(KEY_SEP);
					if (urlName!=null && urlName.length==2) {
						String url = urlName[0];
						String name = urlName[1];
						result += "<" + PREFERENCE_TAG + " "+URL_ATTR + "='" + url + "' ";
						result += NAME_ATTR + "='" + name+ "' ";
						result += VALUE_ATTR + "='" + value+ "' ";
						result += "/>";
					}
			}
			return result;
		}
						
		public void levelUp()
		throws PapillonBusinessException {
			String stars = Utility.getStars(this.getGroupsArray());
			if (stars==null || stars.length()<5) {
				stars += STAR;
				addNewGroup(stars);
			}
		}
	
		public void levelDown()
		throws PapillonBusinessException {
			String stars = Utility.getStars(this.getGroupsArray());
			if (stars!=null) {
				removeGroup(stars);
				if (stars.length()>1) {
					addGroup(stars.substring(0,stars.length()-1));
				}
			}
		}
	
}
