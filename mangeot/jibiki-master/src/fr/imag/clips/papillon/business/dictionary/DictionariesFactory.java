/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.34.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.34  2007/01/15 17:12:18  serasset
 * Several notes added, suppressed the HTMLDOM_CACHE stuff.
 *
 * Revision 1.33  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.32  2006/08/10 22:17:12  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.31  2006/03/10 16:42:31  mangeot
 * Fix for targets.length==0 not satisfied
 *
 * Revision 1.30  2006/03/10 16:21:00  mangeot
 * Hack for targets.length==0, I am not satisfied..
 *
 * Revision 1.29  2006/03/10 15:47:25  mangeot
 * Hack when targets==null added to the findAnswerAndTranslations
 * I am not satified with that.
 *
 * Revision 1.28  2006/03/05 11:07:21  mangeot
 * *** empty log message ***
 *
 * Revision 1.27  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.26  2005/11/24 11:09:53  mangeot
 * *** empty log message ***
 *
 * Revision 1.25  2005/11/24 11:07:43  mangeot
 * *** empty log message ***
 *
 * Revision 1.24  2005/11/24 11:05:22  mangeot
 * *** empty log message ***
 *
 * Revision 1.23  2005/11/14 23:21:57  mangeot
 * *** empty log message ***
 *
 * Revision 1.22  2005/11/14 22:57:27  mangeot
 * *** empty log message ***
 *
 * Revision 1.21  2005/11/14 22:49:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.20  2005/11/14 22:28:45  mangeot
 * *** empty log message ***
 *
 * Revision 1.19  2005/11/09 15:28:44  mangeot
 * *** empty log message ***
 *
 * Revision 1.18  2005/11/09 13:30:31  mangeot
 * *** empty log message ***
 *
 * Revision 1.17.4.2  2006/01/25 15:22:23  fbrunet
 * Improvement of QueryRequest
 * Add new search criteria
 * Add modified status
 *
 * Revision 1.17.4.1  2005/12/02 10:04:09  fbrunet
 * Add Pre/Post edition processing
 * Add index reconstruction
 * Add new query request
 * Add fuzzy search
 * Add new contribution administration
 * Add xsl transformation volume
 *
 * Revision 1.17  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 * Revision 1.16  2005/07/14 13:48:53  serasset
 * Added columns dictionaryname and volumename to the xslsheets.
 * Modified the XslSheet and XslSheetFactory accordingly.
 * Modified AdminXsl interface accordingly.
 * Modified DictionariesFactory and VolumesFactory to allow several xsl-sheets and to
 * correctly provide dictionaryName/volumeName to XslSheetFactory.
 * Cleanup of several errors in papillon_static doml file.
 *
 * Revision 1.15  2005/07/05 09:21:59  serasset
 * Template interface generator now correctly generates attribute names (with an @).
 * Target languages are now correctly handled when querying a pivot multilingual dictionary.
 *
 * Revision 1.14  2005/06/24 10:35:57  mangeot
 * Minor bug fixes
 *
 * Revision 1.13  2005/06/22 15:55:53  mangeot
 * Solved an unresolved prefix bug when the dml prefix was not in the template entry.
 * Now we use the DmlPrefixResolver to solve this issue.
 *
 * Revision 1.12  2005/06/20 16:55:02  mangeot
 * multiple bug fixes
 *
 * Revision 1.11  2005/06/17 12:38:56  mangeot
 * Changed lexiesCollection into lexiesHashtable in order to implement the getDirectTranslations
 *
 * Revision 1.10  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.9  2005/05/25 13:31:08  serasset
 * Return a monolingual entry even if the lexie is not linked to an axie.
 * LexALP transformer now formats simple monolingual query results.
 *
 * Revision 1.8  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.7.4.4  2005/06/01 15:20:33  mangeot
 * Added a boolean for contributionslog
 *
 * Revision 1.7.4.3  2005/05/27 11:53:21  mangeot
 * *** empty log message ***
 *
 * Revision 1.7.4.2  2005/04/29 17:30:30  mangeot
 * *** empty log message ***
 *
 * Revision 1.7.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.7  2005/04/20 10:51:14  mangeot
 * Correction de AddDirectTranslations
 *
 * Revision 1.6  2005/04/13 14:34:38  mangeot
 * Simplified the expert lookup. Now lookup directly the cdm element name
 *
 * Revision 1.5  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.4.2.8  2005/04/06 10:36:01  mangeot
 * Temporary fix for addDirectTranslations because the DOM must not be modified
 *
 * Revision 1.4.2.7  2005/03/29 14:10:46  mangeot
 * Bug fix in addDirectTranslations
 *
 * Revision 1.4.2.6  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.4.2.5  2005/02/06 22:43:49  mangeot
 * Merged the 2 Hashtables CDM Elements and XPaths into one
 * Added a boolean (reverse-lookup) in the volume metadata and functionalities in order to perform a reverse lookup when no direct lookup result is found
 * Added a boolean (index) in the volume metadata for indexing the only specified CDM Elements
 *
 * Revision 1.4.2.4  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
 *
 * Revision 1.4.2.3  2005/01/27 19:29:21  mangeot
 * Implemented the HtmlDom cache, it increases speed drastically.
 * Still does not compile after an ant clean
 *
 * Revision 1.4.2.2  2005/01/27 18:09:28  mangeot
 * Simple dictionary lookup is now working for GDEF.
 * Does not compile yet but cvs commit for backup
 *
 * Revision 1.4.2.1  2005/01/27 15:56:21  mangeot
 * Able to load a volume with XPointers, cannot lookup the result yet.
 * Does not compile but commit for backup
 *
 * Revision 1.4  2005/01/18 12:17:39  mangeot
 * Put default values for MaxDisplayedEntries and MAxRetrievedEntries
 *
 * Revision 1.3  2005/01/18 12:16:10  mangeot
 * Implemented the SQL LIMIT and OFFSET keywords. It allows us to retrieve the entries as blocks and page them. The LIMIT is the DictionariesFactory.MaxRetrievedEntries constant.
 * The implementation may need further tuning
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * Revision 1.18  2004/10/28 10:38:11  mangeot
 * Fixed some bugs that affected the dictd server
 * Modified some methods in order to display a text entry in the dictd server
 *
 * Revision 1.17  2004/05/17 15:38:59  mangeot
 * Added links for bilingual direct dictionaries type GDEF
 *
 * Revision 1.16  2004/05/12 15:35:22  serasset
 * The xml processing instruction also has to be suppressed in the source language
 * entry, when building the artificial Papillon entry.
 *
 * Revision 1.15  2004/05/11 16:15:47  serasset
 * The database now bear a <?xml instruction in the beginning of newly created
 * files.Hence, when building a fake Papillon entry 1 source -> n targets, each
 * translation is preceded by an invalid instruction. Hence the entry could not
 * be parsed for Xsl transformation. As a result, Papillon returned a null pointer
 * exception. Fixed with a hack that suppress the instruction directly in the xml
 * code when the fake entry is built.
 *
 * Revision 1.14  2004/02/10 05:27:12  mangeot
 * The version UIGEN_V2 has been merged with the trunk by MM
 * Be careful because the Volumes and contributions database tables have been modified.
 * You have to drop and rebuild them unless you modify them by hands.
 *
 *-----------------------------------------------
 *
 */

package fr.imag.clips.papillon.business.dictionary;

import com.lutris.dods.builder.generator.query.QueryBuilder;

import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.UsersFactory;
import fr.imag.clips.papillon.business.user.Group;
import fr.imag.clips.papillon.business.user.GroupAnswer;
import fr.imag.clips.papillon.business.user.GroupsFactory;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.data.DictionaryDO;
import fr.imag.clips.papillon.data.DictionaryQuery;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import java.net.URL;
import java.util.*;

/**
 * Used to find the instances of xslsheet.
 */
public class DictionariesFactory {
	
	public final static int MaxDisplayedEntries = 5;
	public final static int MaxRetrievedEntries = 100;
    protected final static String DML_URI = DmlPrefixResolver.DML_URI;
    protected final static String XLINK_URI = DmlPrefixResolver.XLINK_URI;
    protected final static String XSLT_URI = DmlPrefixResolver.XSLT_URI;
    public final static String DICTIONARY_FILES_TAG = "dictionary-metadata-files";
    protected final static String DICTIONARY_TAG="dictionary-metadata";
    protected final static String VOLUME_REF_TAG="volume-metadata-ref";
    protected final static String XSLSHEET_REF_TAG="xsl-stylesheet";
    protected final static String XSLSHEET_TAG="stylesheet";
    protected final static String HREF_ATTRIBUTE="href";
    protected final static String NAME_ATTRIBUTE="name";
	protected final static String DEFAULT_ATTRIBUTE="default";
    protected final static String EXTERNAL_ATTRIBUTE="external";
    
    public final static String PAPILLONAXI="Papillon_axi";
	
    /** 
	 * The initializeDictionaryCache method performs a database query to
	 * initialize the dictionary cache
	 *
	 * @exception PapillonBusinessException
	 *   If there is a problem retrieving disc information.
	 */
    public static void initializeDictionaryCache() 
	throws PapillonBusinessException {
		
		try {
			
			// Initialize cache
			DictionaryCache.dictionaryCacheInit();
			
			// Perform query
			DictionaryQuery query = new DictionaryQuery(CurrentDBTransaction.get());
			query.addOrderByName(true);
			DictionaryDO[] DOarray = query.getDOArray();
			
			// Add dictionaries in cache (keys are the dictionary names)
			for ( int i = 0; i < DOarray.length; i++ ) {
				Dictionary dict = new Dictionary(DOarray[i]);
				DictionaryCache.putDictionaryInCache(dict.getName(), dict);
			}
			
		}catch(Exception ex) {
			throw new PapillonBusinessException("Exception in initializeDictionaryCache()", ex);
		}
	}
    
    
    /** 
	 * The newDictionary method create a new dictionary base on dictionary element into the metadata file
	 *
	 * @param dictionary element into the metadata file
	 *
	 * @return Dictionary
	 *
	 * @exception PapillonBusinessException
	 */
    // FIXME: Most attributes should belong to the dml name space. Currently, only the lang attribute do belong to dml...
    public static Dictionary newDictionary(Element dictionary)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		
		// Cette méthode dépend du schéma des dictionnaires.
		String fullname = dictionary.getAttribute("fullname");
		String name = dictionary.getAttribute("name");
		String category = dictionary.getAttribute("category");
		String type = dictionary.getAttribute("type");
		
		NodeList domains = dictionary.getElementsByTagNameNS(DML_URI,"domain");
		String domain = null;
		if ((null != domains) && (domains.getLength() > 0)) {
			domain = domains.item(0).getFirstChild().getNodeValue();
		}
		
        NodeList legals = dictionary.getElementsByTagNameNS(DML_URI,"legal");
        String legal = null;
        if ((null != legals) && (legals.getLength() > 0)) {
            legal = legals.item(0).getFirstChild().getNodeValue();
        }
        PapillonLogger.writeDebugMsg("Legal: " + legal);
        
        NodeList accessNodes = dictionary.getElementsByTagNameNS(DML_URI,"access");
        String access = null;
        if ((null != accessNodes) && (accessNodes.getLength() > 0)) {
            access = accessNodes.item(0).getFirstChild().getNodeValue();
        }
        PapillonLogger.writeDebugMsg("Access: " + access);
        
		NodeList sourceNodes = dictionary.getElementsByTagNameNS(DML_URI,"source-language");
		String sources = "";
		if ((null != sourceNodes) && (sourceNodes.getLength() > 0)) {
			for (int i=0; i<sourceNodes.getLength(); i++) {
				Element tempElt = (Element)sourceNodes.item(i);
				sources = sources + " " + tempElt.getAttributeNS(DML_URI, "lang");
			}
		}
		sources = sources.trim();
		
		NodeList targetNodes = dictionary.getElementsByTagNameNS(DML_URI,"target-language");
		String targets = "";
		if ((null != targetNodes) && (targetNodes.getLength() > 0)) {
			for (int i=0; i<targetNodes.getLength(); i++) {
				Element tempElt = (Element)targetNodes.item(i);
				targets = targets + " " +  tempElt.getAttributeNS(DML_URI, "lang");
			}
		}
		targets = targets.trim();
		
		// FIXME: Should we store the DOM ?
		String xmlCode= XMLServices.NodeToString(dictionary);
		
		//
		Dictionary newDictionary = createUniqueDictionary(name, fullname, category, type, domain, legal, access, sources, targets, xmlCode);
		
		// Add in cache
		DictionaryCache.putDictionaryInCache(newDictionary.getName(), newDictionary);
		
		//
		return newDictionary;
	}
	
    
    /** 
	 * The createUniqueDictionary method search for an existing dictionary :
	 * If exist, return a null dictionary and write a message into the interface with PapillonLogger.
	 * If not, create the new dictionary in the database (Dictionary method) and return it.
	 *
	 *
	 * @param name
	 * @param fullname
	 * @param category
	 * @param type
	 * @param domain
     * @param legal
     * @param access
	 * @param sources
	 * @param targets
	 * @param xmlCode
	 *
	 * @return Dictionary : an unique dictionary
	 *
	 * @exception PapillonBusinessException
	 */
    protected static Dictionary createUniqueDictionary (String name,
                                                        String fullname,
                                                        String category,
                                                        String type,
                                                        String domain,
                                                        String legal,
                                                        String access,
                                                        String sources,
                                                        String targets,
                                                        String xmlCode)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
        Dictionary myDictionary = null;
        if ((name!=null) && (fullname!=null) && (category!=null) && (type!=null) && (sources!=null) && (xmlCode!=null))
        {
            // search for an existing dictionary
            myDictionary=DictionariesFactory.getDictionaryByName(name);
            if (myDictionary == null || myDictionary.isEmpty()) {//the dictionary is new, create it...
                myDictionary=new Dictionary();
                myDictionary.setName(name);
                myDictionary.setFullName(fullname);
                myDictionary.setCategory(category);
                myDictionary.setType(type);
                myDictionary.setDomain(domain);
                myDictionary.setLegal(legal);
                myDictionary.setAccess(access);
                myDictionary.setSourceLanguages(sources);
                myDictionary.setTargetLanguages(targets);
                myDictionary.setXmlCode(xmlCode);
            } else {
                PapillonLogger.writeDebugMsg("Dico deja existant dans la base");
            }
        } else {
            PapillonLogger.writeDebugMsg("Could not retrieve one of the mandatory elements:");
            PapillonLogger.writeDebugMsg("dictionary name = " + name);
            PapillonLogger.writeDebugMsg("dictionary full name = " + fullname);
            PapillonLogger.writeDebugMsg("dictionary category = " + category);
            PapillonLogger.writeDebugMsg("dictionary type = " + type);
            PapillonLogger.writeDebugMsg("dictionary sources = " + sources);
            if (xmlCode == null) PapillonLogger.writeDebugMsg("dictionary xmlcode is null.");
        }
        return myDictionary;
    }
    
	
	
    /** 
	 * The parseDictionaryMetadata method parse the url file, create the new dictionary in the database, load volumes and entries if specify and return the new dictionary.
	 *
	 * @param fileURL
	 * @param loadVolumes
	 * @param loadEntries
	 * @param logContribs
	 *
	 * @return Dictionary
	 *
	 * @exception PapillonBusinessException
	 */
    public static Dictionary parseDictionaryMetadata
    (URL fileURL, boolean loadVolumes, boolean loadEntries, boolean logContribs)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        Document docXml = XMLServices.buildDOMTree(fileURL);
        return parseDictionaryMetadata(docXml, fileURL, loadVolumes, loadEntries, logContribs);
    }

    public static Dictionary parseDictionaryMetadata
	(Document docXml, URL fileURL, boolean loadVolumes, boolean loadEntries, boolean logContribs)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		Dictionary myDict = null;
		//PapillonLogger.writeDebugMsg("The xml");
		//PapillonLogger.writeDebugMsg(XMLServices.xmlCode(docXml));
		
		// on recupere l'element dictionary
		Element dictionary = (Element)docXml.getElementsByTagNameNS(DML_URI,DICTIONARY_TAG).item(0);
		
		if (dictionary==null) {
			throw new fr.imag.clips.papillon.business.PapillonBusinessException("Error: the XML file does not begin with the tag: " + DICTIONARY_TAG + "!");
		}
		
		// ajout du dico ds la table.
		myDict = DictionariesFactory.newDictionary(dictionary);
		if (null != myDict) {
            myDict.createUserGroups();
			// DONE: allow several stylesheets in metadata
			NodeList stylesheets =(NodeList)docXml.getElementsByTagNameNS(DML_URI,XSLSHEET_REF_TAG);
			for (int i=0; i<stylesheets.getLength(); i++) {
				Element stylesheet = (Element) stylesheets.item(i);
				
                if (null != stylesheet && null != fileURL) {
					String ref = stylesheet.getAttributeNS(XLINK_URI,HREF_ATTRIBUTE);
					String name = stylesheet.getAttribute(NAME_ATTRIBUTE);
					if (name == null) {
						name = "";
					}
					String isDefault = stylesheet.getAttribute(DEFAULT_ATTRIBUTE);
					boolean isDefaultXsl = (null != isDefault && isDefault.equals("true"));
					String isExternal = stylesheet.getAttribute(EXTERNAL_ATTRIBUTE);
					boolean isExternalXsl = (null != isExternal && isExternal.equals("true"));
					URL resultURL = null;
					try  {
						resultURL = new URL(fileURL,ref);
					}
					catch (java.io.IOException ex) {
						throw new PapillonBusinessException("java.io.IOException", ex);
					}
					String xslString = fr.imag.clips.papillon.business.xsl.XslSheetFactory.parseXslSheet(resultURL);
					XslSheetFactory.AddXslSheet(name, myDict.getName(), null , null, xslString, isDefaultXsl, isExternalXsl);
				}
			}
            stylesheets = (NodeList) docXml.getElementsByTagNameNS(XSLT_URI,XSLSHEET_TAG);
            for (int i=0; i<stylesheets.getLength(); i++) {
                Element stylesheet = (Element) stylesheets.item(i);
                
                if (null != stylesheet) {
                    String name = stylesheet.getAttribute(NAME_ATTRIBUTE);
                    if (name == null) {
                        name = "";
                    }
                    String isDefault = stylesheet.getAttribute(DEFAULT_ATTRIBUTE);
                    boolean isDefaultXsl = (null != isDefault && isDefault.equals("true"));
                    String xslString = XMLServices.NodeToString(stylesheet);
                    XslSheetFactory.AddXslSheet(name, myDict.getName(), null , null, xslString, isDefaultXsl, false);
                }
            }
			if (loadVolumes || loadEntries) {
				NodeList links = dictionary.getElementsByTagNameNS(DML_URI,VOLUME_REF_TAG);
				if (null != links && links.getLength() > 0) {
					for (int i=0; i<links.getLength(); i++) {
						Element tempElt = (Element)links.item(i);
						String ref = tempElt.getAttributeNS(XLINK_URI,HREF_ATTRIBUTE);
						URL resultURL = null;
						try  {
							resultURL = new URL(fileURL,ref);
						}
						catch (java.io.IOException ex) {
							throw new PapillonBusinessException("java.io.IOException", ex);
						}
						VolumesFactory.parseVolumeMetadata(myDict, resultURL, loadEntries, logContribs);
					}
				}
			}
		}
		return myDict;
	}
	
    public static void createUserGroup(Element dictionary, String userGroupString, String userGroupXmlName) throws PapillonBusinessException {
        NodeList usersNodeList = (NodeList) dictionary.getElementsByTagNameNS(DML_URI,userGroupXmlName);
        if ((null != usersNodeList) && (usersNodeList.getLength() > 0)) {
            Element usersElement = (Element) usersNodeList.item(0);
            usersNodeList = (NodeList) usersElement.getElementsByTagNameNS(DML_URI,"user-ref");
            for (int i=0; i<usersNodeList.getLength(); i++) {
                Element userElement = (Element) usersNodeList.item(i);
                String userString = userElement.getAttribute("name");
                User aUser = UsersFactory.findUserByLogin(userString);
                if (aUser!= null && !aUser.isEmpty()) {
                    Group aGroup = GroupsFactory.findGroupByName(userGroupString);
                    if (aGroup == null || aGroup.isEmpty()) {
                        GroupAnswer aGroupAnswer = GroupsFactory.createUniqueGroup(userGroupString, aUser.getPassword(),aUser.getPassword(), aUser.getLogin());
                        if (!aGroupAnswer.isEmpty()) {
                            aGroup = aGroupAnswer.getGroup();
                        }
                    }
                    if (aGroup != null && !aGroup.isEmpty()) {
                        aGroup.addUser(aUser.getLogin());
                        //PapillonLogger.writeDebugMsg("Group " + aGroup.getName() + " add user " + aUser.getLogin());
                        aGroup.save();
                        aUser.addGroup(aGroup.getName());
                        //PapillonLogger.writeDebugMsg("User " + aUser.getLogin() + " add group " + aGroup.getName());
                       aUser.save();
                        //PapillonLogger.writeDebugMsg("User " + aUser.getLogin() + " saved.");
                    }
                }
            }
        }
    }
    
    /**
	 * The getDictionaryByName method search a dictionary by its name.
	 *
	 * @param dictionary name
	 *
	 * @return Dictionary
	 *
	 * @exception PapillonBusinessException
	 */
    public static Dictionary getDictionaryByName(String name)
	throws PapillonBusinessException {
		
		//
		return DictionaryCache.getDictionaryInCache(name);
		
	}
    
    /** 
	 * The getDictionaryByHandle method search a dictionary by its handle.
	 *
	 * @param dictionary name
	 *
	 * @return Dictionary
	 *
	 * @exception PapillonBusinessException
	 */
    public static Dictionary getDictionaryByHandle(String handle)
	throws PapillonBusinessException {
		
		//
		return DictionaryCache.getDictionaryInCacheByHandle(handle);
		
	}
	
    
    /** 
	 * The getDictionariesArray method return the collection of dictionaries existing in the system.
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
	 */
    public static Collection getDictionariesArray()
	throws PapillonBusinessException {
		try {
			Collection dictionaries = DictionaryCache.getDictionariesInCache();
			//Collection.sort(dictionaries,new fr.imag.clips.papillon.business.utility.DictionaryComparator());
			return dictionaries;
		}catch(Exception ex) {
			throw new PapillonBusinessException("Exception in getDictionariesArray()", ex);
		}
	}
    
	
    /** 
	 * The getAllDictionariesEntriesCollection method return a collection of entries.
	 *
	 * @param Vector
	 * @param Vector
	 * @param String
	 * @param User
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
	 */
    // used only in DictEngine
    // FIXME: Test this, shouldn't the source language be known ?
    // FIXME: Why Keys1 and Keys2 ? What are they ?
    public static Collection getAllDictionariesEntriesCollection(Vector Keys1,
																 Vector Keys2,
																 String anyContains,
																 User user,
																 int offset,
																 int limit)
	throws PapillonBusinessException {
		Vector entries = new Vector();
		Collection resources = getDictionariesArray();
		for (Iterator iter = resources.iterator(); iter.hasNext();) {
			Dictionary dict = (Dictionary)iter.next();
			entries.addAll(getDictionaryEntriesCollection(dict,
														  null,
														  null,
														  Keys1,
														  Keys2,
														  anyContains,
														  user,
														  offset,
														  limit));
		}
		
		//
		return entries;
	}
	
    
    /** 
	 * The getDictionariesEntriesCollection method return a collection of entries.
	 *
	 * @param String []
	 * @param String
	 * @param Collection
	 * @param Vector
	 * @param Vector
	 * @param String
	 * @param User
	 * @param int
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
	 */
    // FIXME: Why Keys1 and Keys2 ? What are they ?
    // used only in ConsultExpert
    public static Collection getDictionariesEntriesCollection(Collection resources,
															  String source,
															  Collection targets,
															  Vector Keys1,
															  Vector Keys2,
															  String anyContains,
															  User user,
															  int offset, int limit)
	throws PapillonBusinessException {
		Vector entries = new Vector();
		
		if (null != resources) {
			for (Iterator iter = resources.iterator(); iter.hasNext();) {
				Collection myColl = getDictionaryNameEntriesCollection((String)iter.next(),
																	   source,
																	   targets,
																	   Keys1,
																	   Keys2,
																	   anyContains,
																	   user,
																	   offset,
																	   limit);
				if (myColl!=null) {
					entries.addAll(myColl);
				}
			}
		}
		return entries;
	}
	
    
    /** 
	 * The getDictionaryNameEntriesCollection method return a collection of entries.
	 *
	 * @param String
	 * @param String
	 * @param Collection
	 * @param Vector
	 * @param Vector
	 * @param String
	 * @param User
	 * @param int
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
	 */
    // Keys1 = keys and Keys2=clauses (query written by hands)
    public static Collection getDictionaryNameEntriesCollection(String resource,
																String source,
																Collection targets,
																Vector Keys1,
																Vector Keys2,
																String anyContains,
																User user,
																int offset,
																int limit) 
	throws PapillonBusinessException {
		Dictionary dict = getDictionaryByName(resource);
		return getDictionaryEntriesCollection(dict, source,
											  targets,Keys1, 
											  Keys2, anyContains, 
											  user, offset, limit);
	}
	
    
    /** 
	 * The getDictionaryEntriesCollection method return a collection of entries.
	 * Returns a collection of results for a set of headword ???
	 *
	 * @param String
	 * @param String
	 * @param Collection
	 * @param Vector
	 * @param Vector
	 * @param String
	 * @param User
	 * @param int
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
	 */    
    // used only in DictEngine
    public static Collection getDictionaryEntriesCollection(Dictionary dict,
                                                            String source,
                                                            Collection targets,
                                                            Vector Keys1,
                                                            Vector Keys2,
                                                            String anyContains,
                                                            User user,
                                                            int offset,
															int limit)
	throws PapillonBusinessException {
        //	Collection qrset = new HashSet();				
        Collection qrset = new Vector();
        if ((null != dict)
			&& ((source ==null) || dict.getSourceLanguagesArray().contains(source))
			// FIXME: up to now, the preceding statement was always true. I changed it to intersect...
			&& ((targets == null) || Utility.intersect(targets, dict.getTargetLanguagesArray()))) {
            //
            for (Iterator iter = VolumesFactory.getVolumesArray(dict.getName(), source, null).iterator(); iter.hasNext();)
            {
                Volume myVolume = (Volume) iter.next();
				
                // FIXME: get the limit argument
                Vector entriesVector = VolumeEntriesFactory.getVolumeEntriesVector(dict, myVolume, Keys1, Keys2, anyContains, offset, limit);
				
				//System.out.println("entriesVector.size: " + entriesVector.size());
                //FIXME: hack for targets array. If the array is null, it means that all targets are asked
                if (targets == null) {
                    targets = myVolume.getTargetLanguagesArray();
                }
                //
                for (Iterator iter2 = entriesVector.iterator(); iter2.hasNext();) {
                    //
                    VolumeEntry ve = (VolumeEntry) iter2.next();
                    qrset.addAll(expandResult(ve, targets, user));
                }
            }
        }
		//System.out.println("qrset.size: " + qrset.size());
        
        return qrset;
    }
	
	
    /** 
	 * The getAllDictionariesReverseEntriesCollection method return a collection of entries.
	 *
	 * @param Vector
	 * @param Vector
	 * @param String
	 * @param User
	 * @param int
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
	 */ 
	//    protected static Collection getAllDictionariesReverseEntriesCollection(Vector Keys1,
	//																 Vector Keys2,
	//																 String anyContains,
	//																 User user)
	//        throws PapillonBusinessException {
	//            Vector entries = new Vector();
	//            Collection resources = getDictionariesArray();
	//            for (Iterator iter = resources.iterator(); iter.hasNext();) {
	//                Dictionary dict = (Dictionary)iter.next();
	//                entries.addAll(getDictionaryReverseEntriesCollection(dict,
	//                                                                     null,
	//                                                                     null,
	//                                                                     Keys1,
	//                                                                     Keys2,
	//                                                                     anyContains,
	//                                                                     user,
	//                                                                     0));
	//            }
	//
	//            //
	//            return entries;
	//        }
	
    
    /** 
	 * The getDictionariesReverseEntriesCollection method return a collection of entries.
	 *
	 * @param String []
	 * @param String
	 * @param String []
	 * @param Vector
	 * @param Vector
	 * @param String
	 * @param User
	 * @param int
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
     * @deprecated 
	 */  
    // used only in ConsultExpert
    public static Collection getDictionariesReverseEntriesCollection(Collection resources,
																	 String source,
																	 Collection targets,
																	 Vector Keys1,
																	 Vector Keys2,
																	 String anyContains,
																	 User user,
																	 int offset)
	throws PapillonBusinessException {
		Vector entries = new Vector();
		
		if (null != resources) {
			for (Iterator iter = resources.iterator(); iter.hasNext();) {
				Collection myColl = getDictionaryNameReverseEntriesCollection((String)iter.next(),
																			  source,
																			  targets,
																			  Keys1,
																			  Keys2,
																			  anyContains,
																			  user,
																			  offset);
				if (myColl!=null) {
					entries.addAll(myColl);
				}
			}
		}
		return entries;
	}
	
    
    /** 
	 * The getDictionaryNameReverseEntriesCollection method return a collection of entries.
	 *
	 * @param String
	 * @param String
	 * @param String []
	 * @param Vector
	 * @param Vector
	 * @param String
	 * @param User
	 * @param int
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
	 */ 
    protected static Collection getDictionaryNameReverseEntriesCollection(String resource,
																		  String source,
																		  Collection targets,
																		  Vector Keys1,
																		  Vector Keys2,
																		  String anyContains,
																		  User user,
																		  int offset) 
	throws PapillonBusinessException {
		Dictionary dict = getDictionaryByName(resource);
		return getDictionaryReverseEntriesCollection(dict, source,
													 targets,Keys1, 
													 Keys2, anyContains,
													 user, offset);
	}
	
	
    /** 
	 * The getDictionaryReverseEntriesCollection method return a collection of entries.
	 *
	 * @param Dictionary
	 * @param String
	 * @param String []
	 * @param Vector
	 * @param Vector
	 * @param String
	 * @param User
	 * @param int
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
	 */ 
	protected static Collection getDictionaryReverseEntriesCollection(Dictionary dict,
																	  String source,
																	  Collection targets,
																	  Vector theKeys1,
																	  Vector theKeys2,
																	  String anyContains,
																	  User user,
																	  int offset) 
	throws PapillonBusinessException {
		Collection entriesCollection = null;
		
		//
		if (null != dict && dict.getTargetLanguagesArray().contains(source)) {
			
			//
			entriesCollection = (Collection) new Vector();
			
			//
			for (Iterator iter = VolumesFactory.getVolumesArray(dict.getName(), null, null).iterator(); iter.hasNext();) {
				
				// FIXME it depends on the architecture of the dictionaries !
				Volume myVolume = (Volume)iter.next();
				if (myVolume.IsReverseLookup() 
					&& !myVolume.getName().equals(PAPILLONAXI)
					&& myVolume.getTargetLanguagesArray().contains(source)
					&& targets.contains(myVolume.getSourceLanguage())
					&& theKeys1.size()>0) {
					PapillonLogger.writeDebugMsg("Volume Reverse Lookup: " + myVolume.getName() + " " + source + " -> " + myVolume.getSourceLanguage());
					for (java.util.Enumeration enumKeys = theKeys1.elements(); enumKeys.hasMoreElements();) {
						//myKey[0] = key
						//myKey[1] = lang
						//myKey[2] = value
						String[] myKey = (String[]) enumKeys.nextElement();
						if (myKey[0].equals(Volume.CDM_headword)) {
							myKey[0]= Volume.CDM_translation;
						}
					}
					
					// FIXME: get the limit argument
					Vector entriesVector = VolumeEntriesFactory.getVolumeEntriesVector(dict, myVolume, theKeys1, theKeys2, anyContains, offset, 0);
					// TODO: Do we have to expand translations in this case ? Is reverse lookup only for direct dictionaries ?
					for (Iterator iter2 = entriesVector.iterator(); iter2.hasNext();) {
						
						//
						VolumeEntry ve = (VolumeEntry) iter2.next();
						QueryResult qr = new QueryResult(QueryResult.REVERSE_UNIQUE_RESULT, ve);
						entriesCollection.add(qr);
					}
				}
			}
		}
		return entriesCollection;
	}
	
    
    /** 
	 * The getVolumeEntriesCollection method return a collection of Volume Entries (WARN: not a collection of QueryResults).
	 *
	 * @param String
	 * @param User
	 * @param Vector
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
     * @deprecated 
	 */ 
    // FIXME: used in EditAxie and old EditEntry ... remove ?
	public static Collection getVolumeEntriesCollection(String volumeName, User user, Vector Keys) throws PapillonBusinessException {
		Collection entriesCollection = null;
		Vector entriesTable = null;
		Volume volume = null;
		Dictionary dict = null;
		try {
			volume = VolumesFactory.getVolumeByName(volumeName);
			dict = DictionariesFactory.getDictionaryByName(volume.getDictname());
		}
		catch(Exception ex) {
			throw new PapillonBusinessException("Exception in getVolumeNameEntriesVector()", ex);
		}
		entriesTable = VolumeEntriesFactory.getVolumeEntriesVector(dict, volume,Keys, null, null,0, 0);
		return entriesTable;
	}
    
    /*
	 //*********************************************
	 // New set of queries, using QueryParameters
	 // These queries are different, because they do not use a global notion of source language
	 // The source language is attached in the keys of the parameter, meaning we can ask
	 // for entries with a french headword and a german headword...
	 // Question is: what does this mean (german headword is translation or it is a term bank ?)
	 // Problem: One meaning is find an entry with xxx in french, connected to an entry with yyy in german
	 //          Which is meaningfull in the case of a pivot dictionary, but which means that we should detect it
	 //          and make the appropriate query on the axie level...
	 //*********************************************
	 public static Collection doQuery(QueryParameter qp, User user) throws PapillonBusinessException {
	 // FIXME: what about reverse access ?
	 Collection qrset = new Vector();
	 try {
	 // Iterate on all the dictionaries and then on all the volumes...
	 for (int id = 0; id < qp.getDictionaryNames().length; id++) {
	 Dictionary d = getDictionaryByName(qp.getDictionaryNames()[id]);
	 // FIXME: There, I should determine the type of dictionary.
	 // FIXME: and perform the corresponding request
	 Volume[] volumes = VolumesFactory.getVolumesArray(d.getName());
	 for (int iv=0; iv < volumes.length; iv++) {
	 Volume vol = volumes[iv];
	 Vector entriesVector = VolumeEntriesFactory.getVolumeEntriesVector(d, vol, qp.getCriteria(), null, null, qp.getOffset(), qp.getLimit());
	 Iterator iter = entriesVector.iterator();
	 while (iter.hasNext()) {
	 VolumeEntry ve = (VolumeEntry) iter.next();
	 qrset.addAll(expandResult(ve,qp.getTargets(),user));
	 }
	 
	 }
	 }
	 } catch (PapillonBusinessException e) {
	 throw e;
	 }
	 return qrset;
	 }
	 */
    
    
    //*********************************************
    
	
	
	/** 
     * The findAnswerAndTranslations method return a collection of QueryResult.
     *
     * @param String
     * @param String
     * @param String []
     * @param User
     *
     * @return Collection
     *
     * @exception PapillonBusinessException
     */ 
	public static Collection findAnswerAndTranslations(String volumeName, String handle, Collection targets, User user) throws PapillonBusinessException {
        
        // FIXME: should be VolumeEntry...
		VolumeEntry myAnswer = VolumeEntriesFactory.findEntryByHandle(volumeName, handle);
		
        //FIXME: hack for targets array. If the array is null, it means that all targets are asked
		if (targets==null || (targets !=null && targets.size() == 0)) {
			return expandResult(myAnswer, myAnswer.getVolume().getTargetLanguagesArray(), user);
		} else {
			return expandResult(myAnswer, Utility.ArrayIntersection(myAnswer.getVolume().getTargetLanguagesArray(),targets), user);
		}
	}
	
	
    public static Collection expandResult(VolumeEntry ve, Collection targets, User user) 
		throws PapillonBusinessException {
			QueryResult myQR = new QueryResult();
			myQR.addSourceEntry(ve);
			return expandResult(myQR, targets, user);
	}
	
	/** 
     * Create a query result collection that contains all translations of the current entry, for the requested target languages.
     *
     * @param VolumeEntry
     * @param Collection
     * @param User
     *
     * @return Collection
     *
     * @exception PapillonBusinessException
     * @deprecated use expand results with a collection of QueryResults
     */      
    public static Collection expandResult(QueryResult theQR, Collection targets, User user) 
	throws PapillonBusinessException {
        //    
        HashMap allLinks = new HashMap();
		Collection myArrayList = new ArrayList();
		String direction = Link.DIRECTION_DOWN;
 
		VolumeEntry ve = theQR.getSourceEntry();
        //PapillonLogger.writeDebugMsg("expandResult: " + ve.getHeadword());
		Collection realTargets = Utility.ArrayIntersection(ve.getDictionary().getTargetLanguagesArray(), targets);
		String type = ve.getDictionary().getType();
            
		if (type.equals(Dictionary.PIVOT_TYPE)) {
			allLinks.put(ve.getEntryId(),ve);
			direction = Link.DIRECTION_UP;
		}
		else {
			realTargets.remove(ve.getSourceLanguage());
		}
		ArrayList axieLinks = new ArrayList();
		LinkFactory.getLinkedEntriesByEntry(ve, axieLinks, allLinks, realTargets, direction, user);
		theQR.setLexiesHashMap(allLinks);
		myArrayList.add(theQR);
		return myArrayList;
    }
    
	/** 
     * Create a query result collection that contains all translations of the passed entries, for the requested target languages.
     * If merge is true, it does not duplicate axies, hence if several entries do share the same axie, it will result in only one queryResult.
     *
     * @param ves
     * @param targets
     * @param user
     * @param mergeAxies
     *
     * @return Collection of QueryResult
     *
     * @exception PapillonBusinessException
     */      
	public static Collection expandResults(Collection ves, Collection targets, User user, boolean mergeAxies) 
	throws PapillonBusinessException {
		Collection resultsList = new ArrayList();
		// Add axies and remove duplicates entries
		//Collection resultsWithAxies = addAxiesAndTranslations(ves, targets, user, mergeAxies);
		Iterator itve = ves.iterator();
        while (itve.hasNext()) {
			QueryResult qr = (QueryResult) itve.next() ;
			resultsList.addAll(expandResult(qr, targets, user));
		}
		//
		return resultsList;
	}
	
	private static QueryResult findQueryResultWithAxieId(Collection res, String aid) throws PapillonBusinessException {
		Iterator it = res.iterator();
		while (it.hasNext()) {
			QueryResult cr = (QueryResult) it.next();
			if (cr.getResultKind() == QueryResult.AXIE_COLLECTION_RESULT &&
				cr.getResultAxie().getEntryId().equals(aid)) return cr;
		}
		return null;
	}
		
	private static void addPivaxTranslations(QueryResult qr, Collection targets) throws PapillonBusinessException {
		
		String dictName = qr.getSourceEntry().getDictionaryName();
		String sourceLang = qr.getSourceEntry().getSourceLanguage();
		String sourceAxemeLang = sourceLang.toUpperCase();
		String axiLang = "axi";
		
		HashMap resLexies = new HashMap();
		
		java.util.Collection axemesVolumesCollection = VolumesFactory.getVolumesArray(dictName,sourceAxemeLang,null);
		if (axemesVolumesCollection !=null && axemesVolumesCollection.size()>0) {
			Volume axemeVolume = ((Volume)axemesVolumesCollection.iterator().next());
			//PapillonLogger.writeDebugMsg("Pivax entries: volume src axemes: " + axemeVolume.getName() + " entryId: " + qr.getSourceEntry().getEntryId());
			java.util.Vector myKeys = new java.util.Vector();
			String[] Word = new String[4];
			Word[0] = Volume.CDM_translationReflexie;
			Word[1] = sourceLang;
			Word[2] = qr.getSourceEntry().getEntryId();
			Word[3] = QueryBuilder.EQUAL;
			myKeys.add(Word);
			java.util.Collection axemesVector = IndexFactory.getIndexEntriesVector(axemeVolume.getIndexDbname(),
																			   myKeys,
																			   IndexFactory.ORDER_DESCENDING,
																			   100,
																			   0);
			for (java.util.Iterator myIterator = axemesVector.iterator(); myIterator.hasNext(); ) {
				Index axemeEntry = (Index) myIterator.next();
				//PapillonLogger.writeDebugMsg("Pivax entries: volume src axemes: " + axemeVolume.getName() + " axemeId: " + axemeEntry.getEntryId());
				java.util.Collection axemesResultVector = IndexFactory.getIndexVectorByEntryId(axemeVolume, ""+ axemeEntry.getEntryId());
				for (java.util.Iterator myIterator2 = axemesResultVector.iterator(); myIterator2.hasNext(); ) {
					Index axemeResultEntry = (Index) myIterator2.next();
					if (axemeResultEntry.getKey().equals(Volume.CDM_entryId)) {
						java.util.Collection axiesVolumesCollection = VolumesFactory.getVolumesArray(dictName,axiLang,null);
						if (axiesVolumesCollection !=null && axiesVolumesCollection.size()>0) {
							Volume axieVolume = ((Volume)axiesVolumesCollection.iterator().next());
							//PapillonLogger.writeDebugMsg("Pivax entries: volume axie: " + axieVolume.getName() + " axeme: " + axemeResultEntry.getValue());
							myKeys = new java.util.Vector();
							Word = new String[4];
							Word[0] = Volume.CDM_translationReflexie;
							Word[1] = sourceAxemeLang;
							Word[2] = axemeResultEntry.getValue();
							Word[3] = QueryBuilder.EQUAL;
							myKeys.add(Word);
							java.util.Collection axiesVector = IndexFactory.getIndexEntriesVector(axieVolume.getIndexDbname(),
																							  myKeys,
																							  IndexFactory.ORDER_DESCENDING,
																							  100,
																							  0);
							for (java.util.Iterator myIterator3 = axiesVector.iterator(); myIterator3.hasNext(); ) {
								Index axieEntry = (Index) myIterator3.next();
								//PapillonLogger.writeDebugMsg("Pivax entries: volume axie: " + axieVolume.getName() + " axieEntry: " + axieEntry.getValue());
								java.util.Collection axiesResultCollection = IndexFactory.getIndexVectorByEntryId(axieVolume, ""+ axieEntry.getEntryId());
								for (java.util.Iterator myIterator4 = axiesResultCollection.iterator(); myIterator4.hasNext(); ) {
									Index axieResultEntry = (Index) myIterator4.next();
									String targetAxemeLang = axieResultEntry.getLang();
									String targetLang = targetAxemeLang.toLowerCase();
									if (axieResultEntry.getKey().equals(Volume.CDM_translationReflexie) && targets.contains(targetLang)) {
										java.util.Collection unlAxemesVolumesCollection = VolumesFactory.getVolumesArray(dictName,targetAxemeLang,null);
										if (unlAxemesVolumesCollection !=null && unlAxemesVolumesCollection.size()>0) {
											Volume unlAxemeVolume = ((Volume)unlAxemesVolumesCollection.iterator().next());
											//PapillonLogger.writeDebugMsg("Pivax entries: volume trg axeme: " + unlAxemeVolume.getName() + " value: " + axieResultEntry.getValue());
											myKeys = new java.util.Vector();
											Word = new String[4];
											Word[0] = Volume.CDM_entryId;
											Word[1] = Volume.DEFAULT_LANG;
											Word[2] = axieResultEntry.getValue();
											Word[3] = QueryBuilder.EQUAL;
											myKeys.add(Word);
											java.util.Collection unlAxemesVector = IndexFactory.getIndexEntriesVector(unlAxemeVolume.getIndexDbname(),
																												  myKeys,
																												  IndexFactory.ORDER_DESCENDING,
																												  100,
																												  0);
											for (java.util.Iterator myIterator5 = unlAxemesVector.iterator(); myIterator5.hasNext(); ) {
												Index unlAxemeEntry = (Index) myIterator5.next();
												java.util.Collection unlAxemesResultVector = IndexFactory.getIndexVectorByEntryId(unlAxemeVolume, ""+unlAxemeEntry.getEntryId());
												//PapillonLogger.writeDebugMsg("Pivax entries: volume trg axeme: " + unlAxemeVolume.getName() + " axemeId: " + unlAxemeEntry.getEntryId());
												for (java.util.Iterator myIterator6 = unlAxemesResultVector.iterator(); myIterator6.hasNext(); ) {
													Index unlAxemeResultEntry = (Index) myIterator6.next();
													if (unlAxemeResultEntry.getKey().equals(Volume.CDM_translationReflexie)) {
														java.util.Collection unlVolumesCollection = VolumesFactory.getVolumesArray(dictName,targetLang,null);
														if (unlVolumesCollection !=null && unlVolumesCollection.size()>0) {
															Volume unlVolume = ((Volume)unlVolumesCollection.iterator().next());
															//PapillonLogger.writeDebugMsg("Pivax entries: volume trg: " + unlVolume.getName() + " value: " + unlAxemeResultEntry.getValue());
															myKeys = new java.util.Vector();
															Word = new String[4];
															Word[0] = Volume.CDM_entryId;
															Word[1] = Volume.DEFAULT_LANG;
															Word[2] = unlAxemeResultEntry.getValue();
															Word[3] = QueryBuilder.EQUAL;
															myKeys.add(Word);
															java.util.Collection unlLexiesVector = IndexFactory.getIndexEntriesVector(unlVolume.getIndexDbname(),
																																  myKeys,
																																  IndexFactory.ORDER_DESCENDING,
																																  100,
																																  0);
															for (java.util.Iterator myIterator7 = unlLexiesVector.iterator(); myIterator7.hasNext(); ) {
																Index unlIndexEntry = (Index) myIterator7.next();
																VolumeEntry unlEntry = VolumeEntriesFactory.findEntryByHandle(unlVolume.getName(), ""+unlIndexEntry.getEntryId());
																if (unlEntry != null && !unlEntry.isEmpty()) {
																	PapillonLogger.writeDebugMsg("Pivax entries: trg entry: headword: " + unlEntry.getHeadword());
																	resLexies.put(unlEntry.getEntryId(), unlEntry);
																}
																else {
																	PapillonLogger.writeDebugMsg("Pivax entries: Entry null");
																}
																
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		qr.setLexiesHashMap(resLexies);
	}
	
		
	/** 
	 * Find entry by their entry id
	 *
	 * @param String
	 * @param String
	 *
	 * @return IAnswer
	 *
	 * @exception PapillonBusinessException
	 */   
	// FIXME: used QueryRequest !!!
	public static IAnswer findEntryByEntryId(String volumeName, String entryId) throws PapillonBusinessException {
		IAnswer myAnswer = VolumeEntriesFactory.findEntryByEntryId(volumeName, entryId);
		return myAnswer;
		
	}
		
	
	/** 
	 * Returns a collection of Query Results.
	 *
	 * @param QueryResult proto lexie
	 * @param String
	 * @param String []
	 * @param User
	 *
	 * @return Collection
	 *
	 * @exception PapillonBusinessException
	 */   
	protected static QueryResult getDirectResults(QueryResult qr, String source, Collection targets, User myUser) throws PapillonBusinessException {
		//Collection qrset = new HashSet();
		//PapillonLogger.writeDebugMsg("getDirectResults:");
		if (null != qr && null != qr.getSourceEntry()) {
			VolumeEntry mySourceEntry = qr.getSourceEntry();
			String dictName = mySourceEntry.getDictionaryName();
			HashMap resLexies = new HashMap();
			
			// DIRECT TRANSLATION RESULTS + AXIES
			for (Iterator iter = targets.iterator(); iter.hasNext();) {
				String target = (String)iter.next();
				
				if (target != source) {
					PapillonLogger.writeDebugMsg("getDirectResults: " + mySourceEntry.getHeadword() + " target:" + target);
					// get all cdm elements pointing to target entries.
					String[] transIds = mySourceEntry.getTranslationsLexieIds(target);
					
					//
					Collection Volumes = VolumesFactory.getVolumesArray(dictName, target, null);
					if (null != transIds && !Volumes.isEmpty()) {
						
						//
						Volume firstVolume = (Volume)(Volumes.iterator()).next();
						String firstVolumeName = firstVolume.getName();
						for (int j = 0; j < transIds.length; j++) {
							PapillonLogger.writeDebugMsg("getDirectResults: findEntryByEntryId:" + transIds[j]);
							VolumeEntry myEntry = (VolumeEntry) DictionariesFactory.findEntryByEntryId(firstVolumeName, transIds[j]);
							if (myEntry != null && ! myEntry.isEmpty()) {
								PapillonLogger.writeDebugMsg("getDirectResults: targetEntryId:" + myEntry.getEntryId());
								resLexies.put(myEntry.getEntryId(),myEntry);
								PapillonLogger.writeDebugMsg("getDirectResults: target id: " + myEntry.getEntryId());
								
								// pivot volume
								if (target.equals("axi")) {
									for (Iterator iter1 = targets.iterator(); iter1.hasNext();) {
										String target1 = (String)iter1.next();
										
										if (target1 != source && !target1.equals("axi")) {
											// get all cdm elements pointing to target entries.
											String[] transIds1 = myEntry.getTranslationsLexieIds(target1);
											
											Collection Volumes1 = VolumesFactory.getVolumesArray(dictName, target1, null);
											if (null != transIds1 && !Volumes1.isEmpty()) {
												Volume firstVolume1 = (Volume)(Volumes1.iterator()).next();
												String firstVolumeName1 = firstVolume1.getName();
												for (int j1 = 0; j1 < transIds1.length; j1++) {
													VolumeEntry myEntry1 = (VolumeEntry) DictionariesFactory.findEntryByEntryId(firstVolumeName1, transIds1[j1]);
													if (myEntry1 != null && ! myEntry1.isEmpty()) {
														resLexies.put(myEntry1.getEntryId(),myEntry1);
													}
												}
											}
										}
									}
								}
							} 
						}
					}
				}
				
			}
			// PIVOT AXIE RESULTS
			// Done in the Formatter
			qr.setLexiesHashMap(resLexies);
		}
		//PapillonLogger.writeDebugMsg("end of getDirectResults");
		
		return qr;
	}


	public static HashMap expandResultsForRest(Collection ves, Collection targets, User user, boolean mergeAxies) 
	throws PapillonBusinessException {
		HashMap resultsList = new HashMap();
		// Add axies and remove duplicates entries
		//Collection resultsWithAxies = addAxiesAndTranslations(ves, targets, user, mergeAxies);
		Iterator itve = ves.iterator();
        while (itve.hasNext()) {
			QueryResult qr = (QueryResult) itve.next() ;
			resultsList = expandResultForRest(qr, targets, user);
		}
		//
		return resultsList;
	}

	private static HashMap expandResultForRest(QueryResult theQR,
			Collection targets, User user) throws PapillonBusinessException{
		 HashMap allLinks = new HashMap();
			String direction = Link.DIRECTION_DOWN;
	 
			VolumeEntry ve = theQR.getSourceEntry();
	        //PapillonLogger.writeDebugMsg("expandResult: " + ve.getHeadword());
			Collection realTargets = Utility.ArrayIntersection(ve.getDictionary().getTargetLanguagesArray(), targets);
			String type = ve.getDictionary().getType();
	            
			if (type.equals(Dictionary.PIVOT_TYPE)) {
				allLinks.put(ve.getEntryId(),ve);
				direction = Link.DIRECTION_UP;
			}
			else {
				realTargets.remove(ve.getSourceLanguage());
			}
			ArrayList axieLinks = new ArrayList();
			LinkFactory.getLinkedMonoEntriesByEntry(ve, axieLinks, allLinks, realTargets, direction, user);
//			PapillonLogger.writeDebugMsg("axieLinks = "+axieLinks);
//			PapillonLogger.writeDebugMsg("allLinks = "+allLinks.values());
			theQR.setLexiesHashMap(allLinks);
			return allLinks;
	}
}

