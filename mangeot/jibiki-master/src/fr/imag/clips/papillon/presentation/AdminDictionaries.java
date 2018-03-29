/*
 *  papillon
 *
 *  Enhydra super-servlet
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.11.2.1  2007/10/30 13:24:50  serasset
 *  BUG 161: delete action now works in CreateEntryInit page.
 *
 *  Revision 1.11  2007/01/16 13:28:31  serasset
 *  Added cache reinitialization when a metadata is modified.
 *
 *  Revision 1.10  2006/08/10 22:17:13  fbrunet
 *  - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 *  - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 *  - Bug correction : +/- in advanced search
 *
 *  Revision 1.9  2006/02/26 14:04:56  mangeot
 *  Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 *  Revision 1.8  2005/12/01 15:34:28  mangeot
 *  MM: I solved the problem of already created tables by creating an sql query for retrieving the table names. If the name already exists, VolumeEntriesFactory.createVolumeTables do not create the tables.
 *  It allows the administrator to delete and reload only the metadata files without dropping the whole data.
 *  The method is ManageDatabase.getTableNames() and it returns a vector with all the table names created by the database user (usually "papillon").
 *
 *  Revision 1.7  2005/11/22 13:21:02  mangeot
 *  I moved the VolumeEntriesFactory.createVolumeTables out of the database transactions in AdminDictionaries.java and Adminvolumes.java because otherwise, it is not possible to reload metadata when the data tables already exist (in this case, the transaction does not commit).
 *
 *  Revision 1.6  2005/06/15 16:48:28  mangeot
 *  Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *  Revision 1.5  2005/05/24 12:51:22  serasset
 *  Updated many aspect of the Papillon project to handle lexalp project.
 *  1. Layout is now parametrable in the application configuration file.
 *  2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 *  3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 *  4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 *  5. It is now possible to give a name to the cookie key in the app conf file
 *  6. Several bug fixes.
 *
 *  Revision 1.4.4.2  2005/06/01 15:20:33  mangeot
 *  Added a boolean for contributionslog
 *
 *  Revision 1.4.4.1  2005/05/19 17:02:22  mangeot
 *  Importing entries without the contribution element
 *
 *  Revision 1.4  2005/04/11 12:29:59  mangeot
 *  Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 *  Revision 1.3  2005/04/11 08:01:02  fbrunet
 *  Passage en xhtml des ressources Papillon.
 *
 *  Revision 1.2.2.2  2005/03/29 09:50:57  serasset
 *  Added transaction rollback in AdminDictionaries.
 *
 *  Revision 1.2.2.1  2005/03/29 09:41:33  serasset
 *  Added transaction support. Use CurrentDBTransaction class to define a transaction
 *  context in which all db commands will be executed.
 *
 *  Revision 1.2  2004/12/24 14:31:28  mangeot
 *  I merged the latest developments of Papillon5.0 with this version 5.1.
 *  Have to be tested more ...
 *
 *  Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 *  Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 *  There are still bugs in the code.
 *
 *  Revision 1.6  2004/10/28 10:56:21  mangeot
 *  Added the list of connected users on AdminUsers.java,
 *  Added the possibility to sort in columns for some pages
 *  Added persistent preferences for the user. They are saved in the database and retrived when the user reconnects. The user is registered in the enhydra session.
 *
 *  -----------------------------------------------
 *  Papillon Admin page.
 */
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

import fr.imag.clips.papillon.business.message.MessageDBLoader;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.Papillon;
import com.lutris.appserver.server.sql.DBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;

//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// Standard imports
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.Date;
import java.text.DateFormat;
import java.io.*;

import java.net.URL;
import java.net.MalformedURLException;

//pour les dictionary
import fr.imag.clips.papillon.business.dictionary.*;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

/**
 *  Description of the Class
 *
 * @author     serasset
 * @created    December 2, 2004
 */
public class AdminDictionaries extends PapillonBasePO {

    /**
     *  Description of the Field
     */
    protected final static String SEE_PARAMETER = "See";
    /**
     *  Description of the Field
     */
    protected final static String REMOVE_PARAMETER = "Remove";
    /**
     *  Description of the Field
     */
    protected final static String REMOVE_ALL_PARAMETER = "RemoveAll";

    /**
     *  Description of the Method
     *
     * @return    Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return true;
    }


    /**
     *  Description of the Method
     *
     * @return    Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        try {
            return this.getUser().isAdmin();
        } catch (PapillonBusinessException ex) {
            this.getSessionData().writeUserMessage("Error getting the authorisation to use this PO.");
        }
        return false;
    }


    /**
     *  Gets the currentSection attribute of the AdminDictionaries object
     *
     * @return    The currentSection value
     */
    protected int getCurrentSection() {
        return NO_SECTION;
    }


    /**
     *  Gets the content attribute of the AdminDictionaries object
     *
     * @return                                                     The content
     *      value
     * @exception  javax.xml.parsers.ParserConfigurationException  Description
     *      of the Exception
     * @exception  HttpPresentationException                       Description
     *      of the Exception
     * @exception  IOException                                     Description
     *      of the Exception
     * @exception  org.xml.sax.SAXException                        Description
     *      of the Exception
     * @exception  javax.xml.transform.TransformerException        Description
     *      of the Exception
     */
    public Node getContent()
             throws javax.xml.parsers.ParserConfigurationException, HttpPresentationException,
            IOException, org.xml.sax.SAXException, javax.xml.transform.TransformerException, PapillonBusinessException {

        // Création du contenu
        AdminDictionariesXHTML content = (AdminDictionariesXHTML) MultilingualXHtmlTemplateFactory.createTemplate("AdminDictionariesXHTML", this.getComms(), this.getSessionData());

        HttpPresentationRequest req = this.getComms().request;
        // If the page is called with parameters, take the requested action
        if (req.getParameterNames().hasMoreElements()) {
            
            //TEMPORAIRE :avec l URL
            //AJOUT DE DICO
            String userMessage = "";
            String urlString = req.getParameter(AdminDictionariesXHTML.NAME_url);
            //
            if (null != urlString && !urlString.equals("")) {
                // The user asked for a dictionary to be uploaded
                userMessage = handleDictionaryAddition(req);
            
            //
            } else if (null != req.getParameter(SEE_PARAMETER)) {
                String name = req.getParameter(SEE_PARAMETER);
                Dictionary dict = DictionariesFactory.getDictionaryByHandle(name);
                //adding an XML file
                addXml(content, dict.getXmlCode());
                userMessage = "Viewing Dictionary " + dict.getName() + " XML metadata";
            //
            } else if (null != req.getParameter(REMOVE_PARAMETER)) {
                String name = req.getParameter(REMOVE_PARAMETER);
                Dictionary dict = DictionariesFactory.getDictionaryByHandle(name);
                dict.delete();
                Papillon.initializeAllCaches();
                userMessage = "Dictionary " + dict.getName() + " metadata  erased...";
            
            //
            } else if (null != req.getParameter(REMOVE_ALL_PARAMETER)) {
                String name = req.getParameter(REMOVE_ALL_PARAMETER);
                Dictionary dict = DictionariesFactory.getDictionaryByHandle(name);
                // here we should use a truncate sql statement with volume.getDbname();
                dict.deleteAll();
                Papillon.initializeAllCaches();
                userMessage = "Dictionary " + dict.getName() + " metadata and volumes erased...";
            } else {
                String errorMessage = "Error: Wrong arguments";
                this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,errorMessage);
            }
            
            //
            if (userMessage != null && userMessage != "") {
                PapillonLogger.writeDebugMsg(userMessage);
                this.getSessionData().writeUserMessage(userMessage);
            }
        }

        //adding the content of the dictionaries table
        addDictionariesArray(content);

        //On rend le contenu correct
        return content.getElementFormulaire();
    }

    protected String handleDictionaryAddition(HttpPresentationRequest req) throws PapillonBusinessException, HttpPresentationException, java.net.MalformedURLException {
        String userMessage;
        String urlString = req.getParameter(AdminDictionariesXHTML.NAME_url);
        if (null != urlString && urlString.charAt(0) == '/') {
            urlString = "file:" + urlString;
        }
		URL myURL = null;
		try  {
			myURL = new URL(urlString);
			String parseVolumesString = req.getParameter(AdminDictionariesXHTML.NAME_AddVolumes);
			boolean parseVolumes = (parseVolumesString!=null && !parseVolumesString.equals(""));
			String parseEntriesString = req.getParameter(AdminDictionariesXHTML.NAME_AddVolumesAndEntries);
			boolean parseEntries = (parseEntriesString!=null && !parseEntriesString.equals(""));
			String logContribsString = req.getParameter(AdminDictionariesXHTML.NAME_LogContributions);
			boolean logContribs = (logContribsString!=null && !logContribsString.equals(""));
			
			// Create and Register the transaction
            // je supprime la transaction car cela m'empêche d'ajouter plusieurs administrateurs dans le groupe des admins du dictionnaire.
			//CurrentDBTransaction.registerNewDBTransaction();
			Dictionary myDict = null;
			try {
				myDict = DictionariesFactory.parseDictionaryMetadata(myURL, parseVolumes, parseEntries, logContribs);
				if (null != myDict && !myDict.isEmpty()) {
                    myDict.save();
					userMessage = "adding " + myDict.getName() + " dictionary" + " // " + myDict.getCategory() + " // " + myDict.getType() + " // " + myDict.getDomain() + " // " + myDict.getLegal() + " // " + myDict.getSourceLanguages() + " // " + myDict.getTargetLanguages();
				} else {
					userMessage = "Ignoring dictionary";
				}
				// everything was correct, commit the transaction...
				/*try {
					((DBTransaction) CurrentDBTransaction.get()).commit();
				} catch (java.sql.SQLException sqle) {
                    userMessage = "AdminDictionaries: SQLException while commiting the transaction.\n";
                    userMessage += sqle.getMessage();
					PapillonLogger.writeDebugMsg(userMessage);
					sqle.printStackTrace();
				}*/
			} catch (PapillonBusinessException e) {
				userMessage = "Problems while adding the specified dictionary.\n";
				userMessage += e.getMessage();
				userMessage += "\nAll changes to the database have been rolled back.";
               this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
				e.printStackTrace();
				/*try {
					((DBTransaction) CurrentDBTransaction.get()).rollback();
				} catch (java.sql.SQLException sqle) {
					PapillonLogger.writeDebugMsg("ad1.91:AdminDictionaries: SQLException while rolling back failed transaction.");
					sqle.printStackTrace();
				}*/
			}/* finally {
				CurrentDBTransaction.releaseCurrentDBTransaction();
			}*/
			Papillon.initializeAllCaches();
		}
		catch (java.io.IOException ex) {
			userMessage = "Problems while adding the specified dictionary. The following URL: "+ urlString +" is malformed;\n";
            userMessage += ex.getMessage();
			ex.printStackTrace();
		}
        return userMessage;
    }

    /**
     *  Adds a feature to the Xml attribute of the AdminDictionaries object
     *
     * @param  xmlString
     *      The feature to be added to the Xml attribute
     * @exception  fr.imag.clips.papillon.business.PapillonBusinessException
     *      Description of the Exception
     */
    protected void addXml(AdminDictionariesXHTML content, String xmlString)
             throws fr.imag.clips.papillon.business.PapillonBusinessException {

        Node xmlNode = XslSheetFactory.applyXslSheetForXml(xmlString);

        //where we must insert the xml
        HTMLElement theXml = content.getElementXml();

        Node theXmlParent = theXml.getParentNode();

        theXmlParent.appendChild(content.importNode(xmlNode, true));
        theXmlParent.removeChild(theXml);
    }



    /**
     *  Adds a feature to the DictionariesArray attribute of the
     *  AdminDictionaries object
     *
     * @exception  fr.imag.clips.papillon.business.PapillonBusinessException
     *      Description of the Exception
     */
    protected void addDictionariesArray(AdminDictionariesXHTML content)
             throws fr.imag.clips.papillon.business.PapillonBusinessException {

        Collection DictsTable = DictionariesFactory.getDictionariesArray();
        //where we must insert the form
        HTMLTableRowElement theRow = content.getElementTemplateRow();
        HTMLElement theName = content.getElementName();
        HTMLElement theCategory = content.getElementCategory();
        HTMLElement theType = content.getElementType();
        HTMLElement theSources = content.getElementSources();
        HTMLElement theTargets = content.getElementTargets();
        HTMLElement theDomain = content.getElementDomain();
        HTMLElement theLegal = content.getElementLegal();
        HTMLAnchorElement theSeeAnchor = content.getElementSeeAnchor();
        HTMLAnchorElement theRemoveAnchor = content.getElementRemoveAnchor();
        HTMLAnchorElement theRemoveAllAnchor = content.getElementRemoveAllAnchor();

        Node theRowParent = theRow.getParentNode();

        theRow.removeAttribute("id");
        theName.removeAttribute("id");
        theCategory.removeAttribute("id");
        theType.removeAttribute("id");
        theSources.removeAttribute("id");
        theTargets.removeAttribute("id");
        theDomain.removeAttribute("id");
        theLegal.removeAttribute("id");
        theSeeAnchor.removeAttribute("id");
        theRemoveAnchor.removeAttribute("id");
        theRemoveAllAnchor.removeAttribute("id");

        //adding the volumes description
        for (Iterator iter = DictsTable.iterator(); iter.hasNext();) {
            Dictionary dict = (Dictionary)iter.next();
            content.setTextName(dict.getName());
            content.setTextCategory(dict.getCategory());
            content.setTextType(dict.getType());
            content.setTextSources(dict.getSourceLanguages());
            content.setTextTargets(dict.getTargetLanguages());
            content.setTextDomain(dict.getDomain());
            content.setTextLegal(dict.getLegal());

            //
            String handle = dict.getHandle();
            
            //
            theSeeAnchor.setHref(this.getUrl() + "?" + SEE_PARAMETER + "=" + handle);
            theRemoveAnchor.setHref(this.getUrl() + "?" + REMOVE_PARAMETER + "=" + handle);
            theRemoveAllAnchor.setHref(this.getUrl() + "?" + REMOVE_ALL_PARAMETER + "=" + handle);

            theRowParent.appendChild(theRow.cloneNode(true));
        }

        theRowParent.removeChild(theRow);
    }

}
