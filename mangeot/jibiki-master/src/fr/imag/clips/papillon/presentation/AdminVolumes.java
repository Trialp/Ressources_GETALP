/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.17  2007/01/16 13:28:31  serasset
 * Added cache reinitialization when a metadata is modified.
 *
 * Revision 1.16  2007/01/05 13:57:26  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.15  2007/01/05 12:57:49  fbrunet
 * Add undo draft method (bug in EditEntry.java : undo after last finish contribution)
 * Modify transformation method
 *
 * Revision 1.14  2006/11/21 22:51:55  fbrunet
 * Correct UIGenerator bug and another minor bugs
 *
 * Revision 1.13  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.12  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.11  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.10  2005/12/01 15:34:28  mangeot
 * MM: I solved the problem of already created tables by creating an sql query for retrieving the table names. If the name already exists, VolumeEntriesFactory.createVolumeTables do not create the tables.
 * It allows the administrator to delete and reload only the metadata files without dropping the whole data.
 * The method is ManageDatabase.getTableNames() and it returns a vector with all the table names created by the database user (usually "papillon").
 *
 * Revision 1.9  2005/11/22 13:21:02  mangeot
 * I moved the VolumeEntriesFactory.createVolumeTables out of the database transactions in AdminDictionaries.java and Adminvolumes.java because otherwise, it is not possible to reload metadata when the data tables already exist (in this case, the transaction does not commit).
 *
 * Revision 1.8  2005/11/14 21:46:26  mangeot
 * *** empty log message ***
 *
 * Revision 1.7.4.2  2005/12/08 13:03:37  fbrunet
 * Add transformation interface to volume administration
 *
 * Revision 1.7.4.1  2005/12/02 10:04:09  fbrunet
 * Add Pre/Post edition processing
 * Add index reconstruction
 * Add new query request
 * Add fuzzy search
 * Add new contribution administration
 * Add xsl transformation volume
 *
 * Revision 1.7  2005/06/15 16:48:28  mangeot
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
 * Revision 1.5.4.7  2005/06/09 11:28:24  mangeot
 * *** empty log message ***
 *
 * Revision 1.5.4.6  2005/06/09 11:07:45  mangeot
 * Deleted the countEntriesCache. entries counts are not cached any more.
 * Fixed a few bugs.
 *
 * Revision 1.5.4.5  2005/06/02 09:13:50  mangeot
 * *** empty log message ***
 *
 * Revision 1.5.4.4  2005/06/01 15:20:33  mangeot
 * Added a boolean for contributionslog
 *
 * Revision 1.5.4.3  2005/06/01 08:38:43  mangeot
 * Multi bug correction + added the possibility of disabling data edition
 * via the Admin.po page
 *
 * Revision 1.5.4.2  2005/05/19 17:02:22  mangeot
 * Importing entries without the contribution element
 *
 * Revision 1.5.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.5  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.4  2005/04/11 08:01:02  fbrunet
 * Passage en xhtml des ressources Papillon.
 *
 * Revision 1.3.2.2  2005/03/31 09:10:31  mangeot
 * (re) added the code for loading the volume metadata only
 *
 * Revision 1.3.2.1  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
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
 *-----------------------------------------------
 * Papillon Admin Volumes page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// Standard imports
import java.io.IOException;
import java.util.Iterator;

// For URLs
import java.net.URL;


import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.Papillon;
import com.lutris.appserver.server.sql.DBTransaction;

//pour les dictionary
import fr.imag.clips.papillon.business.dictionary.*;

// import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.edition.GenerateTemplate;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;


public class AdminVolumes extends PapillonBasePO {

    protected final static String SEE_METADATA_PARAMETER="SeeMetadata";
    protected final static String SEE_SCHEMA_PARAMETER="SeeSchema";
    protected final static String SEE_TEMPLATE_PARAMETER="SeeTemplate";
    protected final static String SEE_INTERFACE_PARAMETER="SeeInterface";
    protected final static String GENERATE_INTERFACE_PARAMETER="GenerateItf";
    protected final static String REMOVE_DATA_PARAMETER="RemoveData";
    protected final static String REMOVE_METADATA_PARAMETER="RemoveMetadata";
    protected final static String REMOVE_ALL_PARAMETER="RemoveAll";
    protected final static String FLUSH_PARAMETER="Flush";
		
	protected final static String Object_Metadata="Metadata";
	protected final static String Object_Schema="Schema";
	protected final static String Object_Template="Template";
	protected final static String Object_Interface="Interface";
        
    protected final static String URL_PARAMETER="url";
    protected final static String DICTIONARY_PARAMETER="DICTIONARY";

    protected final static String ADD_PARAMETER="Add";
    protected final static String ADD_ENTRIES_PARAMETER="AddEntries";

    protected boolean loggedInUserRequired() {
        return true;
    }

    protected boolean userMayUseThisPO() {
        try {
            return this.getUser().isAdmin();
        } catch (PapillonBusinessException ex) {
            this.getSessionData().writeUserMessage("Error getting the authorisation to use this PO.");
        }
        return false;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }

    public Node getContent()
        throws javax.xml.parsers.ParserConfigurationException,
			HttpPresentationException,
		    IOException, org.xml.sax.SAXException,
			javax.xml.transform.TransformerException,
			java.sql.SQLException
    {
        
        // Création du contenu
        AdminVolumesXHTML content = (AdminVolumesXHTML)MultilingualXHtmlTemplateFactory.createTemplate("AdminVolumesXHTML", this.getComms(), this.getSessionData());
		  
        HttpPresentationRequest req = this.getComms().request;
				String dictName = myGetParameter(AdminVolumesXHTML.NAME_Dictionary);
				String volName = myGetParameter(AdminVolumesXHTML.NAME_Volume);
                String volNameTransformation = myGetParameter(AdminVolumesXHTML.NAME_VolumeTransformation);
        // If the page is called with parameters, take the requested action
        if (req.getParameterNames().hasMoreElements()) {
            //TEMPORAIRE :avec l URL
            //AJOUT DE DICO
            String userMessage = "";
            String urlString = myGetParameter(AdminVolumesXHTML.NAME_URL);
 			URL myURL = null;
            if (null != urlString &&
				null != myGetParameter(AdminVolumesXHTML.NAME_Dictionary)) {
				try  {
                    if (urlString.charAt(0) == '/') {
                        urlString = "file:" + urlString;
                    }
					myURL = new URL(urlString);
                }
				catch (java.io.IOException ex) {
					userMessage += "Problems while adding the specified volume. The following URL: "+ urlString +" is malformed;\n";
					userMessage += ex.getMessage();
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
					ex.printStackTrace();
				}	
				if (myURL != null) {
					userMessage += handleVolumeAddition(req, myURL);
				}
			}
             else if (null != myGetParameter(AdminVolumesXHTML.NAME_Volume) &&
					null != myGetParameter(AdminVolumesXHTML.NAME_URLObject) &&
					null != myGetParameter(AdminVolumesXHTML.NAME_Object)) {
					String object = myGetParameter(AdminVolumesXHTML.NAME_Object);
					String url = myGetParameter(AdminVolumesXHTML.NAME_URLObject);
				 try  {
                     if (url.charAt(0) == '/') {
                         url = "file:" + url;
                     }
					 myURL = new URL(url);
				 }
				 catch (java.io.IOException ex) {
					 userMessage += "Problems while adding the specified volume. The following URL: "+ urlString +" is malformed;\n";
                     this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
					 userMessage += ex.getMessage();
					 ex.printStackTrace();
				 }	
				 if (myURL != null) {
					 userMessage += this.uploadObject(volName, object, url);
				 }            
            } else if (null != myGetParameter(content.NAME_VolumeTransformation) &&
                       null != myGetParameter(content.NAME_URLObjectTransformation)) {
                String url = myGetParameter(content.NAME_URLObjectTransformation);
				try  {
					myURL = new URL(url);
				}
				catch (java.io.IOException ex) {
					userMessage += "Problems while adding the specified volume. The following URL: "+ urlString +" is malformed;\n";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
					userMessage += ex.getMessage();
					ex.printStackTrace();
				}	
				if (myURL != null) {
					userMessage += this.launchTranformation(volNameTransformation, url);
				}            
            //
            } else if (null != myGetParameter(REMOVE_METADATA_PARAMETER)) {
                String handle = myGetParameter(REMOVE_METADATA_PARAMETER);
                Volume volume = VolumesFactory.getVolumeByHandle(handle);
                volume.delete();
                Papillon.initializeAllCaches();
                userMessage = "Volume " + volume.getName() + " metadata erased...";	
            
				//
            } else if (null != myGetParameter(REMOVE_DATA_PARAMETER)) {
                String handle = myGetParameter(REMOVE_DATA_PARAMETER);
                Volume volume = VolumesFactory.getVolumeByHandle(handle);
                if (null != volume && !volume.isEmpty()) {
                    volume.empty();
                    Papillon.initializeAllCaches();
                    userMessage = "Volume " + volume.getName() + " emptied, entries deleted...";
                }
			//
            } else if (null != myGetParameter(REMOVE_ALL_PARAMETER)) {
                String handle = myGetParameter(REMOVE_ALL_PARAMETER);
                Volume volume = VolumesFactory.getVolumeByHandle(handle);
                if (null != volume && !volume.isEmpty()) {
                    volume.deleteAll();
                    Papillon.initializeAllCaches();
                    userMessage = "Volume " + volume.getName() + " entries and metadata erased...";
                }
            
            //
            } else if (null != myGetParameter(SEE_METADATA_PARAMETER)) {
                String handle = myGetParameter(SEE_METADATA_PARAMETER);
                Volume volume = VolumesFactory.getVolumeByHandle(handle);
                //adding an XML file
                addXml(content, volume.getXmlCode());
            
            //
            } else if (null != myGetParameter(SEE_SCHEMA_PARAMETER)) {
                String handle = myGetParameter(SEE_SCHEMA_PARAMETER);
                Volume volume = VolumesFactory.getVolumeByHandle(handle);
                //adding an XML file
                addXml(content, volume.getXmlSchema());
                
            //
            } else if (null != myGetParameter(SEE_TEMPLATE_PARAMETER)) {
                String handle = myGetParameter(SEE_TEMPLATE_PARAMETER);
                Volume volume = VolumesFactory.getVolumeByHandle(handle);
                //adding an XML file
                addXml(content, volume.getTemplateEntry());
                
            //    
            } else if (null != myGetParameter(SEE_INTERFACE_PARAMETER)) {
                String handle = myGetParameter(SEE_INTERFACE_PARAMETER);
                Volume volume = VolumesFactory.getVolumeByHandle(handle);
                //adding an XML file
                addXml(content, volume.getTemplateInterface());
                
            //
            } else if (null != myGetParameter(GENERATE_INTERFACE_PARAMETER)) {
                String handle = myGetParameter(GENERATE_INTERFACE_PARAMETER);
                Volume volume = VolumesFactory.getVolumeByHandle(handle);
                // generating an XNF interface description
				GenerateTemplate.generateInterfaceTemplate(volume);
            } else {
                String errorMessage = "Error: Wrong arguments";
                this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,errorMessage);
            }
            
            //
			if (userMessage != null) {
				this.getSessionData().writeUserMessage(userMessage);
				PapillonLogger.writeDebugMsg(userMessage);
			}
        }
        
        //adding the consult form
        addConsultForm(content, dictName, volName, volNameTransformation);
        
        //adding the content of the volumes table
        addVolumesArray(content);
        
        //On rend le contenu correct
        return content.getElementFormulaire();
    }
    
	protected String handleVolumeAddition(HttpPresentationRequest req, URL myURL) throws PapillonBusinessException, HttpPresentationException, java.net.MalformedURLException {
        String userMessage = "";
        PapillonLogger.writeDebugMsg("Adding volume metadata: " + myURL.toString());
		String logContribsString = req.getParameter(AdminVolumesXHTML.NAME_LogContributions);
		boolean logContribs = (logContribsString!=null && !logContribsString.equals(""));
		String parseEntriesString = req.getParameter(AdminVolumesXHTML.NAME_AddVolumesAndEntries);
		boolean parseEntries = (parseEntriesString!=null && !parseEntriesString.equals(""));
        
        // Create and Register the transaction
        CurrentDBTransaction.registerNewDBTransaction();
		Volume myVolume = null;
        try {
			Dictionary dict = DictionariesFactory.getDictionaryByName(req.getParameter(AdminVolumesXHTML.NAME_Dictionary));
			myVolume = VolumesFactory.parseVolumeMetadata(dict, myURL, parseEntries, logContribs);
			
            if (null != myVolume && !myVolume.isEmpty()) {
				userMessage = "adding " + myVolume.getName() + " volume: " + myVolume.getDictname() + " // "  + myVolume.getDbname() + " // " + myVolume.getSourceLanguage() + " // " + myVolume.getTargetLanguages() + " // " + myVolume.getVolumeRef();
				myVolume.save();
				fr.imag.clips.papillon.business.edition.UITemplates.resetCache();
            } else {
                userMessage = "Ignoring volume";
            }
            // everything was correct, commit the transaction...
			try {
				((DBTransaction) CurrentDBTransaction.get()).commit();
            } catch (java.sql.SQLException sqle) {
                PapillonLogger.writeDebugMsg("AdminVolumes: SQLException while commiting transaction.");
				sqle.printStackTrace();
            }
        } catch (PapillonBusinessException e) {
            userMessage = "Problems while adding the specified volume.\n";
            userMessage += e.getMessage();
            userMessage += "\nAll changes to the database have been rolled back.";
            this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
			e.printStackTrace();
            try {
                ((DBTransaction) CurrentDBTransaction.get()).rollback();
            } catch (java.sql.SQLException sqle) {
                PapillonLogger.writeDebugMsg("AdminVolumes: SQLException while rolling back failed transaction.");
				sqle.printStackTrace();
            }
        } finally {
            CurrentDBTransaction.releaseCurrentDBTransaction();
        }
        // Finally initialize all caches
        // FIXME: other connection can certainly happen during cache initialization... how can we avoid this ?
		VolumeEntriesFactory.resetCountCache(myVolume.getName());
		myVolume.getCount();
        Papillon.initializeAllCaches();
        
        return userMessage;
    }

	
    protected void addVolumesArray(AdminVolumesXHTML content) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            
        //where we must insert the form
        HTMLTableRowElement theRow = content.getElementTemplateRow();
        HTMLElement theDictname = content.getElementDictname();
        HTMLElement theName = content.getElementName();
        HTMLElement theDbname = content.getElementDbname();
        HTMLElement theSource = content.getElementSource();
        HTMLElement theEntries = content.getElementEntries();
        HTMLAnchorElement theSeeMetadataAnchor = content.getElementSeeMetadataAnchor();
        HTMLAnchorElement theSeeSchemaAnchor = content.getElementSeeSchemaAnchor();
		HTMLElement theSeeSchema = content.getElementSeeSchema();
		HTMLAnchorElement theSeeTemplateAnchor = content.getElementSeeTemplateAnchor();
        HTMLAnchorElement theSeeInterfaceAnchor = content.getElementSeeInterfaceAnchor();
        HTMLAnchorElement theRemoveMetadataAnchor = content.getElementRemoveMetadataAnchor();
        HTMLAnchorElement theRemoveAllAnchor = content.getElementRemoveAllAnchor();

        Node theRowParent = theRow.getParentNode();
				
				theRow.removeAttribute("id");
        theDictname.removeAttribute("id");
        theName.removeAttribute("id");
        theDbname.removeAttribute("id");
        theSource.removeAttribute("id");
        theEntries.removeAttribute("id");
        theSeeMetadataAnchor.removeAttribute("id");
				
        theSeeSchemaAnchor.removeAttribute("id");
        content.getElementSeeSchema().removeAttribute("id");
        
		theSeeTemplateAnchor.removeAttribute("id");
        content.getElementSeeTemplate().removeAttribute("id");
       
		 theSeeInterfaceAnchor.removeAttribute("id");
        content.getElementSeeInterface().removeAttribute("id");

        theRemoveMetadataAnchor.removeAttribute("id");
        theRemoveAllAnchor.removeAttribute("id");

        //adding the volumes description
        for (Iterator iter = VolumesFactory.getVolumesArray().iterator(); iter.hasNext();) {
            Volume volume = (Volume)iter.next();
            
        content.setTextDictname(volume.getDictname());
        content.setTextName(volume.getName());
        content.setTextDbname(volume.getDbname());
        content.setTextSource(volume.getSourceLanguage());
        content.setTextEntries("" + volume.getCount());
        
        // Get handle
        String handle =  volume.getHandle();

        //
        theSeeMetadataAnchor.setHref(this.getUrl() + "?" + SEE_METADATA_PARAMETER + "=" + handle);
				
				String schema = volume.getXmlSchema();
				content.setTextSeeSchema("");
				if (schema !=null && !schema.equals("")) {
					content.setTextSeeSchema("See");
					theSeeSchemaAnchor.setHref(this.getUrl() + "?" + SEE_SCHEMA_PARAMETER + "=" + handle);
				}

				String object = volume.getTemplateEntry();
				content.setTextSeeTemplate("");
				if (object !=null && !object.equals("")) {
					content.setTextSeeTemplate("See");
					theSeeTemplateAnchor.setHref(this.getUrl() + "?" + SEE_TEMPLATE_PARAMETER + "=" + handle);
				}
				
				object = volume.getTemplateInterface();
				content.setTextSeeInterface("");
				if (object !=null && !object.equals("")) {
					content.setTextSeeInterface("See");
					theSeeInterfaceAnchor.setHref(this.getUrl() + "?" + SEE_INTERFACE_PARAMETER + "=" + handle);
				}
				else if (schema !=null && !schema.equals("")) {
					content.setTextSeeInterface("Generate");
					theSeeInterfaceAnchor.setHref(this.getUrl() + "?" + GENERATE_INTERFACE_PARAMETER + "=" + handle);
				}
								
        theRemoveMetadataAnchor.setHref(this.getUrl() + "?" + REMOVE_METADATA_PARAMETER + "=" + handle);
        theRemoveAllAnchor.setHref(this.getUrl() + "?" + REMOVE_ALL_PARAMETER + "=" + handle);
            
        theRowParent.appendChild(theRow.cloneNode(true));
				
				
        }
        
        theRowParent.removeChild(theRow);
    }
    
    protected void addXml(AdminVolumesXHTML content, String xmlString) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            
        Node xmlNode = XslSheetFactory.applyXslSheetForXml(xmlString);    
            
        //where we must insert the xml
        HTMLElement theXml = content.getElementXml();

        Node theXmlParent = theXml.getParentNode();

        theXmlParent.appendChild(content.importNode(xmlNode,true));
        theXmlParent.removeChild(theXml);
    }
    
     protected void addConsultForm(AdminVolumesXHTML content, String selectedDict, String selectedVolume, String selectedVolumeTransformation) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException, 
                HttpPresentationException {
           // Adding the appropriate source languages to the source list
        HTMLOptionElement dictionaryOptionTemplate = content.getElementDictionaryOptionTemplate();
        Node dictionarySelect = dictionaryOptionTemplate.getParentNode();
        dictionaryOptionTemplate.removeAttribute("id");
        // We assume that the option element has only one text child 
        // (it should be this way if the HTML is valid...)
        Text dictionaryTextTemplate = (Text)dictionaryOptionTemplate.getFirstChild(); 
                
        //
        for (Iterator iter = DictionariesFactory.getDictionariesArray().iterator(); iter.hasNext();) {   
	    String dictName =((Dictionary)iter.next()).getName();
            dictionaryOptionTemplate.setValue(dictName);
            dictionaryOptionTemplate.setLabel(dictName);
            // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux 
            // specs W3C.
            dictionaryTextTemplate.setData(dictName);
			dictionaryOptionTemplate.setSelected(dictName.equals(selectedDict));

            dictionarySelect.appendChild(dictionaryOptionTemplate.cloneNode(true));            
        }
        dictionarySelect.removeChild(dictionaryOptionTemplate);
				
        // Adding the appropriate source languages to the source lists
        HTMLOptionElement volumeOptionTemplate = content.getElementVolumeOptionTemplate();
        HTMLOptionElement volumeOptionTemplateTransformation = content.getElementVolumeOptionTemplateTransformation();
        Node volumeSelect = volumeOptionTemplate.getParentNode();
        Node volumeSelectTransformation = volumeOptionTemplateTransformation.getParentNode();
        volumeOptionTemplate.removeAttribute("id");
        volumeOptionTemplateTransformation.removeAttribute("id");
        // We assume that the option element has only one text child 
        // (it should be this way if the HTML is valid...)
        Text volumeTextTemplate = (Text)volumeOptionTemplate.getFirstChild();
        Text volumeTextTemplateTransformation = (Text)volumeOptionTemplateTransformation.getFirstChild();
                
       //     
        for (Iterator iter = VolumesFactory.getVolumesArray().iterator(); iter.hasNext();) {
            String volumeName = ((Volume)iter.next()).getName();
            
            //
            volumeOptionTemplate.setValue(volumeName);
            volumeOptionTemplate.setLabel(volumeName);
            // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux 
            // specs W3C.
            volumeTextTemplate.setData(volumeName);
            volumeOptionTemplate.setSelected(volumeName.equals(selectedVolume));
            volumeSelect.appendChild(volumeOptionTemplate.cloneNode(true));
            
            //
            volumeOptionTemplateTransformation.setValue(volumeName);
            volumeOptionTemplateTransformation.setLabel(volumeName);
            // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux 
            // specs W3C.
            volumeTextTemplateTransformation.setData(volumeName);
            if ( (selectedVolumeTransformation != null) && (!selectedVolumeTransformation.equals("")) ) {
                volumeOptionTemplateTransformation.setSelected(volumeName.equals(selectedVolumeTransformation));
            }
            volumeSelectTransformation.appendChild(volumeOptionTemplateTransformation.cloneNode(true));
        }
        volumeSelect.removeChild(volumeOptionTemplate);
        volumeSelectTransformation.removeChild(volumeOptionTemplateTransformation);
    }

	protected String uploadObject(String volName, String object, String url) throws PapillonBusinessException {
        // FIXME: shouldn't this be protected in a transaction ?
        PapillonLogger.writeDebugMsg("uploadObject volName: " + volName);
		Volume myVolume = VolumesFactory.getVolumeByName(volName);
		String result = "Nothing uploaded";
		if (myVolume!=null && !myVolume.isEmpty()) {
			if (object !=null && !object.equals("")) {
				String objectResult = XMLServices.xmlCode(XMLServices.buildDOMTreeFromUrl(url));
				result= object + " reloaded";
				if (object.equals(Object_Schema)) {
					myVolume.setXmlSchema(objectResult);
				}
				else if (object.equals(Object_Metadata)) {
					myVolume.setXmlCode(objectResult);
					myVolume.setCdmElements();
					myVolume.setLinksTable();
				}
				else if (object.equals(Object_Template)) {
					objectResult = VolumesFactory.updateTemplateEntry(objectResult, myVolume.getCdmElements());
					PapillonLogger.writeDebugMsg("uploadObject objectResult: " + objectResult);
					myVolume.setTemplateEntry(objectResult);
					myVolume.setCdmElements();
					myVolume.setLinksTable();
				}
				else if (object.equals(Object_Interface)) {
					myVolume.setTemplateInterface(objectResult);
                }
				else {
						result = "Nothing uploaded";
				}
				fr.imag.clips.papillon.business.edition.UITemplates.resetCache();
				myVolume.save();
                Papillon.initializeAllCaches();
            }
		}
		return result;
	}
    
    //
    protected String launchTranformation(String volName, String url) throws PapillonBusinessException {
		String result = "Nothing uploaded";
        String objectResult = XMLServices.xmlCode(XMLServices.buildDOMTreeFromUrl(url));
        
        //
        if ( (volName != null) && volName.equals("All") ) {
            
            //
            for (Iterator iter = VolumesFactory.getVolumesArray().iterator(); iter.hasNext();) { 
                Volume volume = (Volume)iter.next();
                PapillonLogger.writeDebugMsg("Transformation " + volume.getName() + " volume");

                //
                VolumesFactory.launchTransformation(objectResult, volume, this.getUser());
            }
            
            //
            result = "Transform all volumes";
            PapillonLogger.writeDebugMsg(result + " succeed !");
            
        } else {
            
            //
            Volume myVolume = VolumesFactory.getVolumeByName(volName);
            
            //
            if (myVolume!=null && !myVolume.isEmpty()) {
                
                //
                PapillonLogger.writeDebugMsg("Transformation " + myVolume.getName() + " volume");
                
                //
                VolumesFactory.launchTransformation(objectResult, myVolume, this.getUser());
                result = "Transform " + myVolume.getName() + " volume";
                
            } else {
                
                //
                result = "Empty volume";
            }        
        }
        
        //
		return result;
	}
    
}
