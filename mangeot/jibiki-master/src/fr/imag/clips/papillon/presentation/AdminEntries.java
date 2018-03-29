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
 * Revision 1.13  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.12  2006/04/05 12:38:21  mangeot
 * Fixed a confusion when importing entries versus contributions
 *
 * Revision 1.11  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.10  2006/02/22 19:05:56  mangeot
 * MM: Added default status choice when importing entries
 *
 * Revision 1.9  2006/02/09 10:49:28  mangeot
 * Added 3 new options when importing entries:
 * - ParseVolume.ReplaceExistingEntry_CopyAnyway
 * - ParseVolume.ReplaceExistingEntry_CopyIfSameStatus
 * - ParseVolume.ReplaceExistingEntry_CopyIfFinished
 * Creates a new entry id before importing the entry in the database
 * And added a log for the ParseVolume.MAX_DISCARDED_ENTRIES_LOGGED = 500
 * first entries that are discarded (not imported)
 *
 * Revision 1.8  2006/01/25 14:05:21  mangeot
 * MM: I modified the import of entries
 * Now, I check if an existing entry with the same id already exists in the database.
 * If yes, the user has to choose with several strategies:
 * - ParseVolume.ReplaceExistingEntry_Ignore
 * 	Do not import the new entry if there is already one in the
 * 	DB with the same entry
 * - ParseVolume.ReplaceExistingEntry_ReplaceAnyway
 * 	Replace the existing entry by the new one
 * - ParseVolume.ReplaceExistingEntry_ReplaceIfSameStatus
 * 	Replace the existing entry by the new one if their status is the same
 * - ParseVolume.ReplaceExistingEntry_ReplaceIfFinished
 * 	Replace the existing entry by the new one if the status of the
 * 	existing entry is 'not_finished' and the one of the new entry is
 * 	'finished'
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
 * Revision 1.5.4.6  2005/06/09 11:07:45  mangeot
 * Deleted the countEntriesCache. entries counts are not cached any more.
 * Fixed a few bugs.
 *
 * Revision 1.5.4.5  2005/06/02 09:13:50  mangeot
 * *** empty log message ***
 *
 * Revision 1.5.4.4  2005/06/01 15:25:43  mangeot
 * *** empty log message ***
 *
 * Revision 1.5.4.3  2005/06/01 15:20:33  mangeot
 * Added a boolean for contributionslog
 *
 * Revision 1.5.4.2  2005/06/01 14:17:54  mangeot
 * Bug fix in itf generator & added variant in CDM lookup1
 *
 * Revision 1.5.4.1  2005/06/01 08:38:43  mangeot
 * Multi bug correction + added the possibility of disabling data edition
 * via the Admin.po page
 *
 * Revision 1.5  2005/04/15 15:03:47  mangeot
 * Fixed a bug in setIdIfNull and deleted the empty button on AdminEntries
 *
 * Revision 1.4  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.3  2005/04/11 08:01:02  fbrunet
 * Passage en xhtml des ressources Papillon.
 *
 * Revision 1.2.2.1  2005/01/27 15:56:21  mangeot
 * Able to load a volume with XPointers, cannot lookup the result yet.
 * Does not compile but commit for backup
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * Papillon Admin Entries page.
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

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.util.Iterator;
import java.text.DateFormat;
import java.io.*;


import fr.imag.clips.papillon.CurrentDBTransaction;
import com.lutris.appserver.server.sql.DBTransaction;

//pour le dictionary
import fr.imag.clips.papillon.business.dictionary.*;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.PapillonLogger;

import fr.imag.clips.papillon.business.PapillonBusinessException;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;


public class AdminEntries extends PapillonBasePO {
    
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
        throws HttpPresentationException,
			java.io.UnsupportedEncodingException {
        
        // Création du contenu
        AdminEntriesTmplXHTML content = (AdminEntriesTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("AdminEntriesTmplXHTML", this.getComms(), this.getSessionData());
        
        String volumeString = myGetParameter(content.NAME_VOLUME);
        String urlString = myGetParameter(content.NAME_URL);
        String submitAdd = myGetParameter(content.NAME_ADD);
        String logContributions = myGetParameter(content.NAME_LogContributions);
        String defaultStatus = myGetParameter(content.NAME_DefaultStatus);
        String indexEntriesString = myGetParameter(content.NAME_IndexEntries);
				
		
        String replaceExistingContributionsString = myGetParameter(content.NAME_ReplaceExistingContributions);
		int replaceExistingContributions = ParseVolume.ReplaceExistingContribution_Ignore;
		if (null != replaceExistingContributionsString && !replaceExistingContributionsString.equals("")) {
			replaceExistingContributions = Integer.parseInt(replaceExistingContributionsString);
		}
		
        String replaceExistingEntriesString = myGetParameter(content.NAME_ReplaceExistingEntries);
		int replaceExistingEntries = ParseVolume.ReplaceExistingEntry_Ignore;
		if (null != replaceExistingEntriesString && !replaceExistingEntriesString.equals("")) {
			replaceExistingEntries = Integer.parseInt(replaceExistingEntriesString);
		}
		
		
        boolean logContribs = (logContributions!=null && !logContributions.equals(""));
        boolean indexEntries = (indexEntriesString!=null && !indexEntriesString.equals(""));
        
		String userMessage = "";
        HttpPresentationRequest req = this.getComms().request;
       if (req.getParameterNames().hasMoreElements()) {
            if (volumeString!=null && !volumeString.equals("") &&
                urlString!=null && !urlString.equals("") &&
                submitAdd!=null && !submitAdd.equals("")) {
                java.net.URL myURL = null;
                if (urlString.charAt(0) == '/') {
                    urlString = "file:" + urlString;
                }
                try {
                    myURL = new java.net.URL(urlString);
                    PapillonLogger.writeDebugMsg(myURL.toString());
                }
                catch (java.io.IOException ex) {
                    userMessage += "Problems while adding the specified volume entries. The following URL: "+ urlString +" is malformed;\n";
                    userMessage += ex.getMessage();
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
                   ex.printStackTrace();
                }	
                if (myURL != null) {
                    userMessage += handleVolumeAddition(volumeString, myURL, defaultStatus, replaceExistingEntries, replaceExistingContributions, logContribs, indexEntries);
                }
                if (userMessage != "") {
                    this.getSessionData().writeUserMessage(userMessage);
                    PapillonLogger.writeDebugMsg(userMessage);
                }
            }
            else {
                String errorMessage = "Error: Wrong arguments";
                this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,errorMessage);
            }
        }
        addConsultForm(content, volumeString);
        //On rend le contenu correct
        return content.getElementFormulaire();
    }
	
	protected String handleVolumeAddition(String volumeString, java.net.URL myURL, String defaultStatus, int replaceExistingEntries, int replaceExistingContributions, boolean logContribs, boolean indexEntries)
		throws fr.imag.clips.papillon.business.PapillonBusinessException, 
			HttpPresentationException {
        String userMessage ="";
        
        // Create and Register the transaction
  //      CurrentDBTransaction.registerNewDBTransaction();
        try {
			String message = ParseVolume.parseVolume(volumeString, myURL, defaultStatus, replaceExistingEntries, replaceExistingContributions, logContribs, indexEntries);
			userMessage = "Volume: " + volumeString + " / URL: " + myURL + " downloaded...";
			userMessage += message;
           // everything was correct, commit the transaction...
//            ((DBTransaction) CurrentDBTransaction.get()).commit();
        } catch (PapillonBusinessException e) {
            userMessage = "Problems while adding the specified volume entries.\n";
            userMessage += e.getMessage();
            userMessage += "\nAll changes to the database have been rolled back.";
            this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
			e.printStackTrace();
/*            try {
                ((DBTransaction) CurrentDBTransaction.get()).rollback();
            } catch (java.sql.SQLException sqle) {
                PapillonLogger.writeDebugMsg("AdminVolumes: SQLException while rolling back failed transaction.");
				sqle.printStackTrace();
            }
        } finally {
            CurrentDBTransaction.releaseCurrentDBTransaction(); */
        } 
        return userMessage;
    }


	
	
	protected void addConsultForm(AdminEntriesTmplXHTML content, String selectedVolume) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException, 
                HttpPresentationException {
           // Adding the appropriate source languages to the source list
        HTMLOptionElement volumeOptionTemplate = content.getElementVolumeOptionTemplate();
        Node volumeSelect = volumeOptionTemplate.getParentNode();
        volumeOptionTemplate.removeAttribute("id");
        // We assume that the option element has only one text child 
        // (it should be this way if the HTML is valid...)
        Text volumeTextTemplate = (Text)volumeOptionTemplate.getFirstChild(); 
                
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
        }
        volumeSelect.removeChild(volumeOptionTemplate);
    }
    
}
