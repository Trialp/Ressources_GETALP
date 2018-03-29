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
 * Revision 1.38  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.37  2007/01/12 13:08:51  fbrunet
 * Bug correction : undo error in HandleEntryModifications class
 *
 * Revision 1.36  2007/01/09 17:46:03  fbrunet
 * *** empty log message ***
 *
 * Revision 1.35  2007/01/08 15:13:42  fbrunet
 * Correction of th xml attribut bug in ContributionHeader (VolumeEntry class)
 *
 * Revision 1.34  2007/01/05 12:57:49  fbrunet
 * Add undo draft method (bug in EditEntry.java : undo after last finish contribution)
 * Modify transformation method
 *
 * Revision 1.33  2006/12/13 09:32:00  fbrunet
 * *** empty log message ***
 *
 * Revision 1.32  2006/11/09 09:04:42  fbrunet
 * *** empty log message ***
 *
 * Revision 1.31  2006/05/23 07:57:51  fbrunet
 * Modify edit management : When an user edit a lexie, this lexie doesn't change until an upgrade/finish action (then a new contibution is created link to lexie with a not-finished status).
 *
 * Revision 1.30  2006/04/18 14:30:24  fbrunet
 * Authorize admin to edit all entries
 *
 * Revision 1.29  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 * Revision 1.28  2006/03/13 08:48:00  fbrunet
 * bug corrections before merge
 *
 * Revision 1.26  2006/03/01 15:41:13  mangeot
 * bug fixes
 *
 * Revision 1.25  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 *
 *-----------------------------------------------
 * Papillon Login page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.Element;

// internal imports
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.dictionary.ParseVolume;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.edition.UIGenerator;
import fr.imag.clips.papillon.business.edition.UITemplates;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.Group;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.CurrentRequestContext;
import fr.imag.clips.papillon.facelets.util.JibikiContext;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import java.util.Collection;

public class EditEntry extends EditingBasePO {
    
    //
    private final boolean DEBUG = false;
    
    // Public parameters
    public static String VolumeName_PARAMETER = EditEntryXHTML.NAME_VolumeName;  
    public static String EntryHandle_PARAMETER = EditEntryXHTML.NAME_EntryHandle;
	public static String Referrer_PARAMETER = EditEntryXHTML.NAME_Referrer; 
	
    // URL
    protected final static String EditEntryURL = "EditEntry.po";
	protected final static String EditEntryInitURL = "LexalpEditEntryInit.po";
    protected final static String EditingErrorURL = "EditingError.po";
		
    protected boolean loggedInUserRequired() {
        return true;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }

    public org.w3c.dom.Node getContent() 
		throws java.io.UnsupportedEncodingException, 
			HttpPresentationException {
      /*  this.getComms().response.setHeader("Cache-Control", "private, no-store, no-cache, must-revalidate, max-age=0, post-check=0, pre-check=0"); */
       
		//
        if (DEBUG) PapillonLogger.writeDebugMsg ("EditEntry : getContent");        
                
        // Management of the parameters
	    String volumeName = myGetParameter(EditEntryXHTML.NAME_VolumeName);
		String entryHandle = myGetParameter(EditEntryXHTML.NAME_EntryHandle);
	    String referrer = myGetParameter(EditEntryXHTML.NAME_Referrer);
        
		// Recuperation of parameters
		if (referrer== null || referrer.equals("")) {
			referrer = this.getReferrer();
		}
        
        //
		EditEntryXHTML content = (EditEntryXHTML) MultilingualXHtmlTemplateFactory.createTemplate("EditEntryXHTML", this.myComms, this.sessionData);
        
		// Manage VolumeEntry
		VolumeEntry myVolumeEntry = null;
		if (volumeName!=null && !volumeName.equals("")
			&& entryHandle!=null && !entryHandle.equals("")) {
			
			// VolumeEntry
			myVolumeEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
			if (DEBUG) PapillonLogger.writeDebugMsg("EditEntry : myVolumeEntry.ContributionId " + myVolumeEntry.getContributionId());
            
            /*
			// Verification 
			if (myVolumeEntry!=null && !myVolumeEntry.isEmpty()
				&& !(myVolumeEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS) 
					 && (myVolumeEntry.getModificationAuthor().equals(this.getUser().getLogin())
                        || this.getUser().isInGroup(Group.ADMIN_GROUP)) )) {
				throw new ClientPageRedirectException(EditingErrorURL);
			}
            */ 
             
			// Verification of the DOM
			if (myVolumeEntry.getDom() == null) {
                PapillonLogger.writeDebugMsg ("EditEntry : The dom entry is null.");
				throw new ClientPageRedirectException(EditingErrorURL);
			}
		}
		// TODO answer if no arguments
		else {
            PapillonLogger.writeDebugMsg ("EditEntry : No parameters (volumeName & entryHandle)");
			throw new ClientPageRedirectException(EditEntryInitURL);
		}
        
        // Get interface template
		Element myInterface = UITemplates.getInterface(volumeName, UITemplates.DEFAULT_FORM, this.getSessionData().getUserAcceptLanguages()); 
		Element myItfTemplate = null;	
		if (myInterface !=  null) {
			myInterface = (Element) myInterface.cloneNode(true);
			myItfTemplate = (Element) myInterface.cloneNode(true);	
		}
        
        // Add volume name & entry handle in the form	
		//UIGenerator.setValueInput(myInterface, VolumeName_PARAMETER, volumeName);
		//UIGenerator.setValueInput(myInterface, EntryHandle_PARAMETER, entryHandle);
		//UIGenerator.setValueInput(myInterface, Referrer_PARAMETER, referrer);
        XHTMLInputElement volumeNameElement = content.getElementVolumeName();
		volumeNameElement.setValue(volumeName);
        XHTMLInputElement entryHandleElement = content.getElementEntryHandle();
		entryHandleElement.setValue(entryHandle);
        XHTMLInputElement referrerElement = content.getElementReferrer();
		referrerElement.setValue(referrer);
        
        // Enable undo update button            
  //      String previousNFContributionId = myVolumeEntry.getClassifiedNotFinishedContributionId();
  //      Collection classifiedFinishedContributionIdCollection = myVolumeEntry.getClassifiedFinishedContributionIdCollection();
  //      if (myVolumeEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)
  //          && (    ((previousNFContributionId != null)
  //                  && (!previousNFContributionId.equals("")))
  //              ||  (classifiedFinishedContributionIdCollection.size() != 0))){
  //          XHTMLInputElement undoUpdateElement = content.getElementUndoUpdate();
  //          undoUpdateElement.setDisabled(false);
  //      }
        
        // Fill interface template
		UIGenerator.fillInterfaceTemplate(myVolumeEntry.getDom().getDocumentElement(), myInterface, myItfTemplate);
        
        // Append interface
        XHTMLElement editingInterfaceElement = content.getElementEditEntryContent();
		editingInterfaceElement.appendChild(content.importNode(myInterface, true));
        
        //
		return content.getElementEditEntryForm();
        }   
}
