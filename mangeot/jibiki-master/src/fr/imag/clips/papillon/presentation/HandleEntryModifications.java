/*
 * Jibiki 
 *
 * Enhydra super-servlet
 * 
 * © Francis Brunet-Manquat, Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.19  2007/02/28 09:27:07  fbrunet
 * Added ajax method to AdvancedQueryForm page
 *
 * Revision 1.18  2007/02/09 08:49:10  fbrunet
 * *** empty log message ***
 *
 * Revision 1.17  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.16  2007/01/12 13:08:51  fbrunet
 * Bug correction : undo error in HandleEntryModifications class
 *
 * Revision 1.15  2007/01/09 17:31:04  fbrunet
 * Bug correction : error in user verification in HandleEntryModifications class (induce wrong redirection)
 *
 * Revision 1.14  2007/01/08 15:13:42  fbrunet
 * Correction of th xml attribut bug in ContributionHeader (VolumeEntry class)
 *
 * Revision 1.13  2007/01/05 12:57:49  fbrunet
 * Add undo draft method (bug in EditEntry.java : undo after last finish contribution)
 * Modify transformation method
 *
 * Revision 1.12  2006/12/14 20:03:26  fbrunet
 * Add method to normalize value into XML structure.
 *
 * Revision 1.11  2006/12/13 09:32:00  fbrunet
 * *** empty log message ***
 *
 * Revision 1.10  2006/11/09 09:04:42  fbrunet
 * *** empty log message ***
 *
 * Revision 1.9  2006/05/23 07:57:51  fbrunet
 * Modify edit management : When an user edit a lexie, this lexie doesn't change until an upgrade/finish action (then a new contibution is created link to lexie with a not-finished status).
 *
 * Revision 1.8  2006/05/22 22:45:54  fbrunet
 * LexALP: add merge method in post-save processing (merge axies with same referenced lexies)
 *
 * Revision 1.7  2006/05/05 02:08:23  fbrunet
 * bug correction : url utf8 transfert (in createEntryInit)
 *
 * Revision 1.6  2006/04/24 13:43:29  fbrunet
 * Add new class ViewQueryResult : allow to use one class to create result display in advancedSearch and EditEntryInit (like advancedQueryForm)
 * Improve result display : view n results per page
 *
 * Revision 1.5  2006/04/18 14:30:24  fbrunet
 * Authorize admin to edit all entries
 *
 * Revision 1.4  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 * Revision 1.3  2006/03/13 08:48:00  fbrunet
 * bug corrections before merge
 *
 * Revision 1.2  2006/03/01 15:47:08  mangeot
 * syntax bug fixes
 *
 * Revision 1.1  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 *-----------------------------------------------
 * Jibiki plate-form: handles entry modifications and redirect to the approriate page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.*;

//
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

// Internal imports
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.dictionary.ContributionsFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.edition.UIGenerator;
import fr.imag.clips.papillon.business.edition.UITemplates;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.ResultPreProcessor;
import fr.imag.clips.papillon.business.transformation.ResultPreProcessorFactory;
import fr.imag.clips.papillon.business.transformation.ResultPostSaveProcessor;
import fr.imag.clips.papillon.business.transformation.ResultPostSaveProcessorFactory;
import fr.imag.clips.papillon.business.transformation.ResultPostUpdateProcessor;
import fr.imag.clips.papillon.business.transformation.ResultPostUpdateProcessorFactory;
import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.transformation.ResultFormatterFactory;
import fr.imag.clips.papillon.presentation.ConfirmEntry;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.Group;

// Dom imports
import org.w3c.dom.Document;
import org.w3c.dom.Element;


public class HandleEntryModifications extends EditingBasePO {
    
    //
    private final boolean DEBUG = false;
    
    // Public parameters
    public static String EntryHandle_PARAMETER = "EntryHandle";
    public static String VolumeName_PARAMETER = "VolumeName";  
	public static String Referrer_PARAMETER = "Referrer";
    public static String AddCall_PARAMETER = "AddCall";
    public static String DelCall_PARAMETER = "DelCall";
    public static String MoveUpCall_PARAMETER = "MoveUpCall";
    public static String MoveDownCall_PARAMETER = "MoveDownCall";
    public static String ChooseCall_PARAMETER = "ChooseCall";
    public static String Cancel_PARAMETER = "Cancel";
    public static String Update_PARAMETER = "Update";
    public static String Preview_PARAMETER = "Preview";
    public static String SaveAsDraft_PARAMETER = "SaveAsDraft";
    public static String UndoUpdate_PARAMETER = "UndoUpdate";
    //public static String SaveComment_PARAMETER = "SaveComment";  
    
    // Protected parameters
    protected static String Choose_PARAMETER = UIGenerator.CHOOSE_ATTR_NAME;  
    protected static String Select_PARAMETER = UIGenerator.SELECT_ATTR_NAME;  
    protected static String Boolean_PARAMETER = UIGenerator.BOOLEAN_ATTR_NAME;  
    protected static String BooleanTrue_PARAMETER = UIGenerator.BOOLEAN_TRUE_ATTR_NAME;  
	
    // URL
	protected final static String EditEntryInitURL = "AdvancedLookup.po";
    protected final static String EditingErrorURL = "EditingError.po";
    protected final static String ConfirmEntryURL = "ConfirmEntry.po";
	protected final static String EditEntryURL = "EditEntry.po";
	
    //
	/* 	protected final static String newBlockRedirectionJavascript = "function loadFunction () {\n" */
	/* 		+ "   window.location.hash='NewBlock';\n" */
	/* 		+ "}\n"; */
	
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
			
            this.getComms().response.setHeader("Cache-Control", "private, no-store, no-cache, must-revalidate, max-age=0, post-check=0, pre-check=0");
           //
            if (DEBUG) PapillonLogger.writeDebugMsg ("HandleEntryModifications : getContent");   
            
			// Management of the parameters
			String submitAdd = myGetParameter(AddCall_PARAMETER);
			String submitDelete = myGetParameter(DelCall_PARAMETER);
			String submitMoveUp = myGetParameter(MoveUpCall_PARAMETER);
			String submitMoveDown = myGetParameter(MoveDownCall_PARAMETER);
			String submitChoose = myGetParameter(ChooseCall_PARAMETER);
            String submitCancel = myGetParameter(Cancel_PARAMETER);
			String submitUpdate = myGetParameter(Update_PARAMETER);
            String submitPreview = myGetParameter(Preview_PARAMETER);
            String submitSaveAsDraft = myGetParameter(SaveAsDraft_PARAMETER);
            String submitUndoUpdate = myGetParameter(UndoUpdate_PARAMETER);
			String select = myGetParameter(Select_PARAMETER);
			String choose = myGetParameter(Choose_PARAMETER);
			String volumeName = myGetParameter(VolumeName_PARAMETER);
			String entryHandle = myGetParameter(EntryHandle_PARAMETER);
			//String saveComment = myGetParameter(SaveComment_PARAMETER);
			String referrer = myGetParameter(Referrer_PARAMETER);

            
            //
			if (volumeName==null || volumeName.equals("") ||
				entryHandle==null || entryHandle.equals("")) {
				if (DEBUG) PapillonLogger.writeDebugMsg ("HandleEntryModifications : volumeName || entryHandle null");   
				                
                // FIXME: Add a user error message !!!
				throw new ClientPageRedirectException(EditEntryInitURL);
			}
		
			// INIT
			VolumeEntry theVolumeEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
			
            // 
			if ( theVolumeEntry == null || theVolumeEntry.isEmpty() ) {

                // FIXME: add correct verification process !
                //&&     !( oldVolumeEntry.getModificationAuthor().equals(this.getUser().getLogin())
                //      || this.getUser().isInGroup(Group.ADMIN_GROUP))) {
                
                // FIXME: Add a user error message !!!
				if (DEBUG) PapillonLogger.writeDebugMsg ("HandleEntryModifications : theVolumeEntry null");
				throw new ClientPageRedirectException(EditEntryInitURL);
			
            }
            else if ( theVolumeEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS) ) {
                ;
            }
            else if ( theVolumeEntry.getStatus().equals(VolumeEntry.DRAFT_STATUS) ) {
                theVolumeEntry.setStatus(VolumeEntry.NOT_FINISHED_STATUS);
                ResultPreProcessor preProcessor = ResultPreProcessorFactory.getPreProcessor(theVolumeEntry);
                preProcessor.transformation(theVolumeEntry, this.getUser());
            }
            else if ( theVolumeEntry.getStatus().equals(VolumeEntry.FINISHED_STATUS) ) {
                
                // CREATE EDITABLE COPY
                VolumeEntry newVolumeEntry = VolumeEntriesFactory.newEntryFromExisting(theVolumeEntry);
                newVolumeEntry.setHeadword(theVolumeEntry.getCdmHeadword());
                newVolumeEntry.addClassifiedFinishedContribution(theVolumeEntry);
                newVolumeEntry.setStatus(VolumeEntry.NOT_FINISHED_STATUS);
                newVolumeEntry.setPreviousContributionId(theVolumeEntry.getContributionId());
                ResultPreProcessor preProcessor = ResultPreProcessorFactory.getPreProcessor(newVolumeEntry);
                preProcessor.transformation(newVolumeEntry, this.getUser());
               theVolumeEntry = newVolumeEntry;
            
            } else {
                //
                throw new ClientPageRedirectException(EditEntryInitURL);
            }
            
            
            if (submitCancel!=null && !submitCancel.equals("")) {
                throw new ClientPageRedirectException(
                                                      ConfirmEntryURL + "?" +
                                                      ConfirmEntry.VolumeName_PARAMETER + "=" + theVolumeEntry.getVolumeName() + "&" +
                                                      ConfirmEntry.EntryHandle_PARAMETER + "=" + theVolumeEntry.getHandle() + "&" +
                                                      ConfirmEntry.Delete_PARAMETER + "=" + "Supprimer"
                                                      );
            }
			
			// Call PreProcessor
			// FIXME: Here ? TO BE ADDED TO EDITENTRYINIT
			//ResultPreProcessor preProcessor = ResultPreProcessorFactory.getPreProcessor(newVolumeEntry);
			//preProcessor.transformation(newVolumeEntry, this.getUser());
			
			
            // Get document element
			Element myEntry = theVolumeEntry.getDom().getDocumentElement();
            
            // Get template entry
			Element myTemplateEntry = UITemplates.getTemplateEntry(volumeName);
			
			String anchor = "";
			
			// Fill DOM structure
			if (theVolumeEntry!=null) {
				FillDOMStructure(myEntry, this.getComms().request.getParameterNames());
			}
			
			// Add element
			if (submitAdd!=null && !submitAdd.equals("")) {
				int plus =  submitAdd.indexOf(UIGenerator.PARAMETERS_SEPARATOR);
				if (plus > 0) {
					String elementName = submitAdd.substring(0,plus);
					String parentElement = submitAdd.substring(plus+1);
					String[] siblingElements = myGetParameterValues(Select_PARAMETER);
					if (DEBUG) PapillonLogger.writeDebugMsg ("HandleEntryModifications : + " + elementName);   
					anchor = UIGenerator.addElement(elementName, parentElement, myEntry, myTemplateEntry, siblingElements);
				}
			}
			// Delete elements (MUST be after updateElement because it modifies the element ids.)
			else if (submitDelete!=null && !submitDelete.equals("")
					 && select != null && !select.equals("")) {
				int plus =  submitDelete.indexOf(UIGenerator.PARAMETERS_SEPARATOR);
				if (plus > 0) {
					String elementName = submitDelete.substring(0,plus);
					String parentElement = submitDelete.substring(plus+1);
					String[] selectedElements = myGetParameterValues(Select_PARAMETER);
					anchor = UIGenerator.deleteElements(elementName, parentElement, selectedElements, myEntry, myTemplateEntry);
				}
			}
			// Move elements up
			else if (submitMoveUp!=null && !submitMoveUp.equals("")
					 && select != null && !select.equals("")) {
				int plus =  submitMoveUp.indexOf(UIGenerator.PARAMETERS_SEPARATOR);
				if (plus > 0) {
					String elementName = submitMoveUp.substring(0,plus);
					String parentElement = submitMoveUp.substring(plus+1);
					String[] selectedElements = myGetParameterValues(Select_PARAMETER);
					anchor = UIGenerator.moveElementsUp(elementName, parentElement, selectedElements, myEntry);
				}
			}
			// Move elements Down
			else if (submitMoveDown!=null && !submitMoveDown.equals("")
					 && select != null && !select.equals("")) {
				int plus =  submitMoveDown.indexOf(UIGenerator.PARAMETERS_SEPARATOR);
				if (plus > 0) {
					String elementName = submitMoveDown.substring(0,plus);
					String parentElement = submitMoveDown.substring(plus+1);
					String[] selectedElements = myGetParameterValues(Select_PARAMETER);
					anchor = UIGenerator.moveElementsDown(elementName, parentElement, selectedElements, myEntry);
				}
			}
			// Choose elements 
			else if (submitChoose!=null && !submitChoose.equals("")
					 && choose != null && !choose.equals("")) {
				int plus =  submitChoose.indexOf(UIGenerator.PARAMETERS_SEPARATOR);
				if (plus > 0) {
					String elementId = submitChoose.substring(0,plus);
					String parentId = submitChoose.substring(plus+1);
					anchor = UIGenerator.chooseElement(choose,parentId,myEntry, myTemplateEntry);
				}
			}
			
			//
			updateEditedEntry(theVolumeEntry, referrer);

            if (submitPreview!=null && !submitPreview.equals("")) {
                // redirect preview to confirmEntry
                // Referer is not used !
               // PapillonLogger.writeDebugMsg("HandleEntryModifications: Submit Preview -> confirmEntry");
                throw new ClientPageRedirectException(
                                                      ConfirmEntryURL + "?" +
                                                      ConfirmEntry.VolumeName_PARAMETER + "=" + theVolumeEntry.getVolumeName() + "&" +
                                                      ConfirmEntry.EntryHandle_PARAMETER + "=" + theVolumeEntry.getHandle());
            }
            if (submitSaveAsDraft!=null && !submitSaveAsDraft.equals("")) {
                // Save draft and redirect
                saveEntryAsDraft(theVolumeEntry, referrer);
            }
			// Edit current volume entry
            PapillonLogger.writeDebugMsg("HandleEntryModifs: Referrer: " + referrer);
            throw new ClientPageRedirectException(
                                                  EditEntryURL + "?" + 
                                                  EditEntry.VolumeName_PARAMETER + "=" + theVolumeEntry.getVolumeName() + "&" +
                                                  EditEntry.EntryHandle_PARAMETER + "=" + theVolumeEntry.getHandle() + "&" +
                                                  EditEntry.Referrer_PARAMETER + "=" + myUrlEncode(referrer) + "#" + anchor
												  );
	}
	
	//
	protected void updateBooleanElements(String[] booleanElements, String[] trueElements, Element myEntry) throws fr.imag.clips.papillon.business.PapillonBusinessException {
		// FIXME: a problem here: the element ids change when one element is deleted.
		// the second element of the same type will not be deleted
		java.util.List myList = java.util.Arrays.asList(trueElements);
		for (int i=0; i<booleanElements.length;i++) {
			String elt = booleanElements[i];
			if (elt != null && !elt.equals("")) {
				if (myList.contains(elt)) {
					UIGenerator.updateElement(elt, "true", myEntry);
				}
				else {
					UIGenerator.updateElement(elt, "false", myEntry);
				}
			}
		}
	}
    
    //
    private void FillDOMStructure(Element entryElt, java.util.Enumeration parameterNames)
		throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException, fr.imag.clips.papillon.business.PapillonBusinessException {
            
            // Redirect parameters
            while (parameterNames.hasMoreElements()) {
                String parameterName = (String) parameterNames.nextElement();
                if (parameterName.indexOf(UIGenerator.ID_SEPARATOR)>0) {
                    UIGenerator.updateElement(parameterName, myGetParameter(parameterName), entryElt);
                }
            }
            
            //
            updateBooleanElements(myGetParameterValues(Boolean_PARAMETER),myGetParameterValues(BooleanTrue_PARAMETER), entryElt);
        }
	
	//
	private void updateEditedEntry(VolumeEntry newVolumeEntry, String referrer)
		throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException {
            if (DEBUG) PapillonLogger.writeDebugMsg("HandleEntryModifications : updateEditedEntry");
			
            // Update draft
            newVolumeEntry.setModification(this.getUser().getLogin(), "update");
            
            // Call PostUpdateProcessor
            ResultPostUpdateProcessor postUpdateProcessor = ResultPostUpdateProcessorFactory.getPostUpdateProcessor(newVolumeEntry);
            postUpdateProcessor.transformation(newVolumeEntry, this.getUser());
            newVolumeEntry.save();
        }
	
    private void saveEntryAsDraft(VolumeEntry newVolumeEntry, String referrer)
    throws java.io.UnsupportedEncodingException,
    com.lutris.appserver.server.httpPresentation.HttpPresentationException {
        if (DEBUG) PapillonLogger.writeDebugMsg("HandleEntryModifications : updateDraftEntry");
        
        // Update draft
        newVolumeEntry.setModification(this.getUser().getLogin(), "save as draft");
        newVolumeEntry.setStatus(VolumeEntry.DRAFT_STATUS);
        
        newVolumeEntry.save();
    }
    
	
}
