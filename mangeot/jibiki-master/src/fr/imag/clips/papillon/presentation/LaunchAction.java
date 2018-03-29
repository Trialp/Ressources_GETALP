/*
 *  Jibiki
 *
 *  Enhydra super-servlet presentation object
 *
 * © Francis Brunet-Manquat et Mathieu Mangeot - GETA CLIPS IMAG
 * Projet Papillon
 *
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
*/

package fr.imag.clips.papillon.presentation;

// General java imports
import java.io.UnsupportedEncodingException;
import java.io.IOException;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

// Local imports
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;

// DOM imports
import org.w3c.dom.Node;

public class LaunchAction extends PapillonBasePO {

    //
    private final boolean DEBUG = true;
    
    // PARAMETERS
    protected final static String ACTION_PARAMETER = "action";
    protected final static String HANDLE_PARAMETER = "handle";
    protected final static String DICTIONARY_NAME_PARAMETER = "dictionary";
    protected final static String VOLUME_NAME_PARAMETER = "volumename";
    
    // ACTIONS
    private final static String UNDO_ACTION = "Undo";
    private final static String EDIT_ACTION = "Edit";
    private final static String RE_EDIT_ACTION = "Re-Edit";
    private final static String LOOKUP_ACTION = "Lookup";
    
    // URLs
    private final static String EditingErrorURL = "EditingError.po";
    private final static String ConfirmEntryURL = "ConfirmEntry.po";
    private final static String EditEntryURL = "EditEntry.po";

    protected boolean loggedInUserRequired() {
        return true;
    }
    
    protected boolean userMayUseThisPO() {
        return true;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }
    
    public Node getContent() throws HttpPresentationException, UnsupportedEncodingException, IOException, PapillonBusinessException {
        
        //
        if (DEBUG) PapillonLogger.writeDebugMsg ("LaunchAction : getContent"); 
        
        // Get action
        String action = myGetParameter(ACTION_PARAMETER);		
        
        //
        if ( action == null && action.equals("") ) {
            // Error page
            throw new ClientPageRedirectException(EditingErrorURL);
            
        // EDIT and RE_EDIT
        } else if ( action.equals(EDIT_ACTION) || action.equals(RE_EDIT_ACTION) ) {
            
            // Get parameters
            String volumeName = myGetParameter(VOLUME_NAME_PARAMETER);
            String entryHandle = myGetParameter(HANDLE_PARAMETER);
            
            //
            if ( volumeName!=null && !volumeName.equals("") &&
                 entryHandle!=null && !entryHandle.equals("") ) {
                
                //
                throw new ClientPageRedirectException(EditEntryURL + "?" + 
                                                      EditEntry.VolumeName_PARAMETER + "=" + volumeName + 
                                                      "&" + EditEntry.EntryHandle_PARAMETER + "=" + entryHandle);
            } else {
                // Error page
                throw new ClientPageRedirectException(EditingErrorURL);
            }
            
        // UNDO
        } else if ( action.equals(UNDO_ACTION) ) {
        
            // Get parameters
            String volumeName = myGetParameter(VOLUME_NAME_PARAMETER);
            String entryHandle = myGetParameter(HANDLE_PARAMETER);

            //
            if ( volumeName!=null && !volumeName.equals("") &&
                 entryHandle!=null && !entryHandle.equals("") ) {
                
                // Search contribution corresponding to entryId
                VolumeEntry volumeEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
                    
                //
                PapillonLogger.writeDebugMsg("LaunchAction UNDO_ACTION call undoFinish");
                VolumeEntry previousVolumeEntry = volumeEntry.undoFinish();
                
                //
                if (previousVolumeEntry != null) {
                    //
                    throw new ClientPageRedirectException(EditEntryURL + "?" + 
                                                          EditEntry.VolumeName_PARAMETER + "=" + previousVolumeEntry.getVolumeName() + 
                                                          "&" + EditEntry.EntryHandle_PARAMETER + "=" + previousVolumeEntry.getHandle());
                } else {
                    // FIXME : redirect page ... msg : undo not possible !
                    // Error page
                    throw new ClientPageRedirectException(EditingErrorURL);
                }
                
            } else {
                // Error page
                throw new ClientPageRedirectException(EditingErrorURL);
            }
            
        } else {
            // Error page
            throw new ClientPageRedirectException(EditingErrorURL);
        }
        
        
        /*
        // CREATE NEW ENTRY
        if (action.equals("createAnyway")) {
            if (headword!=null && !headword.equals("")) {
                
                //
                EditEntryInitFactory.createEntry(volume.getName(), headword, this.getUser());
                
            } else {
                
                // Error message
                this.getSessionData().writeUserMessage("Headwords are mandatory for creation");
            }
            
            // EDIT
        } else if (action.equals("edit")) {
            
            //
            EditEntryInitFactory.editEntry(myEntry, this.getUser());
            
            // DUPLICATE
        } else if (action.equals("duplicate")) {
            
            //
            EditEntryInitFactory.duplicateEntry(myEntry, this.getUser());                    
            
            // DELETE
        } else if (action.equals("delete")) {
            
            //
            EditEntryInitFactory.deleteEntry(myEntry, this.getUser()); 
            
            
            // UNDELETE
        } else if (action.equals("undelete")) {
            
            //
            EditEntryInitFactory.undeleteEntry(myEntry, this.getUser());
        }    
        */
    }
}
