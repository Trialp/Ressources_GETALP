/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Francis Brunet-Manquat, Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.1  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 *-----------------------------------------------
 */

package fr.imag.clips.papillon.presentation;

//
import java.util.Iterator;
import java.util.Collection;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.*;

// internal imports
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.transformation.ResultFormatterFactory;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.presentation.xhtml.orig.*;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;


public class MergeEntries extends EditingBasePO {

    
    //
    private final boolean DEBUG = true;
    
    //
    public final static String MergeEntriesURL = "MergeEntries.po";
    
    //
    public static String EntryHandle_PARAMETER = "handle";
    public static String VolumeName_PARAMETER = "volume"; 
    public static String Message_PARAMETER = "message";
	
    protected boolean loggedInUserRequired() {
        return true;
    }
	
    protected boolean userMayUseThisPO() {
        return true;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }
	
    public org.w3c.dom.Node getContent() throws java.io.UnsupportedEncodingException, HttpPresentationException {
        
        //
        if (DEBUG) PapillonLogger.writeDebugMsg ("MergeEntries : getContent");   
        
        //
        MergeEntriesXHTML content = (MergeEntriesXHTML) MultilingualXHtmlTemplateFactory.createTemplate("MergeEntriesXHTML", this.myComms, this.sessionData);
        
        // Management of the parameters
        String entryHandle = myGetParameter(EntryHandle_PARAMETER);
        String volumeName = myGetParameter(VolumeName_PARAMETER);
        String message = myGetParameter(Message_PARAMETER);
        
        //
        if (volumeName!=null && !volumeName.equals("")
            && entryHandle!=null && !entryHandle.equals("")) {
            
            // VolumeEntry
            VolumeEntry volumeEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
           
            // Verification 
            if ( !(volumeEntry.getStatus().equals(VolumeEntry.FINISHED_STATUS) 
                   && volumeEntry.getModificationAuthor().equals(this.getUser().getLogin())) ) {
                
                // Error page
                throw new ClientPageRedirectException(EditingError.EditingErrorURL);
            }
            
            // Fill the LaunchAction parameters
            XHTMLElement value1 = content.getElementValue1();
            value1.setAttribute("name", LaunchAction.VOLUME_NAME_PARAMETER);
            value1.setAttribute("value", volumeEntry.getVolumeName());
            value1.removeAttribute("id");
            XHTMLElement value2 = content.getElementValue2();
            value2.setAttribute("name", LaunchAction.HANDLE_PARAMETER);
            value2.setAttribute("value", volumeEntry.getHandle());
            value2.removeAttribute("id");
            
            // Display volumeEntry
            QueryResult qr = new QueryResult(1, volumeEntry);
            ResultFormatter rf = ResultFormatterFactory.getFormatter(qr, null, ResultFormatterFactory.XHTML_DIALECT, null);
            Element resultElement = (Element)rf.getFormattedResult(qr, this.getUser());
            XHTMLDivElement mergingResultViewElement = content.getElementMergingResultView();
            mergingResultViewElement.appendChild(content.importNode(resultElement, true));
            
            // Display previous entries
            Collection collection = volumeEntry.getClassifiedFinishedContributionIdCollection();
            XHTMLDivElement entriesViewElement = content.getElementEntriesView();
            Node entriesViewParent = entriesViewElement.getParentNode();
            for (Iterator iter = collection.iterator(); iter.hasNext();) {
                volumeEntry = VolumeEntriesFactory.findEntryByContributionId(volumeName, (String)iter.next());
                
                // Create result element
                qr = new QueryResult(1, volumeEntry);
                rf = ResultFormatterFactory.getFormatter(qr, null, ResultFormatterFactory.XHTML_DIALECT, null);
                resultElement = (Element)rf.getFormattedResult(qr, this.getUser());
                
                // Import result element to content
                Node entriesViewElementTmp = content.importNode((XHTMLDivElement)entriesViewElement.cloneNode(true), true);
                entriesViewParent.appendChild(entriesViewElementTmp);
                entriesViewElementTmp.appendChild(content.importNode(resultElement, true));                
            }
            entriesViewParent.removeChild(entriesViewElement);
        }
                
        //
        return content.getElementMergeEntriesContent();
    }            
}
