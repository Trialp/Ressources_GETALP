/*
 *  Jibiki plateform
 *
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.1.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 *
 */
package fr.imag.clips.papillon.facelets.xhtml;

import java.util.Collection;
import java.util.ArrayList;

import org.w3c.dom.Node;
import org.w3c.dom.Text;
import org.enhydra.xml.xhtml.dom.*;

import fr.imag.clips.papillon.presentation.xhtml.orig.ActionNodesXHTML;
import fr.imag.clips.papillon.presentation.MultilingualXHtmlTemplateFactory;
import fr.imag.clips.papillon.CurrentRequestContext;
import fr.imag.clips.papillon.facelets.util.JibikiContext;

import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;

import fr.imag.clips.papillon.business.PapillonBusinessException;

import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.QueryParameter;
import fr.imag.clips.papillon.business.dictionary.QueryCriteria;

import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.PapillonLogger;

import fr.imag.clips.papillon.presentation.EditEntryInitFactory;
import fr.imag.clips.papillon.presentation.AdvancedQueryForm;
import fr.imag.clips.papillon.presentation.PapillonSessionData;

import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.Group;


// import org.apache.xalan.xslt.*;

public class Actions implements fr.imag.clips.papillon.facelets.api.Actions {
    protected final static String HistoryURL = "History.po";
    protected final static String Action = "action";
    
    public Actions() {
        
    }
    
    public Node getActions(String entryid) throws PapillonBusinessException {
        JibikiContext context = CurrentRequestContext.get();
        
        // BUG279: This will be long enough... We really need to work on caching and multiple volume handling (maybe an explicit id...)
  //      VolumeEntry ve = VolumeEntriesFactory.findEntryByEntryId(((PapillonSessionData) context.get("sessionData")).getUser(),entryid);
        VolumeEntry ve = VolumeEntriesFactory.findEntryByContributionId(((PapillonSessionData) context.get("sessionData")).getUser(),entryid);
		return this.getActions(ve);
    }
    
    public Node getActions(VolumeEntry myEntry) throws PapillonBusinessException {
        
        // I need a context object that will contain the comms and sessionData...
        JibikiContext context = CurrentRequestContext.get();
        PapillonSessionData sdata = (PapillonSessionData) context.get("sessionData");
        HttpPresentationComms scomms = (HttpPresentationComms) context.get("comms");

        User user = (null == sdata) ? null : sdata.getUser();
		String url = (String) context.get("absoluteurl");
       
        ActionNodesXHTML content;
        try {
            content = (ActionNodesXHTML) 
            MultilingualXHtmlTemplateFactory.createTemplate("ActionNodesXHTML", scomms, sdata);
        } catch (HttpPresentationException e) {
            throw new PapillonBusinessException("Failed to instanciate XHTML Action Node Template.", e);
        }
        
        XHTMLElement actionsNode = content.getElementActionsNode();
        actionsNode.removeAttribute("id");
        // Actions
		// (action.equals("AUTHOR"))
					XHTMLSpanElement entryAuthor = content.getElementEntryAuthor();
					entryAuthor.removeAttribute("id");
					Text textAuthor = content.createTextNode("unknown");
					textAuthor.setNodeValue(myEntry.getModificationAuthor());
					entryAuthor.appendChild(textAuthor);
				
		//action.equals("AXIE"))
					XHTMLAnchorElement viewAxieAnchor = content.getElementViewAxieAnchor();
					viewAxieAnchor.removeAttribute("id");
					//axie(myEntry,url,user, viewAxieAnchor);
				
		// (action.equals("EDIT"))
					XHTMLAnchorElement editAnchor = content.getElementEditEntryAnchor();
					XHTMLAnchorElement duplicateAnchor = content.getElementDuplicateEntryAnchor();
					XHTMLAnchorElement deleteAnchor = content.getElementDeleteEntryAnchor();
					XHTMLAnchorElement undeleteAnchor = content.getElementUndeleteEntryAnchor();
					editAnchor.removeAttribute("id");
					duplicateAnchor.removeAttribute("id");
					deleteAnchor.removeAttribute("id");
					undeleteAnchor.removeAttribute("id");
					edit(myEntry,url,user,editAnchor,duplicateAnchor,deleteAnchor,undeleteAnchor);
		
		// (action.equals("HISTORY"))
		XHTMLAnchorElement viewHistoryEntryAnchor = content.getElementViewHistoryEntryAnchor();
		viewHistoryEntryAnchor.removeAttribute("id");
		// FIXME: Create static variable ... VolumeName and EntryHandle ...
        String AbsoluteHistoryUrl = url.substring(0,url.lastIndexOf('/')+1) + HistoryURL;
		viewHistoryEntryAnchor.setHref(AbsoluteHistoryUrl + "?" + "VolumeName" + "=" + myEntry.getVolumeName() + "&" + "EntryHandle" + "=" + myEntry.getHandle());
		viewHistoryEntryAnchor.setAttribute("class", "action");
		
		// (action.equals("STATUS"))
					XHTMLSpanElement entryStatus = content.getElementEntryStatus();
					entryStatus.removeAttribute("id");
					Text textStatus = content.createTextNode("unknown");
					if (myEntry.getStatus().equals(VolumeEntry.FINISHED_STATUS)) {
						textStatus.setNodeValue(VolumeEntry.FINISHED_STATUS);
						
						//            if (myEntry.getModificationAuthor().equals(user.getLogin())) {
						//                entryNode.setAttribute("class", "myFinishedEntry");
						//            } else {
						//                entryNode.setAttribute("class", "finishedEntry");
						//            }
						
					} else if (myEntry.getStatus().equals(VolumeEntry.MODIFIED_STATUS)) {
						textStatus.setNodeValue("under edition");
						//            entryNode.setAttribute("class", "modifiedEntry");
						
					} else if (myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)) {
						textStatus.setNodeValue(VolumeEntry.DELETED_STATUS);
						//            entryNode.setAttribute("class", "modifiedEntry");
						
					} else if (myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {
						
						if (myEntry.getModificationAuthor().equals(user.getLogin())) {
							textStatus.setNodeValue("proceed edition");
							//                entryNode.setAttribute("class", "myNotFinishedEntry");
						} else {
							textStatus.setNodeValue("under edition");
							//                entryNode.setAttribute("class", "notFinishedEntry");
						}
                    } else if (myEntry.getStatus().equals(VolumeEntry.CLASSIFIED_FINISHED_STATUS)) {
                        textStatus.setNodeValue("classified");
                        
                    } else if (myEntry.getStatus().equals(VolumeEntry.DRAFT_STATUS)) {
                        textStatus.setNodeValue("draft");
        
                    }
					entryStatus.appendChild(textStatus);

		// (action.equals("XML"))
					XHTMLAnchorElement viewXmlAnchor = content.getElementViewXmlAnchor();
					viewXmlAnchor.removeAttribute("id");
					if (user != null && user.isInGroup(Group.ADMIN_GROUP)) {
						
						// FIXME : create new page.po like history
						QueryParameter qpxml = new QueryParameter();
						qpxml.setXsl("XML");
						ArrayList dicts = new ArrayList();
						dicts.add(myEntry.getDictionary());
						qpxml.setDictionaries(dicts);
						ArrayList crit = new ArrayList();
						String[] idc = new String[4];
						idc[0] = Volume.CDM_contributionId;
						idc[1] = null;
						idc[2] = myEntry.getContributionId();
						idc[3] = QueryCriteria.EQUAL;
						crit.add(idc);
						qpxml.setCriteria(crit);
						
						viewXmlAnchor.setHref(url + "?" + EditEntryInitFactory.ACTION_PARAMETER + "=lookup&" + AdvancedQueryForm.getEncodedUrlForParameter(qpxml));
						viewXmlAnchor.setAttribute("class", "action");
						
					} else {
						viewXmlAnchor.setAttribute("class", "hidden");
					}
        return actionsNode;
    }
	
	
        
	private void edit(VolumeEntry myEntry,String url, User user, XHTMLAnchorElement editAnchor, XHTMLAnchorElement duplicateAnchor,
	XHTMLAnchorElement deleteAnchor, XHTMLAnchorElement undeleteAnchor) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
		//
        String href = url + "?"
		+ EditEntryInitFactory.VOLUME_PARAMETER + "="
		+ myEntry.getVolumeName() + "&"
		+ EditEntryInitFactory.HANDLE_PARAMETER + "="
		+ myEntry.getHandle() + "&"
		+ EditEntryInitFactory.ACTION_PARAMETER + "=";
        
        // Actions
        if (((!myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS))
             && (!myEntry.getStatus().equals(VolumeEntry.MODIFIED_STATUS)))
            
            || (myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)
                && (myEntry.getModificationAuthor().equals(user.getLogin())
                    || user.isInGroup(Group.ADMIN_GROUP)))
            ) 
        {
            
			// Edit button
			if (!myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)) {
				editAnchor.setHref(href + "edit");
				editAnchor.setAttribute("target", "_blank");
				editAnchor.setAttribute("class", "action");
			} else {
				editAnchor.setHref("");
				editAnchor.setAttribute("class", "hidden");
			}					
            // Duplicate button
            if (!myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)
                && !myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {
                duplicateAnchor.setHref(href + "duplicate");
                duplicateAnchor.setAttribute("target", "_blank");
                duplicateAnchor.setAttribute("class", "action");
            } else {
                duplicateAnchor.setHref("");
                duplicateAnchor.setAttribute("class", "hidden");
            }
            
            // Delete button
            if (!myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)) {
                //&& !myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {
                
                // action delete and refresh query with last parameters
                deleteAnchor.setHref(href + "delete");
                deleteAnchor.setAttribute("class", "action");
			} else {
				deleteAnchor.setHref("");
				deleteAnchor.setAttribute("class", "hidden");
			}
            
            // Undelete button
            if (myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)) {
                
                // action undelete and refresh query with last parameters
                undeleteAnchor.setHref(href + "undelete");
                undeleteAnchor.setAttribute("class", "action");
                undeleteAnchor.setAttribute("target", "_blank");
            } else {
                undeleteAnchor.setHref("");
                undeleteAnchor.setAttribute("class", "hidden");
            }
            
        } else {
            // Edit button
            editAnchor.setHref("");
            editAnchor.setAttribute("class", "hidden");
            
            // Duplicate button
            duplicateAnchor.setHref("");
            duplicateAnchor.setAttribute("class", "hidden");
            
            // Delete button
            deleteAnchor.setHref("");
            deleteAnchor.setAttribute("class", "hidden");
            
            // Undelete button
            undeleteAnchor.setHref("");
            undeleteAnchor.setAttribute("class", "hidden");
        }		
	} 
	
	public Node getLinkActions(String entryid) throws PapillonBusinessException {
        JibikiContext context = CurrentRequestContext.get();
        
        // BUG279: This will be long enough... We really need to work on caching and multiple volume handling (maybe an explicit id...)
		VolumeEntry ve = VolumeEntriesFactory.findEntryByContributionId(((PapillonSessionData) context.get("sessionData")).getUser(),entryid);
		return this.getLinkActions(ve);
    }
    
    public Node getLinkActions(VolumeEntry myEntry) throws PapillonBusinessException {
        
        // I need a context object that will contain the comms and sessionData...
        JibikiContext context = CurrentRequestContext.get();
        PapillonSessionData sdata = (PapillonSessionData) context.get("sessionData");
        HttpPresentationComms scomms = (HttpPresentationComms) context.get("comms");
		
        User user = (null == sdata) ? null : sdata.getUser();
		String url = (String) context.get("url");
		
        ActionNodesXHTML content;
        try {
            content = (ActionNodesXHTML) 
            MultilingualXHtmlTemplateFactory.createTemplate("ActionNodesXHTML", scomms, sdata);
        } catch (HttpPresentationException e) {
            throw new PapillonBusinessException("Fail to instanciate XHTML Action Node Template.", e);
        }
        
        XHTMLElement actionsNode = content.getElementActionsNode();
        actionsNode.removeAttribute("id");
        // Actions
		// (action.equals("AUTHOR"))
		XHTMLSpanElement entryAuthor = content.getElementEntryAuthor();
		actionsNode.removeChild(entryAuthor);
		
		//action.equals("AXIE"))
		XHTMLAnchorElement viewAxieAnchor = content.getElementViewAxieAnchor();
		viewAxieAnchor.removeAttribute("id");
		viewAxieAnchor.setHref(
							   url + "?" + EditEntryInitFactory.ACTION_PARAMETER + "=lookup&VOLUME=" + myEntry.getVolume() + "&HANDLE=" + myEntry.getHandle());
		
		// (action.equals("EDIT"))
		XHTMLAnchorElement editAnchor = content.getElementEditEntryAnchor();
		XHTMLAnchorElement duplicateAnchor = content.getElementDuplicateEntryAnchor();
		XHTMLAnchorElement deleteAnchor = content.getElementDeleteEntryAnchor();
		XHTMLAnchorElement undeleteAnchor = content.getElementUndeleteEntryAnchor();
		actionsNode.removeChild(editAnchor);
		actionsNode.removeChild(duplicateAnchor);
		actionsNode.removeChild(deleteAnchor);
		actionsNode.removeChild(undeleteAnchor);
		
		// (action.equals("HISTORY"))
		XHTMLAnchorElement viewHistoryEntryAnchor = content.getElementViewHistoryEntryAnchor();
		actionsNode.removeChild(viewHistoryEntryAnchor);

		// (action.equals("STATUS"))
		XHTMLSpanElement entryStatus = content.getElementEntryStatus();
		actionsNode.removeChild(entryStatus);
		
		// (action.equals("XML"))
		XHTMLAnchorElement viewXmlAnchor = content.getElementViewXmlAnchor();
		actionsNode.removeChild(viewXmlAnchor);
       return actionsNode;
    }
	
}
