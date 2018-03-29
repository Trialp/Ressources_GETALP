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
 * $Log$
 * Revision 1.15  2007/04/10 14:51:14  serasset
 * LexalpEditentryInit now searches for lexies AND translation, with NO TARGET LANGUAGE selected.
 * This way, the axie is retrieved in the result and may be linked from the lexie.
 *
 * Revision 1.14  2007/03/25 22:00:57  fbrunet
 * improved avancedqueryform javascript
 * bug correction: in ViewQueryResult class, encode url criteria in UTF-8
 *
 * Revision 1.13  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.12  2007/01/08 15:13:42  fbrunet
 * Correction of th xml attribut bug in ContributionHeader (VolumeEntry class)
 *
 * Revision 1.11  2006/09/11 19:57:48  fbrunet
 * - bug correction : interface edition (link axie to another axi)
 *
 * Revision 1.10  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.9  2006/06/19 15:27:01  fbrunet
 * Jibiki : improvement of the search result display
 * Lexalp : add help menu (link to wiki and bug tracker)
 *
 * Revision 1.8  2006/06/01 22:05:05  fbrunet
 * New interface, quick search, new contribution management (the first edition not create new contribution. New contribution is created after add, remove element, update, save, etc. in the interface window)
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
 * Revision 1.2  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.1.4.6  2006/02/17 10:41:49  fbrunet
 * Change QueryCriteria parameters
 * Add new windows when editing an entry
 *
 * Revision 1.1.4.5  2006/01/25 16:51:46  fbrunet
 * *** empty log message ***
 *
 * Revision 1.1.4.4  2006/01/25 15:22:23  fbrunet
 * Improvement of QueryRequest
 * Add new search criteria
 * Add modified status
 *
 * Revision 1.1.4.3  2006/01/24 13:39:49  fbrunet
 * Modification view management
 * Modification LexALP postprocessing
 *
 * Revision 1.1.4.2  2005/12/02 10:04:09  fbrunet
 * Add Pre/Post edition processing
 * Add index reconstruction
 * Add new query request
 * Add fuzzy search
 * Add new contribution administration
 * Add xsl transformation volume
 *
 * Revision 1.1.4.1  2005/08/31 15:01:39  serasset
 * Applied modifications done on the LEXALP_1_0 branch to updated sources of the
 * trunk to create a new updated LEXALP_1_1 branch.
 *
 * Revision 1.1.2.1  2005/07/22 13:28:32  serasset
 * Modified EditEntryInit for Lexalp. It now serves as a main page for db maintenance.
 * Added a function to get url for QueryParameter.
 * Modified the way xslsheets are handled in order to allow several xslsheet with the same name, different dicts.
 *
 *
 */

package fr.imag.clips.papillon.presentation;


//General java imports
import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;
import java.util.ArrayList;

// DOM imports
import org.w3c.dom.html.*;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;
import org.enhydra.xml.xhtml.dom.*;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

import com.lutris.dods.builder.generator.query.QueryBuilder;

// HTML source import
import fr.imag.clips.papillon.presentation.xhtmllexalp.orig.*;

//local imports
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.dictionary.QueryRequest;
import fr.imag.clips.papillon.business.dictionary.QueryCriteria;
import fr.imag.clips.papillon.business.dictionary.QueryParameter;
import fr.imag.clips.papillon.business.dictionary.IQuery;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.Group;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.transformation.ResultFormatterFactory;



public class LexalpEditEntryInit extends PapillonBasePO {
    
    protected final static String HANDLE_PARAMETER = "handle";
    protected final static String ACTION_PARAMETER = "action";
	
    protected final static String ContributionsURL = "AdminContributions.po";
    protected final static String ContributionsVolumeParameter = "VOLUME";
    
    protected final static String ConsultExpertURL = "AdvancedSearch.po";
    protected final static String ConsultExpertVolumeParameter = "VOLUME";
    protected final static String ConsultExpertHandleParameter = "handle";
    protected final static String ConsultExpertFormatterParameter = "xslid";
    
    protected final static String EditingErrorURL = "EditingError.po";
    
    protected LexalpEditEntryInitXHTML content;
    
    protected boolean loggedInUserRequired() {
        return true;
    }
    
    protected boolean userMayUseThisPO() {
        return true;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }
    
    public Node getContent()
        throws HttpPresentationException,
        java.io.UnsupportedEncodingException,
        java.io.IOException,
        fr.imag.clips.papillon.business.PapillonBusinessException {
                      
         //   this.getComms().response.setHeader("Cache-Control", "private, no-store, no-cache, must-revalidate, max-age=0, post-check=0, pre-check=0");

            // Content creation
            content = (LexalpEditEntryInitXHTML)MultilingualXHtmlTemplateFactory.createTemplate("fr.imag.clips.papillon.presentation.xhtmllexalp", "LexalpEditEntryInitXHTML", this.getComms(), this.getSessionData());
            // On regarde d'abord les parametres qui nous sont demandes.
            String volumeName = myGetParameter(EditEntryInitFactory.VOLUME_PARAMETER);
            String headword = myGetParameter(content.NAME_Headword);
            String entryHandle = myGetParameter(EditEntryInitFactory.HANDLE_PARAMETER);
            String action = myGetParameter(EditEntryInitFactory.ACTION_PARAMETER);		
            
            //
            if (volumeName != null && !volumeName.equals("")) {
                this.setPreference(content.NAME_VOLUME, volumeName);
            } else {
                volumeName = this.getPreference(content.NAME_VOLUME);
            }
            
            
            //
            if (action!=null && !action.equals("") && !action.equals("lookup") &&
                volumeName!=null && !volumeName.equals("") &&
                entryHandle!=null && !entryHandle.equals("")) {
                
                // Search last contribution corresponding to entry handle
                VolumeEntry myEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
                
                // CREATE NEW ENTRY
                if (action.equals("createAnyway")) {
                    if (headword!=null && !headword.equals("")) {
                        
                        //
                        EditEntryInitFactory.createEntry(volumeName, headword, this.getUser());
                        
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
            }
            
            //
            updateCreationInterface(volumeName, headword);
            User myUser = getUser();
            if (null != myUser) {
                content.setTextUserName(myUser.getName());
            }
            
            // Do query !
            AdvancedQueryForm qf = new AdvancedQueryForm(this.getComms(), this.getSessionData(), false, true);
            
            /*
            //
            QueryRequest queryReq = qf.getQueryRequest();
            
            //
            QueryCriteria criteriaStatus = new QueryCriteria();
            criteriaStatus.add("key", "=", Volume.CDM_contributionStatus);  
            criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.CLASSIFIED_FINISHED_STATUS);
            criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.CLASSIFIED_NOT_FINISHED_STATUS);
            queryReq.addCriteria(criteriaStatus);
            */
            
            //
            QueryRequest queryReq = new QueryRequest(VolumesFactory.getVolumesArray());
            queryReq.setOrTree();   // OR(AND)
            ArrayList criteriaSearchList = qf.getCriteriaList();
                
            
            if ( (criteriaSearchList != null) && (criteriaSearchList.size() != 0) ) {
                //
                ArrayList listFinished = new ArrayList();
                QueryCriteria criteriaFinishedStatus = new QueryCriteria();
                criteriaFinishedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);  
                criteriaFinishedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.FINISHED_STATUS);
                listFinished.add(criteriaFinishedStatus);
                listFinished.addAll(criteriaSearchList);
                queryReq.addAndCriteriaList(listFinished);
                
                // Group management
                if (!this.getUser().isInGroup()) {
                    //
                    ArrayList listModify = new ArrayList();
                    QueryCriteria criteriaModifiedStatus = new QueryCriteria();
                    criteriaModifiedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);  
                    criteriaModifiedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.MODIFIED_STATUS);
                    listModify.add(criteriaModifiedStatus);
                    QueryCriteria criteriaNotLogin = new QueryCriteria();
                    criteriaNotLogin.add("key", QueryCriteria.EQUAL, Volume.CDM_nextContributionAuthor);
                    criteriaNotLogin.add("value", QueryCriteria.NOT_EQUAL, this.getUser().getLogin());
                    listModify.add(criteriaNotLogin);
                    listModify.addAll(criteriaSearchList);
                    queryReq.addAndCriteriaList(listModify);
                    
                    //
                    ArrayList listNotFinished = new ArrayList();
                    QueryCriteria criteriaNFStatus = new QueryCriteria();
                    criteriaNFStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);  
                    criteriaNFStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.NOT_FINISHED_STATUS);
                    listNotFinished.add(criteriaNFStatus);
                    QueryCriteria criteriaLogin = new QueryCriteria();
                    criteriaLogin.add("key", QueryCriteria.EQUAL, Volume.CDM_modificationAuthor);
                    criteriaLogin.add("value", QueryCriteria.EQUAL, this.getUser().getLogin());
                    listNotFinished.add(criteriaLogin);
                    listNotFinished.addAll(criteriaSearchList);
                    queryReq.addAndCriteriaList(listNotFinished);
                } else {
                    //
                    ArrayList listDeleted = new ArrayList();
                    QueryCriteria criteriaDeletedStatus = new QueryCriteria();
                    criteriaDeletedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);  
                    criteriaDeletedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.DELETED_STATUS);
                    listDeleted.add(criteriaDeletedStatus);
                    listDeleted.addAll(criteriaSearchList);
                    queryReq.addAndCriteriaList(listDeleted);
                    
                    //
                    ArrayList listNotFinished = new ArrayList();
                    QueryCriteria criteriaNFStatus = new QueryCriteria();
                    criteriaNFStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);  
                    criteriaNFStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.NOT_FINISHED_STATUS);
                    listNotFinished.add(criteriaNFStatus);
                    listNotFinished.addAll(criteriaSearchList);
                    queryReq.addAndCriteriaList(listNotFinished);
                }
                
                // FIXME : replace by QueryRequest
                QueryParameter qp = qf.getQueryParameter();
                //qp.setTargets(new ArrayList());
                
                // Display query result if no action on form (add ou remove criteria)
                if ( !qf.actionOnFormRequested() ) { 
                    //if (!qf.actionOnFormRequested() || qp.getCriteria().size() != 0) {
                    
                    // Perform the request
                    //Collection qrset = queryReq.findLexieForEdition(this.getUser());
                    queryReq.setLimit(qp.getLimitString());
                    queryReq.setOffset(qp.getOffsetString());
                    Collection qrset = queryReq.findLexieAndTranslation(this.getUser());
                    //Collection qrset = DictionariesFactory.doQuery(qp, this.getUser());
                    
                    /*
                     // FIXME: create QueryRequest method !!!! -> created : findLexieForEdition
                     // Remove contribution linked to another (by getClassifiedFinishedContributionId())
                     // FIXME: change getClassifiedFinishedContributionId() method and tag name
                     ArrayList qrsetClone = new ArrayList();
                     qrsetClone.addAll(qrset);
                     int i = 0;
                     while ( i < qrset.size() ) {
                         QueryResult qr = (QueryResult) qrset.get(i);
                         VolumeEntry ve = qr.getSourceEntry();
                         
                         // Search link
                         int j = 0;
                         boolean isRemove = false;
                         while ( (!isRemove) && (j < qrsetClone.size()) ) {
                             QueryResult qrClone = (QueryResult) qrsetClone.get(j);
                             VolumeEntry veClone = qrClone.getSourceEntry();
                             isRemove = ve.getContributionId().equals(veClone.getClassifiedFinishedContributionId());
                             j++;
                         }
                         
                         // Remove contribution linked with another
                         if ( isRemove ) {
                             qrset.remove(i);
                         } else {
                             i++;
                         }
                     }
                     */
                    
                    //
                    //addEntriesTable(qrset, qp);
                    XHTMLElement queryResultForm = content.getElementQueryResultForm();
                    Node viewQueryResultNode = ViewQueryResult.createNodeResult(this.getComms(), this.getSessionData(), this.getUrl(), this.getUser(), qrset, qp, true);
                    queryResultForm.appendChild(content.importNode(viewQueryResultNode, true));
                    queryResultForm.removeAttribute("id");
                    
                    } else {
                        
                        //
                        //addEntriesTable(new Vector(), qp);
                        XHTMLElement queryResultForm = content.getElementQueryResultForm();
                        Node viewQueryResultNode = ViewQueryResult.createNodeResult(this.getComms(), this.getSessionData(), this.getUrl(), this.getUser(), new Vector(), qp, true);
                        queryResultForm.appendChild(content.importNode(viewQueryResultNode, true));
                    }
                } else {
                    //removeEntryListTemplate();
                }
                    
                //
                XHTMLElement formHolder = content.getElementQueryForm();
                formHolder.appendChild(content.importNode(qf.getQueryFormNode("LexalpEditEntryInit.po"), true));
                
                //
                return content.getElementEditEntryInitContent();
        }
    
    protected void updateCreationInterface(String volumeName, String headword)
        throws fr.imag.clips.papillon.business.PapillonBusinessException,
        HttpPresentationException,
        java.io.UnsupportedEncodingException {
            
            // Addin the headword
            HTMLInputElement headwordInputElt = content.getElementHeadword();
            headwordInputElt.setValue(headword);
            
            // Adding the appropriate source languages to the source list
            HTMLOptionElement volumeOptionTemplate = content.getElementVolumeOptionTemplate();
            Node volumeSelect = volumeOptionTemplate.getParentNode();
            volumeOptionTemplate.removeAttribute("id");
            // We assume that the option element has only one text child
            // (it should be this way if the HTML is valid...)
            Text volumeTextTemplate = (Text)volumeOptionTemplate.getFirstChild();
            
            //
            for (Iterator iter = VolumesFactory.getVolumesArray().iterator(); iter.hasNext();) {
                Volume myVolume = (Volume)iter.next();
                
                //
                String itf = myVolume.getTemplateInterface();
            }
            volumeSelect.removeChild(volumeOptionTemplate);
        }
}

