/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 *
 * © GETA CLIPS IMAG
 *
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.10.2.1  2007/10/30 13:24:50  serasset
 * BUG 161: delete action now works in CreateEntryInit page.
 *
 * Revision 1.10  2007/03/25 22:00:57  fbrunet
 * improved avancedqueryform javascript
 * bug correction: in ViewQueryResult class, encode url criteria in UTF-8
 *
 * Revision 1.9  2007/01/05 12:57:49  fbrunet
 * Add undo draft method (bug in EditEntry.java : undo after last finish contribution)
 * Modify transformation method
 *
 * Revision 1.8  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.7  2006/06/19 15:27:01  fbrunet
 * Jibiki : improvement of the search result display
 * Lexalp : add help menu (link to wiki and bug tracker)
 *
 * Revision 1.6  2006/06/01 22:05:05  fbrunet
 * New interface, quick search, new contribution management (the first edition not create new contribution. New contribution is created after add, remove element, update, save, etc. in the interface window)
 *
 * Revision 1.5  2006/05/05 02:08:23  fbrunet
 * bug correction : url utf8 transfert (in createEntryInit)
 *
 * Revision 1.4  2006/04/24 13:50:37  fbrunet
 * *** empty log message ***
 *
 * Revision 1.3  2006/04/24 13:43:29  fbrunet
 * Add new class ViewQueryResult : allow to use one class to create result display in advancedSearch and EditEntryInit (like advancedQueryForm)
 * Improve result display : view n results per page
 *
 * Revision 1.2  2006/04/18 14:30:24  fbrunet
 * Authorize admin to edit all entries
 *
 * Revision 1.1  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 */

package fr.imag.clips.papillon.presentation;


//General java imports
import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;
import java.util.ArrayList;
import java.util.Hashtable;

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


// HTML source import
import fr.imag.clips.papillon.presentation.xhtml.orig.*;

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



public class CreateEntryInit extends PapillonBasePO {
    
    

    protected final static String ContributionsVolumeParameter = "VOLUME";
    
    protected final static String ConsultExpertVolumeParameter = "VOLUME";
    protected final static String ConsultExpertHandleParameter = "handle";
    protected final static String ConsultExpertFormatterParameter = "xslid";
    protected final static String CreateEntryInitURL = "CreateEntryInit.po";
    protected final static String EditingErrorURL = "EditingError.po";
    protected final static String XML_FORMATTER = fr.imag.clips.papillon.business.transformation.XslTransformation.XML_FORMATTER; 

    protected CreateEntryInitXHTML content;
    private Hashtable contentParameters;
    
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
            
            // Content
            content = (CreateEntryInitXHTML)MultilingualXHtmlTemplateFactory.createTemplate("CreateEntryInitXHTML", this.getComms(), this.getSessionData());
            
            // Header javascript
            XHTMLScriptElement initJavascriptElement = (XHTMLScriptElement)content.getElementInitJavascript();
            this.setHeaderScript(initJavascriptElement);
            removeInitJavascript();
                
            // Parameters
            String volumeName = myGetParameter(EditEntryInitFactory.VOLUME_PARAMETER);
            String volumeAnyway = myGetParameter(EditEntryInitFactory.VOLUME_ANYWAY_PARAMETER);
            String headwordAnyway = myGetParameter(EditEntryInitFactory.HEADWORD_ANYWAY_PARAMETER);
            String entryHandle = myGetParameter(EditEntryInitFactory.HANDLE_PARAMETER);
            String formatter = myGetParameter(EditEntryInitFactory.FORMATTER_PARAMETER);
            String action = myGetParameter(EditEntryInitFactory.ACTION_PARAMETER);
            String go = myGetParameter(content.NAME_GO);
            
            // Headword parameter
            String headword = "";
            if ( myGetParameter(this.getComms().request.getHttpServletRequest(), AdvancedQueryFormXHTML.NAME_FACET + ".0") != null
                 && myGetParameter(this.getComms().request.getHttpServletRequest(), AdvancedQueryFormXHTML.NAME_FACET + ".0").equals(Volume.CDM_headword) ) {
                headword = myGetParameter(this.getComms().request.getHttpServletRequest(), AdvancedQueryFormXHTML.NAME_FACETVALUE + ".0");
            } else {
                removeCreateAnywayForm();
            }
            
            //
            if (volumeName!=null && !volumeName.equals("")) {
                this.setPreference(content.NAME_VOLUME, volumeName);
            }
            else {
                volumeName = this.getPreference(content.NAME_VOLUME);
            }
            
            //
            User myUser = getUser();
            if (null != myUser) {
                content.setTextUserName(myUser.getName());
            }
            
            //
            if (action!=null && !action.equals("")) {

                //PapillonLogger.writeDebugMsg("action " + action);

                    // LOOKUP
                //if (action.equals("lookup")
                //    && volume!=null && !volume.equals("")) {

                    //
                //    displayLookupResults(volume);

                    // LOOKUP or CREATE
                //} else

                if ( (action.equals("create") || action.equals("lookup"))
                           && volumeName!=null && !volumeName.equals("")
                           && headword!=null && !headword.equals("")) {
                    
                    //
                    displayLookupResultsAndCreate(VolumesFactory.getVolumeByName(volumeName), headword);
                    
                    // fill anyway parameter
                    volumeAnyway = new String(volumeName);
                    headwordAnyway = new String(headword);
                    
                    // fill headword parameter (if results !)
                    headword = "";
                    
                    // SHOW AND CREATE
                } else if (action.equals("showAndCreate")
                           && volumeAnyway !=null && !volumeAnyway.equals("") 
                           && headwordAnyway!=null && !headwordAnyway.equals("")) {
                    
                    //
                    content.setTextHEADWORD_FINAL(headwordAnyway);
                    content.setTextVOLUME_FINAL(volumeAnyway);
                    XHTMLInputElement goInputElement = content.getElementGO();
                    goInputElement.setValue("go");
                    
                    //
                    //removeNbResultsSpan();
                    //removeEntryListTable();
                    
                    // CREATE
                } else if (action.equals("createAnyway")
                           && volumeAnyway!=null && !volumeAnyway.equals("") 
                           && headwordAnyway!=null && !headwordAnyway.equals("")) {
                    
                    //
                    EditEntryInitFactory.createEntry(volumeAnyway, headwordAnyway, this.getUser());
                    
                    // EDIT
                } else if (action.equals("edit")
                           && entryHandle!=null && !entryHandle.equals("")) {
                    
                    //
                    VolumeEntry myEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
                    EditEntryInitFactory.editEntry(myEntry, this.getUser());
                    
                    // DUPLICATE
                } else if (action.equals("duplicate")
                           && entryHandle!=null && !entryHandle.equals("")) {
                    
                    //
                    VolumeEntry myEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
                    EditEntryInitFactory.duplicateEntry(myEntry, this.getUser());                    
                    
                    // DELETE
                } else if (action.equals("delete")
                           && entryHandle!=null && !entryHandle.equals("")
                           && volumeName!=null && !volumeName.equals("")) {
                    
                    //
                    VolumeEntry myEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
                    EditEntryInitFactory.deleteEntry(myEntry, this.getUser()); 
                    // FIXME: Here we would like to update the current view. For this we need the headword
                    // But the headword is not needed for deletion. Hence we don't require it.
                    // TODO: use javascript function for delete/undelete/edit/etc... This will allow different
                    // TODO: urls to be used in different contexts (e.g. embed the headword in the url in CreateEntryInit page.
                    // displayLookupResultsAndCreate(VolumesFactory.getVolumeByName(volumeName), headword);
                    removeShowCreation();
                    
                    // UNDELETE
                } else if (action.equals("undelete")
                           && entryHandle!=null && !entryHandle.equals("")
                           && volumeName!=null && !volumeName.equals("") 
                           && headword!=null && !headword.equals("")) {
                    
                    //
                    VolumeEntry myEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
                    EditEntryInitFactory.undeleteEntry(myEntry, this.getUser());
                    displayLookupResultsAndCreate(VolumesFactory.getVolumeByName(volumeName), headword);
                }      
                
            } else {
                
                //
                //removeEntryListTable();
                //removeNbResultsSpan();
                removeShowCreation();
            }
            
            //
            displayLookupInterface(volumeName, headword, volumeAnyway, headwordAnyway);
            return content.getElementEditEntryInitContent();
        }
    
    
    protected void displayLookupInterface(String volumeName, String headword, String volumeAnyway, String headwordAnyway)
        throws fr.imag.clips.papillon.business.PapillonBusinessException,
        HttpPresentationException,
        java.io.UnsupportedEncodingException {
            
            // Addin the headword
            XHTMLInputElement headwordInputElt = content.getElementValueField();
            headwordInputElt.setValue(headword);
            
            XHTMLInputElement headwordHiddenElt = content.getElementHEADWORD_ANYWAY();
            if (headwordAnyway != null) headwordHiddenElt.setValue(headwordAnyway);
            XHTMLInputElement headwordDisabledElt = content.getElementHEADWORD_DISABLED();
            if (headwordDisabledElt != null && headwordAnyway != null) headwordDisabledElt.setValue(headwordAnyway);
            
            XHTMLInputElement volumeHiddenElt = content.getElementVOLUME_ANYWAY();
            if (volumeAnyway != null) volumeHiddenElt.setValue(volumeAnyway);
            XHTMLInputElement volumeDisabledElt = content.getElementVOLUME_DISABLED();
            if (volumeDisabledElt != null && volumeAnyway != null) volumeDisabledElt.setValue(volumeAnyway);
            
            
            // Adding the appropriate source languages to the source list
            XHTMLOptionElement volumeOptionTemplate = content.getElementVolumeOptionTemplate();
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
                volumeOptionTemplate.setValue(myVolume.getName());
                volumeOptionTemplate.setLabel(myVolume.getName());
                // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux
                // specs W3C.
                volumeOptionTemplate.setSelected(myVolume.getName().equals(volumeName));
                volumeTextTemplate.setData(myVolume.getName());
                volumeSelect.appendChild(volumeOptionTemplate.cloneNode(true));
            }
            volumeSelect.removeChild(volumeOptionTemplate);
        }
    
/*
    protected void displayLookupResults (String volumeName)
        throws PapillonBusinessException,
        java.io.UnsupportedEncodingException,
        java.io.IOException, 
        HttpPresentationException {
            
            //System.out.println("displayLookupResults");
            
            //
            AdvancedQueryForm qf = new AdvancedQueryForm(this.getComms(), this.getSessionData(), true, false);
            QueryParameter qp = qf.getQueryParameter();
            qp.setTargets(new String[0]);
            Collection qrset = searchEntry(volumeName, qf, qp);
            
            //
            if (qrset != null) {
            
                //
                //addEntriesTable(qrset, qp);
                XHTMLElement queryResultForm = content.getElementQueryCreateForm();
                Node viewQueryResultNode = ViewQueryResult.createNodeResult(this.getComms(), this.getSessionData(), this.getUrl(), this.getUser(), qrset, qp, true);
                queryResultForm.appendChild(content.importNode(viewQueryResultNode, true));
                queryResultForm.removeAttribute("id");     
                
                //
                removeShowCreation();
                
            } else {
                this.getSessionData().writeUserMessage("Sorry, entry not found! Please check the volume.");
                //removeEntryListTable();
                removeCreateAnywayForm();
                removeShowCreation();
            }
        }
	*/
    
    protected void displayLookupResultsAndCreate (Volume volume, String headword)
		throws PapillonBusinessException,
		java.io.UnsupportedEncodingException,
        java.io.IOException,
		HttpPresentationException {
            boolean resultsEmpty = true;
			
            AdvancedQueryForm qf = new AdvancedQueryForm(this.getComms(), this.getSessionData(), true, false);
           
            // AdvancedSearch query
            QueryRequest queryReq = qf.getQueryRequest();
            if (!queryReq.isEmpty() && !qf.actionOnFormRequested()) {
                
                //PapillonLogger.writeDebugMsg("---qr advanced is not empty");
                
                
                //FIXME: mettre une option sur le volume pour dire s'il est en cours d'édition ou non ?
      /*          if (!queryReq.isOpenRequest()) {
                    // Add status criteria
                    ArrayList listStatus = new ArrayList();
                    
                    QueryCriteria criteriaStatus = new QueryCriteria();
                    criteriaStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);
                    criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.CLASSIFIED_FINISHED_STATUS);
                    criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.NOT_FINISHED_STATUS);
                    criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.DRAFT_STATUS);
                    criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.DELETED_STATUS);
                    listStatus.add(criteriaStatus);
                    
                    queryReq.addOrCriteriaList(listStatus);
                }*/
                
                // Perform the request
                Collection qrset = queryReq.findLexieAndTranslation(this.getUser());
                
            //
            QueryParameter qp = qf.getQueryParameter();
            //qp.setTargets(new ArrayList());
           // Collection qrset = searchEntry(volume, qf, qp);

			//
			if (qrset != null && qrset.size()!=0) {
                resultsEmpty = false;
                PapillonLogger.writeDebugMsg("CreateEntryInit: results not empty!");
                //
                //addEntriesTable(qrset, qp);
                XHTMLElement queryResultForm = content.getElementQueryCreateForm();
                Node viewQueryResultNode = ViewQueryResult.createNodeResult(this.getComms(), this.getSessionData(), this.getUrl(), this.getUser(), qrset, qp, true);
                queryResultForm.appendChild(content.importNode(viewQueryResultNode, true));
                
                //
                removeShowCreation();
                
            }
        }
        if (resultsEmpty) {
            PapillonLogger.writeDebugMsg("CreateEntryInit: results empty!");
               //
                throw new ClientPageRedirectException(CreateEntryInitURL + "?" +
                                                      EditEntryInitFactory.ACTION_PARAMETER + "=createAnyway" +
                                                      "&" + EditEntryInitFactory.VOLUME_ANYWAY_PARAMETER + "=" + volume.getName() +
                                                      "&" + EditEntryInitFactory.HEADWORD_ANYWAY_PARAMETER + "=" + myUrlEncode(headword));
        }
    }


protected void removeShowCreation() {
    Element myElement = content.getElementShowCreation();
    Node myParent = myElement.getParentNode();
    if (myParent != null)
        myParent.removeChild(myElement);
}


protected void removeCreateAnywayForm () {
    Element myElement = content.getElementCreateAnywayForm();
    Node myParent = myElement.getParentNode();
    myParent.removeChild(myElement);
}

protected void removeInitJavascript () {
    Element myElement = content.getElementInitJavascript();
    Node myParent = myElement.getParentNode();
    myParent.removeChild(myElement);
}

	

protected Collection searchEntry(Volume volume, AdvancedQueryForm qf, QueryParameter qp)
      throws PapillonBusinessException,
      java.io.UnsupportedEncodingException,
      HttpPresentationException {
          
          //
          QueryRequest queryReq = new QueryRequest(volume);
          queryReq.setOrTree();   // OR(AND)
          ArrayList criteriaSearchList = qf.getCriteriaList();
          
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
          
          //
          queryReq.setLimit(qp.getLimitString());
          queryReq.setOffset(qp.getOffsetString());
          
          //
          return queryReq.findLexie(this.getUser());          
  }


/*
 protected void addEntriesTable (Collection qrset, QueryParameter qp) throws
 fr.imag.clips.papillon.business.PapillonBusinessException {
     
     
     XHTMLElement entryNode = content.getElementEntryNode();
     HTMLTableRowElement entryListRow = content.getElementEntryListRow();
     //HTMLAnchorElement entryIdAnchor = content.getElementViewEntryAnchor();
     //HTMLElement headwordElement = content.getElementHeadwordList();
     //HTMLElement posElement = content.getElementPosList();
     HTMLElement authorElement = content.getElementEntryAuthorList();
     HTMLAnchorElement editAnchor = content.getElementEditEntryAnchor();
     HTMLAnchorElement duplicateAnchor = content.getElementDuplicateEntryAnchor();
     HTMLAnchorElement deleteAnchor = content.getElementDeleteEntryAnchor();
     HTMLAnchorElement undeleteAnchor = content.getElementUndeleteEntryAnchor();
     HTMLAnchorElement ViewHistoryEntryAnchor = content.getElementViewHistoryEntryAnchor();
     HTMLAnchorElement viewXmlAnchor = content.getElementViewXmlAnchor();
     HTMLElement underEdition = content.getElementUnderEdition();
     HTMLElement proceedEdition = content.getElementProceedEdition();
     //HTMLElement editElement = content.getElementEditMessage();
     //HTMLElement copyElement = content.getElementCopyMessage();
     
     entryNode.removeAttribute("id");
     //entryIdAnchor.removeAttribute("id");
     //headwordElement.removeAttribute("id");
     //posElement.removeAttribute("id");
     editAnchor.removeAttribute("id");
     duplicateAnchor.removeAttribute("id");
     deleteAnchor.removeAttribute("id");
     undeleteAnchor.removeAttribute("id");
     viewXmlAnchor.removeAttribute("id");
     underEdition.removeAttribute("id");
     proceedEdition.removeAttribute("id");
     //editElement.removeAttribute("id");
     //copyElement.removeAttribute("id");
     authorElement.removeAttribute("id");
     
     content.setTextNbResults(Integer.toString(qrset.size()));
     
     Node entryTable = entryListRow.getParentNode();
     
     Iterator iter = qrset.iterator();
     while(iter.hasNext()) {
         QueryResult qr = (QueryResult) iter.next();
         VolumeEntry myEntry = qr.getSourceEntry();             
         ResultFormatter rf = ResultFormatterFactory.getFormatter(qr, qp.getXsl(), ResultFormatterFactory.XHTML_DIALECT, null);
         Utility.removeChildNodes(entryNode);
         Node entryDOM = (Node)rf.getFormattedResult(qr, this.getUser());
         entryNode.appendChild(content.importNode(entryDOM, true));
         
         // Edit actions...
         // The entry id
         //String href = ConsultExpertURL + "?"
         // + ConsultExpert.VOLUME_PARAMETER + "="
         // + myEntry.getVolumeName() + "&"
         // + ConsultExpert.HANDLE_PARAMETER + "="
         // + myEntry.getHandle();
         
         //entryIdAnchor.setHref(href);
         
         // the entry id
         //content.setTextEntryIdList(myEntry.getId());
         
         // The pos
         //content.setTextPosList(myEntry.getPos());
         
         // The author
         //content.setTextEntryAuthorList(myEntry.getAuthor());
         content.setTextEntryAuthorList(myEntry.getModificationAuthor());
         
         // The status
         if ( !myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS) 
              && !myEntry.getStatus().equals(VolumeEntry.MODIFIED_STATUS) ) {
             // view status : finished, deleted
             content.setTextEntryStatusList(myEntry.getStatus());
             underEdition.setAttribute("style","display: none;");
             proceedEdition.setAttribute("style","display: none;");
             if ( myEntry.getStatus().equals(VolumeEntry.FINISHED_STATUS)
                  && myEntry.getModificationAuthor().equals(this.getUser().getLogin()) ) {
                 entryListRow.setAttribute("class", "myContributionTR");
             } else {
                 entryListRow.setAttribute("class", "contributionTR");
             }
             
         } else if ( myEntry.getStatus().equals(VolumeEntry.MODIFIED_STATUS) ) {
             // view modified status like under edition
             content.setTextEntryStatusList("");
             underEdition.setAttribute("style","display: block;");
             proceedEdition.setAttribute("style","display: none;");
             entryListRow.setAttribute("class", "contributionTR");
             
         } else if ( myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS) ) {
             // view not-finished status
             // like proceed edition if my contribution
             // like under edition if not
             if ( myEntry.getModificationAuthor().equals(this.getUser().getLogin()) ) {
                 content.setTextEntryStatusList("");
                 underEdition.setAttribute("style","display: none;");
                 proceedEdition.setAttribute("style","display: block;");
                 entryListRow.setAttribute("class", "myProceedContributionTR");
             } else {
                 content.setTextEntryStatusList("");
                 underEdition.setAttribute("style","display: block;");
                 proceedEdition.setAttribute("style","display: none;");
                 entryListRow.setAttribute("class", "contributionTR");
             }
         }
         
         // The edit anchor
         String href = this.getUrl() + "?"
             + content.NAME_VOLUME + "="
             + myEntry.getVolumeName() + "&"
             + HANDLE_PARAMETER + "="
             + myEntry.getEntryId() + "&"
             + ACTION_PARAMETER + "=";
         
         //
         if ( (   (!myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) 
                  && (!myEntry.getStatus().equals(VolumeEntry.MODIFIED_STATUS)) )
              ||  ( myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)
                    && myEntry.getModificationAuthor().equals(this.getUser().getLogin())
                    )) {
             
             // the edit button
             if ( !myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS) ) {
                 //if ((myEntry.getAuthor().equals(this.getUser().getLogin())
                 //     || this.getUser().isInGroup("validator"))
                 //    && !myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)) {    
                 editAnchor.setHref(href+"edit");
                 editAnchor.setAttribute("class","");
                 editAnchor.setAttribute("target","_blank");
                 } else {
                     editAnchor.setHref("");
                     editAnchor.setAttribute("class","hidden");
                 }
             
             // the duplicate button
             if ( !myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)
                  && !myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {
                 duplicateAnchor.setHref(href+"duplicate");
                 duplicateAnchor.setAttribute("class","");
                 duplicateAnchor.setAttribute("target","_blank");
             } else {
                 duplicateAnchor.setHref("");
                 duplicateAnchor.setAttribute("class","hidden");
             }
             
             // the delete button
             if (    !myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)
                     && !myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {
                 //if ((myEntry.getAuthor().equals(this.getUser().getLogin())
                 //     || this.getUser().isInGroup("validator"))
                 //    && !myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)
                 //    && !myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {
                 
                 // action delete and refresh query with last parameters
                 deleteAnchor.setHref(href + "delete" + "&" + AdvancedQueryForm.getEncodedUrlForParameter(qp));
                 //deleteAnchor.setAttribute("class","");
                 deleteAnchor.setAttribute("style","display: block;");
                 } else {
                     deleteAnchor.setHref("");
                     //deleteAnchor.setAttribute("class","hidden");
                     deleteAnchor.setAttribute("style","display: none;");
                 }
             
             // the undelete button
             if ( myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS) ) {
                 //if ( myEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)
                 //    && (myEntry.getAuthor().equals(this.getUser().getLogin())
                 //        || this.getUser().isInGroup("validator"))) {
                 
                 // action undelete and refresh query with last parameters
                 undeleteAnchor.setHref(href + "undelete" + "&" + AdvancedQueryForm.getEncodedUrlForParameter(qp));
                 //undeleteAnchor.setAttribute("class","");
                 undeleteAnchor.setAttribute("style","display: block;");
                 } else {
                     undeleteAnchor.setHref("");
                     //undeleteAnchor.setAttribute("class","hidden");
                     undeleteAnchor.setAttribute("style","display: none;");
                 }
             
             } else {
                 // the edit button
                 editAnchor.setHref("");
                 editAnchor.setAttribute("class","hidden");
                 
                 // the duplicate button
                 duplicateAnchor.setHref("");
                 duplicateAnchor.setAttribute("class","hidden");
                 
                 // the delete button
                 deleteAnchor.setHref("");
                 deleteAnchor.setAttribute("style","display: none;");
                 
                 // the undelete button
                 undeleteAnchor.setHref("");
                 undeleteAnchor.setAttribute("style","display: none;");
             }
         
         
         if (this.getUser().isInGroup(Group.ADMIN_GROUP)) {
             // The view XML anchor
             // FIXME : create new page.po like history
             QueryParameter qpxml = new QueryParameter();
             qpxml.setXsl("XML");
             String[] dictNames = new String[1];
             dictNames[0] = myEntry.getDictionaryName();
             qpxml.setDictionaryNames(dictNames);
             Vector crit = new Vector();
             String[] idc = new String[4];
             idc[0] = Volume.CDM_entryId;
             idc[1] = null;
             //if (myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)
             //    && myEntry.getModificationAuthor().equals(this.getUser().getLogin())) {
             //        idc[2] = myEntry.getClassifiedFinishedContributionId(); 
             //} else {
             idc[2] = myEntry.getEntryId(); 
             //}
             //idc[3] = QueryBuilder.EQUAL;
             idc[3] = QueryCriteria.EQUAL;
             crit.add(idc);
             
             //String[] idc = new String[4];
             //idc[0] = Volume.CDM_entryId;
             //idc[1] = null;
             //idc[2] = myEntry.getEntryId();
             //idc[3] = QueryBuilder.EQUAL;
             //idc[3] = QueryCriteria.EQUAL;
             //crit.add(idc);
             
             
             qpxml.setCriteria(crit);
             
             viewXmlAnchor.setHref(href + "lookup" + "&" + AdvancedQueryForm.getEncodedUrlForParameter(qpxml));
         } else {
             
             //
             HTMLElement viewXMLElement = content.getElementViewXML();
             viewXMLElement.setAttribute("style","display: none;");
         }
         
         
         // view history
         //if (!myEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)
         //    && (myEntry.getAuthor().equals(this.getUser().getLogin())
         //        || this.getUser().isInGroup("validator"))) {
         href = "History.po?" + "VolumeName" + "=" + myEntry.getVolumeName() + "&" + "EntryHandle" + "=" + myEntry.getHandle();
         ViewHistoryEntryAnchor.setHref(href);
         ViewHistoryEntryAnchor.setAttribute("style","display: block;");
         //} else {
         //    ViewHistoryEntryAnchor.setHref("");
         //     ViewHistoryEntryAnchor.setAttribute("style","display: none;");
         //}
         
 
 HTMLElement cloneEntry = (HTMLElement)entryListRow.cloneNode(true);
 
 //      we have to take off the id attribute because we did not take it off the original
 cloneEntry.removeAttribute("id");
 entryTable.appendChild(cloneEntry);
 
 //entryNode.getParentNode().insertBefore(entryNode.cloneNode(true), entryNode);
             }
 
 //
 entryNode.getParentNode().removeChild(entryNode);
 removeEntryListTemplate();
             }
 */

/*
 protected void removeEntryListTemplate() {
     Element myElement = content.getElementEntryListRow();
     Node myParent = myElement.getParentNode();
     if (myParent != null)
         myParent.removeChild(myElement);
 }
 
 protected void removeEntryListTable () {
     // PB when I remove the entire table, the following elements disappear from the DOM
     // To fix the pb, always embedd a table in a <div> element
     Element myElement = content.getElementEntryListTable();
     Node myParent = myElement.getParentNode();
     myParent.removeChild(myElement);
 }
 
 protected void removeNbResultsSpan () {
     Element myElement = content.getElementNbResultsSpan();
     Node myParent = myElement.getParentNode();
     myParent.removeChild(myElement);
 }
 
 */
/*
 protected void displayEntry (String volumeName, String handle, String formatter)
 throws PapillonBusinessException,
 java.io.UnsupportedEncodingException,
 HttpPresentationException {
     
     removeEntryListTable();
     
     Volume myVolume = VolumesFactory.getVolumeByName(volumeName);
     String[] targets = null;
     if (myVolume != null && !myVolume.isEmpty()) {
         targets = myVolume.getTargetLanguagesArray();
     }
     Collection EntryCollection = DictionariesFactory.findAnswerAndTranslations(volumeName, handle, targets, this.getUser());
     
     //
     if (EntryCollection != null && EntryCollection.size()>0) {
         QueryResult myQueryResult = (QueryResult) EntryCollection.iterator().next();
         VolumeEntry myEntry = myQueryResult.getSourceEntry();
         // get the apropriate transformer.
         fr.imag.clips.papillon.business.transformation.ResultFormatter rf = fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.getFormatter(myQueryResult, formatter, fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XHTML_DIALECT, null);
         
         Element myXhtmlElt = (Element)rf.getFormattedResult(myQueryResult, this.getUser());
         
         XHTMLElement entryDiv = (XHTMLElement) content.getElementEntryDiv();
         entryDiv.appendChild(content.importNode(myXhtmlElt, true));
     }
     
 }
 */

}
