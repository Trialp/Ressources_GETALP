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
 * Revision 1.10  2007/03/25 22:00:57  fbrunet
 * improved avancedqueryform javascript
 * bug correction: in ViewQueryResult class, encode url criteria in UTF-8
 *
 * Revision 1.9  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.8  2006/06/19 15:27:01  fbrunet
 * Jibiki : improvement of the search result display
 * Lexalp : add help menu (link to wiki and bug tracker)
 *
 * Revision 1.7  2006/06/06 09:15:10  fbrunet
 * Bug correction : view action in advanced search page if user is registered.
 *
 * Revision 1.6  2006/06/01 22:05:05  fbrunet
 * New interface, quick search, new contribution management (the first edition not create new contribution. New contribution is created after add, remove element, update, save, etc. in the interface window)
 *
 * Revision 1.5  2006/04/24 13:50:37  fbrunet
 * *** empty log message ***
 *
 * Revision 1.4  2006/04/24 13:43:29  fbrunet
 * Add new class ViewQueryResult : allow to use one class to create result display in advancedSearch and EditEntryInit (like advancedQueryForm)
 * Improve result display : view n results per page
 *
 * Revision 1.3  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 * Revision 1.2  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.1.4.8  2006/02/17 15:16:42  mangeot
 * Do not display the list of all XSL on the search form any more. Displays only a list of XSL descriptions
 *
 * Revision 1.1.4.7  2006/02/17 13:21:25  mangeot
 *
 * MM: modified AdvancedQueryForm. getAllTargetLanguages, getAllSourceLanguages 
 * and getCdmElementsWithDefaultLanguage are now static in AvailableLanguages.java in order to accelerate the execution.
 *
 * Revision 1.1.4.6  2006/01/25 16:51:46  fbrunet
 * *** empty log message ***
 *
 * Revision 1.1.4.5  2006/01/25 15:22:23  fbrunet
 * Improvement of QueryRequest
 * Add new search criteria
 * Add modified status
 *
 * Revision 1.1.4.4  2006/01/24 13:39:49  fbrunet
 * Modification view management
 * Modification LexALP postprocessing
 *
 * Revision 1.1.4.3  2005/12/08 13:44:53  fbrunet
 * *** empty log message ***
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
 * Revision 1.1.2.2  2005/07/22 13:28:32  serasset
 * Modified EditEntryInit for Lexalp. It now serves as a main page for db maintenance.
 * Added a function to get url for QueryParameter.
 * Modified the way xslsheets are handled in order to allow several xslsheet with the same name, different dicts.
 *
 * Revision 1.1.2.1  2005/07/21 12:37:43  serasset
 * Added flags for new legal systems
 * AdvancedSearch now does not query the db when no parameter is given.
 *
 * Revision 1.1  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 *
 *-----------------------------------------------
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.Enhydra;
import com.lutris.appserver.server.session.SessionManager;
import com.lutris.appserver.server.session.SessionException;
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.*;
import com.lutris.dods.builder.generator.query.QueryBuilder;


import fr.imag.clips.papillon.business.message.MessageDBLoader;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.dictionary.QueryParameter;
import fr.imag.clips.papillon.business.dictionary.QueryCriteria;
import fr.imag.clips.papillon.business.dictionary.QueryRequest;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.transformation.ResultFormatterFactory;

import fr.imag.clips.papillon.presentation.AdvancedQueryForm;

// Standard imports
import java.io.IOException;

import java.lang.System;
import java.util.Properties;
import java.util.Enumeration;
import java.util.Collection;
import java.util.Iterator;
import java.util.ArrayList;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

public class AdvancedSearch extends PapillonBasePO {
    
    protected boolean loggedInUserRequired() {
        return false;
    }
    
    protected boolean userMayUseThisPO() {
        return true;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }
    
    public Node getContent()
    throws HttpPresentationException, IOException, SessionException {
        
        
        ////// Handle action events (edit, duplicate, delete, undete)
        // Retrieve parameters 
        String action = myGetParameter(EditEntryInitFactory.ACTION_PARAMETER);
        String volumeName = myGetParameter(EditEntryInitFactory.VOLUME_PARAMETER);
        String entryHandle = myGetParameter(EditEntryInitFactory.HANDLE_PARAMETER);
        
        // 
        if (action!=null && !action.equals("") &&
            volumeName!=null && !volumeName.equals("") &&
            entryHandle!=null && !entryHandle.equals("")) {
            
            // Search last contribution corresponding to entryId
            VolumeEntry myEntry = VolumeEntriesFactory.findEntryByHandle(volumeName, entryHandle);
             
            // EDIT
            if (action.equals("edit")) {
                EditEntryInitFactory.editEntry(myEntry, this.getUser());
                
                // DUPLICATE
            } else if (action.equals("duplicate")) {
                EditEntryInitFactory.duplicateEntry(myEntry, this.getUser());                    
                
                // DELETE
            } else if (action.equals("delete")) {
                EditEntryInitFactory.deleteEntry(myEntry, this.getUser()); 
                
                // UNDELETE
            } else if (action.equals("undelete")) {
                EditEntryInitFactory.undeleteEntry(myEntry, this.getUser());
            }      
        }
        
        
        ////// Create Advanced search page
        AdvancedSearchXHTML content = (AdvancedSearchXHTML) MultilingualXHtmlTemplateFactory.createTemplate("AdvancedSearchXHTML", this.myComms, this.sessionData);
        
        // Retrieve parameters like AdvancedQueryForm object
        AdvancedQueryForm qf = new AdvancedQueryForm(this.getComms(), this.getSessionData(), true, false);
        
        // Import query form node in advanced search page
        XHTMLElement formHolder = content.getElementQueryForm();
        formHolder.appendChild(content.importNode(qf.getQueryFormNode("AdvancedSearch.po"), true));
        
        
        ////// Display query result
        // Create query request
        QueryRequest queryReq = qf.getQueryRequest();
        
		if(queryReq.isEmpty()){
			//PapillonLogger.writeDebugMsg("---qr advanced is empty");
		}
		
        // Display query result if query request have criteria
        // Display query result if no action on form (add ou remove criteria)
        if (!queryReq.isEmpty() && !qf.actionOnFormRequested()) {
    		//PapillonLogger.writeDebugMsg("---qr advanced is not empty");
           
            // Add status criteria
    /*        ArrayList listStatus = new ArrayList();
            
			QueryCriteria criteriaStatus = new QueryCriteria();
			criteriaStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);
			criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.CLASSIFIED_FINISHED_STATUS);
			criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.CLASSIFIED_NOT_FINISHED_STATUS);
			criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.DELETED_STATUS);
			criteriaStatus.add("lang", QueryCriteria.EQUAL, Volume.DEFAULT_LANG);
			listStatus.add(criteriaStatus);
			
			queryReq.addOrCriteriaList(listStatus);*/
            
            // Perform the request
            Collection qrset = queryReq.findLexieAndTranslation(this.getUser());
            
            // Display result
            XHTMLElement queryResultForm = content.getElementQueryResultForm();
            //FIXME: user is identified -> (this.getUser()!=null && !this.getUser().isEmpty())
            // - create a method isIdentify ? create a guest user ? introduce in createNodeResult ?
            Node viewQueryResultNode = ViewQueryResult.createNodeResult(this.getComms(), this.getSessionData(), this.getUrl(), this.getUser(), qrset, qf.getQueryParameter(), (this.getUser()!=null && !this.getUser().isEmpty()));
            queryResultForm.appendChild(content.importNode(viewQueryResultNode, true));
            queryResultForm.removeAttribute("id");
        }
        
        
        //
        return content;
        //return content.getElementAdvancedSearchPage();
    }
    
}

