/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.Enhydra;
import com.lutris.appserver.server.session.SessionManager;
import com.lutris.appserver.server.session.SessionException;
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.*;
import com.lutris.dods.builder.generator.query.QueryBuilder;


import fr.imag.clips.papillon.business.message.MessageDBLoader;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.dictionary.QueryCriteria;
import fr.imag.clips.papillon.business.dictionary.QueryRequest;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.Volume;
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

import fr.imag.clips.papillon.presentation.xhtml.orig.*;


public class History extends PapillonBasePO {
    
    protected final static String VolumeName_PARAMETER = "VolumeName";
    protected final static String EntryHandle_PARAMETER = "EntryHandle";
    
    
    protected boolean loggedInUserRequired() {
        return true;
    }
    
    protected boolean userMayUseThisPO() {
        return true;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }
    
    
    public Node getContent() throws HttpPresentationException {
        
        // Management of the parameters
        String volumeName = myGetParameter(VolumeName_PARAMETER);
        String entryHandle = myGetParameter(EntryHandle_PARAMETER);

        String action = myGetParameter("action");
 
        if (action != null && !action.equals("")) {
            throw new ClientPageRedirectException("Home.po?" + this.getComms().request.getQueryString());
        }
        
        //
        HistoryXHTML content = (HistoryXHTML) MultilingualXHtmlTemplateFactory.createTemplate("HistoryXHTML", this.myComms, this.sessionData);
                
        // Perform the request
        // FIXME: put findLexieHistory in QueryRequestFactory !!
        Collection EntryHistoryCollection = QueryRequest.findLexieHistory(entryHandle, VolumesFactory.getVolumeByName(volumeName), this.getUser());
        
        // Display entries
        if ( (EntryHistoryCollection != null) && (EntryHistoryCollection.size() > 0) ) {
            Utility.removeElement(content.getElementEntryListTable());
            addEntries(content, EntryHistoryCollection);
            Utility.removeElement(content.getElementSorryMessage());
        } else {
            Utility.removeElement(content.getElementEntryListTable());
            Utility.removeElement(content.getElementVolumeEntriesTable());
        }
      
        //
        return content.getElementHistoryContent();
    }
    
    
    /**
    *  Adds a feature to the FewEntries attribute of the Home object
     *
     * @param  EntryCollection  The feature to be added to the FewEntries attribute
     * @exception  fr.imag.clips.papillon.business.PapillonBusinessException    Description of the Exception
     */
    protected void addEntries(HistoryXHTML content, Collection qrset) throws fr.imag.clips.papillon.business.PapillonBusinessException {
        
        //
        if (qrset != null && qrset.size() > 0) {
            
            //
            Iterator myIterator = qrset.iterator();
            while ( myIterator.hasNext() ) {
				QueryResult qr = (QueryResult) myIterator.next();
                
                // get the apropriate transformer.
                ResultFormatter rf = ResultFormatterFactory.getFormatter(qr, null, ResultFormatterFactory.XHTML_DIALECT,null);
                //getFormatter(Dictionary dict, Volume vol, Object parameter, int dialect, String lang)
                addElement(content, (Element)rf.getFormattedResult(qr, this.getUser()), qr.getSourceEntry());
            }
        } else {
            Utility.removeElement(content.getElementEntryListTable());
        }
        Utility.removeElement(content.getElementEntryRow());
    }
    
    
    /**
    *  Adds a feature to the Element attribute of the Home object
     *
     * @param  element          The feature to be added to the Element attribute
     * @param  resourceName     The feature to be added to the Element attribute
     * @exception  PapillonBusinessException  Description of the Exception
     */
    protected void addElement(HistoryXHTML content, Element element, VolumeEntry ve) throws PapillonBusinessException {
        try {
            //for the entry content
            XHTMLTableRowElement originalEntryRow = content.getElementEntryRow();
            Node entryTable = originalEntryRow.getParentNode();
            
            //for the entry content
            content.getElementResourceName().removeAttribute("id");
            content.setTextResourceName(ve.getDictionaryName());
            
            //for the lexie content
            XHTMLElement entryCell = (XHTMLElement) content.getElementEntryDiv();
            entryCell.removeAttribute("id");
            
            // 
            content.getElementModificationAuthor().removeAttribute("id");
            content.setTextModificationAuthor(ve.getModificationAuthor());
            content.getElementModificationDate().removeAttribute("id");
            content.setTextModificationDate(ve.getModificationDate());
            if ( (ve.getModificationComment() != null) 
                && (!ve.getModificationComment().equals("")) ) {
                content.getElementModificationComment().removeAttribute("id");
                content.setTextModificationComment(ve.getModificationComment());
            } else {
                Utility.removeElement(content.getElementModificationComment());
            }
            
            //
            if (entryCell.getChildNodes().getLength() > 0) {
                entryCell.removeChild(entryCell.getFirstChild());
            }
            entryCell.appendChild(content.importNode(element, true));
            
            //
            XHTMLTableRowElement entryRow = (XHTMLTableRowElement) originalEntryRow.cloneNode(true);
            entryRow.removeAttribute("id");
            entryTable.appendChild(entryRow);
            
            // Don't remove the original node in order to add more entries ...
            // entryTable.removeChild(entryRow);
        } catch (Exception ex) {
            throw new PapillonBusinessException("Exception in addEntries: ", ex);
        }
    }
}
