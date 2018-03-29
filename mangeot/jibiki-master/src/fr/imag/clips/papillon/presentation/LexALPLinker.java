/*
 *  papillon
 *
 *  Enhydra super-servlet
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.7  2006/08/10 22:17:13  fbrunet
 *  - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 *  - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 *  - Bug correction : +/- in advanced search
 *
 *  Revision 1.6  2006/05/22 22:45:54  fbrunet
 *  LexALP: add merge method in post-save processing (merge axies with same referenced lexies)
 *
 *  Revision 1.5  2006/03/01 15:12:31  mangeot
 *  Merge between maintrunk and LEXALP_1_1 branch
 *
 *  Revision 1.4.4.3  2006/01/25 15:22:23  fbrunet
 *  Improvement of QueryRequest
 *  Add new search criteria
 *  Add modified status
 *
 *  Revision 1.4.4.2  2006/01/24 13:39:49  fbrunet
 *  Modification view management
 *  Modification LexALP postprocessing
 *
 *  Revision 1.4.4.1  2005/12/02 10:04:09  fbrunet
 *  Add Pre/Post edition processing
 *  Add index reconstruction
 *  Add new query request
 *  Add fuzzy search
 *  Add new contribution administration
 *  Add xsl transformation volume
 *
 *  Revision 1.4  2005/07/21 09:37:47  serasset
 *  LexALPLinker had a pb with package since MM modification.
 *  Lexalp query menu leads to AdvancedSearch.
 *  XslSheetFactory's get default xsl for dict and volume now sets the names to "" during fallback.
 *
 *  Revision 1.3  2005/07/16 16:25:26  mangeot
 *  Adapted the linker to the GDEF project + bug fixes
 *
 *  Revision 1.2  2005/07/16 12:58:31  serasset
 *  Added limit parameter to query functions
 *  Added a parameter to Formater initializations
 *  Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 *  Revision 1.1  2005/07/08 08:22:46  serasset
 *  Reviewed the Abstract/BasePO hierarchy (moved some methods up in the tree).
 *  Added base classes to allow independant browsing window to establish links during edition.
 *
 *  -----------------------------------------------
 *  Class implementing small windows used to link elements.
 */
package fr.imag.clips.papillon.presentation;

import fr.imag.clips.papillon.presentation.xhtmllexalp.orig.LinkerSearchFormXHTML;
import fr.imag.clips.papillon.presentation.xhtmllexalp.orig.LinkerResultListXHTML;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;

import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;
import com.lutris.appserver.server.Enhydra;

import org.w3c.dom.Node;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Comment;
import org.w3c.dom.Text;

import org.enhydra.xml.xhtml.dom.*;

import org.enhydra.xml.xmlc.*;
//import org.enhydra.xml.xmlc.html.*;
import com.lutris.logging.*;
import com.lutris.util.*;

// For the cookies
//import javax.servlet.http.Cookie;
//import fr.imag.clips.papillon.business.user.UsersFactory;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Vector;
import java.util.Collection;
import java.util.Iterator;
import java.text.DateFormat;

//pour le debogage
import fr.imag.clips.papillon.business.utility.*;
import fr.imag.clips.papillon.business.dictionary.*;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.PapillonLogger;

import org.enhydra.xml.io.OutputOptions;
import org.enhydra.xml.io.DOMFormatter;

/**
*  Description of the Class
 *
 * @author     serasset
 * @created    July 6, 2005
 */
public class LexALPLinker extends LinkerBasePO {
        
    
    //
    private static final String PARAMETER_ALL= "*ALL*";
    
    // Search form Parameters
    public class SearchFormParameters {
        public boolean isInitialized = false;
        
        public String [] volumes;
        public String [] dictionaries;
        public String facetValue;
        public String sourceLang;
        public String facet;
        public String comparator;
        public String xsl;
        public String comparisonOperator;
        public String [] targets;
        
        // Parameter initialization does not depend on the exact shape of the search form.
        public void initializeSearchParameters(AbstractPO po) 
        throws java.io.UnsupportedEncodingException, 
            com.lutris.appserver.server.httpPresentation.HttpPresentationException
        {
            if (! isInitialized) {
                // Get parameters
                volumes = po.myGetParameterValues(LinkerSearchFormXHTML.NAME_VOLUMES);
                targets = po.myGetParameterValues(LinkerSearchFormXHTML.NAME_TARGETS);
                dictionaries = po.myGetParameterValues(LinkerSearchFormXHTML.NAME_DICTIONARIES);

                facetValue = po.myGetParameter(LinkerSearchFormXHTML.NAME_FACETVALUE);
                facetValue = (null == facetValue) ? "" : facetValue;
                sourceLang = po.myGetParameter(LinkerSearchFormXHTML.NAME_SOURCE);
                facet = po.myGetParameter(LinkerSearchFormXHTML.NAME_FACET);
                //comparator = po.myGetParameter(LinkerSearchFormXHTML.NAME_OPERATOR);
                //comparisonOperator = (comparator != null) ? Integer.valueOf(comparator).intValue() : 0;
                comparisonOperator = po.myGetParameter(LinkerSearchFormXHTML.NAME_OPERATOR);
                
                xsl = po.myGetParameter(LinkerSearchFormXHTML.NAME_XSL);
                
                isInitialized = true;
            }
        }
        
    }

    SearchFormParameters parameters = new SearchFormParameters();

    /**
    *  This method should be implemented in the subclass so that it returns
     *  true if this particular request requires the user to be logged in,
     *  otherwise false.
     *
     * @return    Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return true;
    }
    
    /**
    *  This method should be implemented in the subclass so that it returns
     *  true if this particular request requires the user to be logged in,
     *  otherwise false.
     *
     * @return    Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        return true;
    }

    public  Node getBrowserForm() throws Exception {
        LinkerSearchFormXHTML searchForm = (LinkerSearchFormXHTML) 
            MultilingualXHtmlTemplateFactory.createTemplate("fr.imag.clips.papillon.presentation.xhtmllexalp", "LinkerSearchFormXHTML", this.myComms, this.sessionData);

        String idFieldName = myGetParameter(LinkerSearchFormXHTML.NAME_idFieldName);
        String langFieldName = myGetParameter(LinkerSearchFormXHTML.NAME_langFieldName);
        searchForm.getElementIdFieldName().setValue(idFieldName);
        searchForm.getElementLangFieldName().setValue(langFieldName);

        // Parameter initialization is generic
        parameters.initializeSearchParameters(this);
        
        // However, building back the desired searchForm depends on it's exact shape.
        // add volumes languages in the sourceLang menu
        HashSet langSet = new HashSet();
        boolean isAllSourcesOption = false;
        
        //
        for (int i=0; i < parameters.volumes.length; i++ ) {
            String volName = parameters.volumes[i];
            if (volName != null && !volName.equals("") && !volName.equals(PARAMETER_ALL)) {
                Volume vol = VolumesFactory.getVolumeByName(parameters.volumes[i]);
                if (null != vol) {
                    langSet.add(vol.getSourceLanguage());
                }
            } else if (volName.equals(PARAMETER_ALL)) {
                isAllSourcesOption = true;
            }
        }
        
        //
        for (int i=0; i < parameters.dictionaries.length; i++ ) {
            String dictName = parameters.dictionaries[i];
            if (dictName != null && dictName != "") {
                Dictionary dict = DictionariesFactory.getDictionaryByName(parameters.dictionaries[i]);
                if (null != dict) {
                    
                    //
                    for (Iterator iter = dict.getSourceLanguagesArray().iterator(); iter.hasNext();) 
                        langSet.add((String)iter.next());
                }
            }
        }
        
        // All languages
        XHTMLOptionElement allSourcesOption = searchForm.getElementAllSourcesOption();
        allSourcesOption.removeAttribute("id");
        if (!isAllSourcesOption) allSourcesOption.getParentNode().removeChild(allSourcesOption);
        
        
        //
        if (!langSet.isEmpty()) {
            String[] langArray = (String[]) langSet.toArray(new String[langSet.size()]);
            Arrays.sort(langArray);
            XHTMLOptionElement sourceOptionTemplate = searchForm.getElementSourceOptionTemplate();
            Node sourceSelect = sourceOptionTemplate.getParentNode();
            sourceOptionTemplate.removeAttribute("id");
            Text sourceTextTemplate = (Text)sourceOptionTemplate.getFirstChild(); 
            for (int i=0; i < langArray.length; i++ ) {
                sourceOptionTemplate.setValue(langArray[i]);
                sourceOptionTemplate.setLabel(langArray[i]);
                sourceTextTemplate.setData(langArray[i]);
                sourceSelect.appendChild(sourceOptionTemplate.cloneNode(true));
            }
            sourceSelect.removeChild(sourceOptionTemplate);
        }
        
        // update searchForm with requested parameters
        searchForm.getElementValueField().setValue(parameters.facetValue);
        searchForm.getElementXsl().setValue(parameters.xsl);

        XHTMLInputElement volTemplate = searchForm.getElementVolumes();
        volTemplate.removeAttribute("id");
        Node volParent = volTemplate.getParentNode();
        for (int i=0; i < parameters.volumes.length; i++) {
            volTemplate.setValue(parameters.volumes[i]);
            volParent.insertBefore(volTemplate.cloneNode(true), volTemplate);
        }
        volParent.removeChild(volTemplate);
        
        XHTMLInputElement dictTemplate = searchForm.getElementDictionaries();
        dictTemplate.removeAttribute("id");
        Node dictParent = dictTemplate.getParentNode();
        for (int i=0; i < parameters.dictionaries.length; i++) {
            dictTemplate.setValue(parameters.dictionaries[i]);
            dictParent.insertBefore(dictTemplate.cloneNode(true), dictTemplate);
        }
        dictParent.removeChild(dictTemplate);
        
        // Targets should be absent in the linker searchForm (monolingual search only...)
        Element targets = searchForm.getElementTargets();
        targets.getParentNode().removeChild(targets);
        //searchForm.getElementTargets().setValue(parameters.targets);
        
        setSelected(searchForm.getElementSourceLang(), parameters.sourceLang);
        setSelected(searchForm.getElementFacet(), parameters.facet);
        setSelected(searchForm.getElementOperator(), parameters.comparisonOperator);
        //setSelected(searchForm.getElementOperator(), Integer.toString(parameters.comparisonOperator));

        return (Node) searchForm.getElementAdvancedSearchForm();
    }
    
    public Node getResultList() throws Exception {
        LinkerResultListXHTML resultsListTmpl = (LinkerResultListXHTML) MultilingualXHtmlTemplateFactory.createTemplate("fr.imag.clips.papillon.presentation.xhtmllexalp", "LinkerResultListXHTML", this.myComms, this.sessionData);

        parameters.initializeSearchParameters(this);
        Collection results = new Vector();
        
        //
        if ((parameters.facetValue != null) && (!parameters.facetValue.equals(""))) {
            
            //
            QueryRequest query = new QueryRequest(VolumesFactory.getVolumesArray());
                        
            //
            QueryCriteria criteria = new QueryCriteria();
            criteria.add("key", QueryCriteria.EQUAL, parameters.facet);
            criteria.add("value", parameters.comparisonOperator, parameters.facetValue);
            if (!parameters.sourceLang.equals(PARAMETER_ALL)) criteria.add("lang", QueryCriteria.EQUAL, parameters.sourceLang);
            query.addCriteria(criteria);
            
            
            /*
            QueryCriteria criteriaFinishedStatus = new QueryCriteria();
            criteriaFinishedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);  
            criteriaFinishedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.FINISHED_STATUS);
            //list.add(criteriaFinishedStatus);
            query.addCriteria(criteriaFinishedStatus);
            */
            
            ArrayList listStatus = new ArrayList();
            
            QueryCriteria criteriaFinishedStatus = new QueryCriteria();
            criteriaFinishedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);  
            criteriaFinishedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.FINISHED_STATUS);
            listStatus.add(criteriaFinishedStatus);
            
            QueryCriteria criteriaValidatedStatus = new QueryCriteria();
            criteriaValidatedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);
            criteriaValidatedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.MODIFIED_STATUS);
            listStatus.add(criteriaValidatedStatus);
            
            query.addOrCriteriaList(listStatus);
            
            
            //
            results = query.findLexie(this.getUser());
        }
        
        
        /*
        String[] key = new String[4];
        key[0] = parameters.facet;
        key[1] = parameters.sourceLang; /// mmm sourceLang determines in which dictionary to look for. should be null here ?
        key[2] = parameters.facetValue;
        key[3] = IQuery.QueryBuilderStrategy[parameters.comparisonOperator];
        
        Vector keys = new Vector();
        keys.add(key);
        
        Collection results = new Vector();
        for (int i=0; i < parameters.volumes.length; i++) {
            String volName = parameters.volumes[i];
            if (volName != null && volName != "") {
                Volume vol = VolumesFactory.getVolumeByName(volName);
                if (vol.getSourceLanguage().equals(parameters.sourceLang)) {
                    Dictionary dict = DictionariesFactory.getDictionaryByName(vol.getDictname());
                    results.addAll(VolumeEntriesFactory.getVolumeEntriesVector(dict,vol,keys,null,null,0, 0));
                }
            }
        }
         */
        
        // Get the list of volumes in which to perform the request (filter the list by source language)
        // Perform the request.
        // display a list view of the result set.
        XHTMLElement noResultsMessage = resultsListTmpl.getElementNoResultsMessage();
        XHTMLElement resultLine = resultsListTmpl.getElementResults();
        XHTMLElement entryNode = resultsListTmpl.getElementEntry();
        XHTMLAnchorElement action = resultsListTmpl.getElementAction();

        noResultsMessage.removeAttribute("id");
        resultLine.removeAttribute("id");
        entryNode.removeAttribute("id");
        action.removeAttribute("id");
        

        if (results.size() != 0) 
            noResultsMessage.getParentNode().removeChild(noResultsMessage);
        
        Iterator iter = results.iterator();
        while(iter.hasNext()) {
            QueryResult qres = (QueryResult) iter.next();
            VolumeEntry ve = qres.getSourceEntry();
            //VolumeEntry ve = (VolumeEntry) iter.next();
            QueryResult qr = new QueryResult(QueryResult.UNIQUE_RESULT, ve);
            //ResultFormatter rf = ResultFormatterFactory.getFormatter(qr, "short-list", ResultFormatterFactory.XHTML_DIALECT, null);
            ResultFormatter rf = ResultFormatterFactory.getFormatter(qr, parameters.xsl, ResultFormatterFactory.XHTML_DIALECT, null);
            removeChildren(entryNode);
            Node entryDOM = (Node)rf.getFormattedResult(qr, this.getUser());
            //Utility.writeNodeToSystemOut(entryDOM);
            entryNode.appendChild(resultsListTmpl.importNode(entryDOM, true));
            action.setAttribute("onClick", "updateParent('" + ve.getEntryId() + "', '" + ve.getSourceLanguage() + "')");
            resultLine.getParentNode().insertBefore(resultLine.cloneNode(true), resultLine);
        }
        
        resultLine.getParentNode().removeChild(resultLine);
        
        return (Node) resultsListTmpl.getElementLinkerResultList();
    }
    
    // FIXME: to me added in a utility class. This is done in many places.
    public static void removeChildren(Node n) {
        while(n.hasChildNodes()) 
            n.removeChild(n.getFirstChild());
    }
    
}

