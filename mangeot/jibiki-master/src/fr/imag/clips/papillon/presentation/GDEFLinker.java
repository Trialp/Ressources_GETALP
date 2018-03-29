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
 *  Revision 1.6  2006/07/25 14:58:36  mangeot
 *  Entries must not be deleted
 *
 *  Revision 1.5  2006/05/22 22:45:54  fbrunet
 *  LexALP: add merge method in post-save processing (merge axies with same referenced lexies)
 *
 *  Revision 1.4  2006/03/01 15:12:31  mangeot
 *  Merge between maintrunk and LEXALP_1_1 branch
 *
 *  Revision 1.3.2.1  2006/01/24 13:39:49  fbrunet
 *  Modification view management
 *  Modification LexALP postprocessing
 *
 *  Revision 1.3  2005/08/01 15:03:41  mangeot
 *  Corrected an important bug in the editor that forbidded to change a boolean value from true to false.
 *  Beware, you have to edit the existing interface templates by hands:
 *  1- duplicate all the input elements with name='boolean' and type='checkbox'.
 *  - for each input element pair,
 *   2- change one input element name into name='booleantrue'
 *   3- change the other input element type to type='hidden'
 *
 *  Revision 1.2  2005/08/01 10:58:22  mangeot
 *  Suppressed the 3rd click on the linker window when only one link has been found
 *
 *  Revision 1.1  2005/07/16 16:42:05  mangeot
 *  iLinker adapted from LexAlpLinker developed by GS
 *
 *
 *  -----------------------------------------------
 *  Class implementing small windows used to link elements.
 */
package fr.imag.clips.papillon.presentation;

import fr.imag.clips.papillon.presentation.xhtmlgdef.orig.LinkerSearchFormXHTML;
import fr.imag.clips.papillon.presentation.xhtmlgdef.orig.LinkerResultListXHTML;

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
public class GDEFLinker extends LinkerBasePO {
        
	protected final static String javascriptHeader = "function loadFunction () {\n";
	protected final static String javascriptBody = "	return 0;\n";
	protected final static String javascriptFooter = "}\n";
		
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
        public int comparisonOperator;
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
                comparator = po.myGetParameter(LinkerSearchFormXHTML.NAME_OPERATOR);
                comparisonOperator = (comparator != null) ? Integer.valueOf(comparator).intValue() : 0;
                
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
            MultilingualXHtmlTemplateFactory.createTemplate("fr.imag.clips.papillon.presentation.xhtmlgdef", "LinkerSearchFormXHTML", this.myComms, this.sessionData);

        String idFieldName = myGetParameter(LinkerSearchFormXHTML.NAME_idFieldName);
        String langFieldName = myGetParameter(LinkerSearchFormXHTML.NAME_langFieldName);
        searchForm.getElementIdFieldName().setValue(idFieldName);
        searchForm.getElementLangFieldName().setValue(langFieldName);

        // Parameter initialization is generic
        parameters.initializeSearchParameters(this);
                        
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
        
        setSelected(searchForm.getElementFacet(), parameters.facet);
        setSelected(searchForm.getElementOperator(), Integer.toString(parameters.comparisonOperator));

        return (Node) searchForm.getElementAdvancedSearchForm();
    }
    
    public Node getResultList() throws Exception {
        LinkerResultListXHTML resultsListTmpl = (LinkerResultListXHTML) MultilingualXHtmlTemplateFactory.createTemplate("fr.imag.clips.papillon.presentation.xhtmlgdef", "LinkerResultListXHTML", this.myComms, this.sessionData);

        parameters.initializeSearchParameters(this);

        Vector keys = new Vector();
        String[] key = new String[4];
        key[0] = parameters.facet;
        key[1] = parameters.sourceLang; /// mmm sourceLang determines in which dictionary to look for. should be null here ?
        key[2] = parameters.facetValue;
        key[3] = IQuery.QueryBuilderStrategy[parameters.comparisonOperator];
        keys.add(key);
        
		/* WE authorize a link to any entry (validated or not) */
		/* 
			String[] key2 = new String[4];
		 key2[0] = Volume.CDM_contributionStatus;
		 key2[1] = Volume.DEFAULT_LANG;
		 key2[2] = VolumeEntry.VALIDATED_STATUS;
		 key2[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];			
		 keys.add(key2);		
		 */
		
		String[] key2 = new String[4];
		key2[0] = Volume.CDM_contributionStatus;
		key2[1] = Volume.DEFAULT_LANG;
		key2[2] = VolumeEntry.REPLACED_STATUS;
		key2[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];			
		keys.add(key2);		
		
		key2[0] = Volume.CDM_contributionStatus;
		key2[1] = Volume.DEFAULT_LANG;
		key2[2] = VolumeEntry.DELETED_STATUS;
		key2[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];			
		keys.add(key2);		
		
        
        
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
        

        if (results.size() == 1) {
			VolumeEntry ve = (VolumeEntry) results.iterator().next();
			String javascriptRedir = javascriptHeader 
				+ "updateParent('" + ve.getEntryId() + "', '" + ve.getSourceLanguage() + "')"
				+ "\n"
				+ javascriptFooter;
			this.addToHeaderScript(javascriptRedir);
			
		} 
        else if (results.size() > 1) {
			this.addToHeaderScript(javascriptHeader + javascriptBody + javascriptFooter);
			noResultsMessage.getParentNode().removeChild(noResultsMessage);
		}
		else {
			String submitButton = myGetParameter(LinkerSearchFormXHTML.NAME_lookup);
			if (submitButton == null || submitButton.equals("")) {
				noResultsMessage.getParentNode().removeChild(noResultsMessage);
			}
			else if (parameters.facet.equals(Volume.CDM_headword)) {
				XHTMLAnchorElement creationAnchor = resultsListTmpl.getElementCreateMissingEntryAnchor();
				creationAnchor.setHref(creationAnchor.getHref() + parameters.facetValue);
			}
			this.addToHeaderScript(javascriptHeader + javascriptBody + javascriptFooter);
		}
        
        Iterator iter = results.iterator();
        while(iter.hasNext()) {
            VolumeEntry ve = (VolumeEntry) iter.next();
            QueryResult qr = new QueryResult(QueryResult.UNIQUE_RESULT, ve);
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

