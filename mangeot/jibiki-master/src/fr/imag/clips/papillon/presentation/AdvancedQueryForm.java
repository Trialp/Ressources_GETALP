/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * ¬© Gilles S√©rasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.13.2.1  2007/10/29 15:11:03  serasset
 * NEW: lexalp css now defines different forms for HARMONISED/REJECTED entries
 * NEW: added new db url/user/password configuration keys in papillon.properties file
 * BUG158: headwords are now harmonised at edition and search time, added a "normalise headword" admin action
 *
 * Revision 1.13  2007/03/25 22:00:57  fbrunet
 * improved avancedqueryform javascript
 * bug correction: in ViewQueryResult class, encode url criteria in UTF-8
 *
 * Revision 1.12  2007/02/28 09:27:07  fbrunet
 * Added ajax method to AdvancedQueryForm page
 *
 * Revision 1.11  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.10  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.9  2006/06/19 15:27:00  fbrunet
 * Jibiki : improvement of the search result display
 * Lexalp : add help menu (link to wiki and bug tracker)
 *
 * Revision 1.8  2006/06/09 10:10:43  fbrunet
 * Add generic components (AdvancedQueryForm, QueryRequest and ViewQueryResult) in Home.java
 *
 * Revision 1.7  2006/06/01 22:05:05  fbrunet
 * New interface, quick search, new contribution management (the first edition not create new contribution. New contribution is created after add, remove element, update, save, etc. in the interface window)
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
 * Revision 1.1.4.8  2006/02/24 13:59:55  fbrunet
 * Postprocessing during update and save
 *
 * Revision 1.1.4.7  2006/02/17 15:16:42  mangeot
 * Do not display the list of all XSL on the search form any more. Displays only a list of XSL descriptions
 *
 * Revision 1.1.4.6  2006/02/17 13:21:25  mangeot
 *
 * MM: modified AdvancedQueryForm. getAllTargetLanguages, getAllSourceLanguages and getCdmElementsWithDefaultLanguage are now static in AvailableLanguages.java in order to accelerate the execution.
 *
 * Revision 1.1.4.5  2006/02/17 10:41:48  fbrunet
 * Change QueryCriteria parameters
 * Add new windows when editing an entry
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
 * Revision 1.1  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 *-----------------------------------------------
 */

package fr.imag.clips.papillon.presentation;

/* standards imports */
import java.util.HashSet;
// Standard imports
import java.util.Vector;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Hashtable;
import java.util.Collection;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Enumeration;

import java.util.Date;
import java.text.DateFormat;

import java.util.regex.Pattern;
import java.util.regex.Matcher;

// Enhydra imports
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import javax.servlet.http.HttpServletRequest;
import org.enhydra.xml.xhtml.dom.*;

// Imported DOM classes
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.Text;
import org.w3c.dom.NodeList;
import org.w3c.dom.Element;

// Papillon imports
import fr.imag.clips.papillon.presentation.xhtml.orig.AdvancedQueryFormXHTML;
import fr.imag.clips.papillon.business.dictionary.AvailableLanguages;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.IQuery;
import fr.imag.clips.papillon.business.dictionary.QueryParameter;
import fr.imag.clips.papillon.business.dictionary.QueryRequest;
import fr.imag.clips.papillon.business.dictionary.QueryCriteria;
import fr.imag.clips.papillon.business.dictionary.VolumeCache;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.utility.StringNormalizer;
import fr.imag.clips.papillon.business.locales.Languages;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;


/**
* A QueryParameter is a business object passed when querying dictionaries.
 */
public class AdvancedQueryForm {
    
    protected Node queryForm;
    
    public final static String ALL_TARGETS = "*ALL*";
	/*public final static String OFFSET_PARAMETER = AdvancedQueryFormXHTML.NAME_OFFSET*/
	public final static String OFFSET_PARAMETER = "OFFSET";
    
    protected static final String actionPatternString = "^(\\+|\\-)([^\\.]+)\\.(\\d+)$";
    protected static Pattern actionPattern = Pattern.compile(actionPatternString);
    protected static Matcher actionMatcher = actionPattern.matcher("");
    protected static java.util.regex.Pattern quotePattern = java.util.regex.Pattern.compile("'");
    protected static Matcher quoteMatcher = quotePattern.matcher("");
   
    QueryParameter qparams;     //FIXME: replace by qrequest
    QueryRequest qrequest;
    ArrayList criteriaList;
    
    boolean IsOpenRequest = false;
    
    String action;
    
    public Collection getRequestedDictionaries(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        //
        String[] d = AbstractPO.myGetParameterValues(req, AdvancedQueryFormXHTML.NAME_DICTIONARIES);
        
        //
        if (null == d) {
            
            // return all Dictionaries
            return null;
        
        } else {
            ArrayList dictionaries = new ArrayList();
            
            //
            for (int i = 0; i < d.length; i++) {
                dictionaries.add(DictionariesFactory.getDictionaryByName(d[i]));
            }
            
            //
            return dictionaries;
        }
    }
	
	public Collection getDisplayableDictionaries(HttpServletRequest req) 
	throws java.io.UnsupportedEncodingException,
	com.lutris.appserver.server.httpPresentation.HttpPresentationException {
		Collection dictionaries  = getRequestedDictionaries(req);
		if (dictionaries == null) {
			// removed in order to avoid selecting all dicts by default. It causes problems when many discts are loaded
			//dictionaries = DictionariesFactory.getDictionariesArray();
		}
		return dictionaries;
	}
	
	
    
    public Collection getRequestedVolumes(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        //
        String[] d = AbstractPO.myGetParameterValues(req, AdvancedQueryFormXHTML.NAME_VOLUMES);
        
        //
        if (null == d) {
            
            // return all Dictionaries
           // return VolumesFactory.getVolumesArray();
			return null;
        
        } else {
            ArrayList volumes = new ArrayList();
            
            //
            for (int i = 0; i < d.length; i++) {
                volumes.add(VolumesFactory.getVolumeByName(d[i]));
            }
            
            //
            return volumes;
        }
    }
	
	public Collection getRequestedDictionariesAndVolumes(HttpServletRequest req) 
	throws java.io.UnsupportedEncodingException,
	com.lutris.appserver.server.httpPresentation.HttpPresentationException {
		Collection volumes  = getRequestedVolumes(req);
			if (volumes == null) {
				Collection dictionaries = getRequestedDictionaries(req);
				if (dictionaries !=null && dictionaries.size()>0) {
					volumes = new ArrayList();
					Iterator iter = dictionaries.iterator();
					while(iter.hasNext()) {
						Dictionary myDict = (Dictionary) iter.next();
						volumes.addAll(VolumesFactory.getVolumesArray(myDict.getName()));
					}
				}
				else {
					volumes = VolumesFactory.getVolumesArray();
				}
			}
			return volumes;
		}
    
    public String getRequestedXsl(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        //
        String xsl = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_XSL);
        
        //
        if (xsl !=null && !xsl.equals("") && !xsl.equalsIgnoreCase("default")) {
            
            //
            return xsl;
            
        } else {
            
            // Default xsl 
            return new String("");
        } 
        
        //return AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_XSL);
    }

    public String getRequestedNumberOfResultsPerPageString(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        String nbr = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_LIMIT);
        if (null == nbr || nbr.equals("")) {
            nbr = "10";         // all (0)
        }
        //System.out.println("getRequestedNumberOfResultsPerPageString  limit = " + nbr);
        return nbr;
    }

    
    public int getRequestedNumberOfResultsPerPage(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        String nbr = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_LIMIT);
        if (null == nbr || nbr.equals("")) {
            nbr = "10";         // all (0)
        }
        //System.out.println("getRequestedNumberOfResultsPerPage  limit = " + nbr);
        return Integer.valueOf(nbr).intValue();
    }

    public String getRequestedOffsetString(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        String ofs = AbstractPO.myGetParameter(req, OFFSET_PARAMETER);
        return (null == ofs || ofs.equals("")) ? "0" : ofs;
    }
    
    public int getRequestedOffset(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        String ofs = AbstractPO.myGetParameter(req, OFFSET_PARAMETER);
        return (null == ofs || ofs.equals("")) ? 0 : Integer.valueOf(ofs).intValue();
    }
    
    public Collection getRequestedTargetLanguages(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        Collection result = new ArrayList();
        String[] t = AbstractPO.myGetParameterValues(req, AdvancedQueryFormXHTML.NAME_TARGETS);
        
        if (t != null) {
            result = Arrays.asList(t);
			if (result.contains(ALL_TARGETS)) {
				result = AvailableLanguages.getTargetLanguagesArray();
			}
		}
         //
        return result;
    }
    
    public List getRequestedCriteria(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        String nbc = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_CRITERIA_NB);
        int nbCriteria = (null == nbc || nbc.equals("")) ? 1 : Integer.valueOf(nbc).intValue();
        
        Vector criteria = new Vector();
        for(int i=0; i < nbCriteria; i++) {
            String[] key = new String[4];
            key[0] = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_FACET + "." + Integer.toString(i));
            key[1] = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_SOURCE + "." + Integer.toString(i));
            String facetValue = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_FACETVALUE + "." + Integer.toString(i));
            key[2] = (null == facetValue) ? "" : facetValue;
            String comparator = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_OPERATOR + "." + Integer.toString(i));
            //int comparisonOperator = (comparator != null) ? Integer.valueOf(comparator).intValue() : 0;
            //key[3] = IQuery.QueryBuilderStrategy[comparisonOperator];
            key[3] = comparator;
            
            // Add the criterion to the set
            if (null != key[0]) {
                criteria.add(key);
            }
        }
        return criteria;
    }
    
    public List getRequestedCriteria2(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        String nbc = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_CRITERIA_NB);
        int nbCriteria = (null == nbc || nbc.equals("")) ? 1 : Integer.valueOf(nbc).intValue();
        
        criteriaList = new ArrayList();
        for(int i=0; i < nbCriteria; i++) {
            String value = StringNormalizer.normalize(AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_FACETVALUE + "." + Integer.toString(i)));
            if (value !=null && !value.equals("")) {
                java.util.regex.Matcher quoteValueMatcher = quotePattern.matcher(value);
                value = quoteValueMatcher.replaceAll("''");
            }
            QueryCriteria criteria = new QueryCriteria();
            
            //
            if ( (null != value) && (!value.equals("")) ) {
                
                //
                String key = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_FACET + "." + Integer.toString(i));
                String language = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_SOURCE + "." + Integer.toString(i));
                String strategy = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_OPERATOR + "." + Integer.toString(i));
                
                //
				if (key != null && !key.equals("")) {

                    
                    /* clés multiples */
                    if (key.indexOf('|')>0) {
                        String keyClause = "(";
                        String[] keynames = key.split("\\|");
                        for (int j=0;j<keynames.length;j++) {
                            java.util.regex.Matcher quoteMatcher = quotePattern.matcher(keynames[j]);
                            String newValue = quoteMatcher.replaceAll("''");
                            
                            keyClause += "key = '" + newValue + "'";
                            if (newValue.equals(Volume.CDM_contributionId) ||
 //                               newValue.equals(VolumeEntry.CDM_originalContributionId) ||
                                newValue.equals(Volume.CDM_contributionStatus)
                                ) {
                                IsOpenRequest = true;
                            }
                            if (j<keynames.length-1) {
                                keyClause += " OR ";
                            }
                        }
                        keyClause += ")";
                        criteria.addClause(keyClause);
                    }
                    else {
                        if (key.equals(Volume.CDM_contributionId) ||
                            //                               newValue.equals(VolumeEntry.CDM_originalContributionId) ||
                            key.equals(Volume.CDM_contributionStatus)
                            ) {
                            IsOpenRequest = true;
                        }
                        criteria.add("key", "=", key);
                    }
				}
				criteria.add("value", strategy, value);
				if (language==null) {
					if (Volume.isDefaultLangCDMElement(key)) {
						language=Volume.DEFAULT_LANG;
					}
				}
                if ( (language != null) && (!language.equals("")) && (!language.equals("All")) ) {
					criteria.add("lang", "=", language);
				}
                
                //
				if (key != null && !key.equals("") && value != null) {
					criteriaList.add(criteria);
				}
            }
        }
        return criteriaList;
    }
    
    //////////该看这了 ZHANG YING
    public String getRequestedAction(HttpServletRequest req) 
        throws java.io.UnsupportedEncodingException,
        com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        return AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_ACTION);
    }
        
    public AdvancedQueryForm(HttpPresentationComms comms, PapillonSessionData sessionData, boolean showTargets, boolean showQuickSearch) throws PapillonPresentationException {
        try {
            
            // Get all the parameters
            qrequest = new QueryRequest(getRequestedDictionariesAndVolumes(comms.request.getHttpServletRequest()));
            Collection criteriaList = getRequestedCriteria2(comms.request.getHttpServletRequest());
            for (Iterator iter = criteriaList.iterator(); iter.hasNext();) {
                qrequest.addCriteria((QueryCriteria)iter.next());
            }
            qrequest.setOpenRequest(IsOpenRequest);
            qrequest.setLimit(getRequestedNumberOfResultsPerPageString(comms.request.getHttpServletRequest()));
            qrequest.setOffset(getRequestedOffsetString(comms.request.getHttpServletRequest()));
            qrequest.setXsl(getRequestedXsl(comms.request.getHttpServletRequest()));
            qrequest.setTargets(getRequestedTargetLanguages(comms.request.getHttpServletRequest()));
            
            //FIXME: replace by qrequest
            // Get all the parameters
            qparams = new QueryParameter();
            qparams.setDictionaries(getDisplayableDictionaries(comms.request.getHttpServletRequest()));
            qparams.setCriteria(getRequestedCriteria(comms.request.getHttpServletRequest()));
            qparams.setLimit(getRequestedNumberOfResultsPerPage(comms.request.getHttpServletRequest()));
            qparams.setOffset(getRequestedOffset(comms.request.getHttpServletRequest()));
            qparams.setXsl(getRequestedXsl(comms.request.getHttpServletRequest()));
            qparams.setTargets(getRequestedTargetLanguages(comms.request.getHttpServletRequest()));
             			
            // Did the user ask the addition or removal of a node ? 
            action = getRequestedAction(comms.request.getHttpServletRequest());
            if (actionOnFormRequested()) {
                                
                // Perform the action
                actionMatcher.reset(action);
                if (actionMatcher.find()) {
                          
                    //
                    String action = actionMatcher.group(1);
                    String tag = actionMatcher.group(2);
                    int num = Integer.valueOf(actionMatcher.group(3)).intValue();
                    
                    //
                    if (action.equals("-")) {
                                                
                        //
                        qparams.getCriteria().remove(num);
                    
                    //
                    } else {
                        
                        String[] key = new String[4];
                        key[0] = Volume.CDM_headword;
                        key[1] = null;
                        key[2] = "";
                        //key[3] = IQuery.QueryBuilderStrategy[0];
                        key[3] = QueryCriteria.EQUAL;
                        qparams.getCriteria().add(key);
                    }
                } 
            }
            
            // Apply those parameters to the form
            buildFormFromParameters(comms, sessionData, qparams, showTargets, showQuickSearch);
            
        } catch (Exception e) {
            throw new PapillonPresentationException("Exception in queryForm", e);
        }
        
    }
    
    /** Called during form initialization.
        If a user wants to modify the parameters and rebuild the form accordingly,
        s/he will have to call this method again.
        */
    public void buildFormFromParameters(HttpPresentationComms comms, PapillonSessionData sessionData, QueryParameter qp, boolean showTargets, boolean showQuickSearch) 
        throws com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {
        String requestURI = comms.request.getPresentationObjectRelativePath();
		// Create the advanced query form and fill it with current requested values
        AdvancedQueryFormXHTML queryDoc = (AdvancedQueryFormXHTML) 
        MultilingualXHtmlTemplateFactory.createTemplate("AdvancedQueryFormXHTML", comms, sessionData);
        
        // Populate form with all known dictionaries
        XHTMLOptionElement dictOption = queryDoc.getElementDictionaryTmpl();
        dictOption.removeAttribute("id");
        Collection knownDictionaries = DictionariesFactory.getDictionariesArray();
        for (Iterator iter = knownDictionaries.iterator(); iter.hasNext();) {
            Dictionary dict = (Dictionary)iter.next();
            String dicName = dict.getName();
            dictOption.setValue(dicName);
            dictOption.setLabel(dicName);
            Text txt = (Text) dictOption.getFirstChild();
            txt.setData(dicName);
            dictOption.getParentNode().appendChild(dictOption.cloneNode(true));
        }
        dictOption.getParentNode().removeChild(dictOption);
        
		
		//populate form with non standard indexed CDM elements in the facet list 
		// disabled because it cannot be controlled when displayed. options cannot be disabled and option names can be unclear
		XHTMLSelectElement facet = queryDoc.getElementFacet();
/*		XHTMLOptionElement templateOption = (XHTMLOptionElement) facet.getLastChild();
        Iterator iterCdmElements = fr.imag.clips.papillon.business.dictionary.VolumeCache.getNonStandardCDMElements().iterator();
        while ( iterCdmElements.hasNext() ) {
            String CdmElement = (String)iterCdmElements.next();
			templateOption = (XHTMLOptionElement) templateOption.cloneNode(true);
			Text optionTextTemplate = (Text)templateOption.getFirstChild();             
            templateOption.setValue(CdmElement);
            templateOption.setLabel(CdmElement);
			optionTextTemplate.setData(CdmElement);
			facet.appendChild(templateOption);
        }
		*/
		
        // Populate form with all know target languages
        XHTMLOptionElement targetOption = queryDoc.getElementTargetTmpl();
        targetOption.removeAttribute("id");
        
        //
        for (Iterator iter = AvailableLanguages.getTargetLanguagesArray().iterator(); iter.hasNext();) {
            String lg = (String)iter.next();
            
            //
            targetOption.setValue(lg);
            targetOption.setLabel(Languages.localizeLabel(sessionData.getUserPreferredLanguage(),lg));
            Text txt = (Text) targetOption.getFirstChild();
            txt.setData(Languages.localizeName(sessionData.getUserPreferredLanguage(),lg));
            targetOption.getParentNode().appendChild(targetOption.cloneNode(true));
        }
        targetOption.getParentNode().removeChild(targetOption);
        
        // Populate form with available xsls
        Collection xslSheetList = XslSheetFactory.getExternalXslSheetsArray();
        XHTMLOptionElement xslOption = queryDoc.getElementXslTmpl();
        xslOption.removeAttribute("id");
		
        for (Iterator iter = xslSheetList.iterator(); iter.hasNext();) {
            XslSheet xsl = (XslSheet) iter.next();
            xslOption.setValue(xsl.getName());
            xslOption.setLabel(xsl.getName());
            Text txt = (Text) xslOption.getFirstChild();
            txt.setData(xsl.getName());
            xslOption.getParentNode().appendChild(xslOption.cloneNode(true));
        }
        xslOption.getParentNode().removeChild(xslOption);
        
        // put parameters back into form
        //queryDoc.getElementOffset().setValue(Integer.toString(qp.getOffset()));
        
        AbstractPO.setSelected(queryDoc.getElementLIMITSelect(), Integer.toString(qp.getLimit()));
        		
			for (Iterator iter = qp.getDictionaries().iterator(); iter.hasNext();) {
				Dictionary dict = (Dictionary)iter.next();
				if (dict !=null && !dict.isEmpty()) {
					AbstractPO.setSelected(queryDoc.getElementDictionaries(), dict.getName());
				}
			}
        //
        for (Iterator iter = qp.getTargets().iterator(); iter.hasNext();) {
            AbstractPO.setSelected(queryDoc.getElementTargets(), (String)iter.next());
        }
        
        AbstractPO.setSelected(queryDoc.getElementXsl(),qp.getXsl());
        
        // Then the criteria...
        XHTMLInputElement nbcrit = queryDoc.getElementNumberOfCriteria();
        Element criterion = queryDoc.getElementCriterion();
        XHTMLImageElement minus = queryDoc.getElementMinus();
        XHTMLImageElement plus = queryDoc.getElementPlus();
        XHTMLSelectElement oper = queryDoc.getElementOperator();
        XHTMLInputElement valuefield = queryDoc.getElementValueField();
        //Element sourceLangSelectArea = queryDoc.getElementSourceLanguageSelection();
        //XHTMLLabelElement sourceLangLabel = queryDoc.getElementSourceLangLabel();
        XHTMLSelectElement sourceLang = queryDoc.getElementSourceLang();
        XHTMLOptionElement sourceOption = queryDoc.getElementSourceOptionTemplate();
        Element targetLanguagesSelectionElement = queryDoc.getElementTargetLanguagesSelection();
        Element QuickLinkElement = queryDoc.getElementQuickLink();
        
        nbcrit.removeAttribute("id");
        criterion.removeAttribute("id");
        minus.removeAttribute("id");
        plus.removeAttribute("id");
        facet.removeAttribute("id");
        //oper.removeAttribute("id");
        //valuefield.removeAttribute("id");
        //sourceLangSelectArea.removeAttribute("id");
        //sourceLangLabel.removeAttribute("id");
        sourceLang.removeAttribute("id");
        sourceOption.removeAttribute("id");
        targetLanguagesSelectionElement.removeAttribute("id");
        QuickLinkElement.removeAttribute("id");
        
        //
        if (!showTargets) targetLanguagesSelectionElement.setAttribute("class","hidden");
        
        if (!showQuickSearch) {
            QuickLinkElement.setAttribute("class","hidden");
        
        } else {
            // Quick links
            XHTMLElement MyFinishedContributionsElement = queryDoc.getElementMyFinishedContributions();
            String href = "EditEntryInit.po?CRITERIA_NB=2&FACET.0=cdm-modification-author&OPERATOR.0=2&FACETVALUE.0=" + sessionData.getUser().getLogin() + "&SOURCE.0=All&FACET.1=cdm-contribution-status&OPERATOR.1=2&FACETVALUE.1=" + VolumeEntry.FINISHED_STATUS + "&SOURCE.1=All&LIMIT=10&lookup=Go";
            MyFinishedContributionsElement.setAttribute("href", href);
            
            XHTMLElement MyNotFinishedContributionsElement = queryDoc.getElementMyNotFinishedContributions();
            href = "EditEntryInit.po?&CRITERIA_NB=2&FACET.0=cdm-modification-author&OPERATOR.0=2&FACETVALUE.0=" + sessionData.getUser().getLogin() + "&SOURCE.0=All&FACET.1=cdm-contribution-status&OPERATOR.1=2&FACETVALUE.1=" + VolumeEntry.NOT_FINISHED_STATUS + "&SOURCE.1=All&LIMIT=10&lookup=Go";
            MyNotFinishedContributionsElement.setAttribute("href", href);
        }
        
        // add all source languages in the lang selector
        for (Iterator iter = AvailableLanguages.getSourceLanguagesArray().iterator(); iter.hasNext();) {
            String lg = (String)iter.next();
            
            //
            sourceOption.setValue(lg);
            sourceOption.setLabel(Languages.localizeLabel(sessionData.getUserPreferredLanguage(),lg));
            Text txt = (Text) sourceOption.getFirstChild();
            txt.setData(Languages.localizeName(sessionData.getUserPreferredLanguage(),lg));
            sourceOption.getParentNode().appendChild(sourceOption.cloneNode(true));
        }
        sourceOption.getParentNode().removeChild(sourceOption);
        
        if (qp.getCriteria().size() == 0) {
            nbcrit.setValue("1");
            AbstractPO.setSelected(facet,Volume.CDM_headword);
            AbstractPO.setSelected(oper,"1");
            facet.setAttribute("name", AdvancedQueryFormXHTML.NAME_FACET + ".0");
            oper.setAttribute("name", AdvancedQueryFormXHTML.NAME_OPERATOR + ".0");
            oper.setAttribute("id", "Operator.0");
           valuefield.setAttribute("name", AdvancedQueryFormXHTML.NAME_FACETVALUE + ".0");
            valuefield.setAttribute("id", "ValueField.0");
            sourceLang.setAttribute("name", AdvancedQueryFormXHTML.NAME_SOURCE + ".0");
            minus.getParentNode().removeChild(minus);
            plus.setAttribute("id", "plus.0");
            sourceLang.setAttribute("id", "SourceLang.0");
			
            String prefSrcLang = sessionData.getPreference(requestURI, sourceLang.getName());
            if (prefSrcLang == null || prefSrcLang.equals("")) {
                prefSrcLang = sessionData.getUserPreferredLanguage();
                sessionData.setPreference(requestURI, sourceLang.getName(), prefSrcLang);
            }
			AbstractPO.setSelected(sourceLang,prefSrcLang);
            //sourceLangLabel.setAttribute("for", "SourceLang.0");
        } else {
            nbcrit.setValue(Integer.toString(qp.getCriteria().size()));
            Iterator iter = qp.getCriteria().iterator();
            int i=0;
            while(iter.hasNext()) {
                String[] key = (String[]) iter.next();
                AbstractPO.setSelected(facet,key[0]);
                //int operValue = index(IQuery.QueryBuilderStrategy, key[3]);
                //AbstractPO.setSelected(oper,Integer.toString(operValue));
                AbstractPO.setSelected(oper, key[3]);
                valuefield.setValue(key[2]);
				String istr = Integer.toString(i);
               if (null != key[1] && !key[1].equals("")) {
                    AbstractPO.setSelected(sourceLang,key[1]);
					if (i==0)  {
						sessionData.setPreference(requestURI, sourceLang.getName() + "." + istr, key[1]);
					}
				}
                facet.setAttribute("name", AdvancedQueryFormXHTML.NAME_FACET + "." + istr);
                oper.setAttribute("name", AdvancedQueryFormXHTML.NAME_OPERATOR + "." + istr);
                oper.setAttribute("id", "Operator." + istr);
               valuefield.setAttribute("name", AdvancedQueryFormXHTML.NAME_FACETVALUE + "." + istr);
                valuefield.setAttribute("id", "ValueField." + istr);
                sourceLang.setAttribute("name", AdvancedQueryFormXHTML.NAME_SOURCE + "." + istr);
                minus.setAttribute("id", "minus." + istr);
                plus.setAttribute("id", "plus." + istr);
                sourceLang.setAttribute("id", "SourceLang." + istr);
                //sourceLangLabel.setAttribute("for", "SourceLang." + istr);
                
                // Specify what cdm-element appears in advanced lookup and if it needs language selector !
                // see below "create table to show/hide parameters"
				if (fr.imag.clips.papillon.business.dictionary.AvailableLanguages.getCdmElementsWithDefaultLanguage().contains(key[0])) {
					((Element)sourceLang.getParentNode()).setAttribute("style", "visibility: hidden;");
				} else {
					((Element)sourceLang.getParentNode()).setAttribute("style", "visibility: visible;");
				}
                
                //
                Node newCriterion = criterion.cloneNode(true);
                
                // Reinit the template node
                minus.removeAttribute("id");
                plus.removeAttribute("id");
                AbstractPO.deselectAll(facet);
                AbstractPO.deselectAll(sourceLang);
                AbstractPO.deselectAll(oper);
                criterion.getParentNode().insertBefore(newCriterion, criterion);
                
                // Remove plus in all but the last element
                if (iter.hasNext()) {
                    Element newPlus = newCriterion.getOwnerDocument().getElementById("plus." + istr);
                    newPlus.getParentNode().removeChild(newPlus);
                } else if (0 == i) {
                    // Remove minus if there is only one element
                    Element newMinus = newCriterion.getOwnerDocument().getElementById("minus." + istr);
                    newMinus.getParentNode().removeChild(newMinus);
                }
                i++;
            }
            criterion.getParentNode().removeChild(criterion);
        }
        
        // Add by Francis : create table to show/hide parameters
        // Specify what cdm-element appears in advanced lookup and if it needs language selector ! 
        XHTMLTableRowElement rowShowHideTemplate = queryDoc.getElementRowShowHideTemplate();
        rowShowHideTemplate.removeAttribute("id");
        
        // Build the data structure in the xhtml
        Iterator iter = fr.imag.clips.papillon.business.dictionary.AvailableLanguages.getCdmElementsWithDefaultLanguage().iterator();
        while ( iter.hasNext() ) {
            String CdmElement = (String)iter.next();
            //
            NodeList list = (NodeList) rowShowHideTemplate.getChildNodes();
            Text parametersToHide = (Text) list.item(0).getFirstChild();
            parametersToHide.setData(CdmElement);
            rowShowHideTemplate.getParentNode().appendChild(rowShowHideTemplate.cloneNode(true));
        }
        rowShowHideTemplate.getParentNode().removeChild(rowShowHideTemplate);
        		
        queryForm = queryDoc.getElementAdvancedSearchForm();
	}
        
    protected static int index(String[] array, String key) {
        int i = 0;
        while( i < array.length && !array[i].equals(key)) i++;
        return i;
    }
    
    // utiliser plutôt & ou &amp; ??? &amp; n'est pas bien décodé par enhydra
    // HttpServletRequest req.getParameterValues(name)

    public static String getEncodedUrlForParameter(QueryParameter qp)
        throws PapillonBusinessException {
        
        // View
        String url = AdvancedQueryFormXHTML.NAME_XSL + "=" + qp.getXsl();
        url += "&" + AdvancedQueryFormXHTML.NAME_LIMIT + "=" + qp.getLimit();
        url += "&" + OFFSET_PARAMETER + "=" + qp.getOffset();
        
        // Dictionnaries
        for (Iterator iter = qp.getDictionaries().iterator(); iter.hasNext();) {
            Dictionary dict = (Dictionary)iter.next();
            url += "&" + AdvancedQueryFormXHTML.NAME_DICTIONARIES  + "=" +  dict.getName();
        }
        
        // target languages
        for (Iterator iter = qp.getTargets().iterator(); iter.hasNext();) {
            url += "&" + AdvancedQueryFormXHTML.NAME_TARGETS  + "=" +  (String)iter.next();
        }
        
        // CriteriaS
        List criteria = qp.getCriteria();
        url+= "&" + AdvancedQueryFormXHTML.NAME_CRITERIA_NB + "=" + criteria.size();
        
        //
        int i = 0;
        for(Iterator iter = criteria.iterator(); iter.hasNext();) {
            String[] key = (String[])iter.next();    
            
            //
            url += "&" + AdvancedQueryFormXHTML.NAME_FACET + "." + Integer.toString(i) + "=" + key[0];
            if (null != key[1])
                url += "&" + AdvancedQueryFormXHTML.NAME_SOURCE + "." + Integer.toString(i) + "=" + key[1];
            url += "&" + AdvancedQueryFormXHTML.NAME_FACETVALUE + "." + Integer.toString(i) + "=" + AbstractPO.myUrlEncode(key[2]);
            //int operValue = index(IQuery.QueryBuilderStrategy, key[3]);
            //url += "&" + AdvancedQueryFormXHTML.NAME_OPERATOR + "." + Integer.toString(i) + "=" + Integer.toString(operValue); 
            url += "&" + AdvancedQueryFormXHTML.NAME_OPERATOR + "." + Integer.toString(i) + "=" + key[3];
            
            //
            i++;
        }
        
        //
        return url;
    }
    
    /** 
    *
    */
    public Node getQueryFormNode(String action) {
        // Create the advanced query form and fill it with current requested values
        return queryForm;
    }
    
   
    public QueryParameter getQueryParameter() {
        // Create the advanced query form and fill it with current requested values
        return qparams;
    }
    
    public QueryRequest getQueryRequest() {
        //
        return qrequest;
    }
    
    public boolean isOpenRequest() {
        //
        return IsOpenRequest;
    }
    
    public ArrayList getCriteriaList() {
        //
        return criteriaList;
    }

    
    
    public boolean actionOnFormRequested() {
        return (null != action && !action.equals(""));
    }
}
