/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  -----------------------------------------------
 *
 */
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports

import com.lutris.appserver.server.httpPresentation.HttpPresentation;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;

//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// Standard imports
import java.io.IOException;
import java.io.*;
import java.util.*;
import java.util.Date;
import java.text.DateFormat;

// Imported TraX classes
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.stream.StreamSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.dom.DOMResult;

import org.xml.sax.InputSource;

// Imported DOM classes
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.Text;
import org.w3c.dom.NodeList;
import org.w3c.dom.Element;
//import org.w3c.dom.html.*;

// Imported JAVA API for XML Parsing classes
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
//for debug
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;

import fr.imag.clips.papillon.business.dictionary.*;
import fr.imag.clips.papillon.business.logs.*;
import fr.imag.clips.papillon.business.user.*;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.utility.*;
import fr.imag.clips.papillon.business.locales.Languages;

//import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.presentation.xhtmlmotamot.orig.*;
import fr.imag.clips.papillon.CurrentRequestContext;

/**
 * Description of the Class
 *
 * @author serasset
 * @created December 8, 2004
 */
public class Motamot extends PapillonBasePO {

    private MotamotContentXHTML content;

    /**
     * Description of the Field
     */
    protected final static String TEXTE = "texte";
    /**
     * Description of the Field
     */
    protected final static String XML = "xml";
    /**
     * Description of the Field
     */
    protected final static String HANDLE = "handle";
    /**
     * Description of the Field
     */
    protected final static String XSLID = "xslid";
    /**
     * Description of the Field
     */
    protected final static String ANY_RESOURCE = "*ANY*";
    /**
     * Description of the Field
     */
    public final static String ALL_TARGETS = "*ALL*";
    /**
     * Description of the Field
     */
    protected final static String VOLUME_PARAMETER = "VOLUME";

    /**
     * Description of the Field
     */
    protected final static String ACTION_PARAMETER = "action";
    /**
     * Description of the Field
     */
    protected final static String LOOKUP_PARAMETER = "lookup";
    /**
     * Description of the Field
     */
    protected final static String PartialMatch_PARAMETER = "PartialMatch";
    /**
     * Description of the Field
     */
    //protected final static String HEADWORD_PARAMETER = "HEADWORD";
    protected final static String HEADWORD_PARAMETER = "FACETVALUE.0";
    /**
     * Description of the Field
     */
    protected final static String RESOURCES_PARAMETER = "RESOURCES";
    /**
     * Description of the Field
     */
    protected final static String TARGETS_PARAMETER = "TARGETS";
    /**
     * Description of the Field
     */
    //protected final static String SOURCE_PARAMETER = "SOURCE";
    protected final static String SOURCE_PARAMETER = "SOURCE.0";

    /**
     * Description of the Field
     */
    protected final static String OFFSET_PARAMETER = "OFFSET";

    /**
     * Description of the Field
     */
    protected final static String ContributionsURL = "AdminContributions.po";
    /**
     * Description of the Field
     */
    protected final static String ContributionsVolumeParameter = "VOLUME";
			
	protected final static String SrcTrg_PARAMETER = "SrcTrg";
	protected final static String SrcTrgAll_PARAMETER = "*all*";

    /*
     *  Parameters used for Sherlock plugin answer
     */
    /**
     * Description of the Field
     */
    protected final static String FORMNAME = "FORMNAME";
    /**
     * Description of the Field
     */
    protected final static String SHERLOCK_FORMNAME = "sherlock";

    /**
     * Description of the Field
     */
    protected String languagesScript = "";

    /**
     * Description of the Field
     */
    protected int XslSheetsNumber = 1;

    /**
     * Description of the Field
     */
    protected Languages Languages;
    /**
     * Description of the Field
     */
    protected String[] allResources;

    protected String[] resources;

    protected String headword;

    protected String partialMatchString;

    private static final String SEARCH_TYPE_PARAMETER = "search_type";
    private static final String EXACT_MATCH = "exact_match";
    private static final String FUZZY_MATCH = "fuzzy_match";

    /**
     *  Description of the Field
     *
     * @return Description of the Return Value
     */
    // protected ConsultXHTML content;


    /**
     * Description of the Method
     *
     * @return Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return false;
    }


    /**
     * Description of the Method
     *
     * @return Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        return true;
    }


    /**
     * Gets the currentSection attribute of the Home object
     *
     * @return The currentSection value
     */
    protected int getCurrentSection() {
        return NO_SECTION;
    }


    /**
     * Gets the content attribute of the Home object
     *
     * @return The content
     *         value
     * @throws HttpPresentationException    Description
     *                                      of the Exception
     * @throws IOException                  Description
     *                                      of the Exception
     * @throws TransformerConfigurationException
     *                                      Description
     *                                      of the Exception
     * @throws org.xml.sax.SAXException     Description
     *                                      of the Exception
     * @throws javax.xml.parsers.ParserConfigurationException
     *                                      Description
     *                                      of the Exception
     * @throws java.io.IOException          Description
     *                                      of the Exception
     * @throws javax.xml.transform.TransformerException
     *                                      Description
     *                                      of the Exception
     * @throws ClassNotFoundException       Description
     *                                      of the Exception
     * @throws PapillonBusinessException    Description
     *                                      of the Exception
     * @throws UnsupportedEncodingException Description
     *                                      of the Exception
     */
    public Node getContent() throws HttpPresentationException, TransformerConfigurationException,
                                    org.xml.sax.SAXException, javax.xml.parsers.ParserConfigurationException,
                                    java.io.IOException, javax.xml.transform.TransformerException,
                                    ClassNotFoundException, PapillonBusinessException, UnsupportedEncodingException 
		{

										
		////// Handle action events (edit, duplicate, delete, undete)
        // Retrieve parameters
        String action = myGetParameter(EditEntryInitFactory.ACTION_PARAMETER);
        String volumeName = myGetParameter(EditEntryInitFactory.VOLUME_PARAMETER);
        String entryHandle = myGetParameter(EditEntryInitFactory.HANDLE_PARAMETER);
		String searchKind = myGetParameter(SEARCH_TYPE_PARAMETER);
		String sourceLanguage = myGetParameter(SOURCE_PARAMETER);
		String targetLanguages[] = myGetParameterValues(TARGETS_PARAMETER);
		String languagePair = myGetParameter(SrcTrg_PARAMETER);		
		String targetLanguage = null;
										
		if (languagePair != null && !languagePair.equals("")) {
			if (languagePair.equals(SrcTrgAll_PARAMETER)) {
				this.getSessionData().setPreference("EditEntry.po","sourceLanguage","");
				this.getSessionData().setPreference("EditEntry.po","targetLanguage","");
			}
			else {
			if (sourceLanguage==null || sourceLanguage.equals("")) {
				sourceLanguage = languagePair.substring(0,3);
			}
			targetLanguage = languagePair.substring(3,6);
			if (sourceLanguage == targetLanguage) {
				targetLanguage = languagePair.substring(0,2);
			}
			this.getSessionData().setPreference("EditEntry.po","sourceLanguage",sourceLanguage);
			this.getSessionData().setPreference("EditEntry.po","targetLanguage",targetLanguage);
			}
		}
		else {
			String sourcePref =  this.getSessionData().getPreference("EditEntry.po","sourceLanguage");
			String targetPref =  this.getSessionData().getPreference("EditEntry.po","targetLanguage");
			if (targetPref !=null && targetPref.equals(sourceLanguage)) {
				this.getSessionData().setPreference("EditEntry.po","sourceLanguage",targetPref);
				this.getSessionData().setPreference("EditEntry.po","targetLanguage",sourcePref);
			}
		}
										
        if (null == searchKind || searchKind.equals("")) {
            searchKind = "exact_match";
        }

        //
        if (action != null && !action.equals("") && volumeName != null && !volumeName.equals(
                "") && entryHandle != null && !entryHandle.equals("")) {

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

        ////// Create Home page
		String page = "Motamot";
		if (languagePair != null && !languagePair.equals("") && !languagePair.equals(SrcTrgAll_PARAMETER)) {
			page = languagePair;
		}
        content = (MotamotContentXHTML) MultilingualXHtmlTemplateFactory.createTemplate("fr.imag.clips.papillon.presentation.xhtmlmotamot",page + "ContentXHTML",
                this.getComms(), this.getSessionData());
        //// Display query result, if action (QueryMenu search)
        if ((action != null) && !action.equals("")) {
		/*
		String source = myGetParameter(AdvancedQueryFormXHTML.NAME_SOURCE + ".0");
		
		Collection volumes = getVolumesInCacheBySourceLanguage(String source);
		
		 Iterator iterator = volumes.iterator();
		 Vector resultEntries
		while ( iterator.hasNext() ) {
			Volume theVolume = (Volume) iterator.next();
		}
		 
		
	public static Vector getVolumeEntriesVector(Dictionary dict, Volume volume, Vector Keys1, Vector Keys2, String any, int offset, int limit) throws PapillonBusinessException {

*/
            //  Retrieve parameters like AdvancedQueryForm object
            AdvancedQueryForm qf = new AdvancedQueryForm(this.getComms(), this.getSessionData(), true, false);

            // Create query request
            QueryRequest queryReq = qf.getQueryRequest();

            // Display query result if query request have criteria
            if (!queryReq.isEmpty()) {
  				
                if (searchKind.equals(EXACT_MATCH)) {
                    //// CLASSIC SEARCH
                    // Perform the request
                    Collection qrset = queryReq.findLexieAndTranslation(this.getUser());
                    // Display classic search result
                    XHTMLElement queryResultForm = content.getElementQueryResultForm();
                    Node viewQueryResultNode = ViewQueryResult.createNodeResult(this.getComms(), this.getSessionData(),
                            this.getUrl(), this.getUser(), qrset, qf.getQueryParameter(),
                            (this.getUser() != null && !this.getUser().isEmpty()));
                    queryResultForm.appendChild(content.importNode(viewQueryResultNode, true));
                    queryResultForm.removeAttribute("id");

                    Integer resSize = (Integer) CurrentRequestContext.get().get("result_size");

                    if (resSize.intValue() != 0) {
                        removeQueryFuzzyResult();
                    } else {
                        searchKind = FUZZY_MATCH;
                    }
                }
                if (searchKind.equals(FUZZY_MATCH)) {
                    //// FUZZY SEARCH

                    // Modify criteria classic -> fuzzy
                    // 1/ find headword value
                    ArrayList criteriaTree = queryReq.getCriteriaTree();
                    ArrayList criteriaOR = (ArrayList) criteriaTree.get(0);
                    QueryCriteria criteria = (QueryCriteria) criteriaOR.get(0);
                    int i = criteria.getCriteriaByColumn("value");
                    // 2/ partial match headword (or words containing in the headword)
                    criteria.addAdvancedValue("value", QueryCriteria.CASE_INSENSITIVE_CONTAINS,
                            criteria.getValue(i));   // like headword (no case sensitive)
                    // 3/ not match headword (no case sensitive)
                    criteria.add("value", QueryCriteria.CASE_INSENSITIVE_NOT_EQUAL, criteria.getValue(i));
                    // 4/ remove headword value
                    criteria.remove(i);
					
                    // Perform the request
                    Collection qrset = queryReq.findLexieAndTranslation(this.getUser());

                    // Display fuzzy search result
                    XHTMLElement queryFuzzyResultForm = content.getElementQueryFuzzyResultForm();
                    Node viewQueryFuzzyResultNode = ViewQueryResult.createNodeResult(this.getComms(),
                            this.getSessionData(), this.getUrl(), this.getUser(), qrset, qf.getQueryParameter(),
                            (this.getUser() != null && !this.getUser().isEmpty()));
                    queryFuzzyResultForm.appendChild(content.importNode(viewQueryFuzzyResultNode, true));
                    queryFuzzyResultForm.removeAttribute("id");
                    removeQueryResult();
					
                }

                //// Initializing cache values for next query
                if (null != sourceLanguage && !sourceLanguage.equals("")) {
                    this.setPreference(SOURCE_PARAMETER, sourceLanguage);
                }
                if (targetLanguage != null) {
                    this.setPreference(TARGETS_PARAMETER, targetLanguage);
               } else if ((targetLanguages != null) && (targetLanguages.length > 1)) {
                    this.setPreference(TARGETS_PARAMETER, ALL_TARGETS);
                }
                String headword = myGetParameter(HEADWORD_PARAMETER);
                if (null != headword && !headword.equals("")) {
                    this.setPreference(HEADWORD_PARAMETER, headword, false);
                }
                removeProjectDescription();

            } else {

                
                // Display project description
                removeQueryResult();
                removeQueryFuzzyResult();
            }

        } else {

            // Display project description
            removeQueryResult();
            removeQueryFuzzyResult();
        }

        //
        return content;
    }


    private void removeProjectDescription() {
        Element myElement = content.getElementProjectDescription();
        Node myParent = myElement.getParentNode();
        if (myParent != null) {
            myParent.removeChild(myElement);
        }
    }

    private void removeQueryResult() {
        Element myElement = content.getElementQueryResult();
        Node myParent = myElement.getParentNode();
        if (myParent != null) {
            myParent.removeChild(myElement);
        }
    }


    private void removeQueryFuzzyResult() {
        Element myElement = content.getElementQueryFuzzyResult();
        Node myParent = myElement.getParentNode();
        if (myParent != null) {
            myParent.removeChild(myElement);
        }
    }


}

