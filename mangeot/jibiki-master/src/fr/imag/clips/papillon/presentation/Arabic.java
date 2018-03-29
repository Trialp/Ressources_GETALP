/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *
 *  -----------------------------------------------
 *  $Id: Home.java 1088 2009-12-06 17:03:17Z mangeot $
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.24.2.3  2007/10/29 15:11:03  serasset
 *  NEW: lexalp css now defines different forms for HARMONISED/REJECTED entries
 *  NEW: added new db url/user/password configuration keys in papillon.properties file
 *  BUG158: headwords are now harmonised at edition and search time, added a "normalise headword" admin action
 *
 *  Revision 1.24.2.2  2007/07/25 15:15:44  serasset
 *  BUGFIX: process and harmo status values were inverted in advanced search
 *  BUGFIX: the source language is now added to the set of target language in simple
 *          search so that synonyms of the source entries are retreived in the result.
 *
 *  Revision 1.24.2.1  2007/07/23 14:23:50  serasset
 *  Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *   - Added XSL extensions callable during xsl transformations
 *   - Implemented new display of query results as requested by EURAC team
 *   - Modified edition interface generator to adapt it to xalan 2.7.0
 *   - Added autocompletion feature to simple search fields
 *   - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 *  Revision 1.24  2007/03/25 22:00:57  fbrunet
 *  improved avancedqueryform javascript
 *  bug correction: in ViewQueryResult class, encode url criteria in UTF-8
 *
 *  Revision 1.23  2006/08/10 22:17:13  fbrunet
 *  - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 *  - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 *  - Bug correction : +/- in advanced search
 *
 *  Revision 1.22  2006/06/19 15:27:01  fbrunet
 *  Jibiki : improvement of the search result display
 *  Lexalp : add help menu (link to wiki and bug tracker)
 *
 *  Revision 1.21  2006/06/09 10:10:43  fbrunet
 *  Add generic components (AdvancedQueryForm, QueryRequest and ViewQueryResult) in Home.java
 *
 *  Revision 1.20  2006/05/05 02:08:23  fbrunet
 *  bug correction : url utf8 transfert (in createEntryInit)
 *
 *  Revision 1.19  2006/04/24 13:43:29  fbrunet
 *  Add new class ViewQueryResult : allow to use one class to create result display in advancedSearch and EditEntryInit (like advancedQueryForm)
 *  Improve result display : view n results per page
 *
 *  Revision 1.18  2006/04/18 14:30:24  fbrunet
 *  Authorize admin to edit all entries
 *
 *  Revision 1.17  2006/04/06 15:06:39  fbrunet
 *  New class 'creationEditInit' : create new entry
 *  Modify LexALPEditEntry : only edit entry
 *
 *  Revision 1.16  2006/03/06 10:06:23  mangeot
 *  Horrible hack sur Home.java pour pouvoir utiliser tout de suite cette version sur le GDEF.
 *  Another thing: the fuzzy search takes a lot of time, so I launch it only if the normal search returns no result.
 *
 *  Revision 1.15  2006/03/01 15:12:31  mangeot
 *  Merge between maintrunk and LEXALP_1_1 branch
 *
 *  Revision 1.14  2006/02/28 18:12:15  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.13.2.6  2006/02/17 13:21:25  mangeot
 *
 *  MM: modified AdvancedQueryForm. getAllTargetLanguages, getAllSourceLanguages and getCdmElementsWithDefaultLanguage are now static in AvailableLanguages.java in order to accelerate the execution.
 *
 *  Revision 1.13.2.5  2006/01/25 15:22:23  fbrunet
 *  Improvement of QueryRequest
 *  Add new search criteria
 *  Add modified status
 *
 *  Revision 1.13.2.4  2006/01/24 13:39:49  fbrunet
 *  Modification view management
 *  Modification LexALP postprocessing
 *
 *  Revision 1.13.2.3  2005/12/02 14:10:29  fbrunet
 *  *** empty log message ***
 *
 *  Revision 1.13.2.2  2005/12/02 10:04:09  fbrunet
 *  Add Pre/Post edition processing
 *  Add index reconstruction
 *  Add new query request
 *  Add fuzzy search
 *  Add new contribution administration
 *  Add xsl transformation volume
 *
 *  Revision 1.13.2.1  2005/10/24 16:29:19  fbrunet
 *  Added fuzzy search capabilities.
 *  Added possibility to rebuild the index DB tables.
 *  Added Pre and post processors that could be defined by the user.
 *
 *  Revision 1.13  2005/08/16 08:09:58  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.12  2005/08/08 07:06:40  mangeot
 *  Added a cache for the validated and reviewed entries count in Home.java
 *
 *  Revision 1.11  2005/08/01 08:34:03  mangeot
 *  Added method getCompleteHeadword for VolumeEntry that concatenates the homograph number and the particule to the headword
 *
 *  Revision 1.10  2005/07/16 12:58:31  serasset
 *  Added limit parameter to query functions
 *  Added a parameter to Formater initializations
 *  Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 *  Revision 1.9  2005/06/15 16:48:28  mangeot
 *  Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *  Revision 1.8  2005/05/24 12:51:22  serasset
 *  Updated many aspect of the Papillon project to handle lexalp project.
 *  1. Layout is now parametrable in the application configuration file.
 *  2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 *  3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 *  4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 *  5. It is now possible to give a name to the cookie key in the app conf file
 *  6. Several bug fixes.
 *
 *  Revision 1.7.4.5  2005/06/15 10:08:06  mangeot
 *  Removed the AND/OR connector, now only AND criteria can be added for dict lookup
 *
 *  Revision 1.7.4.4  2005/06/09 11:07:45  mangeot
 *  Deleted the countEntriesCache. entries counts are not cached any more.
 *  Fixed a few bugs.
 *
 *  Revision 1.7.4.3  2005/05/27 11:53:34  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.7.4.2  2005/05/11 14:45:30  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.7.4.1  2005/04/29 14:50:25  mangeot
 *  New version with contribution infos embedded in the XML of the entries
 *
 *  Revision 1.7  2005/04/20 10:51:14  mangeot
 *  Correction de AddDirectTranslations
 *
 *  Revision 1.6  2005/04/13 14:34:38  mangeot
 *  Simplified the expert lookup. Now lookup directly the cdm element name
 *
 *  Revision 1.5  2005/04/11 12:29:59  mangeot
 *  Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 *  Revision 1.4.2.5  2005/04/01 15:16:05  mangeot
 *  Added validated contributions count on GDEF homepage
 *
 *  Revision 1.4.2.4  2005/02/25 10:22:08  mangeot
 *  Bug fixes and added the use of referrer when exiting from Reviewcontributions.po
 * 
 *  Revision 1.4.2.3  2005/02/06 22:43:49  mangeot
 *  Merged the 2 Hashtables CDM Elements and XPaths into one
 *  Added a boolean (reverse-lookup) in the volume metadata and functionalities in order to perform a reverse lookup when no direct lookup result is found
 *  Added a boolean (index) in the volume metadata for indexing the only specified CDM Elements
 *
 *  Revision 1.4.2.2  2005/01/28 19:45:55  mangeot
 *  First version that runs basically.
 *  Should compile after an ant clean.
 *  XPath loading and virtual volumes for terminological lexicons are OK.
 *  Bugs remain, needs more testings like the editor for example.
 *
 *  Revision 1.4.2.1  2005/01/27 19:29:21  mangeot
 *  Implemented the HtmlDom cache, it increases speed drastically.
 *  Still does not compile after an ant clean
 *
 *  Revision 1.4  2005/01/18 12:16:10  mangeot
 *  Implemented the SQL LIMIT and OFFSET keywords. It allows us to retrieve the entries as blocks and page them. The LIMIT is the DictionariesFactory.MaxRetrievedEntries constant.
 *  The implementation may need further tuning
 *
 *  Revision 1.3  2004/12/24 14:31:28  mangeot
 *  I merged the latest developments of Papillon5.0 with this version 5.1.
 *  Have to be tested more ...
 *
 *  Revision 1.2  2004/12/24 08:57:44  serasset
 *  Premiere version de l'interface avec fond papillon et transparence.
 *
 *  Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 *  Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 *  There are still bugs in the code.
 *
 *  Revision 1.36  2004/10/28 10:56:21  mangeot
 *  Added the list of connected users on AdminUsers.java,
 *  Added the possibility to sort in columns for some pages
 *  Added persistent preferences for the user. They are saved in the database and retrived when the user reconnects. The user is registered in the enhydra session.
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
//import com.lutris.appserveàr.server.httpPresentation.HttpPresentationResponse;

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

import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.CurrentRequestContext;

/**
 * Description of the Class
 *
 * @author serasset
 * @created December 8, 2004
 */
public class Arabic
        extends PapillonBasePO {

    private ArabicContentXHTML content;

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

    protected static int GDEF_estValidatedEntriesCount = 0;

    protected static int GDEF_estReviewedEntriesCount = 0;

    protected static java.util.Calendar myCalendar = new java.util.GregorianCalendar();

    protected static int DAY_OF_MONTH = 0;

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
                                    ClassNotFoundException, PapillonBusinessException, UnsupportedEncodingException {

        ////// Handle action events (edit, duplicate, delete, undete)
        // Retrieve parameters
        String action = myGetParameter(EditEntryInitFactory.ACTION_PARAMETER);
        String volumeName = myGetParameter(EditEntryInitFactory.VOLUME_PARAMETER);
        String entryHandle = myGetParameter(EditEntryInitFactory.HANDLE_PARAMETER);
        String searchKind = myGetParameter(SEARCH_TYPE_PARAMETER);
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
        content = (ArabicContentXHTML) MultilingualXHtmlTemplateFactory.createTemplate("ArabicContentXHTML",
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

                // Add the source language to the targets in order to get synonyms of matching source entries in the
                // result set
                String language = myGetParameter(AdvancedQueryFormXHTML.NAME_SOURCE + ".0");
                if (! queryReq.getTargets().contains(language)) {
                    queryReq.getTargets().add(language);
                }

  
                 // Add status criteria
                ArrayList listStatus = new ArrayList();

				QueryCriteria criteriaStatus = new QueryCriteria();
                criteriaStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);
                criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.CLASSIFIED_FINISHED_STATUS);
                criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.CLASSIFIED_NOT_FINISHED_STATUS);
                criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.DELETED_STATUS);
                criteriaStatus.add("lang", QueryCriteria.EQUAL, Volume.DEFAULT_LANG);
                listStatus.add(criteriaStatus);
				
				queryReq.addOrCriteriaList(listStatus);
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
                String sourceLanguage = myGetParameter(SOURCE_PARAMETER);
                if (null != sourceLanguage && !sourceLanguage.equals("")) {
                    this.setPreference(SOURCE_PARAMETER, sourceLanguage);
                }
                String[] targetsLanguage = myGetParameterValues(TARGETS_PARAMETER);
                if ((targetsLanguage != null) && (targetsLanguage.length == 1)) {
                    this.setPreference(TARGETS_PARAMETER, targetsLanguage[0]);
                } else if ((targetsLanguage != null) && (targetsLanguage.length > 1)) {
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

