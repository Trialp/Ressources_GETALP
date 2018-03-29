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
 *  Revision 1.1.2.1  2007/07/23 14:23:50  serasset
 *  Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *   - Added XSL extensions callable during xsl transformations
 *   - Implemented new display of query results as requested by EURAC team
 *   - Modified edition interface generator to adapt it to xalan 2.7.0
 *   - Added autocompletion feature to simple search fields
 *   - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 *  Revision 1.32  2006/08/10 22:17:13  fbrunet
 *  - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 *  - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 *  - Bug correction : +/- in advanced search
 *
 *  Revision 1.31  2006/03/10 16:23:18  mangeot
 *  Hack for targets.length==0, I am not satisfied...
 *
 *  Revision 1.30  2006/03/01 15:12:31  mangeot
 *  Merge between maintrunk and LEXALP_1_1 branch
 *
 *  Revision 1.28  2005/09/11 16:53:06  mangeot
 *  Bug fix
 *
 *  Revision 1.27  2005/09/08 15:04:25  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.26  2005/09/08 14:13:07  mangeot
 *  Bug fix in Validated contrib deletion process
 *
 *  Revision 1.25.2.2  2006/02/17 13:21:25  mangeot
 *
 *  MM: modified AdvancedQueryForm. getAllTargetLanguages, getAllSourceLanguages and getCdmElementsWithDefaultLanguage are now static in AvailableLanguages.java in order to accelerate the execution.
 *
 *  Revision 1.25.2.1  2006/01/24 13:39:49  fbrunet
 *  Modification view management
 *  Modification LexALP postprocessing
 *
 *  Revision 1.25  2005/08/03 10:22:03  mangeot
 *  Fixed a bug in ConsultExpert for the deleteEntry button
 *  + added a reset Button
 *
 *  Revision 1.24  2005/08/01 08:34:03  mangeot
 *  Added method getCompleteHeadword for VolumeEntry that concatenates the homograph number and the particule to the headword
 *
 *  Revision 1.23  2005/07/21 15:09:20  mangeot
 *  Bug fixes and corrections mainly for the GDEF project
 *
 *  Revision 1.22  2005/07/16 16:25:26  mangeot
 *  Adapted the linker to the GDEF project + bug fixes
 *
 *  Revision 1.21  2005/07/16 13:59:45  mangeot
 *  Fixed XML view bug
 *
 *  Revision 1.20  2005/07/16 12:58:31  serasset
 *  Added limit parameter to query functions
 *  Added a parameter to Formater initializations
 *  Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 *  Revision 1.19  2005/06/24 10:35:57  mangeot
 *  Minor bug fixes
 *
 *  Revision 1.18  2005/06/20 16:55:05  mangeot
 *  multiple bug fixes
 *
 *  Revision 1.17  2005/06/16 16:09:17  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.16  2005/06/16 13:41:15  mangeot
 *  Bugfixed in the default formatter
 *
 *  Revision 1.15  2005/06/15 16:48:28  mangeot
 *  Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *  Revision 1.14  2005/05/24 12:51:22  serasset
 *  Updated many aspect of the Papillon project to handle lexalp project.
 *  1. Layout is now parametrable in the application configuration file.
 *  2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 *  3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 *  4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 *  5. It is now possible to give a name to the cookie key in the app conf file
 *  6. Several bug fixes.
 *
 *  Revision 1.13.4.6  2005/06/15 10:08:06  mangeot
 *  Removed the AND/OR connector, now only AND criteria can be added for dict lookup
 *
 *  Revision 1.13.4.5  2005/05/27 11:53:33  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.13.4.4  2005/05/16 16:45:08  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.13.4.3  2005/05/11 14:45:29  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.13.4.2  2005/04/29 15:34:53  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.13.4.1  2005/04/29 14:50:25  mangeot
 *  New version with contribution infos embedded in the XML of the entries
 *
 *  Revision 1.13  2005/04/20 10:51:14  mangeot
 *  Correction de AddDirectTranslations
 *
 *  Revision 1.12  2005/04/18 13:22:47  mangeot
 *  Fixed a bug with the strategy
 *
 *  Revision 1.11  2005/04/18 11:06:55  mangeot
 *  Bug fix
 *
 *  Revision 1.10  2005/04/15 11:50:20  mangeot
 *  Fixed the same bug for search2text
 *
 *  Revision 1.9  2005/04/15 11:47:10  mangeot
 *  Fixed a bug when search2 can be null
 *
 *  Revision 1.8  2005/04/13 14:34:38  mangeot
 *  Simplified the expert lookup. Now lookup directly the cdm element name
 *
 *  Revision 1.7  2005/04/11 12:29:59  mangeot
 *  Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 *  Revision 1.6.2.3  2005/02/06 22:43:49  mangeot
 *  Merged the 2 Hashtables CDM Elements and XPaths into one
 *  Added a boolean (reverse-lookup) in the volume metadata and functionalities in order to perform a reverse lookup when no direct lookup result is found
 *  Added a boolean (index) in the volume metadata for indexing the only specified CDM Elements
 *
 *  Revision 1.6.2.2  2005/01/28 19:45:55  mangeot
 *  First version that runs basically.
 *  Should compile after an ant clean.
 *  XPath loading and virtual volumes for terminological lexicons are OK.
 *  Bugs remain, needs more testings like the editor for example.
 *
 *  Revision 1.6.2.1  2005/01/27 19:29:21  mangeot
 *  Implemented the HtmlDom cache, it increases speed drastically.
 *  Still does not compile after an ant clean
 *
 *  Revision 1.6  2005/01/18 12:16:10  mangeot
 *  Implemented the SQL LIMIT and OFFSET keywords. It allows us to retrieve the entries as blocks and page them. The LIMIT is the DictionariesFactory.MaxRetrievedEntries constant.
 *  The implementation may need further tuning
 *
 *  Revision 1.5  2005/01/15 12:51:24  mangeot
 *  Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 *  Revision 1.4  2005/01/14 23:08:13  mangeot
 *  Fixed some bugs in ConsultExpert + code cleaning
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
 *
 *
 */
package fr.imag.clips.papillon.presentation.deprecated;

// Enhydra SuperServlet imports
//import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;

//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// Standard imports
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.Collection;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Vector;
import java.util.Iterator;

// Imported TraX classes
        import javax.xml.transform.TransformerConfigurationException;

// Imported DOM classes
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.Text;
import org.w3c.dom.Element;

// Imported JAVA API for XML Parsing classes
//for debug
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;

import fr.imag.clips.papillon.business.dictionary.*;
import fr.imag.clips.papillon.business.user.*;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.utility.*;
import fr.imag.clips.papillon.business.locales.Languages;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.presentation.PapillonBasePO;
import fr.imag.clips.papillon.presentation.deprecated.ReviewContributions;
import fr.imag.clips.papillon.presentation.MultilingualXHtmlTemplateFactory;

/**
 *  Description of the Class
 *
 * @author     serasset
 * @created    December 17, 2004
 * @deprecated Specific page that depends on the application
 */
public class ConsultExpert extends PapillonBasePO {


    /**
     *  Description of the Field
     */
    protected final static String HANDLE_PARAMETER = "handle";
    /**
     *  Description of the Field
     */
    protected final static String FORMATTER_PARAMETER = "formatter";
    protected final static String VOLUME_PARAMETER = "VOLUME";
    protected final static String AnyContains_PARAMETER = "AnyContains";
	protected final static String OFFSET_PARAMETER = "OFFSET";
    protected final static String ANY_RESOURCE = "*ANY*";
    protected final static String ANY_TARGET = "*ANY*";

    protected final static String ContributionsVolumeParameter = "VOLUME";
    protected final static String ContributionsHandleParameter = "HANDLE";
	protected final static String ContributionsDeleteParameter = "REMOVE_VALIDATED_CONTRIB";
	
    /**
     *  Description of the Field
     */
    protected final static String ContributionsURL = "AdminContributions.po";
    /**
     *  Description of the Field
     */
    protected String languagesScript = "";

    /**
     *  Description of the Field
     */
    protected int XslSheetsNumber = 1;
	
    /**
     *  Description of the Field
     */
    protected String login = null;

    /**
     *  Description of the Method
     *
     * @return    Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return false;
    }

    /**
     *  Description of the Method
     *
     * @return    Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        return true;
    }

    /**
     *  Gets the currentSection attribute of the ConsultExpert object
     *
     * @return    The currentSection value
     */
    protected int getCurrentSection() {
        return NO_SECTION;
    }


    /**
     *  Gets the content attribute of the ConsultExpert object
     *
     * @return                                                     The content
     *      value
     * @exception  HttpPresentationException                       Description
     *      of the Exception
     * @exception  IOException                                     Description
     *      of the Exception
     * @exception  TransformerConfigurationException               Description
     *      of the Exception
     * @exception  org.xml.sax.SAXException                        Description
     *      of the Exception
     * @exception  javax.xml.parsers.ParserConfigurationException  Description
     *      of the Exception
     * @exception  java.io.IOException                             Description
     *      of the Exception
     * @exception  javax.xml.transform.TransformerException        Description
     *      of the Exception
     * @exception  ClassNotFoundException                          Description
     *      of the Exception
     * @exception  PapillonBusinessException                       Description
     *      of the Exception
     * @exception  UnsupportedEncodingException                    Description
     *      of the Exception
     */
    public Node getContent()
		throws HttpPresentationException,
		IOException,
		TransformerConfigurationException,
		org.xml.sax.SAXException,
		javax.xml.parsers.ParserConfigurationException,
		java.io.IOException,
		javax.xml.transform.TransformerException,
		ClassNotFoundException,
		PapillonBusinessException,
		UnsupportedEncodingException {

        // Content creation
        ConsultExpertXHTML content = (ConsultExpertXHTML) MultilingualXHtmlTemplateFactory.createTemplate("ConsultExpertXHTML", this.getComms(), this.getSessionData());

        // On regarde d'abord les parametres qui nous sont demandes.
		String submitLookup = myGetParameter(content.NAME_LOOKUP);
		String submitReset = myGetParameter(content.NAME_RESET);
        String sourceLanguage = myGetParameter(content.NAME_SOURCE);
        if (sourceLanguage != null && !sourceLanguage.equals("")) {
            this.setPreference(content.NAME_SOURCE, sourceLanguage);
        } else {
            sourceLanguage = this.getPreference(content.NAME_SOURCE);
        }
        List originalTargets = Arrays.asList(myGetParameterValues(content.NAME_TARGETS));
        ArrayList targetLanguages = new ArrayList(originalTargets);
        List originalResources = Arrays.asList(myGetParameterValues(content.NAME_RESOURCES));
        ArrayList resources = new ArrayList(originalResources);
        String volume = myGetParameter(VOLUME_PARAMETER);

        String search1 = myGetParameter(content.NAME_search1);
        if (search1 != null && !search1.equals("")) {
            this.setPreference(content.NAME_search1, search1);
        } else {
            search1 = this.getPreference(content.NAME_search1);
        }
        String search1text = myGetParameter(content.NAME_search1text);
		
        String search2 = myGetParameter(content.NAME_search2);
        if (search2 != null && !search2.equals("")) {
            this.setPreference(content.NAME_search2, search2);
        } else {
            search2 = this.getPreference(content.NAME_search2);
        }
        String search2text = myGetParameter(content.NAME_search2text);

        String anyContains = null;


        if (null != search1 && null != search1text && !search1text.equals("")) {
            if (search1.equals(AnyContains_PARAMETER)) {
                anyContains = search1text;
            }
        }

        if (null != search2 && null != search2text && !search2text.equals("")) {
            if (search2.equals(AnyContains_PARAMETER)) {
                anyContains = search2text;
            }
        }

        String strategyString1 = myGetParameter(content.NAME_Strategy1);
        if (strategyString1 != null && !strategyString1.equals("")) {
            this.setPreference(content.NAME_Strategy1, strategyString1);
        } else {
            strategyString1 = this.getPreference(content.NAME_Strategy1);
        }
        int strategy1 = IQuery.STRATEGY_NONE;
        if (null != strategyString1 && !strategyString1.equals("")) {
            strategy1 = Integer.parseInt(strategyString1);
        }

        String strategyString2 = myGetParameter(content.NAME_Strategy2);
        if (strategyString2 != null && !strategyString2.equals("")) {
            this.setPreference(content.NAME_Strategy2, strategyString2);
        } else {
            strategyString2 = this.getPreference(content.NAME_Strategy2);
        }
        int strategy2 = IQuery.STRATEGY_NONE;
        if (null != strategyString2 && !strategyString2.equals("")) {
            strategy2 = Integer.parseInt(strategyString2);
        }

        int offset = 0;
        String offsetString = myGetParameter(OFFSET_PARAMETER);
        if (offsetString != null && !offsetString.equals("")) {
            offset = Integer.parseInt(offsetString);
		}

        String handle = myGetParameter(HANDLE_PARAMETER);
        String formatter = myGetParameter(FORMATTER_PARAMETER);

        if (handle != null && !handle.equals("")) {
            submitLookup = "lookup";
        }

        // Consultation of several headwords at one time
        String[] Headwords = null;
		if (search1 !=null && search1.equals(Volume.CDM_headword)
			&& (strategy1 == IQuery.STRATEGY_LEMMATIZE || strategy1 == IQuery.STRATEGY_FOKS)) {
			 Headwords = new String[1];
			 Headwords[0] = search1text;
		}

        // FIXME: Just get the first language for the moment. Architecture of this part should be revised.
        Collection allSourceLanguages = AvailableLanguages.getSourceLanguagesArray();
        Collection allTargetLanguages = AvailableLanguages.getTargetLanguagesArray();
        Collection dictionaries = DictionariesFactory.getDictionariesArray();
        ArrayList allResources = new ArrayList();
        
        int i = 0;
        for (Iterator iter = dictionaries.iterator(); iter.hasNext();) {
            allResources.add(((Dictionary)iter.next()).getName());
        }
    
        //
		if ((targetLanguages.size() == 0) || ((targetLanguages.size() > 0) && ((String)targetLanguages.get(0)).equals(ANY_TARGET))) {
            
            // FIXME: null is important ?
            targetLanguages = null;
        }
        
        //
        if ((resources.size() > 0) && ((String)resources.get(0)).equals(ANY_RESOURCE)) {
            
            //
            resources = allResources;
        }

        // Header Script
        //     setHeaderScript(buildLanguagesScript(AvailableLanguages myAvailableLanguages));


        if (submitLookup != null && !submitLookup.equals("")) {
            // Lemmatize the entry Testing phase ...
            if (Headwords !=null &&
				strategy1 == IQuery.STRATEGY_LEMMATIZE) {
                Wrapper myWrapper = WrappersFactory.createXRCEAnalyzerWrapper(sourceLanguage);
                String result = myWrapper.getResultString(Headwords);
                String[] Results = WrappersFactory.getXRCEAnalyzerAnswers(result);
                if (Results != null && Results.length > 1) {
                    Headwords = Results;
                }
				else if (Results != null && Results.length == 0) {
					Headwords = null;
					search1text = Results[0];
				}
				else {
					Headwords = null;
					
				}
                strategy1 = IQuery.STRATEGY_EXACT;
            }
            // Using FOKS module Testing phase ...
            if (Headwords !=null &&
				strategy1 == IQuery.STRATEGY_FOKS && sourceLanguage.equals("jpn")) {
                Vector foksVector = VolumeEntriesFactory.getFoksEntriesVector(Headwords[0]);
                if (foksVector != null && foksVector.size() > 0) {
                    addFoksEntryTable(content, foksVector, sourceLanguage, originalTargets, originalResources);
                    strategy1 = IQuery.STRATEGY_EXACT;
                    Utility.removeElement(content.getElementSorryMessage());
                } else {
                    Utility.removeElement(content.getElementFoksEntries());
                }
                Utility.removeElement(content.getElementEntryListTable());
                Utility.removeElement(content.getElementVolumeEntries());
            } else {
                // If there is a query, executing it
				// constructions of the Keys vector
				
				Vector myKeys = new Vector();
				Vector Clauses = new Vector();
				if (Headwords == null &&
					search1 !=null && !search1.equals("")  &&
					search1text != null && !search1text.equals("")) {
					String[] key1 = new String[4];
					key1[0] = search1;
					key1[2] = search1text;
					key1[3] = IQuery.QueryBuilderStrategy[strategy1+1];
					myKeys.add(key1);
				}
				if (search2 !=null && !search2.equals("") &&
					search2text != null && !search2text.equals("")) {
					String[] key2 = new String[4];
					key2[0] = search2;
					key2[2] = search2text;
					key2[3] = IQuery.QueryBuilderStrategy[strategy2+1];
					myKeys.add(key2);
				}
				String[] status = new String[4];
				status[0] = Volume.CDM_contributionStatus;
				status[1] = Volume.DEFAULT_LANG;
				status[2] = VolumeEntry.VALIDATED_STATUS;
				status[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];
				myKeys.add(status);
				
                addEntries(content,
						   resources, volume, sourceLanguage, targetLanguages, 
						   Headwords, strategy1, 
						   search1, search1text, strategyString1,
						   search2, search2text, strategyString2,
						   myKeys, Clauses, 
						   anyContains, handle, formatter, 
						   this.getUser(), offset);
                Utility.removeElement(content.getElementFoksEntries());
            }
        } else {
            // If there is no query, ie connection for the first time, adding the Home content
            Utility.removeElement(content.getElementEntryListTable());
            Utility.removeElement(content.getElementVolumeEntries());
            Utility.removeElement(content.getElementFoksEntries());
            Utility.removeElement(content.getElementSorryMessage());
        }

		if (submitReset != null) {
			this.resetPreferences();
			addConsultForm(content,
						   null, allSourceLanguages,
						   null, allTargetLanguages,
						   null, allResources,
						   null, null,
						   null, null,
						   IQuery.STRATEGY_EXACT, IQuery.STRATEGY_EXACT);
		}
		else {
			addConsultForm(content,
						   sourceLanguage, allSourceLanguages,
						   originalTargets, allTargetLanguages, 
						   originalResources, allResources,
						   search1, search1text,
						   search2, search2text,
						   strategy1, strategy2);
		}
		
        return content.getElementConsultContent();
    }


    /**
     *  Adds a feature to the ConsultForm attribute of the ConsultExpert object
     *
     * @exception  PapillonBusinessException     Description of the Exception
     * @exception  HttpPresentationException     Description of the Exception
     * @exception  UnsupportedEncodingException  Description of the Exception
     */
    protected void addConsultForm(ConsultExpertXHTML content,
								  String sourceLanguage, Collection allSourceLanguages,
								  Collection originalTargets, Collection allTargetLanguages, 
								  Collection originalResources, Collection allResources,
								  String search1, String search1text,
								  String search2, String search2text,
								  int strategy1, int strategy2)
		throws PapillonBusinessException,
		HttpPresentationException,
		UnsupportedEncodingException {
				
        // Adding the appropriate source languages to the source list
        XHTMLOptionElement sourceOptionTemplate = content.getElementSourceOptionTemplate();
        Node sourceSelect = sourceOptionTemplate.getParentNode();
        sourceOptionTemplate.removeAttribute("id");
        // We assume that the option element has only one text child
        // (it should be this way if the HTML is valid...)
        Text sourceTextTemplate = (Text) sourceOptionTemplate.getFirstChild();
        String langLoc = getUserPreferredLanguage();
        if (null == sourceLanguage || sourceLanguage.equals("")) {
            sourceLanguage = langLoc;
        }
        
        //
        for (Iterator iter = allSourceLanguages.iterator(); iter.hasNext();) {
            String langi = (String)iter.next();
            
            //
            sourceOptionTemplate.setValue(langi);
            if (this.IsClientWithLabelDisplayProblems()) {
                sourceOptionTemplate.setLabel(Languages.localizeLabel(langLoc, langi));
            } else {
                sourceOptionTemplate.setLabel(Languages.localizeName(langLoc, langi));
            }
            sourceOptionTemplate.setSelected(langi.equals(sourceLanguage));

            // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux
            // specs W3C.
            sourceTextTemplate.setData(Languages.localizeName(langLoc, langi));
            sourceSelect.appendChild(sourceOptionTemplate.cloneNode(true));
        }
        sourceSelect.removeChild(sourceOptionTemplate);

        // Adding the appropriate target languages to the target list
        //if (originalTargets == null || originalTargets.length == 0) {
        //    originalTargets = new String[]{ANY_TARGET};
        //}
        XHTMLOptionElement targetOptionTemplate = content.getElementTargetOptionTemplate();
        XHTMLSelectElement targetSelect = (XHTMLSelectElement) targetOptionTemplate.getParentNode();
        if (!this.IsClientWithLabelDisplayProblems()) {
            this.setUnicodeLabels(targetSelect);
        }
        targetOptionTemplate.removeAttribute("id");
        // We assume that the option element has only one text child
        // (it should be this way if the HTML is valid...)
        Text targetTextTemplate = (Text) targetOptionTemplate.getFirstChild();

        //
        for (Iterator iter = allTargetLanguages.iterator(); iter.hasNext();) {
            String langi = (String)iter.next();
            targetOptionTemplate.setValue(langi);
            if (this.IsClientWithLabelDisplayProblems()) {
                targetOptionTemplate.setLabel(Languages.localizeLabel(langLoc, langi));
            } else {
                targetOptionTemplate.setLabel(Languages.localizeName(langLoc, langi));
            }
            // We should select the previously selected target languages...
            targetTextTemplate.setData(Languages.localizeName(langLoc, langi));
            targetSelect.appendChild(targetOptionTemplate.cloneNode(true));
        }
        targetSelect.removeChild(targetOptionTemplate);
        this.setSelected(content.getElementTargets(), (String[])originalTargets.toArray(new String[originalTargets.size()]));

        // Adding the appropriate resources languages to the resources list
        if (originalResources == null || originalResources.size() == 0) {
            ArrayList list = new ArrayList();
            list.add(ANY_RESOURCE);
            originalResources = list;
        }
        XHTMLOptionElement resourceOptionTemplate = content.getElementResourceOptionTemplate();
        XHTMLSelectElement resourceSelect = (XHTMLSelectElement) resourceOptionTemplate.getParentNode();
        if (!this.IsClientWithLabelDisplayProblems()) {
            this.setUnicodeLabels(resourceSelect);
        }
        resourceOptionTemplate.removeAttribute("id");
        // We assume that the option element has only one text child
        // (it should be this way if the HTML is valid...)
        Text resourceTextTemplate = (Text) resourceOptionTemplate.getFirstChild();
        
        //
        for (Iterator iter = allResources.iterator(); iter.hasNext();) {
            String resource = (String)iter.next();
            
            //
            resourceOptionTemplate.setValue(resource);
            resourceOptionTemplate.setLabel(resource);
            // We should select the previously selected resources...
            // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux
            // specs W3C.
            resourceTextTemplate.setData(resource);
            resourceSelect.appendChild(resourceOptionTemplate.cloneNode(true));
        }
        resourceSelect.removeChild(resourceOptionTemplate);
        this.setSelected(content.getElementResources(), (String[])originalResources.toArray(new String[originalResources.size()]));

        // strategy
        if (!this.IsClientWithLabelDisplayProblems()) {
            this.setUnicodeLabels(content.getElementStrategy1());
        }
        this.setSelected(content.getElementStrategy1(), "" + strategy1);

        if (!this.IsClientWithLabelDisplayProblems()) {
            this.setUnicodeLabels(content.getElementStrategy2());
        }
        this.setSelected(content.getElementStrategy2(), "" + strategy2);

        // Search1 field
        if (search1 == null || search1.equals("")) {
            search1 = Volume.CDM_headword;
        }
        if (!this.IsClientWithLabelDisplayProblems()) {
            this.setUnicodeLabels(content.getElementSearch1());
        }
        this.setSelected(content.getElementSearch1(), search1);

        XHTMLInputElement search1Input = content.getElementSearch1text();
        search1Input.setValue(search1text);

        // Search2 field
        if (search2 == null || search2.equals("")) {
            search2 = Volume.CDM_pos;
        }
        if (!this.IsClientWithLabelDisplayProblems()) {
            this.setUnicodeLabels(content.getElementSearch2());
        }
        this.setSelected(content.getElementSearch2(), search2);

        XHTMLInputElement search2Input = content.getElementSearch2text();
        search2Input.setValue(search2text);
    }


    /**
     *  Adds a feature to the Entries attribute of the ConsultExpert object
     *
     * @param  resources                                           The feature
     *      to be added to the Entries attribute
     * @param  volume                                              The feature
     *      to be added to the Entries attribute
     * @param  source                                              The feature
     *      to be added to the Entries attribute
     * @param  targets                                             The feature
     *      to be added to the Entries attribute
     * @param  Headwords                                           The feature
     *      to be added to the Entries attribute
     * @param  strategy                                            The feature
     *      to be added to the Entries attribute
     * @param  posContains                                         The feature
     *      to be added to the Entries attribute
     * @param  pronContains                                        The feature
     *      to be added to the Entries attribute
     * @param  readingContains                                     The feature
     *      to be added to the Entries attribute
     * @param  transContains                                       The feature
     *      to be added to the Entries attribute
     * @param  anyContains                                         The feature
     *      to be added to the Entries attribute
     * @param  handle                                              The feature
     *      to be added to the Entries attribute
     * @param  formatter                                               The feature
     *      to be added to the Entries attribute
     * @param  myUser                                              The feature
     *      to be added to the Entries attribute
     * @exception  PapillonBusinessException                       Description
     *      of the Exception
     * @exception  ClassNotFoundException                          Description
     *      of the Exception
     * @exception  HttpPresentationException                       Description
     *      of the Exception
     * @exception  IOException                                     Description
     *      of the Exception
     * @exception  TransformerConfigurationException               Description
     *      of the Exception
     * @exception  org.xml.sax.SAXException                        Description
     *      of the Exception
     * @exception  javax.xml.parsers.ParserConfigurationException  Description
     *      of the Exception
     * @exception  javax.xml.transform.TransformerException        Description
     *      of the Exception
     */
    protected void addEntries(ConsultExpertXHTML content,
							  Collection resources, String volume, 
							  String source, Collection targets, 
							  String[] Headwords, int strategy1, 
							  String search1, String search1text, String strategyString1,
							  String search2, String search2text, String strategyString2,
							  Vector myKeys,  Vector myClauses, 
							  String anyContains, String handle, 
							  String formatter, 
							  User myUser, int offset)
             throws PapillonBusinessException,
            ClassNotFoundException,
            HttpPresentationException,
            IOException,
            TransformerConfigurationException,
            org.xml.sax.SAXException,
            javax.xml.parsers.ParserConfigurationException,
            javax.xml.transform.TransformerException {

        Collection EntryCollection = null;
		boolean reverseLookup = false; 

        if (null != handle && null != volume) {
            EntryCollection = DictionariesFactory.findAnswerAndTranslations(volume, handle, targets, myUser);
        } else if (null != volume) {
			EntryCollection = VolumeEntriesFactory.getVolumeNameEntriesVector(volume, myKeys, myClauses, null);
        } else {
			if (Headwords!=null && Headwords.length>0) {
				EntryCollection = (Collection) new Vector();
				for (int i=0; i< Headwords.length; i++) {
					String[] Headword = new String[4];
					Headword[0] = Volume.CDM_headword;
					Headword[1] = source;
					Headword[2] = Headwords[i];
					Headword[3] = IQuery.QueryBuilderStrategy[strategy1+1];
					myKeys.add(Headword);
					EntryCollection.addAll(DictionariesFactory.getDictionariesEntriesCollection(resources, source, targets, myKeys, myClauses, anyContains, myUser, offset, DictionariesFactory.MaxRetrievedEntries));
					myKeys.remove(Headword);
				}
			}
			else {
				EntryCollection = DictionariesFactory.getDictionariesEntriesCollection(resources, source, targets, myKeys, myClauses, anyContains, myUser, offset, DictionariesFactory.MaxRetrievedEntries);
			}
 			if (EntryCollection==null || EntryCollection.size()==0) {
				PapillonLogger.writeDebugMsg("EntryCollection null, getDictionariesReverseEntriesCollection");
				EntryCollection = DictionariesFactory.getDictionariesReverseEntriesCollection(resources, source, targets, myKeys, myClauses, anyContains, myUser,offset);
				reverseLookup = (EntryCollection!=null && EntryCollection.size()>0);
			}
       }
        // If there are too much entries ie > DictionariesFactory.MaxDisplayedEntries,
        // we display a table of entries instead of the entries
        if (null != EntryCollection && EntryCollection.size() > 0) {
			if (reverseLookup) {
				XHTMLElement reverseLookupMsg = content.getElementReverseLookupMessage();
				reverseLookupMsg.setAttribute("class","");
			}
            if (EntryCollection.size() > DictionariesFactory.MaxDisplayedEntries) {
                Utility.removeElement(content.getElementVolumeEntries());
				addEntryTable(content,
							  EntryCollection, source, resources, targets, 
							  search1, search1text, strategyString1,
							  search2, search2text, strategyString2,
							  offset);
			} else {
                Utility.removeElement(content.getElementEntryListTable());
                addFewEntries(content,EntryCollection, formatter);
                // If the entry is remote, it is already an HTML node
            }
            Utility.removeElement(content.getElementSorryMessage());
        } else {
            Utility.removeElement(content.getElementVolumeEntries());
            Utility.removeElement(content.getElementEntryListTable());
        }
    }


    /**
     *  Adds a feature to the EntryTable attribute of the ConsultExpert object
     *
     * @param  EntryCollection                           The feature to be added
     *      to the EntryTable attribute
     * @param  targets                                   The feature to be added
     *      to the EntryTable attribute
     * @exception  PapillonBusinessException             Description of the
     *      Exception
     * @exception  java.io.UnsupportedEncodingException  Description of the
     *      Exception
     */
    protected void addEntryTable(ConsultExpertXHTML content,
								 Collection qrset, 
								 String source,
								 Collection originalResources,
								 Collection targets, 
								 String search1, String search1text, String strategyString1,
								 String search2, String search2text, String strategyString2,
								 int offset)
		throws PapillonBusinessException,
		java.io.UnsupportedEncodingException {

        PapillonLogger.writeDebugMsg("addEntryTable, size: " + qrset.size());
        // On récupère les éléments du layout
        XHTMLTableRowElement entryListRow = content.getElementEntryListRow();
        XHTMLElement vocable = content.getElementVocable();
        XHTMLAnchorElement entryAnchor = content.getElementEntryAnchor();
        XHTMLElement entryId = content.getElementEntryIdList();
        XHTMLElement author = content.getElementAuthorList();
        XHTMLElement pos = content.getElementPosEntry();
        XHTMLElement dictname = content.getElementDictionaryName();

        // Recuperating the elements for the formula
        XHTMLTableRowElement formulaRow = content.getElementFormulaRow();
        XHTMLElement formulaElement = content.getElementFormula();
		
		XHTMLElement entryNumberElement = content.getElementEntryNumber();
		XHTMLAnchorElement previousEntriesAnchor = content.getElementPreviousEntriesAnchor();
		XHTMLAnchorElement nextEntriesAnchor = content.getElementNextEntriesAnchor();

        //      we don't take off the id attribute because we will take the element off later...
        //      entryListRow.removeAttribute("id");
        vocable.removeAttribute("id");
        entryAnchor.removeAttribute("id");
        entryId.removeAttribute("id");
        author.removeAttribute("id");
        pos.removeAttribute("id");
        dictname.removeAttribute("id");
        formulaElement.removeAttribute("id");
        entryNumberElement.removeAttribute("id");
        previousEntriesAnchor.removeAttribute("id");
        nextEntriesAnchor.removeAttribute("id");
		
		content.setTextEntryNumber(""+qrset.size());
		
		
        String href = this.getUrl() + "?"
			+ serializeParameterForUrl(content.NAME_RESOURCES, originalResources) + "&"
			+ content.NAME_search1 + "=" + search1 + "&"
			+ content.NAME_search1text + "=" + search1text + "&"
			+ content.NAME_search2 + "=" + search2 + "&"
			+ content.NAME_search2text + "=" + search2text + "&"
			+ content.NAME_SOURCE + "=" + source + "&"
			+ serializeParameterForUrl(content.NAME_TARGETS, targets) + "&"
			+ content.NAME_Strategy1 + "=" + strategyString1 + "&"
			+ content.NAME_Strategy2 + "=" + strategyString2 + "&"
			+ content.NAME_LOOKUP + "=" + content.NAME_LOOKUP + "&"
			+ OFFSET_PARAMETER + "=";
		if (offset >= DictionariesFactory.MaxRetrievedEntries) {
			int prevOffset = offset-DictionariesFactory.MaxRetrievedEntries;
			previousEntriesAnchor.setHref(href+prevOffset);
		}
		else {
			previousEntriesAnchor.setHref("");
			content.setTextPreviousEntriesAnchor("");			
		}
		int nextOffset = offset+DictionariesFactory.MaxRetrievedEntries;
		nextEntriesAnchor.setHref(href+nextOffset);

        // On récupère le noeud contenant la table...
        Node lexieTable = entryListRow.getParentNode();
        if (null != qrset) {
            for (Iterator myIterator = qrset.iterator(); myIterator.hasNext(); ) {
                QueryResult myQueryResult = (QueryResult) myIterator.next();
                VolumeEntry myEntry = myQueryResult.getSourceEntry();

                // Le vocable
                content.setTextVocable(myEntry.getCompleteHeadword());

                // l'entry
                href = this.getUrl() + "?"
                         + VOLUME_PARAMETER + "=" + myEntry.getVolumeName() + "&"
                         + HANDLE_PARAMETER + "=" + myEntry.getHandle() + "&"
                         + content.NAME_SOURCE + "=" + myEntry.getSourceLanguage() + "&"
                         + serializeParameterForUrl(content.NAME_TARGETS, targets) + "&"
                         + content.NAME_Strategy1 + "=" + strategyString1 + "&"
                         + content.NAME_LOOKUP + "=" + content.NAME_LOOKUP;
                entryAnchor.setHref(href);
                
                // FIXME: is this usefull, shouldn't id be always defined for a db entry...
                String id = myEntry.getId();
                if (id == null) {
                    id = "";
                }
                
                content.setTextEntryIdList(myEntry.getId());

                content.setTextAuthorList(myEntry.getAuthor());

                // Le pos
                String posstr = null;
                
                // FIXME: Apparently, volumeEntries ALWAYS return LocalEntry.... can we have remote entries here ?
                if (myEntry.getType() == IAnswer.LocalEntry) {
					posstr = ((VolumeEntry) myEntry).getPos();
                }
                if (null == posstr || posstr.equals("")) {
                    posstr = "+";
                }
                content.setTextPosEntry(posstr);

                // dictname
                content.setTextDictionaryName(myEntry.getDictionaryName());

                // The formula
                // FIXME: Apparently, volumeEntries ALWAYS return LocalEntry.... can we have remote entries here ?
                 if (myEntry.getType() == IAnswer.LocalEntry) {
					 content.setTextFormula(((VolumeEntry) myEntry).getDefinition());
                }

                XHTMLElement cloneEntry = (XHTMLElement) entryListRow.cloneNode(true);
                XHTMLElement cloneFormula = (XHTMLElement) formulaRow.cloneNode(true);
                //      we have to take off the id attribute because we did not take it off the original
                // FIXME: why did we not tae it off in the original ?
                cloneEntry.removeAttribute("id");
                cloneFormula.removeAttribute("id");
                lexieTable.appendChild(cloneEntry);
                lexieTable.appendChild(cloneFormula);

            }
        }
        lexieTable.removeChild(entryListRow);
        lexieTable.removeChild(formulaRow);
    }


    /**
     *  Adds a feature to the FoksEntryTable attribute of the ConsultExpert
     *  object
     *
     * @param  EntryVector                               The feature to be added
     *      to the FoksEntryTable attribute
     * @exception  PapillonBusinessException             Description of the
     *      Exception
     * @exception  java.io.UnsupportedEncodingException  Description of the
     *      Exception
     */
    protected void addFoksEntryTable(ConsultExpertXHTML content, Vector EntryVector, String sourceLanguage, Collection originalTargets, Collection originalResources)
             throws PapillonBusinessException,
            java.io.UnsupportedEncodingException {

        // On récupère les éléments du layout
        XHTMLTableRowElement entryListRow = content.getElementFoksRow();
        XHTMLElement foksEntry = content.getElementFoksEntry();
        XHTMLElement foksWriting = content.getElementFoksWriting();
        XHTMLElement foksGrade = content.getElementFoksGrade();
        XHTMLAnchorElement foksHref = content.getElementFoksHref();

        //      we don't take off the id attribute because we will take the element off later...
        //      entryListRow.removeAttribute("id");
        foksEntry.removeAttribute("id");
        foksWriting.removeAttribute("id");
        foksGrade.removeAttribute("id");
        foksHref.removeAttribute("id");

        // On récupère le noeud contenant la table...
        Node lexieTable = entryListRow.getParentNode();
        if (null != EntryVector) {
            for (int i = 0; i < EntryVector.size(); i++) {
                String href;
                FoksEntry myEntry = (FoksEntry) EntryVector.get(i);
                String headword = myEntry.getHeadword();
                content.setTextFoksEntry(headword);
                String myWriting = "";
                VolumeEntry myWritingEntry = VolumeEntriesFactory.getJMDictVolumeEntry(headword);
                if (myWritingEntry != null && !myWritingEntry.isEmpty()) {
                    //NOTE: I store the writing of the Japanese entries in the Key1 field.
                    // The reading field is used for transcriptions like romaji
					myWriting = IndexEntry.getCdmString(myWritingEntry,Volume.CDM_writing,"jpn");
                }
                content.setTextFoksWriting(myWriting);
                // Le grade
                content.setTextFoksGrade(myEntry.getScoreString());

                href = this.getUrl() + "?"
                         + content.NAME_SOURCE + "=" + sourceLanguage + "&"
                         + serializeParameterForUrl(content.NAME_TARGETS, originalTargets)  + "&"
                         + serializeParameterForUrl(content.NAME_RESOURCES, originalResources) + "&"
                         + content.NAME_search1 + "=" + Volume.CDM_headword + "&"
                         + content.NAME_search1text + "="  + Utility.convertToUrlForEncoding(headword, "UTF-8") + "&"
                         + content.NAME_Strategy1 + "=" + IQuery.STRATEGY_EXACT + "&"
                         + content.NAME_LOOKUP + "=" + "lookup";
                foksHref.setHref(href);
                XHTMLElement clone = (XHTMLElement) entryListRow.cloneNode(true);
                //      we have to take off the id attribute because we did not take it off the original
                clone.removeAttribute("id");
                lexieTable.appendChild(clone);
            }
        }
        lexieTable.removeChild(entryListRow);
    }


    /**
     *  Gets the xslTable attribute of the ConsultExpert object
     *
     * @param  volume                         Description of the Parameter
     * @param  handle                         Description of the Parameter
     * @param  type                           Description of the Parameter
     * @return                                The xslTable value
     * @exception  HttpPresentationException  Description of the Exception
     * @exception  PapillonBusinessException  Description of the Exception
     */
    protected Node getXslTable(ConsultExpertXHTML content, String volume, String handle, int type)
             throws HttpPresentationException,
            PapillonBusinessException {

        XHTMLElement stylesheetRow = null;

        try {

            Vector xslList = new Vector();

			xslList.add(XslTransformation.XML_FORMATTER);
            // FIXME: We have too much XslSheets so I put only XML and DEC
            // but it should not be hardcoded!
            //    xslList.add("DEC");
            // size + 1 for the default xsl sheet
            XslSheetsNumber = xslList.size() + 1;
            //  XslSheetsNumber = xslList.size();
            //for the stylesheet content
            // On récupère le noeud contenant la table...
            XHTMLTableRowElement originalStylesheetRow = content.getElementStylesheetRow();

            stylesheetRow = (XHTMLElement) originalStylesheetRow.cloneNode(true);
            // The cell element is added in proder to be parsed by enhydra
            // but here we remove it
            XHTMLElement stylesheetCell = (XHTMLElement) stylesheetRow.getFirstChild();

            stylesheetRow.removeAttribute("id");
            stylesheetRow.removeChild(stylesheetCell);

            Document theDoc = stylesheetRow.getOwnerDocument();

            // default XSL = null !
            Element cell = theDoc.createElement("td");
            Element anchor = theDoc.createElement("a");
            Text xslName = theDoc.createTextNode("Default");
            anchor.setAttribute("href", this.getUrl() + "?" +
                    VOLUME_PARAMETER + "=" + volume + "&" +
                    HANDLE_PARAMETER + "=" + handle + "&" +
                    content.NAME_LOOKUP + "=" + "lookup");
            anchor.appendChild(xslName);
            cell.appendChild(anchor);
            stylesheetRow.appendChild(cell);

            //adding xslchoice
            for (int i = 0; i < xslList.size(); i++) {
                // Le nom de la stylesheet
                String name = (String) xslList.elementAt(i);
                //creation de la data cell
                cell = theDoc.createElement("td");
                anchor = theDoc.createElement("a");
                xslName = theDoc.createTextNode(name);
                anchor.setAttribute("href", this.getUrl() + "?" +
                        VOLUME_PARAMETER + "=" + volume + "&" +
                        HANDLE_PARAMETER + "=" + handle + "&" +
                        FORMATTER_PARAMETER + "=" + name + "&" +
                        content.NAME_LOOKUP + "=" + "lookup");
                anchor.appendChild(xslName);
                cell.appendChild(anchor);
                stylesheetRow.appendChild(cell);
            }
        } catch (Exception ex) {
            throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in getXslTable()", ex);
        }
        return stylesheetRow;
    }


    /**
     *  Adds a feature to the Element attribute of the ConsultExpert object
     *
     * @param  element                        The feature to be added to the
     *      Element attribute
     * @param  table                          The feature to be added to the
     *      Element attribute
     * @param  handle                         The feature to be added to the
     *      Element attribute
     * @param  resource                       The feature to be added to the
     *      Element attribute
     * @param  type                           The feature to be added to the
     *      Element attribute
     * @exception  PapillonBusinessException  Description of the Exception
     */
    protected void addElement(ConsultExpertXHTML content, Element element, String table, String handle, String resource, String volume, int type)
             throws PapillonBusinessException {
        try {
            //for the entry content
            XHTMLTableRowElement originalEntryRow = content.getElementEntryRow();
			XHTMLElement deleteButton =  content.getElementDeleteButton();
			XHTMLAnchorElement deleteHref =  content.getElementDeleteHref();
            Node entryTable = originalEntryRow.getParentNode();
            //for the entry content
            content.getElementResourceName().removeAttribute("id");
            content.setTextResourceName(resource);

			// the delete button
			String originalDeleteHref = deleteHref.getHref();
			if (this.getUser() != null && this.getUser().isValidator()) {
				String newHref = originalDeleteHref + "?"
				+ ContributionsDeleteParameter + "=" + ContributionsDeleteParameter + "&"
				+ ContributionsVolumeParameter + "=" + volume + "&"
				+ ContributionsHandleParameter + "=" + handle;
				deleteHref.setHref(newHref);
				deleteButton.setAttribute("class","");
			}
			
            //for the lexie content
            XHTMLTableCellElement entryCell = (XHTMLTableCellElement) originalEntryRow.getFirstChild();

            if (null != table && null != handle) {
                // If we want to put the XSL menu ...
                entryTable.appendChild(getXslTable(content, table, handle, type));
                entryCell.setColSpan(XslSheetsNumber);
            }
            XHTMLElement entryDiv = (XHTMLElement) content.getElementEntryDiv();
            entryDiv.removeAttribute("id");

            while (entryDiv.hasChildNodes()) {
                entryDiv.removeChild(entryDiv.getFirstChild());
            }

            entryDiv.appendChild(content.importNode(element, true));

            XHTMLTableRowElement entryRow = (XHTMLTableRowElement) originalEntryRow.cloneNode(true);
			
			// resetting original deleteHref
			deleteHref.setHref(originalDeleteHref);
			
            entryRow.removeAttribute("id");

            entryTable.appendChild(entryRow);
            // Don't remove the original node in order to add more entries ...
            //entryTable.removeChild(entryRow);
        } catch (Exception ex) {
            throw new PapillonBusinessException("Exception in addEntries: ", ex);
        }
    }


    /**
     *  Adds a feature to the FewEntries attribute of the ConsultExpert object
     *
     * @param  EntryCollection
     *      The feature to be added to the FewEntries attribute
     * @param  formatter
     *      The feature to be added to the FewEntries attribute
     * @exception  fr.imag.clips.papillon.business.PapillonBusinessException
     *      Description of the Exception
     */
    protected void addFewEntries(ConsultExpertXHTML content, Collection EntryCollection, String formatter)
             throws fr.imag.clips.papillon.business.PapillonBusinessException {
        if (EntryCollection != null && EntryCollection.size() > 0) {
            for (Iterator myIterator = EntryCollection.iterator(); myIterator.hasNext(); ) {
				addEntry(content, (QueryResult) myIterator.next(), formatter);
            }
        } else {
            Utility.removeElement(content.getElementEntryListTable());
        }
        // removing
        Utility.removeElement(content.getElementStylesheetRow());
        Utility.removeElement(content.getElementEntryRow());
    }

    /**
     *  Adds an Entry to the Home object
     *
     * @param  EntryCollection
     *      The feature to be added to the FewEntries attribute
     * @param  formatter
     *      The feature to be added to the FewEntries attribute
     * @exception  fr.imag.clips.papillon.business.PapillonBusinessException
     *      Description of the Exception
     */
    protected void addEntry(ConsultExpertXHTML content, QueryResult qr, String formatter)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
            VolumeEntry myEntry = qr.getSourceEntry();
            // get the apropriate transformer.
            ResultFormatter rf = ResultFormatterFactory.getFormatter(qr, formatter, ResultFormatterFactory.XHTML_DIALECT,null);
            //rf.initializeFormatter(qr.getSourceEntry().getDictionary(), qr.getSourceEntry().getVolume() , null, ResultFormatterFactory.XHTML_DIALECT,null);
			Element myHtmlElt = (Element)rf.getFormattedResult(qr, this.getUser());
            
			addElement(content, myHtmlElt, myEntry.getVolumeName(), myEntry.getHandle(),  myEntry.getDictionaryName(), myEntry.getVolumeName(), myEntry.getType());
		}
	
    /**
     *  Description of the Method
     *
     * @return
     *      Description of the Return Value
     * @exception  fr.imag.clips.papillon.business.PapillonBusinessException
     *      Description of the Exception
     */
    protected String buildLanguagesScript(String[] allSourceLanguages)
             throws fr.imag.clips.papillon.business.PapillonBusinessException {

        String script;

        // Heading of the script
        script = "\n<!--\nfunction UpdateList(object) {\n" +
                "     // Remove all existing items from TARGETS list\n" +
                "     object.TARGETS.options.length = 0;\n\n" +
                "     var defaultSelected = false;\n" +
                "     var selected = false;\n\n";

        // Writing one Option variable for each existing language
        String langLoc = getUserPreferredLanguage();
        for (Iterator iter = AvailableLanguages.getAllLanguagesArray().iterator(); iter.hasNext();) {
            String myLang = (String)iter.next();
            
            //
            String varLang = "  var " + myLang + " = new Option(\"" +
                    Languages.localizeName(langLoc, myLang) + "\", \"" + myLang + "\");";
            script = script + "\n" + varLang;
        }

        script = script + "\n\n";

        // Writing one if instruction for each source language
        for (int i = 0; i < allSourceLanguages.length; i++) {
            String mySource = allSourceLanguages[i];
            String ifScript = "  if (object.SOURCE.options[object.SOURCE.selectedIndex].value == \"" + mySource + "\") {";
           
            //
            int j = 0;
            for (Iterator iter = AvailableLanguages.getTargetLanguagesArray(mySource).iterator(); iter.hasNext();) {
                ifScript = ifScript + "\n" + "    object.TARGETS.options["
                         + j + "] = " + (String)iter.next() + ";";
                j++;
            }
            script = script + "\n" + ifScript + "\n}";
        }

        // Tail of the script
        script = script + "\n}\n-->\n";

        return script;
    }
}

