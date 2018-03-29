/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Mathieu Mangeot - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.1.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.35  2007/01/05 13:57:26  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.34  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.33  2006/03/06 10:16:45  mangeot
 * MM: I do not delete any contribution, even not finished.
 * I tag them "deleted"
 *
 * Revision 1.32  2006/03/06 10:11:47  mangeot
 * *** empty log message ***
 *
 * Revision 1.31  2006/03/05 11:54:18  mangeot
 * *** empty log message ***
 *
 * Revision 1.30  2006/03/02 11:55:50  mangeot
 * *** empty log message ***
 *
 * Revision 1.29  2006/03/01 15:47:08  mangeot
 * syntax bug fixes
 *
 * Revision 1.28  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.27  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.23  2005/09/08 15:04:25  mangeot
 * *** empty log message ***
 *
 * Revision 1.22  2005/09/08 14:13:08  mangeot
 * Bug fix in Validated contrib deletion process
 *
 * Revision 1.21  2005/09/03 11:19:19  mangeot
 * Bug fix in redirection after reviewing
 *
 * Revision 1.20.2.1  2006/01/24 13:39:49  fbrunet
 * Modification view management
 * Modification LexALP postprocessing
 *
 * Revision 1.20  2005/08/17 12:58:16  mangeot
 * Fixed a bug when creating an entry from an existing one.
 * From now on, the entry id is the same.
 * Added the links into ReviewContributions.java
 *
 * Revision 1.19  2005/08/02 14:41:49  mangeot
 * Work on stylesheets and
 * added a reset button for Review and AdminContrib forms
 *
 * Revision 1.18  2005/08/01 17:37:33  mangeot
 * Bug fix in sort function
 *
 * Revision 1.17  2005/08/01 08:34:03  mangeot
 * Added method getCompleteHeadword for VolumeEntry that concatenates the homograph number and the particule to the headword
 *
 * Revision 1.16  2005/07/30 16:03:28  mangeot
 * Bug fixes
 *
 * Revision 1.15  2005/07/28 16:40:33  mangeot
 * *** empty log message ***
 *
 * Revision 1.14  2005/07/21 15:09:20  mangeot
 * Bug fixes and corrections mainly for the GDEF project
 *
 * Revision 1.13  2005/07/16 13:43:51  mangeot
 * src/fr/imag/clips/papillon/presentation/ReviewContributions.java
 *
 * Revision 1.12  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 * Revision 1.11  2005/06/15 16:48:28  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.10  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.9.4.16  2005/06/15 10:08:06  mangeot
 * Removed the AND/OR connector, now only AND criteria can be added for dict lookup
 *
 * Revision 1.9.4.15  2005/05/24 11:15:48  mangeot
 * Bug fixes in sort
 *
 * Revision 1.9.4.14  2005/05/19 09:43:26  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.13  2005/05/13 11:08:26  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.12  2005/05/11 15:42:31  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.11  2005/05/11 15:34:00  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.10  2005/05/11 15:16:19  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.9  2005/05/11 15:10:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.8  2005/05/11 14:22:16  mangeot
 * Ajout de entrées précédentes et suivantes
 *
 * Revision 1.9.4.7  2005/04/30 09:11:20  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.6  2005/04/29 18:43:13  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.5  2005/04/29 18:38:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.4  2005/04/29 18:29:59  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.3  2005/04/29 17:30:30  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.2  2005/04/29 17:08:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.9.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.9  2005/04/15 13:20:08  mangeot
 * Added setIdIfNull
 *
 * Revision 1.8  2005/04/15 11:38:05  mangeot
 * Fixed a bug, not using entryHandle from contributions table any more
 *
 * Revision 1.7  2005/04/14 08:25:12  mangeot
 * *** empty log message ***
 *
 * Revision 1.6  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.5  2005/04/11 08:01:02  fbrunet
 * Passage en xhtml des ressources Papillon.
 *
 * Revision 1.4.2.5  2005/04/09 14:51:47  mangeot
 * Added more consult options for AdminContributions page
 *
 * Revision 1.4.2.4  2005/03/30 11:17:07  mangeot
 * Modified table contributions: replaced originalhandle by originalid
 * Corrected a few bugs when validating an already existing entry
 *
 * Revision 1.4.2.3  2005/03/16 09:05:01  mangeot
 * *** empty log message ***
 *
 * Revision 1.4.2.2  2005/02/25 10:22:08  mangeot
 * Bug fixes and added the use of referrer when exiting from Reviewcontributions.po
 *
 * Revision 1.4.2.1  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
 *
 * Revision 1.4  2005/01/15 20:02:19  mangeot
 * Added new search options for ReviewContributions
 *
 * Revision 1.3  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 */

package fr.imag.clips.papillon.presentation.deprecated;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

// Standard imports
import java.io.IOException;
import java.util.*;

//pour le dictionary
import fr.imag.clips.papillon.business.dictionary.*;

import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.user.*;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.transformation.*;


import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.presentation.PapillonBasePO;
import fr.imag.clips.papillon.presentation.EditEntry;
import fr.imag.clips.papillon.presentation.deprecated.AdminContributions;
import fr.imag.clips.papillon.presentation.MultilingualXHtmlTemplateFactory;

// Imported JAVA API for XML Parsing classes

// Imported TraX classes
import javax.xml.transform.TransformerConfigurationException;

/**
 * @deprecated Page is specific to an application. Refactoring is necessary.
 */
public class ReviewContributions extends PapillonBasePO {

	
    protected final static int MaxDisplayedEntries= 5;

	protected final static int STEP_DEFAULT = 1;
	protected final static int STEP_LOOKUP = 2;
	protected final static int STEP_VIEW = 3;
	protected final static int STEP_REMOVE = 4;
	protected final static int STEP_REVISE = 5;
	protected final static int STEP_VALIDATE = 6;
	protected final static int STEP_REMOVE_VALIDATED = 7;
	protected final static int STEP_RESET = 8;
	
	protected final static String ALL="*ALL*";
	protected final static String EditURL="EditEntry.po";
	protected final static String EditVolumeParameter= EditEntry.VolumeName_PARAMETER;
	protected final static String EditHandleParameter=EditEntry.EntryHandle_PARAMETER;
	protected final static String XML_FORMATTER = fr.imag.clips.papillon.business.transformation.XslTransformation.XML_FORMATTER; 

	protected final static String VIEW_CONTRIB_PARAMETER= AdminContributions.VIEW_CONTRIB_PARAMETER;
	protected final static String AnyContains_PARAMETER="AnyContains";
	protected final static String OFFSET_PARAMETER="OFFSET";
	protected final static String REMOVE_CONTRIB_PARAMETER="RemoveContrib";
	protected final static String REMOVE_VALIDATED_CONTRIB_PARAMETER="DeleteContrib";
	protected final static String REVISE_CONTRIB_PARAMETER="ReviseContrib";
	protected final static String VALIDATE_CONTRIB_PARAMETER="ValidateContrib";
	protected final static String HANDLE_PARAMETER=AdminContributions.HANDLE_PARAMETER;
	protected final static String VOLUME_PARAMETER=ReviewContributionsTmplXHTML.NAME_VOLUME;
    protected final static String FORMATTER_PARAMETER="formatter";
    protected final static String SORTBY_PARAMETER="SortBy";
	
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
        IOException,
        TransformerConfigurationException,
        org.xml.sax.SAXException,
        javax.xml.parsers.ParserConfigurationException,
        java.io.IOException,
        javax.xml.transform.TransformerException,
        ClassNotFoundException,
        PapillonBusinessException {
        
        // Création du contenu
        ReviewContributionsTmplXHTML content = (ReviewContributionsTmplXHTML) MultilingualXHtmlTemplateFactory.createTemplate("ReviewContributionsTmplXHTML", this.getComms(), this.getSessionData());

        HttpPresentationRequest req = this.getComms().request;
        
		// decoding the CGI arguments
		String queryString = "";
		
		String lookup = myGetParameter(content.NAME_LOOKUP);
		String reset = myGetParameter(content.NAME_RESET);
			String volume = myGetParameter(VOLUME_PARAMETER);
		// hidden arguments
			String contribid = myGetParameter(HANDLE_PARAMETER);
			String formatter = myGetParameter(FORMATTER_PARAMETER);
			String sortBy = myGetParameter(SORTBY_PARAMETER);
											
			if (volume!=null &&!volume.equals("")) {
				this.setPreference(VOLUME_PARAMETER,volume);
			}
			else {
				volume = this.getPreference(VOLUME_PARAMETER);
			}
			
			if (volume!=null && !volume.equals("")) {
				queryString += "&" + VOLUME_PARAMETER + "=" + volume;
			}
			
			//author
			String author = myGetParameter(content.NAME_AUTHOR);
			if (lookup !=null && !lookup.equals("")) {
				this.setPreference(content.NAME_AUTHOR,author);
			}
			else {
				author = this.getPreference(content.NAME_AUTHOR);
			}
			if (author !=null && !author.equals("")) {
				queryString += "&" + content.NAME_AUTHOR + "=" + author;
			}

			// reviewer
			String reviewer = myGetParameter(content.NAME_REVIEWER);
			if (lookup !=null && !lookup.equals("")) {
				this.setPreference(content.NAME_REVIEWER,reviewer);
			}
			else {
				reviewer =  this.getPreference(content.NAME_REVIEWER);
			}
			if (reviewer !=null && !reviewer.equals("")) {
				queryString += "&" + content.NAME_REVIEWER + "=" + reviewer;
			}

			// CreationDate
			String creationDate = myGetParameter(content.NAME_CreationDate);
			if (lookup !=null && !lookup.equals("")) {
				this.setPreference(content.NAME_CreationDate, creationDate);
			} else {
				creationDate = this.getPreference(content.NAME_CreationDate);
			}
			if (creationDate !=null && !creationDate.equals("")) {
				queryString += "&" + content.NAME_CreationDate + "=" + creationDate;
			}
			String creationDateStrategyString = myGetParameter(content.NAME_StrategyCreationDate);
			if (creationDateStrategyString != null && !creationDateStrategyString.equals("")) {
				this.setPreference(content.NAME_StrategyCreationDate, creationDateStrategyString);
			} else {
				creationDateStrategyString = this.getPreference(content.NAME_StrategyCreationDate);
			}
			int creationDateStrategy = IQuery.STRATEGY_NONE;
			if (null != creationDateStrategyString && !creationDateStrategyString.equals("")) {
				creationDateStrategy = Integer.parseInt(creationDateStrategyString);
			}
			if (creationDateStrategyString !=null && !creationDateStrategyString.equals("")) {
				queryString += "&" + content.NAME_StrategyCreationDate + "=" + creationDateStrategyString;
			}
			
			// ReviewDate
			String reviewDate = myGetParameter(content.NAME_ReviewDate);
			if (lookup !=null && !lookup.equals("")) {
				this.setPreference(content.NAME_ReviewDate, reviewDate);
			} else {
				reviewDate = this.getPreference(content.NAME_ReviewDate);
			}
			if (reviewDate !=null && !reviewDate.equals("")) {
				queryString += "&" + content.NAME_ReviewDate + "=" + reviewDate;
			}
			String reviewDateStrategyString = myGetParameter(content.NAME_StrategyReviewDate);
			if (reviewDateStrategyString != null && !reviewDateStrategyString.equals("")) {
				this.setPreference(content.NAME_StrategyReviewDate, reviewDateStrategyString);
			} else {
				reviewDateStrategyString = this.getPreference(content.NAME_StrategyReviewDate);
			}
			int reviewDateStrategy = IQuery.STRATEGY_NONE;
			if (null != reviewDateStrategyString && !reviewDateStrategyString.equals("")) {
				reviewDateStrategy = Integer.parseInt(reviewDateStrategyString);
			}
			if (reviewDateStrategyString !=null && !reviewDateStrategyString.equals("")) {
				queryString += "&" + content.NAME_StrategyReviewDate + "=" + reviewDateStrategyString;
			}
			
			// search1
			String search1 = myGetParameter(content.NAME_search1);
			if (search1 != null && !search1.equals("")) {
				this.setPreference(content.NAME_search1, search1);
			} else {
				search1 = this.getPreference(content.NAME_search1);
			}
			if (search1 !=null && !search1.equals("")) {
				queryString += "&" + content.NAME_search1 + "=" + search1;
			}
			String search1text = myGetParameter(content.NAME_search1text);
			if (search1text !=null && !search1text.equals("")) {
				queryString += "&" + content.NAME_search1text + "=" + search1text;
			}
			
			// search2
			String search2 = myGetParameter(content.NAME_search2);
			if (search2 != null && !search2.equals("")) {
				this.setPreference(content.NAME_search2, search2);
			} else {
				search2 = this.getPreference(content.NAME_search2);
			}
			if (search2 !=null && !search2.equals("")) {
				queryString += "&" + content.NAME_search2 + "=" + search2;
			}
			String search2text = myGetParameter(content.NAME_search2text);
			if (search2text !=null && !search2text.equals("")) {
				queryString += "&" + content.NAME_search2text + "=" + search2text;
			}
			
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
			
			// strategy1
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
			if (strategyString1 !=null && !strategyString1.equals("")) {
				queryString += "&" + content.NAME_Strategy1 + "=" + strategyString1;
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
			if (strategyString2 !=null && !strategyString2.equals("")) {
				queryString += "&" + content.NAME_Strategy2 + "=" + strategyString2;
			}
						
			int offset = 0;
			String offsetString = myGetParameter(OFFSET_PARAMETER);
			if (offsetString != null && !offsetString.equals("")) {
				offset = Integer.parseInt(offsetString);
			}
			queryString += "&" + OFFSET_PARAMETER + "=" + offset;
			
			// status
			String status = myGetParameter(content.NAME_STATUS);
			if (lookup!=null &&!lookup.equals("")) {
				this.setPreference(content.NAME_STATUS,status);
			}
			else {
				status =  this.getPreference(content.NAME_STATUS);
			}
			if (status !=null && !status.equals("")) {
				queryString += "&" + content.NAME_STATUS + "=" + status;
			}
			if (status != null && status.equals(ALL)) {
				status = null;
			}
			
			Vector myKeys = new Vector();
			Vector myClauses = new Vector();
			if (search1 !=null && !search1.equals("")  &&
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
			if (status !=null && !status.equals("")) {
				String[] key2 = new String[4];
				key2[0] = Volume.CDM_contributionStatus;
				key2[1] = Volume.DEFAULT_LANG;
				key2[2] = status;
				key2[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];			
				myKeys.add(key2);
			}
			else {
				String[] key2 = new String[4];
				key2[0] = Volume.CDM_contributionStatus;
				key2[1] = Volume.DEFAULT_LANG;
				key2[2] = VolumeEntry.VALIDATED_STATUS;
				key2[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];			
				myKeys.add(key2);			
				String[] key3 = new String[4];
				key3[0] = Volume.CDM_contributionStatus;
				key3[1] = Volume.DEFAULT_LANG;
				key3[2] = VolumeEntry.DELETED_STATUS;
				key3[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];			
				myKeys.add(key3);			
				String[] key4 = new String[4];
				key4[0] = Volume.CDM_contributionStatus;
				key4[1] = Volume.DEFAULT_LANG;
				key4[2] = VolumeEntry.REPLACED_STATUS;
				key4[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];			
				myKeys.add(key4);			
			}
			if (author !=null && !author.equals("")) {
				String[] authorKey = new String[4];
				authorKey[0] = Volume.CDM_contributionAuthor;
				authorKey[1] = Volume.DEFAULT_LANG;
				authorKey[2] = author;
				authorKey[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];			
				myKeys.add(authorKey);
			}
			if (reviewer !=null && !reviewer.equals("")) {
				String[] reviewerKey = new String[4];
				reviewerKey[0] = Volume.CDM_contributionReviewer;
				reviewerKey[1] = Volume.DEFAULT_LANG;
				reviewerKey[2] = reviewer;
				reviewerKey[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];			
				myKeys.add(reviewerKey);
			}
			if (reviewDate !=null && !reviewDate.equals("")) {
				String[] reviewDateKey = new String[4];
				reviewDateKey[0] = Volume.CDM_contributionReviewDate;
				reviewDateKey[1] = Volume.DEFAULT_LANG;
				reviewDateKey[2] = reviewDate;
				reviewDateKey[3] = IQuery.QueryBuilderStrategy[reviewDateStrategy+1];			
				myKeys.add(reviewDateKey);
			}
			if (creationDate !=null && !creationDate.equals("")) {
				String[] creationDateKey = new String[4];
				creationDateKey[0] = Volume.CDM_contributionCreationDate;
				creationDateKey[1] = Volume.DEFAULT_LANG;
				creationDateKey[2] = creationDate;
				creationDateKey[3] = IQuery.QueryBuilderStrategy[creationDateStrategy+1];			
				myKeys.add(creationDateKey);
			}
		

		int step = STEP_DEFAULT;
		if (null != lookup) {
			step = STEP_LOOKUP;
		}
		else if (null != contribid && null != myGetParameter(VIEW_CONTRIB_PARAMETER)) {
			step = STEP_VIEW;
		}
		else if (null != contribid && null != myGetParameter(REMOVE_CONTRIB_PARAMETER)) {
			step = STEP_REMOVE;
		}
		else if (null != contribid && null != myGetParameter(REVISE_CONTRIB_PARAMETER)) {
			step = STEP_REVISE;
		}
		else if (null != contribid && null != myGetParameter(VALIDATE_CONTRIB_PARAMETER)) {
			step = STEP_VALIDATE;
		}	
		else if (null != contribid && null != myGetParameter(REMOVE_VALIDATED_CONTRIB_PARAMETER)) {
			step = STEP_REMOVE_VALIDATED;
		}		
		else if (null != reset) {
			step = STEP_RESET;
		}		
		
		String userMessage = null;

		switch (step) {
			case STEP_DEFAULT:
				//addContributions(volume, author, headword, strategy, sortBy, queryString, offset);
				break;
			case STEP_LOOKUP:
				addContributions(content, volume, myKeys, myClauses, sortBy, queryString, offset);
				break;
			case STEP_VIEW:
				addContribution(content, volume, contribid, formatter, queryString);
				break;
			case STEP_REMOVE:
				if (contribid !=null && !contribid.equals("")) {
					VolumeEntry myContrib = VolumeEntriesFactory.findEntryByHandle(volume, contribid);
					if (null != myContrib && !myContrib.isEmpty()) {
						userMessage = "Contribution " +  myContrib.getHandle() + " / " +
						myContrib.getHeadword() + " removed...";
						myContrib.setDeleted(this.getUser());
					}
				}
				addContributions(content, volume, myKeys, myClauses, sortBy, queryString, offset);
				break;
			case STEP_REMOVE_VALIDATED:
				if (contribid !=null && !contribid.equals("") &&
					this.getUser().isValidator()) {
					VolumeEntry myContrib = VolumeEntriesFactory.findEntryByHandle(volume, contribid);
					if (null != myContrib && !myContrib.isEmpty()) {
						userMessage = "Validated contribution " +  myContrib.getHandle() + " / " +
						myContrib.getHeadword() + " deleted...";
						myContrib.setReplaced(this.getUser());
					}
					else {
						PapillonLogger.writeDebugMsg("Remove null contrib!");	
					}
				}
				addContributions(content, volume, myKeys, myClauses, sortBy, queryString, offset);
				break;
			case STEP_REVISE:
				if (contribid !=null && !contribid.equals("") && this.getUser().isSpecialist()) {
					VolumeEntry myContrib = VolumeEntriesFactory.findEntryByHandle(volume, contribid);
					if (null != myContrib && !myContrib.isEmpty()) {
						myContrib.setReviewed(this.getUser());
						userMessage = "Contribution " +  myContrib.getHandle() + " / " +
						myContrib.getHeadword() + " reviewed";
					}
				}
				addContributions(content, volume, myKeys, myClauses, sortBy, queryString, offset);
				break;
			case STEP_VALIDATE:
				if (contribid !=null && !contribid.equals("") && this.getUser().isValidator()) {
					VolumeEntry myContrib = VolumeEntriesFactory.findEntryByHandle(volume, contribid);
					if (null != myContrib && !myContrib.isEmpty()) {
						myContrib.setValidated(this.getUser());
						userMessage = "Contribution " +  myContrib.getContributionId() + " / " +
							myContrib.getHeadword() + " integrated in the dictionary...";
						String originalContribId = myContrib.getOriginalContributionId();
						if (originalContribId != null && !originalContribId.equals("")) {
							VolumeEntry oldContrib = VolumeEntriesFactory.findEntryByContributionId(volume, originalContribId);
							if (null != oldContrib && !oldContrib.isEmpty()) {
								oldContrib.setReplaced(this.getUser());
								userMessage += " Old contribution " + oldContrib.getContributionId() + " / " +
									oldContrib.getHeadword() + " replaced.";
							}
						}
					}
				}
				addContributions(content, volume, myKeys, myClauses, sortBy, queryString, offset);
				break;
			case STEP_RESET:
				this.resetPreferences();
				break;
			default:
				break;
			}	
			
            if (null != userMessage && !userMessage.equals("")){
                this.getSessionData().writeUserMessage(userMessage);
                PapillonLogger.writeDebugMsg(userMessage);
            }
		if (step != STEP_RESET) {
			addConsultForm(content, volume, status, author, reviewer, 
						   creationDate, creationDateStrategyString, 
						   reviewDate, reviewDateStrategyString, 
						   search1, search1text, strategyString1, 
						   search2, search2text, strategyString2);
		}
		else {
			addConsultForm(content, null, null, null, null, 
						   null, null, 
						   null, null, 
						   null, null, null, 
						   null, null, null);
		}

        removeTemplateRows(content);
        
        //On rend le contenu correct
        return content.getElementFormulaire();
    }
        protected void addConsultForm(ReviewContributionsTmplXHTML content, 
			String volume, String status, String author, String reviewer, 
			String creationDate, String creationDateStrategyString, 
			String reviewDate, String reviewDateStrategyString, 
			String search1, String search1text, String strategyString1, 
			String search2, String search2text, String strategyString2)
			throws fr.imag.clips.papillon.business.PapillonBusinessException, 
                HttpPresentationException,
        java.io.UnsupportedEncodingException {
                    
        // Adding the user name
		User user = getUser();

		if (null != user && !user.isEmpty()) {
			content.setTextUserName(user.getName());
		}
                    
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
			String schema = myVolume.getXmlSchema();
			if (schema != null && !schema.equals("")) {

            volumeOptionTemplate.setValue(myVolume.getName());
            volumeOptionTemplate.setLabel(myVolume.getName());
            // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux 
            // specs W3C.
            volumeOptionTemplate.setSelected(myVolume.getName().equals(volume));
            volumeTextTemplate.setData(myVolume.getName());
            volumeSelect.appendChild(volumeOptionTemplate.cloneNode(true));
			}
		}
        volumeSelect.removeChild(volumeOptionTemplate);

		// author
		XHTMLInputElement authorInput = content.getElementAUTHOR();
		authorInput.setValue(author);
		
		// reviewer
		XHTMLInputElement reviewerInput = content.getElementREVIEWER();
		reviewerInput.setValue(reviewer);

		// creationDate
		XHTMLInputElement creationDateInput = content.getElementInputCreationDate();
		creationDateInput.setValue(creationDate);
		this.setSelected(content.getElementStrategyCreationDate(), creationDateStrategyString);

		// reviewDate
		XHTMLInputElement reviewDateInput = content.getElementInputReviewDate();
		reviewDateInput.setValue(reviewDate);
		this.setSelected(content.getElementStrategyReviewDate(), reviewDateStrategyString);
		
		
			// strategy
			if (!this.IsClientWithLabelDisplayProblems()) {
				this.setUnicodeLabels(content.getElementStrategy1());
			}
			this.setSelected(content.getElementStrategy1(), strategyString1);
			
			if (!this.IsClientWithLabelDisplayProblems()) {
				this.setUnicodeLabels(content.getElementStrategy2());
			}
			this.setSelected(content.getElementStrategy2(), strategyString2);
						
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
			
			// status
			XHTMLSelectElement statusSelect = (XHTMLSelectElement) content.getElementSTATUS();
			this.setSelected(statusSelect,status);
    }

	protected void addContribution(ReviewContributionsTmplXHTML content, String volumeString, String entryHandle, String formatter, String queryString) 
		throws PapillonBusinessException,
		java.io.UnsupportedEncodingException,
		HttpPresentationException,
		java.io.IOException {
			java.util.Collection ContribCollection = displayEntry(content, volumeString, entryHandle, formatter);
			addEntryTable(content, ContribCollection, queryString, 0);
	}
	
    protected void addContributions(ReviewContributionsTmplXHTML content, String volume, Vector Keys1, Vector Keys2, String sortBy, String queryString, int offset)
        throws PapillonBusinessException,
        ClassNotFoundException,
        HttpPresentationException,
        IOException,
        TransformerConfigurationException,
        org.xml.sax.SAXException,
        javax.xml.parsers.ParserConfigurationException,
        javax.xml.transform.TransformerException {
            // FIXME: fix the limit parameter
            Vector ContribVector = VolumeEntriesFactory.getVolumeNameEntriesVector(volume, Keys1, Keys2, null, offset, 0);
            if (null != ContribVector) {
				if (sortBy !=null && !sortBy.equals("")) {
					VolumeEntriesFactory.sort(ContribVector, sortBy);
				}
			}
			addContributions(content, ContribVector, queryString, offset);
		}

	protected void addContributions(ReviewContributionsTmplXHTML content, Collection EntryCollection, String queryString, int offset)
		throws PapillonBusinessException,
		ClassNotFoundException,
		HttpPresentationException,
		IOException,
		TransformerConfigurationException,
		org.xml.sax.SAXException,
		javax.xml.parsers.ParserConfigurationException,
		javax.xml.transform.TransformerException {

		// If there are too much entries ie > MaxDisplayedEntries,
        // we display a table of entries instead of the entries
			if (null != EntryCollection && EntryCollection.size()>0) {
				addEntryTable(content, EntryCollection, queryString, offset);
				if (EntryCollection.size() < MaxDisplayedEntries) {
					for(Iterator entriesIterator = EntryCollection.iterator(); entriesIterator.hasNext();) {
						IAnswer myAnswer = (VolumeEntry)entriesIterator.next();
						addElement(content, XslTransformation.applyXslSheets(myAnswer));
					}
				}
			}
		}
   
	protected java.util.Collection displayEntry (ReviewContributionsTmplXHTML content, String volumeName, String handle, String formatter)
		throws PapillonBusinessException,
		java.io.UnsupportedEncodingException,
		HttpPresentationException,
		java.io.IOException {
			
			Volume myVolume = VolumesFactory.getVolumeByName(volumeName);
			java.util.Vector myVector = new Vector();
			Collection targets = null;
			if (myVolume != null && !myVolume.isEmpty()) {
				targets = myVolume.getTargetLanguagesArray();
			}
			java.util.Collection EntryCollection = DictionariesFactory.findAnswerAndTranslations(volumeName, handle, targets, this.getUser());
			
			if (EntryCollection != null && EntryCollection.size()>0) {
				QueryResult myQueryResult = (QueryResult) EntryCollection.iterator().next();
				VolumeEntry myEntry = myQueryResult.getSourceEntry();
				myVector.add(myEntry);
				// get the apropriate transformer.
				fr.imag.clips.papillon.business.transformation.ResultFormatter rf = fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.getFormatter(myQueryResult, formatter, fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XHTML_DIALECT, null);
				
				
				Element myXhtmlElt = (Element)rf.getFormattedResult(myQueryResult, this.getUser());
				addElement(content, myXhtmlElt);
			}
			return (java.util.Collection) myVector;
		}
	

    protected void addEntryTable (ReviewContributionsTmplXHTML content, Collection EntryCollection, String queryString, int offset)
        throws PapillonBusinessException,
        java.io.UnsupportedEncodingException {

            // On récupère les éléments du layout
			String lookupParam = content.NAME_LOOKUP + "=" + "lookup";
			String sortHref = "&" + lookupParam + queryString;
			
			XHTMLAnchorElement authorHeader = content.getElementAuthorHeaderAnchor();
			authorHeader.setHref(authorHeader.getHref() + sortHref);
			XHTMLAnchorElement headwordHeader = content.getElementHeadwordHeaderAnchor();
			headwordHeader.setHref(headwordHeader.getHref()+sortHref);
			XHTMLAnchorElement creationdateHeader = content.getElementCreationdateHeaderAnchor();
			creationdateHeader.setHref(creationdateHeader.getHref()+sortHref);
			XHTMLAnchorElement statusHeader = content.getElementStatusHeaderAnchor();
			statusHeader.setHref(statusHeader.getHref()+sortHref);
			XHTMLAnchorElement reviewerHeader = content.getElementReviewerHeaderAnchor();
			reviewerHeader.setHref(reviewerHeader.getHref()+sortHref);
			XHTMLAnchorElement reviewdateHeader = content.getElementReviewdateHeaderAnchor();
			reviewdateHeader.setHref(reviewdateHeader.getHref()+sortHref);
			
            XHTMLTableRowElement entryListRow = content.getElementEntryListRow();
            XHTMLElement theDate = content.getElementCreationDate();
            XHTMLAnchorElement viewContribAnchor = content.getElementViewContribAnchor();
            XHTMLAnchorElement viewXmlAnchor = content.getElementViewXmlAnchor();
            XHTMLAnchorElement editContribAnchor = content.getElementEditContribAnchor();
            XHTMLAnchorElement removeContribAnchor = content.getElementRemoveContribAnchor();
            XHTMLAnchorElement reviseContribAnchor = content.getElementReviseContribAnchor();
            XHTMLAnchorElement validateContribAnchor = content.getElementValidateContribAnchor();
            XHTMLElement removeMessageElement = content.getElementRemoveMessage();

			XHTMLAnchorElement previousEntriesAnchor = content.getElementPreviousEntriesAnchor();
			XHTMLAnchorElement nextEntriesAnchor = content.getElementNextEntriesAnchor();

            theDate.removeAttribute("id");
            viewContribAnchor.removeAttribute("id");
            viewXmlAnchor.removeAttribute("id");
            editContribAnchor.removeAttribute("id");
            removeContribAnchor.removeAttribute("id");
            reviseContribAnchor.removeAttribute("id");
            validateContribAnchor.removeAttribute("id");
            removeMessageElement.removeAttribute("id");
			
			
			String href =  this.getUrl() + "?" + lookupParam + queryString; // + "&" + OFFSET_PARAMETER + "=";

		if (offset >= DictionariesFactory.MaxRetrievedEntries) {
			int prevOffset = offset-DictionariesFactory.MaxRetrievedEntries;
			String oldOffsetString = OFFSET_PARAMETER + "=" + offset;
			String newOffsetString = OFFSET_PARAMETER + "=" + prevOffset;
			previousEntriesAnchor.setHref(href.replaceFirst(oldOffsetString, newOffsetString));
		}
		else {
			previousEntriesAnchor.setHref("");
			content.setTextPreviousEntriesAnchor("");			
		}
		int nextOffset = offset+DictionariesFactory.MaxRetrievedEntries;
		String oldOffsetString = OFFSET_PARAMETER + "=" +  offset;
		String newOffsetString = OFFSET_PARAMETER + "=" + nextOffset;
		nextEntriesAnchor.setHref(href.replaceFirst(oldOffsetString, newOffsetString));

			String removeMessage = Utility.getText(removeMessageElement);

            // On récupère le noeud contenant la table...
            Node entryTable = entryListRow.getParentNode();
            if (null != EntryCollection && EntryCollection.size()>0) {
				PapillonLogger.writeDebugMsg("addEntryTable " + EntryCollection.size());
				content.setTextContributionsCount("" + EntryCollection.size());
				for(Iterator entriesIterator = EntryCollection.iterator(); entriesIterator.hasNext();) {
                    VolumeEntry myContrib = (VolumeEntry) entriesIterator.next();
					if (myContrib !=null && !myContrib.isEmpty()) {
                       content.setTextViewContribText(myContrib.getCompleteHeadword());
                        viewContribAnchor.setHref(this.getUrl() + "?"
												  + VIEW_CONTRIB_PARAMETER + "=" + VIEW_CONTRIB_PARAMETER
                                                  + "&" + HANDLE_PARAMETER + "=" + myContrib.getHandle()
                                                  + "&" + VOLUME_PARAMETER + "=" + myContrib.getVolumeName());

                        viewXmlAnchor.setHref(this.getUrl() + "?"
											  + VIEW_CONTRIB_PARAMETER + "=" + VIEW_CONTRIB_PARAMETER
                                              + "&" + HANDLE_PARAMETER + "=" + myContrib.getHandle()
											  + "&" + VOLUME_PARAMETER + "=" + myContrib.getVolumeName()
						                      + "&" + FORMATTER_PARAMETER + "="  + XML_FORMATTER);

                        content.setTextVolumeName(myContrib.getVolumeName());
                        content.setTextAuthor(myContrib.getAuthor());
                        content.setTextCreationDate(Utility.PapillonShortDateFormat.format(myContrib.getCreationDate()));
                        content.setTextStatus(myContrib.getStatus());
												
						content.setTextEditMessage("");
						
						// IsNewEntry
						boolean isNewEntry = myContrib.getOriginalContributionId()==null || myContrib.getOriginalContributionId().equals("");
						if (isNewEntry) {
							entryListRow.setAttribute("class","");
						}
						else {
							entryListRow.setAttribute("class","copyentry");
						}
						
						// remove contrib
						if (this.getUser().getLogin().equals(myContrib.getAuthor())
							|| this.getUser().isInNormalGroups(myContrib.getGroups())
							|| this.getUser().isValidator()) {
							removeContribAnchor.setHref(this.getUrl() + "?"
								   + REMOVE_CONTRIB_PARAMETER + "=on"
								   + "&" + HANDLE_PARAMETER + "=" + myContrib.getHandle()
								   + queryString);
							content.setTextRemoveMessage(removeMessage);
						}
						else {
							content.setTextRemoveMessage("");
						}

						// revisions on the contrib
						if (myContrib.getReviewer()!=null && myContrib.getReviewDate()!=null) {
							content.setTextReviewer(myContrib.getReviewer());
							content.setTextReviewDate(Utility.PapillonShortDateFormat.format(myContrib.getReviewDate()));
						}
						else {
							content.setTextReviewer("");
							content.setTextReviewDate("");
						}
						// action on contrib
						if (myContrib.getStatus()!=null) {
							if (myContrib.getStatus().equals(VolumeEntry.FINISHED_STATUS)
								&& this.getUser().isSpecialist() 
								&& this.getUser().isInNormalGroups(myContrib.getGroups())) {
								reviseContribAnchor.setHref(this.getUrl() + "?"
									 + REVISE_CONTRIB_PARAMETER + "=" + REVISE_CONTRIB_PARAMETER
									 + "&" + HANDLE_PARAMETER + "=" + myContrib.getHandle()
								     + queryString);
								reviseContribAnchor.setAttribute("class","");	 
							}
							else {
								reviseContribAnchor.setAttribute("class","hidden");	 
							}
							if (myContrib.getStatus().equals(VolumeEntry.REVIEWED_STATUS)
								&& this.getUser().isValidator()) {
								validateContribAnchor.setHref(this.getUrl() + "?"
									 + VALIDATE_CONTRIB_PARAMETER + "=" + VALIDATE_CONTRIB_PARAMETER
									 + "&" + HANDLE_PARAMETER + "=" + myContrib.getHandle()
								     + queryString);
								validateContribAnchor.removeAttribute("class");	 
							}
							else {
								validateContribAnchor.setAttribute("class","hidden");	 
							}
						}
                        
                        XHTMLElement clone = (XHTMLElement)entryListRow.cloneNode(true);
                        //      we have to take off the id attribute because we did not take it off the original
                        clone.removeAttribute("id");
                        entryTable.appendChild(clone);
                    }
					}
                }
            }

    protected void addElement (ReviewContributionsTmplXHTML content, Element element)
        throws HttpPresentationException,
        PapillonBusinessException,
        java.io.IOException {

            //for the entry content
            XHTMLTableRowElement originalEntryRow = content.getElementEntryRow();
            Node entryTable=originalEntryRow.getParentNode();
            //for the entry content
            XHTMLTableRowElement entryRow = (XHTMLTableRowElement)originalEntryRow.cloneNode(true);

            //for the lexie content
            XHTMLTableCellElement entryCell= (XHTMLTableCellElement)entryRow.getFirstChild();

            entryRow.removeAttribute("id");
            entryCell.removeAttribute("id");

            entryCell.appendChild(content.importNode(element, true));
            entryTable.appendChild(entryRow);
            // Don't remove the original node in order to add more entries ...
            //entryTable.removeChild(entryRow);
        }

    
    protected void removeTemplateRows(ReviewContributionsTmplXHTML content) {
        // EntryListRow
        Element entryListRow = content.getElementEntryListRow();
        Node entryListRowParent = entryListRow.getParentNode();
        entryListRowParent.removeChild(entryListRow);

        // EntryRow
        Element entryRow = content.getElementEntryRow();
        Node entryRowParent = entryRow.getParentNode();
        entryRowParent.removeChild(entryRow);
    }    
    
}
