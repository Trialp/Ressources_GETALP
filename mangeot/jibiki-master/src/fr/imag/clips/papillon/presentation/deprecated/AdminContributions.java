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
 * Revision 1.1.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.35  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.34  2006/03/06 10:16:45  mangeot
 * MM: I do not delete any contribution, even not finished.
 * I tag them "deleted"
 *
 * Revision 1.33  2006/03/06 10:11:47  mangeot
 * *** empty log message ***
 *
 * Revision 1.32  2006/03/05 11:54:18  mangeot
 * *** empty log message ***
 *
 * Revision 1.31  2006/03/02 11:56:00  mangeot
 * *** empty log message ***
 *
 * Revision 1.30  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.29  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.28  2006/02/24 22:59:48  mangeot
 * *** empty log message ***
 *
 * Revision 1.22  2006/01/25 17:06:09  mangeot
 * Fixed the sort buttons
 *
 * Revision 1.21.2.1  2006/01/24 13:39:49  fbrunet
 * Modification view management
 * Modification LexALP postprocessing
 *
 * Revision 1.21  2005/08/17 12:58:16  mangeot
 * Fixed a bug when creating an entry from an existing one.
 * From now on, the entry id is the same.
 * Added the links into ReviewContributions.java
 *
 * Revision 1.20  2005/08/17 09:29:54  mangeot
 * Added links in AdminContributions
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
 * Revision 1.16  2005/07/30 16:23:07  mangeot
 * *** empty log message ***
 *
 * Revision 1.15  2005/07/28 16:40:33  mangeot
 * *** empty log message ***
 *
 * Revision 1.14  2005/07/21 15:09:20  mangeot
 * Bug fixes and corrections mainly for the GDEF project
 *
 * Revision 1.13  2005/07/16 13:43:15  mangeot
 * Added do not search for replaced entries
 *
 * Revision 1.12  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 * Revision 1.11  2005/06/23 09:48:17  mangeot
 * Bug fix in xpath completion and creation-date cdm element
 *
 * Revision 1.10  2005/06/15 16:48:28  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.9  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.8.4.15  2005/06/15 10:08:06  mangeot
 * Removed the AND/OR connector, now only AND criteria can be added for dict lookup
 *
 * Revision 1.8.4.14  2005/05/24 11:15:48  mangeot
 * Bug fixes in sort
 *
 * Revision 1.8.4.13  2005/05/19 09:43:26  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.12  2005/05/14 11:56:28  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.11  2005/05/13 11:52:13  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.10  2005/05/13 11:20:42  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.9  2005/05/13 11:17:07  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.8  2005/04/30 09:11:20  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.7  2005/04/29 18:43:13  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.6  2005/04/29 18:38:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.5  2005/04/29 18:29:59  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.4  2005/04/29 17:30:30  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.3  2005/04/29 16:49:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.2  2005/04/29 16:42:36  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.8  2005/04/15 12:08:30  mangeot
 * *** empty log message ***
 *
 * Revision 1.7  2005/04/15 11:38:05  mangeot
 * Fixed a bug, not using entryHandle from contributions table any more
 *
 * Revision 1.6  2005/04/14 09:21:02  mangeot
 * Changed redirections after marked finish and save a contribution
 *
 * Revision 1.5  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.4  2005/04/11 08:01:02  fbrunet
 * Passage en xhtml des ressources Papillon.
 *
 * Revision 1.3.2.2  2005/04/09 14:51:47  mangeot
 * Added more consult options for AdminContributions page
 *
 * Revision 1.3.2.1  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
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
 *
 *-----------------------------------------------
 * Papillon AdminContributions page.
 */

package fr.imag.clips.papillon.presentation.deprecated;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

import fr.imag.clips.papillon.business.message.MessageDBLoader;

// Standard imports
import java.io.IOException;
import java.util.Vector;
import java.util.Iterator;
import java.text.DateFormat;
import java.io.*;


//pour le dictionary
import fr.imag.clips.papillon.business.dictionary.*;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.user.*;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.xsl.*;


import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.presentation.PapillonBasePO;
import fr.imag.clips.papillon.presentation.EditEntry;
import fr.imag.clips.papillon.presentation.MultilingualXHtmlTemplateFactory;

// Imported JAVA API for XML Parsing classes
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

// Imported TraX classes
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.stream.StreamSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.dom.DOMResult;

/**
 * @deprecated Page is specific to an application. Needs refactoring.
 */
public class AdminContributions extends PapillonBasePO {
	
    protected final static int MaxDisplayedEntries= 5;
	
	protected final static int STEP_DEFAULT = 1;
	protected final static int STEP_LOOKUP = 2;
	protected final static int STEP_VIEW = 3;
	protected final static int STEP_REMOVE = 4;
	protected final static int STEP_MARK_FINISHED = 7;
	protected final static int STEP_RESET = 8;	
	
	protected final static String EditURL="EditEntry.po";
	protected final static String EditVolumeParameter= EditEntry.VolumeName_PARAMETER;
	protected final static String EditHandleParameter=EditEntry.EntryHandle_PARAMETER;
	protected final static String XML_FORMATTER = fr.imag.clips.papillon.business.transformation.XslTransformation.XML_FORMATTER; 
	
	protected final static String ALL="*ALL*";
	protected final static String AnyContains_PARAMETER="AnyContains";
	protected final static String VIEW_CONTRIB_PARAMETER="ViewContrib";
    protected final static String FORMATTER_PARAMETER="formatter";
	protected final static String OFFSET_PARAMETER="OFFSET";
	protected final static String REMOVE_CONTRIB_PARAMETER="RemoveContrib";
	protected final static String MARK_FINISHED_PARAMETER="MarkFinished";
	protected final static String VOLUME_PARAMETER = AdminContributionsTmplXHTML.NAME_VOLUME;
	protected final static String LOOKUP_PARAMETER = AdminContributionsTmplXHTML.NAME_LOOKUP;
	protected final static String HANDLE_PARAMETER = "HANDLE";
    protected final static String SORTBY_PARAMETER = "SortBy";
    	
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
			AdminContributionsTmplXHTML content = (AdminContributionsTmplXHTML) MultilingualXHtmlTemplateFactory.createTemplate("AdminContributionsTmplXHTML", this.getComms(), this.getSessionData());
			
			HttpPresentationRequest req = this.getComms().request;
						
			String URL = null;
			String lookup = myGetParameter(LOOKUP_PARAMETER);
			String reset = myGetParameter(content.NAME_RESET);
			String volumeString = myGetParameter(VOLUME_PARAMETER);
			String entryid = myGetParameter(HANDLE_PARAMETER);
			String formatter = myGetParameter(FORMATTER_PARAMETER);
			String sortBy = myGetParameter(SORTBY_PARAMETER);
			
			if (volumeString!=null &&!volumeString.equals("")) {
				this.setPreference(VOLUME_PARAMETER,volumeString);
			}
			else {
				volumeString = this.getPreference(VOLUME_PARAMETER);
			}
			
			String queryString = "";
			if (volumeString!=null && !volumeString.equals("")) {
				queryString += "&" + VOLUME_PARAMETER + "=" + volumeString;
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
						
			//offset
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
			if (creationDate !=null && !creationDate.equals("")) {
				String[] creationDateKey = new String[4];
				creationDateKey[0] = Volume.CDM_contributionCreationDate;
				creationDateKey[1] = Volume.DEFAULT_LANG;
				creationDateKey[2] = creationDate;
				creationDateKey[3] = IQuery.QueryBuilderStrategy[creationDateStrategy+1];			
				myKeys.add(creationDateKey);
			}
			if (reviewDate !=null && !reviewDate.equals("")) {
				String[] reviewDateKey = new String[4];
				reviewDateKey[0] = Volume.CDM_contributionReviewDate;
				reviewDateKey[1] = Volume.DEFAULT_LANG;
				reviewDateKey[2] = reviewDate;
				reviewDateKey[3] = IQuery.QueryBuilderStrategy[reviewDateStrategy+1];			
				myKeys.add(reviewDateKey);
			}
			String[] authorKey = new String[4];
			authorKey[0] = Volume.CDM_contributionAuthor;
			authorKey[1] = Volume.DEFAULT_LANG;
			authorKey[2] = this.getUser().getLogin();
			authorKey[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];			
			myKeys.add(authorKey);
			
			int step = STEP_DEFAULT;
			if (null != lookup) {
				step = STEP_LOOKUP;
			}
			else if (null != entryid && null != myGetParameter(VIEW_CONTRIB_PARAMETER)) {
				step = STEP_VIEW;
			}
			else if (null != entryid && null != myGetParameter(REMOVE_CONTRIB_PARAMETER)) {
				step = STEP_REMOVE;
			}
			else if (null != entryid && null != myGetParameter(MARK_FINISHED_PARAMETER)) {
				step = STEP_MARK_FINISHED;
			}		
			else if (null != reset) {
				step = STEP_RESET;
			}		
			
			String userMessage = null;
			switch (step) {
				case STEP_DEFAULT:
					addConsultForm(content, volumeString, creationDate, creationDateStrategyString, reviewDate, reviewDateStrategyString, search1, search1text, strategyString1, search2, search2text, strategyString2, status);
					break;
				case STEP_LOOKUP:
					addContributions(content, volumeString, this.getUser(), myKeys, myClauses, anyContains, offset, sortBy, queryString);
					addConsultForm(content, volumeString, creationDate, creationDateStrategyString, reviewDate, reviewDateStrategyString, search1, search1text, strategyString1, search2, search2text, strategyString2, status);
					break;
				case STEP_VIEW:
					addContribution(content, volumeString, entryid, formatter, queryString);
					addConsultForm(content, volumeString, creationDate, creationDateStrategyString, reviewDate, reviewDateStrategyString, search1, search1text, strategyString1, search2, search2text, strategyString2, status);
					break;
				case STEP_MARK_FINISHED:
					if (entryid !=null && !entryid.equals("")) {
						VolumeEntry myContrib = VolumeEntriesFactory.findEntryByHandle(volumeString, entryid);
						if (null != myContrib && !myContrib.isEmpty()) {
							myContrib.setFinished(this.getUser());
							entryid = null;
							userMessage = "Contribution " +  myContrib.getHandle() + " / " +
								myContrib.getHeadword() + " finished";
						}
					}
					addContributions(content, volumeString, this.getUser(), myKeys, myClauses, anyContains, offset, sortBy, queryString);
					addConsultForm(content, volumeString, creationDate, creationDateStrategyString, reviewDate, reviewDateStrategyString, search1, search1text, strategyString1, search2, search2text, strategyString2, status);
					break;
				case STEP_REMOVE:
					VolumeEntry myContrib = VolumeEntriesFactory.findEntryByHandle(volumeString, entryid);
					if (null != myContrib && !myContrib.isEmpty()) {
						myContrib.setDeleted(this.getUser());
						entryid=null;
						userMessage = "Contribution " +  myContrib.getHandle() + " / " +
							myContrib.getHeadword() + " removed...";
					}
					addContributions(content, volumeString, this.getUser(), myKeys, myClauses, anyContains, offset, sortBy, queryString);
					addConsultForm(content, volumeString, creationDate, creationDateStrategyString, reviewDate, reviewDateStrategyString, search1, search1text, strategyString1, search2, search2text, strategyString2, status);
					break;
				case STEP_RESET:
					this.resetPreferences();
					addConsultForm(content, null, null, null, null, null, null, null, null, null, null, null, null);
					break;
				default:
					break;
			}	
			
			if (null != userMessage && !userMessage.equals("")){
				this.getSessionData().writeUserMessage(userMessage);
				PapillonLogger.writeDebugMsg(userMessage);
			}
			removeTemplateRows(content);
			
			//On rend le contenu correct
			return content.getElementFormulaire();
		}
	
	protected void addConsultForm(AdminContributionsTmplXHTML content,
								  String volume, 
								  String creationDate, String creationDateStrategyString,
								  String reviewDate, String reviewDateStrategyString,
								  String search1, String search1text, 
								  String strategy1, 
								  String search2, String search2text,
								  String strategy2, String status)
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
			
			XHTMLSelectElement statusSelect = (XHTMLSelectElement) content.getElementSTATUS();
			this.setSelected(statusSelect,status);
			
		}
	
	protected void addContribution(AdminContributionsTmplXHTML content, 
								   String volumeString, String entryHandle, 
								   String formatter, String queryString) 
		throws PapillonBusinessException,
		java.io.UnsupportedEncodingException,
		HttpPresentationException,
		java.io.IOException {
		java.util.Collection ContribCollection = displayEntry(content, volumeString, entryHandle, formatter);
		addEntryTable(content, ContribCollection, queryString);
	}
	
    protected void addContributions(AdminContributionsTmplXHTML content, String volumeString, User myUser, Vector Keys1, Vector Keys2, String anyContains, int offset, String sortBy, String queryString)
        throws PapillonBusinessException,
        ClassNotFoundException,
        HttpPresentationException,
        IOException,
        TransformerConfigurationException,
        org.xml.sax.SAXException,
        javax.xml.parsers.ParserConfigurationException,
        javax.xml.transform.TransformerException {
			
			Vector ContribVector = VolumeEntriesFactory.getVolumeNameEntriesVector(volumeString, Keys1, Keys2, anyContains, offset, 0);
			// If there are too much entries ie > MaxDisplayedEntries,
			// we display a table of entries instead of the entries
			if (null != ContribVector) {
				if (sortBy !=null && !sortBy.equals("")) {
					VolumeEntriesFactory.sort(ContribVector, sortBy);
				}
				addEntryTable(content, (java.util.Collection) ContribVector, queryString);
				if (ContribVector.size() < MaxDisplayedEntries) {
					for(int i = 0; i < ContribVector.size(); i++) {
						IAnswer myAnswer = (IAnswer)ContribVector.get(i);
						if (myAnswer!=null && !myAnswer.isEmpty()) {
							addElement(content, XslTransformation.applyXslSheets(myAnswer));
						}
					}
				}
			}
		}
    
	
    protected void addEntryTable (AdminContributionsTmplXHTML content, java.util.Collection ContribCollection, String queryString)
        throws PapillonBusinessException,
        java.io.UnsupportedEncodingException {
			
            // On récupère les éléments du layout
			// the headers
			XHTMLAnchorElement headwordHeader = content.getElementHeadwordHeaderAnchor();
			headwordHeader.setHref(headwordHeader.getHref()+queryString);
			XHTMLAnchorElement creationdateHeader = content.getElementCreationdateHeaderAnchor();
			creationdateHeader.setHref(creationdateHeader.getHref()+queryString);
			XHTMLAnchorElement statusHeader = content.getElementStatusHeaderAnchor();
			statusHeader.setHref(statusHeader.getHref()+queryString);
			XHTMLAnchorElement newentryHeader = content.getElementNewentryHeaderAnchor();
			newentryHeader.setHref(newentryHeader.getHref()+queryString);
			
			
            XHTMLTableRowElement entryListRow = content.getElementEntryListRow();
			
			// View Contrib
            XHTMLAnchorElement viewContribAnchor = content.getElementViewContribAnchor();
            viewContribAnchor.removeAttribute("id");
            content.getElementViewContribText().removeAttribute("id");
			
			// EntryId
			content.getElementEntryId().removeAttribute("id");
												
			// Creation date
			content.getElementCreationDate().removeAttribute("id");
												
			// IsNewEntry
			content.getElementIsNewEntry().removeAttribute("id");
			
			// Status
			content.getElementStatus().removeAttribute("id");
			
			//Mark finished
            XHTMLAnchorElement markFinishedAnchor = content.getElementMarkFinishedAnchor();
            markFinishedAnchor.removeAttribute("id");
			
			//View XML
            XHTMLAnchorElement viewXmlAnchor = content.getElementViewXmlAnchor();
            viewXmlAnchor.removeAttribute("id");
			
			// Edit contrib
            XHTMLAnchorElement editContribAnchor = content.getElementEditContribAnchor();
            editContribAnchor.removeAttribute("id");
			
			// remove contrib
            XHTMLAnchorElement removeContribAnchor = content.getElementRemoveContribAnchor();
            removeContribAnchor.removeAttribute("id");
			
			
            // On récupère le noeud contenant la table...
            Node entryTable = entryListRow.getParentNode();
            if (null != ContribCollection && ContribCollection.size()>0) {
				java.util.Iterator contribsIterator = ContribCollection.iterator();
				content.setTextContributionsCount("" + ContribCollection.size());
                while (contribsIterator.hasNext()) {
					VolumeEntry myContrib = (VolumeEntry) contribsIterator.next();
					
					if (myContrib!=null && !myContrib.isEmpty()) {						
                        content.setTextViewContribText(myContrib.getCompleteHeadword());
						viewContribAnchor.setHref(this.getUrl() + "?"
												  + HANDLE_PARAMETER + "=" + myContrib.getHandle()
												  + "&" + VIEW_CONTRIB_PARAMETER + "=" +  VIEW_CONTRIB_PARAMETER
												  + "&" + content.NAME_VOLUME + "=" + myContrib.getVolumeName());
						// entry id
						content.setTextEntryId(myContrib.getEntryId());
						
						// creation date
						if (myContrib.getCreationDate() != null) {
							content.setTextCreationDate(Utility.PapillonShortDateFormat.format(myContrib.getCreationDate()));
						}
						else {
							content.setTextCreationDate("no date!");
						}
						
						// IsNewEntry
						boolean isNewEntry = myContrib.getOriginalContributionId()==null || myContrib.getOriginalContributionId().equals("");
						content.setTextIsNewEntry(new Boolean(isNewEntry).toString());
						if (isNewEntry) {
							entryListRow.setAttribute("class","");
						}
						else {
							entryListRow.setAttribute("class","copyentry");
						}
						
						// Status
						content.setTextStatus(myContrib.getStatus());
						
						// mark finished
						if (myContrib.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {
							markFinishedAnchor.setHref(this.getUrl() + "?"
												   + MARK_FINISHED_PARAMETER + "=" + MARK_FINISHED_PARAMETER
												   + "&" + HANDLE_PARAMETER + "=" + myContrib.getHandle()
												   + queryString);
							markFinishedAnchor.setAttribute("class", "");
						}
						else {
							markFinishedAnchor.setHref(this.getUrl());
							markFinishedAnchor.setAttribute("class", "hidden");
						}
						
						// view XML
						viewXmlAnchor.setHref(this.getUrl() + "?"
											  + HANDLE_PARAMETER + "=" + myContrib.getHandle() 
											  + "&" + VIEW_CONTRIB_PARAMETER + "=" +  VIEW_CONTRIB_PARAMETER
											  + "&" + content.NAME_VOLUME + "=" + myContrib.getVolumeName()
											  + "&" + FORMATTER_PARAMETER + "=" + XML_FORMATTER);
						
						editContribAnchor.setHref(EditURL + "?"
														+ EditVolumeParameter + "=" + myContrib.getVolumeName()
														+ "&"+ EditHandleParameter + "=" + myContrib.getHandle());
						
						// remove contrib
                        removeContribAnchor.setHref(this.getUrl() + "?"
													+ REMOVE_CONTRIB_PARAMETER + "=" + REMOVE_CONTRIB_PARAMETER
													+ "&" + HANDLE_PARAMETER + "=" + myContrib.getHandle()
													+ queryString);                        
                        
                        XHTMLElement clone = (XHTMLElement)entryListRow.cloneNode(true);
                        //      we have to take off the id attribute because we did not take it off the original
                        clone.removeAttribute("id");
                        entryTable.appendChild(clone);
                    }
					else {
						PapillonLogger.writeDebugMsg("contrib empty ");
					}
				}
			}
        }
	
    protected void addElement (AdminContributionsTmplXHTML content, Element element)
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
	
	protected java.util.Collection displayEntry (AdminContributionsTmplXHTML content, String volumeName, String handle, String formatter)
		throws PapillonBusinessException,
		java.io.UnsupportedEncodingException,
		HttpPresentationException,
		java.io.IOException {
			
			java.util.Vector myVector = new Vector();
			Volume myVolume = VolumesFactory.getVolumeByName(volumeName);
			java.util.Collection EntryCollection = DictionariesFactory.findAnswerAndTranslations(volumeName, handle, myVolume.getTargetLanguagesArray(), this.getUser());
			
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
	
    
    
    protected void removeTemplateRows(AdminContributionsTmplXHTML content) {
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
