/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id: ContributorsBoard.java 508 2006-08-10 22:17:13Z fbrunet $
 *-----------------------------------------------
 * $Log$
 * Revision 1.9  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.8  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.7  2005/11/01 14:12:02  mangeot
 * *** empty log message ***
 *
 * Revision 1.6  2005/10/21 12:24:16  mangeot
 * *** empty log message ***
 *
 * Revision 1.5  2005/10/21 12:18:10  mangeot
 * *** empty log message ***
 *
 * Revision 1.4  2005/10/21 12:15:35  mangeot
 * *** empty log message ***
 *
 * Revision 1.3  2005/10/21 11:20:46  mangeot
 * Modified the contributors board to add the best contributor and best reviewer
 *
 * Revision 1.2  2005/06/15 16:48:28  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.1.2.3  2005/05/25 21:00:36  mangeot
 * Bug fixes
 *
 * Revision 1.1.2.2  2005/05/24 11:15:48  mangeot
 * Bug fixes in sort
 *
 * Revision 1.1.2.1  2005/05/20 10:31:46  mangeot
 * Added 2 new classes 1 contributors board and one for exporting a volume
 *
 *
 *-----------------------------------------------
 * Papillon Contributors Board page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.session.Session;

import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

import fr.imag.clips.papillon.business.message.MessageDBLoader;
import fr.imag.clips.papillon.presentation.PapillonSessionData;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.util.Iterator;
import java.text.DateFormat;
import java.io.*;

// for users
import fr.imag.clips.papillon.business.user.*;
import fr.imag.clips.papillon.presentation.xhtml.orig.WelcomeTmplXHTML;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.xml.XMLServices;


public class Welcome extends PapillonBasePO {

    protected final static String SORTBY_PARAMETER="SortBy";
	protected final static String ALL="*ALL*";
	
	protected final static String BASE_DIR_CONFIG = "Papillon.Informations.baseDir";
    protected final static String MEDIA_DIR_CONFIG = "Papillon.Informations.mediaDir";
	protected final static String NEWS_VOLUME_DIR="news";
	protected final static String NEWS_FILE="news.xhtml";
	protected final static String LatestNewsIdString = "LatestNews";
	
	protected static org.w3c.dom.Node LatestNewsDOMCache = null;	
    
	protected static int GDEF_estValidatedEntriesCount = 0;
	protected static int GDEF_estReviewedEntriesCount = 0;	
	protected static int GDEF_estFinishedEntriesCount = 0;
	
	protected static java.util.Calendar myCalendar = new java.util.GregorianCalendar();
	
	protected static int DAY_OF_MONTH = 0;

    protected WelcomeTmplXHTML content;

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
        throws javax.xml.parsers.ParserConfigurationException,
			HttpPresentationException,
		    IOException, org.xml.sax.SAXException,
			javax.xml.transform.TransformerException,
			fr.imag.clips.papillon.presentation.PapillonPresentationException {
        
        // Création du contenu
        content = (WelcomeTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("WelcomeTmplXHTML", this.getComms(), this.getSessionData());
	  
        HttpPresentationRequest req = this.getComms().request;

		addLoggedUsersArray(content);
		addLatestNews(content);

		addEntriesCount(content);
				
		String volume =  "GDEF_est";
		java.util.Date fromDate = null;
		java.util.Date toDate = null;
		java.util.Calendar calendar = new java.util.GregorianCalendar();
		calendar.setTime(new java.util.Date());
		try {
			toDate = Utility.PapillonShortDateFormat.parse("9999/01/01");
			fromDate = Utility.PapillonShortDateFormat.parse(""+calendar.get(java.util.Calendar.YEAR)+"/"+(calendar.get(java.util.Calendar.MONTH)+1)+ "/01");
		}
		catch (java.text.ParseException pe) {
			PapillonLogger.writeDebugMsg("Error in date format, please check!");
		}
		addContributorsBoard(volume, fromDate, toDate);
				
				
        //On rend le contenu correct
        return content.getElementFormulaire();
    }


    protected void addContributorsBoard(String volume, java.util.Date fromDate, java.util.Date toDate)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {

            java.util.Vector UsersVector =  fr.imag.clips.papillon.business.dictionary.ContributionsFactory.getContributorsBoard(volume, fromDate, toDate);
			
            //where we must insert the form
            XHTMLTableRowElement theRow = content.getElementTemplateRow();
            XHTMLElement theName = content.getElementName();
            XHTMLElement theLogin = content.getElementLogin();
			XHTMLElement theTableDiv = content.getElementContributorsTable();
			theTableDiv.setAttribute("class","");

            Node theRowParent = theRow.getParentNode();

            theRow.removeAttribute("id");
            theName.removeAttribute("id");
            theLogin.removeAttribute("id");
			
            //adding the volumes description
			int contributionsMax = 0;
			int revisionsMax = 0;
			int validationsMax = 0;
			int contributionsTotal = 0;
			int revisionsTotal = 0;
			int validationsTotal = 0;
			String bestContributor = "";
			String bestReviewer = "";
			String bestValidator = "";
				
			for (int i=0; i< UsersVector.size(); i++) {
				java.util.Vector myVector = (java.util.Vector) UsersVector.elementAt(i);
				User myUser = (User) myVector.elementAt(0);
				int finished = Integer.parseInt((String) myVector.elementAt(1));
				int reviewed = Integer.parseInt((String) myVector.elementAt(2));
				int validated = Integer.parseInt((String) myVector.elementAt(3));
				contributionsTotal += finished;
				revisionsTotal += reviewed;
				validationsTotal += validated;
				content.setTextName(myUser.getName());
				content.setTextLogin(myUser.getLogin());
				content.setTextFinished("" + finished);
				content.setTextReviewed("" + reviewed);
				content.setTextValidated("" + validated);
				if (contributionsMax < finished) {
					contributionsMax = finished;
					bestContributor = myUser.getName();
				}
				if (revisionsMax < reviewed) {
					revisionsMax = reviewed;
					bestReviewer = myUser.getName();
				}
				theRowParent.appendChild(theRow.cloneNode(true));
            }
            theRowParent.removeChild(theRow);
			content.setTextBestContributor(bestContributor);
			content.setTextMaxContributions("" + contributionsMax);
 			content.setTextBestReviewer(bestReviewer);
			content.setTextMaxRevisions("" + revisionsMax);
			
			content.setTextTotalFinished("" + contributionsTotal);
			content.setTextTotalReviewed("" + revisionsTotal);
			content.setTextTotalValidated("" + validationsTotal);
        }
			
	protected void addLoggedUsersArray(WelcomeTmplXHTML content)
	throws fr.imag.clips.papillon.business.PapillonBusinessException,
	fr.imag.clips.papillon.presentation.PapillonPresentationException {
		
		java.util.Vector ActiveUsersVector = PapillonSessionManager.getActiveUsersVector();	
		
		//where we must insert the form
		XHTMLTableRowElement theRow = content.getElementLoggedTemplateRow();
		XHTMLElement theCount = content.getElementLoggedUsers();
		XHTMLElement theName = content.getElementLoggedName();
		
		Node theRowParent = theRow.getParentNode();
		
		theRow.removeAttribute("id");
		theCount.removeAttribute("id");
		theName.removeAttribute("id");
		
		content.setTextLoggedUsers("" + ActiveUsersVector.size());
		
		//adding the logged users
		for ( int i = 0; i < ActiveUsersVector.size(); i++ ) {
			User myUser = (User) ActiveUsersVector.elementAt(i);
			if (myUser != null) {
				content.setTextLoggedName(myUser.getName());
			}
			else {
				content.setTextLoggedName("");
			} 
			theRowParent.appendChild(theRow.cloneNode(true));
		}
		theRowParent.removeChild(theRow);
	}
	
	protected void addEntriesCount(WelcomeTmplXHTML content)
	throws HttpPresentationException,
	java.io.IOException {
		
 		// code spécifique pour le GDEF
		Element GDEFFinishedEntryCount = content.getElementGDEFFinishedEntryCount();
		Element GDEFReviewedEntryCount = content.getElementGDEFReviewedEntryCount();
		Element GDEFValidatedEntryCount = content.getElementGDEFValidatedEntryCount();
		if (GDEFFinishedEntryCount != null
			&& GDEFReviewedEntryCount != null
			&& GDEFValidatedEntryCount != null) {
			Volume GDEFVolume = VolumesFactory.getVolumeByName("GDEF_est");
			if (GDEFVolume!=null) {
				if (myCalendar.get(myCalendar.DAY_OF_MONTH) != DAY_OF_MONTH) {
					GDEF_estValidatedEntriesCount = GDEFVolume.getCount(VolumeEntry.VALIDATED_STATUS);
					GDEF_estReviewedEntriesCount = GDEFVolume.getCount(VolumeEntry.REVIEWED_STATUS);
					GDEF_estFinishedEntriesCount = GDEFVolume.getCount(VolumeEntry.FINISHED_STATUS);
					DAY_OF_MONTH = myCalendar.get(myCalendar.DAY_OF_MONTH);
				}
				int total = GDEF_estValidatedEntriesCount + GDEF_estReviewedEntriesCount + GDEF_estFinishedEntriesCount;
				Utility.setText(GDEFFinishedEntryCount,"" + GDEF_estFinishedEntriesCount);
				Utility.setText(GDEFReviewedEntryCount,"" + GDEF_estReviewedEntriesCount);
				Utility.setText(GDEFValidatedEntryCount,"" + GDEF_estValidatedEntriesCount);
			}
		}
    }
	
	protected void addLatestNews(WelcomeTmplXHTML content)
    throws HttpPresentationException, java.io.IOException {
		
		Element LatestNewsContainer = content.getElementLatestNewsContainer();
		
		if (LatestNewsDOMCache == null) {
			org.xml.sax.InputSource newsInputSource = getInputSource(getNewsFileAbsolutePath());
			if (newsInputSource != null) {
				org.w3c.dom.Document myNewsDocument = XMLServices.buildDOMTree(newsInputSource);
				if (myNewsDocument != null) {
					LatestNewsDOMCache = myNewsDocument.getElementById(LatestNewsIdString);
				}
			}
		}
        //On rend le contenu correct
		Node newNode = LatestNewsDOMCache.cloneNode(true);
		PapillonLogger.writeDebugMsg("newnode:" + XMLServices.NodeToString(newNode));
		Node importedNode = content.importNode(newNode,true);
		PapillonLogger.writeDebugMsg("importedNode:" + XMLServices.NodeToString(importedNode));
		LatestNewsContainer.appendChild(importedNode);
    }
	
	protected String getNewsFileAbsolutePath() throws PapillonPresentationException {            
		String baseDir = "";
		String mediaDir = "";
		String filePath = "";
		try {
			baseDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(BASE_DIR_CONFIG);
			mediaDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(MEDIA_DIR_CONFIG);
			if (! baseDir.endsWith(java.io.File.separator)) {
				baseDir = baseDir + java.io.File.separator;
			}
			if (! mediaDir.endsWith(java.io.File.separator)) {
				mediaDir = mediaDir + java.io.File.separator;
			}
		}
		catch (com.lutris.util.ConfigException ex) {
			throw new PapillonPresentationException("Error in Papillon Configuration File: ", ex);
		}
		filePath = baseDir + mediaDir + NEWS_VOLUME_DIR + java.io.File.separator + NEWS_FILE;
		return filePath;
	}
	
	protected org.xml.sax.InputSource getInputSource(String filePath) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {	
		org.xml.sax.InputSource myInputSource = null;
		try {
			java.io.FileInputStream myFileInputStream = new java.io.FileInputStream(filePath);
			myInputSource = new org.xml.sax.InputSource(myFileInputStream);
		}
		catch(java.io.IOException ioex) {
			throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in renderOutputStream()", ioex);
		}
		return myInputSource;
	}
	
	
}
