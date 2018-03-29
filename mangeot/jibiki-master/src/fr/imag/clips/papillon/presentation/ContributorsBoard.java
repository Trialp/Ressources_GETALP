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
import fr.imag.clips.papillon.presentation.xhtml.orig.ContributorsBoardTmplXHTML;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.PapillonLogger;


public class ContributorsBoard extends PapillonBasePO {

    protected final static String SORTBY_PARAMETER="SortBy";
	protected final static String ALL="*ALL*";
    
    protected ContributorsBoardTmplXHTML content;

    protected boolean loggedInUserRequired() {
        return true;
    }

    protected boolean userMayUseThisPO() {
        try {
            return this.getUser().isSpecialist();
        } catch (fr.imag.clips.papillon.business.PapillonBusinessException ex) {
            this.getSessionData().writeUserMessage("Error getting the authorisation to use this PO.");
        }
        return false;
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
        content = (ContributorsBoardTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("ContributorsBoardTmplXHTML", this.getComms(), this.getSessionData());
	  
        HttpPresentationRequest req = this.getComms().request;

		String lookup =  myGetParameter(content.NAME_LOOKUP);
		String volume =  myGetParameter(content.NAME_VOLUME);
		String fromDateString =  myGetParameter(content.NAME_FromDate);
		String toDateString =  myGetParameter(content.NAME_ToDate);

		if (lookup !=null && !lookup.equals("")
			&& fromDateString !=null && !fromDateString.equals("")
			&& toDateString !=null && !toDateString.equals("")) {
			
			this.setPreference(content.NAME_VOLUME,volume);
			if (volume != null && (volume.equals("") || volume.equals(ALL))) {
				volume = null;
			}
			
			this.setPreference(content.NAME_FromDate,fromDateString);
			this.setPreference(content.NAME_ToDate,toDateString);

			String userMessage = "";
			java.util.Date fromDate = null;
			java.util.Date toDate = null;
			
			try {
				fromDate = Utility.PapillonShortDateFormat.parse(fromDateString);
				toDate = Utility.PapillonShortDateFormat.parse(toDateString);
			}
			catch (java.text.ParseException pe) {
				userMessage = "Error in date format, please check!";
			}
			
			addContributorsBoard(volume, fromDate, toDate);
			if (userMessage != null) {
				this.getSessionData().writeUserMessage(userMessage);
				PapillonLogger.writeDebugMsg(userMessage);
			}
		}
        else {
			volume = this.getPreference(content.NAME_VOLUME);
			fromDateString = this.getPreference(content.NAME_FromDate);
			toDateString = this.getPreference(content.NAME_ToDate);

		}
		addBoardForm(volume, fromDateString, toDateString);
		
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
		
	protected void addBoardForm(String selectedVolume, String fromDate, String toDate)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
	
		// volume
        XHTMLOptionElement volumeOptionTemplate = content.getElementVolumeOptionTemplate();
        Node volumeSelect = volumeOptionTemplate.getParentNode();
        volumeOptionTemplate.removeAttribute("id");
        // We assume that the option element has only one text child 
        // (it should be this way if the HTML is valid...)
        Text volumeTextTemplate = (Text)volumeOptionTemplate.getFirstChild(); 
                
		//      
        for (Iterator iter = fr.imag.clips.papillon.business.dictionary.VolumesFactory.getVolumesArray().iterator(); iter.hasNext();) {
            String volumeName = ((Volume)iter.next()).getName();
            
            //
            volumeOptionTemplate.setValue(volumeName);
            volumeOptionTemplate.setLabel(volumeName);
            // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux 
            // specs W3C.
            volumeTextTemplate.setData(volumeName);
            volumeOptionTemplate.setSelected(volumeName.equals(selectedVolume));
            volumeSelect.appendChild(volumeOptionTemplate.cloneNode(true));
        }
        volumeSelect.removeChild(volumeOptionTemplate);

		// fromDate
		XHTMLInputElement fromDateElement = content.getElementFromDate();
		fromDateElement.setValue(fromDate);

		// toDate
		XHTMLInputElement toDateElement = content.getElementToDate();
		toDateElement.setValue(toDate);
		
	}
}
