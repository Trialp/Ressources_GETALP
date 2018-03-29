/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 *
 *  © 2005 Mathieu MANGEOT - ATILF
 *  Projet GDEF
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.10  2007/04/09 15:27:19  serasset
 *  Modified xhtml files and applications layout, because the "Targets" id was duplicated
 *  between QueryMenu and AdvancedSearch.
 *  Corrected problems with the Lexalp italian home page and advanced query form.
 *  Added a link to axie when searching for a lexie and user is logged in.
 *
 *  Revision 1.9  2006/08/10 22:17:13  fbrunet
 *  - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 *  - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 *  - Bug correction : +/- in advanced search
 *
 *  Revision 1.8  2006/06/09 10:10:43  fbrunet
 *  Add generic components (AdvancedQueryForm, QueryRequest and ViewQueryResult) in Home.java
 *
 *  Revision 1.7  2006/04/06 15:06:39  fbrunet
 *  New class 'creationEditInit' : create new entry
 *  Modify LexALPEditEntry : only edit entry
 *
 *  Revision 1.6  2006/03/01 15:12:31  mangeot
 *  Merge between maintrunk and LEXALP_1_1 branch
 *
 *  Revision 1.5  2006/02/26 14:04:56  mangeot
 *  Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 *  Revision 1.4.4.1  2006/02/17 13:21:25  mangeot
 *
 *  MM: modified AdvancedQueryForm. getAllTargetLanguages, getAllSourceLanguages and getCdmElementsWithDefaultLanguage are now static in AvailableLanguages.java in order to accelerate the execution.
 *
 *  Revision 1.4  2005/06/23 14:05:36  mangeot
 *  Added EDIT_DATA
 *
 *  Revision 1.3  2005/06/20 16:55:05  mangeot
 *  multiple bug fixes
 *
 *  Revision 1.2  2005/06/16 10:42:15  mangeot
 *  Added and modified files for the GDEF project
 *
 *  Revision 1.1  2005/06/16 08:21:23  mangeot
 *  *** empty log message ***
 *
 *
 *
 *  -----------------------------------------------
 *
 *  Cette classe implémente le layout par défaut de l'ensemble des pages de papillon
 *  Les pages utilisent une instance de Layout et se contentent de remplir son contenu
 *
 */
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
// For the node
import org.w3c.dom.*;
//import org.w3c.dom.html.*;
import org.enhydra.xml.xhtml.dom.*;

import java.io.*;

// For the user
import fr.imag.clips.papillon.business.user.*;
import fr.imag.clips.papillon.business.locales.*;
import fr.imag.clips.papillon.business.dictionary.*;
import fr.imag.clips.papillon.business.utility.Utility;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.presentation.PapillonSessionData;
//for debug
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.util.Iterator;
import java.text.DateFormat;

//import java.io.*;

/**
*  Description of the Class
 *
 * @author     serasset
 * @created    December 8, 2004
 */
public class GDEFLayout implements StdLayout {
    protected LayoutXHTML layout;
    
    // Constructeur avec script
    /**
        *  Constructor for the StdLayout object
     *
     * @param  comms
     *      Description of the Parameter
     * @param  sessionData
     *      Description of the Parameter
     * @param  url
     *      Description of the Parameter
     * @param  script
     *      Description of the Parameter
     * @exception  com.lutris.appserver.server.httpPresentation.HttpPresentationException
     *      Description of the Exception
     * @exception  UnsupportedEncodingException
     *      Description of the Exception
     */
    public void initLayout(HttpPresentationComms comms, PapillonSessionData sessionData, String url, XHTMLScriptElement script)
        throws com.lutris.appserver.server.httpPresentation.HttpPresentationException, UnsupportedEncodingException {
            
            // Création du document
            layout = (LayoutXHTML) MultilingualXHtmlTemplateFactory.createTemplate("LayoutXHTML", comms, sessionData);
           // HeaderXHTML header = (HeaderXHTML) MultilingualXHtmlTemplateFactory.createTemplate("HeaderXHTML", comms, sessionData);
           // Node menuBar = header.getElementMenuBar();
            
            /*
            // adding a script if needed
            XHTMLScriptElement scriptElement = (XHTMLScriptElement) layout.getElementScript();
            if (null != script && !script.equals("")) {
				script = "\n" + script + "\n//";
				while (scriptElement.hasChildNodes()) {
					scriptElement.removeChild(scriptElement.getFirstChild());
				}
				Comment scriptContent = scriptElement.getOwnerDocument().createComment(script);
                scriptElement.appendChild(scriptContent);
            }
            scriptElement.removeAttribute("id");
             */
            
            // adding a new script if needed
            XHTMLScriptElement scriptElement = (XHTMLScriptElement) layout.getElementScript();
            if (null != script) {
                Node scriptParent = scriptElement.getParentNode();
                scriptParent.removeChild(scriptElement);
                scriptParent.appendChild(layout.importNode(script, true));
            }
            scriptElement.removeAttribute("id");
                        
            // Insertion du header et du footer
           // layout.getElementHeaderPlace().appendChild(layout.importNode(menuBar, true));
            // layout.getElementFooterPlace().appendChild(layout.importNode( menuBar, true));
            
            // Gestion du menu :
            // Si les utilisateurs sont logués, on met leur login
            
            // menu
            handleUserMenu(comms, sessionData);
            handleConsultForm(comms, sessionData);
			
			if (Admin.EDIT_DATA) {
				handleLexieMenu(comms, sessionData);
				handleReviewerMenu(comms, sessionData);
				handleValidatorMenu(comms, sessionData);
			}
            handleAdministratorMenu(comms, sessionData);
            
        }
    
    
    /**
        *  Gets the layout attribute of the StdLayout object
     *
     * @return    The layout value
     */
    public Document getLayout() {
        return (Document) layout;
    }
    
    public Node getContentPlaceHolder() {
        return layout.getElementMainColumn();
    }
	
	public Node getBannerPlaceHolder() {
        return layout.getElementBannerPlaceHolder();
    }
    
	public Node getContextualMenuPlaceHolder() {
        return layout.getElementMenuColumn();
    }
    
	public Node getBannerContent() {
        return layout.getElementBannerContent();
    }
	
        
    
    /**
	 *  Description of the Method
     *
     * @param  userMenu                       Description of the Parameter
     * @param  myUser                         Description of the Parameter
     * @exception  PapillonBusinessException  Description of the Exception
     */
    protected void handleUserMenu(HttpPresentationComms comms, PapillonSessionData sessionData)
	throws com.lutris.appserver.server.httpPresentation.HttpPresentationException, PapillonBusinessException {
		
		LangAndUserXHTML userMenu = (LangAndUserXHTML) MultilingualXHtmlTemplateFactory.createTemplate("LangAndUserXHTML", comms, sessionData);
		
		// If the user is logged
		User myUser = sessionData.getUser();
		if (null != myUser && !myUser.isEmpty()) {
			userMenu.setTextUserLogin(myUser.getLogin());
			Utility.removeElement(userMenu.getElementLoginAnchor());
		}                                         // If the user is not logged
		else {
			userMenu.setTextUserLogin("");
			Utility.removeElement(userMenu.getElementUserProfileAnchor());
			Utility.removeElement(userMenu.getElementLogoutAnchor());
		}
		
		layout.getElementUserMenuHolder().appendChild(layout.importNode(userMenu.getElementUserMenu(), true));
	}
    
    /**
        *  Description of the Method
     *
     * @param  comms
     *      Description of the Parameter
     * @param  sessionData
     *      Description of the Parameter
     * @exception  com.lutris.appserver.server.httpPresentation.HttpPresentationException
     *      Description of the Exception
     * @exception  PapillonBusinessException
     *      Description of the Exception
     * @exception  HttpPresentationException
     *      Description of the Exception
     * @exception  UnsupportedEncodingException
     *      Description of the Exception
     */
    protected void handleConsultForm(HttpPresentationComms comms, PapillonSessionData sessionData)
        throws com.lutris.appserver.server.httpPresentation.HttpPresentationException,
        PapillonBusinessException,
        HttpPresentationException,
        UnsupportedEncodingException {
            
            QueryMenuXHTML queryMenu = (QueryMenuXHTML) MultilingualXHtmlTemplateFactory.createTemplate("QueryMenuXHTML", comms, sessionData);
            //XHTMLInputElement headwordInput = queryMenu.getElementHeadwordInput();
            XHTMLInputElement headwordInput = queryMenu.getElementValueField();
            String headtmp = sessionData.getPreference("Home.po", headwordInput.getName());
            if (headtmp != null) {
                headwordInput.setValue(headtmp);
            }
            
            // Adding the appropriate source languages to the source list
            XHTMLOptionElement sourceOptionTemplate = queryMenu.getElementSourceOptionTemplate();
            XHTMLSelectElement sourceSelect = (XHTMLSelectElement) sourceOptionTemplate.getParentNode();
            sourceOptionTemplate.removeAttribute("id");
            // We assume that the option element has only one text child
            // (it should be this way if the HTML is valid...)
            
            String langLoc = sessionData.getUserPreferredLanguage();
            String prefSrcLang = sessionData.getPreference("Home.po", sourceSelect.getName());
            if (prefSrcLang == null || prefSrcLang.equals("")) {
                prefSrcLang = langLoc;
                sessionData.setPreference("Home.po", sourceSelect.getName(), prefSrcLang);
            }
                        
            // Adding the appropriate target languages to the target list
            //XHTMLOptionElement targetOptionTemplate = queryMenu.getElementTargetOptionTemplate();
            XHTMLOptionElement targetOptionTemplate = queryMenu.getElementTargetTmpl();
            XHTMLSelectElement targetSelect = (XHTMLSelectElement) targetOptionTemplate.getParentNode();
            if (!sessionData.getClientWithLabelDisplayProblems()) {
                PapillonBasePO.setUnicodeLabels(targetSelect);
            }
            targetOptionTemplate.removeAttribute("id");
            // We assume that the option element has only one text child
            // (it should be this way if the HTML is valid...)
            Text targetTextTemplate = (Text) targetOptionTemplate.getFirstChild();
            String prefTrgLang = sessionData.getPreference("Home.po", targetSelect.getName());
            if (prefTrgLang == null || prefTrgLang.equals("")) {
                prefTrgLang = Home.ALL_TARGETS;
                sessionData.setPreference("Home.po", targetSelect.getName(), prefTrgLang);
            }
            
            for (Iterator iter = AvailableLanguages.getTargetLanguagesArray().iterator(); iter.hasNext();) {
                String langi = (String)iter.next();
                
                //
                targetOptionTemplate.setValue(langi);
                if (sessionData.getClientWithLabelDisplayProblems()) {
                    targetOptionTemplate.setLabel(Languages.localizeLabel(langLoc, langi));
                } else {
                    targetOptionTemplate.setLabel(Languages.localizeName(langLoc, langi));
                }
                // We should select the previously selected target languages...
                targetTextTemplate.setData(Languages.localizeName(langLoc, langi));
                targetSelect.appendChild(targetOptionTemplate.cloneNode(true));
            }
            targetSelect.removeChild(targetOptionTemplate);
            //PapillonBasePO.setSelected(queryMenu.getElementTarget(), prefTrgLang);
            PapillonBasePO.setSelected(queryMenu.getElementQMTargets(), prefTrgLang);
            
            // Add the menu to the Page
            layout.getElementQueryMenuHolder().appendChild(layout.importNode(queryMenu.getElementQueryMenu(), true));
        }
    
    
    /**
        *  Description of the Method
     *
     * @param  comms
     *      Description of the Parameter
     * @param  sessionData
     *      Description of the Parameter
     * @exception  com.lutris.appserver.server.httpPresentation.HttpPresentationException
     *      Description of the Exception
     */
    protected void handleLexieMenu(HttpPresentationComms comms, PapillonSessionData sessionData)
        throws com.lutris.appserver.server.httpPresentation.HttpPresentationException {
            // If the user is logged in add the lexie menu
            User myUser = sessionData.getUser();
            if (null != myUser && !myUser.isEmpty()) {
                LexiesManagementXHTML lexiesMenu = (LexiesManagementXHTML) MultilingualXHtmlTemplateFactory.createTemplate("LexiesManagementXHTML", comms, sessionData);
				layout.getElementLexiesManagementHolder().appendChild(layout.importNode(lexiesMenu.getElementLexiesManagement(), true));
            }
        }
    
    
    /**
        *  Description of the Method
     *
     * @param  comms
     *      Description of the Parameter
     * @param  sessionData
     *      Description of the Parameter
     * @exception  com.lutris.appserver.server.httpPresentation.HttpPresentationException
     *      Description of the Exception
     */
    protected void handleReviewerMenu(HttpPresentationComms comms, PapillonSessionData sessionData)
        throws com.lutris.appserver.server.httpPresentation.HttpPresentationException {
            // If the user is not a specialist reviewer
            User myUser = sessionData.getUser();
            if (null != myUser && !myUser.isEmpty() && myUser.isSpecialist()) {
                ReviewerMenuXHTML reviewerMenu = (ReviewerMenuXHTML) MultilingualXHtmlTemplateFactory.createTemplate("ReviewerMenuXHTML", comms, sessionData);
                layout.getElementReviewerMenuHolder().appendChild(layout.importNode(reviewerMenu.getElementReviewerMenu(), true));
            }
        }
    
    
	/**
        *  Description of the Method
     *
     * @param  comms
     *      Description of the Parameter
     * @param  sessionData
     *      Description of the Parameter
     * @exception  com.lutris.appserver.server.httpPresentation.HttpPresentationException
     *      Description of the Exception
     */
    protected void handleValidatorMenu(HttpPresentationComms comms, PapillonSessionData sessionData)
        throws com.lutris.appserver.server.httpPresentation.HttpPresentationException {
            // If the user is not a specialist reviewer
            User myUser = sessionData.getUser();
            if (null != myUser && !myUser.isEmpty() && myUser.isValidator()) {
                ValidatorMenuXHTML validatorMenu = (ValidatorMenuXHTML) MultilingualXHtmlTemplateFactory.createTemplate("ValidatorMenuXHTML", comms, sessionData);
                layout.getElementValidatorMenuHolder().appendChild(layout.importNode(validatorMenu.getElementValidatorMenu(), true));
            }
        }

	/**
        *  Description of the Method
     *
     * @param  comms
     *      Description of the Parameter
     * @param  sessionData
     *      Description of the Parameter
     * @exception  com.lutris.appserver.server.httpPresentation.HttpPresentationException
     *      Description of the Exception
     */
    protected void handleAdministratorMenu(HttpPresentationComms comms, PapillonSessionData sessionData)
        throws com.lutris.appserver.server.httpPresentation.HttpPresentationException {
            // If the user is not a specialist reviewer
            User myUser = sessionData.getUser();
            if (null != myUser && !myUser.isEmpty() && myUser.isAdmin()) {
                AdministrationMenuXHTML adminMenu = (AdministrationMenuXHTML) MultilingualXHtmlTemplateFactory.createTemplate("AdministrationMenuXHTML", comms, sessionData);
                layout.getElementAdministrationMenuHolder().appendChild(layout.importNode(adminMenu.getElementAdministrationMenu(), true));
            }
        }
    
}
