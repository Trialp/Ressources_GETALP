/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Mathieu Mangeot & Gilles SÈrasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.6  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.5  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.4  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.3  2005/04/11 08:01:02  fbrunet
 * Passage en xhtml des ressources Papillon.
 *
 * Revision 1.1.2.1  2005/02/25 10:22:08  mangeot
 * Bug fixes and added the use of referrer when exiting from Reviewcontributions.po
 *
 * Revision 1.1  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1  2004/11/13 17:41:27  mangeot
 * New admin page for groups admin
 *
 *-----------------------------------------------
 * Papillon AdminGroups page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.session.Session;

//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

import fr.imag.clips.papillon.business.message.MessageDBLoader;

import fr.imag.clips.papillon.presentation.PapillonSessionData;

//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.text.DateFormat;
import java.io.*;

// for users
import fr.imag.clips.papillon.business.user.*;
import fr.imag.clips.papillon.presentation.xhtml.orig.AdminGroupsTmplXHTML;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;


public class AdminGroups extends PapillonBasePO {

    protected final static String REMOVE_PARAMETER="Remove";
    
    protected boolean loggedInUserRequired() {
        return true;
    }

    protected boolean userMayUseThisPO() {
        try {
            return this.getUser().isAdmin();
        } catch (PapillonBusinessException ex) {
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
        
        // CrÈation du contenu
        AdminGroupsTmplXHTML content = (AdminGroupsTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("AdminGroupsTmplXHTML", this.getComms(), this.getSessionData());
	  
        HttpPresentationRequest req = this.getComms().request;

        // If the page is called with parameters, take the requested action
        if (req.getParameterNames().hasMoreElements()) {

            //TEMPORAIRE :avec l URL
            //AJOUT DE DICO
            String userMessage = null;
            if (myGetParameter(AdminGroupsTmplXHTML.NAME_AddGroup)!=null &&
				myGetParameter(AdminGroupsTmplXHTML.NAME_GroupName)!=null &&
				myGetParameter(AdminGroupsTmplXHTML.NAME_GroupPassword)!=null) {
				String groupName = myGetParameter(AdminGroupsTmplXHTML.NAME_GroupName);
				String groupPassword = myGetParameter(AdminGroupsTmplXHTML.NAME_GroupPassword);
				String groupPassword2 = myGetParameter(AdminGroupsTmplXHTML.NAME_GroupPasswordConfirm);
				
                GroupAnswer myGroupAnswer = GroupsFactory.createUniqueGroup(groupName, groupPassword, groupPassword2, this.getUser().getLogin());
                userMessage = myGroupAnswer.getMessage();
            }
            else if (null != myGetParameter(REMOVE_PARAMETER)) {
                Group myGroup = GroupsFactory.findGroupByHandle(myGetParameter(REMOVE_PARAMETER));
								if (myGroup !=null && !myGroup.isEmpty()) {
									String groupName = myGroup.getName();
									myGroup.delete();
									userMessage = "Group "+  groupName + " has been deleted";
								}
								else {
									userMessage = "Group not in database";
                                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
								}
            } else {
                String errorMessage = "Error: Wrong arguments";
                this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,errorMessage);
            }

			if (userMessage != null) {
				this.getSessionData().writeUserMessage(userMessage);
				PapillonLogger.writeDebugMsg(userMessage);
			}
        }
				
        addGroupsArray(content);
        
        //On rend le contenu correct
        return content.getElementFormulaire();
    }


    protected void addGroupsArray(AdminGroupsTmplXHTML content)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {

            Group[] GroupsTable=GroupsFactory.getGroupsArray();
            //where we must insert the form
            HTMLTableRowElement theRow = content.getElementGrpTemplateRow();
            HTMLElement theCount = content.getElementExistingGroups();
            HTMLElement theName = content.getElementGrpName();
            HTMLElement theAdmins = content.getElementGrpAdmins();
            HTMLAnchorElement theRemoveAnchor = content.getElementGrpRemoveAnchor();

            Node theRowParent = theRow.getParentNode();

            theRow.removeAttribute("id");
            theCount.removeAttribute("id");
            theName.removeAttribute("id");
            theAdmins.removeAttribute("id");
            theRemoveAnchor.removeAttribute("id");
			
			content.setTextExistingGroups("" + GroupsTable.length);

            //adding the volumes description
            for ( int i = 0; i < GroupsTable.length; i++ ) {
				Group tempGroup = GroupsTable[i];
                content.setTextGrpName(tempGroup.getName());
				String[] adminsArray = tempGroup.getAdminsArray();
				String groupsString = "";
				for (int j=0;j< adminsArray.length;j++) {
					groupsString += adminsArray[j] + "";
				}
                content.setTextGrpAdmins(groupsString);
                // à commenter sinon les admin ne peuvent aps supprimer des groupes.
                // je ne me souviens plus pourquoi j'avais bloqué cette fonctionnalité !
			//if (tempGroup.isAdmin(this.getUser().getLogin())) {
                 theRemoveAnchor.setHref(this.getUrl() + "?" + REMOVE_PARAMETER +
                                        "=" + tempGroup.getHandle());

			//	}
			//	else {
			//		theRemoveAnchor.setHref("");
			//	}
                theRowParent.appendChild(theRow.cloneNode(true));
            }

            theRowParent.removeChild(theRow);
        }
}
