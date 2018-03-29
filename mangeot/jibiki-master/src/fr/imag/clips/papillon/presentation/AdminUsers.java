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
 * Revision 1.5.6.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.5  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.4  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.3  2005/04/11 08:01:02  fbrunet
 * Passage en xhtml des ressources Papillon.
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * Papillon Admin Users page.
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
import fr.imag.clips.papillon.business.PapillonBusinessException;

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
import fr.imag.clips.papillon.presentation.xhtml.orig.AdminUsersTmplXHTML;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.PapillonLogger;


public class AdminUsers extends PapillonBasePO {

    protected final static String SEE_PARAMETER="See";
    protected final static String REMOVE_PARAMETER="Remove";
    protected final static String MAKEADMIN_PARAMETER="MakeAdmin";
    protected final static String RESETPASSWORD_PARAMETER="ResetPassword";
    protected final static String MAKESPECIALIST_PARAMETER="MakeSpecialist";
	protected final static String MAKEVALIDATOR_PARAMETER="MakeValidator";
    protected final static String ADD_IN_GROUP_PARAMETER="AddInGroup";
    protected final static String REMOVE_FROM_GROUP_PARAMETER="RemoveFromGroup";
    protected final static String LEVEL_UP_PARAMETER="LevelUp";
    protected final static String LEVEL_DOWN_PARAMETER="LevelDown";
    protected final static String GROUP_PARAMETER="Group";
    protected final static String SORTBY_PARAMETER="SortBy";

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
        
        // Création du contenu
        AdminUsersTmplXHTML content = (AdminUsersTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("AdminUsersTmplXHTML", this.getComms(), this.getSessionData());
	  
        HttpPresentationRequest req = this.getComms().request;

        // If the page is called with parameters, take the requested action
        if (req.getParameterNames().hasMoreElements()) {
            //TEMPORAIRE :avec l URL
            //AJOUT DE DICO
            String userMessage = null;
            if (null != myGetParameter(SEE_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(SEE_PARAMETER));
                if (myUser !=null && !myUser.isEmpty()) {
                    String userName = myUser.getName();
                    userMessage = "User "+  userName + " XML: " + myUser.getXmlCode();
                }
                else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else if (null != myGetParameter(REMOVE_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(REMOVE_PARAMETER));
                if (myUser !=null && !myUser.isEmpty()) {
                    String userName = myUser.getName();
                    myUser.delete();
                    userMessage = "User "+  userName + " has been deleted";
                }
                else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else if (null != myGetParameter(MAKEADMIN_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(MAKEADMIN_PARAMETER));
                if (null != myUser && !myUser.isEmpty()) {
                    myUser.addGroup(Group.ADMIN_GROUP);
                    myUser.save();
                    userMessage = "User "+ myUser.getName() + " is admin";
                } else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else if (null != myGetParameter(RESETPASSWORD_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(RESETPASSWORD_PARAMETER));
                if (null != myUser && !myUser.isEmpty()) {
                    myUser.setPassword(myUser.getLogin());
                    myUser.save();
                    userMessage = "User "+ myUser.getName() + " has a new password with the login " + myUser.getLogin() + " for value";
                } else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else if (null != myGetParameter(MAKESPECIALIST_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(MAKESPECIALIST_PARAMETER));
                if (null != myUser && !myUser.isEmpty()) {
                    myUser.addGroup(Group.SPECIALIST_GROUP);
                    myUser.save();
                    userMessage = "User "+ myUser.getName() + " is a specialist";
                } else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else if (null != myGetParameter(MAKEVALIDATOR_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(MAKEVALIDATOR_PARAMETER));
                if (null != myUser && !myUser.isEmpty()) {
                    myUser.addGroup(Group.VALIDATOR_GROUP);
                    myUser.save();
                    userMessage = "User "+ myUser.getName() + " is a validator";
                } else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else if (null != myGetParameter(ADD_IN_GROUP_PARAMETER) && null != myGetParameter(GROUP_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(ADD_IN_GROUP_PARAMETER));
                if (null != myUser && !myUser.isEmpty()) {
                    myUser.addGroup(myGetParameter(GROUP_PARAMETER));
                    myUser.save();
                    userMessage = "User "+ myUser.getName() + " is in group " + myGetParameter(GROUP_PARAMETER);
                } else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else if (null != myGetParameter(REMOVE_FROM_GROUP_PARAMETER) && null != myGetParameter(GROUP_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(REMOVE_FROM_GROUP_PARAMETER));
                if (null != myUser && !myUser.isEmpty()) {
                    myUser.removeGroup(myGetParameter(GROUP_PARAMETER));
                    myUser.save();
                    userMessage = "User "+ myUser.getName() + " has been removed from group " + myGetParameter(GROUP_PARAMETER);
                } else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else if (null != myGetParameter(LEVEL_UP_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(LEVEL_UP_PARAMETER));
                if (null != myUser && !myUser.isEmpty()) {
                    myUser.levelUp();
                    myUser.save();
                    userMessage = "User "+ myUser.getName() + " has been levelled up";
                } else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else if (null != myGetParameter(LEVEL_DOWN_PARAMETER)) {
                User myUser = UsersFactory.findUserById(myGetParameter(LEVEL_DOWN_PARAMETER));
                if (null != myUser && !myUser.isEmpty()) {
                    myUser.levelDown();
                    myUser.save();
                    userMessage = "User "+ myUser.getName() + " has been levelled down";
                } else {
                    userMessage = "User not in database";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_NOT_FOUND,userMessage);
                }
            }
            else {
                String errorMessage = "Error: Wrong arguments";
                this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,errorMessage);
            }
		
            if (userMessage != null) {
				this.getSessionData().writeUserMessage(userMessage);
				PapillonLogger.writeDebugMsg(userMessage);
			}
        }
		String sortBy = myGetParameter(SORTBY_PARAMETER);
		
        addLoggedUsersArray(content);
        addUsersArray(content, sortBy);
        
        //On rend le contenu correct
        return content.getElementFormulaire();
    }

	protected void addLoggedUsersArray(AdminUsersTmplXHTML content)
        throws fr.imag.clips.papillon.business.PapillonBusinessException,
		 fr.imag.clips.papillon.presentation.PapillonPresentationException {
		
			java.util.Vector ActiveUsersVector = PapillonSessionManager.getActiveUsersVector();	

            //where we must insert the form
            HTMLTableRowElement theRow = content.getElementLoggedTemplateRow();
            HTMLElement theCount = content.getElementLoggedUsers();
            HTMLElement theName = content.getElementLoggedName();
            HTMLElement theLogin = content.getElementLoggedLogin();
            HTMLElement theEmail = content.getElementLoggedEmail();
 
            Node theRowParent = theRow.getParentNode();

            theRow.removeAttribute("id");
            theCount.removeAttribute("id");
            theName.removeAttribute("id");
            theLogin.removeAttribute("id");
            theEmail.removeAttribute("id");

			content.setTextLoggedUsers("" + ActiveUsersVector.size());

            //adding the logged users
            for ( int i = 0; i < ActiveUsersVector.size(); i++ ) {
				User myUser = (User) ActiveUsersVector.elementAt(i);
				if (myUser != null) {
					content.setTextLoggedName(myUser.getName());
					content.setTextLoggedLogin(myUser.getLogin());
					content.setTextLoggedEmail(myUser.getEmail());
				}
				else {
					content.setTextLoggedName("");
					content.setTextLoggedLogin("guest");
					content.setTextLoggedEmail("");
				} 
				theRowParent.appendChild(theRow.cloneNode(true));
            }
            theRowParent.removeChild(theRow);
        }


    protected void addUsersArray(AdminUsersTmplXHTML content, String sortBy)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {

            User[] UsersTable=UsersFactory.getUsersArray(sortBy);
            //where we must insert the form
            HTMLTableRowElement theRow = content.getElementTemplateRow();
            HTMLElement theCount = content.getElementExistingUsers();
            HTMLElement theName = content.getElementName();
            HTMLElement theLogin = content.getElementLogin();
            HTMLElement theEmail = content.getElementEmail();
            HTMLElement theGroups = content.getElementGroups();
            HTMLElement creationDate = content.getElementCreationDate();
            HTMLElement modificationDate = content.getElementModificationDate();
            HTMLAnchorElement theMakeAdminAnchor = content.getElementMakeAdminAnchor();
            HTMLAnchorElement theResetPasswordAnchor = content.getElementResetPasswordAnchor();
            HTMLAnchorElement theMakeSpecialistAnchor = content.getElementMakeSpecialistAnchor();
            HTMLAnchorElement theSeeAnchor = content.getElementSeeAnchor();
            HTMLAnchorElement theRemoveAnchor = content.getElementRemoveAnchor();

            Node theRowParent = theRow.getParentNode();

            theRow.removeAttribute("id");
            theCount.removeAttribute("id");
            theName.removeAttribute("id");
            theLogin.removeAttribute("id");
            theEmail.removeAttribute("id");
            theGroups.removeAttribute("id");
            creationDate.removeAttribute("id");
            modificationDate.removeAttribute("id");
            theMakeAdminAnchor.removeAttribute("id");
            theResetPasswordAnchor.removeAttribute("id");
            theMakeSpecialistAnchor.removeAttribute("id");
            theSeeAnchor.removeAttribute("id");
            theRemoveAnchor.removeAttribute("id");
			
			content.setTextExistingUsers("" + UsersTable.length);

            //adding the volumes description
            for ( int i = 0; i < UsersTable.length; i++ ) {
                content.setTextName(UsersTable[i].getName());
                content.setTextLogin(UsersTable[i].getLogin());
                content.setTextEmail(UsersTable[i].getEmail());
				String [] Groups = UsersTable[i].getGroupsArray();
				String groupsString = "";
				if (Groups != null && Groups.length>0)
					for (int j=0; j<Groups.length;j++) {
						groupsString += " " + Groups[j];
					}
                content.setTextGroups(groupsString);
                
                String creationDateString = "";
                if (UsersTable[i].getCreationDate() != null) {
                    creationDateString = UsersTable[i].getCreationDate().toString();
                }
                content.setTextCreationDate(creationDateString);

                String modificationDateString = "";
                if (UsersTable[i].getModificationDate() != null) {
                    modificationDateString = UsersTable[i].getModificationDate().toString();
                }
                content.setTextModificationDate(modificationDateString);

                
                theMakeAdminAnchor.setHref(this.getUrl() + "?" + MAKEADMIN_PARAMETER +
                                           "=" + UsersTable[i].getHandle());
                theResetPasswordAnchor.setHref(this.getUrl() + "?" + RESETPASSWORD_PARAMETER +
                                           "=" + UsersTable[i].getHandle());
                theMakeSpecialistAnchor.setHref(this.getUrl() + "?" + MAKESPECIALIST_PARAMETER +
                                           "=" + UsersTable[i].getHandle());
                theSeeAnchor.setHref(this.getUrl() + "?" + SEE_PARAMETER +
                                     "=" + UsersTable[i].getHandle());
                theRemoveAnchor.setHref(this.getUrl() + "?" + REMOVE_PARAMETER +
                                        "=" + UsersTable[i].getHandle());

                theRowParent.appendChild(theRow.cloneNode(true));
            }

            theRowParent.removeChild(theRow);
        }
}
