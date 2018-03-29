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
 * Papillon Login page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

import fr.imag.clips.papillon.business.message.MessageDBLoader;

//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.text.DateFormat;
import java.io.*;

import org.apache.xerces.dom.*;

// for users
import fr.imag.clips.papillon.business.user.*;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.PapillonLogger;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;


public class LoginUser extends PapillonBasePO {
    protected final static String LOGIN_OK_MESSAGE = "LOGIN_OK_MESSAGE";
    protected final static String LOGIN_PARAMETER="Login";
    protected final static String REMEMBER_LOGIN_PARAMETER="RememberLogin";
    protected final static String SESSION_WITHOUT_COOKIES_PARAMETER="WithoutCookie";
    protected final static String NO_REDIRECTION_PARAMETER="NoRedirection";
    protected final static String PASSWORD_PARAMETER="Password";
    protected final static String LOGOUT_PARAMETER="Logout";
    // Now called DESTINATION_AFTER_LOGIN_PARAMETER and inherited from BasePO
    //protected final static String DESTINATION_PARAMETER="Destination";

    protected final static String SUBMIT_PARAMETER="Submit";
    
    protected final static String MESSAGE_STYLE = "display: block; background-color: yellow; padding: 5px;";
    
    protected LoginUserTmplXHTML content;


    protected boolean loggedInUserRequired() {
        return false;
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
	       javax.xml.transform.TransformerException
    {

        // Création du contenu
        content = (LoginUserTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("LoginUserTmplXHTML", this.getComms(), this.getSessionData());

        HttpPresentationRequest req = this.getComms().request;
        //HttpPresentationResponse theResponse = this.getComms().response;


        String Login = myGetParameter(LOGIN_PARAMETER);
        String RememberLogin = myGetParameter(REMEMBER_LOGIN_PARAMETER);
		String SessionWithoutCookies = myGetParameter(SESSION_WITHOUT_COOKIES_PARAMETER);
		String NoRedirection = myGetParameter(NO_REDIRECTION_PARAMETER);
        String Logout = myGetParameter(LOGOUT_PARAMETER);
        String Password = myGetParameter(PASSWORD_PARAMETER);
        String Dest = myGetParameter(DESTINATION_AFTER_LOGIN_PARAMETER);
        // If there is no destination, just redirect the user to the home page after a succesfull log in.
		// This redirection does not work with the jessionid cookie in the URL
        //Dest = (Dest != null) ? Dest : req.getAppFileURIPath("/");
		Dest = (Dest != null && !Dest.equals("")) ? Dest : this.getUrl();
       
        String userMessage = "";
        // If the page is called with parameters, take the requested action
       /* String params = "";
        java.util.Enumeration parameterNames = this.getComms().request.getParameterNames();
        while (parameterNames.hasMoreElements()) {
            String parameterName = (String) parameterNames.nextElement();
            String value = myGetParameter(parameterName);
            params += parameterName + "=" + value + "&";
        }
        PapillonLogger.writeDebugMsg("LoginUser.po with parameters: " + params);*/
        if (req.getParameterNames().hasMoreElements()) {
            if (null != myGetParameter(LOGIN_OK_MESSAGE)) {
         /*       this.getComms().response.setHeader("Cache-Control", "private, max-age=10800, pre-check=10800"); */
               content.getElementLoginOKMessage().setAttribute("style",MESSAGE_STYLE);
            }
            else if (null != myGetParameter(SUBMIT_PARAMETER) &&
                null != Login && !Login.equals("") &&
                null != Password && !Password.equals("")) {
                User myUser = UsersFactory.findUserByLogin(Login);
                if (null != myUser && !myUser.isEmpty()) { 
                    if (myUser.HasCorrectPassword(Password)) {
                        setUser(myUser);
                        PapillonLogger.writeDebugMsg("user " + myUser.getLogin() +" logged");
                       if (RememberLogin != null && !RememberLogin.equals("")) {
                            this.setCookie(this.LOGIN_COOKIE,
                            myUser.getHandle());
                        }
                        if (Dest.equals(this.getUrl())) {
                            Dest += "?" + LOGIN_OK_MESSAGE + "=on";
                            userMessage = this.getUrl() + ": Login OK";
                        }
                        if (SessionWithoutCookies != null && !SessionWithoutCookies.equals("") || !getComms().request.isRequestedSessionIdFromCookie()) {
                            try {
                                Dest = ((fr.imag.clips.papillon.Papillon)com.lutris.appserver.server.Enhydra.getApplication()).encodeUrl(Dest,this.getComms().session.getSessionKey());
                                PapillonLogger.writeDebugMsg("Url without cookies: " + Dest);
                            }
                            catch (com.lutris.appserver.server.ApplicationException ae) {
                                PapillonLogger.writeDebugMsg("Error in LoginUser.po: " + ae.toString());
                            }
                            if (SessionWithoutCookies != null && !SessionWithoutCookies.equals("")) {
                                getComms().response.setSessionIdEncodeUrlRequired(true);
                            }
                        }
						if ((Dest != null && !Dest.equals("")) && (NoRedirection == null || NoRedirection.equals(""))) {
                            //Dest += "?" + LOGIN_OK_MESSAGE + "=on";
                            //PapillonLogger.writeDebugMsg("Dest:" + Dest);
                     		throw new ClientPageRedirectException(Dest);
                        }
                        else {
                          /*  this.getComms().response.setHeader("Cache-Control", "private, max-age=10800, pre-check=10800"); */
                            content.getElementLoginOKMessage().setAttribute("style",MESSAGE_STYLE);
                        }
                    } else {
                        userMessage = this.getUrl() + ": Wrong password";
                        String errorMsg = "Error: Wrong password";
                       content.getElementWrongPasswordMessage().setAttribute("style",MESSAGE_STYLE);
                        getComms().response.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
                    }
                } else {
                    userMessage = this.getUrl() + ": User unknown";
                    String errorMsg = "Error: User unknown";
                    content.getElementUserUnknownMessage().setAttribute("style",MESSAGE_STYLE);
                    getComms().response.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
                }
            }
            else if (null != Logout && !Logout.equals("")) {
                this.removeUserFromSession();
                userMessage = this.getUrl() + ": User logged out";
                content.getElementUserLoggedOutMessage().setAttribute("style",MESSAGE_STYLE);
            }
            else {
                String errorMessage = "Error: Wrong arguments";
                this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,errorMessage);
            }
            if (userMessage != null && userMessage != "") {
                PapillonLogger.writeDebugMsg(userMessage);
            }
        }

            // If the login fail, we have to send the login form with the appropriate Destination
        HTMLInputElement DestElement = (HTMLInputElement) content.getElementDestination();
        DestElement.setValue(Dest);
        
        return content.getElementFormulaire();
    }
    
    
    
    //    protected void addMessage(String xmlString)
    //    throws fr.imag.clips.papillon.business.PapillonBusinessException {

            //where we must insert the xml
    //        HTMLDivElement theXml = (HTMLDivElement) content.getElementMessage();
    //        Text textTemplate = new TextImpl();
      //      textTemplate.setData(xmlString);
        //    theXml.appendChild(textTemplate);
    //  }    
}
