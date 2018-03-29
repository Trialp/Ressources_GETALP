/*
 *  Papillon project
 *
 *  © Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.7.2.1  2007/07/23 14:23:50  serasset
 *  Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *   - Added XSL extensions callable during xsl transformations
 *   - Implemented new display of query results as requested by EURAC team
 *   - Modified edition interface generator to adapt it to xalan 2.7.0
 *   - Added autocompletion feature to simple search fields
 *   - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 *  Revision 1.7  2007/01/09 17:31:04  fbrunet
 *  Bug correction : error in user verification in HandleEntryModifications class (induce wrong redirection)
 *
 *  Revision 1.6  2006/08/10 22:17:13  fbrunet
 *  - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 *  - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 *  - Bug correction : +/- in advanced search
 *
 *  Revision 1.5  2006/02/26 14:04:56  mangeot
 *  Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 *  Revision 1.4  2005/07/16 12:58:31  serasset
 *  Added limit parameter to query functions
 *  Added a parameter to Formater initializations
 *  Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 *  Revision 1.3  2005/07/08 08:22:46  serasset
 *  Reviewed the Abstract/BasePO hierarchy (moved some methods up in the tree).
 *  Added base classes to allow independant browsing window to establish links during edition.
 *
 *  Revision 1.2  2005/06/15 20:40:41  serasset
 *  Now serve content as text/html instead of application/xhtml+xml
 *
 *  Revision 1.1  2005/05/24 12:51:22  serasset
 *  Updated many aspect of the Papillon project to handle lexalp project.
 *  1. Layout is now parametrable in the application configuration file.
 *  2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 *  3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 *  4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 *  5. It is now possible to give a name to the cookie key in the app conf file
 *  6. Several bug fixes.
 *
 */
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports

import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

import fr.imag.clips.papillon.facelets.util.JibikiContext;
import fr.imag.clips.papillon.CurrentRequestContext;
import fr.imag.clips.papillon.business.PapillonBusinessException;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;

import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;
import com.lutris.appserver.server.Enhydra;
//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.Node;

import org.enhydra.xml.xhtml.dom.*;

import com.lutris.util.*;

// Standard imports
import java.io.IOException;
import java.util.Vector;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Collection;



import org.enhydra.xml.io.OutputOptions;
import org.enhydra.xml.io.DOMFormatter;

import fr.imag.clips.papillon.Papillon;

import fr.imag.clips.papillon.business.PapillonLogger;

import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.UsersFactory;


/**
 * Description of the Class
 *
 * @author serasset
 * @created March 30, 2005
 * @noinspection deprecation
 */
public abstract class AbstractPO
        implements HttpPresentation {
    /**
     * Holds the current Session data
     */
    public PapillonSessionData sessionData;

    // Cookie keys
    /**
     * Description of the Field
     */
    public final static String LOGIN_COOKIE = ((Papillon) Enhydra.getApplication()).getLoginCookieName();
    protected int PAGE_EXPIRE_TIME = ((Papillon) Enhydra.getApplication()).getPageExpireTime();


    /**
     * Administrator message, to be displayed in all sessions
     */
    public static String adminMessage = "";


    /**
     * Description of the Field
     */
    protected static String LOGIN_PAGE = "LoginUser.po";
    /**
     * Description of the Field
     */
    protected static String REGISTER_PAGE = "Register.po";

    /**
     * Description of the Field
     */
    protected static String DESTINATION_AFTER_LOGIN_PARAMETER = "Destination";
            
            
            protected int responseStatus = HttpPresentationResponse.SC_OK;
            protected String responseMessage = "";

    /**
     * This is the procedure that is called when an HTML request occurs.
     *
     * @return XMLObject The XMLObject (in HTML format) that is
     *         to be sent to the user.
     * @throws Exception Description of the Exception
     */
    public abstract Node getDocument() throws Exception;


    /**
     * This method should be implemented in the subclass so that it returns
     * true if this particular request requires the user to be logged in,
     * otherwise false.
     *
     * @return Description of the Return Value
     */
    protected abstract boolean loggedInUserRequired();


    /**
     * This method should be implemented in the subclass so that it returns
     * true if this particular request requires the user to be logged in,
     * otherwise false.
     *
     * @return Description of the Return Value
     */
    protected abstract boolean userMayUseThisPO();


    /**
     * Saved input and output context, and session data
     */
    protected HttpPresentationComms myComms = null;
    //protected PapillonSessionData mySessionData = null;

						
    /**
     * Gets HttpPresentation object
     *
     * @return The saved comms objects to whichever subclass needs it
     */
    public HttpPresentationComms getComms() {
        return this.myComms;
    }

    /**
     * This implements the run method in HttpPresentation.
     *
     * @param comms Description of the Parameter
     * @throws Exception
     * @throws HttpPresentationException Description of the Exception
     * @throws IOException               Description of the Exception
     */
    public void run(HttpPresentationComms comms) throws HttpPresentationException, IOException, Exception {
        this.myComms = comms;
        initSessionData();

        // Check if the user needs to be logged in for this request.
        if (this.loggedInUserRequired()) {
            checkForUserLogin();                  // This will redirect the user to the login page if necessary
        }

        // After this point, user is logged in if required...

        if (!this.userMayUseThisPO()) {
            userIsNotAuthorized();                // This will redirect the user to the login page if necessary
        }


        HttpPresentationOutputStream out;
        Node document;
        byte[] buffer;

        // setContentType before calling getDocument
        // because getDocument can change the content type
        this.getComms().response.setContentType("text/html");
        this.getComms().response.setHeader("Access-Control-Allow-Origin","*");
        
   /*     if (!this.loggedInUserRequired() && PAGE_EXPIRE_TIME>0) {
            this.getComms().response.setHeader("Cache-Control", "public, s-maxage=" + PAGE_EXPIRE_TIME + ", max-age="+PAGE_EXPIRE_TIME);
        }
        else if (this.loggedInUserRequired()) {
            this.getComms().response.setHeader("Cache-Control", "private, max-age=10800, pre-check=10800");
        } */


        try {
            initPresentationContext();
            document = getDocument();
        } finally {
            flushPresentationContext();
        }

        if (null == document) {
            this.getComms().response.sendError(HttpPresentationResponse.SC_NOT_FOUND, "Page returned a null Document");
        }

        // Preparation de la sortie...
        OutputOptions options = new OutputOptions();
        options.setDropHtmlSpanIds(true);
        options.setXmlEncoding("UTF-8");
        DOMFormatter fFormatter = new DOMFormatter(options);

        buffer = fFormatter.toBytes(document);
        comms.response.setContentLength(buffer.length);
        out = comms.response.getOutputStream();
        out.write(buffer);
        out.flush();
    }


    /**
     * Method to get or create the AgSessionData object from the user session
     * This object is saved in the EbrokerPresentation object
     *
     * @param comms Description of the Parameter
     * @throws HttpPresentationException Description of the Exception
     * @noinspection JavadocReference
     */
    protected void initSessionData() throws HttpPresentationException {

        try {
            Object obj = null;
			if (getComms() != null && getComms().sessionData != null) {
				obj = getComms().sessionData.get(PapillonSessionData.SESSION_KEY);
			}
            // If we found the session data, save it in a private data member
            if (null != obj) {
                this.sessionData = (PapillonSessionData) obj;
            } else if (this.sessionData == null) {
                // If no session data was found, create a new session data instance
                this.sessionData = new PapillonSessionData();

				if (getComms() != null && getComms().sessionData != null) {
					getComms().sessionData.set(PapillonSessionData.SESSION_KEY, this.sessionData);
				}
            }
            com.lutris.http.BasicAuthResult theBasicAuthResult = com.lutris.http.BasicAuth.getAuthentication(this.getComms().request);
            if (theBasicAuthResult!=null) {
                String login = theBasicAuthResult.username;
                String password = theBasicAuthResult.password;
               // PapillonLogger.writeDebugMsg("basic auth: " + login + " password: " + password);
                User theUser = UsersFactory.findUserByLogin(login);
                if (theUser!=null && !theUser.isEmpty()) {
                    if (theUser.HasCorrectPassword(password)) {
                        PapillonLogger.writeDebugMsg("Registered user from BasicAuth " + theUser.getName());
                        this.setUser(theUser);
                    }
                    else {
                        String errorMsg = "Error: Wrong password";
                        //System.out.println(errorMsg);
                        this.getComms().response.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
                        this.responseStatus = HttpPresentationResponse.SC_UNAUTHORIZED;
                        this.responseMessage = errorMsg;
                    }
                }
                else {
                    String errorMsg = "Error: User unknown";
                    //System.out.println(errorMsg);
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
                    this.responseStatus = HttpPresentationResponse.SC_UNAUTHORIZED;
                    this.responseMessage = errorMsg;
              }
            }
            else {
                if (this.getUser() == null || this.getUser().isEmpty()) {
                    User cookieUser = this.getLoginCookieUser();
                    // if the user is registered
                    if (cookieUser != null && !cookieUser.isEmpty()) {
                        PapillonLogger.writeDebugMsg("Registered user from cookie: " + cookieUser.getName());
                        this.setUser(cookieUser);
                    }
                 }
            }
            
        } catch (KeywordValueException ex) {
            System.out.println("Problem getting session data from session: " + ex.getMessage());
        }
    }


    /**
     * Gets the sessionData attribute of the PO object
     *
     * @return The sessionData value
     */
    public PapillonSessionData getSessionData() {
        return this.sessionData;
    }

    protected void initPresentationContext() throws PapillonBusinessException {
        JibikiContext context = new JibikiContext();
        context.set("comms", this.getComms());
        context.set("sessionData", this.getSessionData());
        context.set("url", this.getUrl());
        context.set("absoluteurl", this.getAbsoluteUrl());
        CurrentRequestContext.registerNewPresentationContext(context);
    }

    protected void flushPresentationContext() throws PapillonBusinessException {
        CurrentRequestContext.releasePresentationContext();
    }

    /**
     * converts a table of Strings for writing an URL
     *
     * @param parameter
     * @param table
     * @return a String
     */
    public static String serializeParameterForUrl(String parameter, Collection table) {
        String result = "";
        if (table != null) {
            for (Iterator iter = table.iterator(); iter.hasNext();) {
                result += parameter + "=" + myUrlEncode((String) iter.next()) + "&";
            }
        }
        return result;
    }

    /**
     * Description of the Method
     *
     * @param str Description of the
     *            Parameter
     * @return Description of the
     *         Return Value
     * @throws java.io.UnsupportedEncodingException
     *          Description of the
     *          Exception
     */
    public static String myUrlEncode(String str) {
        if (str != null && !str.equals("")) {
            // WARNING Java 1.4.1 specific.
            try {
                return java.net.URLEncoder.encode(str, "UTF-8");
            } catch (java.io.UnsupportedEncodingException ex) {
                PapillonLogger.writeErrorMsg("UTF8 is not supported ??!!");
            }
        }
        return str;
    }

    // HACK: For proper decoding of UTF-8 urlencoded values
    /**
     * Description of the Method
     *
     * @param name Description of the Parameter
     * @return Description of the Return Value
     * @throws HttpPresentationException Description of the Exception
     */
    public String myGetParameter(String name) throws HttpPresentationException {
        String res = null;
        try {
            String pvalue = this.getComms().request.getParameter(name);
            if (null != pvalue) {
                byte[] pbytes = pvalue.getBytes("ISO-8859-1");
                res = new String(pbytes, "UTF-8");
            }
        } catch (java.io.UnsupportedEncodingException e) {
            throw new HttpPresentationException("UTF-8 encoding is not supported on this plateform.", e);
        }
        return res;
    }

    // HACK: For proper decoding of UTF-8 urlencoded values
    /**
     * Description of the Method
     *
     * @param req  Description of the Parameter
     * @param name Description of the Parameter
     * @return Description of the Return Value
     * @throws HttpPresentationException Description of the Exception
     */
    public static String myGetParameter(HttpServletRequest req, String name) throws HttpPresentationException {
        String res = null;
        try {
            String pvalue = req.getParameter(name);
            if (null != pvalue) {
                byte[] pbytes = pvalue.getBytes("ISO-8859-1");
                res = new String(pbytes, "UTF-8");
            }
        } catch (java.io.UnsupportedEncodingException e) {
            throw new HttpPresentationException("UTF-8 encoding is not supported on this plateform.", e);
        }
        return res;
    }

    // HACK: For proper decoding of UTF-8 urlencoded values
    /**
     * Description of the Method
     *
     * @param name Description of the
     *             Parameter
     * @return Description of the
     *         Return Value
     * @throws java.io.UnsupportedEncodingException
     *                                   Description of the
     *                                   Exception
     * @throws HttpPresentationException Description of the
     *                                   Exception
     */
    public String[] myGetParameterValues(String name)
            throws java.io.UnsupportedEncodingException, HttpPresentationException {
        try {
            String[] res = this.getComms().request.getParameterValues(name);
            if (null != res && res.length > 0) {
                for (int i = 0; i < res.length; i++) {
                    byte[] pbytes = res[i].getBytes("ISO-8859-1");
                    res[i] = new String(pbytes, "UTF-8");
              }
            }
            return res;
        } catch (java.io.UnsupportedEncodingException e) {
            throw new HttpPresentationException("UTF-8 encoding is not supported on this plateform.", e);
        }
    }

    // HACK: For proper decoding of UTF-8 urlencoded values
    /**
     * Description of the Method
     *
     * @param req  Description of the
     *             Parameter
     * @param name Description of the
     *             Parameter
     * @return Description of the
     *         Return Value
     * @throws java.io.UnsupportedEncodingException
     *                                   Description of the
     *                                   Exception
     * @throws HttpPresentationException Description of the
     *                                   Exception
     */
    public static String[] myGetParameterValues(HttpServletRequest req, String name)
            throws java.io.UnsupportedEncodingException, HttpPresentationException {
        try {
            String[] res = req.getParameterValues(name);
            if (null != res && res.length > 0) {
                for (int i = 0; i < res.length; i++) {
                    byte[] pbytes = res[i].getBytes("ISO-8859-1");
                    res[i] = new String(pbytes, "UTF-8");
                 }
            }
            return res;
        } catch (java.io.UnsupportedEncodingException e) {
            throw new HttpPresentationException("UTF-8 encoding is not supported on this plateform.", e);
        }
    }

    /**
     * Sets the selected attribute of the PapillonBasePO class
     *
     * @param mySelect The new selected value
     * @param myArray  The new selected value
     */
    public static void setSelected(XHTMLSelectElement mySelect, String[] myArray) {
        Vector myVector = new Vector();
        myVector.addAll(Arrays.asList(myArray));
        setSelected(mySelect, myVector);
    }


    /**
     * Sets the selected attribute of the PapillonBasePO class
     *
     * @param mySelect The new selected value
     * @param myValue  The new selected value
     */
    public static void setSelected(XHTMLSelectElement mySelect, String myValue) {
        if (myValue != null && !myValue.equals("")) {
            HTMLCollection myCollection = mySelect.getOptions();
            int i = 0;
            while (i < myCollection.getLength()) {
                if (((XHTMLOptionElement) myCollection.item(i)).getValue().equals(myValue)) {
                    // This method does not work any more with enhydra5.1...
                    // mySelect.setSelectedIndex(i);
                    ((XHTMLOptionElement) myCollection.item(i)).setSelected(true);
                    break;
                }
                i++;
            }
        }
    }


    /**
     * Sets the selected attribute of the PapillonBasePO class
     *
     * @param mySelect The new selected value
     * @param myValue  The new selected value
     */
    public static void setSelected(HTMLSelectElement mySelect, String myValue) {
        if (myValue != null && !myValue.equals("")) {
            HTMLCollection myCollection = mySelect.getOptions();
            int i = 0;
            while (i < myCollection.getLength()) {
                if (((HTMLOptionElement) myCollection.item(i)).getValue().equals(myValue)) {
                    // This method does not work any more with enhydra5.1...
                    // mySelect.setSelectedIndex(i);
                    ((HTMLOptionElement) myCollection.item(i)).setSelected(true);
                    break;
                }
                i++;
            }
        }
    }


    /**
     * Sets the selected attribute of the PapillonBasePO class
     *
     * @param mySelect The new selected value
     * @param myVector The new selected value
     */
    public static void setSelected(XHTMLSelectElement mySelect, Vector myVector) {
        if (myVector != null && myVector.size() > 0) {
            HTMLCollection myCollection = mySelect.getOptions();
            int i = 0;
            while (i < myCollection.getLength() && myVector.size() > 0) {
                XHTMLOptionElement myOptionElement = (XHTMLOptionElement) myCollection.item(i);
                String myOption = myOptionElement.getValue();
                if (myVector.contains(myOption)) {
                    myOptionElement.setSelected(true);
                    myVector.remove(myOption);
                }
                i++;
            }
        }
    }

    /**
     * Sets the selected attribute of the PapillonBasePO class
     *
     * @param mySelect The new selected value
     */
    public static void deselectAll(XHTMLSelectElement mySelect) {
        HTMLCollection myCollection = mySelect.getOptions();
        int i = 0;
        while (i < myCollection.getLength()) {
            ((HTMLOptionElement) myCollection.item(i)).setSelected(false);
            i++;
        }
    }


    /**
     * Sets the unicodeLabels attribute of the PapillonBasePO class
     *
     * @param mySelect The new unicodeLabels value
     */
    public static void setUnicodeLabels(XHTMLSelectElement mySelect) {
        HTMLCollection myCollection = mySelect.getOptions();
        for (int i = 0; i < myCollection.getLength(); i++) {
            XHTMLOptionElement myOption = (XHTMLOptionElement) myCollection.item(i);
            myOption.setLabel(myOption.getText());
        }
    }


    /**
     * Gets the url attribute of the PO object
     *
     * @return The url value
     */
			protected String getUrl() {
				String className = this.getClass().getName();
				return className.substring(className.lastIndexOf(".") + 1) + ".po";
			}
			
			protected String getAbsoluteUrl() {
				String className = this.getClass().getName();
				className = className.replace('.','/');
				String applicationPrefix = ((Papillon)Enhydra.getApplication()).getApplicationPrefix();
				if (applicationPrefix != null && applicationPrefix.endsWith("/")) {
					applicationPrefix = applicationPrefix.substring(0,applicationPrefix.length()-1);
				}
				String presentationPrefix = "";
				try {
					presentationPrefix = Enhydra.getApplication().getConfig().getString("Server.PresentationPrefix");
				}
				catch (com.lutris.util.ConfigException ce) {
					PapillonLogger.writeDebugMsg("com.lutris.util.ConfigException: ");
					PapillonLogger.writeDebugMsg(ce.toString());
				}
				if (className.indexOf(presentationPrefix)==0) {
					className = className.substring(presentationPrefix.length());
				}
				if (className != null && !className.startsWith("/")) {
					className += "/";
				}
				return applicationPrefix + className + ".po";
			}
			
			
    /**
     * Description of the Method
     *
     * @throws ClientPageRedirectException Description of the Exception
     * @throws HttpPresentationException   Description of the Exception
     */
    protected void checkForUserLogin() throws ClientPageRedirectException, HttpPresentationException {
        try {

            if (!this.getSessionData().userHasBeenIdentified()) {
                //send to LoginPage if a logged in user is required.
                String requestedPO = this.getComms().request.getRequestURI();
                throw new ClientPageRedirectException(
                        LOGIN_PAGE + "?" + DESTINATION_AFTER_LOGIN_PARAMETER + "=" + myUrlEncode(requestedPO));
            }
        } catch (Exception ex) {
            throw new HttpPresentationException("Trouble checking for user login status", ex);
        }
    }


    /**
     * Description of the Method
     *
     * @throws ClientPageRedirectException Description of the Exception
     * @throws HttpPresentationException   Description of the Exception
     */
    protected void userIsNotAuthorized() throws ClientPageRedirectException, HttpPresentationException {
        this.getSessionData().writeUserMessage("You are not authorized to view the requested page.");
        this.getSessionData().writeUserMessage("Please log in as an authorized user to proceed.");

        try {
            String requestedPO = this.getComms().request.getRequestURI();
            throw new ClientPageRedirectException(
                    LOGIN_PAGE + "?" + DESTINATION_AFTER_LOGIN_PARAMETER + "=" + myUrlEncode(requestedPO));
        } catch (Exception ex) {
            throw new HttpPresentationException("Trouble checking for user login status", ex);
        }
    }

    /**
     * Sets the user into the session
     *
     * @param theUser The new user value
     * @throws PapillonPresentationException
     */
    public void setUser(User theUser) throws PapillonPresentationException {
        try {
            if (this.myComms.session!= null) {
                this.myComms.session.setUser(theUser);
            }
        } catch (com.lutris.appserver.server.session.SessionException SesEx) {
            throw new PapillonPresentationException("PapillonBasePO:Session Error for setUser: ", SesEx);
        }
        this.getSessionData().setUser(theUser);
    }


    /**
     * Gets the user from the session
     *
     * @return the person object in the session
     */
    public User getUser() {
        return this.getSessionData().getUser();
    }

    /**
     *  Gets the loginCookieUser attribute of the PapillonBasePO object
     *
     * @return                                The loginCookieUser value
     * @exception  HttpPresentationException  Description of the Exception
     */
    public User getLoginCookieUser()
    throws HttpPresentationException {
        User cookieUser = null;
        Cookie[] myCookies = this.getCookies();
        int i = 0;
        while (i < myCookies.length && cookieUser == null) {
            Cookie myCookie = myCookies[i];
            if (myCookie.getName().equals(LOGIN_COOKIE)) {
                cookieUser = UsersFactory.findUserById(myCookie.getValue());
                if (cookieUser ==null || cookieUser.isEmpty()) {
                    String errorMsg = "Error: User unknown";
                //System.out.println(errorMsg);
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
                }
            }
            i++;
        }
        return cookieUser;
    }
            
            
            /**
             *  Gets the cookies attribute of the PapillonBasePO object
             *
             * @return                                The cookies value
             * @exception  HttpPresentationException  Description of the Exception
             */
            public Cookie[] getCookies()
            throws HttpPresentationException {
                return this.getComms().request.getCookies();
            }
            


    /**
     * Method to remove the current user from the session
     *
     * @throws PapillonPresentationException Description of the Exception
     */
    public void removeUserFromSession() throws PapillonPresentationException {
        try {
            this.myComms.session.clearUser();
            User newUser = new User();
            newUser.setName(this.myComms.request.getRemoteHost());
            newUser.setLogin("guest");
            newUser.setEmail(this.myComms.request.getRemoteUser() + "@" + this.myComms.request.getRemoteAddr());
            this.myComms.session.setUser(newUser);
        } catch (com.lutris.appserver.server.session.SessionException SesEx) {
            throw new PapillonPresentationException("PapillonBasePO:Session Error for clearUser: ", SesEx);
        } catch (com.lutris.appserver.server.httpPresentation.HttpPresentationException HttpEx) {
            throw new PapillonPresentationException("PapillonBasePO: Error for request.get...: ", HttpEx);
        }
        this.getSessionData().removeUser();
        }
    
}
