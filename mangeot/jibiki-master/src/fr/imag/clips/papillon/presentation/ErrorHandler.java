/*
 * Enhydra Java Application Server Project
 * 
 * The contents of this file are subject to the Enhydra Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License on
 * the Enhydra web site ( http://www.enhydra.org/ ).
 * 
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See 
 * the License for the specific terms governing rights and limitations
 * under the License.
 * 
 * The Initial Developer of the Enhydra Application Server is Lutris
 * Technologies, Inc. The Enhydra Application Server and portions created
 * by Lutris Technologies, Inc. are Copyright Lutris Technologies, Inc.
 * All Rights Reserved.
 * 
 * Contributor(s):
 * 
 * $Id: ErrorHandler.java,v 1.1 2003/11/25 18:17:31 slobodan Exp $
 */

package fr.imag.clips.papillon.presentation;

import com.lutris.logging.*;
import com.lutris.appserver.server.httpPresentation.*;
import com.lutris.appserver.server.*;

import com.lutris.appserver.server.Enhydra;
import fr.imag.clips.papillon.Papillon;

import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.business.PapillonLogger;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

import org.enhydra.xml.io.OutputOptions;
import org.enhydra.xml.io.DOMFormatter;


import java.io.*;
/*
 import java.io.BufferedReader;
 import java.io.IOException;
 import java.io.InputStream;
 import java.io.InputStreamReader;*/

import java.util.Properties;


/**
 * Class to handle exceptions not caught anywhere else in the framework of
 * our application
 *
 * @author
 * @version
 */
public class ErrorHandler extends  fr.imag.clips.papillon.presentation.AbstractPO {
	
    private static final String API_COMMAND = "api";
    private static final String APIUSERS_COMMAND = "apiusers";
    protected final static String JSON_CONTENTTYPE = "text/json";
    protected final static String XML_CONTENTTYPE = "text/xml";
	
	protected final static String ERROR_PAGE = "<?xml version='1.0'?><html></html>";
    protected final static String ENHYDRA_SESSION_COOKIE = "JSESSIONID";
	
    private org.w3c.dom.Document content;
    private ErrorXHTML XHTMLcontent;
    private String jsonString;
    private String acceptContentType = XML_CONTENTTYPE;
    private String sentContentType = XML_CONTENTTYPE;
    


    /**
     * Description of the Method
     *
     * @return Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return false;
    }
	
	
    /**
     * Description of the Method
     *
     * @return Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        return true;
    }
	
	/**
     * This implements the run method in HttpPresentation.
     *
     * @param HttpPresentationComms
     * @exception HttpPresentationException
     */
    public org.w3c.dom.Node getDocument()
	throws HttpPresentationException, java.io.IOException, java.lang.Exception {
		
		
		////// Create Home page
		content = XMLServices.buildDOMTree(ERROR_PAGE);
		
		// special code for handling the root query (/api/)
		// because it is not handled by the api/ErrorHandler.po
        String command = "";
        int status = HttpPresentationResponse.SC_OK;
		
         String prefix = this.getAbsoluteUrl();
		prefix = prefix.substring(0,prefix.lastIndexOf('/') + 1);
        String apiPrefix = prefix + API_COMMAND + "/";
        String apiUsersPrefix = prefix + APIUSERS_COMMAND + "/";
		HttpPresentationRequest theRequest = this.getComms().request;
		HttpPresentationResponse theResponse = this.getComms().response;
        if (null != theRequest.getHeader("Accept") && theRequest.getHeader("Accept").equals("application/json")) {
            acceptContentType = JSON_CONTENTTYPE;
        }
        if (null != theRequest.getHeader("Content-Type") && theRequest.getHeader("Content-Type").startsWith("application/json")) {
            sentContentType = JSON_CONTENTTYPE;
        }
        
        String theURI = theRequest.getPresentationURI();
        if (theURI.indexOf(apiPrefix)==0) {
            theURI = theURI.substring(apiPrefix.length());
            command = API_COMMAND;
        }
        if (theURI.indexOf(apiUsersPrefix)==0) {
            theURI = theURI.substring(apiUsersPrefix.length());
            command = APIUSERS_COMMAND;
        }
        //PapillonLogger.writeDebugMsg("ERROR HANDLER: "+ theRequest.getPresentationURI()+" Accept: "+acceptContentType+" ; command:" + command);
		if ((theURI == null || theURI.equals(""))
		   && (null != this.getComms().exception) 
		   && (this.getComms().exception instanceof com.lutris.appserver.server.httpPresentation.FilePresentationException)) {
            if (this.responseStatus != HttpPresentationResponse.SC_OK) {
                PapillonLogger.writeDebugMsg(this.responseMessage);
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error: " + this.responseStatus + " Unauthorized</h1><p>" + this.responseMessage + "</p></html>");
            }
            else if (command.equals(API_COMMAND)) {
                PapillonLogger.writeDebugMsg("REST API COMMAND: " + theRequest.getMethod() + " DICTLIST");
                if (theRequest.getMethod().equals("GET")) {
                    java.util.Vector responseVector = fr.imag.clips.papillon.business.dictionary.MetadataApi.getDictionaryList(this.getUser());
                    content = (org.w3c.dom.Document) responseVector.elementAt(0);
                    status = ((Integer)responseVector.elementAt(1)).intValue();
                    theResponse.setStatus(status, (String) responseVector.elementAt(2));
                }
                else if (theRequest.getMethod().equals("PUT")) {
                    HttpPresentationInputStream inputStream = theRequest.getInputStream();
                    String dict = convertStreamToString(inputStream);
                    PapillonLogger.writeDebugMsg("Error: put dictlist: not implemented");
                    theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED);
                }
                else if (theRequest.getMethod().equals("POST")) {
                    HttpPresentationInputStream inputStream = theRequest.getInputStream();
                    String dict = convertStreamToString(inputStream);
                    PapillonLogger.writeDebugMsg("Error: post dictlist: not implemented");
                    theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED);
                }
                else if (theRequest.getMethod().equals("DELETE")) {
                    PapillonLogger.writeDebugMsg("Error: delete dictlist: not implemented");
                    theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED);
                }
            }
                else if (command.equals(APIUSERS_COMMAND)) {
                PapillonLogger.writeDebugMsg("REST APIUSERS COMMAND: " + theRequest.getMethod() + " USERSLIST;");
                if (theRequest.getMethod().equals("GET")) {
                    java.util.Vector responseVector = fr.imag.clips.papillon.business.user.UserApi.getUserList(this.getUser());
                    content = (org.w3c.dom.Document) responseVector.elementAt(0);
                    status = ((Integer)responseVector.elementAt(1)).intValue();
                    theResponse.setStatus(status, (String) responseVector.elementAt(2));
                }
                else if (theRequest.getMethod().equals("PUT")) {
                    HttpPresentationInputStream inputStream = theRequest.getInputStream();
                    String parcel = convertStreamToString(inputStream);
                    PapillonLogger.writeDebugMsg("Error: put users list: not implemented");
                    theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED);
                }
                else if (theRequest.getMethod().equals("POST")) {
                    HttpPresentationInputStream inputStream = theRequest.getInputStream();
                    String parcel = convertStreamToString(inputStream);
                    PapillonLogger.writeDebugMsg("Error: post users list: not implemented");
                    theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED);
                    
                }
                else if (theRequest.getMethod().equals("DELETE")) {
                    PapillonLogger.writeDebugMsg("Error: delete users list: not implemented");
                    theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED);
                }
                
                }
            if (acceptContentType.equals(JSON_CONTENTTYPE)) {
                try {
                    String xmlString = XMLServices.NodeToString(content);
                    org.json.JSONObject xmlJSONObj = org.json.XML.toJSONObject(xmlString);
                    jsonString = xmlJSONObj.toString(2);
                } catch (org.json.JSONException je) {
                    theResponse.setStatus(HttpPresentationResponse.SC_INTERNAL_SERVER_ERROR,je.toString());
                    jsonString = je.toString();
                }
            }
			}
			else {
				////// Create Home page
				XHTMLcontent = (ErrorXHTML) MultilingualXHtmlTemplateFactory.createTemplate("ErrorXHTML",
																					   this.getComms(), this.getSessionData());
				//        ErrorHTML errorPage = new ErrorHTML();
				
				if(null != this.getComms() && null != this.getComms().exception) {
					StringWriter stringWriter = new StringWriter();
					this.getComms().exception.printStackTrace(new PrintWriter(stringWriter));
					LogChannel logChannel = Enhydra.getLogChannel();
					int level = logChannel.getLevel("DEBUG");
					
					logChannel.write(level, "jibiki.presentation.ErrorHandler stack trace = ");
					logChannel.write(level, stringWriter.toString());
					logChannel.write(level, "jibiki.presentation.ErrorHandler caught an exception - " 
									 + this.getComms().exception.toString(), this.getComms().exception);
					XHTMLcontent.setTextStackTrace(stringWriter.toString());
					if (this.getComms().exception.getMessage() != null) {
						XHTMLcontent.setTextErrorMessage(this.getComms().exception.getMessage());
					}
					else {
						XHTMLcontent.setTextErrorMessage(this.getComms().exception.toString());
					}
					
				}
				else {
					XHTMLcontent.setTextErrorMessage("Comms or comms.exception is null");
				}
				content = (org.w3c.dom.Document) XHTMLcontent;
			}
		return content;
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
        
        if (PAGE_EXPIRE_TIME>0) {
            this.myComms.response.setHeader("Cache-Control", "s-maxage=" + PAGE_EXPIRE_TIME);
        }
        
        // code spécial pour récupérer la session active car il semble qu'il n'y ait pas de session attachée au ErrorHandler !
        if (this.myComms != null && this.myComms.sessionData == null) {
            javax.servlet.http.Cookie[] cookiesArray = this.myComms.request.getCookies();
            //PapillonLogger.writeDebugMsg("Cookies:");
            for (int i=0; i<cookiesArray.length;i++) {
                //PapillonLogger.writeDebugMsg(cookiesArray[i].getName() + ":" + cookiesArray[i].getValue());
                if (cookiesArray[i].getName().equals(ENHYDRA_SESSION_COOKIE)) {
                    this.myComms.session = PapillonSessionManager.getSession(cookiesArray[i].getValue());
                    if (this.myComms.session!=null) {
                        //PapillonLogger.writeDebugMsg("The Session not null");
                        this.myComms.sessionData = this.myComms.session.getSessionData();
                        if (this.myComms.sessionData != null) {
                            //PapillonLogger.writeDebugMsg("theSessionData not null");
                            Object obj = this.myComms.sessionData.get(PapillonSessionData.SESSION_KEY);
                            if (null != obj) {
                                this.sessionData = (PapillonSessionData) obj;
                            }
                        }
                        else {
                            //PapillonLogger.writeDebugMsg("theSessionData null");
                        }
                    }
                }
            }
        }
        
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
        org.w3c.dom.Node document;
        byte[] buffer;
        
        // setContentType before calling getDocument
        // because getDocument can change the content type
        this.getComms().response.setContentType(acceptContentType);
        this.getComms().response.setEncoding("UTF-8");
        this.getComms().response.setHeader("Access-Control-Allow-Origin","*");
        this.getComms().response.setHeader("Access-Control-Allow-Headers","Origin,Content-Type,Accept,Authorization");
        try {
            initPresentationContext();
            document = getDocument();
        } finally {
            flushPresentationContext();
        }
        
        if (acceptContentType.equals(JSON_CONTENTTYPE)) {
            buffer = jsonString.getBytes("UTF-8");
        }
        else {
            if (null == document) {
                this.getComms().response.sendError(HttpPresentationResponse.SC_NOT_FOUND, "Page returned a null Document");
            }
            // Preparation de la sortie...
            OutputOptions options = new OutputOptions();
            options.setDropHtmlSpanIds(true);
            options.setXmlEncoding("UTF-8");
            DOMFormatter fFormatter = new DOMFormatter(options);
            buffer = fFormatter.toBytes(document);
        }
        
        comms.response.setContentLength(buffer.length);
        out = comms.response.getOutputStream();
        out.write(buffer);
        out.flush();
    }

    
    
	public static String convertStreamToString(InputStream is) {
		/*
		 * To convert the InputStream to String we use the BufferedReader.readLine()
		 * method. We iterate until the BufferedReader return null which means
		 * there's no more data to read. Each line will appended to a StringBuilder
		 * and returned as String.
		 */
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		StringBuilder sb = new StringBuilder();
		
		String line = null;
		try {
			while ((line = reader.readLine()) != null) {
				sb.append(line + "\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		} 
		/*
		 finally {
		 try {
		 is.close();
		 } catch (IOException e) {
		 // HttpPresentationInputStream may not be closed error
		 //e.printStackTrace();
		 }
		 } */
		
		return sb.toString();
	}
	
}
