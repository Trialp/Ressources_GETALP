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

package fr.imag.clips.papillon.presentation.apiusers;

import com.lutris.logging.*;
import com.lutris.appserver.server.httpPresentation.*;
import com.lutris.appserver.server.*;

import com.lutris.appserver.server.Enhydra;
import fr.imag.clips.papillon.Papillon;

import fr.imag.clips.papillon.business.xml.XMLServices;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.user.Group;
import fr.imag.clips.papillon.business.user.GroupApi;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.UserApi;
import fr.imag.clips.papillon.business.user.UsersFactory;
import fr.imag.clips.papillon.presentation.PapillonSessionData;
import fr.imag.clips.papillon.presentation.PapillonSessionManager;

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
public class ErrorHandler extends fr.imag.clips.papillon.presentation.AbstractPO {
 
	protected final static String ERROR_PAGE = "<?xml version='1.0'?><html></html>";
	protected final static String LOGIN_PARAMETER = "login";
    protected final static String PASSWORD_PARAMETER = "password";
	protected final static String STRATEGY_PARAMETER = "strategy";
	protected final static String LIMIT_PARAMETER = "count";
    protected final static String OFFSET_PARAMETER = "startIndex";
    protected final static String ORDERBY_PARAMETER = "sortBy";
    protected final static String JSON_CONTENTTYPE = "text/json";
    protected final static String XML_CONTENTTYPE = "text/xml";
    protected final static String ENHYDRA_SESSION_COOKIE = "JSESSIONID";
 
    protected final static int PAGE_EXPIRE_TIME = ((Papillon) Enhydra.getApplication()).getPageExpireTime();
   
    private final static String USERS_OBJECT = "users";
    private final static String GROUPS_OBJECT = "groups";
    private final static String DICTIONARY_OBJECT = "dictionary";
    private final static String DICTIONARIES_OBJECT = "dictionaries";

    private org.w3c.dom.Document content;
    private String acceptContentType = XML_CONTENTTYPE;
    private String sentContentType = XML_CONTENTTYPE;
    private String jsonString = "";
    
	
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
        int status = HttpPresentationResponse.SC_OK;
        String prefix = this.getAbsoluteUrl();
		prefix = prefix.substring(0,prefix.lastIndexOf('/') + 1);
			
        if (null != this.getComms().exception) {
			if (this.getComms().exception instanceof com.lutris.appserver.server.httpPresentation.FilePresentationException) {
				HttpPresentationRequest theRequest = this.getComms().request;
				HttpPresentationResponse theResponse = this.getComms().response;
				String login =  myGetParameter(LOGIN_PARAMETER);
                String password = myGetParameter(PASSWORD_PARAMETER);
                
                setUserFromLoginPassword(login,password);
                
                if (null != theRequest.getHeader("Accept") && theRequest.getHeader("Accept").equals("application/json")) {
                    acceptContentType = JSON_CONTENTTYPE;
                }
                if (null != theRequest.getHeader("Content-Type") && theRequest.getHeader("Content-Type").startsWith("application/json")) {
                    sentContentType = JSON_CONTENTTYPE;
                }
                //PapillonLogger.writeDebugMsg("sentContentType: " + sentContentType);
                //PapillonLogger.writeDebugMsg("acceptContentType: " + acceptContentType);
              
                PapillonLogger.writeDebugMsg("REST APIUSERS URI : [" + prefix + "] " + theRequest.getPresentationURI()+" Accept: "+theRequest.getHeader("Accept")+" ;");
				String theURI = java.net.URLDecoder.decode(theRequest.getPresentationURI());
				if (theURI.indexOf(prefix)==0) {
					theURI = theURI.substring(prefix.length());
				}
				String[] restStrings = null;
				if (theURI!=null &&!theURI.equals("")) {
					restStrings = theURI.split("/");
				}
				
				String commande = "REST APIUSERS COMMAND: " + theRequest.getMethod();
                String object = "";
                if (restStrings!= null && restStrings.length>0) {
                    object = restStrings[0];
                }
                // Si l'authentificaiton a échoué et ce n'est pas la commande pour créer un nouvel utilisateur
                if (this.responseStatus != HttpPresentationResponse.SC_OK && !(theRequest.getMethod().equals("POST") && object.equals(USERS_OBJECT) && restStrings.length==2)) {
                    PapillonLogger.writeDebugMsg(this.responseMessage);
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + this.responseStatus + " Unauthorized</h1><p>" + this.responseMessage + "</p></html>");
                }
                else if (restStrings== null || restStrings.length==0) {
					PapillonLogger.writeDebugMsg(commande + " USERSLIST;");
					if (theRequest.getMethod().equals("GET")) {
                        java.util.Vector responseVector = UserApi.getUserList(this.getUser());
                        content = (org.w3c.dom.Document) responseVector.elementAt(0);
                        status = ((Integer)responseVector.elementAt(1)).intValue();
                        theResponse.setStatus(status, (String) responseVector.elementAt(2));
					}
                    else if (theRequest.getMethod().equals("OPTIONS")) {
                        theResponse.setHeader("Access-Control-Allow-Methods","GET, OPTIONS");
                        theResponse.setHeader("Allow","GET, OPTIONS");
                    }
                    else {
                        String errorMsg = "Error: method not implemented";
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_IMPLEMENTED + "</h1><p>" + errorMsg + "</p></html>");
                        theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED, errorMsg);
                    }
				}
				else if (restStrings.length==1) {
                    if (theRequest.getMethod().equals("GET")) {
                        PapillonLogger.writeDebugMsg(commande + " OBJECT: " + object + ";");
                        content = null;
                        
                        if (object.equalsIgnoreCase(USERS_OBJECT)) {
                            java.util.Vector responseVector = UserApi.getUserList(this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else if (object.equalsIgnoreCase(GROUPS_OBJECT)) {
                            java.util.Vector responseVector = GroupApi.getGroupsList(this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else {
                            String errorMsg = "Error: object: " + object + " does not exist!";
                            PapillonLogger.writeDebugMsg(errorMsg);
                            theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                        }
                    }
                    else if (theRequest.getMethod().equals("OPTIONS")) {
                        theResponse.setHeader("Access-Control-Allow-Methods","GET, OPTIONS");
                        theResponse.setHeader("Allow","GET, OPTIONS");
                    }
                    else {
                        String errorMsg = "Error: method not implemented";
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_IMPLEMENTED + "</h1><p>" + errorMsg + "</p></html>");
                        theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED, errorMsg);
                    }
                }
                else if (restStrings.length==2) {
                    String objectName = restStrings[1];
                    PapillonLogger.writeDebugMsg(commande + " OBJECT: " + object + " name: " + objectName +";");
                    if (theRequest.getMethod().equals("GET")) {
                        content = null;
                        
                        if (object.equalsIgnoreCase(USERS_OBJECT)) {
                            java.util.Vector responseVector = UserApi.getUser(objectName, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else if (object.equalsIgnoreCase(GROUPS_OBJECT)) {
                            java.util.Vector responseVector = GroupApi.getGroup(objectName, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else if (object.equalsIgnoreCase(DICTIONARY_OBJECT)) {
                            java.util.Vector responseVector = UserApi.getRolesForDictionary(objectName, null, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else {
                            String errorMsg = "Error: object: " + object + " does not exist!";
                            PapillonLogger.writeDebugMsg(errorMsg);
                            theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                        }
                    }
                    // TODO : pour le PUT, trouver une solution pour changer de mot de passe !
                    else if (theRequest.getMethod().equals("POST")) {
                        HttpPresentationInputStream inputStream = theRequest.getInputStream();
                        String objectBody = convertStreamToString(inputStream);
                        content = null;
                        
                        if (object.equalsIgnoreCase(USERS_OBJECT)) {
                            com.lutris.http.BasicAuthResult theBasicAuthResult = com.lutris.http.BasicAuth.getAuthentication(theRequest);
                            if (theBasicAuthResult!=null) {
                                login = theBasicAuthResult.username;
                                password = theBasicAuthResult.password;
                            }
                            java.util.Vector responseVector = UserApi.postUser(login, password, objectName, objectBody, sentContentType, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else {
                            String errorMsg = "Error: object: " + object + " does not exist!";
                            PapillonLogger.writeDebugMsg(errorMsg);
                            theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                        }
                        theResponse.flush();
                    }
                    else if (theRequest.getMethod().equals("PUT")) {
                        HttpPresentationInputStream inputStream = theRequest.getInputStream();
                        String objectBody = convertStreamToString(inputStream);
                        content = null;
                        
                        if (object.equalsIgnoreCase(USERS_OBJECT)) {
                            com.lutris.http.BasicAuthResult theBasicAuthResult = com.lutris.http.BasicAuth.getAuthentication(theRequest);
                            if (theBasicAuthResult!=null) {
                                login = theBasicAuthResult.username;
                                password = theBasicAuthResult.password;
                            }
                            java.util.Vector responseVector = UserApi.putUser(login, password, objectName, objectBody, sentContentType, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else {
                            String errorMsg = "Error: object: " + object + " does not exist!";
                            PapillonLogger.writeDebugMsg(errorMsg);
                            theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                        }
                        theResponse.flush();
                    }
                    else if (theRequest.getMethod().equals("DELETE")) {
                        content = null;
                        
                        if (object.equalsIgnoreCase(USERS_OBJECT)) {
                            java.util.Vector responseVector = UserApi.deleteUser(objectName, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else {
                            String errorMsg = "Error: object: " + object + " does not exist!";
                            PapillonLogger.writeDebugMsg(errorMsg);
                            theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                        }
                    }
                    else if (theRequest.getMethod().equals("OPTIONS")) {
                        theResponse.setHeader("Access-Control-Allow-Methods","GET, OPTIONS");
                        theResponse.setHeader("Allow","GET, OPTIONS");
                    }
                    else {
                        String errorMsg = "Error: method not implemented";
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_IMPLEMENTED + "</h1><p>" + errorMsg + "</p></html>");
                        theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED, errorMsg);
                    }
                }
                else if (restStrings.length==3) {
                    String objectName = restStrings[1];
                    String attributeName = restStrings[2];
                    PapillonLogger.writeDebugMsg(commande + " OBJECT: " + object + " name: " + objectName +", ATTRIBUTE: " + attributeName + ";");
                    if (theRequest.getMethod().equals("GET")) {
                        content = null;
                        if (object.equalsIgnoreCase(USERS_OBJECT)) {
                            java.util.Vector responseVector = UserApi.getGroupsForUser(objectName, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
 /*                       else if (object.equalsIgnoreCase(GROUPS_OBJECT)) {
                            java.util.Vector responseVector = GroupApi.getGroup(objectName, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }*/
                        else if (object.equalsIgnoreCase(DICTIONARY_OBJECT)) {
                            java.util.Vector responseVector = UserApi.getRolesForDictionary(objectName, attributeName, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else {
                            String errorMsg = "Error: object: " + object + " does not exist!";
                            PapillonLogger.writeDebugMsg(errorMsg);
                            theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                        }
                    }
                    else if (theRequest.getMethod().equals("PUT")) {
                        content = null;
                        HttpPresentationInputStream inputStream = theRequest.getInputStream();
                        String objectBody = convertStreamToString(inputStream);
                        /*if (object.equalsIgnoreCase(USERS_OBJECT)) {
                            java.util.Vector responseVector = UserApi.getGroupsForUser(objectName, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }
                        else */if (object.equalsIgnoreCase(GROUPS_OBJECT)) {
                            if (attributeName !=null && attributeName.equals(USERS_OBJECT)) {
                                java.util.Vector responseVector = UserApi.putUsersInGroup(objectBody, objectName, sentContentType, this.getUser());
                                content = (org.w3c.dom.Document) responseVector.elementAt(0);
                                status = ((Integer)responseVector.elementAt(1)).intValue();
                                theResponse.setStatus(status, (String) responseVector.elementAt(2));
                            }
                            else {
                                String errorMsg = "Error: attribute: " + attributeName + " does not exist!";
                                PapillonLogger.writeDebugMsg(errorMsg);
                                theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                            }
                         }/*
                        else if (object.equalsIgnoreCase(DICTIONARY_OBJECT)) {
                            java.util.Vector responseVector = UserApi.getGroupsForDictionary(objectName, attributeName, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        }*/
                        else {
                            String errorMsg = "Error: object: " + object + " does not exist!";
                            PapillonLogger.writeDebugMsg(errorMsg);
                            theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                        }
                        theResponse.flush();
                    }
                    else if (theRequest.getMethod().equals("OPTIONS")) {
                        theResponse.setHeader("Access-Control-Allow-Methods","GET, PUT, OPTIONS");
                        theResponse.setHeader("Allow","GET, PUT, OPTIONS");
                    }
                    else {
                        String errorMsg = "Error: method not implemented";
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_IMPLEMENTED + "</h1><p>" + errorMsg + "</p></html>");
                        theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED, errorMsg);
                    }
                }
                else if (restStrings.length==4) {
                    String objectName = restStrings[1];
                    String secondObject = restStrings[2];
                    String secondObjectName = restStrings[3];
                    PapillonLogger.writeDebugMsg(commande + " OBJECT: " + object + " name: " + objectName +", secondObject: " + secondObject + " second name: " + secondObjectName);
                    if (theRequest.getMethod().equals("PUT")) {
                        content = null;
                        if (object.equalsIgnoreCase(USERS_OBJECT)) {
                            if (secondObject.equalsIgnoreCase(GROUPS_OBJECT)) {
                                java.util.Vector responseVector = UserApi.putUserInGroup(objectName, secondObjectName, this.getUser());
                                content = (org.w3c.dom.Document) responseVector.elementAt(0);
                                status = ((Integer)responseVector.elementAt(1)).intValue();
                                theResponse.setStatus(status, (String) responseVector.elementAt(2));
                            }
                            else {
                                String errorMsg = "Error: second object: " + secondObject + " does not exist!";
                                PapillonLogger.writeDebugMsg(errorMsg);
                                theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                            }
                        }
                        else if (object.equalsIgnoreCase(GROUPS_OBJECT)) {
                            if (secondObject.equalsIgnoreCase(USERS_OBJECT)) {
                                java.util.Vector responseVector = UserApi.putUserInGroup(secondObjectName, objectName, this.getUser());
                                content = (org.w3c.dom.Document) responseVector.elementAt(0);
                                status = ((Integer)responseVector.elementAt(1)).intValue();
                                theResponse.setStatus(status, (String) responseVector.elementAt(2));
                            }
                            else {
                                String errorMsg = "Error: second object: " + secondObject + " does not exist!";
                                PapillonLogger.writeDebugMsg(errorMsg);
                                theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                            }
                        }
                        else if (object.equalsIgnoreCase(DICTIONARY_OBJECT)) {
                            if (secondObject.equalsIgnoreCase(Group.ADMIN_GROUP) ||
                                secondObject.equalsIgnoreCase(Group.VALIDATOR_GROUP) ||
                                secondObject.equalsIgnoreCase(Group.SPECIALIST_GROUP) ||
                                secondObject.equalsIgnoreCase(Group.READER_GROUP)) {
                                String groupName = secondObject + Group.DICT_GROUP_PREFIX + objectName;
                                //if dictionary group does not exist
                                java.util.Vector responseVector = GroupApi.getGroup(groupName, this.getUser());
                                if (((Integer)responseVector.elementAt(1)).intValue() != HttpPresentationResponse.SC_OK) {
                                    responseVector = GroupApi.postGroup(groupName, this.getUser());
                                }
                                responseVector = UserApi.putUserInGroup(secondObjectName, groupName, this.getUser());
                                content = (org.w3c.dom.Document) responseVector.elementAt(0);
                                status = ((Integer)responseVector.elementAt(1)).intValue();
                                theResponse.setStatus(status, (String) responseVector.elementAt(2));
                            }
                            else {
                                String errorMsg = "Error: second object: " + secondObject + " does not exist!";
                                PapillonLogger.writeDebugMsg(errorMsg);
                                theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                            }
                        }
                        else {
                            String errorMsg = "Error: object: " + object + " does not exist!";
                            PapillonLogger.writeDebugMsg(errorMsg);
                            theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                        }
                    }
                    else if (theRequest.getMethod().equals("DELETE")) {
                        content = null;
                        if (object.equalsIgnoreCase(USERS_OBJECT)) {
                            if (secondObject.equalsIgnoreCase(GROUPS_OBJECT)) {
                                java.util.Vector responseVector = UserApi.removeUserFromGroup(objectName, secondObjectName, this.getUser());
                                content = (org.w3c.dom.Document) responseVector.elementAt(0);
                                status = ((Integer)responseVector.elementAt(1)).intValue();
                                theResponse.setStatus(status, (String) responseVector.elementAt(2));
                            }
                            else {
                                String errorMsg = "Error: second object: " + secondObject + " does not exist!";
                                PapillonLogger.writeDebugMsg(errorMsg);
                                theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                            }
                        }
                        else if (object.equalsIgnoreCase(GROUPS_OBJECT)) {
                            if (secondObject.equalsIgnoreCase(USERS_OBJECT)) {
                                java.util.Vector responseVector = UserApi.removeUserFromGroup(secondObjectName, objectName, this.getUser());
                                content = (org.w3c.dom.Document) responseVector.elementAt(0);
                                status = ((Integer)responseVector.elementAt(1)).intValue();
                                theResponse.setStatus(status, (String) responseVector.elementAt(2));
                            }
                            else {
                                String errorMsg = "Error: second object: " + secondObject + " does not exist!";
                                PapillonLogger.writeDebugMsg(errorMsg);
                                theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                            }
                        }
                        else if (object.equalsIgnoreCase(DICTIONARY_OBJECT)) {
                            if (secondObject.equalsIgnoreCase(Group.ADMIN_GROUP) ||
                                secondObject.equalsIgnoreCase(Group.VALIDATOR_GROUP) ||
                                secondObject.equalsIgnoreCase(Group.SPECIALIST_GROUP) ||
                                secondObject.equalsIgnoreCase(Group.READER_GROUP)) {
                                String groupName = secondObject + Group.DICT_GROUP_PREFIX + objectName;
                                java.util.Vector responseVector = UserApi.removeUserFromGroup(secondObjectName, groupName, this.getUser());
                                content = (org.w3c.dom.Document) responseVector.elementAt(0);
                                status = ((Integer)responseVector.elementAt(1)).intValue();
                                theResponse.setStatus(status, (String) responseVector.elementAt(2));
                            }
                            else {
                                String errorMsg = "Error: second object: " + secondObject + " does not exist!";
                                PapillonLogger.writeDebugMsg(errorMsg);
                                theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                            }
                        }
                        else {
                            String errorMsg = "Error: object: " + object + " does not exist!";
                            PapillonLogger.writeDebugMsg(errorMsg);
                            theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                        }
                    }
                    else if (theRequest.getMethod().equals("OPTIONS")) {
                        theResponse.setHeader("Access-Control-Allow-Methods","PUT, DELETE, OPTIONS");
                        theResponse.setHeader("Allow","PUT, DELETE, OPTIONS");
                    }
                    else {
                        String errorMsg = "Error: method not implemented";
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_IMPLEMENTED + "</h1><p>" + errorMsg + "</p></html>");
                        theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED, errorMsg);
                    }
                }
				else {
                    String errorMsg = "Error: method not implemented: " + commande + " OBJECT: " + object + " name: " + restStrings[1]+ " attribute: " + restStrings[2] + ";";
                    //PapillonLogger.writeDebugMsg(errorMsg);
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_IMPLEMENTED + "</h1><p>" + errorMsg + "</p></html>");
                    theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED, errorMsg);
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
				StringWriter stringWriter = new StringWriter();
				this.getComms().exception.printStackTrace(new PrintWriter(stringWriter));
				LogChannel logChannel = Enhydra.getLogChannel();
				int level = logChannel.getLevel("DEBUG");
            
				logChannel.write(level, "jibiki.presentation.ErrorHandler stack trace = ");
				logChannel.write(level, stringWriter.toString());
				logChannel.write(level, "jibiki.presentation.ErrorHandler caught an exception - " 
                             + this.getComms().exception.toString(), this.getComms().exception);
				org.w3c.dom.Node stackNode = content.createTextNode(stringWriter.toString());
				content.getDocumentElement().appendChild(stackNode);
				org.w3c.dom.Node messageNode = content.createTextNode(this.getComms().exception.getMessage());
				content.getDocumentElement().appendChild(messageNode);
			}
        }

		return content;
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
    
    protected void setUserFromLoginPassword(String login, String password)
    throws fr.imag.clips.papillon.business.PapillonBusinessException, fr.imag.clips.papillon.presentation.PapillonPresentationException, HttpPresentationException {
        User user = this.getUser();
        //PapillonLogger.writeDebugMsg("setUserFromLoginPassword: [" + login + "] [" + password + "]");
        if (user==null) {
            if (null != login && !login.equals("") &&
                null != password && !password.equals("")) {
                user = UsersFactory.findUserByLogin(login);
                if (user !=null && !user.isEmpty()) {
                    PapillonLogger.writeDebugMsg("User found: " + user.getLogin());
                    if (user.HasCorrectPassword(password)) {
                        PapillonLogger.writeDebugMsg("User has correct password!");
                        setUser(user);
                    }
                    else {
                        String errorMsg = "Error: Wrong password";
                        System.out.println(errorMsg);
                        this.getComms().response.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
                        this.responseStatus = HttpPresentationResponse.SC_UNAUTHORIZED;
                        this.responseMessage = errorMsg;
                    }
                }
                else {
                    String errorMsg = "Error: User unknown";
                    System.out.println(errorMsg);
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
                    this.responseStatus = HttpPresentationResponse.SC_UNAUTHORIZED;
                    this.responseMessage = errorMsg;
                }
            }
            else {
                // login or password empty
            }
        }
        else {
            PapillonLogger.writeDebugMsg("API User identified: " + user.getLogin());
        }
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
        
    /*    if (PAGE_EXPIRE_TIME>0) {
            this.myComms.response.setHeader("Cache-Control", "s-maxage=" + PAGE_EXPIRE_TIME);
        } */

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


}
