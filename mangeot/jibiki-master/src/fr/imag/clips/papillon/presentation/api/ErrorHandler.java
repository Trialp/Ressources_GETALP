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

package fr.imag.clips.papillon.presentation.api;

import com.lutris.logging.*;
import com.lutris.appserver.server.httpPresentation.*;
import com.lutris.appserver.server.*;

import com.lutris.appserver.server.Enhydra;
import fr.imag.clips.papillon.Papillon;

import fr.imag.clips.papillon.business.xml.XMLServices;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.MetadataApi;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.user.User;
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
 
	private org.w3c.dom.Document content;
    private String jsonString = "";
	
	protected static String ERROR_PAGE = "<?xml version='1.0'?><html></html>";
	protected static String LOGIN_PARAMETER = "login";
    protected static String PASSWORD_PARAMETER = "password";
	protected static String STRATEGY_PARAMETER = "strategy";
	protected static String LIMIT_PARAMETER = "count";
    protected static String OFFSET_PARAMETER = "startIndex";
    protected static String ORDERBY_PARAMETER = "sortBy";
    protected static String ENHYDRA_SESSION_COOKIE = "JSESSIONID";
 
    protected int PAGE_EXPIRE_TIME = ((Papillon) Enhydra.getApplication()).getPageExpireTime();
   
    private String sentContentType = MetadataApi.XML_CONTENTTYPE;
    private String acceptContentType = MetadataApi.XML_CONTENTTYPE;
   
	
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
			
        if(null != this.getComms().exception) {
			if (this.getComms().exception instanceof com.lutris.appserver.server.httpPresentation.FilePresentationException) {
				HttpPresentationRequest theRequest = this.getComms().request;
				HttpPresentationResponse theResponse = this.getComms().response;
				String login =  myGetParameter(LOGIN_PARAMETER);
                String password = myGetParameter(PASSWORD_PARAMETER);
                
                setUserFromLoginPassword(login,password);
                
                if (null != theRequest.getHeader("Accept") && theRequest.getHeader("Accept").startsWith("application/json")) {
                    acceptContentType = MetadataApi.JSON_CONTENTTYPE;
                }
                if (null != theRequest.getHeader("Content-Type") && theRequest.getHeader("Content-Type").startsWith("application/json")) {
                    sentContentType = MetadataApi.JSON_CONTENTTYPE;
                }
                
                PapillonLogger.writeDebugMsg("REST API URI : [" + prefix + "] " + theRequest.getPresentationURI()+" Accept: "+theRequest.getHeader("Accept")+" ;");
				String theURI = java.net.URLDecoder.decode(theRequest.getPresentationURI());
				if (theURI.indexOf(prefix)==0) {
					theURI = theURI.substring(prefix.length());
				}
				String[] restStrings = null;
				if (theURI!=null &&!theURI.equals("")) {
					restStrings = theURI.split("/");
				}
				// theRequest.getRemoteAddr() = 127.0.0.1 toujours
				String commande = " REST API COMMAND: " + theRequest.getMethod() + " SENT_FORMAT: " + sentContentType + " ACCEPT_FORMAT: " + acceptContentType;
                String dictName = "";
                if (restStrings!= null && restStrings.length>0) {
                    dictName = restStrings[0];
                }

                if (this.responseStatus != HttpPresentationResponse.SC_OK) {
                    PapillonLogger.writeDebugMsg(this.responseMessage);
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + this.responseStatus + " Unauthorized</h1><p>" + this.responseMessage + "</p></html>");
                }
                else if (restStrings== null || restStrings.length==0) {
					PapillonLogger.writeDebugMsg(commande + " DICTLIST;");
					if (theRequest.getMethod().equals("GET")) {
                        java.util.Vector responseVector = MetadataApi.getDictionaryList(this.getUser());
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
				else if (restStrings.length==1) {
					if (theRequest.getMethod().equals("GET")) {
                        PapillonLogger.writeDebugMsg(commande + " DICT: " + dictName+ ";");
						if (dictName.equals("*")) {
                            PapillonLogger.writeDebugMsg(commande + " DICTLIST;");
                            java.util.Vector responseVector = MetadataApi.getDictionaryList(this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
						}
						else {
                            java.util.Vector responseVector = MetadataApi.getDictionary(dictName, this.getUser());
                            content = (org.w3c.dom.Document) responseVector.elementAt(0);
                            status = ((Integer)responseVector.elementAt(1)).intValue();
                            theResponse.setStatus(status, (String) responseVector.elementAt(2));
						}
					}
                    else if (theRequest.getMethod().equals("PUT")) {
                        //TODO: 415 Unsupported Media Type si le content type est json
                        //TODO pour tous les dict et les vol et peut-être les entry put post ?
                        HttpPresentationInputStream inputStream = theRequest.getInputStream();
                        String dictXml = convertStreamToString(inputStream);
                        
                        java.util.Vector responseVector = MetadataApi.putDictionary(dictName, dictXml, sentContentType, this.getUser());
                        content = (org.w3c.dom.Document) responseVector.elementAt(0);
                        status = ((Integer)responseVector.elementAt(1)).intValue();
                        theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        theResponse.flush();
                    }
                    else if (theRequest.getMethod().equals("POST")) {
                        HttpPresentationInputStream inputStream = theRequest.getInputStream();
                        String dictXml = convertStreamToString(inputStream);
 
                        java.util.Vector responseVector = MetadataApi.postDictionary(dictName, dictXml, sentContentType, this.getUser());
                        content = (org.w3c.dom.Document) responseVector.elementAt(0);
                        status = ((Integer)responseVector.elementAt(1)).intValue();
                        theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        theResponse.flush();
                    }
					else if (theRequest.getMethod().equals("DELETE")) {
                        java.util.Vector responseVector = MetadataApi.deleteDictionary(dictName, this.getUser());
                        content = (org.w3c.dom.Document) responseVector.elementAt(0);
                        status = ((Integer)responseVector.elementAt(1)).intValue();
                        theResponse.setStatus(status, (String) responseVector.elementAt(2));
					}
                    else if (theRequest.getMethod().equals("OPTIONS")) {
                        // System.out.println("OPTIONS");
                        theResponse.setHeader("Access-Control-Allow-Methods","GET, PUT, POST, DELETE, OPTIONS");
                        theResponse.setHeader("Allow","GET, POST, PUT, DELETE, OPTIONS");
                    }
                    else {
                        theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED);
                        PapillonLogger.writeDebugMsg("Error: method not implemented");
                    }
				}
				else if (restStrings.length==2) {
                    String lang = restStrings[1];
					PapillonLogger.writeDebugMsg(commande + " DICT: " + dictName + " LANG: " + lang+ ";");
					if (theRequest.getMethod().equals("GET")) {
                        java.util.Vector responseVector = MetadataApi.getVolume(dictName, lang, this.getUser());
                        content = (org.w3c.dom.Document) responseVector.elementAt(0);
                        status = ((Integer)responseVector.elementAt(1)).intValue();
                        theResponse.setStatus(status, (String) responseVector.elementAt(2));
					}
					else if (theRequest.getMethod().equals("PUT")) {
                        HttpPresentationInputStream inputStream = theRequest.getInputStream();
                        String volXml = convertStreamToString(inputStream);
                        java.util.Vector responseVector = MetadataApi.putVolume(dictName, lang, volXml, sentContentType, this.getUser());
                        content = (org.w3c.dom.Document) responseVector.elementAt(0);
                        status = ((Integer)responseVector.elementAt(1)).intValue();
                        theResponse.setStatus(status, (String) responseVector.elementAt(2));
                       theResponse.flush();
                    }
					else if (theRequest.getMethod().equals("POST")) {
                        HttpPresentationInputStream inputStream = theRequest.getInputStream();
                        String volXml = convertStreamToString(inputStream);
                        java.util.Vector responseVector = MetadataApi.postVolume(dictName, lang, volXml, sentContentType, this.getUser());
                        content = (org.w3c.dom.Document) responseVector.elementAt(0);
                        status = ((Integer)responseVector.elementAt(1)).intValue();
                        theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        theResponse.flush();
                    }
					else if (theRequest.getMethod().equals("DELETE")) {
                        java.util.Vector responseVector = MetadataApi.deleteVolume(dictName, lang, this.getUser());
                        content = (org.w3c.dom.Document) responseVector.elementAt(0);
                        status = ((Integer)responseVector.elementAt(1)).intValue();
                        theResponse.setStatus(status, (String) responseVector.elementAt(2));
					}
                    else if (theRequest.getMethod().equals("OPTIONS")) {
                        // System.out.println("OPTIONS");
                        theResponse.setHeader("Access-Control-Allow-Methods","GET, PUT, POST, DELETE, OPTIONS");
                        theResponse.setHeader("Allow","GET, POST, PUT, DELETE, OPTIONS");
                    }
                    else {
                        theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED);
                        PapillonLogger.writeDebugMsg("Error: method not implemented");
                    }
				}
				else if (restStrings.length==3) {
					PapillonLogger.writeDebugMsg(commande + " DICT: " + dictName + " LANG: " + restStrings[1]+ " ENTRYID : " + restStrings[2]+ ";");
					if (theRequest.getMethod().equals("GET")) {
						content = Entries.getEntry(dictName, restStrings[1], restStrings[2]);
						if (content==null) {
							String errorMsg = "Error: get: " + dictName + " lang: " +  restStrings[1] + " ID: " + restStrings[2] +" does not exist!";
							System.out.println(errorMsg);
							theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
						}						
					}
					else if (theRequest.getMethod().equals("PUT")) {
						HttpPresentationInputStream inputStream = theRequest.getInputStream();
						String entry = convertStreamToString(inputStream);
						//PapillonLogger.writeDebugMsg("put data: "+entry);
                        org.w3c.dom.Document entryDom = null;
                        if (sentContentType.equals(MetadataApi.XML_CONTENTTYPE)) {
                       try {
                            entryDom = XMLServices.buildDOMTree(entry);
                        }
                        catch (Exception e) {
                            entry = "";
                        }
                        if (entry != null && !entry.equals("")) {
                            if (Entries.userCanPutEntry(getUser(),dictName)) {
                                content = Entries.putEntry(dictName, restStrings[1], restStrings[2], entryDom, this.getUser());
                                if (content==null) {
                                    String errorMsg = "Error: dict: " + dictName + " lang: " +  restStrings[1] + " ENTRY ID: " + restStrings[2] +" does not exist!";
                                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_FOUND + "</h1><p>" + errorMsg + "</p></html>");
                                    theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND,errorMsg);
                                    //PapillonLogger.writeDebugMsg(errorMsg);
                                }
                                else {
                                    theResponse.setStatus(HttpPresentationResponse.SC_CREATED);
                                }
                            }
                            else {
                                String errorMsg = "Error: user: " + login +" not authorized to put entry!";
                                //PapillonLogger.writeDebugMsg(errorMsg);
                                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_FOUND + "</h1><p>" + errorMsg + "</p></html>");
                                theResponse.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
                            }
                        }
                        else {
                            String errorMsg = "Error: entry: <![CDATA["+ entry +"]]> XML is malformed!";
                            //PapillonLogger.writeDebugMsg(errorMsg);
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_BAD_REQUEST + "</h1><p>"+errorMsg+"</p></html>");
                            theResponse.setStatus(HttpPresentationResponse.SC_BAD_REQUEST ,errorMsg);
                        }
                        }
                        else {
                            String errorMsg = "Error: only XML content type allowed!";
                            //PapillonLogger.writeDebugMsg(errorMsg);
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + 415 + " Unsupported Media Type</h1><p>" + errorMsg + "</p></html>");
                            theResponse.setStatus(415 ,errorMsg);
                        }
                        theResponse.flush();
					}
					else if (theRequest.getMethod().equals("POST")) {
						HttpPresentationInputStream inputStream = theRequest.getInputStream();
						String entry = convertStreamToString(inputStream);
                        //PapillonLogger.writeDebugMsg("post data: "+entry);
                        java.util.Vector responseVector = Entries.postEntries(dictName, restStrings[1], restStrings[2], entry, sentContentType, this.getUser());
                        content = (org.w3c.dom.Document) responseVector.elementAt(0);
                        status = ((Integer)responseVector.elementAt(1)).intValue();
                        theResponse.setStatus(status, (String) responseVector.elementAt(2));
                        theResponse.flush();
					}
					else if (theRequest.getMethod().equals("DELETE")) {
						if (Entries.userCanDeleteEntry(getUser(),dictName)) {
							content = Entries.deleteEntry(dictName, restStrings[1], restStrings[2], getUser());
							if (content==null) {
								String errorMsg = "Error: dict: " + dictName + " lang: " +  restStrings[1] + " ID: " + restStrings[2] +" does not exist!";
								//PapillonLogger.writeDebugMsg(errorMsg);
                                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_FOUND + "</h1><p>" + errorMsg + "</p></html>");
								theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND, errorMsg);
							}
							else {
								theResponse.setStatus(HttpPresentationResponse.SC_NO_CONTENT);
							}							
						}
						else {
							String errorMsg = "Error: user: " + login +" not authorized to put entry!";
	//						PapillonLogger.writeDebugMsg(errorMsg);
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_UNAUTHORIZED + "</h1><p>" + errorMsg + "</p></html>");
							theResponse.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
						}
					}
                    else if (theRequest.getMethod().equals("OPTIONS")) {
                        // System.out.println("OPTIONS");
                        theResponse.setHeader("Access-Control-Allow-Methods","GET, PUT, POST, DELETE, OPTIONS");
                        theResponse.setHeader("Allow","GET, PUT, POST, DELETE, OPTIONS");
                    }
                    else {
                        String errorMsg = "Error: method not implemented";
                        //PapillonLogger.writeDebugMsg(errorMsg);
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_IMPLEMENTED + "</h1><p>" + errorMsg + "</p></html>");
                        theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED, errorMsg);
                    }
				}
				else if (restStrings.length==4 || restStrings.length==5) {
					PapillonLogger.writeDebugMsg(commande + " DICT: " + dictName + " LANG: " + restStrings[1]+ " MODE: " + restStrings[2]+ " STRING: " + restStrings[3]+ ";");
					if (theRequest.getMethod().equals("GET")) {
						String strategy = myGetParameter(STRATEGY_PARAMETER);
						String limit = myGetParameter(LIMIT_PARAMETER);
                        String offset = myGetParameter(OFFSET_PARAMETER);
                        String orderby = myGetParameter(ORDERBY_PARAMETER);
						String key=null;
						if (restStrings.length==5) {
							key = restStrings[4];
						}
                        PapillonLogger.writeDebugMsg("Parameters: strategy: " + strategy + " limit: " + limit + " offset: " + offset + " orderby: " + orderby + " key: " + key);
						content = Entries.getEntries(dictName, restStrings[1], restStrings[2], restStrings[3], key, strategy, limit, offset, orderby, getUser());
						if (content==null) {
							String errorMsg = "Error: search: " + dictName + " lang: " +  restStrings[1] + " method: " + restStrings[2] +" does not exist!";
						//	PapillonLogger.writeDebugMsg(errorMsg);
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_FOUND + "</h1><p>" + errorMsg + "</p></html>");
							theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND, errorMsg);
						}
					}
                    else if (theRequest.getMethod().equals("PUT")) {
                        HttpPresentationInputStream inputStream = theRequest.getInputStream();
                        String xpathString = convertStreamToString(inputStream);
                        //inputStream.close();
                        //PapillonLogger.writeDebugMsg("put data: "+ restStrings[3] +" xpath: "+entryPart);
                        try {
                            javax.xml.xpath.XPathFactory factory = javax.xml.xpath.XPathFactory.newInstance();
                            javax.xml.xpath.XPath xpath = factory.newXPath();
                            javax.xml.xpath.XPathExpression expr = xpath.compile(xpathString);
                        }
                        catch (Exception exception) {
                            xpathString = "";
                        }
                        if (xpathString != null && !xpathString.equals("")) {
                            if (Entries.userCanEditEntry(getUser(), dictName)) {
                                //TODO : messages d'erreur plus précis si editEntry rate !
                                content = Entries.editEntry(dictName, restStrings[1], restStrings[2], xpathString, restStrings[3], this.getUser());
                                if (content==null) {
                                    String errorMsg = "Error: dict: " + dictName + " lang: " +  restStrings[1] +" CONTRIB ID: " + restStrings[2] + " does not exist!";
                                    //PapillonLogger.writeDebugMsg(errorMsg);
                                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_FOUND + "</h1><p>" + errorMsg + "</p></html>");
                                   theResponse.setStatus(HttpPresentationResponse.SC_NOT_FOUND, errorMsg);
                                }
                                else {
                                    theResponse.setStatus(HttpPresentationResponse.SC_CREATED);
                                    //PapillonLogger.writeDebugMsg("put data end: "+ restStrings[3] +" result: "+XMLServices.NodeToString(content));
                                }
                            }
                            else {
                                String errorMsg = "Error: user: " + login +" not authorized to put entry part!";
                            //    PapillonLogger.writeDebugMsg(errorMsg);
                                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_UNAUTHORIZED + "</h1><p>" + errorMsg + "</p></html>");
                                theResponse.setStatus(HttpPresentationResponse.SC_UNAUTHORIZED,errorMsg);
                            }
                        //inputStream.close();
                        //theResponse.flush();
                        }
                        else {
                            String errorMsg = "Error: XPath: " + xpathString +" is malformed!";
                            //PapillonLogger.writeDebugMsg(errorMsg);
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_BAD_REQUEST + "</h1><p>" + errorMsg + "</p></html>");
                            theResponse.setStatus(HttpPresentationResponse.SC_BAD_REQUEST ,errorMsg);
                        }
                   }
                    else if (theRequest.getMethod().equals("OPTIONS")) {
                        theResponse.setHeader("Access-Control-Allow-Methods","GET, PUT, POST, DELETE, OPTIONS");
                        theResponse.setHeader("Allow","GET, PUT, POST, DELETE, OPTIONS");
                    }
					else {
						theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED);
						PapillonLogger.writeDebugMsg("Error: method not implemented");
						PapillonLogger.writeDebugMsg("search entries: error message! " + dictName + " "+restStrings[1]);
					}
				}
				else {
                    String errorMsg = "Error: method not implemented: " + commande + " DICT: " + dictName + " LANG: " + restStrings[1]+ " MODE: " + restStrings[2]+ " STRING: " + restStrings[3]+ ";";
                    //PapillonLogger.writeDebugMsg(errorMsg);
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + HttpPresentationResponse.SC_NOT_IMPLEMENTED + "</h1><p>" + errorMsg + "</p></html>");
                    theResponse.setStatus(HttpPresentationResponse.SC_NOT_IMPLEMENTED, errorMsg);
				}
                if (acceptContentType.equals(MetadataApi.JSON_CONTENTTYPE)) {
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
        
        if (acceptContentType.equals(MetadataApi.JSON_CONTENTTYPE)) {
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
