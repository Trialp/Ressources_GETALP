//
//  Register.java
//  PB-Papillon
//
//  Created by liu on Wed Mar 19 2003.
//  Copyright (c) 2003 __MyCompanyName__. All rights reserved.
//
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.*;
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

import fr.imag.clips.papillon.business.message.MessageDBLoader;

import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

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


public class Register extends PapillonBasePO {


    public final static String PREFERENCES_PREFIX = "PREF__";
    
    protected RegisterXHTML content;

    protected static String REGISTER_PAGE = "ReNuser";

    
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
        content = (RegisterXHTML)MultilingualXHtmlTemplateFactory.createTemplate("RegisterXHTML", this.getComms(), this.getSessionData());

        HttpPresentationRequest req = this.getComms().request;

        String name = "";
        String login = "";
        String password = "";
        String password2 = "";
        String email = myGetParameter(content.NAME_NEmail);
        String Dest = myGetParameter(REGISTER_PAGE);
    
        // If there is no destination, just redirect the user to the login page after a succesfull log in.
        Dest = (Dest != null) ? Dest : req.getAppFileURIPath("");
        //Dest = (Dest != null) ? Dest : req.getAppFileURIPath("Home.po");
        
        // If the page is called with parameters, take the requested action
        if (req.getParameterNames().hasMoreElements()) {

            String userMessage = new String("");

            if (null != myGetParameter(content.NAME_Register)) {
                name = myGetParameter(content.NAME_NName);
                login = myGetParameter(content.NAME_NLogin);
                password = myGetParameter(content.NAME_NPassword);
                password2 = myGetParameter(content.NAME_NPassword2);
                email = myGetParameter(content.NAME_NEmail);
                
                User Existe=UsersFactory.findUserByName(name);
                if (!Existe.isEmpty()) {
                    userMessage = "User " + name + " already in the database";
                    this.getComms().response.setStatus(409,userMessage);
               }
                else {
                    Existe=UsersFactory.findUserByLogin(login);
                    if (!Existe.isEmpty()) {
                        userMessage = "Login " + login + " already in the database";
                        this.getComms().response.setStatus(409,userMessage);
                   }
                    else {
                        if (!password.equals(password2)) {
                            userMessage = "Passwords do not match";
                            this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
                        }
                        else {
                UserAnswer myUserAnswer = UsersFactory.createUniqueUser(name, login, password, password2, email, this.getUserPreferredLanguage()); 
                
                userMessage = myUserAnswer.getMessage();

                if (!myUserAnswer.isEmpty()) {
                    this.setUser(myUserAnswer.getUser());
                    savePreferences();
                    throw new ClientPageRedirectException(Dest);
                }
                else {
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
                }
                        }
                    }
                }
                
                this.getSessionData().writeUserMessage(userMessage);
                PapillonLogger.writeDebugMsg(userMessage);
            }
            else {
                String errorMessage = "Error: Wrong arguments";
                this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,errorMessage);
            }
        }
    
    HTMLInputElement DestElement = (HTMLInputElement) content.getElementReNuser();
    DestElement.setValue(Dest);
    
    return content.getElementFormulaire();
    }
    
    protected void savePreferences() throws com.lutris.appserver.server.httpPresentation.HttpPresentationException {
        java.util.Enumeration parameterNames = this.getComms().request.getParameterNames();
        while (parameterNames.hasMoreElements()) {
            String parameterName = (String) parameterNames.nextElement();
            if (parameterName.indexOf(PREFERENCES_PREFIX)==0) {
                String value = myGetParameter(parameterName);
              //  PapillonLogger.writeDebugMsg("setPreference " + parameterName + ": " + value);
                this.setPreference(parameterName,value);
            }
        }

    }
    
    
    
}
