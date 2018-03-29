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
 * This page shows the properties defined on the Virtual Machine the server runs on.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.Enhydra;
import com.lutris.appserver.server.session.SessionManager;
import com.lutris.appserver.server.session.SessionException;
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.*;

import fr.imag.clips.papillon.business.message.MessageDBLoader;

// Standard imports
import java.io.IOException;
import fr.imag.clips.papillon.business.PapillonBusinessException;

import java.lang.System;
import java.util.Properties;
import java.util.Enumeration;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

public class GetSystemProperties extends PapillonBasePO {

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
    throws HttpPresentationException, IOException, SessionException {

        SystemPropertiesListXHTML content = (SystemPropertiesListXHTML)MultilingualXHtmlTemplateFactory.createTemplate("SystemPropertiesListXHTML", this.getComms(), this.getSessionData());

        // Enter Session Properties
		SessionManager mySessionManager = Enhydra.getSessionManager();
        content.setTextActiveSessionCount(new Integer(mySessionManager.activeSessionCount()).toString());
        content.setTextMaxSessionCount(new Integer(mySessionManager.maxSessionCount()).toString());
        content.setTextMaxSessionCountDate(mySessionManager.maxSessionCountDate().toString());


        // Enter System Properties List
        HTMLTableRowElement templateRow = content.getElementTemplateRow();
        HTMLElement propCellTemplate = content.getElementProperty();
        HTMLElement valueCellTemplate = content.getElementValue();
        templateRow.removeAttribute("id");
        propCellTemplate.removeAttribute("id");
        valueCellTemplate.removeAttribute("id");
        
        Node propTable = templateRow.getParentNode();

        
        // Get the properties of the system.
        Properties plist = System.getProperties();
        for (Enumeration e = plist.propertyNames() ; e.hasMoreElements() ;) {
            String propKey = e.nextElement().toString();
            String propValue = plist.getProperty(propKey, "");
            
            // Put the xorrect texts in the template
            content.setTextProperty(propKey);
            content.setTextValue(propValue);
            
            // Clone it and add the clone to the table
            Node clone=templateRow.cloneNode(true);
            propTable.appendChild(clone);
        }
        
        // Finally, delete the template
        propTable.removeChild(templateRow);
        
        //On rends le contenu correst
        return content.getElementPropertiesList();
    }

}
