/*
 *  Jibiki
 *
 *  Enhydra super-servlet presentation object
 *
 *  © Francis Brunet-Manquat & Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *
 *  Cette classe implémente le layout par défaut de l'ensemble des pages de papillon
 *  Les pages utilisent une instance de Layout et se contentent de remplir son contenu
 *
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.4  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 * Revision 1.3  2006/03/13 08:48:00  fbrunet
 * bug corrections before merge
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
import java.text.DateFormat;


/**
*  Description of the Class
 *
 * @author     francis
 * @created    January 31, 2005
 */
public class EditingLayout implements StdLayout {
    EditingLayoutXHTML layout;
    
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
            layout = (EditingLayoutXHTML) MultilingualXHtmlTemplateFactory.createTemplate("EditingLayoutXHTML", comms, sessionData);
            //HeaderXHTML header = (HeaderXHTML) MultilingualXHtmlTemplateFactory.createTemplate("HeaderXHTML", comms, sessionData);
            //Node menuBar = header.getElementMenuBar();
            
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
            //layout.getElementHeaderPlace().appendChild(layout.importNode(menuBar, true));
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
	
	
}
