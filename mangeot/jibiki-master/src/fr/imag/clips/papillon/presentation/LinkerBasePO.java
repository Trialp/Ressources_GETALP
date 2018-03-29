/*
 *  papillon
 *
 *  Enhydra super-servlet
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.2  2005/07/16 16:25:26  mangeot
 *  Adapted the linker to the GDEF project + bug fixes
 *
 *  Revision 1.1  2005/07/08 08:22:46  serasset
 *  Reviewed the Abstract/BasePO hierarchy (moved some methods up in the tree).
 *  Added base classes to allow independant browsing window to establish links during edition.
 *
 *  -----------------------------------------------
 *  Abstract class implementing small windows used to link elements.
 */
package fr.imag.clips.papillon.presentation;

// Papillon import
//import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.presentation.xhtml.orig.*;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;

import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;
import com.lutris.appserver.server.Enhydra;

import org.w3c.dom.Node;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Comment;

import org.w3c.dom.html.HTMLCollection;
import org.w3c.dom.html.HTMLOptionElement;
import org.w3c.dom.html.HTMLSelectElement;

import org.enhydra.xml.xhtml.dom.*;

import org.enhydra.xml.xmlc.*;
//import org.enhydra.xml.xmlc.html.*;
import com.lutris.logging.*;
import com.lutris.util.*;

// For the cookies
//import javax.servlet.http.Cookie;
//import fr.imag.clips.papillon.business.user.UsersFactory;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;
import java.text.DateFormat;

//pour le debogage
import fr.imag.clips.papillon.business.utility.*;
import fr.imag.clips.papillon.business.PapillonLogger;

import org.enhydra.xml.io.OutputOptions;
import org.enhydra.xml.io.DOMFormatter;

/**
*  Description of the Class
 *
 * @author     serasset
 * @created    December 8, 2004
 */
public abstract class LinkerBasePO extends AbstractPO {
        
        // JavaScript to add in the header
    /**
        *  Description of the Field
     */
    protected String headerScript = "";
    
    /**
        *  This is the procedure that is called when an HTML request occurs.
     *
     * @return                XMLObject The XMLObject (in HTML format) that is
     *      to be included in the standard layout.
     * @exception  Exception  Description of the Exception
     */
    public abstract Node getBrowserForm() throws Exception;
    
    public abstract Node getResultList() throws Exception;
    
    /**
        *  Sets the headerScript attribute of the PapillonBasePO object
     *
     * @param  newScript                          The new headerScript value
     * @exception  PapillonPresentationException  Description of the Exception
     */
    public void addToHeaderScript(String newScript)
        throws PapillonPresentationException {
            this.headerScript += "\n\n" + newScript;
        }
    
    /**
        *  Returns the complete document.
     *
     * @exception  Exception
     * @exception  HttpPresentationException  Description of the Exception
     * @exception  IOException                Description of the Exception
     */
    public Node getDocument()
        throws HttpPresentationException, IOException, Exception {
            
            LinkerLayoutXHTML linkerLayout;
            
        
            Node form = getBrowserForm();
            Node results = getResultList();

            
            // Création du document
            linkerLayout = (LinkerLayoutXHTML) MultilingualXHtmlTemplateFactory.createTemplate("LinkerLayoutXHTML", this.myComms, this.sessionData);
                       
            //Insertion du contenu dans le document vide.
            linkerLayout.getElementBrowsingForm().appendChild(linkerLayout.importNode(form, true));
            linkerLayout.getElementResultList().appendChild(linkerLayout.importNode(results, true));
            
            // Add scripts to the layout.
            // adding a script if needed
            XHTMLScriptElement scriptElement = (XHTMLScriptElement) linkerLayout.getElementScript();
            if (null != this.headerScript && !this.headerScript.equals("")) {
				this.headerScript = "\n" + this.headerScript + "\n//";
				Comment scriptContent = scriptElement.getOwnerDocument().createComment(this.headerScript);
                scriptElement.appendChild(scriptContent);
            }
            scriptElement.removeAttribute("id");
            
            return (Node) linkerLayout;
        }
    
}
