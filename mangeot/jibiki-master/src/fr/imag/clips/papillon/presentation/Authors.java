/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id: Information.java 1296 2011-11-29 17:57:22Z mangeot $
 *-----------------------------------------------
 *
 *-----------------------------------------------
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// w3c imports
//import org.w3c.dom.html.HTMLElement;
import org.w3c.dom.Node;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.text.DateFormat;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

import fr.imag.clips.papillon.presentation.MultilingualHtmlTemplateFactory;

public class Authors extends PapillonBasePO {

    public Node getContent()
        throws HttpPresentationException, IOException {

	AuthorsTmplXHTML content;

	// Création du contenu
	//content = (InfosTmplHTML)this.getComms().xmlcFactory.create(InfosTmplHTML.class);
        content = (AuthorsTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("AuthorsTmplXHTML", this.getComms(), this.getSessionData());
      
        return content.getElementInfoContent();
        }

    protected boolean loggedInUserRequired() {
        return false;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }
    
    protected  int getCurrentSection() {
        return INFORMATIONS_SECTION;
    }


}
