/*
 * Jibiki 
 *
 * Enhydra super-servlet
 * 
 * © Francis Brunet-Manquat & Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *
 * 
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.4  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.3  2006/03/13 08:48:00  fbrunet
 * bug corrections before merge
 *
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.*;

import java.util.ArrayList;

// internal imports
import fr.imag.clips.papillon.business.PapillonBusinessException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;


public class EditingError extends EditingBasePO {
 
    // URL
    protected final static String EditingErrorURL = "EditingError.po";
    
    //
    protected boolean loggedInUserRequired() {
        return true;
    }
    
    //
    protected boolean userMayUseThisPO() {
        return true;
    }
    
    //
    protected  int getCurrentSection() {
        return NO_SECTION;
    }

    //
    public org.w3c.dom.Node getContent() 
    throws java.io.UnsupportedEncodingException, 
    HttpPresentationException {
        
        // Management of the parameters
        //String submitAdd = myGetParameter(AddCall_PARAMETER);
                
        //
        EditingErrorXHTML content = (EditingErrorXHTML) MultilingualXHtmlTemplateFactory.createTemplate("EditingErrorXHTML", this.myComms, this.sessionData);
        
        //
        return content.getElementEditingErrorContent();
    }        
    
}
    