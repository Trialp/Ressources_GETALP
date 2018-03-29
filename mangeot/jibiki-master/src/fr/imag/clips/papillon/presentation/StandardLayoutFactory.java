/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 *
 *  © 2005 Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.2  2006/04/06 15:06:39  fbrunet
 *  New class 'creationEditInit' : create new entry
 *  Modify LexALPEditEntry : only edit entry
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
 *  -----------------------------------------------
 *
 */
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
// For the node
import org.w3c.dom.*;
//import org.w3c.dom.html.*;
import org.enhydra.xml.xhtml.dom.*;

import java.io.*;

import com.lutris.appserver.server.Enhydra;
import fr.imag.clips.papillon.Papillon;

//for debug
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.text.DateFormat;

//import java.io.*;

/**
*  Description of the Class
 *
 * @author     serasset
 * @created    December 8, 2004
 */
public class StandardLayoutFactory {

    public static StdLayout createStandardLayout(HttpPresentationComms comms, PapillonSessionData sessionData, String url) 
    throws HttpPresentationException {
        return createStandardLayout(comms, sessionData, url, null);
    }    
    
    public static StdLayout createStandardLayout(HttpPresentationComms comms, PapillonSessionData sessionData, String url, XHTMLScriptElement script) 
    throws HttpPresentationException {
        try {
            String layoutClassName = ((Papillon)Enhydra.getApplication()).getLayoutClassName();
            StdLayout layout = (StdLayout) Class.forName(layoutClassName).newInstance();
            layout.initLayout(comms, sessionData, url, script);
            
            return layout;
        } catch (java.lang.ClassNotFoundException e) {
            throw new HttpPresentationException("Could not find a suitable Layout. Please contact administrator ASAP.", e);
        } catch (java.lang.InstantiationException e) {
            throw new HttpPresentationException("Could not instanciate the Layout. Please contact administrator ASAP.", e);
        } catch (java.lang.IllegalAccessException e) {
            throw new HttpPresentationException("Illegal access exception while instanciating the Layout. Please contact administrator ASAP.", e);
        } catch (java.io.UnsupportedEncodingException e) {
            throw new HttpPresentationException("UnsupportedEncodingException while instanciating the Layout. Please contact administrator ASAP.", e);
        }
    }
}
