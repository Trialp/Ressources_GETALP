/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 *
 *  © 2001 Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.9  2006/04/06 15:06:39  fbrunet
 *  New class 'creationEditInit' : create new entry
 *  Modify LexALPEditEntry : only edit entry
 *
 *  Revision 1.8  2006/03/01 15:12:31  mangeot
 *  Merge between maintrunk and LEXALP_1_1 branch
 *
 *  Revision 1.7.4.1  2006/01/10 12:33:26  serasset
 *  Lexalp does not use a specific BasePO anymore. Search of adequate XHTML pages is done dynamically using lexalp specific package when necessary.
 *
 *  Revision 1.7  2005/06/15 16:48:28  mangeot
 *  Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *  Revision 1.6  2005/05/24 12:51:22  serasset
 *  Updated many aspect of the Papillon project to handle lexalp project.
 *  1. Layout is now parametrable in the application configuration file.
 *  2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 *  3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 *  4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 *  5. It is now possible to give a name to the cookie key in the app conf file
 *  6. Several bug fixes.
 *
 *  Revision 1.5.4.2  2005/06/01 08:38:43  mangeot
 *  Multi bug correction + added the possibility of disabling data edition
 *  via the Admin.po page
 *
 *  Revision 1.5.4.1  2005/05/27 11:53:34  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.5  2005/04/18 10:50:26  mangeot
 *  Bug fix when displaying with IExplorer,
 *  Bug fixes when seqencial request
 *
 *  Revision 1.4  2005/01/15 12:51:24  mangeot
 *  Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 *  Revision 1.3  2005/01/14 22:33:57  mangeot
 *  Fixed the BasePO.setSelected
 *  It seems that the enhydra setSelectedIndex does not work any more with enhydra5.1
 *
 *  Revision 1.2  2004/12/24 08:57:44  serasset
 *  Premiere version de l'interface avec fond papillon et transparence.
 *
 *  Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 *  Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 *  There are still bugs in the code.
 *
 *
 *  -----------------------------------------------
 *
 *  Cette classe implémente le layout par défaut de l'ensemble des pages de papillon
 *  Les pages utilisent une instance de Layout et se contentent de remplir son contenu
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
import org.enhydra.xml.xmlc.XMLObject;

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

//import java.io.*;

/**
 *  Description of the Class
 *
 * @author     serasset
 * @created    December 8, 2004
 */
public interface StdLayout {

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
        throws com.lutris.appserver.server.httpPresentation.HttpPresentationException, UnsupportedEncodingException;

    /**
     *  Gets the layout attribute of the StdLayout object
     *
     * @return    The layout value
     */
    public Document getLayout();
    
    /**
	 *  Returns the node that will hold the main content of the page.
     *
     * @return    The layout value
     */    
    public Node getContentPlaceHolder();

    /**
	 *  Returns the node that will hold the main content of the page.
     *
     * @return    The layout value
     */    
    public Node getContextualMenuPlaceHolder();
	
	/**
	 *  Returns the node that will hold the main content of the page.
     *
     * @return    The layout value
     */    
    public Node getBannerPlaceHolder();
	
	/**
	 *  Returns the node that will hold the main content of the page.
     *
     * @return    The layout value
     */    
    public Node getBannerContent();
	
    
}
