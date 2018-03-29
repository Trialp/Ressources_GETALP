/*
 *  LexALP
 *
 *  Enhydra super-servlet presentation object
 *
 *  © Gilles Sérasset - GETA CLIPS IMAG
 *  Projet LexALP
 *
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.3  2006/06/09 10:10:43  fbrunet
 *  Add generic components (AdvancedQueryForm, QueryRequest and ViewQueryResult) in Home.java
 *
 *  Revision 1.2  2006/03/01 15:12:31  mangeot
 *  Merge between maintrunk and LEXALP_1_1 branch
 *
 *  Revision 1.1.4.1  2006/01/10 12:33:26  serasset
 *  Lexalp does not use a specific BasePO anymore. Search of adequate XHTML pages is done dynamically using lexalp specific package when necessary.
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
 */
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;

//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;


// Standard imports
import java.io.IOException;
import java.io.*;
import java.util.*;
import java.util.Date;
import java.text.DateFormat;

// Imported TraX classes
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.stream.StreamSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.dom.DOMResult;
import org.xml.sax.InputSource;

// Imported DOM classes
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.Text;
import org.w3c.dom.NodeList;
import org.w3c.dom.Element;
//import org.w3c.dom.html.*;

// Imported JAVA API for XML Parsing classes
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
//for debug
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;

import fr.imag.clips.papillon.business.dictionary.*;
import fr.imag.clips.papillon.business.logs.*;
import fr.imag.clips.papillon.business.user.*;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.utility.*;
import fr.imag.clips.papillon.business.locales.Languages;

import fr.imag.clips.papillon.presentation.xhtmllexalp.orig.*;

/**
 *  Description of the Class
 *
 * @author     serasset
 * @created    December 8, 2004
 */
public class LexALP extends PapillonBasePO {

    /**
     *  Description of the Method
     *
     * @return    Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return false;
    }

    protected  int getCurrentSection() {
        return NO_SECTION;
    }
    
    /**
     *  Description of the Method
     *
     * @return    Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        return true;
    }


    /**
     *  Gets the content attribute of the Home object
     *
     * @return                                                     The content
     *      value
     * @exception  HttpPresentationException                       Description
     *      of the Exception
     * @exception  IOException                                     Description
     *      of the Exception
     * @exception  TransformerConfigurationException               Description
     *      of the Exception
     * @exception  org.xml.sax.SAXException                        Description
     *      of the Exception
     * @exception  javax.xml.parsers.ParserConfigurationException  Description
     *      of the Exception
     * @exception  java.io.IOException                             Description
     *      of the Exception
     * @exception  javax.xml.transform.TransformerException        Description
     *      of the Exception
     * @exception  ClassNotFoundException                          Description
     *      of the Exception
     * @exception  PapillonBusinessException                       Description
     *      of the Exception
     * @exception  UnsupportedEncodingException                    Description
     *      of the Exception
     */
    public Node getContent()
    throws 
    HttpPresentationException, IOException,
    TransformerConfigurationException,
    org.xml.sax.SAXException,
    javax.xml.parsers.ParserConfigurationException,
    java.io.IOException,
    javax.xml.transform.TransformerException,
    ClassNotFoundException,
    PapillonBusinessException,
    UnsupportedEncodingException 
{
    HomeContentXHTML homeContent = (HomeContentXHTML) MultilingualXHtmlTemplateFactory.createTemplate("HomeContentXHTML", this.getComms(), this.getSessionData());
    Element home = homeContent.getElementHomeContent();
    
    //
    removeQueryResult(homeContent);
    removeQueryFuzzyResult(homeContent);
    
    return (Node) home;
}

private void removeQueryResult(HomeContentXHTML homeContent) {
    Element myElement = homeContent.getElementQueryResult();
    Node myParent = myElement.getParentNode();
    if (myParent != null)
        myParent.removeChild(myElement);
}


private void removeQueryFuzzyResult(HomeContentXHTML homeContent) {
    Element myElement = homeContent.getElementQueryFuzzyResult();
    Node myParent = myElement.getParentNode();
    if (myParent != null)
        myParent.removeChild(myElement);
}

}

