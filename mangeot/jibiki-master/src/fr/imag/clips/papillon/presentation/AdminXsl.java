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
 * Revision 1.9  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.8  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.7  2005/08/05 18:44:38  mangeot
 * Bug fixes + ProcessVolume.po page creation
 *
 * Revision 1.6  2005/07/14 13:48:53  serasset
 * Added columns dictionaryname and volumename to the xslsheets.
 * Modified the XslSheet and XslSheetFactory accordingly.
 * Modified AdminXsl interface accordingly.
 * Modified DictionariesFactory and VolumesFactory to allow several xsl-sheets and to
 * correctly provide dictionaryName/volumeName to XslSheetFactory.
 * Cleanup of several errors in papillon_static doml file.
 *
 * Revision 1.5  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.4  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.3  2005/04/11 08:01:02  fbrunet
 * Passage en xhtml des ressources Papillon.
 *
 * Revision 1.2.2.2  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
 *
 * Revision 1.2.2.1  2005/01/27 15:56:21  mangeot
 * Able to load a volume with XPointers, cannot lookup the result yet.
 * Does not compile but commit for backup
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * Papillon Admin page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
//test
import com.lutris.appserver.server.httpPresentation.HttpUtil;
//fin test

import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationInputStream;
//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

import fr.imag.clips.papillon.business.message.MessageDBLoader;

//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// Standard imports
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.Date;
import java.text.DateFormat;
import java.io.*;

// For URLs
import java.net.URL;

//pour les stylesheet
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.PapillonLogger;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.business.PapillonBusinessException;

public class AdminXsl extends PapillonBasePO {
    
    protected final static String SEE_PARAMETER="See";
    protected final static String ADD_PARAMETER="Add";
    //protected final static String FLUSH_PARAMETER="Flush";
    protected final static String REMOVE_PARAMETER="Remove";

    protected final static String NAME_PARAMETER="name";
    protected final static String DESC_PARAMETER="description";
    protected final static String URL_PARAMETER="url";
    protected final static String DEFAULT_XSL_PARAMETER="defaultxsl";
    protected final static String EXTERNAL_XSL_PARAMETER="externalxsl";
    protected final static String NO_EXTERNAL_XSL_PARAMETER="noexternalxsl";

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
        throws javax.xml.parsers.ParserConfigurationException,HttpPresentationException, IOException
    {

        // Création du contenu
        AdminXslTmplXHTML content = (AdminXslTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("AdminXslTmplXHTML", this.getComms(), this.getSessionData());
		  
        HttpPresentationRequest req = this.getComms().request;
        
        // If the page is called with parameters, take the requested action
        if (req.getParameterNames().hasMoreElements()) {
            String userMessage = null;
            
            /* Remove all xsl sheets
            if (null != req.getParameter(FLUSH_PARAMETER)) {
                XslSheetFactory.removeAllXslSheets();
                userMessage = "All XslSheets removed...";
            } else
            */
            
             
            
            // Add xsl sheet
            if ((null != req.getParameter(AdminXslTmplXHTML.NAME_name)) && 
                    (null != req.getParameter(AdminXslTmplXHTML.NAME_url))) {
                
                // Get the main parameters
                String Nom = req.getParameter(AdminXslTmplXHTML.NAME_name);
                String dictionaryName = req.getParameter(AdminXslTmplXHTML.NAME_dictionaryName);
                String volumeName = req.getParameter(AdminXslTmplXHTML.NAME_volumeName);
                String Description= req.getParameter(AdminXslTmplXHTML.NAME_description);
                String urlString = req.getParameter(AdminXslTmplXHTML.NAME_url);
                if (null != urlString && urlString.charAt(0) == '/') {
                    urlString = "file:" + urlString;
                }
                URL theURL= new URL(urlString);
                String leCode = fr.imag.clips.papillon.business.xsl.XslSheetFactory.parseXslSheet(theURL);
                boolean defaultXsl = (null!=req.getParameter(AdminXslTmplXHTML.NAME_defaultxsl));
                boolean externalXsl = (null!=req.getParameter(AdminXslTmplXHTML.NAME_externalxsl));
                
                /*
                // Get default xsl parameter
                boolean defaultXsl;
                if (null!=req.getParameter(AdminXslTmplXHTML.NAME_type)) {
                    defaultXsl=true;
                    
                    XslSheet theOldDefaultSheet=XslSheetFactory.findDefaultXslSheet();
                    if ( !(theOldDefaultSheet.isEmpty()) ) {
                        theOldDefaultSheet.setDefaultxsl(false);
                        theOldDefaultSheet.save();
                    }   
                } else {
                    defaultXsl=false;
                }
                */
                
                if (leCode !=null && !leCode.equals("")) {
                    XslSheetFactory.AddXslSheet(Nom, dictionaryName, volumeName, Description,leCode, defaultXsl, externalXsl);
                    userMessage = "XslSheet " + Nom + " added...";
                }
                else {
                    userMessage = "XslSheet " + Nom + " not uploaded, please check the URL...";
                    this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,userMessage);
               }
                
                
            // Remove xsl sheet    
            } else if ( (null != req.getParameter(REMOVE_PARAMETER))) {
                String handle = req.getParameter(REMOVE_PARAMETER);
                //XslSheet theSheet = XslSheetFactory.findXslSheetByHandle(handle);
                XslSheet theSheet = XslSheetFactory.getXslSheetByHandle(handle);
                theSheet.delete();
                XslSheetFactory.initializeXslSheetCache();
                userMessage = "XslSheet " + theSheet.getName() + " removed...";
            
            // set default xsl sheet
            } else if ((null != req.getParameter(DEFAULT_XSL_PARAMETER))) {
                String handle= req.getParameter(DEFAULT_XSL_PARAMETER);
                
                //
                //XslSheet theSheet = XslSheetFactory.findXslSheetByHandle(handle);
                XslSheet theSheet = XslSheetFactory.getXslSheetByHandle(handle);
                
                //
                XslSheet theOldDefaultSheet = XslSheetFactory.findDefaultXslSheet(theSheet.getDictionaryName(), theSheet.getVolumeName());
                
                //
                if ( (theOldDefaultSheet != null) && !(theOldDefaultSheet.isEmpty()) ) {
                    theOldDefaultSheet.setDefaultxsl(false);
                     theOldDefaultSheet.save();
                }                     
                
                //
                theSheet.setDefaultxsl(true);
                theSheet.save();
                
                //
                XslSheetFactory.initializeXslSheetCache();
                
                //
                userMessage = "XslSheet " + theSheet.getName() + " set default ...";
            
            // set external xsl sheet
            } else if ((null != req.getParameter(EXTERNAL_XSL_PARAMETER))) {
                String handle= req.getParameter(EXTERNAL_XSL_PARAMETER);
                
                //
                //XslSheet theSheet = XslSheetFactory.findXslSheetByHandle(handle);
                XslSheet theSheet = XslSheetFactory.getXslSheetByHandle(handle);
                
                //
                theSheet.setExternalxsl(true);
                theSheet.save();
                
                //
                XslSheetFactory.initializeXslSheetCache();
                
                //
                userMessage = "XslSheet " + theSheet.getName() + " set external xsl ...";
                
                // set external xsl sheet
            } else if ((null != req.getParameter(NO_EXTERNAL_XSL_PARAMETER))) {
                String handle= req.getParameter(NO_EXTERNAL_XSL_PARAMETER);
                
                //
                //XslSheet theSheet = XslSheetFactory.findXslSheetByHandle(handle);
                XslSheet theSheet = XslSheetFactory.getXslSheetByHandle(handle);
                
                //
                theSheet.setExternalxsl(false);
                theSheet.save();
                
                //
                XslSheetFactory.initializeXslSheetCache();
                
                //
                userMessage = "XslSheet " + theSheet.getName() + " set internal xsl ...";
                
            // see xsl sheet
            } else if ( (null != req.getParameter(SEE_PARAMETER))) {
                String handle = req.getParameter(SEE_PARAMETER);
                //XslSheet theSheet = XslSheetFactory.findXslSheetByHandle(handle);
                XslSheet theSheet = XslSheetFactory.getXslSheetByHandle(handle);
                //adding an XML file
                addXml(content, theSheet.getXmlCode());
            }
            else {
                String errorMessage = "Error: Wrong arguments";
                this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,errorMessage);
            }
            
            
            this.getSessionData().writeUserMessage(userMessage);
            PapillonLogger.writeDebugMsg(userMessage);
        }
        
        
        //
        HTMLTableRowElement theOriginalRow = content.getElementTemplateRow();
        HTMLElement theXslDefault = content.getElementXslDefault();
        HTMLElement theXslExternal = content.getElementXslExternal();
        HTMLAnchorElement theSeeAnchor = content.getElementSeeAnchor();
        HTMLAnchorElement theRemoveAnchor = content.getElementRemoveAnchor();
        HTMLAnchorElement theDefaultAnchor = content.getElementDefaultAnchor();
        HTMLAnchorElement theExternalAnchor = content.getElementExternalAnchor();
        HTMLAnchorElement theNoExternalAnchor = content.getElementNoExternalAnchor();
        
        Node theOriginalRowParent = theOriginalRow.getParentNode();
        
        theOriginalRow.removeAttribute("id");
        theXslDefault.removeAttribute("id");
        theXslExternal.removeAttribute("id");
        theSeeAnchor.removeAttribute("id");	
        theRemoveAnchor.removeAttribute("id");
        theDefaultAnchor.removeAttribute("id");
        theExternalAnchor.removeAttribute("id");

        // FIXME: create a original template AND tempory template 
        // -> don't set attribute class here (setAttribute("class","action")), only if jibiki hidden anchor (setAttribute("class","hidden"))
        // use NAME (like AdminXslTmplXHTML.NAME_defaultxsl) and duplicate node !
        
        //
        Collection XslSheetsTable = XslSheetFactory.getXslSheetsArray();
        for ( Iterator iter = XslSheetsTable.iterator(); iter.hasNext(); ) {
            
            //
            XslSheet xsl = (XslSheet)iter.next();
            content.setTextXslName(xsl.getName());
            content.setTextXslDictionaryName((null == xsl.getDictionaryName()) ? "" : xsl.getDictionaryName());
            content.setTextXslVolumeName((null == xsl.getVolumeName()) ? "" :  xsl.getVolumeName());
            content.setTextXslDesc(xsl.getDescription());
            
            //
            theSeeAnchor.setHref(this.getUrl() + "?" + SEE_PARAMETER + "=" + xsl.getHandle());
            theRemoveAnchor.setHref(this.getUrl()  + "?" + REMOVE_PARAMETER + "=" + xsl.getHandle());
            
            //
            if (xsl.isDefaultxsl()) {
                theXslDefault.setAttribute("class","");
                theDefaultAnchor. setHref("");
                theDefaultAnchor.setAttribute("class","hidden");
            } else {
                theXslDefault.setAttribute("class","hidden");
                theDefaultAnchor.setHref(this.getUrl() + "?" + DEFAULT_XSL_PARAMETER + "=" + xsl.getHandle());
                theDefaultAnchor.setAttribute("class","");
            }
            
            //
            if (xsl.isExternalxsl()) {
                theXslExternal.setAttribute("class","");
                theExternalAnchor.setHref("");
                theExternalAnchor.setAttribute("class","hidden");
                theNoExternalAnchor.setHref(this.getUrl() + "?" + NO_EXTERNAL_XSL_PARAMETER + "=" + xsl.getHandle());
                theNoExternalAnchor.setAttribute("class","");
            } else {
                theXslExternal.setAttribute("class","hidden");
                theNoExternalAnchor.setHref("");
                theNoExternalAnchor.setAttribute("class","hidden");
                theExternalAnchor.setHref(this.getUrl() + "?" + EXTERNAL_XSL_PARAMETER + "=" + xsl.getHandle());
                theExternalAnchor.setAttribute("class","");
            }
            
            theOriginalRowParent.appendChild(theOriginalRow.cloneNode(true));
        }
        
        theOriginalRowParent.removeChild(theOriginalRow);
        

        //On rends le contenu correst
        return content.getElementFormulaire();
    }
    protected void addXml(AdminXslTmplXHTML content, String xmlString) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            
        Node xmlNode = XslSheetFactory.applyXslSheetForXml(xmlString);    
            
        //where we must insert the xml
        HTMLElement theXml = content.getElementXml();

        Node theXmlParent = theXml.getParentNode();

        theXmlParent.appendChild(content.importNode(xmlNode,true));
        theXmlParent.removeChild(theXml);
    }
}
