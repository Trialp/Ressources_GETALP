/*
 *  jibiki platform
 *
 *  Enhydra super-servlet presentation object
 *
 * © GETA CLIPS IMAG
 *
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.2  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.1  2006/07/10 11:07:15  serasset
 * Added a way to call system commands.
 * Added a call to a saxon 8b transformer.
 * Added a Faceless base PO (used for PO that always redirect the client somewhere)
 *
 *
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;

import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.system.Saxon8bTransformer;

public class LaunchSaxon8bTransformation extends FacelessBasePO {
   
    final static String XSL_PARAMETER = "xsl";
    final static String XML_PARAMETER = "xml";
    
    protected boolean loggedInUserRequired() {
        return true;
    }
	
    protected boolean userMayUseThisPO() {
        return true;
    }
        
    public void processRequest() 
    throws HttpPresentationException,
    java.io.IOException,
    PapillonBusinessException {

//        HttpPresentationRequest req = this.getComms().request;
        String redirectUrl = "";        

        String xsl = myGetParameter(XSL_PARAMETER);
        String xml = myGetParameter(XML_PARAMETER);
        
        String out = "/tmp/foresult.fo";
        
        Saxon8bTransformer.doSaxonTransformation(xsl,xml,out);
        
        throw new ClientPageRedirectException("Home.po");
    }
    
}

