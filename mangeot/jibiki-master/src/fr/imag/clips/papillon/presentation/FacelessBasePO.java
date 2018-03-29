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
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

import fr.imag.clips.papillon.business.PapillonBusinessException;

import java.io.IOException;

import org.w3c.dom.Node;

public abstract class FacelessBasePO extends AbstractPO {
    

    /** This method should process the request and throw a Redirect exception !
    */
    public abstract void processRequest() throws ClientPageRedirectException, Exception;
    
    /**
    *  Returns the complete document.
     *
     * @exception  Exception
     * @exception  HttpPresentationException  Description of the Exception
     * @exception  IOException                Description of the Exception
     */
    public Node getDocument()
    throws HttpPresentationException, IOException, Exception {
        
        // The process request method MUST return a ReirectException as this faceless presentation object
        // is to be used to redirect the user somewhere AFTER a processing has been done.
        processRequest();
        return null;
    }
}
