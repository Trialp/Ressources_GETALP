/*
 *  Jibiki plate-form
 *
 *  © Gilles Sérasset - GETALP LIG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.1.2.2  2008/03/04 20:48:15  serasset
 *  Lazy DOM building in VolumeEntry.
 *  Export to Zip file now gives a correct filename.
 *
 *  Revision 1.1.2.1  2008/02/14 17:09:27  serasset
 *  Created an export into a zip file.
 *  Cosmetic changes in other files.
 *
 */
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports

import java.io.IOException;

import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;


/**
 * An HttpPresentation that allows for streaming of output directly to the client
 *
 */
public abstract class DownloadBasePO
        extends AbstractPO {
 
	/**
	 * decode request to determine content type and filename
	 * @throws HttpPresentationException
	 * @throws IOException
	 * @throws Exception
	 */
	public abstract void prepareRequest()throws HttpPresentationException, IOException, Exception;
	
	/**
	 * return the content type of the answer (e.g. binary/octet-stream; name="filename")
	 * @return content type
	 */
	public abstract String getContentType();
	
	/**
	 * return the content disposition (e.g. attachment; filename="tagada")
	 * @return content disposition
	 */
	public abstract String getFilename(); // return filename
	
	/**
	 * dump the requested data in the output stream
	 * @param out the HttpPresentationOutput to dump the data into
	 * @throws HttpPresentationException
	 * @throws IOException
	 * @throws Exception
	 */
	public abstract void dumpRequestedData(HttpPresentationOutputStream out) throws HttpPresentationException, IOException, Exception;

	
    /**
     * This implements the run method in HttpPresentation.
     *
     * @param comms Description of the Parameter
     * @throws Exception
     * @throws HttpPresentationException Description of the Exception
     * @throws IOException               Description of the Exception
     */
    public void run(HttpPresentationComms comms) throws HttpPresentationException, IOException, Exception {
        this.myComms = comms;
        initSessionData();

        // Check if the user needs to be logged in for this request.
        if (this.loggedInUserRequired()) {
            checkForUserLogin();                  // This will redirect the user to the login page if necessary
        }

        // After this point, user is logged in if required...

        if (!this.userMayUseThisPO()) {
            userIsNotAuthorized();                // This will redirect the user to the login page if necessary
        }
        
        // setContentType before calling getDocument
        // because getDocument can change the content type
        this.getComms().response.setContentType("text/html");

        try {
            initPresentationContext();
            prepareRequest();
            comms.response.setContentType( getContentType() );
            comms.response.setHeader("Content-Disposition", "attachment; filename=\"" + getFilename() + "\"");
            // comms.response.setContentLength( buffer.length );
            // dump...
            
            HttpPresentationOutputStream out = comms.response.getOutputStream();
            dumpRequestedData(out);
            out.flush();
        } finally {
            flushPresentationContext();
        }      
    }
}
