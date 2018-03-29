/*
 *  papillon
 *
 *  Enhydra super-servlet
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id: XmlBasePO.java 700 2006-09-10 09:18:25Z mangeot $
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.1.2.1  2006/09/10 09:18:25  mangeot
 *  Resoures modified and added for playing with authentic xml plugin
 *
 *  Revision 1.7  2006/07/15 08:55:14  mangeot
 *  The BrowseVolumePage opens an HTML form that is used to lookup a volume in alphabetical order.
 *  The BrowseVolume is the server side of the AJAX script for retrieving the entries in alphabetical order
 *
 *  Revision 1.6  2006/02/27 00:04:01  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.5  2006/02/26 22:05:02  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.4  2006/02/26 20:24:30  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.3  2006/02/26 19:58:18  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.2  2006/02/26 19:21:38  mangeot
 *  Work on BrowseVolume
 *
 *  Revision 1.1  2006/02/26 14:09:32  mangeot
 *  *** empty log message ***
 *
 *
 *  -----------------------------------------------
 *  beta version
 */
package fr.imag.clips.papillon.presentation;

import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.dods.builder.generator.query.QueryBuilder;

import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.Index;
import fr.imag.clips.papillon.business.dictionary.IndexFactory;

import fr.imag.clips.papillon.business.utility.Utility;

/**
*  Description of the Class
 *
 * @author     mangeot
 * @created    February 24, 2006
 */
public abstract class XmlBasePO extends AbstractPO {
    	
    /**
        *  This is the procedure that is called when an HTTP request occurs.
     *
     * @return                XMLObject The XMLObject (in XML format) that is
     *      to be included in the standard layout.
     * @exception  Exception  Description of the Exception
     */
    public abstract org.w3c.dom.Document getContent() throws Exception;
	
	
    /**
        *  Returns the complete document.
     *
     * @exception  Exception
     * @exception  HttpPresentationException  Description of the Exception
     * @exception  IOException                Description of the Exception
     */
    public org.w3c.dom.Node getDocument()
        throws HttpPresentationException, java.io.IOException, Exception {
			
			getComms().response.setContentType("text/xml");
			getComms().response.setEncoding("UTF-8");
						
            return (org.w3c.dom.Node) getContent();
        }
    
}
