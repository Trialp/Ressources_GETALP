/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.informationfile;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonImportException;

public abstract class InformationFileAction {

    public static String handlerName = null;

    /**
     * perform the file addition.
     */
    public abstract void addFile( de.opus5.servlet.UploadedFile file,
                                  InformationDocument doc,
                                  RelativeURLMapper env,
				  String lang)
                
            throws PapillonBusinessException, PapillonImportException,
		   javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException;
            
    public abstract void addFile( java.io.File file,
                                  InformationDocument doc,
                                  RelativeURLMapper env,
				  String lang)
                
            throws PapillonBusinessException, PapillonImportException,
		   javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException;
            
    /**
     * Return the url where the given file will be found after an import...
     */
    public abstract String getNameOnServer(String filename, String oid); 
    

    /**
     * Return the name of the handler
     */
    public String getHandlerName() {
	return handlerName;
    }

}
