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
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.informationfile;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;

public class DoNothing extends InformationFileAction {

    public DoNothing() {
        // Nada
    }
    /**
     * perform the file addition.
     */
    public void addFile( de.opus5.servlet.UploadedFile file,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang) 
                
            throws PapillonBusinessException {
        PapillonLogger.writeDebugMsg("File \""+ file.getName() + "\"ignored");
        
    }
    
    public void addFile( java.io.File file,
			   InformationDocument doc,
			   RelativeURLMapper env,
			   String lang) 
                
            throws PapillonBusinessException {
	PapillonLogger.writeDebugMsg("File \""+ file.getName() + "\"ignored");
    }
    
    /**
     * Return the url where the given file will be found after an import...
     */
    public String getNameOnServer(String filename, String oid) {
        return "";
    }


}
