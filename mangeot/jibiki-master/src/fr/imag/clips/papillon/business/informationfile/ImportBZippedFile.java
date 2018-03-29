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
 * Revision 1.3  2006/08/10 17:32:27  mangeot
 * *** empty log message ***
 *
 * Revision 1.2  2006/08/10 17:18:34  mangeot
 * *** empty log message ***
 *
 * Revision 1.1  2006/08/10 09:27:59  mangeot
 * Importing a bzipped .bz2 file
 *
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
import fr.imag.clips.papillon.business.PapillonImportException;
import fr.imag.clips.papillon.business.PapillonLogger;

import java.io.InputStream;
import java.io.IOException;

public class ImportBZippedFile extends InformationFileAction {
 
    MediaPreferences prefs;

    // Constructor...
    public ImportBZippedFile() {
        handlerName = "ImportBZippedFile";
	prefs = new MediaPreferences();
    }
    
    /**
     * perform the file addition.
     */
    public void addFile( de.opus5.servlet.UploadedFile file,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang)
            throws PapillonImportException, PapillonBusinessException,
		   javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
        PapillonLogger.writeDebugMsg("bunzipping File \""+ file.getName());
        try {
            // WARNING: we assume that the imported file is a DiskFile...
            // We should not. The location of the tmp folder should be known...
            // First, computing the output name
            String source = file.getName();
            String dest = file.getFile().getParent() + java.io.File.separator + 
                            "unbz-" + doc.getHandle();
            int len = source.length();
            if (len > 3
                && ".bz".equalsIgnoreCase(source.substring(len-3))) {
                dest = dest + source.substring(0, len-3);
            } else {
                dest = dest + source;
            }

            InputStream is = file.getInputStream();
            
            this.addFile(is, dest, doc, env, lang);
         } catch (java.io.IOException e) {
            PapillonLogger.writeDebugMsg("IOException encountered while importing GZipped file.");
        }
        
        PapillonLogger.writeDebugMsg("File \""+ file.getName() + "\" Imported !");
    }
           
    public void addFile( java.io.File file,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang)
            throws PapillonImportException, PapillonBusinessException,
		   javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
        PapillonLogger.writeDebugMsg("bunzipping File \""+ file.getName());
        try {
            // First, computing the output name
            String source = file.getName();
            String dest = file.getParent() + java.io.File.separator + 
                            "unbz-" + doc.getHandle();
            int len = source.length();
            if (len > 3
                && ".bz".equalsIgnoreCase(source.substring(len-3))) {
                dest = dest + source.substring(0, len-3);
            } else {
                dest = dest + source;
            }

            
            InputStream is = (InputStream) (new java.io.FileInputStream(file));
            this.addFile(is, dest, doc, env, lang);
        } catch (java.io.IOException e) {
            PapillonLogger.writeDebugMsg("IOException encountered while importing GZipped file.");
        } 
        
        PapillonLogger.writeDebugMsg("File \""+ file.getName() + "\" Imported !");
    }

    public void addFile( InputStream is,
                         String dest,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang) throws java.io.IOException,
					     PapillonImportException, fr.imag.clips.papillon.business.PapillonBusinessException,
					     javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
            
        if (is.available() == 0) {
            PapillonLogger.writeDebugMsg("ImportBZippedFile: Encountered an empty GZipped file.");
	    throw new PapillonImportException("the BZipped file is empty.");
        } else {
            // bunzip the file, then apply the appropriate action, based on gunzipped file type.
            
            java.io.FileOutputStream out = null;
            org.apache.tools.bzip2.CBZip2InputStream zIn = null;
            try {
                out = new java.io.FileOutputStream(dest);
                zIn = new org.apache.tools.bzip2.CBZip2InputStream(is);
                byte[] buffer = new byte[8 * 1024];
                int count = 0;
                do {
                    out.write(buffer, 0, count);
                    count = zIn.read(buffer, 0, buffer.length);
                } while (count != -1);
            } catch (IOException ioe) {
                String msg = "Problem expanding bzip " + ioe.getMessage();
                PapillonLogger.writeDebugMsg(msg);
                throw new PapillonBusinessException(msg, ioe);
            } finally {
                if (out != null) {
                    try {
                        out.close();
                    } catch (IOException ioex) {}
                }
                if (zIn != null) {
                    try {
                        zIn.close();
                    } catch (IOException ioex) {}
                }
            }
        
        java.io.File unbzFile = new java.io.File(dest);
        
        // Then, we should recursively call the addFile with the apprpriate handler...
        InformationFileAction handlerObject = InformationFileActionFactory.getAction(unbzFile);
        
        handlerObject.addFile(unbzFile, doc, env, lang);
        unbzFile.deleteOnExit();
        }
    }
    
    /**
     * Return the url where the given file will be found after an import...
     */
    public String getNameOnServer(String filename, String oid) {
        int dot = filename.lastIndexOf(".");
        String ext = filename.substring(dot+1);
        
        return prefs.getMediaDirName() + oid + "." + ext;
    }
 
}

