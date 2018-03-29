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
 * Revision 1.4  2006/08/10 17:32:27  mangeot
 * *** empty log message ***
 *
 * Revision 1.3  2006/08/10 17:18:34  mangeot
 * *** empty log message ***
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

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonImportException;
import fr.imag.clips.papillon.business.PapillonLogger;

public class ImportMediaFile extends InformationFileAction {
    MediaPreferences prefs;


    // Constructor...
    public ImportMediaFile() {
        handlerName = "ImportMediaFile";
        prefs = new MediaPreferences();
    }
    
    /**
     * perform the file addition.
     */
    public void addFile( de.opus5.servlet.UploadedFile file,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang)
            throws PapillonBusinessException ,PapillonImportException{
        String name = file.getName();
        PapillonLogger.writeDebugMsg("importing Media File \""+ name);
        try {
            // WARNING: we assume that the imported file is a DiskFile...
            // We should not. The location of the tmp folder should be known...
            // First, computing the output name
            String path = file.getFile().getCanonicalPath();
            String oid = env.getFileIDForURL(path);
                        
            InputStream is = file.getInputStream();
//            PapillonLogger.writeDebugMsg("oid --> \""+ oid + "\"; path --> \"" + path + "\"");

            this.addFile(is, name, oid, doc, lang);
         } catch (java.io.IOException e) {
            PapillonLogger.writeDebugMsg("IOException encountered while importing Media file.");
        }
        
        PapillonLogger.writeDebugMsg("Media File \""+ name + "\" Imported !");
    }
           
    public void addFile( java.io.File file,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang)
            throws PapillonBusinessException ,PapillonImportException{
        String name = file.getName();
        PapillonLogger.writeDebugMsg("copying File \""+ name);
        try {
            String path = file.getCanonicalPath();
            String oid = env.getFileIDForURL(path);
            
            InputStream is = (InputStream) new FileInputStream(file);
            
            this.addFile(is, name, oid, doc, lang);
        } catch (java.io.IOException e) {
            PapillonLogger.writeDebugMsg("IOException encountered while importing Media file.");
        } 
        
        PapillonLogger.writeDebugMsg("Media File \""+ name + "\" Imported !");
    }

    public void addFile( InputStream is,
                         String name,
                         String oid,
                         InformationDocument doc,
			 String lang) throws java.io.IOException,
                                             fr.imag.clips.papillon.business.PapillonBusinessException ,PapillonImportException{
            
        if (is.available() == 0) {
            PapillonLogger.writeDebugMsg("ImportMediaFile: Encountered an empty media file.");
	    throw new PapillonImportException("a Media file is empty.");
        } else {
            // copy the file to dest.
            if (null == oid) {
                // The file object has not been created already because it is the document (and not a part of it)
		// check that the document has a title
                if (doc.getTitle().equals("")) {
		    throw new PapillonImportException("please, enter a title for this document");
		}
		int dot = name.lastIndexOf(".");
                String ext = name.substring(dot+1);
                InformationFile theFileObject=InformationFileFactory.RegisterNewInformationFile("", ext, "", doc);
                oid = theFileObject.getHandle();
		theFileObject.setFilename(this.getNameOnServer(name, oid));
                theFileObject.setIsIndexFile(true);
		theFileObject.setLanguage(lang);
                theFileObject.save();
            }
            String dest = prefs.getBaseDirName() + this.getNameOnServer(name, oid);

            FileOutputStream out = null;
            try {
                out = new FileOutputStream(dest);
                byte[] buffer = new byte[8 * 1024];
                int count = 0;
                do {
                    out.write(buffer, 0, count);
                    count = is.read(buffer, 0, buffer.length);
                } while (count != -1);
            } catch (IOException ioe) {
                String msg = "Problem copying media file: " + ioe.getMessage();
                PapillonLogger.writeDebugMsg(msg);
                throw new PapillonBusinessException(msg, ioe);
            } finally {
                if (out != null) {
                    try {
                        out.close();
                    } catch (IOException ioex) {}
                }
            }
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

