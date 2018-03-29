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

import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonImportException;
import fr.imag.clips.papillon.business.PapillonLogger;

import org.apache.tools.tar.*;

import java.io.*;
import java.util.Vector;
import java.util.Iterator;

public class ImportTarFile extends ImportArchive {

    MediaPreferences prefs;

    // Constructor...
    public ImportTarFile() {
        handlerName = "ImportTarFile";
	prefs = new MediaPreferences();
    }
    
    /**
     * perform the file addition.
     */
    public void addFile( de.opus5.servlet.UploadedFile file,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang)
            throws PapillonImportException, PapillonBusinessException, ClientPageRedirectException,
		   javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {

        PapillonLogger.writeDebugMsg("untaring File \""+ file.getName());
        try {
            // Get the data
            InputStream is = file.getInputStream();
            String outputDirName = file.getFile().getParent() + File.separator + "untar-"+ doc.getHandle();

            this.addFile(is, outputDirName, doc, env, lang);
        } catch (java.io.IOException e) {
            PapillonLogger.writeDebugMsg("IOException encountered while importing Tar file.");
        }
        
        PapillonLogger.writeDebugMsg("File \""+ file.getName() + "\" Imported !");
    }
           
           
    public void addFile( java.io.File file,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang)
            throws PapillonImportException, PapillonBusinessException, ClientPageRedirectException,
		   javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
        PapillonLogger.writeDebugMsg("untaring File \""+ file.getName());
        try {
            InputStream is = (InputStream) new FileInputStream(file);
            String outputDirName = file.getParent() + File.separator + "untar-"+ doc.getHandle();
            this.addFile(is, outputDirName, doc, env, lang);
        } catch (java.io.IOException e) {
            PapillonLogger.writeDebugMsg("IOException encountered while importing Tar file.");
        }
        
        PapillonLogger.writeDebugMsg("File \""+ file.getName() + "\" Imported !");
    }

    public void addFile( InputStream is,
                         String outputDirName,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang)
            throws java.io.IOException, PapillonImportException, PapillonBusinessException, ClientPageRedirectException,
		   javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
        
	if (is.available() == 0) {
            PapillonLogger.writeDebugMsg("ImportTarFile: Encountered an empty Tar file.");
	    throw new PapillonImportException("the Tar file is empty.");
        } else {
            // Create the output directory
            File outputDir = new File(outputDirName);
            outputDir.mkdirs();
            
            // untar the file, 
            TarInputStream tis = new TarInputStream(is);
            TarEntry te = null;
            Vector files = new Vector();
            
            while ((te = tis.getNextEntry()) != null) {
                File f = new File(outputDir, te.getName());            
                PapillonLogger.writeDebugMsg("expanding " + te.getName() + " to "+ f); 
                File dirF=new File(f.getParent());
                dirF.mkdirs();
                dirF.deleteOnExit();
                if (te.isDirectory()) {
                    f.mkdirs();
                    f.deleteOnExit();
                } else {
                    byte[] buffer = new byte[1024];
                    int length = 0;
                    FileOutputStream fos = new FileOutputStream(f);
                    while ((length = tis.read(buffer)) >= 0) {
                        fos.write(buffer, 0, length);
                    }
                    fos.close();
                    try {
                         
                        // Create an empty information file object and associates its Handle to its name
                        InformationFile inf = 
                            InformationFileFactory.RegisterNewInformationFile("" , "", "", doc);
                        env.setFileIDForURL(f.getCanonicalPath(), inf.getHandle());
                        
                        // Calculate its name on the server...
                        InformationFileAction handlerObject = InformationFileActionFactory.getAction(f);
                        inf.setFilename(handlerObject.getNameOnServer(te.getName(), inf.getHandle()));
			inf.save();
                        files.addElement(te.getName());
                    } catch (java.io.IOException e) {
                        PapillonLogger.writeDebugMsg("getCanonicalPath threw an exception !!!");
                    }
                }
            }
            tis.close();
            
            // Perform effective files addition
	    addArchiveFiles(files, outputDir, doc, env, lang);


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


