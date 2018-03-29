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
 * Revision 1.3  2006/08/10 16:34:03  mangeot
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

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;

import com.lutris.appserver.server.Enhydra;
import com.lutris.util.KeywordValueException;
import com.lutris.util.KeywordValueTable;


public class InformationFileActionFactory {

    public static String CONTENT_TYPE_ACTION_CONFIG_GROUP = "Papillon.Informations.contentTypeHandler";
    public static String FILE_EXTENSION_ACTION_CONFIG_GROUP = "Papillon.Informations.fileExtensionHandler";
    //    public static String NO_ACTION_CLASS = "fr.imag.clips.papillon.business.informationfile.DoNothing";
    // Here I cheat because I am tired of adding new importMediaFile for each file extension ...
    public static String NO_ACTION_CLASS = "fr.imag.clips.papillon.business.informationfile.ImportMediaFile";

    public static InformationFileAction getAction(de.opus5.servlet.UploadedFile file) throws 
                                PapillonBusinessException {
        String fileName = file.getName();
        String contentType = file.getContentType();
        return getAction(fileName, contentType);
    }
    
    public static InformationFileAction getAction(java.io.File file) throws 
                                PapillonBusinessException {
        
        String fileName = file.getName();
        return getAction(fileName, null);
     }
    
    public static InformationFileAction getAction(String name, String contentType) throws PapillonBusinessException {
            // According to the file type and/or name, determine the actions to perform

        String handler = null;
       // PapillonLogger.writeDebugMsg("MIME type:" + contentType);
       // PapillonLogger.writeDebugMsg("file name:" + name);

        if (null != contentType) {
            try {
                int i = contentType.indexOf("/");
                String category = contentType.substring(0, i).replace('-', '_');
                String type = contentType.substring(i+1).replace('-', '_');
                try {
                    KeywordValueTable handlers = 
                        Enhydra.getApplication().getConfig().getSection(CONTENT_TYPE_ACTION_CONFIG_GROUP);
                    try {
                        handler = handlers.getString(category + "." + type);
                    } catch (KeywordValueException kve) {
                        PapillonLogger.writeDebugMsg("Unknown MIME type: " + category + "/" + type 
                            + " I will try to use filename..." );
                    }
                } catch (KeywordValueException kve) {
                    PapillonLogger.writeDebugMsg("Unexpected Error: Incorrect configuration file for group: "
                        + CONTENT_TYPE_ACTION_CONFIG_GROUP);
                }
            } catch (IndexOutOfBoundsException e) {
                PapillonLogger.writeDebugMsg("Unexpected Error: malformed content type: " + contentType);
            }
        }
        
        // If the handler is not found yet, try to find it from the filename extensions.
        if (null == handler) {
            if (null != name) {
                try {
                    int i = name.lastIndexOf(".");
                    String ext = name.substring(i+1).toLowerCase();
                    try {
                        KeywordValueTable handlers = 
                            Enhydra.getApplication().getConfig().getSection(FILE_EXTENSION_ACTION_CONFIG_GROUP);
                        try {
                            handler = handlers.getString(ext);
                        } catch (KeywordValueException kve) {
                            PapillonLogger.writeDebugMsg("Unknown file extension : \"" + ext + 
                                "\" The file will be ignored");
                        }
                    } catch (KeywordValueException kve) {
                        PapillonLogger.writeDebugMsg("Unexpected Error: Incorrect configuration file for group: "
                            + FILE_EXTENSION_ACTION_CONFIG_GROUP);
                    }
                } catch (IndexOutOfBoundsException e) {
                    PapillonLogger.writeDebugMsg("Unexpected Error: Can't determine file extension for file: " + name);
                }
            }
        }
        
        // If the handler is still not found, use the DoNothing Action.
        if (null == handler) {
            handler = NO_ACTION_CLASS;
        }
        
        // Now, get an instance of the handling class.
        InformationFileAction handlerObject = null;
        
        try {
            handlerObject = (InformationFileAction)Class.forName(handler).newInstance();
        } catch (Exception e) {
            throw new PapillonBusinessException("Information File handler not found", e);
        }
        
        return handlerObject;

    }
}
