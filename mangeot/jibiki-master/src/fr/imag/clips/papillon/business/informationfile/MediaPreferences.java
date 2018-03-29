/*
 *  papillon
 *
 *  Preference object for the information file management 
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
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

import fr.imag.clips.papillon.business.PapillonLogger;

import com.lutris.appserver.server.Enhydra;
import com.lutris.util.KeywordValueException;

import java.io.File;

public class MediaPreferences {
    public final static String MEDIA_DIR_CONFIG = "Papillon.Informations.mediaDir";
    public final static String BASE_DIR_CONFIG = "Papillon.Informations.baseDir";
    public final static String RELATIVE_DIR_CONFIG = "Papillon.Informations.relativeDir";
    public final static String FILE_VIEWER_URL = "ConsultInformations.po";
    public final static String FILE_ID_PARAMETER = "fileid";
    public final static String DOC_ID_PARAMETER = "docid";
    public final static String DOC_LANG_PARAMETER = "docLang";
    
    protected String mediaDir;
    protected String baseDir;
    protected String relativeDir;

    public MediaPreferences() {
        try {
            mediaDir = Enhydra.getApplication().getConfig().getString(MEDIA_DIR_CONFIG);
            if (! mediaDir.endsWith(File.separator)) {
                mediaDir = mediaDir + File.separator;
            }
            baseDir = Enhydra.getApplication().getConfig().getString(BASE_DIR_CONFIG);
            if (! baseDir.endsWith(File.separator)) {
                baseDir = baseDir + File.separator;
            }
            relativeDir = Enhydra.getApplication().getConfig().getString(RELATIVE_DIR_CONFIG);
            if (! relativeDir.endsWith(File.separator)) {
                relativeDir = relativeDir + File.separator;
            }
            File baseFolder = new File(baseDir);
            File mediaFolder = new File(baseFolder, mediaDir);
            mediaFolder.mkdirs();
        } catch (KeywordValueException kve) {
            PapillonLogger.writeDebugMsg("Unexpected Error: Incorrect configuration file for group: "
                            + MEDIA_DIR_CONFIG);
        } catch (Exception e) {
            PapillonLogger.writeDebugMsg("Unexpected IO Exception while importing a media file.");
        }

    }
    
    public String getMediaDirName() {
        return mediaDir;
    }
    
    public String getBaseDirName() {
        return baseDir;
    }
    
    public String getRelativeDirName() {
        return relativeDir;
    }
    
    public String getRelativeMediaDirName() {
        return relativeDir + mediaDir;
    }
    
    public String getConsultFileUrl() {
        return FILE_VIEWER_URL + "?" + FILE_ID_PARAMETER + "=";
    }
    
    public String getConsultDocumentUrl() {
        return FILE_VIEWER_URL + "?" + DOC_ID_PARAMETER + "=";
    }
    
}
