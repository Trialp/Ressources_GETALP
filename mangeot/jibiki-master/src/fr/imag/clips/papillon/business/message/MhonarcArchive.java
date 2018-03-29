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

package fr.imag.clips.papillon.business.message;

import fr.imag.clips.papillon.data.*;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Iterator;
import java.util.Vector;

import com.lutris.appserver.server.Enhydra;
import com.lutris.logging.Logger;

/**
 * Represents a Mailing list message. 
 */
public class MhonarcArchive {

    final static String DEFAULT_DB_NAME = ".dbpap";
    final static String LOCK_FILE_NAME = DEFAULT_DB_NAME + ".lock";
    final static String MSG_HEADER = "---+++Messages+++---";
    final static String ATTACHMENTS_HEADER = "---+++Attachments+++---";
    final static String DB_FOOTER = "---+++EndOfPapillonDb+++---";
    
    String pDBName = null;
    Vector messages = new Vector();;
    Vector attachments = new Vector();;
    boolean isDirty = false;               // Boolean stating if the database has to be changed or not
    File lockFile;
    File pDBFile;
    
    
    
    // Création de la base à partir d'un fichier ou d'un dossier
    // La base est une simple collection de nom de fichiers.
    MhonarcArchive(String dbName) throws NotAPapillonArchive, PapillonArchiveIsLocked {
        // Le nom donné est-il celui d'un dossier ou d'un fichier ?
        File f = new File(dbName);
        if (f.exists()) {
            if (f.isDirectory()) {
                pDBName = dbName +  File.separator + DEFAULT_DB_NAME;
            } else {
                pDBName = dbName;
            }
        } else {
            File p = f.getParentFile();
            if (p.exists()) {
                pDBName = dbName;
            } else {
                throw new NotAPapillonArchive("Specified directory not found");
            }
        }
        this.loadDB();
    }
    
    
    private void loadDB() throws NotAPapillonArchive, PapillonArchiveIsLocked {
	try {
            File f = new File(pDBName);
            File dir = f.getParentFile();
	    // First, create a lock file to avoid confusion
	    lockFile = new File(dir.toString() + File.separator + LOCK_FILE_NAME);
            
	    if (lockFile.createNewFile() ) {
                // On demande à java d'effacer ce fichier lorsque la machine virtuelle s'arrêtera.
		lockFile.deleteOnExit();
            
                pDBFile = new File(pDBName);
		InputStreamReader reader = new InputStreamReader(new FileInputStream(pDBFile));
		BufferedReader lect = new BufferedReader (reader);
		String line;
		
		// Lecture de l'entête des messages
		while (! lect.readLine().equals(MSG_HEADER)) ;
		// Accumulation des messages
		line = lect.readLine();
		while (!line.equals(ATTACHMENTS_HEADER)) {
		    messages.addElement(new String(line));
		    line = lect.readLine();
		}
		// Accumulation des Attachements
		line = lect.readLine();
		while (!line.equals(DB_FOOTER)) {
		    attachments.addElement(new String(line));
		    line = lect.readLine();
		}
	    } else {
                if (!lockFile.canWrite()) {
                    throw new PapillonArchiveIsLocked();
                } else {
                    throw new NotAPapillonArchive("Error creating the lock file. Are you sure the server has the adequate rights ?");
                }
                
	    }
	    
	} catch (FileNotFoundException e) {
	    // Cette exception survient si le fichier n'est pas trouvé, il devra donc être créé
	    // à la sauvegarde.
	    isDirty = true;
	} catch (IOException e) {
	    // Là, je ne sais pas trop ce qu'il faut faire...
	    throw new NotAPapillonArchive();
	} catch (NullPointerException e) { 
	    // Cette exception ne peut survenir que si on n'a pas trouvé l'un des éléments
	    // propres à la DB.
	    throw new NotAPapillonArchive();
	}
    }

        
    public boolean containsMessage(String msg) {
	return messages.contains(msg);
    }

    public boolean containsAttachment(String att) {
	return attachments.contains(att);
    }
    
    public void addMessage(String msgName) {
	messages.addElement(msgName);
	isDirty = true;
    }
    
    public void addAttachment(String attachName) {
	attachments.addElement(attachName);
	isDirty = true;
    }


    /**
     * This method should be called when the Database is no longer in use.
     * It deletes the lock file of the archive, hence allowing further access
     * to the same archive.
     */
    public void releaseDB() {
	try {
	    if (isDirty) {
		// On doit sauver les changements
		File newFile = new File(pDBName + ".new");
		BufferedWriter out = 
		    new BufferedWriter(new OutputStreamWriter(new FileOutputStream(newFile)));
		
		out.write(MSG_HEADER, 0, MSG_HEADER.length());
		out.newLine();
		Iterator iter = messages.iterator();
		while (iter.hasNext()) {
		    String elt = (String) iter.next();
		    out.write(elt, 0, elt.length());
		    out.newLine();
		}
		out.write(ATTACHMENTS_HEADER, 0, ATTACHMENTS_HEADER.length());
		out.newLine();
		iter = attachments.iterator();
		while (iter.hasNext()) {
		    String elt = (String) iter.next();
		    out.write(elt, 0, elt.length());
		    out.newLine();
		}
		out.write(DB_FOOTER, 0, DB_FOOTER.length());
		out.newLine();
		
		out.close();
		File saveFile = new File(pDBName + ".save");
		pDBFile.renameTo(saveFile);
		newFile.renameTo(pDBFile);
		saveFile.delete();
	    }
	} catch (Exception e) {
	    // Peut mieux faire... Il faudrait voir quelle exception a eu lieu.
            writeDebugMsg(e.getMessage());
	}
       
	// Finallement, on supprime le fichier de lock.
	lockFile.delete();
    }

    /**
     * Method to write a debugging message to the debug log
     * channel when the DEBUG flag is turned on
     * 
     * @param msg The message to write to the DEBUG log channel
     */
    public static void writeDebugMsg(String msg) {
        Enhydra.getLogChannel().write(Logger.DEBUG,msg);
    }

}
