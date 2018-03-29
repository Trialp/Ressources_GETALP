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
 * Revision 1.3  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.2.2.1  2005/03/29 09:41:33  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
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

package fr.imag.clips.papillon.business.message;


//import java.net.URL;
//import java.sql.*;
import java.util.*;
import java.io.*;

import fr.imag.clips.papillon.business.message.*;
import fr.imag.clips.papillon.business.*;
import fr.imag.clips.papillon.CurrentDBTransaction;

import fr.imag.clips.papillon.data.*;

//import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonLogger;

public class MessageDBLoader extends PapillonUserLogger {
    protected String repositoryPathname;
    
    public MessageDBLoader(String pathname) {
        this.repositoryPathname = pathname;
    }
    
    // Filtre des fichiers message
    static class MessageFileFilter implements FilenameFilter {
	
	public boolean accept(File dir, String name) {
	    return (name.startsWith("msg") && name.endsWith(".html"));
	}
	
    }
    
    // Filtre des fichiers attachements
    static class AttachmentsFileFilter implements FilenameFilter {
	// La liste des extension connues par mhonarc 2.4.8, telle que définie dans
	// mhmimetypes.pl


	String knownExtensions[] = {"bin", "123", "aif", "apr", "asd", "asn", "asp", "avi",
				    "bcpio", "bin", "bmp", "bz2", "c3d", "cdf", "cgm", "chm",
				    "cif", "clp", "cml", "cpio", "crd", "csh", "css", "cxf",
				    "doc", "dot", "dvi", "dwg", "emb", "es", "etc", "evy", "fif", 
				    "fm", "fvi", "g3f", "gal", "gau", "gcg", "gen", "gif", "gtar",
				    "gz", "hdf", "hlp", "hpg", "hqx", "html", "ief", "ifs", "iges", 
				    "ins", "jdx", "jpg", "js", "kin", "ksh", "latex", "lcc", "lwp", 
				    "m14", "ma", "man", "mbd", "mdb", "me", "mesh", "midi", "mif", "mmd",
				    "mny", "moc", "mol", "mop", "mov", "movie", "mp2", "mpg", "mpp", "ms", 
				    "oda", "org", "pac", "pae", "patch", "pbm", "pcx", "pdb", "pdf", "pgm", 
				    "pgp", "pict", "pl", "png", "pnm", "ppm", "ppt", "proxy", "prz", "ps", 
				    "pub", "ra", "ras", "rdf", "rgb", "roff", "ros", "rtf", "rtx", "rxn", 
				    "scd", "scm", "script", "sdf", "sgml", "sh", "shar", "sit", "slc", 
				    "smi", "smp", "snd", "spr", "src", "stx", "sv4cpio", "sv4crc", "talk",
				    "tar", "tbp", "tbt", "tcl", "tex", "texinfo", "tgf", "tif", "tki",
				    "trm", "tsv", "txt", "ustar", "v5d", "viv", "wav", "wi", "wmf",
				    "wp", "wri", "wrl", "wv", "xbm", "xls", "xpm", "xwd", "xyz", "Z", "zip"} ;
	public boolean accept(File dir, String name) {
	    int i=0;
	    while (i != knownExtensions.length &&    
		   !(name.startsWith(knownExtensions[i]) && name.endsWith("." + knownExtensions[i]))) {
		i++;
	    }
	    
	    return i != knownExtensions.length;
	}
    }
    
    // return a status message...
    public void updateDatabase() throws PapillonBusinessException {
        
	try {
            // First, create and load the archive
            MhonarcArchive dbpap = new MhonarcArchive(repositoryPathname);
	    
            // Then, check the directory in order to find new messages.
	    File dir = new File(repositoryPathname);
            if (!dir.isDirectory()) {
                dir = dir.getParentFile();
            }
	    
            // Make sure we are all right
            // This should never happen as the directory is checked when creating the archive object.
	    if (!dir.exists()) throw new java.io.FileNotFoundException("Specified directory does not exist.");
	    if (!dir.isDirectory()) throw new java.io.FileNotFoundException("Specified Path is not a directory.");
            
            // Load the names of messages.
	    String messages [] = dir.list(new MessageFileFilter());
	    
	    Vector newMessages = new Vector();
	    for (int i=0; i<messages.length; i++) {
		if (! dbpap.containsMessage(messages[i])) {
		    newMessages.addElement(messages[i]);
		}
	    }
	    
	    String attachments [] = dir.list(new AttachmentsFileFilter());
	    Vector newAttachments = new Vector();
	    for (int i=0; i<attachments.length; i++) {
		//System.out.println(messages[i]);
		if (! dbpap.containsAttachment(attachments[i])) {
		    newAttachments.addElement(attachments[i]);
		}
	    }
	    if (newMessages.size() != 0) { 
		    // Put the new messages in the database
		    Iterator iter = newMessages.iterator();
		    while (iter.hasNext()) { 
			String elt = (String) iter.next();
			try {
			    // Ajout du message dans la base de donnée
			    this.insertMessageDOFromMhonarcFile(repositoryPathname + File.separator + elt);
			    
			    // Puis ajout du message dans la minibase papillon
			    dbpap.addMessage(elt);
			    
			} catch (PapillonBusinessException e) {
                            this.writeUserLog("Warning: Business Exception when inserting Message\"" + elt + ", ignored.");
                            this.writeUserLog(e.getMessage());
			} catch (NotAValidUTF8Message e) {
                            this.writeUserLog("Warning: " + elt + " is not a valid UTF8 message, ignored");
                            this.writeUserLog(e.getMessage());
                        } catch (Exception e) {
                            this.writeUserLog("Warning: Unexpected Exception when inserting Message \"" + elt + ", ignored.");
                            this.writeUserLog(e.getMessage());
                        }
		    }
	    }
	    // Je ne sais pas encore comment se fait le traitement des attachements
	    // Soit on les laisse sur place, soit on les déplace vers un dossier web.
	    // Pour l'instant, on se contente de les noter dans la minibase
            Iterator iter = newAttachments.iterator();
	    while (iter.hasNext()) { 
		String elt = (String) iter.next();
		dbpap.addAttachment(elt);
		
	    }
	    // Enfin, on sauvegarde la minibase le cas échéant.
	    dbpap.releaseDB();
	} catch (NotAPapillonArchive e) {
	    this.writeUserLog("Invalid Papillon database, aborting.");
            if (null != e.getMessage()) {
                this.writeUserLog(e.getMessage());
            }
	} catch (PapillonArchiveIsLocked e) {
	    this.writeUserLog("The Papillon database appears to be locked by another process.");
	    this.writeUserLog("  --> Please wait for the other process to terminate.");
	    this.writeUserLog("  --> If no other process is currently running, just remove the db lock file");
	} catch (Exception e) {
            throw new PapillonBusinessException("Exception caught during database update.", e);
        }
	
    } 
    
    /** 
     * The insertMessageDOFromMhonarcFile method returns a <CODE>Message</CODE>
     * object corresponding to the MhonarcMessage filename given in parameter.
     * The author and Threads are retreived from the database if necessary.
     *
     * @param filename, the Mhonarc Message filename.
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
    public void insertMessageDOFromMhonarcFile(String filename) 
        throws PapillonBusinessException, Exception {
        try {
            MhonarcMessage msg = new MhonarcMessage(filename);
        
            // The msgid field is unique... So first check that is is not in the database already
            // Can't this be done at the database level ?
            MessageQuery mq = new MessageQuery(CurrentDBTransaction.get());
            mq.setQueryMsgid(new String(msg.msgId));
            // Throw an exception if more than one message is found
            mq.requireUniqueInstance();
            MessageDO theMessageDO = mq.getNextDO();
            
            if (null != theMessageDO) {
                throw new DuplicateMessageException("A message with the same id is already in the database.");
            }
            
            String authorName;
            theMessageDO = MessageDO.createVirgin(CurrentDBTransaction.get());
            if ((null == msg.auteur) || (msg.auteur.trim().equals(""))) {
                authorName = new String(msg.adresseAuteur);
            } else {
                authorName = new String(msg.auteur);
            }

            theMessageDO.setAuthor(authorName);
            theMessageDO.setAuthorAddress(new String(msg.adresseAuteur));
            
            theMessageDO.setThread(new String(msg.sujetCannonique()));
            
            theMessageDO.setDate(new java.sql.Date(msg.date.getTime()));
            theMessageDO.setMsg(new String(msg.corps));
            theMessageDO.setMsgid(new String(msg.msgId));
            theMessageDO.setSubject(new String(msg.sujet));
            
            theMessageDO.save();
            
        } catch (DuplicateMessageException ex) {
            // C'est pas grave, on ne fait rien...
            PapillonLogger.writeDebugMsg(ex.getMessage());
	} catch(Exception ex) {
            PapillonLogger.writeDebugMsg(ex.getMessage());
            throw ex;
        }
    }    

}
