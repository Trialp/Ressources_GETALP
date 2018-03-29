/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.3  2006/08/10 16:34:03  mangeot
 * *** empty log message ***
 *
 * Revision 1.2  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.1.1.1.2.2  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.1.1.1.2.1  2005/03/16 13:24:31  serasset
 * Modified all boolean fields in table to CHAR(1) in order to be more db independant.
 * Suppressed ant.jar from class path, informationfiles (which rely on it) should be corrected.
 * The version of Xerces is now displayed on application init.
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * Revision 1.1.1.1  2002/10/28 16:49:14  serasset
 * Creation of the papillon CVS repository for enhydra 5.0
 *
 * Revision 1.8  2002/07/26 16:51:11  tache
 * The document repository now manages multilingual documents.
 *
 * Revision 1.7  2002/02/06 15:03:02  serasset
 * Consultation and Administration of the information files now correctly use the Sections.
 * We do not display the files directly anymore, but display a document. This allows the display of
 * images as embedded docs in the default layout.
 *
 * Revision 1.6  2002/02/06 10:39:39  serasset
 * Remove document is now implemented.
 * The information sections are now defined in papillon's config file and handled in the Administration page.
 *
 * Revision 1.5  2001/10/29 13:32:18  serasset
 * Information section is now managed with Documents containing files. We now allow the addition of tar.gz files
 * containing several html files. images are still not supported.
 *
 * Revision 1.4  2001/10/17 12:59:56  serasset
 * L'ajout de document d'information se fait via une action specifique par type de fichier.
 * Distinction entre Document (abstrait) et Fichier (qui composent concretement un document).
 *
 * Revision 1.3  2001/09/07 06:53:45  serasset
 * File addition is now done with a set of Handler classes that are invoked depending on
 * the application configuration file.
 *
 * Revision 1.2  2001/08/28 17:58:32  salvati
 * Added the mainFile attribut and then the good listing.
 *
 * Revision 1.1  2001/08/01 15:15:29  salvati
 * Adding the interface with database.
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.informationfile;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.CurrentDBTransaction;

import fr.imag.clips.papillon.business.PapillonLogger;

import fr.imag.clips.papillon.business.utility.*;

import com.lutris.appserver.server.sql.ObjectId;


/**
 * Used to find the instances of InformationFile.
 */
public class InformationFileFactory {
    
    /** 
     * The getInformationFilesArray method performs a database query to
     * return an array of InformationFile
     * @return
     *     array of InformationFile. 
     * @exception PapillonBusinessException
     *   If there is a problem retrieving data.
     */
    public static InformationFile[] getInformationFilesArray() 
        throws PapillonBusinessException {
        InformationFile[] theFileArray = null;
        
        try {
            InformationFileQuery query = new InformationFileQuery(CurrentDBTransaction.get());
         
            InformationFileDO[] DOarray = query.getDOArray();
            theFileArray = new InformationFile[ DOarray.length ];
            for ( int i = 0; i < DOarray.length; i++ )
	    	theFileArray[i] = new InformationFile(DOarray[i]);
        return theFileArray;

        }catch(Exception ex) {
            throw new PapillonBusinessException("Exception in getInformationFilesArray()", ex);
        }
         
    }
    
    
    
    /** 
     * The findMessageByID method performs a database query to
     * return a <CODE>Message</CODE> object
     * representing the row in the <CODE>message</CODE> table
     * that matches the object id. 
     *
     * @param id, the object id of the message table.
     * @return
     *    the message. null if there isn't a message associated
     *    to the id
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
    public static InformationFile findInformationFileByID(String id) 
        throws PapillonBusinessException {
        InformationFile theFile = null;
        
        try {
            InformationFileQuery query = new InformationFileQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryOId(new ObjectId(id));
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            InformationFileDO theInformationFileDO = query.getNextDO();
            theFile = new InformationFile(theInformationFileDO);
            return theFile;
        } catch(Exception ex) {
            throw new PapillonBusinessException("Exception in findInformationFileByID()", ex);
        }
    }
    
    
    public static InformationFile findInformationFileByName(String name) 
        throws PapillonBusinessException {
        InformationFile theFile = null;
        
        try {
            InformationFileQuery query = new InformationFileQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryFilename(name);
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            InformationFileDO theInformationFileDO = query.getNextDO();			
				
            theFile = new InformationFile(theInformationFileDO);
            return theFile;
        } catch(Exception ex) {
            throw new PapillonBusinessException("Exception in findInformationFileByName()", ex);
        }
    }
    
    
    /** 
     * The findLocIndexFileForDocument method performs a database query to
     * return an <CODE>InformationFile</CODE> object
     * that is the index file for the passed document in the given lang. 
     *
     * @param doc, the InformationDocument;
     *        lang, the queried language.
     * @return
     *    the InformationFile. null if there isn't an index file associated
     *    to the document with that language
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
    public static InformationFile findLocIndexFileForDocument(InformationDocument doc, String lang) 
        throws PapillonBusinessException {
        InformationFile theFile = null;
        
        try {
            InformationFileQuery query = new InformationFileQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryDocument(doc.myDO);
            query.setQueryIsIndexFile("Y");
            query.setQueryLanguage(lang);
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            InformationFileDO theInformationFileDO = query.getNextDO();
            if (null != theInformationFileDO) {
                theFile = new InformationFile(theInformationFileDO);
            }
            return theFile;
        }catch(Exception ex) {
            throw new PapillonBusinessException("Exception in findInformationFileByID()", ex);
        }
    }

    
    public static InformationFile RegisterNewInformationFile(
            String filename, 
            String filetype,
	    String lang,
            InformationDocument doc) 
            throws PapillonBusinessException {
        InformationFile f = new InformationFile();
        
        f.setFilename(filename);
        f.setFiletype(filetype);
		f.setLanguage(lang);
        f.setDocument(doc.myDO);
        f.setFilecode("");
		f.setIsIndexFile(true);
       // f.save();
        
        return f;
    }
	 
    /** 
     * The getInformationFilesArrayForDocument method performs a database query to
     * return an array of InformationFile that belong to the given document.
     * @param doc, the Information Document
     * @return
     *     array of InformationFile. 
     * @exception PapillonBusinessException
     *   If there is a problem retrieving data.
     */
    public static InformationFile[] getInformationFilesArrayForDocument(InformationDocument doc) 
        throws PapillonBusinessException {
        InformationFile[] theFileArray = null;
        
        try {
            InformationFileQuery query = new InformationFileQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryDocument(doc.myDO);
 
            InformationFileDO[] DOarray = query.getDOArray();
            theFileArray = new InformationFile[ DOarray.length ];
            for ( int i = 0; i < DOarray.length; i++ )
	    	theFileArray[i] = new InformationFile(DOarray[i]);
            return theFileArray;
        } catch(Exception ex) {
            throw new PapillonBusinessException("Exception in getInformationFilesArray()", ex);
        }
         
    }

}

