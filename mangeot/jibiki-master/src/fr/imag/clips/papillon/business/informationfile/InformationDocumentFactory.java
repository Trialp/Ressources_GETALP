/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.4  2006/08/10 16:32:34  mangeot
 * Added methods in order to add a file from an URL.
 * In is used in conjunction with UploadSimpleFile.java in order to upload correctly documents
 *
 * Revision 1.3  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.2  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.1.1.1.2.1  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * Revision 1.8  2003/08/25 08:27:27  mangeot
 * *** empty log message ***
 *
 * Revision 1.7  2003/06/30 08:22:50  serasset
 * Ajout de sections dans les documents d'informations;
 * Modification de build.xml pour faire la transcription Unicode -> ASCII a la compil.
 *
 * Revision 1.6  2003/03/26 04:53:31  mangeot
 * *** empty log message ***
 *
 * Revision 1.5  2003/03/26 04:20:50  mangeot
 * Added order in informationdocument and xslsheet queries
 *
 * Revision 1.4  2002/12/09 05:29:29  mangeot
 * Bug correction for replacing a document
 *
 * Revision 1.3  2002/12/09 03:35:07  mangeot
 * *** empty log message ***
 *
 * Revision 1.2  2002/11/22 13:04:10  mangeot
 * Nouvelle version Papillon enhydra 5.0
 *
 * Revision 1.1.1.1  2002/10/28 16:49:14  serasset
 * Creation of the papillon CVS repository for enhydra 5.0
 *
 * Revision 1.14  2002/10/25 14:10:30  mangeot
 * merge between PAPILLON_1_4 and trunk
 * CVg: 	papillon-enh-src/docs/Bugs-papillon.rtf
 *
 * Revision 1.13.2.3  2002/10/23 09:51:08  serasset
 * Clean up of the source tree,
 *     Every source file is now encoded in ISO-Latin-1,
 *     Every html file is still encoded in UTF8.
 *
 * Revision 1.13.2.2  2002/10/01 10:44:16  mangeot
 * Modified to import archives with .ds_store in it
 *
 * Revision 1.13.2.1  2002/10/01 05:49:16  mangeot
 * Corrected a few bugs for AdminInformations.java
 *
 * Revision 1.13  2002/09/16 09:55:05  mangeot
 * added a field in informationdocument, the reference,
 * tested and modified olivier's code
 *
 * Revision 1.12  2002/08/19 11:14:12  tache
 * Minor corrections of code presentation (comments, etc).
 *
 * Revision 1.11  2002/08/16 11:46:09  tache
 * Improved errors handle. Now all kind of exceptions thrown during the
 * import are caught.
 *
 * Revision 1.10  2002/08/07 14:09:58  tache
 * The XML information file (info_doc.xml) now can be used for importing
 * monolingual archives and multilingual HTML archives.
 *
 * Revision 1.9  2002/08/01 12:40:00  tache
 * Cleaned up a lot of code in import classes. Moved HTML parsing methods from
 * ImportHTMLFile to new class HTMLParser to reuse them in ImportArchive.
 * Moved XMLTitleParser to fr.imag.clips.papillon.presentation.
 *
 * Revision 1.8  2002/07/26 16:51:11  tache
 * The document repository now manages multilingual documents.
 *
 * Revision 1.7  2002/06/12 06:54:39  mangeot
 * added a message for the registration to the conference
 * and modified AdminInformation to change a doc
 *
 * Revision 1.6  2002/06/04 03:59:39  mangeot
 * bugs corrected for volumeEntriesFactory plus
 * I added the research for title and author in consltInformations
 *
 * Revision 1.5  2002/02/06 15:03:02  serasset
 * Consultation and Administration of the information files now correctly use the Sections.
 * We do not display the files directly anymore, but display a document. This allows the display of
 * images as embedded docs in the default layout.
 *
 * Revision 1.4  2002/02/06 10:39:39  serasset
 * Remove document is now implemented.
 * The information sections are now defined in papillon's config file and handled in the Administration page.
 *
 * Revision 1.3  2001/11/15 15:13:41  serasset
 * Ajout de l'import d'un fichier image/pdf/autre par stockage sur disque dur.
 *
 * Revision 1.2  2001/10/30 15:23:00  serasset
 * Images and other format are now supported as media files (i.e. files that are locally saved in a special directory).
 *
 * Revision 1.1  2001/10/29 13:32:17  serasset
 * Information section is now managed with Documents containing files. We now allow the addition of tar.gz files
 * containing several html files. images are still not supported.
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.informationfile;

import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.CurrentDBTransaction;


// classes for creation of an XML Document
import org.apache.xerces.dom.*;
import org.apache.xml.serialize.XMLSerializer;
import org.apache.xml.serialize.OutputFormat;
import org.w3c.dom.*;
import java.io.StringWriter;

/* for the query sql clauses */
import com.lutris.dods.builder.generator.query.QueryBuilder;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonImportException;

import fr.imag.clips.papillon.business.utility.*;

import com.lutris.appserver.server.sql.ObjectId;
import java.util.Date;

/**
 * Used to find the instances of InformationDocument.
 */
public class InformationDocumentFactory {
    

    /** 
     * The getInformationDocumentArray method performs a database query to
     * return an array of InformationDocument
     * @return
     *     array of InformationDocument. 
     * @exception PapillonBusinessException
     *   If there is a problem retrieving data.
     */
    public static InformationDocument[] getInformationDocumentArray() 
        throws PapillonBusinessException {
        InformationDocument[] theDocArray = null;
        
        try {
            InformationDocumentQuery query = new InformationDocumentQuery(CurrentDBTransaction.get());
	    query.addOrderByAuthor(true);
            InformationDocumentDO[] DOarray = query.getDOArray();
            theDocArray = new InformationDocument[ DOarray.length ];
            for ( int i = 0; i < DOarray.length; i++ )
	    	theDocArray[i] = new InformationDocument(DOarray[i]);
            return theDocArray;

        } catch(Exception ex) {
            throw new PapillonBusinessException("Exception in getInformationDocumentArray()", ex);
        }
         
    }
    
    /** 
     * The getInformationDocumentArrayForSection method performs a database query to
     * return an array of InformationDocument for the given section.
     * @parameter section: the name of the section as in the database.
     * @return
     *     array of InformationDocument. 
     * @exception PapillonBusinessException
     *   If there is a problem retrieving data.
     */
    public static InformationDocument[] getInformationDocumentArrayForSection(String section, String title, String author, String reference) 
        throws PapillonBusinessException {
        InformationDocument[] theDocArray = null;
        
        try {
            InformationDocumentQuery query = new InformationDocumentQuery(CurrentDBTransaction.get());
            query.setQuerySection(section);
	    query.addOrderByAuthor(true);
            String CIC=QueryBuilder.CASE_INSENSITIVE_CONTAINS;
            String E=QueryBuilder.EQUAL;

            if (null != title && !title.equals("")) {
                query.getQueryBuilder().addWhereClause("title", title, CIC);
            }
            if (null != author && !author.equals("")) {
                query.getQueryBuilder().addWhereClause("author", author, CIC);
            }
            if (null != reference && !reference.equals("")) {
                query.getQueryBuilder().addWhereClause("reference", reference, CIC);
            }

            InformationDocumentDO[] DOarray = query.getDOArray();
            theDocArray = new InformationDocument[ DOarray.length ];
            for ( int i = 0; i < DOarray.length; i++ )
	    	theDocArray[i] = new InformationDocument(DOarray[i]);
            return theDocArray;

        } catch(Exception ex) {
            throw new PapillonBusinessException("Exception in getInformationDocumentArray()", ex);
        }
         
    }

    
    /** 
     * The findInformationDocumentByID method performs a database query to
     * return a <CODE>InformationDocument</CODE> object
     * representing the row in the <CODE>informationdocument</CODE> table
     * that matches the object id. 
     *
     * @param id, the object id of the informationdocument table.
     * @return
     *    the InformationDocument. null if there isn't a InformationDocument associated
     *    to the id
     * @exception PapillonBusinessException
     *    if there is a problem retrieving InformationDocument.
     */
    public static InformationDocument findInformationDocumentByID(String id) 
        throws PapillonBusinessException {
        InformationDocument theDoc = null;
        
        try {
            InformationDocumentQuery query = new InformationDocumentQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryOId(new ObjectId(id));
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            InformationDocumentDO theInformationDocDO = query.getNextDO();
            theDoc = new InformationDocument(theInformationDocDO);
            return theDoc;
        } catch(Exception ex) {
            throw new PapillonBusinessException("Exception in findInformationDocumentByID()", ex);
        }
    }
    

    /**
     * Creates a String containing the XML document representing the title
     * for a monolingual document.
     *
     * @param title the title string
     * @param lang the language of the document
     */
    protected static String createTitleDoc(String title, String lang) 
	throws PapillonImportException {
/*	Document titleDoc = new DocumentImpl(null);
	Element titleTag = titleDoc.createElement("title");
	Element locTitle = titleDoc.createElement(lang);
	Text titleTextNode = new TextImpl((DocumentImpl)titleDoc, title);
	locTitle.appendChild(titleTextNode);
	titleTag.appendChild(locTitle);
	titleDoc.appendChild(titleTag);
	StringWriter strw = new StringWriter();
	OutputFormat of=new OutputFormat("text", "UTF-8", true);
	XMLSerializer serial = new XMLSerializer(strw, of);
	try {
	    serial.serialize(titleDoc);
	} catch (java.io.IOException e) {
	    throw new PapillonImportException("Error creating Document's title", e);
	}
	return strw.toString();
*/
// FIXME: MML urgent patch ! Maybe enough ?
	    title = Utility.encodeXMLEntities(title);
        return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><title><" + 
            lang + ">" + title + "</" + lang + "></title>";
    }
    


    protected static void setTitleAndAuthorForDoc(InformationDocument d, String title, String lang, String author) 
	throws PapillonImportException, PapillonBusinessException {

	if (!title.equals("")) {
	    if (!lang.equals("")) {
		String serialTitleDoc = createTitleDoc(title.trim(), lang);
		d.setTitle(serialTitleDoc);
	    } else {		
		throw new PapillonImportException("please, choose a language for this document");
	    }
	    
	} else {
	    d.setTitle("");
	}
        d.setAuthor(author.trim());
    }


    public static InformationDocument newInformationDocument(String title, String author, String owner, String section, String lang, String date, String reference) 
	throws PapillonImportException, PapillonBusinessException {
        InformationDocument d = new InformationDocument();
	d.setSection(section);	
        if (null == date || date.equals("")) {
            d.setCreationDate(new Date());
        }
        else {
            d.setCreationDate(date);
        }
        d.setOwner(owner);
        d.setReference(reference.trim());
	setTitleAndAuthorForDoc(d, title, lang, author);
        d.save();        
	return d;
    }


    public static void addNewInformationDocument(de.opus5.servlet.UploadedFile file, 
                                                 String docTitle,
                                                 String docAuthor,
                                                 String owner,
                                                 String section,
												 String lang,
                                                 String date,
                                                 String reference)
	
	throws PapillonImportException, Exception {
		// We should keep a copy of the added document in order to survive a database crash...                                
		// Should we ?
		
		// Create a new information document. Then, create a file name -> objectId mapper that will be used
		// when adding files to the database (eg. correct relative URLs of HTML docs, etc...)
		InformationDocument doc = null;	
		RelativeURLMapper mapper = new RelativeURLMapper();       	    
		// Then, process the file (with the corresponding file action...)
		InformationFileAction handlerObject = InformationFileActionFactory.getAction(file);
		
		try {
			doc  = newInformationDocument(docTitle, docAuthor, owner, section, lang, date, reference);
			handlerObject.addFile(file, doc, mapper, lang);
		} catch (PapillonImportException pie) {
			PapillonLogger.writeDebugMsg("PapillonImportException Error importing document: "+pie.getMessage());	 
			if (doc != null) {
				deleteInformationDocument(doc.getHandle());
			}
			throw pie;
		}
			catch (PapillonBusinessException pbe) {
				PapillonLogger.writeDebugMsg("PapillonBusinessException Error importing document: "+pbe.getMessage());	 
				if (doc != null) {
					deleteInformationDocument(doc.getHandle());
				}
				throw pbe;
			} 
		}
	
    public static void addNewInformationDocument(String url, 
                                                 String docTitle,
                                                 String docAuthor,
                                                 String owner,
                                                 String section,
												 String lang,
                                                 String date,
                                                 String reference)
	
	throws PapillonImportException, Exception {
		java.net.URI newURI = new java.net.URI(url);
		java.io.File newFile = new java.io.File(newURI);
		addNewInformationDocument(newFile, 
								   docTitle,
								   docAuthor,
								   owner,
								   section,
								   lang,
								   date,
								   reference);
		
	}
	
	public static void addNewInformationDocument(java.io.File file, 
                                                 String docTitle,
                                                 String docAuthor,
                                                 String owner,
                                                 String section,
												 String lang,
                                                 String date,
                                                 String reference)
	
	throws PapillonImportException, PapillonBusinessException, javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
		InformationDocument doc = null;	
		RelativeURLMapper mapper = new RelativeURLMapper();       	    
		// Then, process the file (with the corresponding file action...)
		InformationFileAction handlerObject = InformationFileActionFactory.getAction(file);
		
		try {
			doc  = newInformationDocument(docTitle, docAuthor, owner, section, lang, date, reference);
			handlerObject.addFile(file, doc, mapper, lang);
		} catch (PapillonImportException pie) {
			PapillonLogger.writeDebugMsg("PapillonImportException Error importing document: "+pie.getMessage());	 
			if (doc != null) {
				deleteInformationDocument(doc.getHandle());
			}
			throw pie;
		} 
		catch (PapillonBusinessException pbe) {
			PapillonLogger.writeDebugMsg("PapillonBusinessException Error importing document: "+pbe.getMessage());	 
			if (doc != null) {
				deleteInformationDocument(doc.getHandle());
			}
			throw pbe;
		} 
		catch (javax.xml.parsers.ParserConfigurationException pce) {
			PapillonLogger.writeDebugMsg("ParserConfigurationException Error importing document: "+pce.getMessage());	 
			if (doc != null) {
				deleteInformationDocument(doc.getHandle());
			}
			throw pce;
		} 
		catch (org.xml.sax.SAXException saxe) {
			PapillonLogger.writeDebugMsg("ParserConfigurationException Error importing document: "+saxe.getMessage());	 
			if (doc != null) {
				deleteInformationDocument(doc.getHandle());
			}
			throw saxe;
		} 
    }

	public static void replaceInformationDocument(String id,
                                                  String url,
                                                  String title,
                                                  String author,
												  String lang,
                                                  String date,
                                                  String reference)
	throws Exception, PapillonImportException {
		java.net.URI newURI = new java.net.URI(url);
		java.io.File newFile = new java.io.File(newURI);
		replaceInformationDocument( id,
								    newFile,
								    title,
								    author,
								    lang,
								    date,
									reference);
	}

	public static void replaceInformationDocument(String id,
                                                  java.io.File file,
                                                  String title,
                                                  String author,
												  String lang,
                                                  String date,
                                                  String reference)
	throws Exception, PapillonImportException {
        InformationDocument doc = findInformationDocumentByID(id);
		
        if (null != doc && !doc.isEmpty()) {
			
            // remove files that belong to this document...
            InformationFile[] files = InformationFileFactory.getInformationFilesArrayForDocument(doc);
            String origLang = "";
            for (int i=0; i<files.length; i++) {
                // Here I backup the language of the original file for
                // replacement
                if (i==0) {
                    origLang = files[i].getLanguage();
                }
                files[i].delete();
            }
            // add new files for this document
            RelativeURLMapper mapper = new RelativeURLMapper();
			
            // Then, process the file (with the corresponding file action...)
            InformationFileAction handlerObject = InformationFileActionFactory.getAction(file);
			
            if ((lang == null) || lang.equals("")) {
                lang=origLang;
            }
            if (null != date && !date.equals("")) {
                doc.setCreationDate(date);
            }
            if (null != reference && !reference.equals("")) {
                doc.setReference(reference);
            }           
			try {
                if (title != null && !title.equals("")) {
                    if ((author == null) || author.equals("")) {
                        author=doc.getAuthor();
                    }                    
                    setTitleAndAuthorForDoc(doc, title, lang, author);
                }
                if (file != null) {
                    handlerObject.addFile(file, doc, mapper, lang);
                }
                doc.save();
			} catch (PapillonImportException pie) {
				PapillonLogger.writeDebugMsg("PapillonImportException Error importing document: "+pie.getMessage());
				throw pie;
			}
			catch (PapillonBusinessException pbe) {
				PapillonLogger.writeDebugMsg("PapillonBusinessException Error importing document: "+pbe.getMessage());	 
				if (doc != null) {
					deleteInformationDocument(doc.getHandle());
				}
				throw pbe;
			}
		}
    }
	
		
	
    public static void replaceInformationDocument(String id,
                                                  de.opus5.servlet.UploadedFile file,
                                                  String title,
                                                  String author,
						  String lang,
                                                  String date,
                                                  String reference)
	throws Exception, PapillonImportException {
        InformationDocument doc = findInformationDocumentByID(id);
            
        if (null != doc && !doc.isEmpty()) {

            // remove files that belong to this document...
            InformationFile[] files = InformationFileFactory.getInformationFilesArrayForDocument(doc);
            String origLang = "";
            for (int i=0; i<files.length; i++) {
                // Here I backup the language of the original file for
                // replacement
                if (i==0) {
                    origLang = files[i].getLanguage();
                }
                files[i].delete();
            }
            // add new files for this document
            RelativeURLMapper mapper = new RelativeURLMapper();

            // Then, process the file (with the corresponding file action...)
            InformationFileAction handlerObject = InformationFileActionFactory.getAction(file);

            if ((lang == null) || lang.equals("")) {
                lang=origLang;
            }
            if (null != date && !date.equals("")) {
                doc.setCreationDate(date);
            }
            if (null != reference && !reference.equals("")) {
                doc.setReference(reference);
            }           
	    try {
                if (title != null && !title.equals("")) {
                    if ((author == null) || author.equals("")) {
                        author=doc.getAuthor();
                    }                    
                    setTitleAndAuthorForDoc(doc, title, lang, author);
                }
                if (file != null) {
                    handlerObject.addFile(file, doc, mapper, lang);
                }
                doc.save();
	    } catch (PapillonImportException pie) {
		PapillonLogger.writeDebugMsg("PapillonImportException Error importing document: "+pie.getMessage());
		throw pie;
		}
		catch (PapillonBusinessException pbe) {
			PapillonLogger.writeDebugMsg("PapillonBusinessException Error importing document: "+pbe.getMessage());	 
			if (doc != null) {
				deleteInformationDocument(doc.getHandle());
			}
			throw pbe;
		} 
		
            // Modify and save the document itself
	    /* if (null != title && !title.equals("")) {
                doc.setTitle(title);
            }
            if (null != author && !author.equals("")) {
                doc.setAuthor(author);
            }
            doc.save();*/
        }
    }

   
    public static void deleteInformationDocument(String id)
    throws PapillonBusinessException {
        InformationDocument doc = findInformationDocumentByID(id);
        
        // remove files that belong to this document...
        InformationFile[] files = InformationFileFactory.getInformationFilesArrayForDocument(doc);
        for (int i=0; i<files.length; i++) {
            files[i].delete();
        }
        // then remove the document.
        doc.delete();
    }
}

