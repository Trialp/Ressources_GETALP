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
import fr.imag.clips.papillon.business.PapillonImportException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.informationfile.InformationSections;

// classes for creation of an XML Document
import org.apache.xerces.dom.*;
import org.w3c.dom.*;

//import org.apache.xerces.parsers;
import javax.xml.parsers.*;
import org.xml.sax.InputSource;
import org.apache.xml.serialize.XMLSerializer;
import org.apache.xml.serialize.OutputFormat;

import java.io.*;
import java.util.Vector;
import java.util.Iterator;

// For languages
import fr.imag.clips.papillon.business.locales.Languages;


/** 
 * The class in charge of importing archive files. ImportZipFile and ImportTarFile extend it.
 */
public abstract class ImportArchive extends InformationFileAction {

    protected final static String TarHandler = "ImportTarFile";
    protected final static String ZipHandler = "ImportZipFile";
    protected final static String GZipHandler = "ImportGZippedFile";

    protected final static String infoFileName = "info_doc.xml";
    //protected final static String XML_SCHEMA_LOCATION = "/home/tache/"+infoFileName;

    protected final static String DOCUMENT_ELEMENT = "document";
    protected final static String TITLE_ELEMENT = "title";
    protected final static String SECTION_ATTRIBUTE = "section";
    protected final static String AUTHORS_ELEMENT = "authors";
    protected final static String DATE_ELEMENT = "date";
    protected final static String REFERENCE_ELEMENT = "reference";

    
    // Constructor...
    public ImportArchive() {
    }
    

    /** 
     * Guess if the archive is a single compressed directory or not.
     *
     *@param dir the location of the archive.
     *@return the name of the eventual single directory or "" if it doesn't exist.
     */ 
    protected String getRoot(File dir) {
	String root;
	File[] dirContent = dir.listFiles();
	if ((dirContent.length == 1) && dirContent[0].isDirectory()) {
	    //the archive is a compressed directory
	    String dirName = dirContent[0].toString();
	    int lastSlash = dirName.lastIndexOf('/');
	    root = dirName.substring(lastSlash+1);
	} else {
	    root = "";
	}
	return root;
    }


    /**
        * Guess if an HTML file is an index, based on its name.
     * It is regarded as an index file if its name is
     * index.htm(l) or default.htm(l)
     *
     * @param n the file name.
     * @return true if the file name indicates that this is an index file, false otherwise.
     */
    protected boolean isIndexName(String n) {
        return (  n.endsWith("index.html")
                  || n.endsWith("index.htm")
                  || n.endsWith("default.html")
                  || n.endsWith("default.htm"));
    }


    /**
     * Guess if a file is a hidden one (name beginning with a '.'
     * @param n the file name.
     * @return true if the file name indicates that this is an index file, false otherwise.
     */
    protected boolean isHiddenName(String n) {
        boolean result = false;
        int lastSlash = n.lastIndexOf('/');
        if (lastSlash >= 0 && n.length() > 0) {
            n = n.substring(lastSlash+1);
        }
        result = (n.charAt(0) == '.');
        return result;
    }


    
    /**
     * List all directories in the given directory
     * @param dir the directory.
     * @return a Vector of Strings containing the name of directories found in dir.
    */
    protected Vector getDirs(File dir) {
	Vector foundDirs = new Vector();

	File[] dirContent = dir.listFiles();
	for (int i=0; i < dirContent.length; i++) {
	    if (dirContent[i].isDirectory()) {
		foundDirs.add(dirContent[i].toString());
	    }
	}
	return foundDirs;
    }

    /**
     * List all files (and not directories) in the given directory
     * @param dir the directory.
     * @return a Vector of Strings containing the name of files found in dir.
    */   
    protected Vector getFiles(File dir) {
	Vector foundFiles = new Vector();
	
	File[] dirContent = dir.listFiles();

	for (int i=0; i < dirContent.length; i++) {
	    if (!dirContent[i].isDirectory()) {
		foundFiles.add(dirContent[i].toString());

	    }
	}
	return foundFiles;
    }    



    /**
     * Finds the value of <title> tag in the given HTML file
     * @param f the file to search the title in
     * @param dir the name of directory where f is placed.
     * @return the title string
     */
    protected String getTitle(File f, String dir) 
	throws PapillonImportException, java.io.IOException, java.io.FileNotFoundException {
	// Parse the HTML to find the title
	InputStream is = (InputStream)new FileInputStream(f);
	int encoding = HTMLParser.guessEncoding(is);
	
	// re-open InputStream
	is = (InputStream)new FileInputStream(f);
	Document htmlDoc = null;
	// parse the file and throw an exception if Tidy encountered errors (not warnings)
	try {
	    htmlDoc = HTMLParser.parseHTML(is, encoding);
	} catch (PapillonImportException pie) {
	    throw new PapillonImportException("there are errors in "+dir+"/"+f.getName()+". Correct them before sending the document to the server.");
	}

	NodeList titleList = htmlDoc.getElementsByTagName("title");
	String title = null;
	
	if (titleList.getLength() > 0) {
	    title = titleList.item(0).getFirstChild().getNodeValue();
	} else {
	    title = "";
	}

	if (title.equals("")) {
	    throw new PapillonImportException("the index file " +f.toString()+ " has no valid title. Please, check its <title> tag.");
	}
	 
	return title;
    }



    /**
     * Iterate over archive files to add them to the database
     * @param files a Vector of the archive files names.
     * @param outputDir the path where archive files are temporary saved.
     * @param doc the current information document.
     * @param env the RelativeURLMapper.
     * @throws PapillonImportException if the archive doesn't have a correct format.
    */
    protected void performAddition(Vector files, File outputDir, InformationDocument doc, RelativeURLMapper env)
	throws java.io.IOException, PapillonBusinessException, PapillonImportException,
	       javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
	
	
	Iterator iter = files.iterator();
	while (iter.hasNext()) {
	    String fname = (String) iter.next();

	    // Get the appropriate action on the file...
	    File f = new File(outputDir, fname);
	    
	    InformationFileAction handlerObject = InformationFileActionFactory.getAction(f);
	    String handlerName = handlerObject.getHandlerName();
	    if (handlerName.equals(TarHandler) || handlerName.equals(ZipHandler) || handlerName.equals(GZipHandler)) {
		// this file is an embedded archive, we don't uncompress it
		// it is considered as a Media file
		try {
		    handlerObject = 
			(InformationFileAction)Class.forName("fr.imag.clips.papillon.business.informationfile.ImportMediaFile").newInstance();
		} catch (Exception e) {
		    throw new PapillonBusinessException("Information File handler not found", e);
		}
	    }
	    
	    handlerObject.addFile(f, doc, env, "");
	    
	    f.deleteOnExit();
	}       
    }


    /**
     * Updates an InformationDocument with informations contained in the given Xml information file.
     * If the information file is for a monolingual document, returns its language, otherwise return null.
     *
     * @param theInfoFileName the pathname of the XML information file.
     * @param doc the document to update.
     * @param archiveRootDirs the archive root directories
     * @param outputDirName the pathname to the archive root
     * @param langCodes all known language codes in papillon
     * @param monolingual indicates if the document is monolingual or not
     */
    protected String updateInfoDoc(String theInfoFileName,
                                   InformationDocument doc,
                                   Vector archiveRootDirs,
                                   String outputDirName,
				   Vector langCodes,
                                   boolean monolingual)
	throws java.io.IOException,
        PapillonBusinessException,
        PapillonImportException,
        org.xml.sax.SAXException {
	
	String res = null;
	Vector xmlTitleLangs = null;
	
	File theInfoFile = new File(theInfoFileName);
	theInfoFile.deleteOnExit();

	/******* Parse the Xml information file *******/
        try {
            // buiding tools to serialize the XML
            OutputFormat of=new OutputFormat("text", "UTF-8", true);

            // buiding tools to parse the XML
            DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
            dbf.setValidating(true);
            DocumentBuilder docBuilder = dbf.newDocumentBuilder();
            Document docXml=docBuilder.parse(theInfoFile);
	// we should check that info_doc.xml is an instance of info_doc.xsd...


	/****************************************************************  
	 *******  get the title Node and store it in the database *******
	 ****************************************************************/

	NodeList titleList = docXml.getElementsByTagName(TITLE_ELEMENT);
	if (titleList.getLength() != 1) {
	    throw new PapillonImportException("<title> tag badly defined or not defined in " + infoFileName + ".");
        }

	// Build language list
	Element title = (Element)titleList.item(0);
	xmlTitleLangs = new Vector();
	NodeList langNodesList = title.getChildNodes();

	for (int i = 0; i < langNodesList.getLength(); i++){
	    String langName = langNodesList.item(i).getNodeName();
	    if (!langName.equals("#text")) {
		
		// check that the language code is known
		if (!langCodes.contains(langName)) {
		    throw new PapillonImportException("the language code "+langName+" is unknown.");
		}
		
		// check that the title is not empty		    
		Node locTitleTag = langNodesList.item(i).getFirstChild();
		if (locTitleTag == null) {
		    throw new PapillonImportException("the title for language "+langName+" is not defined in " + infoFileName + ".");
		}
		
		// no errors : add the language to the list
		xmlTitleLangs.add(outputDirName+'/'+langName);
	    }
	}
	
	
	// check that the language declaration is correct & get the result string if the doc is monolingual
	if (monolingual) {
	    if (xmlTitleLangs.size()!=1) {
		throw new PapillonImportException("more than one language is declared in info_doc.xml while the document is monolingual.");
	    } 
	    else {
		// the language name is the second node of langNodesList because the first one is "#text"
		res = langNodesList.item(1).getNodeName();
	    }
	} else {
	    // multilingual document : check that the archive directories names matches the language list
	    if (xmlTitleLangs.size() != archiveRootDirs.size() || !archiveRootDirs.containsAll(xmlTitleLangs))
		throw new PapillonImportException("the languages declaration in " + infoFileName +" doesn't match the directories names.");
	}



	/*******************************
	 ******* Get the section *******
	 *******************************/
        NodeList documentList = docXml.getElementsByTagName(DOCUMENT_ELEMENT);
        String theSection = "";
        if (null != documentList && documentList.getLength() >0) {
            theSection = ((Element)documentList.item(0)).getAttribute(SECTION_ATTRIBUTE);
        }
	if (!theSection.equals("")) {
	    // check that the section is correct
	    // this will be useless when we are able to validate info_doc.xml against info_doc.xsd
	    InformationSections IS = new InformationSections();
	    String[] sections = IS.getInformationSectionArray();
	    boolean sectionFound = false;
	    int i = 0;
	    while (i < sections.length && !sectionFound) {
		if (sections[i].equals(theSection))
		    sectionFound=true;
		i++;
	    }
	    if (!sectionFound) {
		throw new PapillonImportException("The "+ SECTION_ATTRIBUTE + " attribute of <" + DOCUMENT_ELEMENT + "> element has an incorrect value.");
	    }
	    doc.setSection(theSection);
	} else {
	    throw new PapillonImportException("The "+ SECTION_ATTRIBUTE + " attribute of <" + DOCUMENT_ELEMENT + "> element is missing in " + infoFileName + ".");
	}

        /******************************
            ******* Get the author *******
            ******************************/
        NodeList authorList = docXml.getElementsByTagName(AUTHORS_ELEMENT);
        if (null != authorList && authorList.getLength() >0) {
            Element authorElement = (Element)authorList.item(0);

            //FIXME: The parser does not verify if the info_doc is correctly encoded. I don't know why.
            // Thus I verify here with the serializer and parser:
            StringWriter myStringWriter = new StringWriter();
            XMLSerializer serial = new XMLSerializer(myStringWriter, of);
            serial.serialize(authorElement);
            String authorXML = myStringWriter.toString();
            PapillonLogger.writeDebugMsg("author String: " + authorXML);
            docBuilder.parse(new InputSource(new StringReader(authorXML)));

            String author = authorElement.getFirstChild().getNodeValue();
            doc.setAuthor(author);
        }

        /******************************
            ******* Get the date *******
            ******************************/
        NodeList dateList = docXml.getElementsByTagName(DATE_ELEMENT);
        if (null != dateList && dateList.getLength() >0) {
            doc.setCreationDate(dateList.item(0).getFirstChild().getNodeValue());
        }

        /******************************
            ******* Get the reference *******
            ******************************/
        NodeList refList = docXml.getElementsByTagName(REFERENCE_ELEMENT);
        if (null != refList && refList.getLength() >0) {
            Element refElement = (Element)refList.item(0);
            //FIXME: The parser does not verify if the info_doc is correctly encoded. I don't know why.
            // Thus I verify here with the serializer and parser:
            StringWriter myStringWriter = new StringWriter();
            XMLSerializer serial = new XMLSerializer(myStringWriter, of);
            serial.serialize(refElement);
            String refString = myStringWriter.toString();
            PapillonLogger.writeDebugMsg("ref String: " + refString);
            docBuilder.parse(new InputSource(new StringReader(refString)));

            String ref = refElement.getFirstChild().getNodeValue();
            doc.setReference(ref);
        }
        
        
	/************************************** 
	 ******* Serialize the Document *******
	 **************************************/
        StringWriter myStringWriter = new StringWriter();
        XMLSerializer serial = new XMLSerializer(myStringWriter, of);
	serial.serialize(title);
        String titleString = myStringWriter.toString();

        //FIXME: The parser does not verify if the info_doc is correctly encoded. I don't know why.
        // Thus I verify here with the serializer and parser:
        PapillonLogger.writeDebugMsg("title String: " + titleString);
        docBuilder.parse(new InputSource(new StringReader(titleString)));
        
        doc.setTitle(titleString);
        } catch(Exception e) {
            throw new PapillonImportException("Error parsing document " + infoFileName, e);
        }
       	PapillonLogger.writeDebugMsg(infoFileName + " parsed!");
        doc.save();        

	return res;
    }


    /**
     * Sets isindex and lang fields for the information file corresponding to the given file.
     *
     * @param fname the pathname to the file
     * @param env the current URL Mapper
     * @param lang the language of the file
     */
    protected void setIndexAndLangForFile(String fname, RelativeURLMapper env, String lang) 
	throws PapillonBusinessException{
	String oid = env.getFileIDForURL(fname);
	InformationFile inf = InformationFileFactory.findInformationFileByID(oid);
	inf.setIsIndexFile(true);
	inf.setLanguage(lang);
	inf.save();
    }


    /**
     * Finds index files for a multilingual non-html document and sets their language.
     * Parses the given XML file to extract the section, the author name
     * and the titles in the document languages.
     *
     * @param theInfoFileName the path of the XML information file to parse.
     * @param archiveRootDirs the Vector of the archive root directories.
     * @param doc the current InformationDocument.
     * @param env the current URL Mapper.
     * @throws PapillonImportException if the archive doesn't have a correct format.
     */
    protected void findInfoForMultilingualMediaDoc(String theInfoFileName,
                                                   Vector archiveRootDirs,
                                                   InformationDocument doc,
                                                   RelativeURLMapper env)
        throws java.io.IOException,
        PapillonBusinessException,
        PapillonImportException,
        javax.xml.parsers.ParserConfigurationException,
        org.xml.sax.SAXException {
	

            Iterator iterDirs = archiveRootDirs.iterator();
            int lastSlash;
        
	while (iterDirs.hasNext()) {
            File dir = new File((String)iterDirs.next());
	    File[] dirFiles = dir.listFiles();
	    lastSlash = dir.toString().lastIndexOf('/');
	    String lang = dir.toString().substring(lastSlash+1);
            boolean indexFound = false;
            
            for (int i=0; i< dirFiles.length; i++) {

                // I should perform a better test. Here I discard only fils like .DS_store ...
                File file = dirFiles[i];
                if (!isHiddenName(file.toString().toLowerCase())) {
                    if (!indexFound) {
                            indexFound=true;
                            setIndexAndLangForFile(file.getCanonicalPath(), env, lang);
                        }
              //      else {
             //           throw new PapillonImportException("the /"+lang+" directory must contain exactly one file.");
             //       }
                    }
                    else {
                        file.deleteOnExit();
                    }
                }
            }
        }
	

     /**
     * Finds index files for a multilingual HTML document (with no XML information file)
     * and sets the correct language for each of them
     * @param archiveRootDirs the Vector of the archive root directories.
     * @param doc the current InformationDocument
     * @param env the current URL Mapper.
     * @throws PapillonImportException if the archive doesn't have a correct format.
     */
    protected void findInfoForMultilingualHTMLDoc(Vector archiveRootDirs, InformationDocument doc, RelativeURLMapper env)
	throws java.io.IOException, PapillonBusinessException, PapillonImportException {

	Iterator iterDirs = archiveRootDirs.iterator();
	boolean indexFound;
	int lastSlash;

	// create the DOM Document representing the document titles
	Document titleDoc = new DocumentImpl(null);
	Element titleTag = titleDoc.createElement("title");	

	while (iterDirs.hasNext()) {
	    File dir = new File((String)iterDirs.next());
	    File[] dirFiles = dir.listFiles();
	    lastSlash = dir.toString().lastIndexOf('/');
	    String lang = dir.toString().substring(lastSlash+1);
	    indexFound = false;

	    for (int i = 0; i < dirFiles.length; i++) {
		if (!dirFiles[i].isDirectory() && isIndexName(dirFiles[i].toString().toLowerCase())) {
		    if (!indexFound) {
			indexFound=true;
			
			String title = getTitle(dirFiles[i], lang);

			// add the new title to the DOM Document
			Element locTitle = titleDoc.createElement(lang);
			Text titleTextNode = new TextImpl((DocumentImpl)titleDoc, title);
			locTitle.appendChild(titleTextNode);
			titleTag.appendChild(locTitle);			
			// save language and set isIndexFile to true for the current file
			setIndexAndLangForFile(dirFiles[i].getCanonicalPath(), env, lang);
		    } else 
			    throw new PapillonImportException("too much index files found in /"+lang+" directory.");
		    
		}
	    }	    
		       
	    if (!indexFound)
		throw new PapillonImportException("no index file found in /"+lang+" directory.");
	    dir.deleteOnExit();
	}
	titleDoc.appendChild(titleTag);

	// serialize the doc
	StringWriter strw = new StringWriter();
	OutputFormat of=new OutputFormat("text", "UTF-8", true);
	XMLSerializer serial = new XMLSerializer(strw, of);
	serial.serialize(titleDoc);
        
	doc.setTitle(strw.toString());
	doc.save();

    }

     /**
     * Finds informations for a multilingual HTML document (with an XML information file)
     * Finds index files and sets the correct language for each of them
     * @param archiveRootDirs the Vector of the archive root directories.
     * @param doc the current InformationDocument
     * @param env the current URL Mapper.
     * @throws PapillonImportException if the archive doesn't have a correct format.
     */
    protected void findInfoForMultilingualHTMLDoc(String theInfoFileName, String outputDirName, Vector archiveRootDirs, 
						  InformationDocument doc, RelativeURLMapper env)
	throws java.io.IOException, PapillonBusinessException, PapillonImportException,
	       javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
	
	Iterator iterDirs = archiveRootDirs.iterator();
	boolean indexFound;
	int lastSlash;

	
	while (iterDirs.hasNext()) {
	    File dir = new File((String)iterDirs.next());
	    File[] dirFiles = dir.listFiles();
	    lastSlash = dir.toString().lastIndexOf('/');
	    String lang = dir.toString().substring(lastSlash+1);
	    indexFound = false;
	    
	    for (int i = 0; i < dirFiles.length; i++) {
		if (!dirFiles[i].isDirectory() && isIndexName(dirFiles[i].toString().toLowerCase())) {
		    if (!indexFound) {
			indexFound=true;
			
			// save language and set isIndexFile to true for the current file
			setIndexAndLangForFile(dirFiles[i].getCanonicalPath(), env, lang);
		    } else
			throw new PapillonImportException("too much index files found in /"+lang+" directory.");		    
		}
	    }	    
	    
	    if (!indexFound)
		throw new PapillonImportException("no index file found in /"+lang+" directory.");
	    dir.deleteOnExit();
	}
	
    }
    

    /**
     * Finds the index file for a monolingual document
     * and sets the correct language for it.
     * @param archiveRootDirs the Vector of the archive root directories.
     * @param doc the current InformationDocument.
     * @param env the current URL Mapper.
     * @param lang the language of the document.
     * @throws PapillonImportException if the archive doesn't have a correct format.
     */
    protected void findIndexForMonolingualDoc(Vector archiveRootFiles, InformationDocument doc, RelativeURLMapper env, String lang)
	throws java.io.IOException, PapillonBusinessException, PapillonImportException {

	if (doc.getTitle().equals(""))
	    throw new PapillonImportException("please, type in a title for this document.");
      
	if (lang.equals(""))
	    throw new PapillonImportException("please, choose a language for this document.");

	if (archiveRootFiles.size() == 1) {
	    File file = new File((String)archiveRootFiles.get(0));
	    setIndexAndLangForFile(file.getCanonicalPath(), env, lang);
	} else {

            Iterator iterFiles = archiveRootFiles.iterator();
            boolean indexFound=false;

            while (iterFiles.hasNext()) {
                File file = new File((String)iterFiles.next());
                if (isIndexName(file.toString().toLowerCase())) {
                    if (!indexFound) {
                        indexFound=true;
                        setIndexAndLangForFile(file.getCanonicalPath(), env, lang);
                    } else
                        throw new PapillonImportException("too much index files found.");
                }
            }

            // code added for monolingual non HTML archives ! Not well tested.
            if (!indexFound) {
                iterFiles = archiveRootFiles.iterator();
                while (iterFiles.hasNext()) {
                File file = new File((String)iterFiles.next());

                // I should perform a better test. Here I discard only fils like .DS_store ...
                if (!isHiddenName(file.toString().toLowerCase())) {
                    if (!indexFound) {
                        indexFound=true;
                        setIndexAndLangForFile(file.getCanonicalPath(), env, lang);
                    }
                }
                else {
                    file.deleteOnExit();
                }                
            }
            }
            if (!indexFound)
		throw new PapillonImportException("no index file found.");
	}
    }



    /**
     * Guess if the multilingual Doc is HTML or not and take the appropriate action
     *
     * @param theInfoFileName the pathname of the XML information file
     * @param outputDirName the path of the archive root in the tmp directory
     * @param archiveRootDirs the Vector of the archive root directories
     * @param doc the current InformationDocument
     * @param env the current URL Mapper
     */
    protected void findInfoForMultilingualDoc(String theInfoFileName, String outputDirName, 
					      Vector archiveRootDirs, InformationDocument doc,
					      RelativeURLMapper env)
	throws java.io.IOException, PapillonBusinessException, PapillonImportException,
	       javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
	
	boolean multiHTML = false;
	int i = 0;
	while (i < archiveRootDirs.size() && !multiHTML) {
	    File dir = new File((String)archiveRootDirs.get(i));
	    dir.deleteOnExit();
	    File[] listDir = dir.listFiles();
            int j=0;
            if (listDir != null && listDir.length>0) {
                while (j<listDir.length && !multiHTML) {
                    String fname = listDir[j].toString();
                    if (fname.endsWith(".html") || fname.endsWith(".htm")) {
                        // there's only one HTML file => HTML archive
                        multiHTML = true;
                    }
                    j++;
                }
            }
	    else {
		// Error: a directory is empty while it should contain an index file
		int lastSlash = dir.toString().lastIndexOf('/');
		String dirName = dir.toString().substring(lastSlash+1);
		throw new PapillonImportException("the "+dirName+"/ directory is empty");
	    }
	    i++;
	}
	if (multiHTML) {
	    PapillonLogger.writeDebugMsg("Importing multilingual HTML document");
	    findInfoForMultilingualHTMLDoc(theInfoFileName, outputDirName,  archiveRootDirs, doc, env);
	} else {
	    PapillonLogger.writeDebugMsg("Importing multilingual media document");
	    findInfoForMultilingualMediaDoc(theInfoFileName, archiveRootDirs, doc, env);
	}
	
    }




    /**
     * Guess the type of the document (monolingual, multilingual HTML, multilingual non-HTML)
     * and adds files according to this type.
     * @param files the vector of files to add.
     * @param outputDir the pathname to files to add.
     * @param doc the current InformationDocument.
     * @param env the current URL Mapper.
     * @param lang the language of the document, if it is monolingual.
     * @throws PapillonImportException if the archive doesn't have a correct format.
     */
    protected void addArchiveFiles(Vector files, File outputDir, InformationDocument doc, RelativeURLMapper env, String lang)
	throws java.io.IOException, PapillonImportException, PapillonBusinessException, 
	       javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException {
		
	// Check if the archive is a single compressed directory. In that case, root is the name of this directory
	String root = getRoot(outputDir);

	// get the real root of the archive, different from ouputdir.toString() if the archive contains a single directory.
	String outputDirName;	
	if (!root.equals(""))
	    outputDirName = outputDir.toString()+"/"+root;
	else
	    outputDirName = outputDir.toString();

	File archiveRoot = new File(outputDirName);
	archiveRoot.deleteOnExit();

	/**** look for directories bearing the name of a language  ****/

	int dirsFound = 0;
	String[] langCodedNamesArray = Languages.getArray();
	
	// transform the languages array into a Vector
	Vector langCodedNames = new Vector();
	for (int i = 0; i < langCodedNamesArray.length; i++) {
	    langCodedNames.add(langCodedNamesArray[i]);
	}

	Vector archiveRootDirs = getDirs(archiveRoot);

	for (int i = 0; i < archiveRootDirs.size(); i++) {
	    int lastSlash = ((String)archiveRootDirs.get(i)).lastIndexOf('/');
	    if (langCodedNames.contains(((String)archiveRootDirs.get(i)).substring(lastSlash+1))) {
		dirsFound++;
	    }
	}

 
	Vector archiveRootFiles = getFiles(archiveRoot);
	String theInfoFileName = outputDirName+'/'+infoFileName;

	/******* Guess the archive type and take the appropriate action *******/

	if (dirsFound == 0 || dirsFound != archiveRootDirs.size() ) {
            // MML: I do not check any more if there are other files than info_doc.xml
            // because otherwise, archives with dummy files like .DS_store are considered
            // as monolingual
            // || archiveRootFiles.size()>1) {

            //there's more than one file on the root or not all directories bear a language name
	    // => monolingual archive
	    PapillonLogger.writeDebugMsg("Importing monolingual document");
	    String realLang;
	    if (archiveRootFiles.contains(theInfoFileName)) {	      
		archiveRootFiles.remove(theInfoFileName);
		files.remove(infoFileName);
		realLang = updateInfoDoc(theInfoFileName, doc, archiveRootDirs, outputDirName, langCodedNames, true);
	    } else {
		realLang = lang;
	    }

	    findIndexForMonolingualDoc(archiveRootFiles, doc, env, realLang);

	} else {
	    // multilingual document except if the single root file is not the XML information file
     // MML here also I modify the code because otherwise, archives with
     // dummy files like .DS_store are considered as monolingual

            if (archiveRootFiles.contains(theInfoFileName)) {
            // remove info_doc.xml from files list: it is useless to store it in the database
                files.remove(infoFileName);
                updateInfoDoc(theInfoFileName, doc, archiveRootDirs, outputDirName, langCodedNames, false);
                findInfoForMultilingualDoc(theInfoFileName, outputDirName,  archiveRootDirs, doc, env);

            } else {
            // there's no file in the archive root => multilingual archive without info_doc.xml
                PapillonLogger.writeDebugMsg("Importing multilingual HTML document");
                findInfoForMultilingualHTMLDoc(archiveRootDirs, doc, env);
            }

            
     //       if (archiveRootFiles.size() == 1) {
//		if (archiveRootFiles.contains(theInfoFileName)) {
		    // remove info_doc.xml from files list: it is useless to store it in the database
//		    files.remove(infoFileName);
	//	    updateInfoDoc(theInfoFileName, doc, archiveRootDirs, outputDirName, langCodedNames, false);
	//	    findInfoForMultilingualDoc(theInfoFileName, outputDirName,  archiveRootDirs, doc, env);

	//	} else { 
		    // the single file on the root is not info_doc.xml => monolingual archive
	//	    PapillonLogger.writeDebugMsg("Importing monolingual document");
	//	    findIndexForMonolingualDoc(archiveRootFiles, doc, env, lang);
	//	}

	  //  } else { 
		// there's no file in the archive root => multilingual archive without info_doc.xml
	//	PapillonLogger.writeDebugMsg("Importing multilingual HTML document");
	//	findInfoForMultilingualHTMLDoc(archiveRootDirs, doc, env);
	//    }
	}
	performAddition(files, outputDir, doc, env);
    }
}


