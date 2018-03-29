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

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.StringWriter;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonImportException;
import fr.imag.clips.papillon.business.PapillonLogger;

import org.apache.xml.serialize.OutputFormat;
import org.apache.xml.serialize.XMLSerializer;
import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList;

public class ImportHTMLFile extends InformationFileAction {

    public final static String FILE_VIEWER_URL="ConsultInformations.po?fileid="; 

 
    // Constructor...
    public ImportHTMLFile() {
        handlerName = "ImportHTMLFile";
    }
    
    /**
     * perform the file addition.
     */
    public void addFile( de.opus5.servlet.UploadedFile file,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang)
            throws PapillonBusinessException ,PapillonImportException{

        PapillonLogger.writeDebugMsg("Importing HTML File \"" + file.getName() + "\"");
        try {
            // Get the data
            InputStream is = file.getInputStream();
            String path = file.getFile().getAbsolutePath();
            
            // First, try to guess the encoding to tell tidy what to use...
            // THOUGHT: Maybe it will be better to convert the stream to UTF8 before parsing it with Tidy
            // This way, we will perhaps accept more encoding (not only Tidy's encodings)...
            int encoding = HTMLParser.guessEncoding(is);
            
            is = file.getInputStream();
            this.addFile(is, doc, path, encoding, env, lang);
            
        } catch (java.io.IOException e) {
            PapillonLogger.writeDebugMsg("IOException encountered while importing HTML file.");
        }
        
        PapillonLogger.writeDebugMsg("File \""+ file.getName() + "\" Imported !");
    }
    
    public void addFile( java.io.File file,
                         InformationDocument doc,
                         RelativeURLMapper env,
			 String lang)
            throws PapillonBusinessException ,PapillonImportException {

	PapillonLogger.writeDebugMsg("Importing HTML File "+ file.getName());

	try {
            // Get the data
            InputStream is = (InputStream) (new FileInputStream(file));
            String path = file.getAbsolutePath();
            
            // First, try to guess the encoding to tell tidy what to use...
            // THOUGHT: Maybe it will be better to convert the stream to UTF8 before parsing it with Tidy
            // This way, we will perhaps accept more encoding (not only Tidy's encodings)...
            int encoding = HTMLParser.guessEncoding(is);
            
            is = (InputStream) (new FileInputStream(file));
            this.addFile(is, doc, path, encoding, env, lang);
            
        } catch (java.io.IOException e) {
            PapillonLogger.writeDebugMsg("IOException encountered while importing HTML file.");
        }
        
        PapillonLogger.writeDebugMsg("File \""+ file.getName() + "\" Imported !");
    } 
    
    /**
     * Return the url where the given file will be found after an import...
     */
    public String getNameOnServer(String filename, String oid) {
        return FILE_VIEWER_URL + oid;
    }

    
    public void addFile( InputStream is,
                         InformationDocument doc,
                         String pathname,
                         int encoding,
                         RelativeURLMapper env,
			 String lang ) throws java.io.IOException, PapillonBusinessException ,PapillonImportException{
                                     

        if (is.available() == 0) {
            PapillonLogger.writeDebugMsg("ImportHTMLFile: Encountered an empty HTML file.");
	    throw new PapillonImportException("an HTML file is empty.");
        } else {
            // Get the oid of the file
            File currentFile = new File(pathname);
            String oid = env.getFileIDForURL(currentFile.getCanonicalPath());
            PapillonLogger.writeDebugMsg(currentFile.getCanonicalPath() + " --> " + oid);
            InformationFile theFileObject;
            if (null != oid && !oid.equals("")) {
                theFileObject=InformationFileFactory.findInformationFileByID(oid);
            } 
			else {
                // check that the document has a title
				if (doc.getTitle().equals("")) {
					throw new PapillonImportException("please, enter a title for this document");
				}
                theFileObject=InformationFileFactory.RegisterNewInformationFile("", "", "", doc);
               oid = theFileObject.getHandle();
                theFileObject.setFilename(this.getNameOnServer(pathname, oid));
                theFileObject.setIsIndexFile(true);
				theFileObject.setLanguage(lang);
                theFileObject.save();
            }
            // Parse it with tidyHtml to accept/correct ill formed documents            
            Document myDoc = HTMLParser.parseHTML(is, encoding);
            
            // Walk the document in order to spot relative links (that will likely 
            // be broken when imported into the web site). If the html file is part of
            // a document, change the links accordingly.
            myDoc = cleanUpHTMLDoc(myDoc, pathname, env);
	    
            String docStr=serializeDoc(myDoc);

            // Then store it into the database
            // WARNING: filename parameter should reflect the name of the file locally on the server, if available...
            theFileObject.setFilecode(docStr);
            theFileObject.save();
        }
    }


    public static Document cleanUpHTMLDoc(Document doc, String pathname, RelativeURLMapper env)  
                                throws java.io.IOException, PapillonBusinessException {
        PapillonLogger.writeDebugMsg("Cleaning up HTML file: " + pathname);

        // First get all the nodes of the document.
        NodeList nl = doc.getElementsByTagName("*");
        
        // Iterate over it and correct the relative URIs
        for (int i=0; i<nl.getLength();i++) {
        
            NamedNodeMap atl=nl.item(i).getAttributes();
            //search for internal links
            for (int j=0; j<atl.getLength();j++)
            { 
                if (atl.item(j).getNodeName().toLowerCase().equals("href") ||
                    atl.item(j).getNodeName().toLowerCase().equals("src") ||
                    atl.item(j).getNodeName().toLowerCase().equals("background")) {
                    String url = atl.item(j).getNodeValue();
                    
                    if (urlHasToBeConverted(url)) {
                        String newUrl = convertUrl(url, pathname, env);
                        
                        atl.item(j).setNodeValue(newUrl);
                    }  
                }
            }
        }
        return doc;
    }
    
    public static boolean urlHasToBeConverted(String url) {
        String upUrl = url.toUpperCase();
        return !( upUrl.startsWith("MAILTO:") ||
                  (upUrl.indexOf("://") != -1) ||
                  upUrl.startsWith("/") ||
                  upUrl.startsWith("#"));
                
    }
    
    /**
    *	Give the url, relative to the root folder of the tar file
    *	@param url is the local url
    *	@param pathname is the absolute pathname of the file containing the url
    *
    *	@return the absolute url 
    */
    public static String toAbsoluteUrl(String url, String pathname) {
        String aurl = "";
        try {
            File f = new File(pathname);
            File parent = f.getParentFile();	// The folder containing the HTML File
            File furl = new File(parent, url);
            
            aurl = furl.getCanonicalPath();
        } catch (java.io.IOException e) {
            PapillonLogger.writeDebugMsg("IOException when retrieving the files canonical path.");
        }
        return aurl;
                
    }
    
    
    /**
    *	Convert the url, relative to the root folder of the information section
    *	@param url is the local url
    *   @param pathname is the path to the HTML file currently analyzed
    *	@param env is the url to oid Mapper.
    *
    *	@return the new url 
    */
    public static String convertUrl(String url, String pathname, RelativeURLMapper env) throws PapillonBusinessException {

	String cleanUrl;
	String anchor;
	// check if the url contains an anchor. If so, get the url without the anchor and the anchor
	int anchorIndex = url.indexOf('#');
	if (anchorIndex != -1) {
	    cleanUrl = url.substring(0, anchorIndex);
	    anchor = url.substring(anchorIndex);
	} else {
	    cleanUrl = url;
	    anchor = "";
	}

        String absoluteTarget = toAbsoluteUrl(cleanUrl, pathname);
        String oid = env.getFileIDForURL(absoluteTarget);
        PapillonLogger.writeDebugMsg("HREF: " + url + " leads to: " + absoluteTarget + " i.e: " + oid);
        
        
        if (null != oid) {
            InformationFile inf = InformationFileFactory.findInformationFileByID(oid);
            return inf.getFilename()+anchor;
        } else {
            return url;
        }
    }


    public static String serializeDoc(Document d) throws java.io.IOException {
        if ((null != d) || (d.getDocumentElement() != null)) {
            StringWriter strw = new StringWriter();
            OutputFormat of=new OutputFormat("text", "UTF-8", true);
            XMLSerializer serial = new XMLSerializer(strw, of);
            serial.serialize(d.getDocumentElement());
            return strw.toString();
        } else {
            return null;
        }
    }
    
}


