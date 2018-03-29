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
 * Revision 1.4  2006/08/10 16:33:43  mangeot
 * Bug fixes...
 *
 * Revision 1.3  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.informationfile;

import java.io.InputStream;
import java.io.InputStreamReader;

import java.util.regex.*;

import fr.imag.clips.papillon.business.PapillonImportException;
import fr.imag.clips.papillon.business.PapillonLogger;

import org.w3c.dom.Document;
import org.w3c.tidy.Configuration;
import org.w3c.tidy.Tidy;


public class HTMLParser {

    public final static String FILE_VIEWER_URL="ConsultInformations.po?fileid="; 

	protected final static Pattern encodingHeaderRegex = Pattern.compile(".*content=\\\"text/html;\\s*charset=([^\\\"]*)\\\".*",Pattern.CASE_INSENSITIVE);

    // Constructor...
    public HTMLParser() {
	// Nada...
    }
    

    /**
     * Parses an HTML File in an InputStream to a DOM Document
     *
     * @param is the InputStream to parse
     * @param encoding the encoding type to use for the parsing
     * @return the resulting document
     */
    public static Document parseHTML(InputStream is, int encoding)  throws java.io.IOException, PapillonImportException {
        
        Tidy myTidy=new Tidy();
        myTidy.setCharEncoding(encoding);

        myTidy.setMakeClean(true);
        myTidy.setXHTML(true);
        myTidy.setXmlOut(true);
        myTidy.setUpperCaseTags(true);
        myTidy.setTidyMark(false);
        myTidy.setQuiet(true);
        myTidy.setQuoteAmpersand(false);
        myTidy.setQuoteMarks(true);        
        myTidy.setSmartIndent(true);
        myTidy.setIndentContent(true);
        Document myDoc=myTidy.parseDOM(is,null);
        if (myTidy.getParseErrors()>0) {
	    throw new PapillonImportException("there are errors in an HTML file. Correct them before sending the document to the server.");
	}
        return myDoc;

    }
    
    

    /**
     * Tries to guess the encoding of an InputStream
     *
     * @param is the InputStream
     * @return an integer representing the encoding type
     */
    public static int guessEncoding(InputStream is) {
        // returns the encoding of the HTML document
        // Pretend the stream is iso latin 1...
        int res = Configuration.UTF8;
        try {
			java.io.BufferedReader inBuff = new java.io.BufferedReader(new InputStreamReader(is, "ISO-8859-1"));
    
            // First, search for a encoding declaration. If present, it should be something like:
            // <META content="text/html; charset=UTF-8" http-equiv="Content-Type"> 
            // WARNING, will will only search for \"text/html; charset=...\"
			boolean eof = false;
			CharSequence line = (CharSequence) inBuff.readLine();
			Matcher encodingMatcher = encodingHeaderRegex.matcher((CharSequence) inBuff.readLine());

			while (!encodingMatcher.find() && line!=null) {
				line = (CharSequence) inBuff.readLine();
				if (line !=null && !line.equals("")) {
					encodingMatcher = encodingHeaderRegex.matcher(line);
				}
			}
			if (encodingMatcher.find()) {
                String enc = encodingMatcher.group();
                if (enc.equalsIgnoreCase("UTF-8")) {
                    res = Configuration.UTF8;
                } else if (enc.equalsIgnoreCase("ISO-8859-1")) {
                    res = Configuration.LATIN1;
                } else {
                    PapillonLogger.writeDebugMsg("Unrecognized encoding: " + enc);
                }
            }
        } catch (java.io.UnsupportedEncodingException e) {
            PapillonLogger.writeDebugMsg("unsupported encoding.");
        } catch (java.io.IOException e) {
	    PapillonLogger.writeDebugMsg("Error closing InputStreamReader.");
	}
        return res;
    }

}



