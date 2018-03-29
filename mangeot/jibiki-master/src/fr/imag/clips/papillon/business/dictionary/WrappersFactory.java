/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.4  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
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
 *
 *-----------------------------------------------
 *
 */

package fr.imag.clips.papillon.business.dictionary;

// General import

import java.util.TreeSet;

/**
* Used to manage wrappers.
 */
public class WrappersFactory {

    // This function is written ad-hoc but has to be generated from an XML file
    // stored in the database...
    public static Wrapper createXRCEAnalyzerWrapper(String lang) {
        String url = "http://www.xrce.xerox.com/cgi-bin/mltt/demos/";
        Wrapper myWrapper = null;
        String CGI;
        if (lang.equals("deu")) CGI = "german.cgi";
        else if (lang.equals("eng")) CGI = "english.cgi";
        else if (lang.equals("esp")) CGI = "spanish.cgi";
        else if (lang.equals("fra")) CGI = "french.cgi";
        else if (lang.equals("ita")) CGI = "italian.cgi";
        else if (lang.equals("hun")) CGI = "hungarian.cgi";
        else CGI = null;

        if (null != CGI) {
            url = url + CGI;
            myWrapper = new Wrapper();
            myWrapper.setName("XRCE_" + lang);
            myWrapper.setURL(url);
            myWrapper.setMethod("POST");
            myWrapper.setParameters("step=2&txt=");
            myWrapper.setStartMatch("<listing><font size=+1>");
            myWrapper.setEndMatch("</font></listing>");
            myWrapper.setEncoding("ISO-8859-1");
        }
        return myWrapper;
    }

    // This function is written ad-hoc but has to be generated from an XML file
    // stored in the database...
    public static Wrapper createLexicoWrapper() {
        Wrapper myWrapper = new Wrapper();
        myWrapper.setName("Lexico_eng");
        myWrapper.setURL("http://www.dictionary.com/search");
        myWrapper.setMethod("GET");
        myWrapper.setParameters("q=");
        myWrapper.setStartMatch("<!-- Content -->");
        myWrapper.setEndMatch("<!-- End content -->");
        myWrapper.setEncoding("ISO-8859-1");
        return myWrapper;
    }

		public static String[] getXRCEAnalyzerAnswers(String response)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        String[] Array = null;
        String[] ArrayResponse = null;
        int offset;
        Array = response.split("\\s");
        TreeSet mySet = new TreeSet();
        String word;
        
        if (null != Array && Array.length > 0) {
            for (int i=0; i< Array.length; i++) {
                word = Array[i];
                if ((offset = word.indexOf("+")) > -1) {
                    if (offset>0) {
                        word = word.substring(0,offset);
                    }
                    else {
                        word = null;
                    }
                }
                if (null != word && !word.equals("")) {
                    mySet.add(word);
                }
            }
        }
        return (String[])mySet.toArray(new String[0]);
    }
}

