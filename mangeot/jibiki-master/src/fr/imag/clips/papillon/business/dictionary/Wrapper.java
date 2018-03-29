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
 * Revision 1.4  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.3  2005/01/23 17:21:16  mangeot
 * *** empty log message ***
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
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


import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.xml.XMLServices;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.tidy.Configuration;
import org.w3c.tidy.Tidy;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;


/**
 * Represents a Volume. 
 */
public class Wrapper {

    protected String name = "";
    protected String url = "";
    protected String parameters = "";
    protected String method = "GET";
    protected String encoding = "UTF-8";
    protected String preStartMatch = "";
    protected String startMatch = "";
    protected String endMatch = "";
            
    
    // constructor
   public Wrapper() {
   }

   public void setURL(String newURL) {
       url = newURL;
   }

   public void setName(String newName) {
       name = newName;
   }

   public void setEncoding(String newEncoding) {
       encoding = newEncoding;
   }

   public void setParameters(String params) {
        parameters = params;
    }

   public void setMethod(String newMethod) {
       method = newMethod;
   }

   public void setPreStartMatch(String match) {
       preStartMatch = match;
   }

   public void setStartMatch(String match) {
       startMatch = match;
   }

   public void setEndMatch(String match) {
       endMatch = match;
   }
   public String getEncoding() {
       return encoding;
   }
   public String getName() {
       return name;
   }
   
   public Element getResultElement (String[] Headwords)
       throws PapillonBusinessException, java.io.IOException {
           String resString = "<div>" + getResultString(Headwords) + "</div>";
           Tidy tidyParser = new Tidy();
           tidyParser.setQuiet(true);
           tidyParser.setShowWarnings(false);
           tidyParser.setOnlyErrors(true);
           //   tidyParser.setErrout(new NullPrintWriter());
           // tidyParser.setErrout(null);
           tidyParser.setXmlOut(true);
           // tidyParser.setXHTML(true);
           tidyParser.setCharEncoding(Configuration.UTF8);
           // tidyParser.setXmlTags(true);
           Document docDOM = tidyParser.parseDOM(new ByteArrayInputStream(resString.getBytes()), null);
           // It is dangerous but this config of tidy produces an html document
           Element elt = docDOM.getDocumentElement();
           elt = (Element)elt.getElementsByTagName("body").item(0).getFirstChild();
           return elt;
       }


   /* FIXME: this method has two big problems:
       1) It exists because I don't know how to insert the result of tidy directly into another DOM document for display
       2) I have to cut the XML header "by hands" and it is not nice ! */
   public String getResultXmlCode (String[] Headwords)
       throws PapillonBusinessException, java.io.IOException {
           Element myElement = getResultElement(Headwords);
           String xmlCode =  XMLServices.NodeToString(myElement);
		   xmlCode = Utility.trimXMLHeader(xmlCode);
           return xmlCode;
       }

   
   public String getResultString (String[] Headwords) throws PapillonBusinessException {
     return getResultString(concatenateHeadwords(Headwords));
   }

   protected String concatenateHeadwords(String[] Headwords) {
       String headwords = null;
       if (null != Headwords && Headwords.length > 0) {
           headwords = Headwords[0];
           int length = Headwords.length;
           if (length > 1) {
               for (int i=1;i<length;i++) {
                   headwords = headwords + "+" + Headwords[i];
               }
           }
       }
       return headwords;
   }
   
       public String getResultString (String headwords) throws PapillonBusinessException {
           String resultString = null;

       try {
           headwords=Utility.convertToUrlForEncoding(headwords,encoding);
           if (method.equalsIgnoreCase("GET")) {
               url = url + "?" + parameters + headwords;
           }
           URL myURL = null;
           myURL = new URL(url);

           HttpURLConnection myConnection = (HttpURLConnection)myURL.openConnection();
           myConnection.setRequestMethod(method);
           fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Connected to: " + url);

           if (method.equalsIgnoreCase("POST")) {
               myConnection.setDoOutput(true);

               OutputStream myOutputStream = myConnection.getOutputStream();
               myOutputStream.write((parameters+headwords).getBytes());
               fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Parameters: " + parameters+headwords);

               myOutputStream.flush();
               myOutputStream.close();
           }

           InputStream myInputStream = myConnection.getInputStream();
           ByteArrayOutputStream bytearrayout = new ByteArrayOutputStream();
           byte data1[] = new byte[1];

           int length;
           while(true) {
               length = myInputStream.read(data1);
               if (length < 0) break;
               bytearrayout.write(data1);
           }
           resultString = bytearrayout.toString();
           bytearrayout.close();
           myInputStream.close();
           myConnection.disconnect();

           resultString = extractResponse(resultString);
           resultString = Utility.convertFromEncoding(resultString,this.encoding);
       }
       catch (Exception e) {
           throw new PapillonBusinessException("Error in Wrapper.getResult() :", e);
       }

       return resultString;
    }
   
   protected String extractResponse(String response) {
       int offset = 0;
       if (null != preStartMatch && !preStartMatch.equals("")) {
           if ((offset=response.indexOf(preStartMatch))> 0) {
               response = response.substring(offset + preStartMatch.length());
           }
       }
       if (null != startMatch && !startMatch.equals("")) {
           if ((offset=response.indexOf(startMatch))> 0) {
               response = response.substring(offset + startMatch.length());
           }
       }
       if (null != endMatch && !endMatch.equals("")) {
           if ((offset=response.indexOf(endMatch))> 0) {
               response = response.substring(0, offset);
           }
       }
       return response;
   }

 }
