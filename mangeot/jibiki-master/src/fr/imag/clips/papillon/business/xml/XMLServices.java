/*
 * Jibiki project
 *
 * � Gilles S�rasset and Jibiki development team - GETA CLIPS IMAG
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.2.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.2  2007/01/15 17:12:18  serasset
 * Several notes added, suppressed the HTMLDOM_CACHE stuff.
 *
 * Revision 1.1  2007/01/05 13:57:26  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 *
 */
package fr.imag.clips.papillon.business.xml;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
//import org.apache.xalan.serialize.SerializerToText;
import org.apache.xml.serialize.OutputFormat;
import org.apache.xml.serialize.XMLSerializer;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.xml.sax.InputSource;

import javax.xml.parsers.DocumentBuilder;
import java.io.*;
import java.net.URL;

/**
 * Wrapper around XML parsing classes. This uses the XMLParserPool to reduce
 * parser initialisation time.
 */
public class XMLServices {

 //   private static SerializerToText mySerializerToText = new SerializerToText();
    // FIXME: there may be an encoding problem because we use the default encoding here instead of UTF-8
    // in order to build the Outputformat("text","UTF-8",true);
    private static OutputFormat myOutputFormat = new OutputFormat();


    /**
     * Builds an empty DOM document from scratch
     *
     * @return the DOM Document
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    public static Document newDOMTree() throws PapillonBusinessException {
        Document resultDoc = null;

        DocumentBuilder parser = XMLParsersPool.allocateParser();
        resultDoc = parser.newDocument();
        XMLParsersPool.releaseParser(parser);
        return resultDoc;
    }

    /**
     * Builds a DOM document from an URL
     *
     * @param theUrl URL of the source xml
     * @return the DOM Document
     */
    public static Document buildDOMTree(URL theUrl) throws PapillonBusinessException {
        return buildDOMTreeFromUrl(theUrl.toString());
    }

    /**
     * Builds a DOM document from an XML String
     *
     * @param xmlInputString The XML source as a String
     * @return the DOM Document
     */
    public static Document buildDOMTree(String xmlInputString) throws PapillonBusinessException {
        Reader myReader = new StringReader(xmlInputString);
        return buildDOMTree(new InputSource(myReader));
    }

    /**
     * Builds a DOM document from an XML InputSource
     *
     * @param mySource the input source
     * @return the DOM Document
     */
    public static Document buildDOMTree(InputSource mySource) throws PapillonBusinessException {
        Document contentDocument = null;

        try {
            DocumentBuilder parser = XMLParsersPool.allocateParser();
            contentDocument = parser.parse(mySource);
            XMLParsersPool.releaseParser(parser);
        } catch (org.xml.sax.SAXException saxe) {
            //PapillonLogger.writeDebugMsg("buildDOMTree: org.xml.sax.SAXException" + saxe);
			throw new PapillonBusinessException("buildDOMTree: org.xml.sax.SAXException", saxe);
        } catch (java.io.IOException ioe) {
            //PapillonLogger.writeDebugMsg("buildDOMTree: java.io.IOException" + ioe);
 			throw new PapillonBusinessException("buildDOMTree: java.io.IOException", ioe);
       }
        if (null == contentDocument) {
            //PapillonLogger.writeDebugMsg("buildDOMTree: DOCUMENT IS NULL !!!! ");
        }
        return contentDocument;
    }

    /**
     * Builds a DOM document from an URL String
     *
     * @param url * @return the DOM Document
     */
    public static Document buildDOMTreeFromUrl(String url) throws PapillonBusinessException {
        Document contentDocument = null;

        try {
            DocumentBuilder parser = XMLParsersPool.allocateParser();
            contentDocument = parser.parse(url);
            XMLParsersPool.releaseParser(parser);
        } catch (org.xml.sax.SAXException saxe) {
            PapillonLogger.writeDebugMsg("buildDOMTree from "+ url + " : org.xml.sax.SAXException" + saxe);
			throw new PapillonBusinessException("buildDOMTree from "+ url + " : org.xml.sax.SAXException", saxe);
        } catch (java.io.IOException ioe) {
            PapillonLogger.writeDebugMsg("buildDOMTree from "+ url + " : java.io.IOException" + ioe);
 			throw new PapillonBusinessException("buildDOMTree from "+ url + " : java.io.IOException", ioe);
		}
        return contentDocument;
    }

    public static String xmlCode(Document doc) throws PapillonBusinessException {
        StringWriter sw = new StringWriter();
        formatXml(doc, sw, false);
        return sw.toString();
    }

    public static String xmlCodePrettyPrinted(Document doc) throws PapillonBusinessException {
        StringWriter sw = new StringWriter();
        formatXml(doc, sw, true);
        return sw.toString();
    }

    public static void formatXml(Document doc, Writer wrt, boolean pretty) throws PapillonBusinessException {
        try {
            OutputFormat format = new OutputFormat(doc);
            format.setOmitXMLDeclaration(true);
            if (pretty) {
                format.setIndenting(true);
                format.setIndent(2);
            }
            XMLSerializer output = new XMLSerializer(wrt, format);
            output.serialize(doc);
        }
        catch (IOException e) {
            throw new PapillonBusinessException("IOException while serializing document.", e);
        }
    }

    public static void formatXml(Document doc, OutputStream outs, boolean pretty) throws PapillonBusinessException {
        try {
            OutputFormat format = new OutputFormat(doc);
            format.setOmitXMLDeclaration(true);
            if (pretty) {
                format.setIndenting(true);
                format.setIndent(2);
            }
            XMLSerializer output = new XMLSerializer(outs, format);
            output.serialize(doc);
        }
        catch (IOException e) {
            throw new PapillonBusinessException("IOException while serializing document.", e);
        }
    }

    /**
     * Serialize the DOM element
     *
     * @return the xml code as a string.
     * @deprecated Use xmlCode or xmlPretty unless you have a special requirement.
     */
    /* prints XML declaration */
    public static String xmlCodeOld(Document doc) {
        return ElementToString(doc.getDocumentElement(), true, false);
    }

    public static String DocumentToString(Document doc, boolean printXmlDeclaration, boolean setIndenting) {
        return DocumentToString(doc, printXmlDeclaration, setIndenting, false);
    }

    private static String DocumentToString(Document doc, boolean printXmlDeclaration, boolean setIndenting, boolean printDoctype) {
        String res = "";
        if (doc != null) {
            try {
                StringWriter myStringWriter = new StringWriter();
                myOutputFormat.setMethod("text");   //.setMethod("XML"); ?
                myOutputFormat.setIndenting(setIndenting);
                myOutputFormat.setOmitDocumentType(!printDoctype);
                myOutputFormat.setPreserveSpace(false);
//myOutputFormat.setEncoding("UTF-8");
                myOutputFormat.setOmitXMLDeclaration(!printXmlDeclaration);
                XMLSerializer myXMLSerializer = new XMLSerializer(myStringWriter, myOutputFormat);
                myXMLSerializer.serialize(doc);
                res = myStringWriter.toString();
            }
            catch (java.io.IOException ioe) {
                PapillonLogger.writeDebugMsg("NodeToString: java.io.IOException" + ioe);
            }
        }
        return res;
    }

    public static String ElementToString(Element elt) {
        return ElementToString(elt, true, true);
    }

    public static String ElementToString(Element elt, boolean printXmlDeclaration, boolean setIndenting) {
        String res = "";
        if (elt != null) {
            try {
                StringWriter myStringWriter = new StringWriter();
                myOutputFormat.setMethod("text");
                myOutputFormat.setIndenting(setIndenting);
                myOutputFormat.setOmitDocumentType(true);
                myOutputFormat.setOmitXMLDeclaration(!printXmlDeclaration);
                XMLSerializer myXMLSerializer = new XMLSerializer(myStringWriter, myOutputFormat);
                myXMLSerializer.serialize(elt);
                res = myStringWriter.toString();
            }
            catch (java.io.IOException ioe) {
                PapillonLogger.writeDebugMsg("NodeToString: java.io.IOException" + ioe);
            }
        }
        return res;
    }

   public static String DocumentFragmentToString(org.w3c.dom.DocumentFragment docfrag, boolean printXmlDeclaration, boolean setIndenting) {
        String res = "";
        if (docfrag != null) {
            try {
                StringWriter myStringWriter = new StringWriter();
                myOutputFormat.setMethod("text");
                myOutputFormat.setIndenting(setIndenting);
                myOutputFormat.setOmitDocumentType(true);
                myOutputFormat.setOmitXMLDeclaration(!printXmlDeclaration);
                XMLSerializer myXMLSerializer = new XMLSerializer(myStringWriter, myOutputFormat);
                myXMLSerializer.serialize(docfrag);
                res = myStringWriter.toString();
            }
            catch (java.io.IOException ioe) {
                PapillonLogger.writeDebugMsg("NodeToString: java.io.IOException" + ioe);
            }
        }
        return res;
    }

    /**
     * Serialize the DOM node
     *
     * @return the xml code as a string.
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */

/*
The unsolved dilemna is the following:
- the first implementation does not seem to work when the XML is not prefixed
or there is no accessible xml schema
- the second implementation does not output the namespace URIs when the elements are prefixed
*/
    public static String NodeToString(Node N) {
        return NodeToString(N, true);
    }

    public static String NodeToString(Node N, boolean printXmlDeclaration) {
        String res = "";
        if (N != null) {
  //          try {
 				switch (N.getNodeType()) {
					case org.w3c.dom.Node.DOCUMENT_NODE:               
						res = DocumentToString((org.w3c.dom.Document) N,printXmlDeclaration, true, true);
						break;
					case org.w3c.dom.Node.ELEMENT_NODE:
						res = ElementToString((org.w3c.dom.Element) N,printXmlDeclaration, true);
						break;
					case org.w3c.dom.Node.DOCUMENT_FRAGMENT_NODE:
						res = DocumentFragmentToString((org.w3c.dom.DocumentFragment) N,printXmlDeclaration, true);
						break;
				}				
/*                mySerializerToText.m_shouldNotWriteXMLHeader = (!printXmlDeclaration);
                mySerializerToText.setWriter(myStringWriter);
                mySerializerToText.serialize(N); 
                res = myStringWriter.toString();*/
//            }
  //          catch (java.io.IOException ioe) {
    //            PapillonLogger.writeDebugMsg("NodeToString: java.io.IOException: " + ioe);
      //      }
        }
        return res;
    }/*
	 public static String NodeToString(Node N) {
         NodeToString(N,true);
}
         public static String NodeToString(Node N, boolean printXmlDeclaration) {
        StringWriter myStringWriter = new StringWriter();
        try {
            // indent and omit xml declaration
            Transformer myTransformer = myTransformerFactory.newTransformer();
            myTransformer.setOutputProperty(javax.xml.transform.OutputKeys.INDENT,"yes");
            if (!printXmlDeclaration) {
                myTransformer.setOutputProperty(javax.xml.transform.OutputKeys.OMIT_XML_DECLARATION,"yes");
            }
            myTransformer.setOutputProperty(javax.xml.transform.OutputKeys.STANDALONE,"yes");
            DOMSource source = new DOMSource(n);
            StreamResult result = new StreamResult(myStringWriter);

            myTransformer.transform(source, result);
        } catch (TransformerConfigurationException tce) {
            // Error generated by the parser
            PapillonLogger.writeDebugMsg("NodeToString: Transformer Factory error " + tce.getMessage());

            // Use the contained exception, if any
            Throwable x = tce;
            if (tce.getException() != null)
                x = tce.getException();
            x.printStackTrace();
        } catch (TransformerException te) {
            // Error generated by the parser
            PapillonLogger.writeDebugMsg("NodeToString: Transformation error " + te.getMessage());
            // Use the contained exception, if any
            Throwable x = te;
            if (te.getException() != null)
                x = te.getException();
            x.printStackTrace();
        }
        return myStringWriter.toString();
}
*/

    /**
     * serialize a DOM document into a file
     *
     * @return void
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    public static void printToFile(Document theDoc, String filePath)
            throws PapillonBusinessException {
        try {
            java.io.OutputStream outStream = new java.io.FileOutputStream(filePath);
            java.io.PrintStream myPrintStream = new java.io.PrintStream(outStream, true, "UTF-8");
            myPrintStream.print(DocumentToString(theDoc, true, true, true));
            myPrintStream.close();
            outStream.close();
        }
        catch (java.io.UnsupportedEncodingException ueex) {
            throw new PapillonBusinessException("Exception in printToFile()", ueex);
        }
        catch (java.io.IOException ioex) {
            throw new PapillonBusinessException("Exception in printToFile()", ioex);
            }
        }
    
    public static String trimXmlDeclaration(String XmlString) {
        return XmlString.replaceAll("\\<\\?xml(.+?)\\?\\>", "").trim();
    }
}
