/*
*----------------------------------
*$Id$
*---------------------------------
*$log$
*
*-----------------------------------
*
*/

package fr.imag.clips.papillon.presentation;

//import org.enhydra.xml.xmlc.XMLObject;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Vector;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.informationfile.InformationDocument;
import fr.imag.clips.papillon.business.locales.LanguageNegotiator;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

/**
 * this class allows to parse titles stored in XML format in the database and to retrieve informations.
 */
public class XMLTitleParser {

    protected DocumentBuilderFactory docBuilderFactory = null;
    protected DocumentBuilder docBuilder = null;
    protected Document xmlTitleDoc = null;

    public final static String TITLE_ELEMENT = "title";
    
    // Constructor
    public XMLTitleParser () {
	//Nothing
    }




    /**
     * Parses the given String
     * @param theTitle the XML String to parse.
     */
    protected void parse(String theTitle) 
	throws javax.xml.parsers.ParserConfigurationException, org.xml.sax.SAXException, java.io.IOException {

        docBuilderFactory = DocumentBuilderFactory.newInstance();
        docBuilder = docBuilderFactory.newDocumentBuilder();
        InputSource titleIS=new InputSource(new StringReader(theTitle));
        xmlTitleDoc=docBuilder.parse(titleIS);
    }

	
    /**
     * Return the available languages in the parsed title.
     *
     * @return the vector of available languages codes in the xml title
     */
    public Vector getLangs() {

	Vector availLangs = new Vector();

	Node titleTag = xmlTitleDoc.getElementsByTagName(TITLE_ELEMENT).item(0);
	NodeList langList = titleTag.getChildNodes();

	for (int i = 0; i < langList.getLength(); i++) {
	    if (!langList.item(i).getNodeName().equals("#text")) {
		// Ignore blank text nodes added by the parser
		availLangs.add(langList.item(i).getNodeName());
	    }
	}
	return availLangs;
    }


    /**
     * Returns the title in the given language.
     *
     * @param lang, the language.
     *
     * @return the localized title in a string or null if the language is not available.
     */
    public String getTitle(String lang) {

	String locTitle = null;

	Element titleTag = (Element)xmlTitleDoc.getElementsByTagName(TITLE_ELEMENT).item(0);
	NodeList langTags = titleTag.getElementsByTagName(lang);
	if (!(langTags.getLength()==0)) {
	    locTitle = langTags.item(0).getFirstChild().getNodeValue();
	}
	return locTitle;
    }

    /**
     * Parses the document title and returns the list of languages that are available for
     * this document. The first language is the one to use to display the title.
     * 
     * @param theDoc the InformationDocument to add to the doc list
     * @param accLang the languages accepted by the user
     * @return the titles Vector
     */
    public Vector buildLangs(InformationDocument theDoc, ArrayList accLang) 
	throws PapillonPresentationException, PapillonBusinessException {

	Vector theLangs = new Vector();
	
	//parse the XML title stored in the DB
	String XMLTitle = theDoc.getTitle();
	String lang = null;
	
	try {
	    parse(XMLTitle);
	} catch(Exception e) {
	    throw new PapillonPresentationException("Error parsing document title", e);
	}
	
	Vector availLang = getLangs();
	
	lang = LanguageNegotiator.negotiateLanguage(accLang, availLang, (String)availLang.get(0));
	availLang.remove(lang);
	
	theLangs.add(lang);
	
	// add other available languages
	for (int j = 0; j < availLang.size(); j++) {
	    String langCode = (String)availLang.get(j);	    
	    theLangs.add(langCode);
	}
	return theLangs;
    }
}
