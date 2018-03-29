/* ConsultInformations.java
 * User interface to the document repository.
 *----------------------------------
 * $Id$
 *---------------------------------
 * $log$
 *
 *-----------------------------------
 *
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.*;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

//pour les upload
import com.lutris.mime.*;

//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;
import org.w3c.dom.Document;
import org.w3c.dom.html.HTMLImageElement;

import javax.xml.parsers.*;
import org.xml.sax.InputSource;

import fr.imag.clips.papillon.business.message.MessageDBLoader;
import fr.imag.clips.papillon.presentation.PapillonPresentationException;

//for the choice of the type and encoding
import fr.imag.clips.papillon.business.filetype.*;
import fr.imag.clips.papillon.business.encoding.*;
import fr.imag.clips.papillon.business.user.User;

//for the languages
import fr.imag.clips.papillon.business.locales.Languages;


// Standard imports
import java.io.IOException;
import java.util.Date;
import java.text.DateFormat;
import java.io.*;
import java.util.Enumeration;
import java.util.Vector;
import java.util.ArrayList;
//pour les Information file
import fr.imag.clips.papillon.business.informationfile.*;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.*;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.utility.*;

import de.opus5.servlet.*;
import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.presentation.XMLTitleParser;

import fr.imag.clips.papillon.business.PapillonBusinessException;
      

public class ConsultInformations extends PapillonBasePO {

    protected MediaPreferences prefs;

    protected final static String TITLE_CONTAINS_PARAMETER="titlecontains";
    protected final static String AUTHOR_CONTAINS_PARAMETER="authorcontains";

    /* The any contains is not yet implemented because we need to query informationFile instead of informationDocument */
    protected final static String ANY_CONTAINS_PARAMETER="anycontains";

    protected ConsultInformationsTmplXHTML content;        


    public ConsultInformations () {
        prefs = new fr.imag.clips.papillon.business.informationfile.MediaPreferences();
    }

    protected boolean loggedInUserRequired() {
        return false;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }

    protected  int getCurrentSection() {
        return NO_SECTION;
    }

 
    
public Node getContent() throws javax.xml.parsers.ParserConfigurationException,
                        com.lutris.appserver.server.httpPresentation.HttpPresentationException,
                        fr.imag.clips.papillon.business.PapillonBusinessException,
                        java.io.UnsupportedEncodingException,
                        org.xml.sax.SAXException,
                        java.io.IOException {
    
    HttpPresentationRequest req = this.getComms().request;
    HttpPresentationComms comms=this.getComms();

    // Création du contenu
    content = (ConsultInformationsTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("ConsultInformationsTmplXHTML", this.getComms(), this.getSessionData());

    String fileId=req.getParameter(prefs.FILE_ID_PARAMETER);
    String docId = req.getParameter(prefs.DOC_ID_PARAMETER);
    String docLang = req.getParameter(prefs.DOC_LANG_PARAMETER);

    // on utilise myGetParameters qui permet de traiter les encodages
    String titleContains = myGetParameter(content.NAME_TitleContains);
    String authorContains = myGetParameter(content.NAME_AuthorContains);
    String refContains = myGetParameter(content.NAME_RefContains);

    if (docLang == null || docLang.equals("")) {
        docLang= "eng";
    }
    
    if ((docId != null) && (docLang != null)) {
        return getInformationDocument(docId,docLang);
    } else if (fileId!=null) {
        return getInformationFile(fileId);
    } else {
        return getInformationDocList(titleContains,authorContains,refContains);
    }
}

public Node getInformationDocument(String docId, String lang) throws
                        javax.xml.parsers.ParserConfigurationException,
                        com.lutris.appserver.server.httpPresentation.HttpPresentationException,
                        fr.imag.clips.papillon.business.PapillonBusinessException,
                        java.io.UnsupportedEncodingException,
                        org.xml.sax.SAXException,
                        java.io.IOException, 
                        com.lutris.appserver.server.httpPresentation.ClientPageRedirectException {
    // If the index is a html file, just get the file.
    // If the index is a gif, just return an IMG pointing to it.
    InformationDocument theDoc = InformationDocumentFactory.findInformationDocumentByID(docId);
    InformationFile theIndex = InformationFileFactory.findLocIndexFileForDocument(theDoc, lang);
    
    if (theIndex.isAnImage()) {
        HTMLPaletteXHTML palette = (HTMLPaletteXHTML)MultilingualXHtmlTemplateFactory.createTemplate("HTMLPaletteXHTML", this.getComms(), this.getSessionData());

        HTMLImageElement img = palette.getElementHTMLImageElementTemplate();
        img.removeAttribute("id");
        
        img.setSrc(prefs.getRelativeDirName() + theIndex.getFilename());
        
        return (Node)img;
    } else if (theIndex.isAHTMLFile()) {
        return getInformationFile(theIndex);
    } else {
        // Just redirect the client to the file Name...
        throw new ClientPageRedirectException(prefs.getRelativeDirName() + theIndex.getFilename());
    }
    
} 

public Node getInformationFile(InformationFile theFile) throws
                        javax.xml.parsers.ParserConfigurationException,
                        com.lutris.appserver.server.httpPresentation.HttpPresentationException,
                        fr.imag.clips.papillon.business.PapillonBusinessException,
                        java.io.UnsupportedEncodingException,
                        org.xml.sax.SAXException,
                        java.io.IOException {

        DocumentBuilderFactory usineDocB = DocumentBuilderFactory.newInstance();
        //on cree le constructeur de document
        DocumentBuilder docBuilder = usineDocB.newDocumentBuilder();
        
        InputSource htmlCode=new InputSource(new StringReader(theFile.getFilecode()));
        Document docXml=docBuilder.parse(htmlCode);
        
        // WARNING: What happens with javascripts, etc.
        return docXml;
}

public Node getInformationFile(String file_id) throws
                        javax.xml.parsers.ParserConfigurationException,
                        com.lutris.appserver.server.httpPresentation.HttpPresentationException,
                        fr.imag.clips.papillon.business.PapillonBusinessException,
                        java.io.UnsupportedEncodingException,
                        org.xml.sax.SAXException,
                        java.io.IOException {   

    // A FAIRE: comment gérer les fichiers non trouvés...
    InformationFile theFile=InformationFileFactory.findInformationFileByID(file_id);
    return getInformationFile(theFile);
}



    public Node getInformationDocList(String titleContains, String authorContains, String refContains)
	throws fr.imag.clips.papillon.presentation.PapillonPresentationException, 
	       fr.imag.clips.papillon.business.PapillonBusinessException,
	       com.lutris.appserver.server.httpPresentation.HttpPresentationException
    {

    /*
    <TD><SPAN id="DocumentAuthor">AUTHOR</SPAN><SPAN id="CreationDate">DATE</SPAN>
    <SPAN id="CreationDate">DATE</SPAN>
    <A id="DocumentAnchor" HREF="see"><SPAN id="DocumentTitle">TITLE</SPAN></A>
    <SPAN id="Reference">Reference</SPAN><SPAN id="NBSpace">&nbsp;&nbsp;</SPAN></TD>*/
        
	
        //where we must insert the form
        HTMLTableRowElement theSectionRow = content.getElementSectionRowTemplate();
        HTMLElement theSectionHeader = content.getElementSectionHeader();
	
        HTMLTableRowElement thedocRow = content.getElementDocumentRowTemplate();
        HTMLElement theDocTitle = content.getElementDocumentTitle();
        HTMLElement theDate = content.getElementCreationDate();
        HTMLElement theAuthor = content.getElementDocumentAuthor();
        HTMLAnchorElement theDocumentAnchor = content.getElementDocumentAnchor();
        HTMLElement theRef = content.getElementReference();
        HTMLElement theSpaces = content.getElementNBSpace();

	Node theRefCell = theRef.getParentNode();
        Node theRowParent = theSectionRow.getParentNode();
	
        theSectionRow.removeAttribute("id");
        theSectionHeader.removeAttribute("id");
	
        thedocRow.removeAttribute("id");
        theDocTitle.removeAttribute("id");
        theDate.removeAttribute("id");
        theAuthor.removeAttribute("id");
        theRef.removeAttribute("id");
        theDocumentAnchor.removeAttribute("id");
	theSpaces.removeAttribute("id");

        // First get the list of sections
        InformationSections sections = new InformationSections();
	
        String[] sectionNames = sections.getInformationSectionArray();
        // FIXME: We should use the first available accepted language. Currently we only use the first accept language.
        String[] sectionLocalizedNames = sections.getAcceptedLocalizedSectionArray((ArrayList)this.getSessionData().getUserAcceptLanguages());
        
        for (int si=0; si < sectionNames.length; si++) {
            // Get the documents for each section
            InformationDocument[] docTable=InformationDocumentFactory.getInformationDocumentArrayForSection(sectionNames[si], titleContains, authorContains, refContains);

            if (docTable.length > 0) {
                // Put the section header into the html, followed by all docs
                content.setTextSectionHeader(sectionLocalizedNames[si]);
                theRowParent.appendChild(theSectionRow.cloneNode(true));
		
                for (int di = 0; di < docTable.length; di++) {

		    // parse the XML title stored in the DB
		    XMLTitleParser xtp = new XMLTitleParser();
		    // get available languages
		    // the first language of the vector is the one to use to display the title
		    Vector theLangs = xtp.buildLangs(docTable[di], this.getSessionData().getUserAcceptLanguages());    
		    
		    // add available languages next to the document title
		    // i will replace this by flags pictures
		    for (int j = 1; j < theLangs.size(); j++) {
			String langCode = (String)theLangs.get(j);
			String locLang = Languages.localizeName(getUserPreferredLanguage(),langCode);
			content.setTextDocumentTitle(locLang);
			theDocumentAnchor.setHref(prefs.getConsultDocumentUrl() +
                             docTable[di].getHandle() + "&" +
                             prefs.DOC_LANG_PARAMETER + "=" +
                             langCode);
			theRefCell.appendChild(theDocumentAnchor.cloneNode(true));
			theRefCell.appendChild(theSpaces.cloneNode(true));
		    }
		    
		    // add the title in the preferred language
		    String title = xtp.getTitle((String)theLangs.get(0));
		    // it should not happend
		    if (title.trim().equals("")) title = "Untitled Document";

		    content.setTextDocumentTitle(title);
		    theDocumentAnchor.setHref(prefs.getConsultDocumentUrl() +
                                docTable[di].getHandle() + "&" +
                                prefs.DOC_LANG_PARAMETER + "=" + theLangs.get(0));
                    content.setTextDocumentAuthor(docTable[di].getAuthor());

                    // Creation date: dislapy only the year
                    java.text.SimpleDateFormat myDateFormat = new java.text.SimpleDateFormat("yyyy");
                    content.setTextCreationDate(myDateFormat.format(docTable[di].getCreationDate()));
                    content.setTextReference(docTable[di].getReference());
		    
		    theRowParent.appendChild(thedocRow.cloneNode(true));

                    // Here I remove the childs precedently added for multilingual docs
                    if (theLangs.size() > 1) {
                        int nbChilds = theRefCell.getChildNodes().getLength();
                        for (int ic = 0; ic <= theLangs.size()-1; ic++) {
                            theRefCell.removeChild(theRefCell.getLastChild());

                        }
                    }
                    
		}
	    }
	}
	theRowParent.removeChild(theSectionRow);
	theRowParent.removeChild(thedocRow);
	
	Element contenu=content.getElementInformationsFileListPlace();
	return contenu;
    }
}
