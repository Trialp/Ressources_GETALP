/*
 *  papillon
 *
 *  Enhydra super-servlet
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.8  2006/08/10 22:17:13  fbrunet
 *  - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 *  - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 *  - Bug correction : +/- in advanced search
 *
 *  Revision 1.7  2006/04/06 15:06:39  fbrunet
 *  New class 'creationEditInit' : create new entry
 *  Modify LexALPEditEntry : only edit entry
 *
 *  Revision 1.6  2006/02/26 14:04:56  mangeot
 *  Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 *  Revision 1.5  2005/06/15 16:48:28  mangeot
 *  Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *  Revision 1.4  2005/05/24 12:51:22  serasset
 *  Updated many aspect of the Papillon project to handle lexalp project.
 *  1. Layout is now parametrable in the application configuration file.
 *  2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 *  3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 *  4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 *  5. It is now possible to give a name to the cookie key in the app conf file
 *  6. Several bug fixes.
 *
 *  Revision 1.3.4.2  2005/06/09 11:28:24  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.3.4.1  2005/06/09 11:07:45  mangeot
 *  Deleted the countEntriesCache. entries counts are not cached any more.
 *  Fixed a few bugs.
 *
 *  Revision 1.3  2005/04/11 08:01:02  fbrunet
 *  Passage en xhtml des ressources Papillon.
 *
 *  Revision 1.2  2005/01/15 12:51:24  mangeot
 *  Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 *  Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 *  Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 *  There are still bugs in the code.
 *
 *  -----------------------------------------------
 */
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

import fr.imag.clips.papillon.business.message.MessageDBLoader;

//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// Standard imports
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.Date;
import java.text.DateFormat;
import java.io.*;

//pour les dictionary
import fr.imag.clips.papillon.business.dictionary.*;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.*;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

/**
 *  Description of the Class
 *
 * @author     serasset
 * @created    December 2, 2004
 */
public class ViewDictionaries extends PapillonBasePO {

    /**
     *  Description of the Field
     */
    protected final static String SEE_VOLUME_PARAMETER = "SeeVolume";
    /**
     *  Description of the Field
     */
    protected final static String SEE_DICTIONARY_PARAMETER = "SeeDictionary";

    /**
     *  Description of the Field
     */
    protected ViewDictionariesXHTML content;


    /**
     *  Description of the Method
     *
     * @return    Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return false;
    }


    /**
     *  Description of the Method
     *
     * @return    Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        return true;
    }


    /**
     *  Gets the currentSection attribute of the ViewDictionaries object
     *
     * @return    The currentSection value
     */
    protected int getCurrentSection() {
        return NO_SECTION;
    }


    /**
     *  Gets the content attribute of the ViewDictionaries object
     *
     * @return                                                     The content
     *      value
     * @exception  javax.xml.parsers.ParserConfigurationException  Description
     *      of the Exception
     * @exception  HttpPresentationException                       Description
     *      of the Exception
     * @exception  IOException                                     Description
     *      of the Exception
     * @exception  org.xml.sax.SAXException                        Description
     *      of the Exception
     * @exception  javax.xml.transform.TransformerException        Description
     *      of the Exception
     */
    public Node getContent()
             throws javax.xml.parsers.ParserConfigurationException, HttpPresentationException,
            IOException, org.xml.sax.SAXException, javax.xml.transform.TransformerException {

        // Création du contenu
        content = (ViewDictionariesXHTML) MultilingualXHtmlTemplateFactory.createTemplate("ViewDictionariesXHTML", this.getComms(), this.getSessionData());

        HttpPresentationRequest req = this.getComms().request;
        // If the page is called with parameters, take the requested action
        if (req.getParameterNames().hasMoreElements()) {
            
            //
            if (null != req.getParameter(SEE_DICTIONARY_PARAMETER)) {
                String handle = req.getParameter(SEE_DICTIONARY_PARAMETER);
                Dictionary dict = DictionariesFactory.getDictionaryByHandle(handle);
                //adding an XML file
                addXml(dict.getXmlCode());
            }
            
            //
            if (null != req.getParameter(SEE_VOLUME_PARAMETER)) {
                String handle = req.getParameter(SEE_VOLUME_PARAMETER);
                Volume vol = VolumesFactory.getVolumeByHandle(handle);
                //adding an XML file
                addXml(vol.getXmlCode());
            }
        }

        //adding the content of the dictionaries table
        addDictionariesArray();

        //On rend le contenu correct
        return content.getElementFormulaire();
    }


    /**
     *  Adds a feature to the Xml attribute of the ViewDictionaries object
     *
     * @param  xmlString
     *      The feature to be added to the Xml attribute
     * @exception  fr.imag.clips.papillon.business.PapillonBusinessException
     *      Description of the Exception
     */
    protected void addXml(String xmlString)
             throws fr.imag.clips.papillon.business.PapillonBusinessException {

        Node xmlNode = XslSheetFactory.applyXslSheetForXml(xmlString);

        //where we must insert the xml
        HTMLElement theXml = content.getElementXml();

        Node theXmlParent = theXml.getParentNode();

        theXmlParent.appendChild(content.importNode(xmlNode, true));
        theXmlParent.removeChild(theXml);
    }



    /**
     *  Adds a feature to the DictionariesArray attribute of the
     *  ViewDictionaries object
     *
     * @exception  fr.imag.clips.papillon.business.PapillonBusinessException
     *      Description of the Exception
     */
    protected void addDictionariesArray()
             throws fr.imag.clips.papillon.business.PapillonBusinessException {

        Collection DictsTable = DictionariesFactory.getDictionariesArray();
        //where we must insert the form
        HTMLTableRowElement theRow = content.getElementTemplateRow();
        HTMLElement theDictionary = content.getElementDictionary();
        HTMLElement theVolume = content.getElementVolume();
        HTMLElement theName = content.getElementName();
        HTMLElement theCategory = content.getElementCategory();
        HTMLElement theType = content.getElementType();
        HTMLElement theSources = content.getElementSources();
        HTMLElement theTargets = content.getElementTargets();
        HTMLElement theEntries = content.getElementEntries();
        HTMLElement theDomain = content.getElementDomain();
        HTMLElement theLegal = content.getElementLegal();
        HTMLAnchorElement theSeeAnchor = content.getElementSeeAnchor();

        Node theRowParent = theRow.getParentNode();

        theRow.removeAttribute("id");
        theName.removeAttribute("id");
        theCategory.removeAttribute("id");
        theType.removeAttribute("id");
        theSources.removeAttribute("id");
        theTargets.removeAttribute("id");
        theEntries.removeAttribute("id");
        theDomain.removeAttribute("id");
        theLegal.removeAttribute("id");
        theSeeAnchor.removeAttribute("id");

        //adding the volumes description
        for (Iterator iter = DictsTable.iterator(); iter.hasNext();) {
            Utility.removeElement(theVolume);
            Dictionary dict = (Dictionary)iter.next();
            content.setTextName(dict.getName());
            content.setTextCategory(dict.getCategory());
            content.setTextType(dict.getType());
            content.setTextSources(dict.getSourceLanguages());
            content.setTextTargets(dict.getTargetLanguages());
            content.setTextDomain(dict.getDomain());
            content.setTextLegal(dict.getLegal());
            content.setTextEntries("Entries");

            theSeeAnchor.setHref(this.getUrl() + "?"
                     + SEE_DICTIONARY_PARAMETER + "="
                     + dict.getHandle());

            theRowParent.appendChild(theRow.cloneNode(true));
            theDictionary.getParentNode().appendChild(theVolume);

            //
            for (Iterator iter2 = VolumesFactory.getVolumesArray(dict.getName()).iterator(); iter2.hasNext();) {
                Volume volume = (Volume)iter2.next();
                
                Utility.removeElement(theDictionary);
                content.setTextName(volume.getName());
                content.setTextCategory("");
                content.setTextType("");
                content.setTextSources(volume.getSourceLanguage());
                content.setTextTargets(volume.getTargetLanguages());
                content.setTextDomain("");
                content.setTextLegal("");
                // FIXME: Finished entry but not-finished entry ?
                //int nb = volume.getCount(VolumeEntry.FINISHED_STATUS) + volume.getCount(VolumeEntry.NOT_FINISHED_STATUS);
                content.setTextEntries("" + volume.getEntries());

                theSeeAnchor.setHref(this.getUrl() + "?"
                                     + SEE_VOLUME_PARAMETER + "="
                                     + volume.getHandle());
                theRowParent.appendChild(theRow.cloneNode(true));
                theVolume.getParentNode().appendChild(theDictionary);
            }
        }

        theRowParent.removeChild(theRow);
    }

}
