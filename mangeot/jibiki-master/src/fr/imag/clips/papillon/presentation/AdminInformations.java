/*
 * papillon
 *
 * Enhydra super-servlet
 *
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.11  2006/08/13 14:19:04  mangeot
 * changed req.getParameter into myGetParameter
 *
 * Revision 1.10  2006/08/10 19:21:37  mangeot
 * *** empty log message ***
 *
 * Revision 1.9  2006/08/10 19:12:57  mangeot
 * *** empty log message ***
 *
 * Revision 1.8  2006/08/10 18:36:26  mangeot
 * *** empty log message ***
 *
 * Revision 1.7  2006/08/10 16:43:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.6  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 * Revision 1.5  2005/06/15 16:48:28  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.4  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.3.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.3  2005/04/11 08:01:02  fbrunet
 * Passage en xhtml des ressources Papillon.
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *-----------------------------------------------
 *
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.*;

//pour les erreurs dans l'importation de fichiers
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;


//pour les upload
import com.lutris.mime.*;

//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.html.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;
import org.w3c.dom.Document;

import javax.xml.parsers.*;
import org.xml.sax.InputSource;

//import fr.imag.clips.papillon.business.message.MessageDBLoader;
import com.lutris.util.KeywordValueTable;
import com.lutris.util.Config;
import com.lutris.util.KeywordValueException;
import com.lutris.util.ConfigException;
import com.lutris.appserver.server.Enhydra;



// Standard imports
import java.io.IOException;
import java.util.Date;
import java.util.Vector;
import java.util.ArrayList;
import java.text.DateFormat;
import java.io.*;
import java.lang.Integer;

//pour les Information file
import fr.imag.clips.papillon.business.informationfile.*;
//for the choice of the type and encoding
import fr.imag.clips.papillon.business.filetype.*;
import fr.imag.clips.papillon.business.encoding.*;


import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.*;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.locales.Languages;
import fr.imag.clips.papillon.business.dictionary.AvailableLanguages;

import de.opus5.servlet.*;

import fr.imag.clips.papillon.business.PapillonImportException;

import fr.imag.clips.papillon.presentation.PapillonPresentationException;
import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.presentation.XMLTitleParser;

public class AdminInformations extends PapillonBasePO {

    //the parameters
    //protected final static String SEE_URL="ConsultInformations.po?fileid=";
    protected final static String REMOVE_PARAM = "remove";
    protected final static String ADD=AdminInformationsTmplXHTML.NAME_Add;
    protected final static String REPLACE=AdminInformationsTmplXHTML.NAME_Replace;
    protected final static String FLUSH="Flush";

    protected final static String FILE_URL=AdminInformationsTmplXHTML.NAME_url;
    protected final static String FILE_TITLE=AdminInformationsTmplXHTML.NAME_title;
    protected final static String FILE_SECTION=AdminInformationsTmplXHTML.NAME_section;
    protected final static String FILE_AUTHOR=AdminInformationsTmplXHTML.NAME_author;
    protected final static String FILE_LANGUAGE=AdminInformationsTmplXHTML.NAME_language;
    protected final static String FILE_DATE=AdminInformationsTmplXHTML.NAME_date;
    protected final static String FILE_REFERENCE=AdminInformationsTmplXHTML.NAME_Reference;

    //protected final static String UPLOAD_DIRECTORY="/Users/enhydra/papillon/classes/upload/";
    protected final static String TMP_DIR_CONFIG="Papillon.Informations.tmpDir";
    protected final static String IMPORT_HELP_PAGE = "ImportHelp.po";

    protected fr.imag.clips.papillon.business.informationfile.MediaPreferences prefs;

    protected AdminInformationsTmplXHTML content;

    protected Languages Languages;



    public AdminInformations () {
        prefs = new MediaPreferences();
    }

    protected boolean loggedInUserRequired() {
        return true;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }

    protected  int getCurrentSection() {
        return NO_SECTION;
    }

    private void removeInformationDocument(String docID)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            InformationDocumentFactory.deleteInformationDocument(docID);
        }


    public Node getContent() throws javax.xml.parsers.ParserConfigurationException,
				    PapillonPresentationException,
        HttpPresentationException,
        IOException,
        IllegalArgumentException ,
        com.lutris.mime.MimeException,
        javax.xml.transform.TransformerConfigurationException,
        javax.xml.transform.TransformerException,
        org.xml.sax.SAXException
    {

        /*********************************/
        // Reading the request
        /*********************************/
        // Avant toute chose, on récupère les paramêtres selon la méthode que le client a utilisé
        // Ce peut-être du GET (c'est standard), du POST en url-encoded (c'est standard aussi dans enhydra)
        // ou bien du POST en multipart (non géré par enhydra, mais par le multipart request).
        HttpPresentationComms comms=this.getComms();

        String contentType = comms.request.getContentType();
        if (contentType == null) {
            contentType = "";
        }
        ContentHeader contentHdr =new ContentHeader("Content-Type: " + contentType);
        javax.servlet.http.HttpServletRequest req = null;

        if (contentHdr.getValue().toLowerCase().startsWith("multipart/")) {
            // On a affaire à un multipart request, on construit donc la requête appropriée.
            // ATTENTION: Pour simplifier la gestion du fichier chargé plus tard, je choisi le constructeur
            // qui force la sauvegarde des données dans un fichier temporaire.
					PapillonLogger.writeDebugMsg("Multipart request");
            String tmpDir="/tmp/";      // Dangereux ???
            try {
                tmpDir = Enhydra.getApplication().getConfig().getString(TMP_DIR_CONFIG);
                if (! tmpDir.endsWith(File.separator)) {
                    tmpDir = tmpDir + File.separator;
                }
            } catch (KeywordValueException kve) {
                PapillonLogger.writeDebugMsg("Unexpected Error: Incorrect configuration file for group: "
                                             + TMP_DIR_CONFIG);
            }
            req = new MultipartRequest(comms.request.getHttpServletRequest(), 30000000, tmpDir,"papillon-tmp", false);
        }
				else {
					PapillonLogger.writeDebugMsg("Normal request");
					req = comms.request.getHttpServletRequest();
				}

        /*********************************/
        // Managing the request, it may be a flush, remove or add action.
        /*********************************/
        if (req.getParameter(FLUSH)!=null) {
            // InformationFile[] fileArray=InformationFileFactory.getInformationFilesArray();
            // for(int file=0;file<fileArray.length;file++) {
            //     removeInformationFile(fileArray[file]);
            // }
            // writeDebugMsg("database erased...");
        } else if (req.getParameter(REMOVE_PARAM) != null) {
            removeInformationDocument(req.getParameter(REMOVE_PARAM));
        } else if (req.getParameter(FILE_URL) != null && !req.getParameter(FILE_URL).equals("")) {
            // On a une demande d'ajout de document. On la traite...
            addInformationDocument(req);
        }

        /*********************************/
        // Creating the document Layout
        /*********************************/
        content = (AdminInformationsTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("AdminInformationsTmplXHTML", this.getComms(), this.getSessionData());

        addConsultForm();

        addFileListing();

        return content.getElementFormulaire();

    }

    protected void addFileListing ()
        throws fr.imag.clips.papillon.business.PapillonBusinessException,
        PapillonPresentationException {

        /*********************************/
        //adding the file listing
        /*********************************/

        InformationDocument[] docTable=InformationDocumentFactory.getInformationDocumentArray();

        HTMLTableRowElement theRow = content.getElementTemplateRow();
        HTMLElement theDocTitle = content.getElementDocumentTitle();
        HTMLElement theAuthor = content.getElementDocumentAuthor();
        HTMLElement theDate = content.getElementDocumentDate();
        HTMLElement theReference = content.getElementDocumentReference();
        HTMLAnchorElement theDocumentAnchor = content.getElementDocumentAnchor();
        HTMLInputElement theReplaceButton = content.getElementReplaceButton();
        HTMLAnchorElement theRemoveAnchor = content.getElementRemoveAnchor();
        HTMLElement theSpaces = content.getElementNBSpace();

        Node theTitleCell = theDocumentAnchor.getParentNode();

        theRow.removeAttribute("id");
        theDocTitle.removeAttribute("id");
        theAuthor.removeAttribute("id");
        theDate.removeAttribute("id");
        theReference.removeAttribute("id");
        theReplaceButton.removeAttribute("id");
        theRemoveAnchor.removeAttribute("id");
        theSpaces.removeAttribute("id");
        theDocumentAnchor.removeAttribute("id");
        

    Node theRowParent = theRow.getParentNode();
    Node theReplaceButtonParent = theReplaceButton.getParentNode();
    Node theRemoveAnchorParent = theRemoveAnchor.getParentNode();

        //adding the file list
        for ( int i = 0; i < docTable.length; i++ ) {
            theReplaceButtonParent.removeChild(theReplaceButtonParent.getFirstChild());
            theRemoveAnchorParent.removeChild(theRemoveAnchorParent.getFirstChild());
            content.setTextDocumentAuthor(docTable[i].getAuthor());
            content.setTextDocumentDate(Utility.PapillonPrintDateFormat.format(docTable[i].getCreationDate()));
            content.setTextDocumentReference(docTable[i].getReference());

            //parse the XML title stored in the DB
            XMLTitleParser xtp = new XMLTitleParser();

            // get available languages
            // the first language of the vector is the one to use to display the title
            Vector theLangs = xtp.buildLangs(docTable[i], this.getSessionData().getUserAcceptLanguages());

            // add available languages next to the document title
            // we should replace this by flags pictures
            
            for (int j = 1; j < theLangs.size(); j++) {
                String langCode = (String)theLangs.get(j);
                String locLang = Languages.localizeName(getUserPreferredLanguage(),langCode);
                content.setTextDocumentTitle(locLang);
                theDocumentAnchor.setHref(prefs.getConsultDocumentUrl() +
                                          docTable[i].getHandle() + "&" +
                                          prefs.DOC_LANG_PARAMETER + "=" +
                                          langCode);
                theTitleCell.appendChild(theDocumentAnchor.cloneNode(true));
                theTitleCell.appendChild(theSpaces.cloneNode(true));
            }

            // add the title in the preferred language
            String title = xtp.getTitle((String)theLangs.get(0));

            // it should not happen
            if (title.trim().equals("")) title = "Untitled Document";
            content.setTextDocumentTitle(title);
            theDocumentAnchor.setHref(prefs.getConsultDocumentUrl() +
                                      docTable[i].getHandle() + "&" +
                                      prefs.DOC_LANG_PARAMETER + "=" +
                                      theLangs.get(0));

            // The replace button and remove anchor are put only if
            // - the user logged is the owner of the document or
            // - the user is admin
            String replaceString = "this.form." + REPLACE + ".value='" + 
            docTable[i].getHandle() + "'";

            if (this.getUser().isAdmin() ||
                this.getUser().getLogin().equals(docTable[i].getOwner())) {
                theReplaceButton.setAttribute("onclick", replaceString);
                theRemoveAnchor.setHref(this.getUrl() + "?" + REMOVE_PARAM + "=" +
                                        docTable[i].getHandle());
                theReplaceButtonParent.appendChild(theReplaceButton.cloneNode(true));
                theRemoveAnchorParent.appendChild(theRemoveAnchor.cloneNode(true));
            }
            theRowParent.appendChild(theRow.cloneNode(true));
            // Here I remove the childs precedently added for multilingual docs
            if (theLangs.size() > 1) {
                int nbChilds = theTitleCell.getChildNodes().getLength();
                for (int j = 0; j <= theLangs.size()-1; j++) {
                    theTitleCell.removeChild(theTitleCell.getLastChild());

                }
            }
        }
        theRowParent.removeChild(theRow);
    }


    public void addInformationDocument(javax.servlet.http.HttpServletRequest mReq)
        throws 	javax.xml.parsers.ParserConfigurationException,
        HttpPresentationException,
        IOException,
        IllegalArgumentException ,
        com.lutris.mime.MimeException,
        javax.xml.transform.TransformerConfigurationException,
        javax.xml.transform.TransformerException,
        org.xml.sax.SAXException  {

			
			String docUrl = mReq.getParameter(FILE_URL);

            // Perhaps we should give default values (problem: in what language ?)

            String docTitle= myGetParameter(FILE_TITLE);
            //file author
            String docAuthor= myGetParameter(FILE_AUTHOR);
            //file section
            String docSection= myGetParameter(FILE_SECTION);
            //file language
            String docLanguage= myGetParameter(FILE_LANGUAGE);
            //file date
            String docDate = myGetParameter(FILE_DATE);
            //file reference
            String docReference= myGetParameter(FILE_REFERENCE);


            // In case of Replace, existing doc id
            String docId= myGetParameter(REPLACE);

            String docOwner = this.getUser().getLogin();
			String userMessage = "";
            
            // these strings should not be null !!
            if (null == docTitle) { docTitle = ""; }
            if (null == docAuthor) { docAuthor = ""; }
            if (null == docSection) { docSection = ""; }
            if (null == docOwner) { docOwner = ""; }
            if (null == docLanguage) { docLanguage = ""; }
            if (null == docDate) { docDate = ""; }
            if (null == docReference) { docReference = ""; }

			/*
            if (mReq.getClass().getName().equals("de.opus5.servlet.MultipartRequest")) {

                de.opus5.servlet.UploadedFile f = ((MultipartRequest)mReq).getUploadedFile(FILE_FIELD);
                if (null != f && !f.getName().equals("")) {
                    try {
                        if (null != docId && !docId.equals("")){
                            InformationDocumentFactory.replaceInformationDocument(docId,
                                                                                  ((MultipartRequest)mReq).getUploadedFile(FILE_FIELD),
                                                                                  docTitle, docAuthor, docLanguage, docDate, docReference);
                        }
                        else {
                            InformationDocumentFactory.addNewInformationDocument(((MultipartRequest)mReq).getUploadedFile(FILE_FIELD),
                                                                                 docTitle, docAuthor, docOwner, docSection, docLanguage, docDate, docReference);
                        }
                    } catch(PapillonImportException pie) {
                        this.getSessionData().writeUserMessage("PapillonImportException ERROR IMPORTING DOCUMENT: "+pie.getMessage());
                        // redirect the user to import help page
                        throw new ClientPageRedirectException(IMPORT_HELP_PAGE);
                    } catch(Exception e) {
						 userMessage = "Exception ERROR IMPORTING DOCUMENT: "+e.getMessage();
                    }
					if (userMessage==null || userMessage.equals("")) {
					 userMessage = "Document imported";
					}
					 this.getSessionData().writeUserMessage(userMessage);
                }

            } */
			if (docUrl != null && !docUrl.equals("")) {
				try {
					if (null != docId && !docId.equals("")){
						InformationDocumentFactory.replaceInformationDocument(docId,
																			  docUrl,
																			  docTitle, docAuthor, docLanguage, docDate, docReference);
					}
					else {
						InformationDocumentFactory.addNewInformationDocument(docUrl,
																			 docTitle, docAuthor, docOwner, docSection, docLanguage, docDate, docReference);
					}
				} catch(PapillonImportException pie) {
					this.getSessionData().writeUserMessage("PapillonImportException ERROR IMPORTING DOCUMENT: "+pie.getMessage());
					// redirect the user to import help page
					throw new ClientPageRedirectException(IMPORT_HELP_PAGE);
				} catch(java.lang.Exception e) {
					userMessage = "java.lang.Exception ERROR IMPORTING DOCUMENT: "+e.getMessage();
				}
				if (userMessage==null || userMessage.equals("")) {
					userMessage = "Document imported";
				}
				this.getSessionData().writeUserMessage(userMessage);				
			} 
        }

    protected void addConsultForm()
        throws fr.imag.clips.papillon.business.PapillonBusinessException,
        HttpPresentationException,
        java.io.UnsupportedEncodingException {

            // Adding the appropriate options to the sections list
            /*********************************/
            HTMLInputElement theAuthorInputTemplate = content.getElementAuthor();
            HTMLOptionElement theOptionTemplate = content.getElementOptionTemplate();
            HTMLOptionElement theLanguageTemplate = content.getElementLanguageTemplate();
            Node theSelect = theOptionTemplate.getParentNode();
            Node theLanguageSelect = theLanguageTemplate.getParentNode();

            theOptionTemplate.removeAttribute("id");
            theLanguageTemplate.removeAttribute("id");

            theAuthorInputTemplate.setValue(this.getUser().getName());

            // We assume that the option element has only one text child
            // (it should be this way if the HTML is valid...)
            Text theTextTemplate = (Text)theOptionTemplate.getFirstChild();
            Text theLanguageTextTemplate = (Text)theLanguageTemplate.getFirstChild();

            InformationSections sections = new InformationSections();

            String[] sectionNames = sections.getInformationSectionArray();

            String[] sectionLocalizedNames =
                sections.getAcceptedLocalizedSectionArray((ArrayList)this.getSessionData().getUserAcceptLanguages());
            for (int i = 0; i < sectionNames.length; i++) {
                theOptionTemplate.setValue(sectionNames[i]);
				// Les infos récupérées des fichiers de config sont encodées en codage local.
				// Il faut donc les redécoder à l'envers pour avoir de l'unicode 
                theOptionTemplate.setLabel(Utility.convertFromLocalEncoding(sectionLocalizedNames[i]));
                // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux
                // specs W3C.
                theTextTemplate.setData(sectionLocalizedNames[i]);
                theSelect.appendChild(theOptionTemplate.cloneNode(true));
            }
						

            /*********************************/
            // Adding the languages list
            /*********************************/

            String[] allLanguages = Languages.getArray();
						String langLoc = getUserPreferredLanguage();
						for (int i = 0; i < allLanguages.length; i++) {
							String langi = allLanguages[i];
							theLanguageTemplate.setValue(langi);
							if (this.IsClientWithLabelDisplayProblems()) {
								theLanguageTemplate.setLabel(Languages.localizeLabel(langLoc,langi));
							}
							else {
								theLanguageTemplate.setLabel(Languages.localizeName(langLoc,langi));
							}
							theLanguageTemplate.setSelected(langi.equals(langLoc));

							// Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux
			 // specs W3C.
							theLanguageTextTemplate.setData(Languages.localizeName(langLoc, langi));
							theLanguageSelect.appendChild(theLanguageTemplate.cloneNode(true));
            }
						

            theSelect.removeChild(theOptionTemplate);
            theLanguageSelect.removeChild(theLanguageTemplate);
        }
}














