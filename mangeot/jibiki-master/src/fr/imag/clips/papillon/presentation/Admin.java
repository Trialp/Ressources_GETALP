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
 * Revision 1.15.2.1  2007/10/29 15:11:03  serasset
 * NEW: lexalp css now defines different forms for HARMONISED/REJECTED entries
 * NEW: added new db url/user/password configuration keys in papillon.properties file
 * BUG158: headwords are now harmonised at edition and search time, added a "normalise headword" admin action
 *
 * Revision 1.15  2007/01/15 19:44:33  serasset
 * Corrected an error due to HTMLDOM removal.
 *
 * Revision 1.14  2007/01/15 17:12:18  serasset
 * Several notes added, suppressed the HTMLDOM_CACHE stuff.
 *
 * Revision 1.13  2006/12/14 20:03:26  fbrunet
 * Add method to normalize value into XML structure.
 *
 * Revision 1.12  2006/12/13 09:32:00  fbrunet
 * *** empty log message ***
 *
 * Revision 1.11  2006/08/17 21:26:13  mangeot
 * Now uses myGetParameter() instead of req.getParameter() because of encoding problem
 *
 * Revision 1.10  2006/04/06 11:44:21  mangeot
 * Fixed a pb after the merge: EditData handle was broken
 *
 * Revision 1.9  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.8.2.3  2006/02/17 13:21:25  mangeot
 *
 * MM: modified AdvancedQueryForm. getAllTargetLanguages, getAllSourceLanguages and getCdmElementsWithDefaultLanguage are now static in AvailableLanguages.java in order to accelerate the execution.
 *
 * Revision 1.8.2.2  2006/01/25 15:22:23  fbrunet
 * Improvement of QueryRequest
 * Add new search criteria
 * Add modified status
 *
 * Revision 1.8.2.1  2005/10/24 16:29:19  fbrunet
 * Added fuzzy search capabilities.
 * Added possibility to rebuild the index DB tables.
 * Added Pre and post processors that could be defined by the user.
 *
 * Revision 1.8  2005/07/28 14:36:56  mangeot
 * Added a News presentation page that takes a static xhtml page and displays it.
 * People can edit this static page on the server by accessing it via ftp or ssh.
 * There is a cache that can be reset via the Admin.po page.
 *
 * Revision 1.7  2005/06/15 16:48:28  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.6  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.5.4.4  2005/06/09 11:07:45  mangeot
 * Deleted the countEntriesCache. entries counts are not cached any more.
 * Fixed a few bugs.
 *
 * Revision 1.5.4.3  2005/06/01 08:38:43  mangeot
 * Multi bug correction + added the possibility of disabling data edition
 * via the Admin.po page
 *
 * Revision 1.5.4.2  2005/05/25 22:06:11  mangeot
 * *** empty log message ***
 *
 * Revision 1.5.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.5  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.4  2005/04/11 08:01:02  fbrunet
 * Passage en xhtml des ressources Papillon.
 *
 * Revision 1.3.2.1  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
 *
 * Revision 1.3  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *
 *
 *-----------------------------------------------
 * Papillon Admin page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.Enhydra;
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;

import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.Node;

import fr.imag.clips.papillon.business.message.MessageDBLoader;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.PapillonBusinessException;


// Standard imports
import java.io.IOException;
import java.util.Date;
import java.text.DateFormat;

import fr.imag.clips.papillon.Papillon;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

public class Admin extends PapillonBasePO {

	/* by default, data is editable */
	public static boolean EDIT_DATA = true;
    public static int pageExpireTime = ((Papillon) Enhydra.getApplication()).getPageExpireTime();

	protected static String ALL_VOLUMES = "*all*";
	
	protected String selectedVolume = null;

    protected boolean loggedInUserRequired() {
        return true;
    }

    protected boolean userMayUseThisPO() {
        try {
            return this.getUser().isAdmin();
        } catch (PapillonBusinessException ex) {
            this.getSessionData().writeUserMessage("Error getting the authorisation to use this PO.");
        }
        return false;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }

    public Node getContent()
        throws HttpPresentationException, IOException {
        AdminTmplXHTML content;
        
        // Création du contenu
        content = (AdminTmplXHTML) MultilingualXHtmlTemplateFactory.createTemplate("AdminTmplXHTML", this.getComms(), this.getSessionData());
		
		//boolean cacheSet = fr.imag.clips.papillon.business.dictionary.VolumeEntry.CACHE_HTMLDOM;

		
        // If the page is called with parameters, take the requested action
        if (this.getComms().request.getParameterNames().hasMoreElements()) {
			
			selectedVolume = myGetParameter(content.NAME_VOLUME);
            // Get the main parameters
            String mhonarcFolder = myGetParameter(content.NAME_Folder);
     //       String dbUrl= myGetParameter(DB_URL);
            
            // Get the requested action, THERE IS NO DEFAULT...
            if (null != myGetParameter(content.NAME_AdminMessageSubmit)) {
                AbstractPO.adminMessage=myGetParameter(content.NAME_AdminMessage);
			}
            else if (null != myGetParameter(content.NAME_Flush)) {
                this.getSessionData().writeUserMessage("Flushing is not provided for the moment...");
            } else if (null != myGetParameter(content.NAME_Reload)) {
                this.getSessionData().writeUserMessage("Flushing&Reload is not provided for the moment...");
            } else if (null != myGetParameter(content.NAME_Update)) {
                // Updating is the default action...
                MessageDBLoader loader = new MessageDBLoader(mhonarcFolder);
                loader.updateDatabase();
                this.getSessionData().writeUserMessage(loader.getUserLog());
                this.getSessionData().writeUserMessage("Updating... done.");
            } else if (null != myGetParameter(content.NAME_ResetInterfaceDescriptionCache)) {
				fr.imag.clips.papillon.business.edition.UITemplates.resetCache();
            }
			else if (null != myGetParameter(content.NAME_ResetXslSheetTransformersCache)) {
				fr.imag.clips.papillon.business.transformation.XslTransformation.resetCache();
            }
			else if (null != myGetParameter(content.NAME_ResetLanguagesCache)) {
				fr.imag.clips.papillon.business.locales.Languages.resetCache();
				fr.imag.clips.papillon.business.dictionary.AvailableLanguages.resetCache();
            }
			else if (null != myGetParameter(content.NAME_ResetNewsCache)) {
				fr.imag.clips.papillon.presentation.News.resetCache();
            }
			//else if (null != myGetParameter(content.NAME_SetHTMLDomCaches)) {
				//String setCacheString = myGetParameter(content.NAME_HTMLDomCaches);
				//cacheSet = (setCacheString!=null && !setCacheString.equals(""));
				//fr.imag.clips.papillon.business.dictionary.VolumeEntry.setCacheHtmlDom(cacheSet);
                //this.getSessionData().writeUserMessage("HTML DOM cache is set? " + cacheSet);
            //}
			else if (null != myGetParameter(content.NAME_SetEditData)) {
				String setEditDataString = myGetParameter(content.NAME_EditData);
				EDIT_DATA = (setEditDataString!=null && !setEditDataString.equals(""));
                this.getSessionData().writeUserMessage("EditData is set? " + EDIT_DATA);
            }
			else if (null != myGetParameter(content.NAME_ReConstructionIndex) &&
					 null != selectedVolume) {
				if (selectedVolume.equals(ALL_VOLUMES)) {
					fr.imag.clips.papillon.business.dictionary.VolumesFactory.reConstructionIndex();
				}
				else {
					fr.imag.clips.papillon.business.dictionary.VolumesFactory.volumeNameReConstructionIndex(myGetParameter(content.NAME_VOLUME));
				}
            }
            else if (null != myGetParameter(content.NAME_SetPageExpireTime)) {
                pageExpireTime = Integer.parseInt(myGetParameter(content.NAME_PageExpireTime));
                ((Papillon) Enhydra.getApplication()).setPageExpireTime(pageExpireTime);
                this.getSessionData().writeUserMessage("PageExpireTime is " + pageExpireTime);
            }
					// FIXME: supress
			 else if (null != myGetParameter(content.NAME_ModifiedStatus)) {
					this.getSessionData().writeUserMessage("Status modification has been desactivated...");				
                //fr.imag.clips.papillon.business.dictionary.VolumesFactory.modifiedStatus(this.getUser());
            
            } else if (null != myGetParameter(content.NAME_Standardization)) {
                this.getSessionData().writeUserMessage("Standardization of xml data has been desactivated...");
                //fr.imag.clips.papillon.business.dictionary.VolumesFactory.standardization();
                
            } else if (null != myGetParameter(content.NAME_NormalizeHeadword)) {
                fr.imag.clips.papillon.business.dictionary.VolumesFactory.normalizeHeadwords();

            }
            else {
                String errorMessage = "Error: Wrong arguments";
                this.getComms().response.setStatus(HttpPresentationResponse.SC_BAD_REQUEST,errorMessage);
            }
        }
		
		addAdminForm(content, EDIT_DATA, pageExpireTime, selectedVolume);

        //On rends le contenu correct
        return content.getElementHomeContent();
    }
	
	protected static void addAdminForm (AdminTmplXHTML content, boolean editData, int pageExpireTime, String selectedVolume)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
	
		XHTMLOptionElement volumeOptionTemplate = content.getElementVolumeOptionTemplate();
        Node volumeSelect = volumeOptionTemplate.getParentNode();
        volumeOptionTemplate.removeAttribute("id");
        // We assume that the option element has only one text child 
        // (it should be this way if the HTML is valid...)
         org.w3c.dom.Text volumeTextTemplate = ( org.w3c.dom.Text)volumeOptionTemplate.getFirstChild(); 
		
        //
        for (java.util.Iterator iter = VolumesFactory.getVolumesArray().iterator(); iter.hasNext();) {
            String volumeName = ((Volume)iter.next()).getName();
            
            //
            volumeOptionTemplate.setValue(volumeName);
            volumeOptionTemplate.setLabel(volumeName);
            // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux 
            // specs W3C.
            volumeTextTemplate.setData(volumeName);
            volumeOptionTemplate.setSelected(volumeName.equals(selectedVolume));
            volumeSelect.appendChild(volumeOptionTemplate.cloneNode(true));
        }
        volumeSelect.removeChild(volumeOptionTemplate);
		
		
		//XHTMLInputElement cacheElement = content.getElementHTMLDomCaches();
		//cacheElement.setChecked(cacheSet);
		
            XHTMLInputElement editDataElement = content.getElementEditData();
            editDataElement.setChecked(editData);

            XHTMLInputElement pageExpireTimeElement = content.getElementPageExpireTime();
            pageExpireTimeElement.setValue(""+pageExpireTime);
		
	}
	
	
}
