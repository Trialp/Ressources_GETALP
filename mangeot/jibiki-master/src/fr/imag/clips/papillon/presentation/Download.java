/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id: Information.java 1296 2011-11-29 17:57:22Z mangeot $
 *-----------------------------------------------
 * $Log$
 * Revision 1.4  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.3  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.2  2004/12/24 08:57:44  serasset
 * Premiere version de l'interface avec fond papillon et transparence.
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// w3c imports
//import org.w3c.dom.html.HTMLElement;
import org.w3c.dom.Node;

// Standard imports
import java.io.IOException;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

import fr.imag.clips.papillon.presentation.MultilingualHtmlTemplateFactory;

public class Download extends PapillonBasePO {

    public Node getContent()
        throws HttpPresentationException, IOException {

	DownloadTmplXHTML content;

	// Création du contenu
	//content = (InfosTmplHTML)this.getComms().xmlcFactory.create(InfosTmplHTML.class);
        content = (DownloadTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("DownloadTmplXHTML", this.getComms(), this.getSessionData());
      
		String volume = myGetParameter("VOLUME");
			
				// Adding the volume list
				org.enhydra.xml.xhtml.dom.XHTMLSelectElement volumeSelect = content.getElementFileName();
				org.enhydra.xml.xhtml.dom.XHTMLOptionElement volumeOptionTemplate = content.getElementFileOptionTemplate();
				volumeOptionTemplate.removeAttribute("id");
				org.w3c.dom.Text volumeTextTemplate = (org.w3c.dom.Text)volumeOptionTemplate.getFirstChild();
				java.util.Collection AllVolumes = fr.imag.clips.papillon.business.dictionary.VolumesFactory.getVolumesArray();
				for (java.util.Iterator iter = AllVolumes.iterator(); iter.hasNext(); ) {
					fr.imag.clips.papillon.business.dictionary.Volume myVolume = (fr.imag.clips.papillon.business.dictionary.Volume)iter.next();
					volumeOptionTemplate.setValue(myVolume.getName());
					fr.imag.clips.papillon.business.dictionary.Dictionary myDict = fr.imag.clips.papillon.business.dictionary.DictionariesFactory.getDictionaryByName(myVolume.getDictname());
					//volumeOptionTemplate.setLabel(myVolume.getName());
					// Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux
					// specs W3C.
					volumeTextTemplate.setData(myDict.getFullName() + " " + myVolume.getName());
					volumeOptionTemplate.setSelected(myVolume.getName().equals(volume));
					volumeSelect.appendChild(volumeOptionTemplate.cloneNode(true));
				}
				volumeSelect.removeChild(volumeOptionTemplate);
			
        return content.getElementInfoContent();
        }

    protected boolean loggedInUserRequired() {
        return false;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }
    
    protected  int getCurrentSection() {
        return INFORMATIONS_SECTION;
    }


}
