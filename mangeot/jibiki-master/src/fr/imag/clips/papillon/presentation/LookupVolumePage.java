/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Mathieu Mangeot & Jibiki team - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 *-----------------------------------------------
 * Lookup Volume page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import org.enhydra.xml.xmlc.XMLObject;

// Standard imports
import java.io.IOException;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

public class LookupVolumePage extends PapillonBasePO {


    protected boolean loggedInUserRequired() {
        return false;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }

    protected  int getCurrentSection() {
        return NO_SECTION;
    }

	public org.w3c.dom.Node getContent() throws HttpPresentationException, IOException {
		org.w3c.dom.Node contentNode = null;
		return contentNode;
	}
	
	
    public org.w3c.dom.Node getDocument()
	throws HttpPresentationException, java.io.IOException, Exception {
//    public Node getContent()
//    throws HttpPresentationException, IOException {
		
		String status = myGetParameter(LookupVolumePageXHTML.NAME_STATUS);
		String volume = myGetParameter(LookupVolumePageXHTML.NAME_VOLUME);

        // Création du contenu
        LookupVolumePageXHTML page =
            (LookupVolumePageXHTML)MultilingualXHtmlTemplateFactory.createTemplate("LookupVolumePageXHTML", this.getComms(), this.getSessionData());

		org.enhydra.xml.xhtml.dom.XHTMLElement volumeLabel = page.getElementVolumeLabel();
		
		if (!volumeLabel.hasAttribute("style")) {
			// Adding the volume list
			org.enhydra.xml.xhtml.dom.XHTMLSelectElement volumeSelect = page.getElementVOLUME();
			org.enhydra.xml.xhtml.dom.XHTMLOptionElement volumeOptionTemplate = page.getElementVolumeOptionTemplate();
			volumeOptionTemplate.removeAttribute("id");
			org.w3c.dom.Text volumeTextTemplate = (org.w3c.dom.Text)volumeOptionTemplate.getFirstChild();
			java.util.Collection AllVolumes = fr.imag.clips.papillon.business.dictionary.VolumesFactory.getVolumesArray();
			for (java.util.Iterator iter = AllVolumes.iterator(); iter.hasNext(); ) {
				fr.imag.clips.papillon.business.dictionary.Volume myVolume = (fr.imag.clips.papillon.business.dictionary.Volume)iter.next();
				volumeOptionTemplate.setValue(myVolume.getName());
				volumeOptionTemplate.setLabel(myVolume.getName());
				// Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux
				// specs W3C.
				volumeTextTemplate.setData(myVolume.getName());
				volumeSelect.appendChild(volumeOptionTemplate.cloneNode(true));
            }
			volumeSelect.removeChild(volumeOptionTemplate);
		}
		
		if (status != null && !status.equals("")) {
			org.enhydra.xml.xhtml.dom.XHTMLInputElement statusInput = page.getElementSTATUS();
			statusInput.setValue(status);
		}
		//On rend le contenu correct
       return (org.w3c.dom.Document) page;
    }

}
