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
 * Revision 1.4  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.3  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 * Revision 1.2  2005/06/15 16:48:28  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.1.2.1  2005/05/27 09:54:38  mangeot
 * *** empty log message ***
 *
 *
 *-----------------------------------------------
 * Papillon Choose Translation page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.Element;

//pour le dictionary
import fr.imag.clips.papillon.business.dictionary.*;
import fr.imag.clips.papillon.business.transformation.XslTransformation;
import fr.imag.clips.papillon.business.utility.Utility;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.PapillonLogger;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import org.w3c.dom.Node;

public class ChooseEntry extends PapillonBasePO {

    protected final static String ChooseEntry_PARAMETER="Entry";
    protected final static String QueryString_PARAMETER="QueryString";
    protected final static String Volume_PARAMETER="VOLUME";

    protected final static String Ampersand="_@@_";
    protected final static String Equal="_@_";

    protected boolean loggedInUserRequired() {
        return true;
    }

    protected boolean userMayUseThisPO() {
        return false;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }

    public Node getContent() 
        throws HttpPresentationException,
		java.io.UnsupportedEncodingException
    {
        
        // Création du contenu
        ChooseEntryTmplXHTML content = (ChooseEntryTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("ChooseEntryTmplXHTML", this.getComms(), this.getSessionData());
        
        String volume = myGetParameter(Volume_PARAMETER);
        String queryString = myGetParameter(QueryString_PARAMETER);
        String word = myGetParameter(ChooseEntry_PARAMETER);

        String submit = myGetParameter(content.NAME_Choose);
        String redirection = myGetParameter(content.NAME_Redirection);
        String entryId = myGetParameter(content.NAME_EntryId);
        
		if (volume !=null && !volume.equals("") 
			&& queryString !=null && !queryString.equals("")
			&& word !=null && !word.equals("")) {
			
			String[] wordKey = new String[4];
			wordKey[0] = Volume.CDM_headword;
			wordKey[2] = word;
			wordKey[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];
			
			java.util.Vector Keys = new java.util.Vector();
			Keys.add(wordKey);
			
            // FIXME: fix the limit parameter
			java.util.Vector ContribVector = VolumeEntriesFactory.getVolumeNameEntriesVector(volume, Keys, null, null, 0,0);
			addEntryTable(content, ContribVector);
			addRedirection(content, queryString);
		}
		else if (submit !=null && !submit.equals("") 
			&& redirection !=null && !redirection.equals("")
			&& entryId !=null && !entryId.equals("")) {

			redirection = redirection.replaceAll(Ampersand,"&");
			redirection = redirection.replaceAll(Equal,"=");
			
			redirection += entryId;
			
			throw new ClientPageRedirectException(redirection);
		
		}
		
        //On rend le contenu correct
        return content.getElementFormulaire();
    }
	
	protected void addRedirection (ChooseEntryTmplXHTML content, String redirection)
        throws fr.imag.clips.papillon.business.PapillonBusinessException,
        java.io.UnsupportedEncodingException {

		XHTMLInputElement redirectionInput = content.getElementRedirection();
		redirectionInput.setValue(redirection);

	}
	    
	protected void addEntryTable (ChooseEntryTmplXHTML content, java.util.Vector ContribVector)
        throws fr.imag.clips.papillon.business.PapillonBusinessException,
        java.io.UnsupportedEncodingException {
			
            // On récupère les éléments du layout			
            XHTMLTableRowElement entryListRow = content.getElementEntryListRow();
            XHTMLElement entryCell = content.getElementEntryListCell();
            XHTMLInputElement entryButton = content.getElementEntryListButton();
						
			// EntryId
			entryListRow.removeAttribute("id");
			entryCell.removeAttribute("id");
			
            // On récupère le noeud contenant la table...
            Node entryTable = entryListRow.getParentNode();
            if (null != ContribVector && ContribVector.size()>0) {
                for(int i = 0; i < ContribVector.size(); i++) {
					VolumeEntry myContrib = (VolumeEntry) ContribVector.elementAt(i);
					if (myContrib!=null && !myContrib.isEmpty()) {					
					
						// the radio button
						entryButton.setValue(myContrib.getEntryId());
					
						// the XML	
						Element myElement = XslTransformation.applyXslSheets(myContrib);
						Utility.removeChildNodes(entryCell);
						entryCell.appendChild(content.importNode(myElement, true));

                        XHTMLElement clone = (XHTMLElement)entryListRow.cloneNode(true);
                        entryTable.appendChild(clone);
                    }
					else {
						PapillonLogger.writeDebugMsg("contrib empty ");
					}
				}
			}
			entryTable.removeChild(entryListRow);
        }

}
