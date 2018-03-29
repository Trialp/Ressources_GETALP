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
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;
import org.enhydra.xml.xmlc.XMLObject;

// Standard imports
import java.io.IOException;

import com.lutris.dods.builder.generator.query.QueryBuilder;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.Index;
import fr.imag.clips.papillon.business.dictionary.IndexFactory;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.transformation.ResultFormatterFactory;
import fr.imag.clips.papillon.business.utility.HeadwordComparator;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.xml.XMLServices;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

public class ConsultPage extends DilafBasePO {

	protected ConsultPageXHTML content = null;
	protected String volume = null;
    
    protected static final int MAX_HOMOGRAPHS = 50;

    protected boolean loggedInUserRequired() {
        return false;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }

    protected  int getCurrentSection() {
        return NO_SECTION;
    }
	
    public org.w3c.dom.Node getContent() throws HttpPresentationException,
	java.io.IOException,
	ClassNotFoundException, fr.imag.clips.papillon.business.PapillonBusinessException 
	{
        String javascript = myGetParameter("JAVASCRIPT");
        String status = myGetParameter(ConsultPageXHTML.NAME_STATUS);
		volume = myGetParameter(ConsultPageXHTML.NAME_VOLUME);
		String word = myGetParameter(ConsultPageXHTML.NAME_HEADWORD);
		
		// Création du contenu
		content =
		(ConsultPageXHTML)MultilingualXHtmlTemplateFactory.createTemplate("ConsultPageXHTML", this.getComms(), this.getSessionData());
		
//		org.enhydra.xml.xhtml.dom.XHTMLSpanElement queryString = content.getElementQueryString();
//		queryString.setAttribute("data-query-string",this.getComms().request.getQueryString());
		
		org.enhydra.xml.xhtml.dom.XHTMLElement volumeLabel = content.getElementVolumeLabel();
		
		// Adding the volume list
		//org.enhydra.xml.xhtml.dom.XHTMLSelectElement volumeSelect = content.getElementVOLUME();
		org.enhydra.xml.xhtml.dom.XHTMLOptionElement volumeOptionTemplate = content.getElementVolumeOptionTemplate();
		org.enhydra.xml.xhtml.dom.XHTMLSelectElement volumeSelect = (org.enhydra.xml.xhtml.dom.XHTMLSelectElement) volumeOptionTemplate.getParentNode();
		volumeOptionTemplate.removeAttribute("id");
		org.w3c.dom.Text volumeTextTemplate = (org.w3c.dom.Text)volumeOptionTemplate.getFirstChild();
		java.util.Collection AllVolumes = fr.imag.clips.papillon.business.dictionary.VolumesFactory.getVolumesArray();
		for (java.util.Iterator iter = AllVolumes.iterator(); iter.hasNext(); ) {
			fr.imag.clips.papillon.business.dictionary.Volume iterVolume = (fr.imag.clips.papillon.business.dictionary.Volume)iter.next();
			volumeOptionTemplate.setValue(iterVolume.getName());
			volumeOptionTemplate.setLabel(iterVolume.getName());
			// Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux
			// specs W3C.
			volumeTextTemplate.setData(iterVolume.getName());
			if (iterVolume.getName().equals(volume)) {
				volumeOptionTemplate.setAttribute("selected","selected");
			}
			else {
				volumeOptionTemplate.removeAttribute("selected");
			}
			volumeSelect.appendChild(volumeOptionTemplate.cloneNode(true));
		}
		volumeSelect.removeChild(volumeOptionTemplate);
		
		if (status != null && !status.equals("")) {
			org.enhydra.xml.xhtml.dom.XHTMLInputElement statusInput = content.getElementSTATUS();
			statusInput.setValue(status);
		}
		

		// Établissement de la requête
			
			/* initialize response */
			java.util.Collection EntryCollection = null;
		Volume myVolume = null;
			org.w3c.dom.Document docResponse = XMLServices.buildDOMTree("<?xml version='1.0' encoding='UTF-8' ?><div id='entries'></div>");
			// Intialize QueryRequest
			/* volume */
			String key = myGetParameter("KEY");
			String handle = myGetParameter("HANDLE");
			String action = myGetParameter("action");
			String source = null;
			if (key==null || key.equals("HEADWORD")|| key.equals("")) {
				key=Volume.CDM_headword;
			}			
		
			//PapillonLogger.writeDebugMsg("ConsultPage: " + action + " volume:" + volume + " WORD: " + word + " KEY: " + key);
			if (action != null && !action.equals("")) {
				throw new ClientPageRedirectException("Home.po?" + this.getComms().request.getQueryString());
			}
			if (volume != null && volume.equals("Motamot_khm_api")) {
				volume = "Motamot_khm";
				key=Volume.CDM_pronunciation;
			}
			if (volume != null) {
				myVolume = VolumesFactory.getVolumeByName(volume);
				source = myVolume.getSourceLanguage();
			}
		// Query an entry from its headword
			if (myVolume != null && 
				word != null && !word.equals("")) {
                content.getElementLookupcontent().removeChild(content.getElementWelcomeMessage());
						java.util.Collection targets = myVolume.getTargetLanguagesArray();
			java.util.Vector myKeys = new java.util.Vector();
			String[] Headword = new String[4];
			//PapillonLogger.writeDebugMsg("ConsultPage entry: " + word + " key: " + key);
			Headword[0] = key;
			Headword[1] = source;
			Headword[2] = word;
			Headword[3] = QueryBuilder.EQUAL;
			myKeys.add(Headword);
			
			EntryCollection = DictionariesFactory.getDictionaryNameEntriesCollection(myVolume.getDictname(),
																					 source,
																					 targets,
																					 myKeys,
																					 null,
																					 null,
																					 this.getUser(),
																					 0, MAX_HOMOGRAPHS);
            if (EntryCollection!=null) {
                java.util.Iterator myIterator = EntryCollection.iterator();
                /* Opération trop lente ! */
   /*             if (!myIterator.hasNext()) {
                    Headword[3] = QueryBuilder.GREATER_THAN_OR_EQUAL;
                    EntryCollection = DictionariesFactory.getDictionaryNameEntriesCollection(myVolume.getDictname(),
                                                                                             source,
                                                                                             targets,
                                                                                             myKeys,
                                                                                             null,
                                                                                             null,
                                                                                             this.getUser(),
                                                                                             0, 1);
                }
                myIterator = EntryCollection.iterator(); */
                 if (!myIterator.hasNext()) {
                     Headword[3] = QueryBuilder.CASE_INSENSITIVE_STARTS_WITH;
                     EntryCollection = DictionariesFactory.getDictionaryNameEntriesCollection(myVolume.getDictname(),
                                                                                              source,
                                                                                              targets,
                                                                                              myKeys,
                                                                                              null,
                                                                                              null,
                                                                                              this.getUser(),
                                                                                              0, 1);
                     if (EntryCollection!=null) {
                         myIterator = EntryCollection.iterator();
                         if (myIterator.hasNext()) {
                             content.getElementPrefixLookupMessage().setAttribute("style","display: block");
                         }
                     }
                 }
                if (myIterator.hasNext()) {
                    java.util.Collections.sort((java.util.List) EntryCollection,new HeadwordComparator());
                   for (myIterator = EntryCollection.iterator(); myIterator.hasNext(); ) {
                    QueryResult myQueryResult = (QueryResult) myIterator.next();
                    ResultFormatter myResultFormater = ResultFormatterFactory.getFormatter(myQueryResult, null, ResultFormatterFactory.XHTML_DIALECT,null);
                    org.w3c.dom.Element newEntry = (org.w3c.dom.Element)myResultFormater.getFormattedResult(myQueryResult, this.getUser());
                    content.getElementLookupcontent().appendChild(content.importNode(newEntry, true));
                    }
                }
                else {
                    content.getElementEmptyMessage().setAttribute("style","display: block");
                }
			}
		}
		// Query an entry from its database handle
			if (myVolume != null && handle != null && !handle.equals("")) {
				java.util.Collection targets = myVolume.getTargetLanguagesArray();
				EntryCollection = DictionariesFactory.findAnswerAndTranslations(volume, handle, targets, this.getUser());
				
				if (EntryCollection!=null) {
					for (java.util.Iterator myIterator = EntryCollection.iterator(); myIterator.hasNext(); ) {
						QueryResult myQueryResult = (QueryResult) myIterator.next();
						VolumeEntry firstEntry = myQueryResult.getFirstSourceEntry();
						if (firstEntry != null) {
							if (key !=null && key.equals(Volume.CDM_pronunciation)) {
								word = myQueryResult.getFirstSourceEntry().getCdmString(Volume.CDM_pronunciation);
							}
							else {
								word = myQueryResult.getFirstSourceEntry().getHeadword();
							}
						}
						ResultFormatter myResultFormater = ResultFormatterFactory.getFormatter(myQueryResult, null, ResultFormatterFactory.XHTML_DIALECT,null);
						org.w3c.dom.Element newEntry = (org.w3c.dom.Element)myResultFormater.getFormattedResult(myQueryResult, this.getUser());
						content.getElementLookupcontent().appendChild(content.importNode(newEntry, true));
					}
				}
			}
		
		// Query an alphabetical list of entries from a word prefix
			if (myVolume != null && 
					 (word != null && !word.equals(""))) {
                if (javascript != null && !javascript.equals("")) {
                    org.w3c.dom.Document javascriptResponse = XMLServices.buildDOMTree("<?xml version='1.0' encoding='UTF-8' ?><script type='text/javascript'><!-- \n\n queryPrefixVolume('"+word+"','"+volume+"'); \n\n// --></script>");
                    content.getElementMainContent().appendChild(content.importNode(javascriptResponse.getDocumentElement(), true));
                }
                else {
				int limit = 30;
				/*String limitString = myGetParameter("LIMIT");
				if (limitString!=null && !limitString.equals("")) {
					limit = Integer.parseInt(limitString);
				}*/
				String strategy = QueryBuilder.GREATER_THAN_OR_EQUAL;
				String order = IndexFactory.ORDER_ASCENDING;
				
				java.util.Vector myKeys = new java.util.Vector();
				String[] Headword = new String[4];
				Headword[0] = key;
				Headword[1] = source;
				Headword[2] = word;
				Headword[3] = strategy;
				myKeys.add(Headword);
				PapillonLogger.writeDebugMsg("ConsultPage: [" + myVolume.getIndexDbname() + "] source: [" + source + "] WORD: [" + word + "] KEY: [" + key + "] strat: [" + strategy + "] order: [" + order + "] limit: [" + limit + "]");
				EntryCollection = IndexFactory.getIndexEntriesVector(myVolume.getIndexDbname(), myKeys, order,limit);
				String stringResponse = "";
				if (EntryCollection!=null) {
					for (java.util.Iterator myIterator = EntryCollection.iterator(); myIterator.hasNext(); ) {
						Index myIndex = (Index) myIterator.next();
						String entry = "<div class='lookupentry' msort='"+ Utility.encodeXMLEntities(myIndex.getMsort())+"'><a href='javascript:void(0);' style='display:block; margin:5px;' onclick='lookupVolume(\"VOLUME="+volume+"&amp;HANDLE="+myIndex.getEntryId()+"\");$(this).parent().css(\"font-weight\",\"bold\")'>"+Utility.encodeXMLEntities(myIndex.getValue())+"</a></div>";
						if (order.equals(IndexFactory.ORDER_DESCENDING)) {
							stringResponse = entry + stringResponse;
						}
						else {
							stringResponse += entry;
						}
					}
				}
				strategy = QueryBuilder.LESS_THAN;
				order = IndexFactory.ORDER_DESCENDING;
				myKeys = new java.util.Vector();
				Headword[3] = strategy;
				myKeys.add(Headword);
					//PapillonLogger.writeDebugMsg("ConsultPage: [" + myVolume.getIndexDbname() + "] source: [" + source + "] WORD: [" + word + "] KEY: [" + key + "] strat: [" + strategy + "] order: [" + order + "] limit: [" + limit + "]");
				EntryCollection = IndexFactory.getIndexEntriesVector(myVolume.getIndexDbname(), myKeys, order,limit);
				String stringResponse2 = "";
				if (EntryCollection!=null) {
					for (java.util.Iterator myIterator = EntryCollection.iterator(); myIterator.hasNext(); ) {
						Index myIndex = (Index) myIterator.next();
						String entry = "<div class='lookupentry' msort='"+ Utility.encodeXMLEntities(myIndex.getMsort())+"'><a href='javascript:void(0);' style='display:block; margin:5px;' onclick='lookupVolume(\"VOLUME="+volume+"&amp;HANDLE="+myIndex.getEntryId()+"\");$(this).parent().css(\"font-weight\",\"bold\")'>"+Utility.encodeXMLEntities(myIndex.getValue())+"</a></div>";
						if (order.equals(IndexFactory.ORDER_DESCENDING)) {
							stringResponse2 = entry + stringResponse2;
						}
						else {
							stringResponse2 += entry;
						}
					}
				}
				
				//PapillonLogger.writeDebugMsg("stringResponse.length(): "+ stringResponse.length());
				stringResponse = "<?xml version='1.0' encoding='UTF-8' ?><div class='entries'>" + stringResponse2 + stringResponse + "</div>";
				docResponse = XMLServices.buildDOMTree(stringResponse);
				content.getElementLookupentries().appendChild(content.importNode(docResponse.getDocumentElement(), true));
                }
			}

		//On rend le contenu correct
       return content.getElementMainContent();
    }
	
    public org.w3c.dom.Node getBannerContent() throws HttpPresentationException,
	java.io.IOException,
	ClassNotFoundException, fr.imag.clips.papillon.business.PapillonBusinessException 
	{
		//On rend le contenu correct
		org.enhydra.xml.xhtml.dom.XHTMLElement selectedVolume = (org.enhydra.xml.xhtml.dom.XHTMLElement) content.getElementById("Description_"+volume);
		if (selectedVolume!=null) {
			selectedVolume.setAttribute("style","display:block");
		}
		else {
			selectedVolume = (org.enhydra.xml.xhtml.dom.XHTMLElement) content.getElementById("Description_Default");
			if (selectedVolume!=null) {
				selectedVolume.setAttribute("style","display:block");
			}
		}
		org.w3c.dom.Node banner = content.getElementBannerContent();
		return banner;
		//return null;
    }

	public org.w3c.dom.Node getContextualMenuContent() throws HttpPresentationException,
	java.io.IOException,
	ClassNotFoundException, fr.imag.clips.papillon.business.PapillonBusinessException 
	{
		//On rend le contenu correct
		return content.getElementContextualMenuContent();
		//return null;
    }
}
