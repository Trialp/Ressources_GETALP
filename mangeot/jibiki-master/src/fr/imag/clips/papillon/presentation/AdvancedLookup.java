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
import org.enhydra.xml.xhtml.dom.*;


// Standard imports
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

import com.lutris.dods.builder.generator.query.QueryBuilder;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.Index;
import fr.imag.clips.papillon.business.dictionary.IndexFactory;
import fr.imag.clips.papillon.business.dictionary.QueryCriteria;
import fr.imag.clips.papillon.business.dictionary.QueryParameter;
import fr.imag.clips.papillon.business.dictionary.QueryRequest;
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

public class AdvancedLookup extends DilafBasePO {

	protected AdvancedLookupXHTML content = null;

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
        //PapillonLogger.writeDebugMsg("orig qs:" + this.getComms().request.getQueryString());

        // Création du contenu
		content =
		(AdvancedLookupXHTML)MultilingualXHtmlTemplateFactory.createTemplate("AdvancedLookupXHTML", this.getComms(), this.getSessionData());

		// Établissement de la requête
			
			// Intialize QueryRequest
		String action = myGetParameter("action");
		if (action != null && !action.equals("") && !action.equals("advancedLookup")) {
			throw new ClientPageRedirectException("Home.po?" + this.getComms().request.getQueryString());
		}
		
        
        // Retrieve parameters like AdvancedQueryForm object
        AdvancedQueryForm qf = new AdvancedQueryForm(this.getComms(), this.getSessionData(), true, false);
        
        // Import query form node in advanced search page
        XHTMLElement formHolder = content.getElementQueryMenuHolder();
        formHolder.appendChild(content.importNode(qf.getQueryFormNode("AdvancedSearch.po"), true));

        /* initialize response */
		java.util.Collection EntryCollection = null;
		Volume myVolume = null;
		org.w3c.dom.Document docResponse = XMLServices.buildDOMTree("<?xml version='1.0' encoding='UTF-8' ?><div id='entries'></div>");
		
			// AdvancedSearch query
        QueryRequest queryReq = qf.getQueryRequest();
 		if (!queryReq.isEmpty() && !qf.actionOnFormRequested()) {
			
            //PapillonLogger.writeDebugMsg("---qr advanced is not empty");
			
            // Perform the request
            Collection qrset = queryReq.findIndex(this.getUser());
            
            // Display result
			String stringResponse = "<?xml version='1.0' encoding='UTF-8' ?><div class='entries'>";
            String firstEntryStyle = " style='font-weight:bold;'";
			String volumeName = null;
            Volume theVolume = null;

			if (qrset!=null) {
				for (java.util.Iterator myIterator = qrset.iterator(); myIterator.hasNext(); ) {
					Index myIndex = (Index) myIterator.next();
					//if (theVolume == null && volumeName==null) {
						theVolume = VolumesFactory.getVolumeByIndexDbname(myIndex.getTableName());
						volumeName = theVolume.getName();
					//}
                    String displayValue = Utility.encodeXMLEntities(myIndex.getValue());
 					String entry = "<div class='lookupentry' msort='"+ Utility.encodeXMLEntities(myIndex.getMsort())+"' "+firstEntryStyle+"><a href='javascript:void(0);' style='display:block; margin:5px;' onclick='lookupVolume(\"VOLUME="+volumeName+"&amp;HANDLE="+myIndex.getEntryId()+"\");$(this).parent().css(\"font-weight\",\"bold\")'>"+displayValue+"</a></div>";
                    firstEntryStyle = "";
					stringResponse += entry;
				}
				if (qrset.size()==0) {
					stringResponse += "<!-- // -->";
				}
			}
			else {
				stringResponse += "<!-- // -->";
			}
			if (qrset.size()==qf.getQueryParameter().getLimit()) {
				QueryParameter qp = qf.getQueryParameter();
				int newOffset = qp.getOffset() + qp.getLimit();
				qp.setOffset(newOffset);
				org.enhydra.xml.xhtml.dom.XHTMLSpanElement queryString = content.getElementQueryString();
                String previousQueryString = this.getComms().request.getQueryString();
                previousQueryString = previousQueryString.replaceAll("&OFFSET=\\d*","");
				queryString.setAttribute("data-query-string",previousQueryString+"&OFFSET="+newOffset);
			}
			
			//PapillonLogger.writeDebugMsg("stringResponse.length(): "+ stringResponse.length());
			stringResponse +=  "</div>";
			docResponse = XMLServices.buildDOMTree(stringResponse);
			content.getElementLookupentries().appendChild(content.importNode(docResponse.getDocumentElement(), true));
        
			// Query the first entry from its database handle
			if (qrset!=null && qrset.iterator().hasNext()) {
				Index myIndex = (Index) qrset.iterator().next();
				Volume tempVolume = VolumesFactory.getVolumeByIndexDbname(myIndex.getTableName());
				java.util.Collection targets = tempVolume.getTargetLanguagesArray();
				EntryCollection = DictionariesFactory.findAnswerAndTranslations(tempVolume.getName(), myIndex.getEntryId()+"", targets, this.getUser());
				
				if (EntryCollection!=null) {
                    java.util.Collections.sort((java.util.List) EntryCollection,new HeadwordComparator());
					for (java.util.Iterator myIterator = EntryCollection.iterator(); myIterator.hasNext(); ) {
						QueryResult myQueryResult = (QueryResult) myIterator.next();
						ResultFormatter myResultFormater = ResultFormatterFactory.getFormatter(myQueryResult, null, ResultFormatterFactory.XHTML_DIALECT,null);
						org.w3c.dom.Element newEntry = (org.w3c.dom.Element)myResultFormater.getFormattedResult(myQueryResult, this.getUser());
						content.getElementLookupcontent().removeChild(content.getElementWelcomeMessage());
						content.getElementLookupcontent().appendChild(content.importNode(newEntry, true));
					}
				}
			}
		}
		
		//On rend le contenu correct
       return content.getElementMainContent();
    }
	
    public org.w3c.dom.Node getBannerContent() throws HttpPresentationException,
	java.io.IOException,
	ClassNotFoundException, fr.imag.clips.papillon.business.PapillonBusinessException 
	{
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
