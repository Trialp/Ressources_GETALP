/*
 *  papillon
 *
 *  Enhydra super-servlet
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id: BrowseVolume.java,v 1.7.2.2 2006/11/26 12:04:29 mangeot Exp $
 *  -----------------------------------------------
 *  $Log: BrowseVolume.java,v $
 *  Revision 1.7.2.2  2006/11/26 12:04:29  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.7.2.1  2006/09/10 09:18:25  mangeot
 *  Resoures modified and added for playing with authentic xml plugin
 *
 *  Revision 1.7  2006/07/15 08:55:14  mangeot
 *  The BrowseVolumePage opens an HTML form that is used to lookup a volume in alphabetical order.
 *  The BrowseVolume is the server side of the AJAX script for retrieving the entries in alphabetical order
 *
 *  Revision 1.6  2006/02/27 00:04:01  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.5  2006/02/26 22:05:02  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.4  2006/02/26 20:24:30  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.3  2006/02/26 19:58:18  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.2  2006/02/26 19:21:38  mangeot
 *  Work on BrowseVolume
 *
 *  Revision 1.1  2006/02/26 14:09:32  mangeot
 *  *** empty log message ***
 *
 *
 *  -----------------------------------------------
 *  beta version
 */
package fr.imag.clips.papillon.presentation;

import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;
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


import java.util.ArrayList;
import java.util.Collection;

/**
*  Description of the Class
 *
 * @author     mangeot
 * @created    February 24, 2006
 */
public class LookupVolume extends AbstractPO {
    
	protected static final String ALL_STATUS = "*ALL*";
 
    protected static final int MAX_HOMOGRAPHS = 50;
	
	/* Beware, This feature is Postgresql only!!! */
	//protected static final String DOLLAR_QUOTING = "$GETA$";
	
    /**
    *  This method should be implemented in the subclass so that it returns
     *  true if this particular request requires the user to be logged in,
     *  otherwise false.
     *
     * @return    Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return false;
    }
    
    /**
    *  This method should be implemented in the subclass so that it returns
     *  true if this particular request requires the user to be logged in,
     *  otherwise false.
     *
     * @return    Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        return true;
    }
		
    /**
        *  Returns the complete document.
     *
     * @exception  Exception
     * @exception  HttpPresentationException  Description of the Exception
     * @exception  IOException                Description of the Exception
     */
    public org.w3c.dom.Node getDocument()
	throws HttpPresentationException, java.io.IOException, Exception {
			
			/* initialize response */
			java.util.Collection EntryCollection = null;
			org.w3c.dom.Document docResponse = XMLServices.buildDOMTree("<?xml version='1.0' encoding='UTF-8' ?><div id='entries'><script type='text/javascript'><!-- \n\n var em = document.getElementById('EmptyMessage'); if (em != null) em.setAttribute('style','display:none;');\n var plm = document.getElementById('PrefixLookupMessage'); if (plm != null) plm.setAttribute('style','display:none;');\n\n // --></script></div>");
			// Intialize QueryRequest
			/* volume */
			String volume = myGetParameter("VOLUME");
			String key = myGetParameter("KEY");
			String lang = myGetParameter("LANG");
			String word = myGetParameter("WORD");
			String handle = myGetParameter("HANDLE");
			String oneentry = myGetParameter("ENTRY");
			String msort = myGetParameter("MSORT");
			String action = myGetParameter("action");
			String order = myGetParameter("DIRECTION");
        
        if (word==null && oneentry!=null) {
            word=oneentry;
        }

		
		PapillonLogger.writeDebugMsg("LookupVolume: action: " + action + " VOLUME: " + volume + " WORD: " + word + " KEY: " + key + " ORDER: " + order);
		// advanced lookup
			if (action!=null && action.equals("advancedLookup")) {
				if (order!=null && order.equals(IndexFactory.ORDER_ASCENDING)) {
				AdvancedQueryForm qf = new AdvancedQueryForm(this.getComms(), this.getSessionData(), true, false);
				QueryRequest queryReq = qf.getQueryRequest();
				if (!queryReq.isEmpty() && !qf.actionOnFormRequested()) {
					//PapillonLogger.writeDebugMsg("---qr advanced is not empty");
										
					// Perform the request
					Collection qrset = queryReq.findIndex(this.getUser());
					
					// Display result
					String stringResponse = "<?xml version='1.0' encoding='UTF-8' ?><div class='entries'>";
					
                    String volumeName = null;
                    Volume theVolume = null;
                    
                    if (qrset!=null) {
                        for (java.util.Iterator myIterator = qrset.iterator(); myIterator.hasNext(); ) {
                            Index myIndex = (Index) myIterator.next();
                            if (theVolume == null && volumeName==null) {
                                theVolume = VolumesFactory.getVolumeByIndexDbname(myIndex.getTableName());
                                volumeName = theVolume.getName();
                            }
                            String displayValue = "";
                            if (myIndex.getKey().equals(Volume.CDM_headword)) {
                                displayValue = Utility.encodeXMLEntities(myIndex.getValue());
                            }
                            else {
                                java.util.Collection resultsVector = IndexFactory.getIndexVectorByEntryId(theVolume, myIndex.getEntryId()+"");
                                String cdmHeadword = "";
                                java.util.Iterator indexIterator = resultsVector.iterator();
                                while (cdmHeadword == "" &&  indexIterator.hasNext()) {
                                    Index myEntry = (Index) indexIterator.next();
                                    if (myEntry.getKey().equals(Volume.CDM_headword)) {
                                        cdmHeadword = myEntry.getValue();
                                    }
                                }
                                if (cdmHeadword!="") {
                                    displayValue = Utility.encodeXMLEntities(cdmHeadword);
                                }
                            }
							String entry = "<div class='lookupentry' msort='"+ Utility.encodeXMLEntities(myIndex.getMsort())+"'><a href='javascript:void(0);' style='display:block; margin:5px;' onclick='lookupVolume(\"VOLUME="+volumeName+"&amp;HANDLE="+myIndex.getEntryId()+"\");$(this).parent().css(\"font-weight\",\"bold\")'>"+displayValue+"</a></div>";
							stringResponse += entry;
						}
					}					
					//PapillonLogger.writeDebugMsg("stringResponse.length(): "+ stringResponse.length());
					stringResponse +=  "</div>";
					docResponse = XMLServices.buildDOMTree(stringResponse);					
				}
				}
			}		
			//alphabetical lookup
			else if (action!=null && action.equals("lookupVolume") && volume != null && !volume.equals("") && 
				(word != null && !word.equals("") || msort != null && !msort.equals(""))) {
				Volume myVolume = VolumesFactory.getVolumeByName(volume);
				int limit = 100;
				String limitString = myGetParameter("LIMIT");
				if (limitString!=null && !limitString.equals("")) {
					try {
						limit = Integer.parseInt(limitString);
					}
					catch (java.lang.NumberFormatException nfe) {
						//PapillonLogger.writeDebugMsg("myGetParameter(LIMIT): " + limitString);
					}
				}
				String strategy = myGetParameter("STRATEGY");
				if (strategy==null || strategy.equals("")) {
					strategy = QueryBuilder.GREATER_THAN;
				}
				/* bug when I send strategy >= it is equals to QueryBuilder.GREATER_THAN_OR_EQUAL but does not work! */
				else if (strategy!=null && !strategy.equals("") && strategy.equals(QueryBuilder.GREATER_THAN_OR_EQUAL)) {
					//PapillonLogger.writeDebugMsg("strategy: "+ strategy + " = " + strategy.equals(QueryBuilder.GREATER_THAN_OR_EQUAL));
					strategy = QueryBuilder.GREATER_THAN_OR_EQUAL;
				}

				if (order==null || order.equals("")) {
					order = IndexFactory.ORDER_ASCENDING;
				}
				if (order.equals(IndexFactory.ORDER_DESCENDING)) {
					strategy = QueryBuilder.LESS_THAN;
				}
				//PapillonLogger.writeDebugMsg("LookupVolume: " + volume + " WORD: " + word + " KEY: " + key + " Order: " + order + " Strategy: " + strategy);
				if (lang==null || lang.equals("")) {
					lang = myVolume.getSourceLanguage();
				}
				
				if (key==null || key.equals("HEADWORD")|| key.equals("")) {
					key=Volume.CDM_headword;
				}
				
				if (word != null && !word.equals("")) {
					java.util.Vector myKeys = new java.util.Vector();
					String[] Headword = new String[4];
					Headword[0] = key;
					Headword[1] = lang;
					Headword[2] = word;
					Headword[3] = strategy;
					myKeys.add(Headword);
                    
                    /* si ça prend du temps, faire une table d'index juste pour les vedettes */
                   QueryCriteria criteriaStatus = new QueryCriteria();
                    criteriaStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);
                    criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.CLASSIFIED_FINISHED_STATUS);
                    criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.NOT_FINISHED_STATUS);
                    criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.DRAFT_STATUS);
                    criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.DELETED_STATUS);

                    java.util.Vector myClauses = new java.util.Vector();
                    myClauses.add(criteriaStatus.getFullClause());
                    
                    
					// PapillonLogger.writeDebugMsg("LookupVolume: lookupvolume [" + myVolume.getIndexDbname() + "] lang: [" + lang + "] WORD: [" + word + "] KEY: [" + key + "] strat: [" + strategy + "] order: [" + order + "] limit: [" + limit + "]");
					EntryCollection = IndexFactory.getIndexEntriesVector(myVolume.getIndexDbname(), myKeys, myClauses, order,limit, 0);
				}
				else if (msort != null && !msort.equals("")) {
 /* si ça prend du temps, faire une table d'index juste pour les vedettes */
                    
                    //PapillonLogger.writeDebugMsg("LookupVolume: lookupvolume msort [" + myVolume.getIndexDbname() + "] lang: [" + lang + "] WORD: [" + word + "] KEY: [" + key + "] msort: [ " + msort + "] strat: [" + strategy + "] order: [" + order + "] limit: [" + limit + "]");
                   EntryCollection = IndexFactory.getIndexEntriesVector(myVolume.getIndexDbname(), key, null, msort, strategy, order,limit,0);
				
				}
				if (EntryCollection!=null) {
					String stringResponse = "";
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
					//PapillonLogger.writeDebugMsg("stringResponse.length(): "+ stringResponse.length());
					stringResponse = "<?xml version='1.0' encoding='UTF-8' ?><div class='entries'>" + stringResponse + "</div>";
					docResponse = XMLServices.buildDOMTree(stringResponse);
                    //PapillonLogger.writeDebugMsg("LookupVolume fin lookupVolume: " + oneentry + " key: " + key);

				}
			}
			// one entry by handle lookup
			else if (action!=null && action.equals("queryHandle") && volume != null && !volume.equals("") && 
					 handle != null && !handle.equals("")) {
				Volume myVolume = VolumesFactory.getVolumeByName(volume);
				java.util.Collection targets = myVolume.getTargetLanguagesArray();
				EntryCollection = DictionariesFactory.findAnswerAndTranslations(volume, handle, targets, this.getUser());
				
				if (EntryCollection!=null) {
					org.w3c.dom.Element rootElement = docResponse.getDocumentElement();
					for (java.util.Iterator myIterator = EntryCollection.iterator(); myIterator.hasNext(); ) {
						QueryResult myQueryResult = (QueryResult) myIterator.next();
						ResultFormatter myResultFormater = ResultFormatterFactory.getFormatter(myQueryResult, null, ResultFormatterFactory.XHTML_DIALECT,null);
						org.w3c.dom.Element newEntry = (org.w3c.dom.Element)myResultFormater.getFormattedResult(myQueryResult, this.getUser());
						rootElement.appendChild(docResponse.importNode(newEntry, true));
					}
				}
			}
			// one entry by headword lookup
			else if (action!=null && action.equals("queryOneEntry") && volume != null && !volume.equals("") && 
					 oneentry != null && !oneentry.equals("")) {

				if (key==null || key.equals("HEADWORD")|| key.equals("")) {
					key=Volume.CDM_headword;
				}
				
				Volume myVolume = VolumesFactory.getVolumeByName(volume);
				String source = myVolume.getSourceLanguage();
				java.util.Collection targets = myVolume.getTargetLanguagesArray();
				/*if (lang==null || lang.equals("")) {
					if (myVolume.isDefaultLangCDMElement(key)) {
						lang = Volume.DEFAULT_LANG;
					}
					else {
						lang = source;
					}
				}*/
				
				java.util.Vector myKeys = new java.util.Vector();
				String[] Headword = new String[4];
				//PapillonLogger.writeDebugMsg("LookupVolume queryOneEntry: " + oneentry + " key: " + key);
				Headword[0] = key;
				Headword[1] = lang;
				Headword[2] = oneentry;
				Headword[3] = QueryBuilder.CASE_INSENSITIVE_EQUAL;
				myKeys.add(Headword);
                
                QueryCriteria criteriaStatus = new QueryCriteria();
                criteriaStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);
                criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.CLASSIFIED_FINISHED_STATUS);
                criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.NOT_FINISHED_STATUS);
                criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.DRAFT_STATUS);
                criteriaStatus.add("value", QueryCriteria.NOT_EQUAL, VolumeEntry.DELETED_STATUS);
               
                java.util.Vector myClauses = new java.util.Vector();
                myClauses.add(criteriaStatus.getFullClause());


				EntryCollection = DictionariesFactory.getDictionaryNameEntriesCollection(myVolume.getDictname(),
																			source,
																			targets,
																			myKeys,
																			myClauses,
																			null,
																			this.getUser(),
																						 0, MAX_HOMOGRAPHS);
 				if (EntryCollection!=null) {
                    /* Opération trop lente ! */
                    /* Headword[3] = QueryBuilder.GREATER_THAN_OR_EQUAL; */
                    java.util.Iterator myIterator = EntryCollection.iterator();
                    org.w3c.dom.Element prefixLookupElement = null;
                   if (!myIterator.hasNext()) {
                        Headword[3] = QueryBuilder.CASE_INSENSITIVE_STARTS_WITH;
                       EntryCollection = DictionariesFactory.getDictionaryNameEntriesCollection(myVolume.getDictname(),
                                                                                                 source,
                                                                                                 targets,
                                                                                                 myKeys,
                                                                                                 myClauses,
                                                                                                 null,
                                                                                                 this.getUser(),
                                                                                                 0, 1);
                       if (EntryCollection!=null) {
                           myIterator = EntryCollection.iterator();
                           if (myIterator.hasNext()) {
                               String prefixLookup = "<?xml version='1.0' encoding='UTF-8' ?><div><script type='text/javascript'><!-- \n\n document.getElementById('PrefixLookupMessage').setAttribute('style','display:block;');\n\n // --></script></div>";
                               prefixLookupElement = XMLServices.buildDOMTree(prefixLookup).getDocumentElement();
                           }
                       }
                   }
                    if (myIterator.hasNext()) {
                        java.util.Collections.sort((java.util.List) EntryCollection,new HeadwordComparator());
                       org.w3c.dom.Element rootElement = docResponse.getDocumentElement();
                        for (myIterator = EntryCollection.iterator(); myIterator.hasNext(); ) {
						QueryResult myQueryResult = (QueryResult) myIterator.next();
						ResultFormatter myResultFormater = ResultFormatterFactory.getFormatter(myQueryResult, null, ResultFormatterFactory.XHTML_DIALECT,null);
						org.w3c.dom.Element newEntry = (org.w3c.dom.Element)myResultFormater.getFormattedResult(myQueryResult, this.getUser());
						rootElement.appendChild(docResponse.importNode(newEntry, true));
                        }
                        if (prefixLookupElement != null) {
                            rootElement.appendChild(docResponse.importNode(prefixLookupElement, true));
                        }
                        // PapillonLogger.writeDebugMsg("Fin LookupVolume queryOneEntry");
                    }
                    else {
                        //PapillonLogger.writeDebugMsg("Pas de réponse!");
                        String stringResponse = "<?xml version='1.0' encoding='UTF-8' ?><div><script type='text/javascript'><!-- \n\n document.getElementById('EmptyMessage').setAttribute('style','display:block;');\n\n // --></script></div>";
                        docResponse = XMLServices.buildDOMTree(stringResponse);
                    }
				}
			}
			else if (action != null && !action.equals("")) {
						 throw new ClientPageRedirectException("Home.po?" + this.getComms().request.getQueryString());
			}
					 
			return docResponse;			
        }
}
