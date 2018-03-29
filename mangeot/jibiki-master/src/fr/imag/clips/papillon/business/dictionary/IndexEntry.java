/*
 -----------------------------
 * $Id: IndexEntry.java 1316 2012-07-26 12:51:38Z zhangying $
 *-----------------------------------------------
 * $Log$
  *
 *-----------------------------------------------
 *
 * Created by Mathieu MANGEOT on Tuesday July 31 2012
 *  Copyright (c) 2003-2012 mangeot & zhangying GETALP-LIG. All rights reserved.
 */

package fr.imag.clips.papillon.business.dictionary;

// internal imports

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.business.dictionary.IndexEntry;
import fr.imag.clips.papillon.CurrentDBTransaction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.awt.List;
import java.sql.SQLException;

public class IndexEntry {
    
    protected static class IndexData {
        String CdmElement, lang, value, handle;

        public IndexData(String cdme, String lang, String value, String handle) {
            CdmElement = cdme;
            this.lang = lang;
            this.value = value;
            this.handle = handle;
        }

        public Index toIndex(String volumeidx) throws PapillonBusinessException {
            return IndexFactory.newIndex(volumeidx, CdmElement, lang, value, handle);

        }
    }

    public static void saveIndexes(ArrayList indexes, String volumeidx) throws PapillonBusinessException {
        Iterator i = indexes.iterator();
        while (i.hasNext()) {
            IndexData id = (IndexData) i.next();
			if (id==null) {PapillonLogger.writeDebugMsg("Index null!!! ");}
            Index myIndex = id.toIndex(volumeidx);
            if (myIndex != null && !myIndex.isEmpty()) {
                myIndex.save();
            }
        }
    }

    public static void saveLinks(ArrayList links) throws PapillonBusinessException {
        Iterator i = links.iterator();
        while (i.hasNext()) {
            Link myLink = (Link) i.next();
            if (myLink != null && !myLink.isEmpty()) {
                myLink.save();
            }
        }
    }
    
    public static boolean isFinishedEntry(ArrayList indexes) throws PapillonBusinessException {
        Iterator i = indexes.iterator();
        boolean isFinished = true;
        while (i.hasNext() && isFinished) {
            IndexData id = (IndexData) i.next();
            if (id.CdmElement.equals(Volume.CDM_contributionStatus) && !id.value.equals(VolumeEntry.FINISHED_STATUS)) {
                isFinished = false;
            }
         }
        return isFinished;
    }
    
    public static void saveNotFinishedStatusContributionIdIndexes(ArrayList indexes, String volumeidx) throws PapillonBusinessException {
        Iterator i = indexes.iterator();
        while (i.hasNext()) {
            IndexData id = (IndexData) i.next();
            if (id==null) {PapillonLogger.writeDebugMsg("Index null!!! ");}
            else if (id.CdmElement.equals(Volume.CDM_history)||
                     id.CdmElement.equals(Volume.CDM_modification)||
                     id.CdmElement.equals(Volume.CDM_modificationAuthor)||
                     id.CdmElement.equals(Volume.CDM_modificationDate)||
                     id.CdmElement.equals(Volume.CDM_modificationComment)||
                     id.CdmElement.equals(Volume.CDM_contributionAuthor)||
                     id.CdmElement.equals(Volume.CDM_contributionCreationDate)||
                     id.CdmElement.equals(Volume.CDM_contributionGroup)||
                     id.CdmElement.equals(Volume.CDM_contributionGroups)||
                     id.CdmElement.equals(Volume.CDM_contributionId)||
                     id.CdmElement.equals(Volume.CDM_contributionFinitionDate)||
                     id.CdmElement.equals(Volume.CDM_contributionReviewDate)||
                     id.CdmElement.equals(Volume.CDM_contributionReviewer)||
                     id.CdmElement.equals(Volume.CDM_contributionValidationDate)||
                     id.CdmElement.equals(Volume.CDM_contributionValidator)||
                     id.CdmElement.equals(Volume.CDM_contributionStatus)||
                     id.CdmElement.equals(Volume.CDM_originalContributionId)||
                     // FBM: Added by francis
                     // link to previous classified finished/not-finished contribution
                     id.CdmElement.equals(Volume.CDM_previousClassifiedFinishedContribution)||
                     id.CdmElement.equals(Volume.CDM_previousClassifiedNotFinishedContribution)||
                     // link to next contribution author
                     id.CdmElement.equals(Volume.CDM_nextContributionAuthor)||
                     // MM: Ajouté par Mathieu: simplifications !
                     id.CdmElement.equals(Volume.CDM_previousContribution)
                     ) {
                Index myIndex = id.toIndex(volumeidx);
                if (myIndex != null && !myIndex.isEmpty()) {
                    myIndex.save();
                }
            }
        }
    }
   
	
    // boolean yeld = true if everything is OK
    public static boolean indexEntry(VolumeEntry myEntry)
            throws PapillonBusinessException {
				
				//linksEntry(myEntry);
				
        org.w3c.dom.Document entryDoc = myEntry.getDom();
        boolean result = false;
				
			if (entryDoc != null) {
				java.util.HashMap CdmElementsTable = myEntry.getVolume().getCdmElements();
				org.w3c.dom.Element myRootElt = entryDoc.getDocumentElement();
					
				ArrayList indexes = indexEntry(CdmElementsTable, myRootElt, myEntry.getVolume().getPrefixResolver(), myEntry.getHandle());
				String volumeidx = myEntry.getVolume().getIndexDbname();
                if (isFinishedEntry(indexes)) {
                    saveIndexes(indexes, volumeidx);
                    java.util.HashMap linksTable = myEntry.getVolume().getLinksTable();
                    ArrayList links = indexEntryLinks(linksTable, myEntry.getVolume(), myRootElt, myEntry.getHandle());
                    saveLinks(links);
                }
                else {
                    saveNotFinishedStatusContributionIdIndexes(indexes, volumeidx);
                }
                result = true;
			}
        return result;
    }
    
    public static ArrayList indexEntry(java.util.HashMap CdmElementsTable, org.w3c.dom.Element myRootElt, org.apache.xml.utils.PrefixResolver myPrefixResolver, String handle)
            throws PapillonBusinessException {
				ArrayList indexes = new ArrayList();				
				
            for (Iterator langKeys = CdmElementsTable.keySet().iterator(); langKeys.hasNext();) {
                String lang = (String) langKeys.next();
                
                java.util.HashMap tmpTable = (java.util.HashMap) CdmElementsTable.get(lang);
                for (Iterator keys = tmpTable.keySet().iterator(); keys.hasNext();) {
                    String CdmElement = (String) keys.next();
      
					//PapillonLogger.writeDebugMsg("Index entry, key " + CdmElement + ":");
                    ArrayList myArrayList = (ArrayList) tmpTable.get(CdmElement);
                    org.apache.xpath.XPath myXPath = null;
                    boolean isIndex = false;
                    if (myArrayList != null) {
						isIndex = ((Boolean) myArrayList.get(1)).booleanValue();
						if (isIndex) {
							myXPath = (org.apache.xpath.XPath) myArrayList.get(2);
 							
							//PapillonLogger.writeDebugMsg("Index entry: lang: "+ lang +" /key: " + CdmElement + " /xpath: " + (String) myArrayList.elementAt(0));
							indexes.addAll(createIndexData (CdmElement, myXPath, 
										  lang, handle,
										  myRootElt,myPrefixResolver
										  ));
						}
					}
                }
            }
        return indexes;
    }
		
	protected static ArrayList createIndexData (String theElement, org.apache.xpath.XPath theXPath, String theLang, String theHandle,
										org.w3c.dom.Element theRootElement,
										org.apache.xml.utils.PrefixResolver thePrefixResolver) 
		throws PapillonBusinessException {
		ArrayList resultData = new ArrayList();
		org.w3c.dom.NodeList resNodeList = null;
		try {
			org.apache.xpath.objects.XObject myXObject = theXPath.execute(
																		 new org.apache.xpath.XPathContext(), theRootElement, thePrefixResolver);
			resNodeList = myXObject.nodelist();
			//PapillonLogger.writeDebugMsg("Index entry, xPath.execute res: ");
		} catch (javax.xml.transform.TransformerException e) {
			throw new PapillonBusinessException("javax.xml.transform.TransformerException: ", e);
		}
		if (resNodeList != null) {
			for (int i = 0; i < resNodeList.getLength(); i++) {
				org.w3c.dom.Node myNode = resNodeList.item(i);
				String value = myNode.getNodeValue();
				//PapillonLogger.writeDebugMsg("Index entry, node " + myNode.getNodeName() + " /value: " + value);
				if (value != null) {
					value = value.trim();
					if (!value.equals("")) {
						//PapillonLogger.writeDebugMsg("Index entry, node value: " + value);
						resultData.add(new IndexData(theElement, theLang, value, theHandle));
					}
				}
			}
		} else {
			//PapillonLogger.writeDebugMsg("Index entry, node list null for CdmElement: " + theElement + ":");
		}
		return resultData;
	}

	public static ArrayList indexEntryLinks(java.util.HashMap linksTable, Volume theVolume, org.w3c.dom.Element theRootElement, String theHandle)
	throws PapillonBusinessException {
		//PapillonLogger.writeDebugMsg("indexEntryLinks on: " + theVolume.getName() + " linksTable.size: " + linksTable.size());
		ArrayList linksArray = new ArrayList();
		String linkDbtableName = theVolume.getLinkDbname();
		org.apache.xml.utils.PrefixResolver  thePrefixResolver = theVolume.getPrefixResolver();
		for (Iterator linksKeys = linksTable.keySet().iterator(); linksKeys.hasNext();) {
			String linkName = (String) linksKeys.next();
			HashMap linkTable = (HashMap) linksTable.get(linkName);
			ArrayList linkElementArrayList = (ArrayList) linkTable.get(Volume.LINK_ELEMENT_TYPE);
			org.apache.xpath.XPath linkElementXPath = (org.apache.xpath.XPath) linkElementArrayList.get(1);
			org.w3c.dom.NodeList resNodeList = null;
			try {
				org.apache.xpath.objects.XObject myXObject = linkElementXPath.execute(new org.apache.xpath.XPathContext(), theRootElement, thePrefixResolver);
				resNodeList = myXObject.nodelist();
			} catch (javax.xml.transform.TransformerException e) {
				throw new PapillonBusinessException("javax.xml.transform.TransformerException: ", e);
			}
			String theLang = Volume.DEFAULT_LANG;
			String newLang = (String) linkTable.get(Volume.LINK_STRING_LANG_TYPE);
			if (newLang != null && !newLang.equals("")) {
				theLang = newLang;
			}

			if (resNodeList != null) {
				for (int i = 0; i < resNodeList.getLength(); i++) {
					org.w3c.dom.Node myNode = resNodeList.item(i);
					//PapillonLogger.writeDebugMsg("Index entry link, Node " + myNode.getNodeName());
					Link myLink = LinkFactory.newLink(linkDbtableName, linkName, theLang, theHandle);
					
					for (Iterator linkKeys = linkTable.keySet().iterator(); linkKeys.hasNext();) {
						String linkType = (String) linkKeys.next();
						if (linkType != Volume.LINK_STRING_LANG_TYPE && linkType != Volume.LINK_STRING_LANG_TYPE) {
						ArrayList linkArrayList = (ArrayList) linkTable.get(linkType);
						org.apache.xpath.XPath linkXPath = (org.apache.xpath.XPath) linkArrayList.get(1);
						org.w3c.dom.NodeList linkNodeList = null;
						try {
							org.apache.xpath.objects.XObject myXObject = linkXPath.execute(
																						  new org.apache.xpath.XPathContext(), myNode, thePrefixResolver);
							linkNodeList = myXObject.nodelist();
							//PapillonLogger.writeDebugMsg("Index entry, xPath.execute res: ");
						} catch (javax.xml.transform.TransformerException e) {
							throw new PapillonBusinessException("javax.xml.transform.TransformerException: ", e);
						}
						if (linkNodeList != null && linkNodeList.getLength()>0) {
							org.w3c.dom.Node typeNode = linkNodeList.item(0);
							String value = typeNode.getNodeValue();
							//PapillonLogger.writeDebugMsg("Index entry link, node " + typeNode.getNodeName() + " /value: " + value);
							if (value != null) {
								value = value.trim();
								if (!value.equals("")) {
									//PapillonLogger.writeDebugMsg("Index entry link, node value: " + value);
									if (linkType.equals(Volume.CDM_linkValue)) {
										myLink.setTargetId(value);
									}	
									else if (linkType.equals(Volume.CDM_linkVolume)) {
										myLink.setVolumeTarget(value);
									}
									else if (linkType.equals(Volume.CDM_linkWeight)) {
										myLink.setWeight(value);
									}
									else if (linkType.equals(Volume.CDM_linkLang)) {
										myLink.setLang(value);
									}									
									else if (linkType.equals(Volume.CDM_linkElementId)) {
										myLink.setElementId(value);
									}									
									else if (linkType.equals(Volume.CDM_linkType)) {
										myLink.setType(value);
									}									
									else if (linkType.equals(Volume.CDM_linkLabel)) {
										myLink.setLabel(value);
									}									
								}
							}
						}
					}
					}
					if (myLink.getVolumeTarget() == null || myLink.getVolumeTarget().equals("")) {
						String targetVolumeName = theVolume.getDefaultTargetVolumeName(myLink.getLang());
						myLink.setVolumeTarget(targetVolumeName);
					}	
					/* Attention! des fois le lien peut être vide et non final! */
					/*if (myLink.getType() == null || myLink.getType().equals("")) {
						myLink.setType(Link.FINAL_TYPE);
					}*/
					if (myLink.getTargetId() != null && !myLink.getTargetId().equals("")) {
						linksArray.add(myLink);
					}
				}
			} 
			else {
				PapillonLogger.writeDebugMsg("Index entry link, node list null for CdmElement: " + linkName);
			}
		}
		return linksArray;
	}
    
    /*
     *decide that the recode should be putted into links table or into index table by CdmElement
     * */
    public static boolean chooseTable(String CdmElement){
    	ArrayList cdmList = new ArrayList(); 
    	if(Volume.linkElements.length>0){
    		for (int i = 0; i<Volume.linkElements.length; i++){
    			cdmList.add(Volume.linkElements[i]);
    		}
    	}
//    	cdmList.add("cdm-translation-ref");
//    	cdmList.add("cdm-links-weight");
    	
    	boolean tableIsLinks = cdmList.contains(CdmElement);   	
    	return tableIsLinks;
    }

    public static org.w3c.dom.NodeList getCdmElements(IAnswer myEntry, String CdmElement)
            throws PapillonBusinessException {
        return getCdmElements(myEntry, CdmElement, Volume.DEFAULT_LANG);
    }

    /**
     * Gets a CDM element of the VolumeEntry
     *
     * @param CdmElement the name of the CDM element as a String.
     * @param lang       the language of the CDM element as a ISO 639-2/T 3 letters String.
     * @return the CDM volume as a String.
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    public static org.w3c.dom.NodeList getCdmElements(IAnswer myEntry, String CdmElement, String lang)
            throws PapillonBusinessException {
//    	PapillonLogger.writeDebugMsg("myEntry = "+myEntry+", CdmElement ="+CdmElement+" , lang = "+lang);  
        return getCdmElements(myEntry.getDom(), CdmElement, lang, myEntry.getVolume().getCdmElements(), myEntry.getVolume().getPrefixResolver());
    }

    public static org.w3c.dom.NodeList getCdmElements(org.w3c.dom.Document myEntryDOM, String CdmElement, String lang,
                                                      HashMap CdmElementsTable, org.apache.xml.utils.PrefixResolver thePrefixResolver)
            throws PapillonBusinessException {
        org.w3c.dom.NodeList resNodeList = null;
        //fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("getCdmElements: " + CdmElement + " " + lang);
        if (lang != null && !lang.equals("") && CdmElementsTable != null) {
            java.util.HashMap tmpTable = (java.util.HashMap) CdmElementsTable.get(lang);
            if (tmpTable != null) {
                ArrayList myArrayList = (ArrayList) tmpTable.get(CdmElement);
                org.apache.xpath.XPath myXPath = null;
                if (myArrayList != null && myArrayList.size() == 3) {
                    myXPath = (org.apache.xpath.XPath) myArrayList.get(2);
					resNodeList = getNodeListFromXPath(myEntryDOM.getDocumentElement(), myXPath, thePrefixResolver);
                 } else {
                    fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("getCdmElements: ArrayList: null for CdmElement: " + CdmElement + " lang: " + lang);
                }
            } else {
                fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("getCdmElements: " + CdmElement + " tmpTable == null for lang: " + lang);
            }
        } else {
            fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("getCdmElements: " + CdmElement + " CdmElementsTable == null");
        }
        return resNodeList;
    }
	
	public static org.w3c.dom.NodeList getNodeListFromXPath(org.w3c.dom.Element myEntryDOM, org.apache.xpath.XPath myXPath, org.apache.xml.utils.PrefixResolver tmpPrefixResolver) 
	throws PapillonBusinessException {
		org.w3c.dom.NodeList resNodeList = null;
		
		if (myXPath != null && myEntryDOM != null) {
			try {
				//PapillonLogger.writeDebugMsg("executexPath: 0" + myArrayList.get(0) + " 1"+ myArrayList.get(1));
				org.apache.xpath.objects.XObject myXObject = myXPath.execute(new org.apache.xpath.XPathContext(), myEntryDOM,tmpPrefixResolver);
				resNodeList = myXObject.nodelist();
			} catch (javax.xml.transform.TransformerException e) {
				throw new PapillonBusinessException("javax.xml.transform.TransformerException: ", e);
			}
		}
		return resNodeList;
	}
	
	public static String getStringFromXPath(org.w3c.dom.Element myEntryDOM, org.apache.xpath.XPath myXPath, org.apache.xml.utils.PrefixResolver tmpPrefixResolver) 
	throws PapillonBusinessException {
		String resultString = null;
		
		if (myXPath != null && myEntryDOM != null) {
			try {
				//PapillonLogger.writeDebugMsg("executexPath: 0" + myArrayList.get(0) + " 1"+ myArrayList.get(1));
				org.apache.xpath.objects.XObject myXObject = myXPath.execute(new org.apache.xpath.XPathContext(), myEntryDOM,tmpPrefixResolver);
				org.w3c.dom.NodeList resNodeList = myXObject.nodelist();
				if (resNodeList != null && resNodeList.getLength() > 0) {
					org.w3c.dom.Node resNode = resNodeList.item(0);
					if (resNode!=null) {
						resultString = resNode.getNodeValue();
					}
				}
			} catch (javax.xml.transform.TransformerException e) {
				throw new PapillonBusinessException("javax.xml.transform.TransformerException: ", e);
			}
		}
		return resultString;
	}
	
    public static org.w3c.dom.Node getCdmElement(IAnswer myEntry, String CdmElement)
            throws PapillonBusinessException {
        return getCdmElement(myEntry, CdmElement, Volume.DEFAULT_LANG);
    }

    public static org.w3c.dom.Node getCdmElement(IAnswer myEntry, String CdmElement, String lang)
            throws PapillonBusinessException {
        org.w3c.dom.Node resNode = null;
        org.w3c.dom.NodeList myList = getCdmElements(myEntry, CdmElement, lang);
        if (myList != null && myList.getLength() > 0) {
            resNode = myList.item(0);
        }
        return resNode;
    }

    public static String getCdmString(IAnswer myEntry, String CdmElement)
            throws PapillonBusinessException {
        return getCdmString(myEntry, CdmElement, Volume.DEFAULT_LANG);
    }

    public static String getCdmString(IAnswer myEntry, String CdmElement, String lang)
            throws PapillonBusinessException {
        String resString = "";
        org.w3c.dom.Node myNode = getCdmElement(myEntry, CdmElement, lang);
        if (myNode != null) {
            resString = myNode.getNodeValue();
        }
		//PapillonLogger.writeDebugMsg("getCdmString: " + CdmElement + " "+ lang + "" + resString);
        return resString;
    }

    public static String[] getCdmStrings(IAnswer myEntry, String CdmElement)
            throws PapillonBusinessException {
        return getCdmStrings(myEntry, CdmElement, Volume.DEFAULT_LANG);
    }

    public static String[] getCdmStrings(IAnswer myEntry, String CdmElement, String lang)
            throws PapillonBusinessException {
        String[] ResStrings = null;
        org.w3c.dom.Node resNode = null;
        org.w3c.dom.NodeList myList = getCdmElements(myEntry, CdmElement, lang);
        if (myList != null && myList.getLength() > 0) {
            ResStrings = new String[myList.getLength()];
            for (int i = 0; i < myList.getLength(); i++) {
                org.w3c.dom.Node tmpNode = myList.item(i);
                if (tmpNode != null) {
                    ResStrings[i] = tmpNode.getNodeValue();
                } else {
                    ResStrings[i] = null;
                }
            }
        }
        return ResStrings;
    }

    public static void setCdmElement(IAnswer myEntry, String CdmElement, String value)
            throws PapillonBusinessException {
        setCdmElement(myEntry, CdmElement, value, Volume.DEFAULT_LANG);
    }

    public static void setCdmElement(IAnswer myEntry, String CdmElement, String value, String lang)
            throws PapillonBusinessException {
      // fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("setCdmElement: " + CdmElement + " value: " + value + " lang:" + lang);
        org.w3c.dom.Node eltNode = getCdmElement(myEntry, CdmElement, lang);
        if (eltNode != null) {
           // PapillonLogger.writeDebugMsg("setCdmElement: eltNode != null, append");
            org.w3c.dom.Node textNode = eltNode.getOwnerDocument().createTextNode(value);
            if (eltNode.hasChildNodes()) {
                org.w3c.dom.NodeList childNodes = eltNode.getChildNodes();
                for (int i = 0; i < childNodes.getLength(); i++) {
                    eltNode.removeChild(childNodes.item(i));
                }
            }
            eltNode.appendChild(textNode);
        }
        else {
         //   PapillonLogger.writeDebugMsg("setCdmElement: eltNode == null");
        }
    }

    public static void initCdmElement(IAnswer myEntry, String CdmElement)
            throws PapillonBusinessException {
        initCdmElement(myEntry, CdmElement, Volume.DEFAULT_LANG);
    }

    public static void initCdmElement(IAnswer myEntry, String CdmElement, String lang)
            throws PapillonBusinessException {
        org.w3c.dom.NodeList myList = getCdmElements(myEntry, CdmElement, lang);
        if (myList != null && myList.getLength() > 0) {
            org.w3c.dom.Node currentEltNode = (org.w3c.dom.Node) myList.item(0);
            org.w3c.dom.Node parentNode = currentEltNode.getParentNode();
            if (currentEltNode.hasChildNodes()) {
                org.w3c.dom.NodeList childNodes = currentEltNode.getChildNodes();
                for (int i = 0; i < childNodes.getLength(); i++) {
                    currentEltNode.removeChild(childNodes.item(i));
                }
            }
            for (int i = 0; i < myList.getLength(); i++) {
                parentNode.removeChild(myList.item(i));
            }
            parentNode.appendChild(currentEltNode);
        }
    }

    public static void addCdmElement(IAnswer myEntry, String CdmElement, String value)
            throws PapillonBusinessException {
        addCdmElement(myEntry, CdmElement, value, Volume.DEFAULT_LANG);
    }

    public static void addCdmElement(IAnswer myEntry, String CdmElement, String value, String lang)
            throws PapillonBusinessException {
        //fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("addCdmElement: " + CdmElement + " value: " + value);
        org.w3c.dom.NodeList myList = getCdmElements(myEntry, CdmElement, lang);
        if (myList != null && myList.getLength() > 0) {
            org.w3c.dom.Node eltNode = (org.w3c.dom.Node) myList.item(0);
            org.w3c.dom.Node currentEltNode = (org.w3c.dom.Node) eltNode.cloneNode(true);
            org.w3c.dom.Node textNode = currentEltNode.getOwnerDocument().createTextNode(value);
            if (currentEltNode.hasChildNodes()) {
                org.w3c.dom.NodeList childNodes = currentEltNode.getChildNodes();
                for (int i = 0; i < childNodes.getLength(); i++) {
                    currentEltNode.removeChild(childNodes.item(i));
                }
            }
            currentEltNode.appendChild(textNode);
            eltNode.getParentNode().appendChild(currentEltNode);
        }
    }

}
