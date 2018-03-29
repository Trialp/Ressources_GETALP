/*
 *
 *-----------------------
 * $Id$
 *------------------------
 * $Log$
 *
 *------------------------
 */

package fr.imag.clips.papillon.business.pivax;

import java.util.Hashtable;
import java.util.Iterator;
import java.util.Collection;
import java.util.ArrayList;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMResult;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Text;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.dictionary.IAnswer;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.xsl.JibikiXsltExtension;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.dictionary.QueryRequest;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.dictionary.QueryCriteria;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;
//import fr.imag.clips.papillon.presentation.Quick;


public class PivaxFormatter implements ResultFormatter {
    
    
	// Constants
	// Note: I use constants extensively because the XSL transformations are a little
	// bit slow
	protected static Hashtable XslSheetCache = new Hashtable();
    
    //outils pour les transformation
	protected static final TransformerFactory myTransformerFactory = TransformerFactory.newInstance();
	protected static final DocumentBuilderFactory myDocumentBuilderFactory = DocumentBuilderFactory.newInstance();
	protected static DocumentBuilder myDocumentBuilder; 	
    
    protected XslSheet dictXsl = null;
    
    protected Dictionary currentDictionary;
    protected Volume sourceVolume;
    
    //
    private final boolean DEBUG = false;
    
    //
    public void initializeFormatter(Dictionary dictionary, Volume volume, Object parameter, int dialect, String lang) throws PapillonBusinessException {
        
        try {
            
            //
            String dictionaryName = dictionary.getName();
            String volumeName = volume.getName();
            
            //
            if (myDocumentBuilder==null) {
                myDocumentBuilder = myDocumentBuilderFactory.newDocumentBuilder();
            }
            
            //
            dictXsl = getXslSheet(dictionaryName, volumeName, (String) parameter);
            
            //PapillonLogger.writeDebugMsg("PivaxFormatter.initializeFormatter() dictXsl.getName(): " + dictXsl.getName());
			
        } catch (javax.xml.parsers.ParserConfigurationException e) {
            throw new PapillonBusinessException("CRITICAL: error initializing document builder !", e);
        }
        
    }
    
    
    //
    public Node getFormattedResult(QueryResult qr, User usr) throws PapillonBusinessException {
        try {
            //
            if (DEBUG) PapillonLogger.writeDebugMsg("PivaxFormater : begin getFormattedResult");
                        
			// Get document source
            Document docSource = qr.getSourceEntry().getDom();
            // Element handler = docSource.createElement("handler");
            // handler.setAttribute("handler", String.valueOf(qr.getSourceEntry().getHandle()));
            // docSource.appendChild(handler);
			
            // Create document result
            Document resultDoc = myDocumentBuilder.newDocument();
            Element div = resultDoc.createElement("div");
            resultDoc.appendChild(div);
            
            if (null != dictXsl && !dictXsl.isEmpty()) {
				
                // PapillonLogger.writeDebugMsg("PivaxFormatter.getFormattedResult dictXsl.getCode(): " + dictXsl.getCode());
                // PapillonLogger.writeDebugMsg("PivaxFormatter.getFormattedResult qr.getSourceEntry().getHeadword(): " + qr.getSourceEntry().getHeadword());
                
                // Format document source
                Node resultNode = formatResult(docSource, dictXsl, usr);
                // Node srcEntry = addButtons(resultNode, qr.getSourceEntry());
                div.appendChild(resultDoc.importNode(resultNode, true));
            }

            // Add 
            // FIXME : suppress, find another solution ()
            if (qr.getResultKind() == QueryResult.AXIE_COLLECTION_RESULT) {
            	
            	// PapillonLogger.writeDebugMsg("PivaxFormatter.getFormattedResult qr.getLexiesCollection(): " + qr.getLexiesCollection().size());
				//Element hr = resultDoc.createElement("hr");
				//div.appendChild(hr);

                // Then append each translation
                Iterator iter = qr.getLexiesCollection().iterator();
                while (iter.hasNext()) {
                    VolumeEntry ve = (VolumeEntry) iter.next();
                    
                    // PapillonLogger.writeDebugMsg("PivaxFormatter.getFormattedResult target " + ve.getHeadword());
                    //if (!ve.getHandle().equals(qr.getSourceEntry().getHandle()) ) { //don't show reverse
					if (!ve.getSourceLanguage().equals(qr.getSourceEntry().getSourceLanguage()) ) { //don't show source language entries
                        Document doc = ve.getDom();
                        
                        if (null != dictXsl && ! dictXsl.isEmpty()) {
                            
                        	// PapillonLogger.writeDebugMsg("docSourceLexieRef: " + dictXsl.getName() + " " + XMLServices.xmlCodePrettyPrinted(doc));
                            Element resultNode = (Element) formatResult(doc, dictXsl, usr);
                            resultNode.setAttribute("class", "translation");
                            
                            div.appendChild(resultDoc.importNode(resultNode, true));
                        } else
                        	PapillonLogger.writeDebugMsg("PivaxFormatter.getFormattedResult dictXsl is NULL");
                    }
                }
            }            
 			
            //
            if (DEBUG) PapillonLogger.writeDebugMsg("PivaxFormater : end getFormattedResult");
            
            // PapillonLogger.writeDebugMsg("docSourceRoot: " + XMLServices.xmlCodePrettyPrinted(res));
            
            return (Node) resultDoc.getDocumentElement();
            
        } catch(Exception ex) {
            // throw new PapillonBusinessException("Exception in PIVAX getFormattedResult()", ex);
        	return null;
        }	
    }
    
    
    /**
	 ...
     */
    private Node formatResult(Document docSource, XslSheet xsl, User usr) throws PapillonBusinessException {
        try {
        	//Test
        	// PapillonLogger.writeDebugMsg(xsl.getName());
        	// PapillonLogger.writeDebugMsg(XMLServices.xmlCodePrettyPrinted(docSource));
        	
            // init
            boolean isTransform = true;
            Document docCible = null;
            
            docCible = Transform(docSource, xsl);
            /*
			 while (isTransform) {
			 
			 // Transform
			 docCible = Transform(docSource, xsl);
			 
			 // Replace FindReference element by ResultReference
			 // attributes :  termReference, lang, xpath, xslName, dictionaryName, volumeName, ...
			 NodeList list = docCible.getElementsByTagName("FindReference");
			 while (list.getLength() > 0) {
			 Node node = list.item(0);
			 Node parentNode = node.getParentNode();
			 
			 
			 //
			 Node termRefAttribut = node.getAttributes().getNamedItem("termReference");
			 String termRef = termRefAttribut.getNodeValue();
			 if (DEBUG) PapillonLogger.writeDebugMsg("PivaxFormatter: " + termRef);
			 Node langAttribut = node.getAttributes().getNamedItem("lang");
			 String lang = langAttribut.getNodeValue();
			 if (DEBUG) PapillonLogger.writeDebugMsg("PivaxFormatter: " + lang);
			 Node xpathAttribut = node.getAttributes().getNamedItem("xpath");
			 String xpath = xpathAttribut.getNodeValue();
			 if (DEBUG) PapillonLogger.writeDebugMsg("PivaxFormatter: " + xpath);
			 Node xslNameAttribut = node.getAttributes().getNamedItem("xslName");
			 Node dictionaryNameAttribut = node.getAttributes().getNamedItem("dictionaryName");
			 Node volumeNameAttribut = node.getAttributes().getNamedItem("volumeName");
			 //Node namespaceAttribut = node.getAttributes().getNamedItem("namespace");
			 
			 //
			 String xslName = "";
			 if ( xslNameAttribut != null ) { xslName = xslNameAttribut.getNodeValue(); }
			 
			 //
			 String dictionaryName = "";
			 if ( dictionaryNameAttribut != null ) { dictionaryName = dictionaryNameAttribut.getNodeValue(); }
			 
			 //
			 String volumeName = "";
			 if ( volumeNameAttribut != null ) { volumeName = volumeNameAttribut.getNodeValue(); }
			 
			 //FIXME: search the specified entry, then gets its dictionary and volume to correctly select the XSL
			 //newXsl = XslSheetFactory.findXslSheetByName(xslNameAttribut.getNodeValue());
			 XslSheet newXsl = XslSheetFactory.getXslSheet(dictionaryName, volumeName, xslName);
			 //System.out.println("XslSheetFactory.getXslSheet(" + dictionaryName + ", " + volumeName + ", " + xslName + ") -> " + xpath);
			 
			 
			 
			 // Find entry
			 // FIXME : Change DB to do quick entry id search !
			 QueryRequest queryReq = new QueryRequest(VolumesFactory.getVolumesArray(null, lang, null));
			 QueryCriteria criteria = new QueryCriteria();
			 criteria.add("key", QueryCriteria.EQUAL, Volume.CDM_entryId);
			 criteria.add("value", QueryCriteria.EQUAL, termRef);     // Termref is an entryid
			 queryReq.addCriteria(criteria);
			 
			 ArrayList listStatus = new ArrayList();
			 QueryCriteria criteriaFinishedStatus = new QueryCriteria();
			 criteriaFinishedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);  
			 criteriaFinishedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.FINISHED_STATUS);
			 listStatus.add(criteriaFinishedStatus);
			 QueryCriteria criteriaModifiedStatus = new QueryCriteria();
			 criteriaModifiedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);
			 criteriaModifiedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.MODIFIED_STATUS);
			 listStatus.add(criteriaModifiedStatus);
			 queryReq.addOrCriteriaList(listStatus);
			 
			 //
			 Collection qrset = queryReq.findLexie(usr);
			 
			 //
			 Iterator iter = qrset.iterator();
			 while(iter.hasNext()) {
			 QueryResult relatedQr = (QueryResult) iter.next();
			 
			 // Find nodes with xpath
			 NodeList nodeL = relatedQr.getSourceEntry().getNodes(xpath);
			 
			 // Insert new nodes
			 for (int j=0; j <  nodeL.getLength(); j++) {
			 Document docXpath = myDocumentBuilder.newDocument();
			 Element result = docXpath.createElement("ResultReference");
			 docXpath.appendChild(result);                        
			 Node nodeXpath = docXpath.importNode(nodeL.item(j), true);
			 result.appendChild(nodeXpath);
			 
			 //
			 if ( newXsl != null ) {
			 result = (Element) formatResult(docXpath, newXsl, usr);
			 }
			 
			 //
			 if (DEBUG) PapillonLogger.writeDebugMsg("PivaxFormatter: " +  fr.imag.clips.papillon.business.xml.XMLServices.NodeToString(result));
			 parentNode.insertBefore(docCible.importNode(result, true), node);
			 }
			 }
			 
			 //
			 parentNode.removeChild(node);
			 
			 // Here because Res change after removeChild and appendChild methods
			 list = docCible.getElementsByTagName("FindReference");
			 }
			 
			 
			 // Manage xsl cascading !!
			 NodeList nextXslList = docCible.getElementsByTagName("nextXsl"); 
			 /// fixme : id !!! supress ... not in xsl file !
			 
			 if (nextXslList.getLength() > 0) {
			 Node nextXsl = nextXslList.item(0);
			 
			 //
			 Node nextXslNameAttribut = nextXsl.getAttributes().getNamedItem("xslName");
			 Node nextDictionaryNameAttribut = nextXsl.getAttributes().getNamedItem("dictionaryName");
			 Node nextVolumeNameAttribut = nextXsl.getAttributes().getNamedItem("volumeName");
			 
			 //
			 String nextXslName = "";
			 if ( nextXslNameAttribut != null ) { nextXslName = nextXslNameAttribut.getNodeValue(); }
			 
			 //
			 String nextDictionaryName = "";
			 if ( nextDictionaryNameAttribut != null ) { nextDictionaryName = nextDictionaryNameAttribut.getNodeValue(); }
			 
			 //
			 String nextVolumeName = "";
			 if ( nextVolumeNameAttribut != null ) { nextVolumeName = nextVolumeNameAttribut.getNodeValue(); }
			 
			 //
			 xsl = XslSheetFactory.getXslSheet(nextDictionaryName, nextVolumeName, nextXslName);
			 isTransform = true;
			 docSource = docCible;
			 
			 // erase node nextXsl !!!!
			 
			 } else {
			 isTransform = false;
			 }
			 
			 
			 }
			 
			 /*
			 // Manage entry actions
			 Element actionElement = docCible.getElementById("actions");
			 if (true && (actionElement!=null)) {
			 // Replace actions
			 
			 
			 } else if (actionElement!=null) {
			 // Delete actions ...
			 
			 actionElement.getParentNode().removeChild(actionElement);
			 
			 }
			 */
            
            //
            return docCible.getDocumentElement();
			
        } catch(Exception ex) {
            throw new PapillonBusinessException("Exception in PIVAX getFormattedResult()", ex);
        }	
    }    
    
    
	/**
	 Transform the xml source by processing it with an xsl sheet.
	 */
    protected static Document Transform(Node xmlSource, XslSheet xslSheet)
	throws fr.imag.clips.papillon.business.PapillonBusinessException,
	javax.xml.transform.TransformerConfigurationException,
	javax.xml.parsers.ParserConfigurationException,
	javax.xml.transform.TransformerException,
	java.io.UnsupportedEncodingException,
	java.io.IOException {
		
		// Find transformer
		// FIXME: if transformer is null ? 
		Transformer myTransformer = xslSheet.getTransformer();
		
		// PapillonLogger.writeDebugMsg("Class: " + myTransformer.getClass());
		// PapillonLogger.writeDebugMsg("Factory: " + TransformerFactory.class.toString());
		// The xalan transformer uses the context class loader instead of the system class loader
		// But in enhydra 5.1, the system class loader is a custom class loader, hence we have
		// to set the context class loader to the enhydra classloader, or the Extension classes
		// would not be visible.
		try {
			Thread currentThread = Thread.currentThread();
			currentThread.setContextClassLoader(JibikiXsltExtension.class.getClassLoader());
		} catch (SecurityException e) {
			PapillonLogger.writeDebugMsg("Could not modify the context class loader.");
		}
		
		// The source
		if (myDocumentBuilder==null) {
			myDocumentBuilder = myDocumentBuilderFactory.newDocumentBuilder();
		}
		Document newDocument= myDocumentBuilder.newDocument();
		
		//
		myTransformer.transform (new DOMSource(xmlSource), new DOMResult(newDocument));
		
		//
		return newDocument;
	}
	
	
	/** 
	 * Return the xsl sheet corresponding to the name, dictionary name and volumename
	 *
	 * @param String
	 * @param String
	 * @param String
	 *
	 * @return XslSheet
	 *     
	 * @exception PapillonBusinessException
	 */    
	public static XslSheet getXslSheet(String dictionaryName, String volumeName, String Name) 
	throws PapillonBusinessException {
		
		XslSheet theXsl = null;
		
		// find specific xsl
		
		// PapillonLogger.writeDebugMsg("PivaxFormatter.getXslSheet() " + dictionaryName + " " + volumeName + " " + Name);
		theXsl = XslSheetFactory.getXslSheet(dictionaryName, volumeName, Name);
		
		// find specific xsl
		if (theXsl == null) {
			theXsl =  XslSheetFactory.getXslSheet(dictionaryName, "", Name);
		}
		
		// find specific xsl
		if (theXsl == null) {
			theXsl =  XslSheetFactory.getXslSheet("", "", Name);
		}    
		
		// find defaut
		if (theXsl == null) {
			theXsl =  XslSheetFactory.getXslSheet(dictionaryName, volumeName, "");
		}
		
		// find defaut
		if (theXsl == null) {
			theXsl =  XslSheetFactory.getXslSheet(dictionaryName, "", "");
		}
		
		
		// find defaut
		if (theXsl == null) {
			theXsl =  XslSheetFactory.getXslSheet("", "", "");
		}
		
		if (theXsl == null)
			PapillonLogger.writeDebugMsg("PivaxFormatter.getXslSheet() XslSheet is NULL");
		//
		return theXsl;
	}
	
	/**
	 * 
	 * @param entryNode: la presentation HTML d'une entree
	 * @param entry: Contenue d'entree
	 * @return Node racine de la presentation HTML d'un tableau contenant la presentation d'entree avec les buttons a manipuler 
	 */
	private Node addButtons(Node entryNode, VolumeEntry entry) throws 
	fr.imag.clips.papillon.business.PapillonBusinessException {
		String tableau = "<table class=\"tbl_entry\">";
		tableau += "<tr><td></td><td></td></tr>";
		tableau += "</table>";
		Document doc = null;
		doc = XMLServices.buildDOMTree(tableau);
		Element tdContent = (Element) doc.getElementsByTagName("td").item(0);
		Element tdButtons = (Element) tdContent.getNextSibling();
		
		tdContent.appendChild(doc.importNode(entryNode, true));
		//tdButtons.appendChild(doc.importNode(new Quick().getButtons(entry), true));
		
		return doc.getElementsByTagName("table").item(0);
	}
}



















