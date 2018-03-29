/*
 *
 *-----------------------
 *$Id$
 *------------------------
 *$Log$
 *
 *------------------------
 */

package fr.imag.clips.papillon.business.motamot;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.IAnswer;
import fr.imag.clips.papillon.business.dictionary.IndexEntry;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.business.xsl.JibikiXsltExtension;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMResult;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.ByteArrayOutputStream;
import java.util.Collection;
import java.util.Iterator;
import java.util.HashMap;
import java.util.Hashtable;


public class MotamotFormatter implements ResultFormatter {

    // Constants
    // Note: I use constants extensively because the XSL transformations are a little
    // bit slow
    protected static Hashtable XslSheetCache = new Hashtable();

    //outils pour les transformation
    protected static final TransformerFactory myTransformerFactory = TransformerFactory.newInstance();
    // preparation du resultat//usine a document duilder
    protected static final DocumentBuilderFactory myDocumentBuilderFactory = DocumentBuilderFactory.newInstance();
    //on cree le constructeur de document
    protected static DocumentBuilder myDocumentBuilder = null;


    protected XslSheet dictXsl = null;

    protected Dictionary currentDictionary;
    protected Volume sourceVolume;
	
	public static String XML_FORMATTER = "XML";

    //
    private final boolean DEBUG = false;

    public void initializeFormatter(Dictionary dictionary, Volume volume, Object parameter, int dialect, String lang) throws PapillonBusinessException {
        try {
            //
            String dictionaryName = dictionary.getName();
            String volumeName = volume.getName();

            //
            if (myDocumentBuilder == null) {
                myDocumentBuilder = myDocumentBuilderFactory.newDocumentBuilder();
            }
            //
            dictXsl = getXslSheet(dictionaryName, volumeName, (String) parameter);
			/*
			 System.out.println("Document Builder Factory is: " + myDocumentBuilderFactory.getClass());
			 System.out.println("Transformer Factory is: " + myTransformerFactory.getClass());
			 System.out.println("Document Builder is: " + myDocumentBuilder.getClass());
			 System.out.println("XSLSheet is: " + dictXsl.getName() + " " + dictXsl.getHandle());
			 */
			
        } catch (javax.xml.parsers.ParserConfigurationException e) {
            throw new PapillonBusinessException("CRITICAL: error initializing document builder !", e);
        }

    }

    public Node getFormattedResult(QueryResult qr, User usr)
            throws PapillonBusinessException {

        Document res = myDocumentBuilder.newDocument();
        Element rootdiv = res.createElement("div");
        rootdiv.setAttribute("class", "entry");
        res.appendChild(rootdiv);
		
		// FIXME: determine what to do for direct translation result:
        // hint, use jibikiXsltExtensions to resolve the links and format everything...
		// for the moment, old school way...
		
                VolumeEntry myAnswer = qr.getSourceEntry();

                Volume myVolume = myAnswer.getVolume();
                Dictionary myDictionary = myAnswer.getDictionary();
				String dictName = myDictionary.getName();
				String sourceLanguage = myVolume.getSourceLanguage();
				Collection targets = myVolume.getTargetLanguagesArray();

				HashMap lexiesTable = qr.getLexiesHashMap();
				java.util.Iterator e = lexiesTable.keySet().iterator();
				while( e.hasNext() ){
					PapillonLogger.writeDebugMsg("LexieshashMap: " + e.next());
				}
				
                for (Iterator iter = targets.iterator(); iter.hasNext();) {
                    String target = (String) iter.next();

					// DIRECT TRANSLATION RESULTS
                    if (target != null && !target.equals("") && target!= sourceLanguage) {
						//PapillonLogger.writeDebugMsg("getFormattedDirectResult: " + target);
                        NodeList myNodeList = IndexEntry.getCdmElements(myAnswer, Volume.CDM_translationReflexie, target);
                        if ((myNodeList != null) && (myNodeList.getLength() > 0)) {
                            for (int i = 0; i < myNodeList.getLength(); i++) {
                                Node myNode = myNodeList.item(i);
                                String translationId = myNode.getNodeValue();
								if (translationId !=null && !translationId.equals("")) {
								if (myNode.getNodeType() == Node.TEXT_NODE) {
                                   Node textNode = myNode;
                                    myNode = myNode.getParentNode();
                                    myNode.removeChild(textNode);
                                }
								if (myNode.getNodeType() == Node.ATTRIBUTE_NODE) {
									if ( myNode.getParentNode()==null) {
										myNode = ((org.w3c.dom.Attr)myNode).getOwnerElement();
									}
									else {
										myNode = myNode.getParentNode();
									}
								}
								PapillonLogger.writeDebugMsg("Node type: " + myNode.getNodeType() + ", name: " + myNode.getNodeName() + " , value: " + myNode.getNodeValue());
                                VolumeEntry newEntry = (VolumeEntry) qr.getLexiesHashMap().get(translationId);
                                if (newEntry != null && !newEntry.isEmpty()) {
									if (target.equals("axi")) {
										for (Iterator itera = targets.iterator(); itera.hasNext();) {
											String targeta = (String)itera.next();
											
											if (targeta != sourceLanguage && !targeta.equals("axi")) {
												// get all cdm elements pointing to target entries.
												String[] transIds = newEntry.getTranslationsLexieIds(targeta);
												
												//
												Collection Volumes = VolumesFactory.getVolumesArray(dictName, targeta, null);
												if (null != transIds && !Volumes.isEmpty()) {
													Volume firstVolume = (Volume)(Volumes.iterator()).next();
													String volumeName = firstVolume.getName();
													for (int j = 0; j < transIds.length; j++) {
														VolumeEntry myEntry = (VolumeEntry) qr.getLexiesHashMap().get(transIds[j]);
														if (myEntry != null && ! myEntry.isEmpty()) {
															Node tempNode = myAnswer.getDom().importNode((Node) myEntry.getDom().getDocumentElement(), true);
															myNode.appendChild(tempNode);
														} 
													}
												}
											}					
										}
									}
									else {
										Node tempNode = myAnswer.getDom().importNode((Node) newEntry.getDom().getDocumentElement(), true);
										myNode.appendChild(tempNode);
									}
                                }
								}
                            }
                        }
                    }
                }
				
				if (null != dictXsl && !dictXsl.isEmpty()) {
                // Format document source
				//PapillonLogger.writeDebugMsg("ResultNode before format: " + qr.getSourceEntry().getHeadword() + " node: " + XMLServices.NodeToString(qr.getSourceEntry().getDom()));
                Node resultNode = formatResult(qr.getSourceEntry().getDom(), dictXsl, usr);
				//PapillonLogger.writeDebugMsg("ResultNode after format: " + qr.getSourceEntry().getHeadword() + " node: " + XMLServices.NodeToString(resultNode));
                rootdiv.appendChild(res.importNode(resultNode, true));
            }
		
       return rootdiv;
    }

    /**
     * ...
     *
     * @param docSource DOM version of the volume entry to be formatted
     * @param xsl       XslSheet used for formatting
     * @param usr       User currently requesting the result.
     * @return Root node representing the
     * @throws fr.imag.clips.papillon.business.PapillonBusinessException
     *
     */
    private Node formatResult(Document docSource, XslSheet xsl, User usr)
            throws PapillonBusinessException {
        try {

            // init
            Document docCible = null;
			
            docCible = Transform(docSource, xsl);

            return docCible.getDocumentElement();

        } catch (Exception ex) {
            throw new PapillonBusinessException("Exception in formatResult()", ex);
        }
    }

    /**
     * Transform the xml source by processing it with an xsl sheet.
     */
    public static Document Transform(Node xmlSource, XslSheet xslSheet)
            throws PapillonBusinessException, javax.xml.parsers.ParserConfigurationException,
            javax.xml.transform.TransformerException, java.io.IOException {

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
        if (myDocumentBuilder == null) {
            myDocumentBuilder = myDocumentBuilderFactory.newDocumentBuilder();
        }
        Document newDocument = myDocumentBuilder.newDocument();

        //
        myTransformer.transform(new DOMSource(xmlSource), new DOMResult(newDocument));

        //
        return newDocument;
    }

     public static XslSheet getXslSheet(String dictionaryName, String volumeName, String name)
            throws PapillonBusinessException {

        XslSheet theXsl = null;

        // find specific xsl
        theXsl = XslSheetFactory.getXslSheet(dictionaryName, volumeName, name);

        // find specific xsl
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet(dictionaryName, "", name);
        }

        // find specific xsl
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet("", "", name);
        }

        // find defaut
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet(dictionaryName, volumeName, "");
        }

        // find defaut
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet(dictionaryName, "", "");
        }

        // find defaut
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet("", "", "");
        }

        //
        return theXsl;
    }

}


















