/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 *
 */

package fr.imag.clips.papillon.business.dictionary;

import com.lutris.appserver.server.sql.DBTransaction;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.edition.GenerateTemplate;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.data.VolumeDO;
import fr.imag.clips.papillon.data.VolumeQuery;
import org.w3c.dom.*;

import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.dom.DOMResult;
import javax.xml.transform.stream.StreamSource;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Enumeration;
import java.io.StringReader;

/**
 * Used to find the instances of xslsheet.
 */
public class VolumesFactory {

    protected final static String DML_URI = DmlPrefixResolver.DML_URI;
    protected final static String XLINK_URI = DmlPrefixResolver.XLINK_URI;
    protected final static String XSLT_URI = DmlPrefixResolver.XSLT_URI;
    public final static String XMLSCHEMA_URI = DmlPrefixResolver.XMLSCHEMA_URI;

    public final static String VOLUME_FILES_TAG = "volume-metadata-files";
    protected final static String VOLUME_TAG = "volume-metadata";
    protected final static String CDM_ELEMENTS_TAG = "cdm-elements";
    protected final static String LINKS_TAG = "links";
    protected final static String XSLSHEET_REF_TAG = "xsl-stylesheet";
    protected final static String XMLSCHEMA_REF_TAG = "xmlschema-ref";
    protected final static String TEMPLATE_INTERFACE_REF_TAG = "template-interface-ref";
    protected final static String TEMPLATE_ENTRY_REF_TAG = "template-entry-ref";
    protected final static String XSLSHEET_TAG = "stylesheet";
    protected final static String XMLSCHEMA_TAG = "schema";
    public final static String TEMPLATE_INTERFACE_TAG = "template-interface";
    public final static String TEMPLATE_ENTRY_TAG = "template-entry";
    protected final static String XML_FOOTER_TAG = "volume-xml-footer";

    protected final static String HREF_ATTRIBUTE = "href";
    protected final static String INDEX_ATTRIBUTE = "index";
    protected final static String LANG_ATTRIBUTE = "lang";
    protected final static String LOCATION_ATTRIBUTE = "location";
    protected final static String VIRTUAL_ATTRIBUTE = "virtual";
    protected final static String XPATH_ATTRIBUTE = "xpath";
    protected final static String NAME_ATTRIBUTE = "name";
    protected final static String DEFAULT_ATTRIBUTE = "default";
    protected final static String EXTERNAL_ATTRIBUTE = "external";
	
	public final static String XMLNAMESPACE = "xmlns";
	public final static String DEFAULT_DML_PREFIX = "d";

    /**
     * The initializeVolumeCache method performs a database query to
     * initialize the volume cache
     *
     * @throws PapillonBusinessException If there is a problem retrieving disc information.
     */
    public static void initializeVolumeCache()
            throws PapillonBusinessException {

        try {

            // Initialize cache
            VolumeCache.volumeCacheInit();

            // Perform query
            VolumeQuery query = new VolumeQuery(CurrentDBTransaction.get());
            query.addOrderByName(true);
            VolumeDO[] DOarray = query.getDOArray();

            // Add volumes in cache (keys are the volume names)
            for (int i = 0; i < DOarray.length; i++) {
                Volume vol = new Volume(DOarray[i]);
				updateCdmElementsAndLinksTables(vol);
                VolumeCache.putVolumeInCache(vol.getName(), vol);
				VolumeEntriesFactory.putNbEntriesNumberInCache(vol.getName(),vol.getEntries());
            }

        } catch (Exception ex) {
            throw new PapillonBusinessException("Exception in initializeVolumeCache()", ex);
        }
    }


    /**
     * The newVolume method create a new volume base on volume element into the metadata file
     *
     * @param dictname
     * @param volume   : dictionary element into the metadata file
     * @param fileURL
     * @return Volume
     * @throws PapillonBusinessException, IOException
     */
    public static Volume newVolume(String dictname, Element volume, URL fileURL)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        return newVolume(dictname, volume, fileURL, "","","");
    }
    
    public static Volume newVolume(String dictname, Element volume, URL fileURL, String schema, String tmplEntry, String tmplInterface)
            throws fr.imag.clips.papillon.business.PapillonBusinessException {

				String name = volume.getAttribute("name");
				if (name == null || name.equals("")) {
					throw new PapillonBusinessException("Exception in newVolume: there is no volume name, there is no 'name' attribute  for the 'volume-metadata' element or it is empty!");
				}
				String dbname = volume.getAttribute("dbname");
				String location = volume.getAttribute(LOCATION_ATTRIBUTE);
				String source = volume.getAttribute("source-language");
				if (source == null || source.equals("")) {
					throw new PapillonBusinessException("Exception in newVolume: there is no volume source, there is no 'source' attribute  for the 'volume-metadata' element or it is empty!");
				}
				String targets = volume.getAttribute("target-languages");
				String reverseLookup = volume.getAttribute("reverse-lookup");
				boolean reverse = (reverseLookup != null && reverseLookup.equals("true"));
				
				if (null == dbname || dbname.equals("")) {
					dbname = name;
				}
				
				NodeList refNodes = volume.getElementsByTagNameNS(DML_URI,"volume-ref");
				String href = "";
				if (null != refNodes && refNodes.getLength() > 0) {
					Element tempElt = (Element) refNodes.item(0);
					href = tempElt.getAttributeNS(XLINK_URI, "href");
				} else {
					PapillonLogger.writeDebugMsg("No volume-ref");
				}
                if (schema == null || schema.equals("")) {
                    schema = getXmlCode(volume, XMLSCHEMA_REF_TAG, fileURL);
                }
                if (tmplInterface == null || tmplInterface.equals("")) {
                    tmplInterface = getXmlCode(volume, TEMPLATE_INTERFACE_REF_TAG, fileURL);
                }
                if (tmplEntry == null || tmplEntry.equals("")) {
                    tmplEntry = getXmlCode(volume, TEMPLATE_ENTRY_REF_TAG, fileURL);
                }
 				
		if (tmplEntry == null || tmplEntry.equals("")) {
			throw new PapillonBusinessException("Exception in newVolume: there is no template entry, the element " + TEMPLATE_ENTRY_REF_TAG + " does not exist or is empty!");
		}
		else {
			tmplEntry = addDmlUrlInTemplateEntry(tmplEntry);
		}
		upgradeCdmLinkElement(volume.getOwnerDocument());
		
      //  PapillonLogger.writeDebugMsg("Call createCdmElementsTable for volume " + name);
                HashMap cdmElements = createCdmElementsTable(volume, source, tmplEntry);
				
				HashMap linksTable = createLinksTable(volume, tmplEntry, cdmElements);
				
		PapillonLogger.writeDebugMsg("The CDM elements and links hashtables for volume " + name + " have been created");
				
        // Embedding the entry into a contribution element
        tmplEntry = updateTemplateEntry(tmplEntry, cdmElements);
		PapillonLogger.writeDebugMsg("The template entry has been updated");

        String xmlCode = XMLServices.NodeToString(volume);

        //
        Volume newVolume = createUniqueVolume(name, dictname, dbname, location, source, targets, href, cdmElements,
                linksTable, xmlCode, schema, tmplInterface, tmplEntry, reverse);

        // Add in cache
        VolumeCache.putVolumeInCache(newVolume.getName(), newVolume);

        // reset Available languages so that they are correctly computed afterwards
        AvailableLanguages.resetCache();

        //
        return newVolume;
    }
	
	protected static void updateCdmElementsAndLinksTables(Volume theVolume) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
	
	Document docXml = XMLServices.buildDOMTree(theVolume.getXmlCode());
	
	//on recupere le dictionnaire
	Element volume;
	boolean virtual = false;
	volume = (Element) docXml.getElementsByTagNameNS(DML_URI,VOLUME_TAG).item(0);
	if (volume != null) {
   //     PapillonLogger.writeDebugMsg("Call createCdmElementsTable for volume " + theVolume.getName());
	HashMap cdmElements = createCdmElementsTable(volume, theVolume.getSourceLanguage(),theVolume.getTemplateEntry());
		theVolume.setCdmElements(cdmElements);
		HashMap linksTable = createLinksTable(volume, theVolume.getTemplateEntry(), cdmElements);
		theVolume.setLinksTable(linksTable);
	}
	}
		
    public static org.apache.xpath.XPath compileXPath(String xpathString,
                                                      org.apache.xml.utils.PrefixResolver aPrefixResolver)
	throws PapillonBusinessException {
        javax.xml.transform.SourceLocator mySourceLocator = new org.apache.xml.utils.SAXSourceLocator();
        org.apache.xpath.XPath myXPath = null;
        try {
            myXPath = new org.apache.xpath.XPath(xpathString, mySourceLocator, aPrefixResolver,
												 org.apache.xpath.XPath.SELECT);
        } catch (javax.xml.transform.TransformerException e) {
            throw new PapillonBusinessException("javax.xml.transform.TransformerException with XPath "+xpathString+" : ", e);
        }
        return myXPath;
    }
	
    /* cdmElements HashMap = {lang => HashMap} = {CDM_element => ArrayList} = (xpathString, isIndex, XPath)*/
    protected static boolean compileXPathTable(java.util.HashMap cdmElements, org.apache.xml.utils.PrefixResolver aPrefixResolver)
	throws PapillonBusinessException {
        boolean result = false;
        for (Iterator langKeys = cdmElements.keySet().iterator(); langKeys.hasNext();) {
            String lang = (String) langKeys.next();
            java.util.HashMap tmpTable = (java.util.HashMap) cdmElements.get(lang);
            for (Iterator keys = tmpTable.keySet().iterator(); keys.hasNext();) {
                String CDM_element = (String) keys.next();
                ArrayList eltArrayList = (ArrayList) tmpTable.get(CDM_element);
                if (eltArrayList != null && eltArrayList.size() == 2) {
                    String xpathString = (String) eltArrayList.get(0);
                    org.apache.xpath.XPath myXPath = compileXPath(xpathString, aPrefixResolver);
                    if (myXPath != null) {
						 //PapillonLogger.writeDebugMsg("compileXPathTable: CDM element: " + CDM_element + " xpath not null: " + xpathString);
                        eltArrayList.add(myXPath);
                        result = true;
                    } else {
						//PapillonLogger.writeDebugMsg("compileXPathTable: CDM element: " + CDM_element + " xpath null: " + xpathString);
                    }
                }
            }
        }
        return result;
    }
	
	
	protected static HashMap createCdmElementsTable(Element volume, String source, String tmplEntry) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		HashMap cdmElements = null;
		
        // Cette méthode dépend du schéma des volumes.
 		
		if (volume == null) {
			throw new PapillonBusinessException("Error in createCdmElementsTable: the volume element is null!");
		}
		if (source == null || source.equals("")) {
			throw new PapillonBusinessException("Error in createCdmElementsTable: there is no source language!");
		}
		if (tmplEntry == null || tmplEntry.equals("")) {
			throw new PapillonBusinessException("Error in createCdmElementsTable: there is no template entry!");
		}
		
        org.w3c.dom.Document myDoc = XMLServices.buildDOMTree(tmplEntry);
        String dml_prefix = "";
		if (myDoc == null) {
           PapillonLogger.writeDebugMsg("Error in createCdmElementsTable: the template entry doc is empty!");
			throw new PapillonBusinessException("Error in createCdmElementsTable: the template entry doc is empty!");
		}
		else {
			dml_prefix = getDmlPrefix(myDoc.getDocumentElement());
		}
		
        NodeList cdmElts = volume.getElementsByTagNameNS(DML_URI,CDM_ELEMENTS_TAG);
        if (null != cdmElts && cdmElts.getLength() > 0) {
            Element cdmElt = (Element) cdmElts.item(0);
            cdmElements = buildCdmElementsTable(cdmElt, source, dml_prefix);
        } else {
            PapillonLogger.writeDebugMsg("No " + CDM_ELEMENTS_TAG + " tag");
			throw new PapillonBusinessException("Error in createCdmElementsTable: there is no " + CDM_ELEMENTS_TAG + " tag!");
        }
				
		org.apache.xml.utils.PrefixResolver tmplPrefixResolver = new org.apache.xml.utils.PrefixResolverDefault(myDoc.getDocumentElement());
		compileXPathTable(cdmElements, tmplPrefixResolver);
		return cdmElements;
	}

    /* linksTable HashMap = {name => HashMap} = {element => ArrayList} = (xpathString, XPath)*/
    protected static boolean compileLinksXPathTable(java.util.HashMap linksTable, org.apache.xml.utils.PrefixResolver aPrefixResolver)
	throws PapillonBusinessException {
        boolean result = false;
        for (Iterator nameKeys = linksTable.keySet().iterator(); nameKeys.hasNext();) {
            String name = (String) nameKeys.next();
            java.util.HashMap tmpTable = (java.util.HashMap) linksTable.get(name);
            for (Iterator keys = tmpTable.keySet().iterator(); keys.hasNext();) {
                String element = (String) keys.next();
				if (!element.equals(Volume.LINK_STRING_LANG_TYPE)) {
                ArrayList eltArrayList = (ArrayList) tmpTable.get(element);
                if (eltArrayList != null && eltArrayList.size() == 1) {
                    String xpathString = (String) eltArrayList.get(0);
                    org.apache.xpath.XPath myXPath = compileXPath(xpathString, aPrefixResolver);
                    if (myXPath != null) {
						 //PapillonLogger.writeDebugMsg("compileLinksXPathTable: CDM element: " + element + " xpath not null: " + xpathString);
                        eltArrayList.add(myXPath);
                        result = true;
                    } else {
						//PapillonLogger.writeDebugMsg("compileXPathTable: CDM element: " + element + " xpath null: " + xpathString);
                    }
                }
				}
            }
        }
        return result;
    }	
	

	protected static HashMap createLinksTable(Element volume, String tmplEntry, HashMap cdmElements) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		HashMap linksTable = null;

		
        // Cette méthode dépend du schéma des volumes.
 		
		if (volume == null) {
			throw new PapillonBusinessException("Error in createLinksTable: the volume element is null!");
		}
		if (tmplEntry == null || tmplEntry.equals("")) {
			throw new PapillonBusinessException("Error in createLinksTable: there is no template entry!");
		}

		org.w3c.dom.Document myDoc = XMLServices.buildDOMTree(tmplEntry);
		String dml_prefix = "";
        if (myDoc != null) {
			dml_prefix = getDmlPrefix(myDoc.getDocumentElement());
		}
		else {
            PapillonLogger.writeDebugMsg("Error in createLinksTable: the template entry doc is empty!");
			throw new PapillonBusinessException("Error in createLinksTable: the template entry doc is empty!");
		}
		NodeList cdmElts = volume.getElementsByTagNameNS(DML_URI,CDM_ELEMENTS_TAG);
        if (null != cdmElts && cdmElts.getLength() > 0) {
            Element cdmElt = (Element) cdmElts.item(0);
            linksTable = buildLinksTable(cdmElt, cdmElements, dml_prefix);
        } else {
            PapillonLogger.writeDebugMsg("No " + CDM_ELEMENTS_TAG + " tag");
			throw new PapillonBusinessException("Error in createLinksTable: there is no " + CDM_ELEMENTS_TAG + " tag!");
        }
		
		org.apache.xml.utils.PrefixResolver tmplPrefixResolver = new org.apache.xml.utils.PrefixResolverDefault(myDoc.getDocumentElement());
		compileLinksXPathTable(linksTable, tmplPrefixResolver);
		return linksTable;
	}
	
	

    /**
     * Create an unique volume
     *
     * @param String
     * @param String
     * @param String
     * @param String
     * @param String
     * @param String
     * @param String
     * @param HashMap
     * @param String
     * @param String
     * @param String
     * @param String
     * @param boolean
     * @return Volume
     * @throws PapillonBusinessException
     */
    protected static Volume createUniqueVolume(String name, String dictname, String dbname, String location,
                                               String source, String targets, String href,
                                               java.util.HashMap cdmElements, java.util.HashMap linksTable, String xmlCode, String xmlSchema,
                                               String tmplInterface, String tmplEntry, boolean reverse)
            throws fr.imag.clips.papillon.business.PapillonBusinessException {
        Volume myVolume = null;
        if ((name != null) && (dictname != null) && (dbname != null) && (source != null) && (href != null) && (xmlCode != null)) {
            //search for an existing dictionary
            myVolume = VolumesFactory.getVolumeByName(name);
            if (myVolume == null || myVolume.isEmpty()) {
                //does'nt exist, create :
                myVolume = new Volume();
                myVolume.setName(name);
                myVolume.setDictname(dictname);
                myVolume.setDbname(dbname);
                myVolume.setLocation(location);
                myVolume.setSourceLanguage(source);
                myVolume.setTargetLanguages(targets);
                myVolume.setVolumeRef(href);
                myVolume.setCdmElements(cdmElements);
                myVolume.setLinksTable(linksTable);
                myVolume.setXmlCode(xmlCode);
                myVolume.setXmlSchema(xmlSchema);
                myVolume.setTemplateInterface(tmplInterface);
                myVolume.setTemplateEntry(tmplEntry);
                myVolume.setReverseLookup(reverse);
            } else {
                PapillonLogger.writeDebugMsg("Volume already in the Database");
            }
        }
        return myVolume;
    }

    /**
     * Find volume by name
     *
     * @param String
     * @return Volume
     * @throws PapillonBusinessException
     */
    public static Volume getVolumeByName(String name)
            throws PapillonBusinessException {

        //
        return VolumeCache.getVolumeInCache(name);

        /*
        Volume theVolume = null;

       if (null != name && !name.equals("")) {
           try {
               VolumeQuery query = new VolumeQuery(CurrentDBTransaction.get());
               //set query
               query.setQueryName(name,QueryBuilder.CASE_INSENSITIVE_EQUAL);
               // Throw an exception if more than one message is found
               query.requireUniqueInstance();
               VolumeDO theVolumeDO = query.getNextDO();
               theVolume = new Volume(theVolumeDO);
           }catch(Exception ex) {
               throw new PapillonBusinessException("Exception in getVolumeByName()", ex);
           }
       }
       return theVolume;
        */
    }

	/**
     * Find volume by name
     *
     * @param String
     * @return Volume
     * @throws PapillonBusinessException
     */
    public static Volume getVolumeByIndexDbname(String indexDbname)
	throws PapillonBusinessException {
		
        //
        return VolumeCache.getVolumeInCacheByIndexDbname(indexDbname);
    }
	

    /**
     * Get volume by handle
     *
     * @param String
     * @return Volume
     * @throws PapillonBusinessException
     */
    public static Volume getVolumeByHandle(String handle)
            throws PapillonBusinessException {

        //
        return VolumeCache.getVolumeInCacheByHandle(handle);
    }

    /**
     * Find volume by database name
     *
     * @param String
     *
     * @return Volume
     *
     * @exception PapillonBusinessException
     *
    public static Volume findVolumeByDbname(String name)
    throws PapillonBusinessException {
    Volume theVolume = null;
    if (null != name && !name.equals("")) {
    try {
    VolumeQuery query = new VolumeQuery(CurrentDBTransaction.get());
    //set query
    query.setQueryDbname(name,QueryBuilder.CASE_INSENSITIVE_EQUAL);
    // Throw an exception if more than one message is found
    query.requireUniqueInstance();
    VolumeDO theVolumeDO = query.getNextDO();
    theVolume = new Volume(theVolumeDO);
    }
    catch(Exception ex) {
    throw new PapillonBusinessException("Exception in findVolumeByDbame()", ex);
    }
    }
    return theVolume;
    }
     */


    /**
     * Parse url file to create a new volume
     *
     * @param Dictionary
     * @param URL
     * @param boolean
     * @param boolean
     * @return Volume
     * @throws PapillonBusinessException
     */
    public static Volume parseVolumeMetadata(Dictionary dict, URL fileURL, boolean parseEntries, boolean logContribs) throws fr.imag.clips.papillon.business.PapillonBusinessException {
        Document docXml = XMLServices.buildDOMTree(fileURL);
        return parseVolumeMetadata(dict, docXml, fileURL, parseEntries, logContribs);
    }

    public static Volume parseVolumeMetadata(Dictionary dict, Document docXml, URL fileURL, boolean parseEntries, boolean logContribs)
            throws fr.imag.clips.papillon.business.PapillonBusinessException {

        Volume resVolume = null;

        try {
            //PapillonLogger.writeDebugMsg("The xml code:");
            //PapillonLogger.writeDebugMsg(XMLServices.xmlCodePrettyPrinted(docXml));

            //on recupere le volume
            Element volume;
            boolean virtual = false;
            volume = (Element) docXml.getElementsByTagNameNS(DML_URI,VOLUME_TAG).item(0);
            if (volume != null) {
                String virtualString = volume.getAttribute(VIRTUAL_ATTRIBUTE);
                virtual = (virtualString != null && virtualString.equals("true"));
            }
			else {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException("Error: the XML file does not begin with the tag: " + VOLUME_TAG + "!");
			}

            Element schemaElement = (Element) docXml.getElementsByTagNameNS(XMLSCHEMA_URI,XMLSCHEMA_TAG).item(0);
            Element tmplEntryElement = (Element) docXml.getElementsByTagNameNS(DML_URI,TEMPLATE_ENTRY_TAG).item(0);
            Element tmplInterfaceElement = (Element) docXml.getElementsByTagNameNS(DML_URI,TEMPLATE_INTERFACE_TAG).item(0);
            String schemaString = "";
            if (schemaElement != null) {
                schemaString = XMLServices.NodeToString(schemaElement);
            }
           String tmplEntryString = "";
            if (tmplEntryElement != null) {
                tmplEntryString = XMLServices.NodeToString(getFirstElement(tmplEntryElement));
            }
            String tmplInterfaceString = "";
            if (tmplInterfaceElement != null) {
                tmplInterfaceString = XMLServices.NodeToString(tmplInterfaceElement.getFirstChild());
            }

            // ajout du volume ds la table.
            resVolume = VolumesFactory.newVolume(dict.getName(), volume, fileURL, schemaString, tmplEntryString, tmplInterfaceString);
			//PapillonLogger.writeDebugMsg("parseVolumeMetadata: linksTable size: " + resVolume.getLinksTable().size());
            if (null != resVolume) {
				if ((resVolume.getTemplateInterface() == null || resVolume.getTemplateInterface() == "") && resVolume.getTemplateEntry() !=null && resVolume.getTemplateEntry() != "" && resVolume.getXmlSchema() !=null && resVolume.getXmlSchema() !="") {
					GenerateTemplate.generateInterfaceTemplate(resVolume);
				}
                resVolume.save();

                NodeList stylesheets = (NodeList) docXml.getElementsByTagNameNS(DML_URI,XSLSHEET_REF_TAG);
                for (int i = 0; i < stylesheets.getLength(); i++) {
                    Element stylesheet = (Element) stylesheets.item(i);

                    if (null != stylesheet && null != fileURL) {
                        String ref = stylesheet.getAttributeNS(XLINK_URI, HREF_ATTRIBUTE);
                        String name = stylesheet.getAttribute(NAME_ATTRIBUTE);
                        if (name == null) {
                            name = "";
                        }

                        String isDefault = stylesheet.getAttribute(DEFAULT_ATTRIBUTE);
                        boolean isDefaultXsl = (null != isDefault && isDefault.equals("true"));

                        String isExternal = stylesheet.getAttribute(EXTERNAL_ATTRIBUTE);
                        boolean isExternalXsl = (null != isExternal && isExternal.equals("true"));

                        URL resultURL = new URL(fileURL, ref);
                        String xslString = fr.imag.clips.papillon.business.xsl.XslSheetFactory.parseXslSheet(resultURL);
                        XslSheetFactory.AddXslSheet(name, dict.getName(), resVolume.getName(), null, xslString,
                                isDefaultXsl, isExternalXsl);
                    }
                }
                stylesheets = (NodeList) docXml.getElementsByTagNameNS(XSLT_URI,XSLSHEET_TAG);
                for (int i = 0; i < stylesheets.getLength(); i++) {
                    Element stylesheet = (Element) stylesheets.item(i);
                    
                    if (null != stylesheet) {
                        String name = stylesheet.getAttribute(NAME_ATTRIBUTE);
                        if (name == null) {
                            name = "";
                        }
                        String isDefault = stylesheet.getAttribute(DEFAULT_ATTRIBUTE);
                        boolean isDefaultXsl = (null != isDefault && isDefault.equals("true"));
                        String xslString = XMLServices.NodeToString(stylesheet);
                        XslSheetFactory.AddXslSheet(name, dict.getName(), resVolume.getName(), null, xslString,
                                                    isDefaultXsl, false);
                    }
                }

                if (resVolume.getLocation().equals(Volume.LOCAL_LOCATION) && !virtual) {
                    VolumeEntriesFactory.createVolumeTables(resVolume);
                    if (parseEntries) {
                        URL resultURL = new URL(fileURL, resVolume.getVolumeRef());
                        ParseVolume.parseVolume(dict, resVolume, resultURL, logContribs, false);
                    } else {
                        PapillonLogger.writeDebugMsg("parseVolumeMetadata: do not parse entries!");
                    }
                }
            } else {
                PapillonLogger.writeDebugMsg("parseVolumeMetadata: volume null!");
            }
        } catch (Exception ex) {
            throw new PapillonBusinessException("Exception in parseVolumeMetadata()", ex);
        }
        return resVolume;
    }
    
    /*
     public static Volume findVolumeByID(String id)
         throws PapillonBusinessException {
             Volume theVolume = null;
             VolumeDO theVolumeDO = null;

             try {
                 VolumeQuery query = new VolumeQuery(CurrentDBTransaction.get());
                 //set query
                 query.setQueryOId(new ObjectId(id));
                 // Throw an exception if more than one message is found
                 query.requireUniqueInstance();
                 theVolumeDO = query.getNextDO();
                 theVolume = new Volume(theVolumeDO);
                 return theVolume;

             }
             catch(Exception ex) {
                 throw new PapillonBusinessException("Exception in findVolumeByID()", ex);
             }
         }
     */

    /**
     * Return a list of volume names
     *
     * @return String[]
     *
     * @exception PapillonBusinessException
    +/
    // FIXME: return Collection
    public static String[] getVolumesArrayName()
    throws PapillonBusinessException {
    return getVolumesArrayName(null, null, null);
    }
     */

    /**
     * Return a list of volumes
     *
     * @return Collection
     * @throws PapillonBusinessException
     */
    public static Collection getVolumesArray()
            throws PapillonBusinessException {
        return getVolumesArray(null, null, null);
    }

    /**
     * Return a list of volume names
     *
     * @return String[]
     *
     * @exception PapillonBusinessException
    +/
    // FIXME: return Collection
    public static String[] getVolumesArrayName(String dictName)
    throws PapillonBusinessException {
    return getVolumesArrayName(dictName, null, null);
    }
     */

    /**
     * Return a list of volumes
     *
     * @return Volume[]
     * @throws PapillonBusinessException
     */
    public static Collection getVolumesArray(String dictName)
            throws PapillonBusinessException {
        return getVolumesArray(dictName, null, null);
    }

    /**
     * Return a list of volume names
     *
     * @return String[]
     *
     * @exception PapillonBusinessException
    +/
    // FIXME: return Collection
    public static String[] getVolumesArrayName(String dictname, String source, String target)
    throws PapillonBusinessException {
    Volume[] volumeList = getVolumesArray(dictname, source, target);
    String[] nameList = new String[ volumeList.length ];
    for ( int i = 0; i < volumeList.length; i++ )
    nameList[i] = volumeList[i].getName();
    return nameList;
    }
     */

    /**
     * Return a list of volumes
     *
     * @return Collection
     * @throws PapillonBusinessException
     */
    // FIXME: should provide the same method with a dictionary instead of a dictionary name. And use it when possible.
    public static Collection getVolumesArray(String dictname, String source, String target)
            throws PapillonBusinessException {

        try {

            //
            ArrayList volumes = new ArrayList(VolumeCache.getVolumesInCache());

            if ((dictname != null) && !dictname.equals("")) {

                //
                Collection volumesByDictionaryName = VolumeCache.getVolumesInCacheByDictionaryName(dictname);
                volumes.retainAll(volumesByDictionaryName);
            }

            if ((source != null) && !source.equals("")) {

                //
                Collection volumesBySource = VolumeCache.getVolumesInCacheBySourceLanguage(source);
                volumes.retainAll(volumesBySource);
            }

            if ((target != null) && !target.equals("")) {

                //
                Collection volumesByTarget = VolumeCache.getVolumesInCacheByTargetLanguage(target);
                volumes.retainAll(volumesByTarget);
            }
			//Collection.sort(volumes,new fr.imag.clips.papillon.business.utility.VolumeComparator());
            //
            return volumes;

            /*
           Volume[] theDictArray = null;
           ArrayList theDictArrayList = new ArrayList();

           for (Iterator iter = volumes.iterator(); iter.hasNext();) {
               Volume volume = (Volume)iter.next();

               if ( ((dictname == null) || dictname.equals(""))
                    && ((source == null) || source.equals(""))
                    && ((target == null) || target.equals("")) ) {

                   //
                   theDictArrayList.add(volume);

               } else if ( (dictname != null) && !dictname.equals("") && dictname.equals(dictname)
                           && ((source == null) || source.equals(""))
                           && ((target == null) || target.equals("")) ) {

                   //
                   theDictArrayList.add(volume);

               } else if ( (dictname != null) && !dictname.equals("") && dictname.equals(dictname)
                           && (source != null) && !source.equals("")  && volume.getSourceLanguage().equals(source)
                           && ((target == null) || target.equals("")) ) {

                   //
                   theDictArrayList.add(volume);

               }  else if ( (dictname != null) && !dictname.equals("") && dictname.equals(dictname)
                            && (source != null) && !source.equals("") && volume.getSourceLanguage().equals(source)
                            && (target != null) && !target.equals("") ) {

                   //
                   String[] targets = volume.getTargetLanguagesArray();
                   int j = 0;
                   while ( (j < targets.length) && (!targets[j].equals(target)) ) {
                       j++;
                   }

                   if (j < targets.length) {
                       theDictArrayList.add(volume);
                   }
               }
           }

           // A ENLEVER !  ... mettre Collection a la place de Volume[]
           theDictArray = new Volume[theDictArrayList.size()];
           int i = 0;
           for (Iterator iter = theDictArrayList.iterator(); iter.hasNext();) {
               theDictArray[i] = (Volume)iter.next();
               i++;
           }
            //return theDictArray;
            */

        } catch (Exception ex) {
            throw new PapillonBusinessException("Exception in getVolumesArray()", ex);
        }
    }
	

    /**
     * Return the collection of all volumes
     *
     * @return Collection
     * @throws PapillonBusinessException
     */
    public static Collection getVolumes()
            throws PapillonBusinessException {

        return VolumeCache.getVolumesInCache();

        /*
       Volume[] theDictArray = null;

       try {
           VolumeQuery query = new VolumeQuery(CurrentDBTransaction.get());

           query.addOrderByName(true);
           VolumeDO[] DOarray = query.getDOArray();
           theDictArray = new Volume[ DOarray.length ];
           for ( int i = 0; i < DOarray.length; i++ )
               theDictArray[i] = new Volume(DOarray[i]);

       }catch(Exception ex) {
           throw new PapillonBusinessException("Exception in getVolumesArray()", ex);
       }

       return theDictArray;
        */
    }

    /**
     * Return the list of all source languages
     *
     * @return Enumeration
     * @throws PapillonBusinessException
     */
    public static Enumeration getSourceLanguages()
            throws PapillonBusinessException {

        return VolumeCache.getSourceLanguagesInCache();

    }

    /**
     * Return the list of all target languages
     *
     * @return Enumeration
     * @throws PapillonBusinessException
     */
    public static Enumeration getTargetLanguages()
            throws PapillonBusinessException {

        return VolumeCache.getTargetLanguagesInCache();

    }


    /**
     * Return the xml code of a tag in the url file
     *
     * @param Element
     * @param String
     * @param URL
     * @return String
     * @throws PapillonBusinessException
     */
    protected static String getXmlCode(Element myDOMElement, String tag, URL fileURL)
            throws PapillonBusinessException {
        String res = "";
        String href = "";
        try {
            NodeList refNodes = myDOMElement.getElementsByTagName(tag);
            if (null != refNodes && refNodes.getLength() > 0) {
                Element tempElt = (Element) refNodes.item(0);
                href = tempElt.getAttributeNS(XLINK_URI, "href");
                if (href != null && !href.equals("") && fileURL!=null) {
                    URL resultURL = new URL(fileURL, href);
                    PapillonLogger.writeDebugMsg("getXmlCode with URL: " + resultURL.toString());
                    res = XMLServices.xmlCode(XMLServices.buildDOMTree(resultURL));
                }
            } else {
                PapillonLogger.writeDebugMsg("No " + tag);
            }
        } catch (MalformedURLException ex) {
            throw new PapillonBusinessException("Error in getXmlCode with the href " + href + ":" + ex.getMessage(), ex);
        }
        return res;
    }


    /**
     * @param String
     * @param String
     * @param String
     * @return HashMap
     * @throws PapillonBusinessException
     */
    /* cdmElements HashMap = {lang => HashMap} = {CDM_element => ArrayList} = (xpathString, isIndex, XPath)*/
    public static HashMap buildCdmElementsTable(String xmlCode, String sourceLanguage, org.apache.xml.utils.PrefixResolver thePrefixResolver, String dml_prefix)
            throws fr.imag.clips.papillon.business.PapillonBusinessException {
        HashMap cdmElements = new HashMap();
        Document docXml = XMLServices.buildDOMTree(xmlCode);
        NodeList cdmElts = docXml.getElementsByTagName(CDM_ELEMENTS_TAG);
        if (null != cdmElts && cdmElts.getLength() > 0) {
            Element cdmElt = (Element) cdmElts.item(0);
            cdmElements = buildCdmElementsTable(cdmElt, sourceLanguage, dml_prefix);
        } else {
            PapillonLogger.writeDebugMsg("No " + CDM_ELEMENTS_TAG + " tag!");
        }
		compileXPathTable(cdmElements, thePrefixResolver);
        return cdmElements;
    }
	

    /**
     * @param Node
     * @param String
     * @return HashMap
     */
    /* cdmElements HashMap = {lang => HashMap} = {CDM_element => ArrayList} = (xpathString, isIndex, XPath)*/
    protected static HashMap buildCdmElementsTable(Node cdmElt, String sourceLanguage, String dml_prefix) {
        HashMap cdmElements = new HashMap();
       NodeList cdmChilds = cdmElt.getChildNodes();
        for (int i = 0; i < cdmChilds.getLength(); i++) {
            Node myNode = cdmChilds.item(i);
            if (myNode.getNodeType() == Node.ELEMENT_NODE) {
                Element myElt = (Element) myNode;
                String eltName = myElt.getTagName();
				
				if (!eltName.equals(LINKS_TAG)) {
					/* determine the language of the cdm element */
					String lang = Volume.DEFAULT_LANG;
					Attr langAttr = myElt.getAttributeNodeNS(DML_URI, LANG_ATTRIBUTE);
					if (langAttr != null) {
						lang = langAttr.getValue();
					} else if (Volume.isLangCDMElement(eltName)) {
						lang = sourceLanguage;
					}
					
					/* determine if the cdm element has to be indexed */
					boolean isIndex = false;
					String index = myElt.getAttribute(INDEX_ATTRIBUTE);
					isIndex = (index != null && index.equals("true"));
					if (!isIndex) {
						isIndex = Volume.isIndexCDMElement(eltName);
					}
					
					
					String xpath = myElt.getAttribute(XPATH_ATTRIBUTE);
					// PapillonLogger.writeDebugMsg("addCdmElementInTable: " + eltName + " lang: " + lang + " index: " + isIndex + " xpath: " + xpath);
					addCdmElementInTable(cdmElements, eltName, lang, xpath, isIndex);
				}
            }
        }
        completeCdmElementsTable(cdmElements, sourceLanguage, dml_prefix);
        updateCdmElementsTable(cdmElements, dml_prefix);
        return cdmElements;
    }
	
	
 	/**
     * @param String
     * @param String
     * @param String
     * @return HashMap
     * @throws PapillonBusinessException
     */
    /* cdmElements HashMap = {lang => HashMap} = {CDM_element => ArrayList} = (xpathString, isIndex, XPath)*/
    public static HashMap buildLinksTable(String xmlCode, org.apache.xml.utils.PrefixResolver thePrefixResolver, HashMap cdmElements, String dml_prefix)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
        HashMap linksTable = new HashMap();
        Document docXml = XMLServices.buildDOMTree(xmlCode);
        NodeList cdmElts = docXml.getElementsByTagName(CDM_ELEMENTS_TAG);
        if (null != cdmElts && cdmElts.getLength() > 0) {
            Element cdmElt = (Element) cdmElts.item(0);
            linksTable = buildLinksTable(cdmElt, cdmElements, dml_prefix);
        } else {
            PapillonLogger.writeDebugMsg("No " + CDM_ELEMENTS_TAG + " tag!");
        }
		compileLinksXPathTable(linksTable, thePrefixResolver);
        return linksTable;
    }

	
	/**
     * @param Node
     * @param String
     * @return HashMap
     */
    /* linksTable HashMap = {name => HashMap} = {element => ArrayList} (xpathString, XPath) */
    protected static HashMap buildLinksTable(Node cdmElt, HashMap cdmElements, String dml_prefix) {
		HashMap linksTable = new HashMap();
		
		String contribPath = getCdmXPathString(cdmElements, Volume.CDM_contribution, Volume.DEFAULT_LANG);
        String entryPath = getCdmXPathString(cdmElements, Volume.CDM_templateEntry, Volume.DEFAULT_LANG);
        String entryTag = Volume.getTagNameFromXPath(entryPath);
		if (dml_prefix != null && !dml_prefix.equals("")) {
			dml_prefix += ":";
		}
		
        NodeList cdmChilds = cdmElt.getChildNodes();
		for (int i = 0; i < cdmChilds.getLength(); i++) {
            Node myNode = cdmChilds.item(i);
            if (myNode.getNodeType() == Node.ELEMENT_NODE) {
                Element myElt = (Element) myNode;
                String eltName = myElt.getTagName();
				
				if (eltName.equals(LINKS_TAG)) {
					NodeList cdmLinksChilds = myElt.getChildNodes();
					for (int j = 0; j < cdmLinksChilds.getLength(); j++) {
						Node linkNode = cdmLinksChilds.item(j);
						if (linkNode.getNodeType() == Node.ELEMENT_NODE) {
							Element linkElt = (Element) linkNode;
							String linkEltName = linkElt.getTagName();
							if (linkEltName.equals(Volume.CDM_link)) {
								String name = Volume.LINK_defaultLinkName;
								Attr nameAttr = linkElt.getAttributeNode(NAME_ATTRIBUTE);
								if (nameAttr != null) {
									name = nameAttr.getValue();
								}
								String xpathString = linkElt.getAttribute(XPATH_ATTRIBUTE);
								/* update xpath with DML metadata elements */
								if (xpathString.indexOf(contribPath) < 0 && xpathString.indexOf(entryPath) == 0) {
									xpathString = searchAndReplace(xpathString, entryPath,
																   contribPath + "/" + dml_prefix + VolumeEntry.dataTag + "/" + entryTag);
								}
								
								HashMap linkTable = new HashMap();
								ArrayList theArrayList = new ArrayList();
								theArrayList.add(xpathString);
								//PapillonLogger.writeDebugMsg("addLinkInTable: " + name + " xpath: " + xpathString);
								linkTable.put(Volume.LINK_ELEMENT_TYPE,theArrayList);
								
								String lang = Volume.DEFAULT_LANG;
								Attr langAttr = linkElt.getAttributeNodeNS(DML_URI, LANG_ATTRIBUTE);
								if (langAttr != null) {
									lang = langAttr.getValue();
								}
								//PapillonLogger.writeDebugMsg("addLinkInTable: " + name + " xpath: " + lang);
								linkTable.put(Volume.LINK_STRING_LANG_TYPE,lang);
								
								NodeList cdmLinkChilds = linkElt.getChildNodes();
								for (int k = 0; k < cdmLinkChilds.getLength(); k++) {
									Node linkChildNode = cdmLinkChilds.item(k);
									if (linkChildNode.getNodeType() == Node.ELEMENT_NODE) {
										Element linkChildElt = (Element) linkChildNode;
										String linkChildName = linkChildElt.getTagName();
										String xpath = linkChildElt.getAttribute(XPATH_ATTRIBUTE);
										ArrayList secArrayList = new ArrayList();
										secArrayList.add(xpath);
									linkTable.put(linkChildName,secArrayList);
										//PapillonLogger.writeDebugMsg("addLinkInTable: " + linkChildName + " xpath: " + xpath);
									}
								}
								linksTable.put(name, linkTable); 
								//PapillonLogger.writeDebugMsg("addLinkTableinLinksTable: " + name);
							}
						}
					}
				}
            }
        }
        return linksTable;
    }

	protected static void upgradeCdmLinkElement(Document docXml ) {

        NodeList cdmElts = docXml.getElementsByTagName(CDM_ELEMENTS_TAG);
        if (null != cdmElts && cdmElts.getLength() > 0) {
            Element cdmElt = (Element) cdmElts.item(0);
			
			NodeList translationsRef = cdmElt.getElementsByTagName(Volume.CDM_translationReflexie);
			if (null != translationsRef && translationsRef.getLength() > 0) {
				Element translationRef = (Element) translationsRef.item(0);

				Element newLink = docXml.createElement(Volume.CDM_link);
				newLink.setAttribute(NAME_ATTRIBUTE,Volume.LINK_translationLinkName);
				
				Attr langAttr = translationRef.getAttributeNodeNS(DML_URI, LANG_ATTRIBUTE);
				if (langAttr != null) {
					newLink.setAttribute("d:"+ LANG_ATTRIBUTE, langAttr.getValue());
				}

				Element value = docXml.createElement(Volume.CDM_linkValue);
				String xpath = translationRef.getAttribute(XPATH_ATTRIBUTE);
				String xpathStart = xpath.substring(0,xpath.lastIndexOf('/'));
				String xpathEnd = xpath.substring(xpath.lastIndexOf('/')+1);
				newLink.setAttribute(XPATH_ATTRIBUTE, xpathStart);
				value.setAttribute(XPATH_ATTRIBUTE, xpathEnd);

				newLink.appendChild(value);

				Element linksElement = null;
				NodeList linksList = cdmElt.getElementsByTagName(LINKS_TAG);
				if (null != linksList && linksList.getLength() > 0) {
					linksElement = (Element) linksList.item(0);
				}
				else {
					linksElement = docXml.createElement(LINKS_TAG);
					cdmElt.appendChild(linksElement);
				}
				
				linksElement.appendChild(newLink);
				cdmElt.removeChild(translationRef);
				//PapillonLogger.writeDebugMsg("docXml modifié: " + XMLServices.NodeToString(docXml));
			} 
		} 
	}
	

    /**
     * @param HashMap
     */
    /* cdmElements HashMap = {lang => HashMap} = {CDM_element => ArrayList} = (xpathString, isIndex, XPath)*/
    protected static void updateCdmElementsTable(HashMap cdmElements, String dml_prefix) {

		if (dml_prefix != null && !dml_prefix.equals("")) {
			dml_prefix += ":";
		}

        String contribPath = getCdmXPathString(cdmElements, Volume.CDM_contribution, Volume.DEFAULT_LANG);
        String entryPath = getCdmXPathString(cdmElements, Volume.CDM_entry, Volume.DEFAULT_LANG);
        String entryTag = Volume.getTagNameFromXPath(entryPath);

        for (Iterator  langKeys = cdmElements.keySet().iterator(); langKeys.hasNext();) {
            String lang = (String) langKeys.next();
            HashMap tmpTable = (java.util.HashMap) cdmElements.get(lang);
            for (Iterator keys = tmpTable.keySet().iterator(); keys.hasNext();) {
                String CDM_element = (String) keys.next();
                ArrayList eltArrayList = (ArrayList) tmpTable.get(CDM_element);
                if (eltArrayList != null && eltArrayList.size() >= 2) {
                    String xpathString = (String) eltArrayList.get(0);
                    if (xpathString.indexOf(contribPath) < 0 && xpathString.indexOf(entryPath) == 0) {
                        xpathString = searchAndReplace(xpathString, entryPath,
                                contribPath + "/" + dml_prefix + VolumeEntry.dataTag + "/" + entryTag);
                        eltArrayList.remove(0);
                        eltArrayList.add(0, xpathString);
                    }
                }
            }
        }
        /* Here I keep the original entry xpath in order to use it when I load the template entry */
        addCdmElementInTable(cdmElements, Volume.CDM_templateEntry, Volume.DEFAULT_LANG, entryPath, false);
    }	
	

    /**
     * @param HashMap
     * @param String
     */
    /* cdmElements HashMap = {lang => HashMap} = {CDM_element => ArrayList} = (xpathString, isIndex, XPath)*/
    protected static void completeCdmElementsTable(HashMap elementsTable, String sourceLanguage, String dml_prefix) {
		
		if (dml_prefix != null && !dml_prefix.equals("")) {
			dml_prefix += ":";
		}
		
        String currentXpath = getCdmXPathString(elementsTable, Volume.CDM_volume, Volume.DEFAULT_LANG);
        // PapillonLogger.writeDebugMsg("completeCdmElementsTable: currentXpath: " + currentXpath);

        // contribution tags
        currentXpath += "/" + dml_prefix + VolumeEntry.contributionTag;
        if (getCdmXPathString(elementsTable, Volume.CDM_contribution, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contribution, Volume.DEFAULT_LANG, currentXpath,
                    Volume.isIndexCDMElement(Volume.CDM_contribution));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionId, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionId, Volume.DEFAULT_LANG,
                    currentXpath + "/@" + dml_prefix + VolumeEntry.contributionIdAttr,
                    Volume.isIndexCDMElement(Volume.CDM_contributionId));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_originalContributionId, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_originalContributionId, Volume.DEFAULT_LANG,
                    currentXpath + "/@" + dml_prefix + VolumeEntry.originalContributionIdAttr,
                    Volume.isIndexCDMElement(Volume.CDM_originalContributionId));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionDataElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionDataElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.dataTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionDataElement));
        }
        currentXpath += "/" + dml_prefix + VolumeEntry.metadataTag;
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionAuthorElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionAuthorElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.authorTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionAuthorElement));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionAuthor, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionAuthor, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.authorTag + "/text()",
                    Volume.isIndexCDMElement(Volume.CDM_contributionAuthor));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionGroups, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionGroups, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.groupsTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionGroups));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionGroup, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionGroup, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.groupsTag + "/" + dml_prefix + VolumeEntry.groupTag + "/text()",
                    Volume.isIndexCDMElement(Volume.CDM_contributionGroup));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionCreationDateElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionCreationDateElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.creationDateTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionCreationDateElement));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionCreationDate, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionCreationDate, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.creationDateTag + "/text()",
                    Volume.isIndexCDMElement(Volume.CDM_contributionCreationDate));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionFinitionDateElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionFinitionDateElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.finitionDateTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionFinitionDateElement));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionFinitionDate, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionFinitionDate, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.finitionDateTag + "/text()",
                    Volume.isIndexCDMElement(Volume.CDM_contributionFinitionDate));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionReviewDateElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionReviewDateElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.reviewDateTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionReviewDateElement));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionReviewDate, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionReviewDate, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.reviewDateTag + "/text()",
                    Volume.isIndexCDMElement(Volume.CDM_contributionReviewDate));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionReviewerElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionReviewerElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.reviewerTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionReviewerElement));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionReviewer, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionReviewer, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.reviewerTag + "/text()",
                    Volume.isIndexCDMElement(Volume.CDM_contributionReviewer));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionValidationDateElement,
                Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionValidationDateElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.validationDateTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionValidationDateElement));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionValidationDate, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionValidationDate, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.validationDateTag + "/text()",
                    Volume.isIndexCDMElement(Volume.CDM_contributionValidationDate));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionValidatorElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionValidatorElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.validatorTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionValidatorElement));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionValidator, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionValidator, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.validatorTag + "/text()",
                    Volume.isIndexCDMElement(Volume.CDM_contributionValidator));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionStatusElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionStatusElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.statusTag,
                    Volume.isIndexCDMElement(Volume.CDM_contributionStatusElement));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_contributionStatus, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_contributionStatus, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.statusTag + "/text()",
                    Volume.isIndexCDMElement(Volume.CDM_contributionStatus));
        }
        // Previous contribution
        if (getCdmXPathString(elementsTable, Volume.CDM_previousContributionElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_previousContributionElement, Volume.DEFAULT_LANG,
                                 currentXpath + "/" + dml_prefix + VolumeEntry.previousContributionTag,
                                 Volume.isIndexCDMElement(Volume.CDM_previousContributionElement));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_previousContribution, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_previousContribution, Volume.DEFAULT_LANG,
                                 currentXpath + "/" + dml_prefix + VolumeEntry.previousContributionTag + "/text()",
                                 Volume.isIndexCDMElement(Volume.CDM_previousContribution));
        }
       // Previous classified finished contribution
        if (getCdmXPathString(elementsTable, Volume.CDM_previousClassifiedFinishedContributionElement,
                Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_previousClassifiedFinishedContributionElement,
                    Volume.DEFAULT_LANG, currentXpath + "/" + dml_prefix + VolumeEntry.previousClassifiedFinishedContributionTag,
                    false);
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_previousClassifiedFinishedContribution,
                Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_previousClassifiedFinishedContribution, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.previousClassifiedFinishedContributionTag + "/text()", Volume.isIndexCDMElement(Volume.CDM_previousClassifiedFinishedContribution));
        }
        // Previous classified not finished contribution
        if (getCdmXPathString(elementsTable, Volume.CDM_previousClassifiedNotFinishedContributionElement,
                Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_previousClassifiedNotFinishedContributionElement,
                    Volume.DEFAULT_LANG, currentXpath + "/" + dml_prefix + VolumeEntry.previousClassifiedNotFinishedContributionTag,
                    false);
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_previousClassifiedNotFinishedContribution,
                Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_previousClassifiedNotFinishedContribution,
                    Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.previousClassifiedNotFinishedContributionTag + "/text()", Volume.isIndexCDMElement(Volume.CDM_previousClassifiedNotFinishedContribution));
        }
        // Next contribution author
        if (getCdmXPathString(elementsTable, Volume.CDM_nextContributionAuthorElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_nextContributionAuthorElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.nextContributionAuthorTag, false);
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_nextContributionAuthor, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_nextContributionAuthor, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.nextContributionAuthorTag + "/text()", Volume.isIndexCDMElement(Volume.CDM_nextContributionAuthor));
        }


        currentXpath += "/" + dml_prefix + VolumeEntry.historyTag;
        if (getCdmXPathString(elementsTable, Volume.CDM_history, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_history, Volume.DEFAULT_LANG, currentXpath,
                    Volume.isIndexCDMElement(Volume.CDM_history));
        }

        currentXpath += "/" + dml_prefix + VolumeEntry.modificationTag;
        if (getCdmXPathString(elementsTable, Volume.CDM_modification, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_modification, Volume.DEFAULT_LANG, currentXpath,
                    Volume.isIndexCDMElement(Volume.CDM_modification));
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_modificationAuthorElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_modificationAuthorElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.authorTag, false);
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_modificationAuthor, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_modificationAuthor, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.authorTag + "/text()", true);
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_modificationDateElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_modificationDateElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.dateTag, false);
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_modificationDate, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_modificationDate, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.dateTag + "/text()", true);
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_modificationCommentElement, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_modificationCommentElement, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.commentTag, false);
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_modificationComment, Volume.DEFAULT_LANG) == null) {
            addCdmElementInTable(elementsTable, Volume.CDM_modificationComment, Volume.DEFAULT_LANG,
                    currentXpath + "/" + dml_prefix + VolumeEntry.commentTag + "/text()", true);
        }
		// Creating the CDM_entryString from CDM_entry if it does not exist
        if (getCdmXPathString(elementsTable, Volume.CDM_entryString, Volume.DEFAULT_LANG) == null) {
            String entryString = getCdmXPathString(elementsTable, Volume.CDM_entry, Volume.DEFAULT_LANG);
			addCdmElementInTable(elementsTable, Volume.CDM_entryString, Volume.DEFAULT_LANG, entryString,false);
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_headwordElement, sourceLanguage) == null) {
            String headwordXpath = getCdmXPathString(elementsTable, Volume.CDM_headword, sourceLanguage);
            if (headwordXpath != null) {
                if (headwordXpath.indexOf("/text()") > 0) {
                    headwordXpath = headwordXpath.substring(0, headwordXpath.indexOf("/text()"));
                }
                addCdmElementInTable(elementsTable, Volume.CDM_headwordElement, sourceLanguage, headwordXpath,
                        Volume.isIndexCDMElement(Volume.CDM_headwordElement));
            }
        }
        if (getCdmXPathString(elementsTable, Volume.CDM_entryIdElement, Volume.DEFAULT_LANG) == null) {
            String entryidxpath = getCdmXPathString(elementsTable, Volume.CDM_entryId, Volume.DEFAULT_LANG);
            if (entryidxpath != null) {
                if (entryidxpath.indexOf("/text()") > 0) {
                    entryidxpath = entryidxpath.substring(0, entryidxpath.indexOf("/text()"));
                }
                addCdmElementInTable(elementsTable, Volume.CDM_entryIdElement, Volume.DEFAULT_LANG, entryidxpath,
                        Volume.isIndexCDMElement(Volume.CDM_entryIdElement));
            }
        }
    }


    /**
     * @param String
     * @param HashMap
     * @return String
     * @throws PapillonBusinessException
     */
    // embeds the template entry into a contribution element
    public static String updateTemplateEntry(String tmplEntry, HashMap cdmElements)
            throws fr.imag.clips.papillon.business.PapillonBusinessException {
        if (tmplEntry != null && !tmplEntry.equals("")) {
            //PapillonLogger.writeDebugMsg("updateTemplateEntry: " + tmplEntry);
            org.w3c.dom.Document templateDoc = XMLServices.buildDOMTree(tmplEntry);			
			org.apache.xml.utils.PrefixResolver tmplPrefixResolver = new org.apache.xml.utils.PrefixResolverDefault(templateDoc);

            org.w3c.dom.NodeList contribNodeList = IndexEntry.getCdmElements(templateDoc, Volume.CDM_contribution,
                    Volume.DEFAULT_LANG, cdmElements, tmplPrefixResolver);
            org.w3c.dom.NodeList entryNodeList = IndexEntry.getCdmElements(templateDoc, Volume.CDM_templateEntry,
                    Volume.DEFAULT_LANG, cdmElements, tmplPrefixResolver);

            if ((contribNodeList == null || contribNodeList.getLength() == 0) && (entryNodeList != null && entryNodeList.getLength() == 1)) {
                Node myEntryNode = entryNodeList.item(0);
                // FIXME: maybe use XMLC to have an already compiled DOM ?
              String contributionString = VolumeEntry.getContributionHeader(tmplEntry) + VolumeEntry.getContributionFooter(tmplEntry);
                org.w3c.dom.Document contributionDoc = XMLServices.buildDOMTree(contributionString);
                Node contributionNode = templateDoc.importNode(contributionDoc.getDocumentElement(), true);
                Node entryParent = myEntryNode.getParentNode();
                entryParent.replaceChild(contributionNode, myEntryNode);
                NodeList dataNodeList = IndexEntry.getCdmElements(templateDoc, Volume.CDM_contributionDataElement,
                        Volume.DEFAULT_LANG, cdmElements, tmplPrefixResolver);
               if (dataNodeList != null && dataNodeList.getLength() == 1) {
                    Node dataNode = dataNodeList.item(0);
                    dataNode.appendChild(myEntryNode);
                    tmplEntry = XMLServices.xmlCode(templateDoc);
                }
            } else {
                PapillonLogger.writeDebugMsg(
                        "updateTemplateEntry: contribNodeList null? " + (contribNodeList == null) + " entryNodeList null?: " + (entryNodeList == null));
            }
        }
        return tmplEntry;
    }
	
    protected static String addDmlUrlInTemplateEntry(String tmplEntry)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		String result = null;
        if (tmplEntry != null && !tmplEntry.equals("")) {
            org.w3c.dom.Document templateDoc = XMLServices.buildDOMTree(tmplEntry);
			org.w3c.dom.Element rootElement = templateDoc.getDocumentElement();
			if (getDmlPrefix(rootElement) == null) {
				String newDmlPrefix = DEFAULT_DML_PREFIX;
				ArrayList prefixes = getPrefixes(rootElement);
				while (prefixes.contains(newDmlPrefix)) {
					newDmlPrefix += DEFAULT_DML_PREFIX;
				}
				rootElement.setAttribute(XMLNAMESPACE + ":" + newDmlPrefix, DML_URI);
			}
			result = XMLServices.xmlCode(templateDoc);
		}
		return result;
	}
			
	public static String getDmlPrefix(Element element) {
		return getPrefix(element, DML_URI);
	}
	
	public static String getPrefix(Element element, String namespaceUri) {
		//PapillonLogger.writeDebugMsg("get dml prefix: " + element.getNodeName() + " " + DML_URI);
		String prefix = null;
		org.w3c.dom.NamedNodeMap atts = element.getAttributes();
		int i=0;
		while (i < atts.getLength() && prefix ==null) {
			Node node = atts.item(i);
			String name = node.getNodeName();
			//PapillonLogger.writeDebugMsg("get dml prefix: name : " + name + " value: " + node.getNodeValue());
			if (namespaceUri.equals(node.getNodeValue())
				&& (name != null && (XMLNAMESPACE.equals(name) || name.startsWith(XMLNAMESPACE + ":")))) {
				prefix = node.getLocalName();
				if (prefix.equals(XMLNAMESPACE)) {
					prefix = "";
				}
			}
			i++;
		}
		return prefix;
	}
	
	public static ArrayList getPrefixes(Element element) {
		org.w3c.dom.NamedNodeMap atts = element.getAttributes();
		ArrayList prefixes = new ArrayList(atts.getLength());
		for (int i=0; i < atts.getLength(); i++) {
			Node node = atts.item(i);
			String name = node.getNodeName();
			if (name != null && (XMLNAMESPACE.equals(name) || name.startsWith(XMLNAMESPACE + ":"))) {
				prefixes.add(node.getLocalName());
			}
		}
		return prefixes;
	}
	
	

    /**
     * @param HashMap
     * @param String
     * @param String
     * @return String
     */
    protected static String getCdmXPathString(HashMap table, String name, String lang) {
        String res = null;
        if (table != null) {
            java.util.HashMap tmpTable = (java.util.HashMap) table.get(lang);
            if (tmpTable != null) {
                java.util.ArrayList myArrayList = (java.util.ArrayList) tmpTable.get(name);
                if (myArrayList != null && myArrayList.size() > 0) {
                    res = (String) myArrayList.get(0);
                }
            }
        }
        return res;
    }


    /**
     * @param HashMap
     * @param String
     * @param String
     * @param String
     * @param boolean
     */
    protected static void addCdmElementInTable(HashMap table, String elt, String lang, String xpathString,
                                               boolean isIndex) {
        /* cdmElements HashMap = {lang => HashMap} = {CDM_element => ArrayList} = (xpathString, isIndex, XPath)*/
		//PapillonLogger.writeDebugMsg("addCdmElementInTable: elt: " + elt + " lang: " + lang + " xpath: " + xpathString + " isIndex: " + isIndex);
        ArrayList xpathAndIndexList = new ArrayList();
        xpathAndIndexList.add(xpathString);
        xpathAndIndexList.add(new Boolean(isIndex));
        HashMap tmpTable = (HashMap) table.get(lang);
        if (tmpTable == null) {
            tmpTable = new HashMap();
            table.put(lang, tmpTable);
        }
        tmpTable.put(elt, xpathAndIndexList);
    }


    /**
     * @param String
     * @param String
     * @param String
     * @return String
     */
    protected static String searchAndReplace(String theString, String search, String replace) {
        String searchedRegex = "\\Q" + search + "\\E";
        theString = theString.replaceAll(searchedRegex, replace);
        return theString;
    }

    /**
     * The reConstructionIndex rebuild the volume index.
     *
     * @throws PapillonBusinessException
     */
    protected static void volumeReConstructionIndex(Volume volume)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {

		//
		VolumeEntriesFactory.resetCountCache(volume.getName());
		int count = volume.getCount();
		//volume.setEntries(Integer)
		int delta = 20; // buffer limit
		PapillonLogger.writeDebugMsg("Volume : " + volume.getName() + " - " + count + " entries");
		IndexFactory.truncateIndexTable(volume);
		//
		for (int i = 0; i < count; i = i + delta) {
			
			try {
				
				// Begin transaction
				CurrentDBTransaction.registerNewDBTransaction();
				
				// Buffer volumeEntries
				Collection bufferResults = VolumeEntriesFactory.getVolumeEntries(volume, i, delta);
				
				//
				Iterator buffer = bufferResults.iterator();
				while (buffer.hasNext()) {
					VolumeEntry ve = (VolumeEntry) buffer.next();
					
					//
					IndexEntry.indexEntry(ve);					
				}
				
				// End transaction : a part was correct, commit the transaction ...
				((DBTransaction) CurrentDBTransaction.get()).commit();
				PapillonLogger.writeDebugMsg(Integer.toString(i) + " to " + Integer.toString(i + delta - 1));
				
			} catch (Exception e) {
				String userMessage = "Problems when re-index entries.";
				PapillonLogger.writeDebugMsg(userMessage);
				e.printStackTrace();
				
				//
				try {
					((DBTransaction) CurrentDBTransaction.get()).rollback();
				} catch (java.sql.SQLException sqle) {
					PapillonLogger.writeDebugMsg(
												 "reConstructionIndex: SQLException while rolling back failed transaction.");
					sqle.printStackTrace();
				}
				
			} finally {
				CurrentDBTransaction.releaseCurrentDBTransaction();
			}
		}
		
		//
		PapillonLogger.writeDebugMsg(volume.getName() + " index re-construction succeed !");
		
	}
	
    /**
     * The reConstructionIndex rebuild the volume index.
     *
     * @throws PapillonBusinessException
     */
    public static void volumeNameReConstructionIndex(String volumeName) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			if (volumeName != null && !volumeName.equals("")) {
				Volume theVolume = VolumesFactory.getVolumeByName(volumeName);
				if (theVolume != null && !theVolume.isEmpty()) {
					volumeReConstructionIndex(theVolume);
				}
			}
	}

    /**
     * The reConstructionIndex rebuild the volume index.
     *
     * @throws PapillonBusinessException
     */
    public static void reConstructionIndex()
            throws fr.imag.clips.papillon.business.PapillonBusinessException {

        //
        for (Iterator iter = getVolumesArray().iterator(); iter.hasNext();) {
            Volume volume = (Volume) iter.next();
			volumeReConstructionIndex(volume);
		}
        //
        PapillonLogger.writeDebugMsg("Index re-construction process succeed !");
    }


    /**
     * The modifiedStatus method modify the status of the contributions appearing "before" a not-finished contribution to modified status.
     *
     * @param User
     * @throws PapillonBusinessException
     */
    // FIXME : supress when GDEF and Papillon project will be import in Jibiki
    public static void modifiedStatus(fr.imag.clips.papillon.business.user.User user)
            throws PapillonBusinessException {
        try {

            //
            for (Iterator iter = getVolumesArray().iterator(); iter.hasNext();) {

                //
                Volume volume = (Volume) iter.next();

                //
                // FIXME : change request (see launchtransformation in volume.java)
                ArrayList allEntries = QueryRequest.findAllLexies(volume, user);

                //
                for (int i = 0; i < allEntries.size(); i++) {
                    //
                    QueryResult qr = (QueryResult) allEntries.get(i);
                    VolumeEntry ve = qr.getSourceEntry();

                    if (ve.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {

                        Collection classifiedFinishedContributionIdCollection = ve.getClassifiedFinishedContributionIdCollection();
                        for (Iterator iter2 = classifiedFinishedContributionIdCollection.iterator(); iter2.hasNext();) {
                            VolumeEntry finishedVe = VolumeEntriesFactory.findEntryByContributionId(ve.getVolumeName(),
                                    (String) iter2.next());

                            if (finishedVe.getStatus().equals(VolumeEntry.CLASSIFIED_FINISHED_STATUS)) {
                                finishedVe.setStatus(VolumeEntry.MODIFIED_STATUS);
                                finishedVe.setNextContributionAuthor(ve.getModificationAuthor());
                                finishedVe.save();
                            }
                        }
                    }

                }
            }

        } catch (Exception ex) {
            throw new PapillonBusinessException("Error when modify status", ex);
        }
    }


    /**
     * Standardization of all volumes.
     *
     * @throws PapillonBusinessException
     */
    public static void standardization()
            throws fr.imag.clips.papillon.business.PapillonBusinessException {

        // Standardization launch on all volumes
        for (Iterator iter = getVolumesArray().iterator(); iter.hasNext();) {
            Volume volume = (Volume) iter.next();

            //
            int count = volume.getCount();
            int delta = 20; // buffer limit
            PapillonLogger.writeDebugMsg("Volume : " + volume.getName() + " - " + count + " entries");

            //
            for (int i = 0; i < count; i = i + delta) {

                try {

                    // Begin transaction
                    CurrentDBTransaction.registerNewDBTransaction();

                    // Buffer volumeEntries
                    Collection bufferResults = VolumeEntriesFactory.getVolumeEntries(volume, i, delta);

                    //
                    Iterator buffer = bufferResults.iterator();
                    while (buffer.hasNext()) {
                        VolumeEntry ve = (VolumeEntry) buffer.next();

                        //
                        if (!ve.isEmpty()) {
                            //&& (ve.getXmlCode() != null)
                            //&& !ve.getXmlCode().equals("")) {

                            //
                            Document dom = ve.getDom();
                            Element root = dom.getDocumentElement();
                            normalizeNode((Node) root);
                            //PapillonLogger.writeDebugMsg(XMLServices.xmlCodePrettyPrinted(dom));
                            ve.save();

                        } else {

                            //
                            PapillonLogger.writeDebugMsg("Database return a null DO while applying a transformation");
                        }

                    }

                    // End transaction : a part was correct, commit the transaction ...
                    ((DBTransaction) CurrentDBTransaction.get()).commit();
                    PapillonLogger.writeDebugMsg(Integer.toString(i) + " to " + Integer.toString(i + delta - 1));

                } catch (Exception e) {
                    String userMessage = "Problems when normalize entries.";
                    PapillonLogger.writeDebugMsg(userMessage);
                    e.printStackTrace();

                    //
                    try {
                        ((DBTransaction) CurrentDBTransaction.get()).rollback();
                    } catch (java.sql.SQLException sqle) {
                        PapillonLogger.writeDebugMsg(
                                "standardization: SQLException while rolling back failed transaction.");
                        sqle.printStackTrace();
                    }

                } finally {
                    CurrentDBTransaction.releaseCurrentDBTransaction();
                }
            }

            //
            PapillonLogger.writeDebugMsg(volume.getName() + " : standardization succeed !");

        }

        //
        PapillonLogger.writeDebugMsg("Standardization process succeed !");
    }


    /**
     * normalize value node and child nodes.
     *
     * @param Node
     * @throws PapillonBusinessException
     */
    private static void normalizeNode(Node node) {

        while (node != null) {
            String value = node.getNodeValue();

            // Normalize value
            if (value != null && !value.equals("")) {
                node.setNodeValue(VolumeEntriesFactory.normalizeValue(value));
            }

            //
            normalizeNode(node.getFirstChild());
            node = node.getNextSibling();
        }
    }


    /**
     * Apply xsl transformation to all entries of the volume
     *
     * @param xslTransformation the xsl code
     * @param volume the volume on which to apply the transformation
     * @param user the user that is connected and applies the transformation
     * @throws PapillonBusinessException
     */
    public static void launchTransformation(String xslTransformation, Volume volume, User user)
            throws PapillonBusinessException {
        /*
        try {
            //
            TransformerFactory myTransformerFactory = TransformerFactory.newInstance();
            Transformer myTransformer = myTransformerFactory.newTransformer(new StreamSource(new StringReader(xslTransformation)));

            //
            ArrayList allEntries = QueryRequest.findAllLexies(this.getName(), user);

            //
            for (int i = 0; i < allEntries.size(); i++) {
                //
                QueryResult qr = (QueryResult) allEntries.get(i);
                VolumeEntry ve = qr.getSourceEntry();

                //
                DOMSource xmlSource = new DOMSource(Utility.buildDOMTree(ve.getXmlCode()));
                DOMResult xmlTarget = new DOMResult();

                //
                myTransformer.transform(xmlSource, xmlTarget);

                //
                //System.out.println(ve.getXmlCode());
                //System.out.println(Utility.NodeToString((Document) xmlTarget.getNode()))

                //
                ve.setDom((Document) xmlTarget.getNode());
                ve.save();
            }
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error deleting Volume", ex);
        }
        */


        try {
            // Get transformer
            TransformerFactory myTransformerFactory = TransformerFactory.newInstance();
            Transformer myTransformer = myTransformerFactory.newTransformer(
                    new StreamSource(new StringReader(xslTransformation)));

            // Truncate index volumes
            IndexFactory.truncateIndexTable(volume);

            //
            int count = volume.getCount();
            int delta = 20; // buffer limit
            PapillonLogger.writeDebugMsg("Volume : " + volume.getName() + " - " + count + " entries");

            //
            for (int i = 0; i < count; i = i + delta) {

                try {

                    // Begin transaction
                    CurrentDBTransaction.registerNewDBTransaction();

                    // Buffer volumeEntries
                    // Replace getVolumeEntriesVector because "order by" expression
                    //Collection bufferResults = VolumeEntriesFactory.getVolumeEntriesVector(DictionariesFactory.getDictionaryByName(volume.getDictname()), volume, null, null, null, z, delta);
                    Collection bufferResults = VolumeEntriesFactory.getVolumeEntries(volume, i, delta);

                    //
                    Iterator buffer = bufferResults.iterator();
                    while (buffer.hasNext()) {
                        VolumeEntry ve = (VolumeEntry) buffer.next();

                        //
                        if (!ve.isEmpty()) {
                            //&& (ve.getXmlCode() != null)
                            //&& !ve.getXmlCode().equals("")) {

                            //
                            //PapillonLogger.writeDebugMsg("--------------> Entry ID : " + ve.getEntryId());
                            //PapillonLogger.writeDebugMsg(XMLServices.xmlCodePrettyPrinted((Document)ve.getDom()));

                            //
                            DOMSource xmlSource = new DOMSource(ve.getDom());
                            DOMResult xmlTarget = new DOMResult();

                            //
                            myTransformer.transform(xmlSource, xmlTarget);
                            //PapillonLogger.writeDebugMsg(Utility.NodeToString((Document) xmlTarget.getNode()))

                            //
                            ve.setDom((Document) xmlTarget.getNode());
                            ve.save();

                        } else {

                            //
                            PapillonLogger.writeDebugMsg("Database return a null DO while applying a transformation");
                        }

                    }

                    // End transaction : a part was correct, commit the transaction ...
                    ((DBTransaction) CurrentDBTransaction.get()).commit();
                    PapillonLogger.writeDebugMsg(Integer.toString(i) + " to " + Integer.toString(i + delta - 1));


                } catch (Exception e) {
                    String userMessage = "Problems when transform entries.";
                    PapillonLogger.writeDebugMsg(userMessage);
                    e.printStackTrace();

                    try {
                        ((DBTransaction) CurrentDBTransaction.get()).rollback();
                    } catch (java.sql.SQLException sqle) {
                        PapillonLogger.writeDebugMsg(
                                "launchTransformation: SQLException while rolling back failed transaction.");
                        sqle.printStackTrace();
                    }
                } finally {
                    CurrentDBTransaction.releaseCurrentDBTransaction();
                }
            }

        } catch (javax.xml.transform.TransformerConfigurationException e) {
            e.printStackTrace();
        }

        //
        PapillonLogger.writeDebugMsg(volume.getName() + " : transformation succeed !");

    }


    public static void normalizeHeadwords()
            throws PapillonBusinessException {
        // TODO: process all entries and normalize its headword.
        java.util.Vector myKeys = new java.util.Vector();
        java.util.Vector myClauses = new java.util.Vector();

        String[] CNFS = new String[4];
        CNFS[0] = Volume.CDM_contributionStatus;
        CNFS[1] = Volume.DEFAULT_LANG;
        CNFS[2] = VolumeEntry.FINISHED_STATUS;
        CNFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL + 1];
        myKeys.add(CNFS);

        String[] CFS = new String[4];
        CFS[0] = Volume.CDM_contributionStatus;
        CFS[1] = Volume.DEFAULT_LANG;
        CFS[2] = VolumeEntry.MODIFIED_STATUS;
        CFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL + 1];
        myKeys.add(CFS);
        //
        for (Iterator iter = (VolumesFactory.getVolumes()).iterator(); iter.hasNext();) {
            Volume myVolume = (Volume) iter.next();
			if (myVolume !=null && !myVolume.isEmpty()) {
				Dictionary myDict = DictionariesFactory.getDictionaryByName(myVolume.getDictname());
				if (myDict !=null && !myDict.isEmpty()) {

						fr.imag.clips.papillon.business.dictionary.IVolumeEntryProcessor myProcessor = new fr.imag.clips.papillon.business.dictionary.NormalizeHeadwordProcessor();
						PapillonLogger.writeDebugMsg("Processor created");

						fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory.processVolume(myDict, myVolume, myKeys, myClauses, myProcessor, false);
						PapillonLogger.writeDebugMsg("Volume processed");

				}
			}
        }
    }
    
    public static Element getFirstElement(Element parent)
    {
        for(Node child = parent.getFirstChild(); child != null; child = child.getNextSibling())
        {
            if (child instanceof Element) return (Element) child;
        }
        return null;
    }

}

