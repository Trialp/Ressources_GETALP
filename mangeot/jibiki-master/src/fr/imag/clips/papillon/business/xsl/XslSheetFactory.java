/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.16  2007/04/05 12:55:54  serasset
 * Added a DBLayer Version management with an auto-update of db layer.
 *
 * Revision 1.15  2007/01/05 13:57:26  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.14  2006/08/10 22:56:01  fbrunet
 * *** empty log message ***
 *
 * Revision 1.13  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.12  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.11  2005/11/24 13:29:56  mangeot
 * findXslSheetByName does not throw an exception if more than one xslsheet bear the same name.
 *
 * Revision 1.10.2.2  2006/02/17 15:16:42  mangeot
 * Do not display the list of all XSL on the search form any more. Displays only a list of XSL descriptions
 *
 * Revision 1.10.2.1  2005/08/31 15:01:39  serasset
 * Applied modifications done on the LEXALP_1_0 branch to updated sources of the
 * trunk to create a new updated LEXALP_1_1 branch.
 *
 * Revision 1.10  2005/08/05 18:44:38  mangeot
 * Bug fixes + ProcessVolume.po page creation
 *
 * Revision 1.9.2.1  2005/07/22 13:28:32  serasset
 * Modified EditEntryInit for Lexalp. It now serves as a main page for db maintenance.
 * Added a function to get url for QueryParameter.
 * Modified the way xslsheets are handled in order to allow several xslsheet with the same name, different dicts.
 *
 * Revision 1.9  2005/07/21 09:37:47  serasset
 * LexALPLinker had a pb with package since MM modification.
 * Lexalp query menu leads to AdvancedSearch.
 * XslSheetFactory's get default xsl for dict and volume now sets the names to "" during fallback.
 *
 * Revision 1.8  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 * Revision 1.7  2005/07/16 12:33:23  mangeot
 * dictionaryname and volumename of xslsheets are now not null
 *
 * Revision 1.6  2005/07/14 13:48:53  serasset
 * Added columns dictionaryname and volumename to the xslsheets.
 * Modified the XslSheet and XslSheetFactory accordingly.
 * Modified AdminXsl interface accordingly.
 * Modified DictionariesFactory and VolumesFactory to allow several xsl-sheets and to
 * correctly provide dictionaryName/volumeName to XslSheetFactory.
 * Cleanup of several errors in papillon_static doml file.
 *
 * Revision 1.5  2005/06/15 16:48:28  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.4  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.3.4.1  2005/06/01 08:38:43  mangeot
 * Multi bug correction + added the possibility of disabling data edition
 * via the Admin.po page
 *
 * Revision 1.3  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.2.2.3  2005/03/29 09:41:33  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.2.2.2  2005/03/16 13:24:31  serasset
 * Modified all boolean fields in table to CHAR(1) in order to be more db independant.
 * Suppressed ant.jar from class path, informationfiles (which rely on it) should be corrected.
 * The version of Xerces is now displayed on application init.
 *
 * Revision 1.2.2.1  2005/01/27 15:56:21  mangeot
 * Able to load a volume with XPointers, cannot lookup the result yet.
 * Does not compile but commit for backup
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:38  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.xsl;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.CurrentDBTransaction;

import fr.imag.clips.papillon.data.XslSheetDO;
import fr.imag.clips.papillon.data.XslSheetQuery;

//for URLs
import java.net.URL;

import java.lang.Exception;
import java.util.Collection;
import java.util.Iterator;
import java.util.ArrayList;

//pour les nouvelles entrees
import org.w3c.dom.*;
import java.util.Properties;
//import org.apache.xalan.xsltc.runtime.AbstractTranslet;
import javax.xml.transform.Transformer;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.dom.DOMResult;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;


/**
* Used to find the instances of xslsheet.
 */
public class XslSheetFactory {
    
    // FIXME: for Xalan 2_7_0
    // Transformer factory properties
    private static final String TransformerFactoryKey = "javax.xml.transform.TransformerFactory";
     private static final String TransformerFactoryValue = "org.apache.xalan.processor.TransformerFactoryImpl";
//   private static final String TransformerFactoryValue = "org.apache.xalan.xsltc.trax.TransformerFactoryImpl";
    private static final String SAXParserFactoryKey = "javax.xml.parsers.SAXParserFactory";
    private static final String SAXParserFactoryValue = "org.apache.xerces.jaxp.SAXParserFactoryImpl";
    
    // FIXME: for Xalan 2_7_0
    // Jibiki xsl translet URIs
    //private static final String jibikiXslTransletXMLtoXMLURI = "fr.imag.clips.papillon.resources.xsl.XML-view";
    //private static final String jibikiXslTransletXMLtoTEXTURI = "fr.imag.clips.papillon.resources.xsl.TEXT-view";
    //private static final String jibikiXslTransletXMLtoFOURI = "fr.imag.clips.papillon.resources.xsl.FormatingObject-view";
    
    // FIXME: for Xalan 2_7_0
    // Jibiki xsl templates
    //private static Templates jibikiXslTemplateXMLtoXML = null;
    //private static Templates jibikiXslTemplateXMLtoText = null;
    //private static Templates jibikiXslTemplateXMLtoFO = null;
    
    // FIXME: for Xalan 2_4_1
    // Jibiki xsl XslSheet
    private static XslSheet jibikiXslXMLtoXML = null;
    private static XslSheet jibikiXslXMLtoText = null;
    private static XslSheet jibikiXslXMLtoFO = null;
    
	// Result preparation : Document builder
	protected static final DocumentBuilderFactory myDocumentBuilderFactory = DocumentBuilderFactory.newInstance();
	protected static DocumentBuilder myDocumentBuilder = null; 
    
    
    /** 
        * This method initialize the transformer factory properties.
        *
        * @exception PapillonBusinessException
        */
    // FIXME: for Xalan 2_7_0
    public static void initializeTransformerFactory() 
        throws PapillonBusinessException {
            
            try {
                
                // Set the TransformerFactory system property.
                // Note: For more flexibility, load properties from a properties file.
                Properties props = System.getProperties();
                props.put(TransformerFactoryKey, TransformerFactoryValue);
                props.put(SAXParserFactoryKey, SAXParserFactoryValue);
                System.setProperties(props);
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in initializeTransformerFactory()", ex);
            }
         }

     /** 
        * This method initialize the jibiki xsl.
        *
        * @exception PapillonBusinessException
        */
    public static void initializeJibikiXslSheet() 
        throws PapillonBusinessException {
            try {
                
                // FIXME: for Xalan 2_4_1
                jibikiXslXMLtoXML = getXslSheet("", "", "XML");
                if (jibikiXslXMLtoXML == null) {
					new PapillonBusinessException("jibiki XML to XML internal xsl is not in cache");
				}
                jibikiXslXMLtoText = getXslSheet("", "", "TEXT");
                jibikiXslXMLtoFO = getXslSheet("", "", "FO");
                
                // FIXME: for Xalan 2_7_0
                //Class transletClass = Class.forName(jibikiXslTransletXMLtoXMLURI);
                //AbstractTranslet translet = (AbstractTranslet)transletClass.newInstance();
                //jibikiXslXMLtoXML = translet.getTemplates();
                
                
            } catch (PapillonBusinessException ex) {
                  throw ex;
            } catch (Exception ex) {
                throw new PapillonBusinessException("Exception in initializeJibikiXslSheet()", ex);
            }
        }
    
    
    /** 
        * The initializeXslSheets method performs a database query to
        * initialize the xsl sheet cache
        *
        * @exception PapillonBusinessException
        *   If there is a problem retrieving disc information.
        */
    public static void initializeXslSheetCache() 
        throws PapillonBusinessException {
            
            try {
                
                // Initialize cache
                XslSheetCache.xslSheetCacheInit();
                
                // Perform query
                XslSheetQuery query = new XslSheetQuery(CurrentDBTransaction.get());
                query.addOrderByName(true);
                XslSheetDO[] DOarray = query.getDOArray();
                
                //
                for ( int i = 0; i < DOarray.length; i++ ) {
                    XslSheet xsl = new XslSheet(DOarray[i]);
                    
                    //
                    AddXslSheetInCache(xsl);
                }
                
            }catch(Exception ex) {
                throw new PapillonBusinessException("Exception in initializeXslSheetsCache()", ex);
            }
        }
    
        
    /** 
        * Return a collection of xsl sheets
        *
        * @return collection of xslSheets. 
        *     
        * @exception PapillonBusinessException
        */
    public static Collection getXslSheetsArray() 
        throws PapillonBusinessException {
            
            try {
                
                //
                return XslSheetCache.getXslSheetsInCache();
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getXslSheetsArray()", ex);
            }
        }
    
    /** 
        * Return a collection of external xsl sheets (visibles by the users).
        *
        * @return collection of xslSheets. 
        *     
        * @exception PapillonBusinessException
        */
    public static Collection getExternalXslSheetsArray() 
        throws PapillonBusinessException {
            
            try {
                
                //
                Collection xslSheetList = XslSheetCache.getXslSheetsInCache();
                
                //
                ArrayList UserXslSheetList = new ArrayList();
                
                //
                for (Iterator iter = xslSheetList.iterator(); iter.hasNext();) {
                    XslSheet xsl = (XslSheet) iter.next();
                    
                    //
                    if ( xsl.isExternalxsl()) {
                        UserXslSheetList.add(xsl);
                    }
                }
                
                //
                return UserXslSheetList;
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getUserXslSheetsArray()", ex);
            }
        }
    
    
    /** 
        * Return xsl string of the xsl url file  
        *
        * @param URL 
        *
        * @return String
        *     
        * @exception PapillonBusinessException
        */
    public static String parseXslSheet(URL fileURL) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            
            String result = null;
            
            try {
                Document docXml = XMLServices.buildDOMTree(fileURL);
                if (null != docXml) result= XMLServices.xmlCode(docXml);
                
                //  PapillonLogger.writeDebugMsg("The XSL sheet:");
                //  PapillonLogger.writeDebugMsg(result);
            }
            catch(Exception ex) {
                throw new PapillonBusinessException("Exception in parseXslSheet()", ex);
            }
            return result;
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
            
            // Initialize
            if ( (Name == null) || (Name.equalsIgnoreCase("Default"))) Name = "";
            if ( dictionaryName == null ) dictionaryName = "";
            if ( volumeName == null ) volumeName = "";
            
            //
            //PapillonLogger.writeDebugMsg("getXslSheet " + dictionaryName + "/" + volumeName + "/" + Name);
            return XslSheetCache.getXslSheetInCache(dictionaryName, volumeName, Name);

        }

	
    /** 
        * Return the xsl sheet corresponding to the 'handle' (database object id)
        *
        * @param String
        *
        * @return XslSheet
        *     
        * @exception PapillonBusinessException
        */    
    public static XslSheet getXslSheetByHandle(String handle) 
        throws PapillonBusinessException {
            
            //
            return XslSheetCache.getXslSheetInCacheByHandle(handle);
        }    
    
    
    
    /** 
        * Return the defaut xsl sheet  
        *
        * @param String
        * @param String
        *
        * @return XslSheet
        *     
        * @exception PapillonBusinessException
        */
    public static XslSheet findDefaultXslSheet(String dictionaryName, String volumeName)
        throws PapillonBusinessException {
            
            // find defaut xsl
            return getXslSheet(dictionaryName, volumeName, "");
            
        }
    
    
    /** 
        * Add a new xsl sheet in database. If xsl exist, this xsl is delete and reload
        *
        * @param String
        * @param String
        * @param String
        * @param String
        * @param String
        * @param boolean
        * @param boolean
        *     
        * @exception PapillonBusinessException
        */    
    public static void AddXslSheet(String name, String dictionaryName, String volumeName, String description, String code, boolean defaultXsl, boolean externalXsl)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            
            if ((name!=null) && (code!=null)) {
                
                //search for an existing
                //XslSheetFactory XslFactory=new XslSheetFactory();
                //XslSheet Existe=XslFactory.findXslSheetByName(name, dictionaryName, volumeName);
				String existName = defaultXsl?"":name;
                XslSheet Existe = XslSheetFactory.getXslSheet(dictionaryName, volumeName, existName);
                
                //
                if ((Existe != null) && (!Existe.isEmpty())) {
                    PapillonLogger.writeDebugMsg("Delete " + Existe.getName() + " xsl");
                    Existe.delete();
                }
                
                //
                XslSheet mySheet=new XslSheet();
                mySheet.setName(name);
                if (dictionaryName == null) {
                    dictionaryName = "";
                }
                mySheet.setDictionaryName(dictionaryName);
                if (volumeName == null) {
                    volumeName = "";
                }
                mySheet.setVolumeName(volumeName);
                mySheet.setDescription(description);
                mySheet.setCode(code);
                mySheet.setDefaultxsl(defaultXsl);
                mySheet.setExternalxsl(externalXsl);
                mySheet.save();
                PapillonLogger.writeDebugMsg("XslSheet: " + name + " is stored in the database");
                
                
                //
				initializeXslSheetCache();
            }
            else {
                PapillonLogger.writeDebugMsg("XslSheet ignored");
            }        
        }
    
    /** 
        * Add a new xsl sheet in cache.
        *
        * @param XslSheet
        *     
        * @exception PapillonBusinessException
        */    
    private static void AddXslSheetInCache(XslSheet xsl)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            
            // if xsl is defaut sheet, add it in cache with key composed of dictionary name and volume name (no xsl name)
            if (xsl.isDefaultxsl()) {
                PapillonLogger.writeDebugMsg("XslSheet " + xsl.getName() +" is default xsl");
                XslSheetCache.putXslSheetInCache(xsl.getDictionaryName(), xsl.getVolumeName(), "", xsl);
            }
			else {
				// Add xsl sheet in cache : key is composed of dictionary name, volume name and xsl name
				XslSheetCache.putXslSheetInCache(xsl.getDictionaryName(), xsl.getVolumeName(), xsl.getName(), xsl);
			}
        }
    
    public static void removeXslSheet(String name, String dictionaryName, String volumeName)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        XslSheet Existe = XslSheetFactory.getXslSheet(dictionaryName, volumeName, name);
        
        //
        if ((Existe != null) && (!Existe.isEmpty())) {
            PapillonLogger.writeDebugMsg("Delete " + Existe.getName() + " xsl");
            Existe.delete();
            initializeXslSheetCache();
        }
    }

    
    /**
        *
        *
        * @param String
        *     
        * @exception PapillonBusinessException
        */ 
    public static Node applyXslSheetForXml(String xmlString)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            
            try {   
                
                //
                Document xmlSource = XMLServices.buildDOMTree(xmlString);
                if (myDocumentBuilder == null) {
                    myDocumentBuilder = myDocumentBuilderFactory.newDocumentBuilder();
                }        
                Document xmlTarget = myDocumentBuilder.newDocument();
                
                // FIXME: for Xalan 2_4_1
                Transformer transformer = jibikiXslXMLtoXML.getTransformer();
                transformer.transform(new DOMSource(xmlSource), new DOMResult(xmlTarget));
                
                // FIXME: for Xalan 2_7_0
                //Transformer transformer = jibikiXslTemplateXMLtoXML.newTransformer();
                //transformer.transform(new DOMSource(xmlSource), new DOMResult(xmlTarget));
                
                //
                return xmlTarget.getDocumentElement();
                
            } catch(Exception ex) {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in applyXslSheetForXml()", ex);
			}
        }
    
    
    
    /** 
        * Remove all xsl sheets in database and in cache
        *
        * @exception PapillonBusinessException
        *   If there is a problem retrieving disc information.
        +/
        public static void removeAllXslSheets()
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            
            // Delete all xsl sheets in database
            Collection TheXslSheets= XslSheetFactory.getXslSheetsArray();
            for ( Iterator iter = TheXslSheets.iterator(); iter.hasNext(); ) {
                
                //
                XslSheet xsl = (XslSheet)iter.next();
                xsl.delete();
            }
            
            // Re-Initialize cache
            XslSheetFactory.initializeXslSheetCache();
        }
    */
    
    
	/*
     public static void AddXslSheet(String name, String dictionaryName, String volumeName, String description,String code, boolean defaultXsl)
     throws fr.imag.clips.papillon.business.PapillonBusinessException{
         if ((name!=null) && (code!=null)) {
             
             //search for an existing
             //XslSheetFactory XslFactory=new XslSheetFactory();
             //XslSheet Existe=XslFactory.findXslSheetByName(name);
             XslSheet Existe = XslSheetFactory.getXslSheet(dictionaryName, volumeName, name);
             
             if (Existe.isEmpty()) {
                 XslSheet mySheet=new XslSheet();
                 mySheet.setName(name);
                 if (dictionaryName == null) {
                     dictionaryName = "";
                 }
                 mySheet.setDictionaryName(dictionaryName);
                 if (volumeName == null) {
                     volumeName = "";
                 }
                 mySheet.setVolumeName(volumeName);
                 mySheet.setDescription(description);
                 mySheet.setCode(code);
                 mySheet.setDefaultxsl(defaultXsl);
                 mySheet.save();
                 PapillonLogger.writeDebugMsg("XslSheet: " + mySheet.getName() + " is stored in the database");
             }
             else {
                 PapillonLogger.writeDebugMsg("Existing XslSheet in the database");
                 PapillonLogger.writeDebugMsg("Name: "+Existe.getName());
                 PapillonLogger.writeDebugMsg("Description: "+Existe.getDescription());
             }
         }
         else {
             PapillonLogger.writeDebugMsg("XslSheet ignored");
         }
     }
     */
    
    
    
    // A MODIFIER ... mettre en place une collection de sheets externe (visible par l'utilisateur)
	//protected static java.util.Set descriptionSet = null;  
    
    /** 
        * A MODIFIER ... mettre en place une collection de sheets externe (visible par l'utilisateur)
        * The getDescriptionSet method returns a set of XslSheets descriptions
        * @return
        *    set of XslSheet descriptions. 
        * @exception PapillonBusinessException
        *   If there is a problem retrieving disc information.
        +/
        public static java.util.Set getDescriptionSet()
        throws PapillonBusinessException {
            
            if (descriptionSet == null) {
                descriptionSet = new java.util.TreeSet();
                
                Collection theArray = getXslSheetsArray();
                for (Iterator iter = theArray.iterator() ; iter.hasNext();) {
                    String description = ((XslSheet)iter.next()).getDescription();
                    if (description != null && !description.equals("")) {
                        descriptionSet.add(description);
                    }
                }
            }
            return descriptionSet;
        }
    */
    
    
    
    /*
     public static boolean resetCache() {
         descriptionSet = null;
         return true;
     }
     */
    
    /** 
        * The getXslSheetArray method performs a database query to
        * return an array of XslSheets for the given dictionary
        * @param dictionaryName
        * @return
        *     array of xslSheets for the given dictionary + global xslSheets. 
        * @exception PapillonBusinessException
        *   If there is a problem retrieving disc information.
        *
        public static XslSheet[] getApplicableXslSheets(String dictionaryName) 
        throws PapillonBusinessException {
            Vector xslSheets = new Vector();
            
            try {
                XslSheetQuery query = new XslSheetQuery(CurrentDBTransaction.get());
                query.setQueryDictionaryName(dictionaryName);
                query.addOrderByName(true);
                XslSheetDO[] DOarray = query.getDOArray();
                for ( int i = 0; i < DOarray.length; i++ )
                    xslSheets.add(new XslSheet(DOarray[i]));
                
                query.reset();
                query.setQueryDictionaryName(null);
                query.addOrderByName(true);
                DOarray = query.getDOArray();
                for ( int i = 0; i < DOarray.length; i++ )
                    xslSheets.add(new XslSheet(DOarray[i]));
                
            }catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getXslSheetsArray()", ex);
            }
            
            return (XslSheet []) xslSheets.toArray((XslSheet []) null);
        }
    
    /** 
        * The getXslSheetArray method performs a database query to
        * return an array of XslSheets for the given dictionary
        * @param dictionaryName
        * @param volumeName
        * @return
        *     array of xslSheets for the given dictionary + global xslSheets. 
        * @exception PapillonBusinessException
        *   If there is a problem retrieving disc information.
        *
        public static XslSheet[] getApplicableXslSheets(String dictionaryName, String volumeName) 
        throws PapillonBusinessException {
            Vector xslSheets = new Vector();
            
            try {
                XslSheetQuery query = new XslSheetQuery(CurrentDBTransaction.get());
                query.setQueryDictionaryName(dictionaryName);
                query.setQueryVolumeName(volumeName);
                query.addOrderByName(true);
                XslSheetDO[] DOarray = query.getDOArray();
                for ( int i = 0; i < DOarray.length; i++ )
                    xslSheets.add(new XslSheet(DOarray[i]));
                
                query.reset();
                query.setQueryDictionaryName(null);
                query.addOrderByName(true);
                DOarray = query.getDOArray();
                for ( int i = 0; i < DOarray.length; i++ )
                    xslSheets.add(new XslSheet(DOarray[i]));
                
            }catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getXslSheetsArray()", ex);
            }
            
            return (XslSheet []) xslSheets.toArray((XslSheet []) null);
        }*/
    
    /**
        *
     *
     *
     *
     public static XslSheet getDefaultXslSheet(String dictionaryName, String volumeName) throws PapillonBusinessException {
         XslSheet theXsl = null;
         
         System.out.println("getDefaultXslSheet : " + dictionaryName + " " + volumeName);
         
         try {
             XslSheetQuery query = new XslSheetQuery(CurrentDBTransaction.get());
             //set query
             query.setQueryDictionaryName(dictionaryName);
             query.setQueryVolumeName(volumeName);
             query.setQueryDefaultxsl("Y");
             
             // Throw an exception if more than one query is found
             query.requireUniqueInstance();
             XslSheetDO theXslSheetDO = query.getNextDO();
             
             if (null == theXslSheetDO) {
                 query.reset();
                 query.setQueryDictionaryName(dictionaryName);
                 query.setQueryVolumeName("");
                 query.setQueryDefaultxsl("Y");
                 query.requireUniqueInstance();
                 theXslSheetDO = query.getNextDO();
                 if (null == theXslSheetDO) {
                     query.reset();
                     query.setQueryDictionaryName("");
                     query.setQueryVolumeName("");
                     query.setQueryDefaultxsl("Y");
                     query.requireUniqueInstance();
                     theXslSheetDO = query.getNextDO();
                 }
             }
             theXsl = new XslSheet(theXslSheetDO);
             if (null == theXsl) {
                 // Here we should generate an exception or display an HTML message
                 fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Exception in findDefaultxslsheet(): no default existing XSL sheet");
             }
             return theXsl;
         } catch(Exception ex) {
             throw new PapillonBusinessException("Exception in getDefaultXslSheet()", ex);
         }
     }*/
    
    /** 
        * The getXslSheetArray method performs a database query to
        * return an array of XslSheets
        * @return
        *     array of xslSheets. 
        * @exception PapillonBusinessException
        *   If there is a problem retrieving disc information.
        *
        public static XslSheet[] getXslSheetsArray() 
        throws PapillonBusinessException {
            XslSheet[] theXslArray = null;
            
            try {
                XslSheetQuery query = new XslSheetQuery(CurrentDBTransaction.get());
                query.addOrderByName(true);
                XslSheetDO[] DOarray = query.getDOArray();
                theXslArray = new XslSheet[ DOarray.length ];
                for ( int i = 0; i < DOarray.length; i++ )
                    theXslArray[i] = new XslSheet(DOarray[i]);
                
            }catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getXslSheetsArray()", ex);
            }
            
            return theXslArray;
        }*/
    
    /** 
        * The findMessageByID method performs a database query to
        * return a <CODE>Message</CODE> object
        * representing the row in the <CODE>message</CODE> table
        * that matches the object id. 
        *
        * @param id, the object id of the message table.
        * @return
        *    the message. null if there isn't a message associated
        *    to the id
        * @exception PapillonBusinessException
        *    if there is a problem retrieving message.
        *
        public static XslSheet findXslSheetByHandle(String id) 
        throws PapillonBusinessException {
            XslSheet theXsl = null;
            
            System.out.println("findXslSheetByHandle : " + id);
            
            try {
                XslSheetQuery query = new XslSheetQuery(CurrentDBTransaction.get());
                //set query
                query.setQueryOId(new ObjectId(id));
                // Throw an exception if more than one message is found
                query.requireUniqueInstance();
                XslSheetDO theXslSheetDO = query.getNextDO();
                theXsl = new XslSheet(theXslSheetDO);
                return theXsl;
            }catch(Exception ex) {
                throw new PapillonBusinessException("Exception in findXslSheetByHandle()", ex);
            }
        }*/
    
    
    /** 
        * Return the xsl sheet corresponding to the name 
        *
        * @param String
        *
        * @return XslSheet
        *     
        * @exception PapillonBusinessException
        +/
        public static XslSheet findXslSheetByName(String Name) 
        throws PapillonBusinessException {
            
            //
            return XslSheetCache.getXslSheetInCache(name);
            
            /*
             XslSheet theXsl = null;
             
             System.out.println("findXslSheetByName : " + Name);
             
             try {
                 XslSheetQuery query = new XslSheetQuery(CurrentDBTransaction.get());
                 //set query
                 query.setQueryName(Name);
                 // Now several xslsheets can use the same name.
                 //query.requireUniqueInstance();
                 XslSheetDO theXslSheetDO = query.getNextDO();			
                 theXsl = new XslSheet(theXslSheetDO);
             }catch(Exception ex) {
                 throw new PapillonBusinessException("Exception in findXslSheetByID()", ex);
             }
             return theXsl;
             +/
        }
             */
            
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
            +/
            public static XslSheet findXslSheetByName(String Name, String dictionaryName, String volumeName) 
            throws PapillonBusinessException {
                XslSheet theXsl = null;
                
                System.out.println("findXslSheetByName : " + Name  + " " +  dictionaryName  + " " +  volumeName);
                
                try {
                    XslSheetQuery query = new XslSheetQuery(CurrentDBTransaction.get());
                    //set query
                    query.setQueryDictionaryName(dictionaryName);
                    query.setQueryVolumeName(volumeName);
                    query.setQueryName(Name);
                    // Throw an exception if more than one message is found
                    query.requireUniqueInstance();
                    XslSheetDO theXslSheetDO = query.getNextDO();			
                    theXsl = new XslSheet(theXslSheetDO);
                }catch(Exception ex) {
                    throw new PapillonBusinessException("Exception in findXslSheetByName()", ex);
                }
                return theXsl;
            }*/
}

