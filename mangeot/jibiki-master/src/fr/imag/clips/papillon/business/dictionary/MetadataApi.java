/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 *
 */

package fr.imag.clips.papillon.business.dictionary;

import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
import com.lutris.appserver.server.sql.DBTransaction;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.user.Group;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.Papillon;


public class MetadataApi {

    protected static final String DICTLIST_XMLSTRING = "<?xml version='1.0' encoding='UTF-8'?><d:dictionary-metadata-list "
    + "xmlns:d='http://www-clips.imag.fr/geta/services/dml'>"
    + "</d:dictionary-metadata-list>";

    public static String JSON_CONTENTTYPE = "text/json";
    public static String XML_CONTENTTYPE = "text/xml";
    
    public static java.util.Vector getDictionaryList(User theUser)
    throws PapillonBusinessException {
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = XMLServices.buildDOMTree(DICTLIST_XMLSTRING);
        String errorMsg = "";
        
        java.util.Collection dictCollection = DictionariesFactory.getDictionariesArray();
        if (dictCollection !=null && dictCollection.size()>0) {
            java.util.Iterator dictItr = dictCollection.iterator();
            while (dictItr.hasNext()) {
                Dictionary theDict = (Dictionary) dictItr.next();
                if (userCanAccessMetadata(theUser,theDict)) {
                    
                    String dictFiles = "<" + DictionariesFactory.DICTIONARY_FILES_TAG + ">"
                    + XMLServices.trimXmlDeclaration(theDict.getXmlCode());
                    XslSheet defaultDictXslSheet = XslSheetFactory.getXslSheet(theDict.getName(),null,"");
                    if (defaultDictXslSheet!=null) {
                        dictFiles += XMLServices.trimXmlDeclaration(defaultDictXslSheet.getXmlCode());
                    }
                    dictFiles += "</" + DictionariesFactory.DICTIONARY_FILES_TAG + ">";
                    org.w3c.dom.Document theDictDom = XMLServices.buildDOMTree(dictFiles);
                    org.w3c.dom.Node theDictImported = content.importNode(theDictDom.getDocumentElement(), true);
                    content.getDocumentElement().appendChild(theDictImported);
                }
            }
        }
    
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        
        return responseVector;
    }

    public static java.util.Vector getDictionary(String dictName, User theUser)
        throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        Dictionary theDict  = DictionariesFactory.getDictionaryByName(dictName);
        if (theDict !=null && !theDict.isEmpty()) {
            if (userCanAccessMetadata(theUser,theDict)) {
                String dictFiles = "<d:" + DictionariesFactory.DICTIONARY_FILES_TAG + " xmlns:d='http://www-clips.imag.fr/geta/services/dml'>"
        + XMLServices.trimXmlDeclaration(theDict.getXmlCode());
                XslSheet defaultDictXslSheet = XslSheetFactory.getXslSheet(theDict.getName(),null,"");
                if (defaultDictXslSheet!=null) {
                    dictFiles += XMLServices.trimXmlDeclaration(defaultDictXslSheet.getXmlCode());
                }
                dictFiles += "</d:" + DictionariesFactory.DICTIONARY_FILES_TAG + ">";
                content = XMLServices.buildDOMTree(dictFiles);
            }
        }
        if (content == null) {
            errorMsg = "Error: dict: " + dictName + " does not exist!";
            status = HttpPresentationResponse.SC_NOT_FOUND;
        }

        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    public static java.util.Vector postDictionary(String dictName, String dictXml, String contentType, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_CREATED;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
       
        org.w3c.dom.Document dictDom = null;
        if (contentType.equals(XML_CONTENTTYPE)) {
            try {
                dictDom = XMLServices.buildDOMTree(dictXml);
            }
            catch (Exception e) {
                dictXml = "";
            }
            if (dictXml != null && !dictXml.equals("")) {
                Dictionary existDict = DictionariesFactory.getDictionaryByName(dictName);
                if (existDict ==null || existDict.isEmpty()) {
                    Dictionary theDict = DictionariesFactory.parseDictionaryMetadata(dictDom, null, false, false, false);
                    if (theDict !=null && !theDict.isEmpty()) {
                        if (userCanHandleMetadata(theUser,theDict)) {
                            theDict.save();
                            String userMessage = "adding " + theDict.getName() + " dictionary" + " // " + theDict.getCategory() + " // " + theDict.getType() + " // " + theDict.getDomain() + " // " + theDict.getLegal() + " // " + theDict.getSourceLanguages() + " // " + theDict.getTargetLanguages();
                            PapillonLogger.writeDebugMsg(userMessage);
                           content = XMLServices.buildDOMTree(theDict.getXmlCode());
                            status = HttpPresentationResponse.SC_CREATED;
                        }
                        else {
                            // il faut tout supprimer
                            theDict.deleteUserGroups();
                            theDict.deleteDefaultXslSheet();
                            
                            // Il faut vider le cache
                            DictionariesFactory.initializeDictionaryCache();
                            String login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():"";
                            errorMsg = "Error: user: " + login +" not authorized to post dict!";
                            status = HttpPresentationResponse.SC_UNAUTHORIZED;
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                        }
                    }
                    else {
                        errorMsg = "Error: dictionary metadata for dict: " + dictName + " is not semantically correct!";
                        status = 422;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                    }
                 }
                else {
                    errorMsg = "Error: conflict, dict: " + dictName + " already exists. Try to choose another name!";
                    status = 409;
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Conflict</h1><p>" + errorMsg + "</p></html>");
                }
            }
            else {
                errorMsg = "Error: dictionary metadata: <![CDATA["+ dictXml +"]]> XML is malformed!";
                status = HttpPresentationResponse.SC_BAD_REQUEST;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            errorMsg = "Error: only XML content type allowed!";
            status = 415;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unsupported Media Type</h1><p>" + errorMsg + "</p></html>");
        }
        
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    public static java.util.Vector putDictionary(String dictName, String dictXml, String contentType, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        org.w3c.dom.Document dictDom = null;
        if (contentType.equals(XML_CONTENTTYPE)) {
            try {
                dictDom = XMLServices.buildDOMTree(dictXml);
            }
            catch (Exception e) {
                dictXml = "";
            }
            if (dictXml != null && !dictXml.equals("")) {
                Dictionary theDict = DictionariesFactory.parseDictionaryMetadata(dictDom, null, false, false, false);
                if (theDict !=null && !theDict.isEmpty()) {
                    if (userCanHandleMetadata(theUser, theDict)) {
                        Dictionary existDict = DictionariesFactory.getDictionaryByName(dictName);
                        if (existDict !=null && !existDict.isEmpty()) {
                            existDict.delete();
                            String userMessage = "modifying " + theDict.getName() + " dictionary" + " // " + theDict.getCategory() + " // " + theDict.getType() + " // " + theDict.getDomain() + " // " + theDict.getLegal() + " // " + theDict.getSourceLanguages() + " // " + theDict.getTargetLanguages();
                            PapillonLogger.writeDebugMsg(userMessage);
                            theDict.save();
                            content = XMLServices.buildDOMTree(theDict.getXmlCode());
                            status = HttpPresentationResponse.SC_NO_CONTENT;
                        }
                        else {
                            errorMsg = "Error: dict: " + dictName + " does not exist!";
                            status = HttpPresentationResponse.SC_NOT_FOUND;
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                        }
                    }
                    else {
                        String login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():"";
                        errorMsg = "Error: user: " + login +" not authorized to post dict!";
                        status = HttpPresentationResponse.SC_UNAUTHORIZED;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                    }
                }
                else {
                    errorMsg = "Error: dictionary metadata for dict: " + dictName + " is not semantically correct!";
                    status = 422;
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                }
            }
            else {
                errorMsg = "Error: dictionary metadata: <![CDATA["+ dictXml +"]]> XML is malformed!";
                status = HttpPresentationResponse.SC_BAD_REQUEST;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            errorMsg = "Error: only XML content type allowed!";
            status = 415;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unsupported Media Type</h1><p>" + errorMsg + "</p></html>");
        }
        
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    public static java.util.Vector deleteDictionary(String dictName, User theUser)
    throws PapillonBusinessException {
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_NO_CONTENT;
        org.w3c.dom.Document content = null;
        String errorMsg = "";

        Dictionary theDict = DictionariesFactory.getDictionaryByName(dictName);
        if (theDict !=null && !theDict.isEmpty()) {
            if (userCanHandleMetadata(theUser,theDict)) {
                content = XMLServices.buildDOMTree(theDict.getXmlCode());
                theDict.deleteAll();
                fr.imag.clips.papillon.Papillon.initializeAllCaches();
                if (content==null) {
                    errorMsg = "Unknown Error: dictionary metadata for dict: " + dictName + " not deleted!";
                    status = 422;
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                }
                else {
                    status = HttpPresentationResponse.SC_NO_CONTENT;
                }
            }
            else {
                String login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():"";
                errorMsg = "Error: user: " + login +" not authorized to delete dict!";
                status = HttpPresentationResponse.SC_UNAUTHORIZED;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            errorMsg = "Error: dict: " + dictName + " does not exist!";
            status = HttpPresentationResponse.SC_NOT_FOUND;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    
    public static java.util.Vector getVolume(String dictName, String lang, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
 
        java.util.Collection volumesCollection = VolumesFactory.getVolumesArray(dictName,lang,null);
        if (volumesCollection !=null && volumesCollection.size()>0) {
            Volume theVolume = (Volume) volumesCollection.iterator().next();
            Dictionary theDict = DictionariesFactory.getDictionaryByName(dictName);
            if (userCanAccessMetadata(theUser,theDict)) {
                PapillonLogger.writeDebugMsg("Volume metadata: " + theVolume.getName());
                String volumeFiles = "<d:" + VolumesFactory.VOLUME_FILES_TAG + " xmlns:d='http://www-clips.imag.fr/geta/services/dml'>"
                + XMLServices.trimXmlDeclaration(theVolume.getXmlCode())
                + "<d:" + VolumesFactory.TEMPLATE_ENTRY_TAG + ">"
                + XMLServices.trimXmlDeclaration(theVolume.getTemplateEntry())
                + "</d:" + VolumesFactory.TEMPLATE_ENTRY_TAG + ">";
                if (theVolume.getTemplateInterface()!=null) {
                    volumeFiles += "<d:" + VolumesFactory.TEMPLATE_INTERFACE_TAG + ">"
                    + XMLServices.trimXmlDeclaration(theVolume.getTemplateInterface())
                    + "</d:" + VolumesFactory.TEMPLATE_INTERFACE_TAG + ">";
                }
                if (theVolume.getXmlSchema()!=null) {
                    volumeFiles += XMLServices.trimXmlDeclaration(theVolume.getXmlSchema());
                }
                XslSheet defaultVolumeXslSheet = XslSheetFactory.getXslSheet(theVolume.getDictname(),theVolume.getName(),"");
                if (defaultVolumeXslSheet!=null) {
                    volumeFiles += XMLServices.trimXmlDeclaration(defaultVolumeXslSheet.getXmlCode());
                }
                volumeFiles += "</d:" + VolumesFactory.VOLUME_FILES_TAG + ">";
                //PapillonLogger.writeDebugMsg(volumeFiles);
                content = XMLServices.buildDOMTree(volumeFiles);
            }
        }
        if (content == null) {
            errorMsg = "Error: volume: " + dictName + " lang: " + lang + " does not exist!";
            status = HttpPresentationResponse.SC_NOT_FOUND;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
 
    public static java.util.Vector postVolume(String dictName, String lang, String volumeXml, String contentType, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_CREATED;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        String exceptionMessage = "";
        
        org.w3c.dom.Document volDom = null;
        if (contentType.equals(XML_CONTENTTYPE)) {
            try {
                volDom = XMLServices.buildDOMTree(volumeXml);
            }
            catch (Exception e) {
                volumeXml = "";
            }
            if (volumeXml != null && !volumeXml.equals("")) {
                Dictionary theDict = DictionariesFactory.getDictionaryByName(dictName);
                if (theDict !=null && !theDict.isEmpty()) {
                    if (userCanHandleMetadata(theUser, theDict)) {
                        java.util.Collection volumesCollection = VolumesFactory.getVolumesArray(dictName,lang,null);
                        if (volumesCollection ==null || volumesCollection.size()==0) {
                            
                            // Create and Register the transaction
                            CurrentDBTransaction.registerNewDBTransaction();
                            Volume theVolume = null;
                            try {
                                theVolume = VolumesFactory.parseVolumeMetadata(theDict, volDom, null, false, false);
                                
                                if (theVolume !=null && !theVolume.isEmpty()) {
                                    fr.imag.clips.papillon.business.edition.UITemplates.resetCache();
                                    VolumeEntriesFactory.resetCountCache(theVolume.getName());
                                    theVolume.getCount();
                                    Papillon.initializeAllCaches();
                                    String userMessage = "Adding " + theVolume.getName() + " volume: " + theVolume.getDictname() + " // "  + theVolume.getDbname() + " // " + theVolume.getSourceLanguage() + " // " + theVolume.getTargetLanguages() + " // " + theVolume.getVolumeRef();
                                    PapillonLogger.writeDebugMsg(userMessage);
                                    
                                }
                                else {
                                    errorMsg = "Error: volume metadata for volume: " + dictName + " lang: " + lang + " is not semantically correct! ";
                                    status = 422;
                                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                                }
                                // everything was correct, commit the transaction...
                                try {
                                    ((DBTransaction) CurrentDBTransaction.get()).commit();
                                } catch (java.sql.SQLException sqle) {
                                    PapillonLogger.writeDebugMsg("AdminVolumes: SQLException while commiting transaction.");
                                    sqle.printStackTrace();
                                    errorMsg = "Error: volume metadata for volume: " + dictName + " lang: " + lang + " is not semantically correct! " + sqle.getMessage();
                                    status = 422;
                                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                                }
                            } catch (PapillonBusinessException e) {
                                e.printStackTrace();
                                errorMsg = "Error: volume metadata for volume: " + dictName + " lang: " + lang + " is not semantically correct! " + e.getMessage();
                                status = 422;
                                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                                try {
                                    ((DBTransaction) CurrentDBTransaction.get()).rollback();
                                } catch (java.sql.SQLException sqle) {
                                    PapillonLogger.writeDebugMsg("AdminVolumes: SQLException while rolling back failed transaction.");
                                    sqle.printStackTrace();
                                    errorMsg = "Error: volume metadata for volume: " + dictName + " lang: " + lang + " is not semantically correct! " + sqle.getMessage();
                                    status = 422;
                                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");

                                }
                            } finally {
                                CurrentDBTransaction.releaseCurrentDBTransaction();
                                content = XMLServices.buildDOMTree(theVolume.getXmlCode());
                                status = HttpPresentationResponse.SC_CREATED;
                            }
                       }
                        else {
                            errorMsg = "Error: conflict, volume in dict: " + dictName + " lang: "+lang+" already exists!";
                            status = 409;
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Conflict</h1><p>" + errorMsg + "</p></html>");
                        }
                    }
                    else {
                        String login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():"";
                        errorMsg = "Error: user: " + login +" not authorized to post volume!";
                        status = HttpPresentationResponse.SC_UNAUTHORIZED;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                    }
                }
                else {
                    errorMsg = "Error: dictionary " + dictName + " for volume " + lang + " does not exist!";
                    status = HttpPresentationResponse.SC_NOT_FOUND;
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                }
            }
            else {
                errorMsg = "Error: volume metadata: <![CDATA["+ volumeXml +"]]> XML is malformed!";
                status = HttpPresentationResponse.SC_BAD_REQUEST;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            errorMsg = "Error: only XML content type allowed!";
            status = 415;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unsupported Media Type</h1><p>" + errorMsg + "</p></html>");
        }
        
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    

    
    public static java.util.Vector putVolume(String dictName, String lang, String volumeXml, String contentType, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        org.w3c.dom.Document volDom = null;
        if (contentType.equals(XML_CONTENTTYPE)) {
            try {
                volDom = XMLServices.buildDOMTree(volumeXml);
            }
            catch (Exception e) {
                volumeXml = "";
            }
            if (volumeXml != null && !volumeXml.equals("")) {
                Dictionary theDict = DictionariesFactory.getDictionaryByName(dictName);
                if (theDict !=null && !theDict.isEmpty()) {
                Volume theVolume = VolumesFactory.parseVolumeMetadata(theDict, volDom, null, false, false);
                if (theVolume !=null && !theVolume.isEmpty()) {
                    if (userCanHandleMetadata(theUser, theDict)) {
                        java.util.Collection volumesCollection = VolumesFactory.getVolumesArray(dictName,lang,null);
                        if (volumesCollection !=null || volumesCollection.size()>0) {
                            Volume existVolume = (Volume) volumesCollection.iterator().next();
                            existVolume.delete();
                            theVolume.save();
                            fr.imag.clips.papillon.business.edition.UITemplates.resetCache();
                            VolumeEntriesFactory.resetCountCache(theVolume.getName());
                            theVolume.getCount();
                            Papillon.initializeAllCaches();
                            String userMessage = "modifying " + theVolume.getName() + " volume: " + theVolume.getDictname() + " // "  + theVolume.getDbname() + " // " + theVolume.getSourceLanguage() + " // " + theVolume.getTargetLanguages() + " // " + theVolume.getVolumeRef();
                            PapillonLogger.writeDebugMsg(userMessage);

                            content = XMLServices.buildDOMTree(theVolume.getXmlCode());
                            status = HttpPresentationResponse.SC_NO_CONTENT;
                        }
                        else {
                            errorMsg = "Error: volume: " + dictName + " " + lang + " does not exist!";
                            status = HttpPresentationResponse.SC_NOT_FOUND;
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                        }
                    }
                    else {
                        String login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():"";
                        errorMsg = "Error: user: " + login +" not authorized to post volume!";
                        status = HttpPresentationResponse.SC_UNAUTHORIZED;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                    }
                }
                else {
                    errorMsg = "Error: volume metadata for volume: " + dictName + " lang: " + lang + " is not semantically correct!";
                    status = 422;
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                }
                }
                else {
                    errorMsg = "Error: dictionary " + dictName + " for volume " + lang + " does not exist!";
                    status = HttpPresentationResponse.SC_NOT_FOUND;
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                }
            }
            else {
                errorMsg = "Error: volume metadata: <![CDATA["+ volumeXml +"]]> XML is malformed!";
                status = HttpPresentationResponse.SC_BAD_REQUEST;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            errorMsg = "Error: only XML content type allowed!";
            status = 415;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unsupported Media Type</h1><p>" + errorMsg + "</p></html>");
        }
        
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    
    public static java.util.Vector deleteVolume(String dictName, String lang, User theUser)
    throws PapillonBusinessException {
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_NO_CONTENT;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        
        java.util.Collection volumesCollection = VolumesFactory.getVolumesArray(dictName,lang,null);
        if (volumesCollection !=null && volumesCollection.size()>0) {
            Volume theVolume = (Volume) volumesCollection.iterator().next();
            Dictionary theDict = DictionariesFactory.getDictionaryByName(dictName);
            if (userCanAccessMetadata(theUser,theDict)) {
                content = XMLServices.buildDOMTree(theVolume.getXmlCode());
                theVolume.deleteAll();
                Papillon.initializeAllCaches();
                if (content==null) {
                    errorMsg = "Unknown Error: volume metadata for dict: " + dictName + " not deleted!";
                    status = 422;
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                }
                else {
                    status = HttpPresentationResponse.SC_NO_CONTENT;
                }
            }
            else {
                String login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():"";
                errorMsg = "Error: user: " + login +" not authorized to delete dict!";
                status = HttpPresentationResponse.SC_UNAUTHORIZED;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            errorMsg = "Error: volume: " + dictName + " lang: " + lang + " does not exist!";
            status = HttpPresentationResponse.SC_NOT_FOUND;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }

    protected static boolean userCanAccessMetadata(User theUser, Dictionary theDict)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        boolean answer = false;
        if (theDict.getAccess() == null ||
            theDict.getAccess().equals("") ||
            theDict.getAccess().equals(Dictionary.PUBLIC_ACCESS) ||
            (theDict.getAccess().equals(Dictionary.RESTRICTED_ACCESS) && theUser != null) ||
            (theDict.getAccess().equals(Dictionary.PRIVATE_ACCESS) && theUser != null && (theUser.isAdmin() || theUser.isInGroup(Group.ADMIN_DICT_GROUP_PREFIX + theDict.getName()) || theUser.isInGroup(Group.VALIDATOR_DICT_GROUP_PREFIX + theDict.getName()) || theUser.isInGroup(Group.SPECIALIST_DICT_GROUP_PREFIX + theDict.getName()) || theUser.isInGroup(Group.READER_DICT_GROUP_PREFIX + theDict.getName())))
            ) {
            answer=true;
        }
        return answer;
    }
    
    public static boolean userCanHandleMetadata(User myUser, Dictionary theDict)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        boolean answer = false;
        PapillonLogger.writeDebugMsg("userCanHandleMetadata: user: " + myUser.getLogin() + " dict: " + theDict.getName());
        if (null != myUser && !myUser.isEmpty() && (myUser.isAdmin() || myUser.isInGroup(Group.ADMIN_DICT_GROUP_PREFIX + theDict.getName()))) {
            answer=true;
        }
        return answer;
    }


}

