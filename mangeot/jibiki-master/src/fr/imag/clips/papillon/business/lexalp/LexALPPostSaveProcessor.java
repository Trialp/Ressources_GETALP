/*
 *  papillon
 *
 * (c) Francis Brunet-Manquat - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------
 * $Id$
 *------------------------
 * $Log$
 * Revision 1.7.2.1  2007/10/29 15:11:03  serasset
 * NEW: lexalp css now defines different forms for HARMONISED/REJECTED entries
 * NEW: added new db url/user/password configuration keys in papillon.properties file
 * BUG158: headwords are now harmonised at edition and search time, added a "normalise headword" admin action
 *
 * Revision 1.7  2007/02/08 15:52:47  fbrunet
 * *** empty log message ***
 *
 * Revision 1.6  2007/02/08 15:24:07  fbrunet
 * *** empty log message ***
 *
 * Revision 1.5  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.4  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.3  2006/06/01 22:05:05  fbrunet
 * New interface, quick search, new contribution management (the first edition not create new contribution. New contribution is created after add, remove element, update, save, etc. in the interface window)
 *
 * Revision 1.2  2006/05/22 22:45:54  fbrunet
 * LexALP: add merge method in post-save processing (merge axies with same referenced lexies)
 *
 * Revision 1.1  2006/05/05 02:08:23  fbrunet
 * bug correction : url utf8 transfert (in createEntryInit)
 *
 *
 *
 *------------------------
 */

package fr.imag.clips.papillon.business.lexalp;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

import java.util.ArrayList;
import java.util.Enumeration;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.QueryRequest;
import fr.imag.clips.papillon.business.dictionary.QueryCriteria;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.presentation.MergeEntries;

import fr.imag.clips.papillon.business.transformation.ResultPostSaveProcessor;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.utility.StringNormalizer;


public class LexALPPostSaveProcessor implements ResultPostSaveProcessor {
    
    //
    protected final static String ConfirmEntryURL = "ConfirmEntry.po";
    
    // FIXME: All this code is duplicate code... Can we call the post update processing before doing a post save ?
    public void transformation(VolumeEntry volumeEntry, User user) throws PapillonBusinessException {
        try {
            
            //
            if (volumeEntry != null) {
                                              
                ///////////////////////////////////////////////////
                // ADD JIBIKI DATA TO ENTRY DATA 

                //
                Document dom = volumeEntry.getDom();

                // FIXME: use another method !!!!!!!!!!!!!
                // d:metadata and admin
                NodeList dMetadataList = dom.getElementsByTagName("d:metadata");
                NodeList adminList = dom.getElementsByTagName("admin");
                if ((null != dMetadataList) && (dMetadataList.getLength() > 0) 
                    && (null != adminList) && (adminList.getLength() > 0)) {
                    Element dMetadata = (Element) dMetadataList.item(0);
                    Element admin = (Element) adminList.item(0);
                    
                    // created-by = d:author
                    NodeList dAuthorList = dMetadata.getElementsByTagName("d:author");
                    NodeList createdByList = admin.getElementsByTagName("created-by");
                    if ((null != dAuthorList) && (dAuthorList.getLength() > 0)
                        && (null != createdByList) && (createdByList.getLength() > 0)) {
                        
                        Node dAuthorNode = dAuthorList.item(0).getFirstChild();
                        
                        if (dAuthorNode != null) {
                            Element createdBy = (Element) createdByList.item(0);
                            if (createdBy.hasChildNodes()) {
                                createdBy.getFirstChild().setNodeValue(dAuthorNode.getNodeValue());
                            } else {   
                                Node createdByNode = dom.importNode(dAuthorNode, false);
                                createdBy.appendChild(createdByNode);
                            }
                        }
                    }
                    
                    // creation-date = d:creation-date
                    NodeList dCreationDateList = dMetadata.getElementsByTagName("d:creation-date");
                    NodeList creationDateList = admin.getElementsByTagName("creation-date");
                    if ((null != dCreationDateList) && (dCreationDateList.getLength() > 0)
                        && (null != creationDateList) && (creationDateList.getLength() > 0)) {
                        
                        Node dCreationDateNode = dCreationDateList.item(0).getFirstChild();
                        
                        if (dCreationDateNode != null) {
                            Element creationDate = (Element) creationDateList.item(0);
                            if (creationDate.hasChildNodes()) {
                                creationDate.getFirstChild().setNodeValue(dCreationDateNode.getNodeValue());
                            } else {   
                                Node creationDateNode = dom.importNode(dCreationDateNode, false);
                                creationDate.appendChild(creationDateNode);
                            }
                        }
                    }
                    
                    // review-by = d:reviewer
                    NodeList dReviewerList = dMetadata.getElementsByTagName("d:reviewer");
                    NodeList reviewByList = admin.getElementsByTagName("review-by");
                    if ((null != dReviewerList) && (dReviewerList.getLength() > 0)
                        && (null != reviewByList) && (reviewByList.getLength() > 0)) {
                        
                        Node dReviewerNode = dReviewerList.item(0).getFirstChild();
                        
                        if (dReviewerNode != null) {
                            Element reviewBy = (Element) reviewByList.item(0);
                            if (reviewBy.hasChildNodes()) {
                                reviewBy.getFirstChild().setNodeValue(dReviewerNode.getNodeValue());
                            } else {   
                                Node reviewByNode = dom.importNode(dReviewerNode, false);
                                reviewBy.appendChild(reviewByNode);
                            }
                        }
                    }
                    
                    // review-date = d:review-date
                    NodeList dReviewDateList = dMetadata.getElementsByTagName("d:review-date");
                    NodeList reviewDateList = admin.getElementsByTagName("review-date");
                    if ((null != dReviewDateList) && (dReviewDateList.getLength() > 0)
                        && (null != reviewDateList) && (reviewDateList.getLength() > 0)) {
                        
                        Node dReviewDateNode = dReviewDateList.item(0).getFirstChild();
                        
                        if (dReviewDateNode != null) {
                            Element reviewDate = (Element) reviewDateList.item(0);
                            if (reviewDate.hasChildNodes()) {
                                reviewDate.getFirstChild().setNodeValue(dReviewDateNode.getNodeValue());
                            } else {   
                                Node reviewDateNode = dom.importNode(dReviewDateNode, false);
                                reviewDate.appendChild(reviewDateNode);
                            }
                        }
                    }
                    
                    // last-modification-by = d:history.d:modification.last-modification-by
                    NodeList dLastModifByList = dMetadata.getElementsByTagName("last-modification-by");
                    NodeList lastModifByList = admin.getElementsByTagName("last-modification-by");
                    if ((null != dLastModifByList) && (dLastModifByList.getLength() > 0)
                        && (null != lastModifByList) && (lastModifByList.getLength() > 0)) {
                        
                        Node dLastModifByNode = dLastModifByList.item(dLastModifByList.getLength()-1).getFirstChild();
                        
                        if (dLastModifByNode != null) {
                            Element lastModifBy = (Element) lastModifByList.item(0);
                            if (lastModifBy.hasChildNodes()) {
                                lastModifBy.getFirstChild().setNodeValue(dLastModifByNode.getNodeValue());
                            } else {   
                                Node lastModifByNode = dom.importNode(dLastModifByNode, false);
                                lastModifBy.appendChild(lastModifByNode);
                            }
                        }
                    }
                    
                    // last-modification-by = d:history.d:modification.last-modification-by
                    NodeList dLastModifDateList = dMetadata.getElementsByTagName("last-modification-date");
                    NodeList lastModifDateList = admin.getElementsByTagName("last-modification-date");
                    if ((null != dLastModifDateList) && (dLastModifDateList.getLength() > 0)
                        && (null != lastModifDateList) && (lastModifDateList.getLength() > 0)) {
                        
                        Node dLastModifDateNode = dLastModifDateList.item(dLastModifDateList.getLength()-1).getFirstChild();
                        
                        if (dLastModifDateNode != null) {
                            Element lastModifDate = (Element) lastModifDateList.item(0);
                            if (lastModifDate.hasChildNodes()) {
                                lastModifDate.getFirstChild().setNodeValue(dLastModifDateNode.getNodeValue());
                            } else {   
                                Node lastModifDateNode = dom.importNode(dLastModifDateNode, false);
                                lastModifDate.appendChild(lastModifDateNode);
                            }
                        }
                    }
                }

                // Normalise headword
                NodeList hwList = dom.getElementsByTagName("term");
                if ((null != hwList) && (hwList.getLength() > 0)) {
                    for (int i=0; i < hwList.getLength(); i++) {

                        //
                        Node termNode = (Node)hwList.item(i);
                        Node termValue = termNode.getFirstChild(); // term node contains text node
                        String term = termValue.getNodeValue();
                        termValue.setNodeValue(StringNormalizer.normalize(term));

                    }
                }

                // Source:  add http:// if no http:// or ftp:// in url source
                NodeList sourceList = dom.getElementsByTagName("source");
                if ((null != sourceList) && (sourceList.getLength() > 0)) {
                    for (int i=0; i < sourceList.getLength(); i++) {
                        
                        //
                        Node source = (Node)sourceList.item(i);
                        Node sourceTypeAttribut = source.getAttributes().getNamedItem("sourceType");
                        String type = sourceTypeAttribut.getNodeValue();
                        
                        //
                        if ( type.equals("Url") ) {
                            String text = source.getFirstChild().getNodeValue();
                            if (!text.startsWith("http://") && !text.startsWith("ftp://")) {
                                source.getFirstChild().setNodeValue("http://" + text);
                            }
                        }
                    }
                }
                
                
                // Legal system post-process
                if (!volumeEntry.getVolumeName().equals("LexALP_axi")) {
                    NodeList entryList = dom.getElementsByTagName("entry");
                    NodeList usageList = dom.getElementsByTagName("usage");
                    if ((null != entryList) && (entryList.getLength() > 0)
                        && (null != usageList) && (usageList.getLength() > 0)) {
                        
                        Element usage = (Element) usageList.item(0);
                        Node codeAttribut = usage.getAttributes().getNamedItem("geographical-code");
                        String code = codeAttribut.getNodeValue();
                        Element entry = (Element) entryList.item(0);
                        Node legalAttribut = entry.getAttributes().getNamedItem("legalSystem");
                        String legalSystem = "UNKNOWN";
                        
                        if ( code.equals("AC")) {               // Alpine Convention
                            legalSystem = "AC";
                            
                        } else if ( code.equals("AT")) {               // AUSTRIA
                            legalSystem = "AT";
                            
                        } else if ( code.equals("DE")) {        // GERMANY
                            legalSystem = "DE";
                            
                        } else if ( code.equals("IT")) {        // ITALY
                            legalSystem = "IT";
                            
                        } else if ( code.equals("CH")) {        // SWITZERLAND
                            legalSystem = "CH";
                            
                        } else if ( code.equals("FR")) {        // FRANCE
                            legalSystem = "FR";
                            
                        } else if ( code.equals("SL")) {        // SLOVENIA
                            legalSystem = "SL";
                            
                        } else if ( code.equals("EU")) {         // EUROPE
                            legalSystem = "EU";
                            
                        } else if ( code.equals("INT")) {       // INTERNATIONAL
                            legalSystem = "INT";
                            
                        } else if ( code.equals("FVG")) {       // Regione Friuli Venezia Giulia -> IT
                            legalSystem = "IT";
                         
                        } else if ( code.equals("BAY")) {         // Bayern -> DE
                            legalSystem = "DE";
                            
                        } else if ( code.equals("N")) {         // Niederösterreich -> AT
                            legalSystem = "AT";
                            
                        } else if ( code.equals("O")) {         // Oberösterreich -> AT
                            legalSystem = "AT";
                            
                        } else if ( code.equals("T")) {         // TIROL -> AT
                            legalSystem = "AT";
                            
                        } else if ( code.equals("St")) {        // Steiermark -> AT
                            legalSystem = "AT";
                            
                        } else if ( code.equals("S")) {         // Salzburg -> AT
                            legalSystem = "AT";
                            
                        } else if ( code.equals("V")) {         // Vorarlberg -> AT
                            legalSystem = "AT";
                            
                        } else if ( code.equals("K")) {         // Kärnten -> AT
                            legalSystem = "AT";
                            
                        } else if ( code.equals("UNKNOWN")) {
                            legalSystem = "UNKNOWN";
                        }
                        
                        // Set legal system
                        if (legalAttribut == null) {
                            entry.setAttribute("legalSystem", legalSystem);
                        } else {
                            legalAttribut.setNodeValue(legalSystem);
                        }
                    }
                }
                
                //
                volumeEntry.save();
                
                
                
                ///////////////////////////////////////////////////
                // FUSION AXI
                // only axi with referenced lexies !
                if (volumeEntry.getVolumeName().equals("LexALP_axi") && getReferencedLexieIds(volumeEntry).size()!=0) {
                    
                    // - Search referenced lexies link with axi
                    // - Search referenced axies link with axi
                    // - Search axies to merge
                    ArrayList referencedLexieList = new ArrayList();
                    ArrayList referencedAxieList = new ArrayList();
                    ArrayList AxieList = new ArrayList();
                    mergeReferencedLexie(volumeEntry, referencedLexieList, referencedAxieList, AxieList, user);
                    
                    //
                    if (AxieList.size() != 0) {
                        
                        // Create new contribution with NOT_FINISHED_STATUS
                        // FIXME: create methods in VolumeEntriesFactory class to manage contributions !
                        VolumeEntry newAxi = VolumeEntriesFactory.newEntryFromExisting(volumeEntry);
                        newAxi.initClassifiedNotFinishedContribution();
                        newAxi.initClassifiedFinishedContribution();
                        newAxi.addClassifiedFinishedContribution(volumeEntry);
                        for ( int i=0; i < AxieList.size(); i++) {
                            newAxi.addClassifiedFinishedContribution((String)AxieList.get(i));
                        }
                        newAxi.setModification(user.getLogin(), "Merge");
                        newAxi.setStatus(VolumeEntry.NOT_FINISHED_STATUS);
                        newAxi.save();
                        
                        // Old entry modification
                        volumeEntry.setStatus(VolumeEntry.MODIFIED_STATUS);
                        volumeEntry.setNextContributionAuthor(user.getLogin());
                        volumeEntry.save();
                        
                        
                        //// Add new referenced lexies
                        Document domNewAxi = newAxi.getDom();
                        NodeList termRefNodeList = domNewAxi.getElementsByTagName("termref");
                        
                        //
                        if ( (null != termRefNodeList) && (termRefNodeList.getLength() > 0) ) {
                            
                            // first term reference
                            Node termRefNode = (Node)termRefNodeList.item(0);
                            Node ParentNode = termRefNode.getParentNode();
                            
                            // create template
                            Node templateTermRefNode = (Node)termRefNode.cloneNode(true);
                            
                            // Remove 
                            ArrayList newReferencedLexieList = notContainLexies(getReferencedLexieIds(newAxi), referencedLexieList);
                                                                                   
                            //
                            for ( int i=0; i < newReferencedLexieList.size(); i++) {
                                
                                // Add to template
                                ArrayList couple = (ArrayList)newReferencedLexieList.get(i);
                                String refLang = (String)couple.get(0);
                                String refLexie = (String)couple.get(1);
                                Node idrefAttribut = templateTermRefNode.getAttributes().getNamedItem("idref");
                                idrefAttribut.setNodeValue(refLexie);
                                Node idlangAttribut = templateTermRefNode.getAttributes().getNamedItem("lang");
                                idlangAttribut.setNodeValue(refLang);
                                
                                // Add to dom
                                ParentNode.insertBefore(domNewAxi.importNode(templateTermRefNode, true), termRefNode);
                            }
                        }
                        
                        
                        //// Add new referenced axies
                        NodeList axieRefNodeList = domNewAxi.getElementsByTagName("axieref");
                        
                        //
                        if ( (null != axieRefNodeList) && (axieRefNodeList.getLength() > 0) ) {
                            
                            // first term reference
                            Node axieRefNode = (Node)axieRefNodeList.item(0);
                            Node ParentNode = axieRefNode.getParentNode();
                            
                            // create template
                            Node templateAxieRefNode = (Node)axieRefNode.cloneNode(true);
                            
                            // Remove 
                            ArrayList newReferencedAxieList = notContainAxies(getReferencedAxieIds(newAxi), referencedAxieList);
                            
                            //
                            for ( int i=0; i < newReferencedAxieList.size(); i++) {
                                
                                // Add to template
                                Node idrefAttribut = templateAxieRefNode.getAttributes().getNamedItem("idref");
                                idrefAttribut.setNodeValue((String)newReferencedAxieList.get(i));
                                
                                // Add to dom
                                ParentNode.insertBefore(domNewAxi.importNode(templateAxieRefNode, true), axieRefNode);
                            }
                        }
                        
                        // Save Axis
                        newAxi.setStatus(VolumeEntry.FINISHED_STATUS);
                        newAxi.save();
                        volumeEntry.setStatus(VolumeEntry.CLASSIFIED_FINISHED_STATUS);
                        volumeEntry.save();
                    
                        
                        // FIXME : action UNDO MERGE ! enlever Re-edit ...
                        throw new ClientPageRedirectException(
                                                              MergeEntries.MergeEntriesURL + "?" + 
                                                              MergeEntries.VolumeName_PARAMETER + "=" + newAxi.getVolumeName() + "&" + 
                                                              MergeEntries.EntryHandle_PARAMETER + "=" + newAxi.getHandle());
                        
                    }
                }
            }  
            
                
        } catch(Exception ex) {
            
            //
            throw new PapillonBusinessException("Exception in LexALPPostProcessor.transformation()", ex);
        }	
        
        
    }
    

    // get referenced lexies to an axi
    private ArrayList getReferencedLexieIds(VolumeEntry axi) throws PapillonBusinessException {
        
        // Search referenced lexies
        ArrayList referencedLexieList = new ArrayList(); 
        
        // Search by lang
        for (Enumeration iter = VolumesFactory.getSourceLanguages(); iter.hasMoreElements ();) {
            String lang = (String)iter.nextElement();
            
            // get referenced lexies
            String[] referencedLexieListTmp = axi.getReferencedLexieIds(lang);
            
            // add referenced lexies and their lang
            if ( referencedLexieListTmp != null ) {
                for (int j = 0; j < referencedLexieListTmp.length; j++) {
                    
                    ArrayList couple= new ArrayList(2);
                    couple.add(lang);
                    couple.add(referencedLexieListTmp[j]);
                    referencedLexieList.add(couple);
                    
                    //
                    //System.out.println("get referencedLexieList ... " + referencedLexieListTmp[j]);
                }
            }
        }
        
        //
        return referencedLexieList;
    }
    
    // get referenced axies to an axi
    private ArrayList getReferencedAxieIds(VolumeEntry axi) throws PapillonBusinessException {
        
        // Search referenced Axies
        ArrayList referencedAxieList = new ArrayList();
        
        // get referenced lexies
        String[] referencedAxieListTmp = axi.getReferencedAxieIds();
        
        // add referenced axies
        if ( referencedAxieListTmp != null ) {
            for (int j = 0; j < referencedAxieListTmp.length; j++) {
                referencedAxieList.add(referencedAxieListTmp[j]);
            }
        }
        
        //
        return referencedAxieList;
    }
    
    

    // find lexies in newList not contain in list
    private ArrayList notContainLexies(ArrayList list, ArrayList newList) {
        
        //
        ArrayList notContainList = new ArrayList();
        
        //
        for (int i=0; i < newList.size(); i++) {
            ArrayList couple = (ArrayList)newList.get(i);
            String refLang = (String)couple.get(0);
            String refLexie = (String)couple.get(1);
            
            //
            int j=0;
            while ( (j < list.size()) 
                    && !(   ((String)((ArrayList)list.get(j)).get(0)).equals(refLang)
                            && ((String)((ArrayList)list.get(j)).get(1)).equals(refLexie)) ) {
                j++;
            }
            
            //
            if ( j == list.size() ) {
                
                // add lexie
                //System.out.println("add lexie : " + refLexie + " " + refLang);
                notContainList.add(couple);      
            }
        }
        
        //
        return notContainList;
    }

    // find axies in newList not contain in list
    private ArrayList notContainAxies(ArrayList list, ArrayList newList) {
        
        //
        ArrayList notContainList = new ArrayList();
        
        //
        for (int i=0; i < newList.size(); i++) {
            String axieId = (String)newList.get(i);
            
            //
            if ( !isContained(list, axieId) ) {
                
                // add lexie
                //System.out.println("add axieId : " + axieId);
                notContainList.add(axieId);      
            }
        }
        
        //
        return notContainList;
    }
    
    
    // true if string id is in list
    private boolean isContained(ArrayList list, String id) {
        
        //
        int j=0;
        while ( (j < list.size()) 
                && !((String)list.get(j)).equals(id) ) {
            j++;
        }
        
        //
        return ( j != list.size() );
    }
    


    /**
     * Find axies Ai link with axi by their referenced lexies, deleted them and add the new referenced lexies to list referencedLexieList
     * 
     * @param axi                   
     * @param referencedLexieList   list of lexies contains in Ai
     * @param referencedAxieList    list of axies Ai
     * @param axieList              list of axies Ai
     * @param user
     *
     */
    private void mergeReferencedLexie(VolumeEntry axi, ArrayList referencedLexieList, ArrayList referencedAxieList, ArrayList axieList, User user) throws PapillonBusinessException {
            
        //
        //System.out.println("find in axi : " + axi.getId());
        
        //// Search new referenced axies
        ArrayList newReferencedAxieList = notContainAxies(referencedAxieList, getReferencedAxieIds(axi));
        referencedAxieList.addAll(newReferencedAxieList);
        
        //// Search new referenced lexies
        for (int i=0; i<referencedLexieList.size(); i++) {
            ArrayList couple = (ArrayList)referencedLexieList.get(i);
            //System.out.println("referencedLexieList ... " + (String)couple.get(1));
        }
        ArrayList newReferencedLexieList = notContainLexies(referencedLexieList, getReferencedLexieIds(axi));
        referencedLexieList.addAll(newReferencedLexieList);
        
        //// Find axies containing the new referenced lexies
        if ( newReferencedLexieList.size()!=0 ) {
                
            //
            QueryRequest qr = new QueryRequest(axi.getVolume());
            
            // ReferencedLexie
            ArrayList listLexies = new ArrayList();
            for (int i=0; i < newReferencedLexieList.size(); i++) {
                
                //
                QueryCriteria criteriaRefLexie = new QueryCriteria();
                criteriaRefLexie.add("key", QueryCriteria.EQUAL, Volume.CDM_axiReflexie);  
                ArrayList couple = (ArrayList)newReferencedLexieList.get(i);
                String refLang = (String)couple.get(0);
                String refLexie = (String)couple.get(1);
                criteriaRefLexie.add("value", QueryCriteria.EQUAL, refLexie);
                criteriaRefLexie.add("lang", QueryCriteria.EQUAL, refLang);
                listLexies.add(criteriaRefLexie);
            }
            qr.addOrCriteriaList(listLexies);
            
            // ID
            QueryCriteria criteriaId = new QueryCriteria();
            criteriaId.add("key", QueryCriteria.EQUAL, Volume.CDM_entryId);  
            criteriaId.add("value", QueryCriteria.NOT_EQUAL, axi.getId());
            qr.addCriteria(criteriaId);
            
            // Status
            QueryCriteria criteriaFinishedStatus = new QueryCriteria();
            criteriaFinishedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);  
            criteriaFinishedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.FINISHED_STATUS);
            qr.addCriteria(criteriaFinishedStatus);
            
            // Find axies
            ArrayList axieResult = qr.findLexie(user);

            
            //// 
            //System.out.println("result axies : " + Integer.toString(axieResult.size()));
            for (int i=0; i < axieResult.size(); i++) {
                QueryResult result = (QueryResult)axieResult.get(i);
                VolumeEntry axiTmp = result.getSourceEntry();
                
                //
                if ( !isContained(axieList, axiTmp.getId()) ) {
                    
                    // Delete axi
                    // FIXME: create methods in VolumeEntriesFactory class to manage contributions !
                    VolumeEntry newAxi = VolumeEntriesFactory.newEntryFromExisting(axiTmp);
                    newAxi.initClassifiedNotFinishedContribution();
                    newAxi.initClassifiedFinishedContribution();
                    newAxi.addClassifiedFinishedContribution(axiTmp);
                    newAxi.setModification(user.getLogin(), "deleted");
                    newAxi.setStatus(VolumeEntry.DELETED_STATUS);
                    newAxi.save();
                    
                    // Old entry modification
                    axiTmp.setStatus(VolumeEntry.CLASSIFIED_FINISHED_STATUS);
                    axiTmp.setNextContributionAuthor(user.getLogin());
                    axiTmp.save();
                    
                    // Add axi
                    axieList.add(newAxi.getContributionId());
                    
                    //
                    mergeReferencedLexie(newAxi, referencedLexieList, newReferencedAxieList, axieList, user);
                }
            }
        }
    }
    
}


















