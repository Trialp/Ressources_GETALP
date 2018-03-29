/*
 *  papillon
 *
 * (c) Francis Brunet-Manquat - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------
 * $Id$
 *------------------------
 * $Log$
 * Revision 1.2.4.1  2007/10/29 15:11:03  serasset
 * NEW: lexalp css now defines different forms for HARMONISED/REJECTED entries
 * NEW: added new db url/user/password configuration keys in papillon.properties file
 * BUG158: headwords are now harmonised at edition and search time, added a "normalise headword" admin action
 *
 * Revision 1.2  2006/06/01 22:05:05  fbrunet
 * New interface, quick search, new contribution management (the first edition not create new contribution. New contribution is created after add, remove element, update, save, etc. in the interface window)
 *
 * Revision 1.1  2006/05/05 02:08:23  fbrunet
 * bug correction : url utf8 transfert (in createEntryInit)
 *
 *
 *
 *------------------------
 */

package fr.imag.clips.papillon.business.lexalp;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.user.User;

import fr.imag.clips.papillon.business.transformation.ResultPostUpdateProcessor;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.utility.StringNormalizer;


public class LexALPPostUpdateProcessor implements ResultPostUpdateProcessor {
    
    
    public void transformation(VolumeEntry volumeEntry, User user) throws PapillonBusinessException {
        try {
            
            //
            if (volumeEntry != null) {
                
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
            }    
            
        } catch(Exception ex) {
            
            //
            throw new PapillonBusinessException("Exception in LexALPPostProcessor.transformation()", ex);
        }	
        
        
    }


    
}


















