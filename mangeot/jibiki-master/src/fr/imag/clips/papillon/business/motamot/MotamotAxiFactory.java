/*
 *  papillon
 *
 * (c) Francis Brunet-Manquat - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------
 * $Id: LexALPPostUpdateProcessor.java 712 2007-10-29 15:11:03Z serasset $
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

package fr.imag.clips.papillon.business.motamot;

import java.util.Collection;
import java.util.Iterator;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.xml.XMLServices;

import fr.imag.clips.papillon.business.transformation.ResultPostUpdateProcessor;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;

public class MotamotAxiFactory {
    
	protected static String AXI_LANG = "axi";
	protected static String ENTRY_TAG = "m:entry";
	protected static String ID_TAG = "id";
	protected static String IDMAX_TAG = "idmax";
	protected static String SENSE_TAG = "m:sense";
	protected static String SID_PREFIX = "s";
	protected static String TRANSLATIONS_TAG = "m:translations";
	protected static String TRANSLATION_TAG = "m:translation";
	protected static String REFLEXIES_TAG = "m:reflexies";
	protected static String REFLEXIE_TAG = "m:reflexie";
	protected static String REFAXIE_TAG = "m:refaxie";
	protected static String IDREFENTRY_TAG = "idrefentry";
	protected static String IDREFLEXIE_TAG = "idreflexie";
	protected static String IDREFAXIE_TAG = "idrefaxie";
	protected static String LANG_TAG = "lang";
    
	protected static void updateOrCreateAxies(VolumeEntry theEntry, Node sense, String senseId, User user) 
	throws PapillonBusinessException {
		Volume axieVolume = theEntry.getAxieVolume();
		String existingRefaxie = null;
		NodeList senseChildsList = sense.getChildNodes();
		if ((null != senseChildsList) && (senseChildsList.getLength() > 0)) {
			for (int j=0; j < senseChildsList.getLength(); j++) {
				Node senseChild = senseChildsList.item(j);
				if (senseChild.getNodeType()==org.w3c.dom.Node.ELEMENT_NODE && senseChild.getNodeName().equals(REFAXIE_TAG) && existingRefaxie == null) {
					existingRefaxie = ((Element) senseChild).getAttribute(IDREFAXIE_TAG);
					if (existingRefaxie != null && existingRefaxie.equals("")) {
						existingRefaxie = null;
					}
				}
				else if (senseChild.getNodeType()==org.w3c.dom.Node.ELEMENT_NODE && senseChild.getNodeName().equals(TRANSLATIONS_TAG)) {
		NodeList translationList = senseChild.getChildNodes();
		if ((null != translationList) && (translationList.getLength() > 0)) {
			for (int i=0; i < translationList.getLength(); i++) {
				Node translation = translationList.item(i);
				if (translation.getNodeType()==org.w3c.dom.Node.ELEMENT_NODE && translation.getNodeName().equals(TRANSLATION_TAG)) {
					Node idrefentryAttribute = translation.getAttributes().getNamedItem(IDREFENTRY_TAG);
					String idrefentry = null;
					if (idrefentryAttribute != null) {
						idrefentry =  idrefentryAttribute.getNodeValue();
						if (idrefentry!=null && !idrefentry.equals("")) {
							Node langAttribute = translation.getAttributes().getNamedItem(LANG_TAG);
							String targetLang = null;
							if (langAttribute != null) {
								targetLang = langAttribute.getNodeValue();
								if (targetLang!=null && !targetLang.equals("")) {
					Node idreflexieAttribute =  ((Element)translation).getAttributes().getNamedItem(IDREFLEXIE_TAG);
				String idreflexie = null;
				if (idreflexieAttribute != null) {
					idreflexie = idreflexieAttribute.getNodeValue();
				}
				// IDREFLEXIE NOT NULL => linked to target lexie, creating an axie
				if (idreflexie!=null && !idreflexie.equals("")) {
									String idrefaxie = null;
									Node idrefaxieAttribute = translation.getAttributes().getNamedItem(IDREFAXIE_TAG);
									if (idrefaxieAttribute != null) {
										idrefaxie = idrefaxieAttribute.getNodeValue();
									}
					updateSourceAxieTarget(axieVolume, theEntry, targetLang, sense, senseId, idrefaxie, existingRefaxie, idrefentry, idreflexie, user);
					((Element)translation).setAttribute(IDREFLEXIE_TAG,"");
					((Element)translation).setAttribute(IDREFAXIE_TAG,"");
					((Element)translation).setAttribute(IDREFENTRY_TAG,"");
					((Element)translation).setAttribute(LANG_TAG,"");
									
									//senseChild.removeChild(translation);
								}
			// IDREFLEXIE NULL => linked to target entry directly, without axie

							}
						}
					}
				}
			}
		}
	}
}
			}
		}
	}

	
	protected static void updateSourceAxieTarget(Volume axieVolume, VolumeEntry theEntry, String targetLang, Node sense, String senseId, String idrefaxie, String existingRefaxie, String idrefentry, String idreflexie, User user)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
		// Target not linked to an existing axie
		if (idrefaxie==null || idrefaxie.equals("")) {
			// but source linked to an existing axie
			String refaxie = "";
			if (existingRefaxie != null && !existingRefaxie.equals("")) {
				refaxie = updateAxie(axieVolume, existingRefaxie, targetLang, idrefentry, idreflexie, user);
			}
			// and source not linked to an existing axie
			else {
				refaxie = createAxie (axieVolume, theEntry.getSourceLanguage(), theEntry.getEntryId(), senseId, targetLang, idrefentry, idreflexie, user);
				addRefaxieInSense(sense,refaxie);
			}
			addRefaxieInEntry(theEntry.getDictionary().getName(), targetLang, idrefentry, idreflexie, refaxie, user);
		}
		// Target linked to an existing axie
		else {
			updateAxie(axieVolume, idrefaxie,theEntry.getSourceLanguage(), idrefentry, idreflexie, user);
			// and source linked to another existing axie
			if (existingRefaxie != null && !existingRefaxie.equals("") && !existingRefaxie.equals(idrefaxie)) {
				mergeAxies(axieVolume, existingRefaxie, idrefaxie, user);
			}
			// and source not linked to an existing axie
			else {
				addRefaxieInSense(sense,idrefaxie);
			}
		}
	}
	
	protected static String createAxie (Volume axieVolume, String lang1, String refEntry1, String refLexie1, String lang2, String refEntry2, String refLexie2, User user)
		throws PapillonBusinessException {

			PapillonLogger.writeDebugMsg("createAxie: lang1: " + lang1 + " e1: " + refEntry1 + "." + refLexie1 + " ; lang2: " + lang2 + " e2: " + refEntry2 + "." + refLexie2);

			VolumeEntry myAxie = VolumeEntriesFactory.createEmptyEntry(axieVolume.getName());
		myAxie.setCreationDate();
		myAxie.setAuthor(user.getLogin());
		myAxie.setGroups(user.getGroupsArray());
		myAxie.setStatus(VolumeEntry.FINISHED_STATUS);
		myAxie.setHeadword(createAxieHeadword(lang1,refEntry1,lang2,refEntry2));
		myAxie.setEntryId();
		myAxie.setContributionId();
        myAxie.setOriginalContributionId(myAxie.getContributionId());
		myAxie.setStatus(VolumeEntry.FINISHED_STATUS);
		myAxie.setModification(user.getLogin(), "creation");
		myAxie.initClassifiedFinishedContribution();
		myAxie.initClassifiedNotFinishedContribution();
		myAxie.setNextContributionAuthor("");			
			
		org.w3c.dom.Document theDom = myAxie.getDom();
		Element reflexies = (Element)theDom.getElementsByTagName(REFLEXIES_TAG).item(0);
		Element reflexie = theDom.createElement(REFLEXIE_TAG);
		reflexie.setAttribute(IDREFENTRY_TAG, refEntry1);
		reflexie.setAttribute(IDREFLEXIE_TAG, refLexie1);
		reflexie.setAttribute(LANG_TAG, lang1);
		reflexies.appendChild(reflexie.cloneNode(true));
		reflexie.setAttribute(IDREFENTRY_TAG, refEntry2);
		reflexie.setAttribute(IDREFLEXIE_TAG, refLexie2);
		reflexie.setAttribute(LANG_TAG, lang2);
		reflexies.appendChild(reflexie);
			
			// FIXME: if I don't recompute the DOM, it does not take into account some parts like the reflexies
		myAxie.setDom(XMLServices.buildDOMTree(XMLServices.xmlCode(theDom)));

		myAxie.save();
		return myAxie.getEntryId();
	}

	protected static String createAxieHeadword(String lang1,String refEntry1,String lang2,String refEntry2) {
		if (lang1.compareTo(lang2)>0) {
			String tmp = refEntry1;
			refEntry1 = refEntry2;
			refEntry2 = tmp;
		}
		return "["+transformEntryId(refEntry1) + "," + transformEntryId(refEntry2) +"]";
	}
    
	protected static String transformEntryId(String id) {
		int first = id.indexOf('.');
		int next =id.indexOf('.',first+1);
		if (first >=0 && next > first) {
			id = id.substring(0,next);
			id = id.replace('.',':');
		}
		return id;
	}
	
	protected static void addRefaxieInEntry(String dictname, String targetLanguage, String entryId, String reflexie, String refaxie, User user) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		PapillonLogger.writeDebugMsg("addRefaxieInEntry：" + entryId + " trg: " + targetLanguage + " sid:" + reflexie + " axi: " + refaxie);
		Collection targetVolumes = VolumesFactory.getVolumesArray(dictname, targetLanguage, null);
		
		if (targetVolumes != null && targetVolumes.size()==1) {
			Volume targetVolume = (Volume) targetVolumes.iterator().next();
			VolumeEntry myEntry = VolumeEntriesFactory.findEntryByEntryId(user, targetVolume, entryId);
			Document theDom = myEntry.getDom();
			// getElementById does not work unless validating is on
			//Node theSense = theDom.getElementById(reflexie);
			
			NodeList senses = theDom.getElementsByTagName(SENSE_TAG);
			Node theSense = null;
			if (senses != null) {
				int i=0;
				while (theSense ==null && i<senses.getLength()) {
					Element senseE = (Element) senses.item(i);
					String senseId = senseE.getAttribute(ID_TAG);
					if (senseId != null && senseId.equals(reflexie)) {
						theSense = senseE;
					}
					i++;
				}
			}
			
			if (theSense !=null) {
				addRefaxieInSense(theSense, refaxie);
			}
			
			myEntry.setModification(user.getLogin()," axie link added: " + refaxie);
			
			// FIXME: if I don't recompute the DOM, it does not take into account some parts like the reflexies
			myEntry.setDom(XMLServices.buildDOMTree(XMLServices.xmlCode(theDom)));
			myEntry.save();
		}
	}
	
	
	protected static void addRefaxieInSense(Node sense, String idrefaxie) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		Element refaxie = sense.getOwnerDocument().createElement(REFAXIE_TAG);
		refaxie.setAttribute(IDREFAXIE_TAG, idrefaxie);

		NodeList childsList = sense.getChildNodes();
		if ((null != childsList) && (childsList.getLength() > 0)) {
			int i=0;
			while (refaxie !=null && i< childsList.getLength()) {
				Node child = childsList.item(i);
				if (child.getNodeType()==org.w3c.dom.Node.ELEMENT_NODE && child.getNodeName().equals(TRANSLATIONS_TAG)) {
					sense.insertBefore(refaxie,child);
					refaxie = null;
				}
				i++;
			}
		}
		if (refaxie != null) {
			sense.appendChild(refaxie);
		}
	}
	
	
	protected static String updateAxie(Volume axieVolume, String refaxie, String lang, String refEntry, String refLexie, User user) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		PapillonLogger.writeDebugMsg("updateAxie: axi: " + refaxie + " lang: " + lang + " e1: " + refEntry + "." + refLexie);
		VolumeEntry myAxie = VolumeEntriesFactory.findEntryByEntryId(user, axieVolume,refaxie);
		org.w3c.dom.Document theDom = myAxie.getDom();
		Element reflexies = (Element)theDom.getElementsByTagName(REFLEXIES_TAG).item(0);
		Element reflexie = theDom.createElement(REFLEXIE_TAG);
		reflexie.setAttribute(IDREFENTRY_TAG, refEntry);
		reflexie.setAttribute(IDREFLEXIE_TAG, refLexie);
		reflexie.setAttribute(LANG_TAG, lang);
		reflexies.appendChild(reflexie);
		
		myAxie.setModification(user.getLogin(),lang + " language link added");

		// FIXME: if I don't recompute the DOM, it does not take into account some parts like the reflexies
		myAxie.setDom(XMLServices.buildDOMTree(XMLServices.xmlCode(theDom)));
		myAxie.save();
		return myAxie.getEntryId();
	}
	
	protected static void mergeAxies(Volume axieVolume, String refaxie1, String refaxie2, User user) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		VolumeEntry myAxie1 = VolumeEntriesFactory.findEntryByEntryId(user, axieVolume,refaxie1);
		VolumeEntry myAxie2 = VolumeEntriesFactory.findEntryByEntryId(user, axieVolume,refaxie2);
		if (myAxie1 != null && myAxie2 != null) {
			org.w3c.dom.Document theDom1 = myAxie1.getDom();
			Element reflexies1 = (Element)theDom1.getElementsByTagName(REFLEXIES_TAG).item(0);
			org.w3c.dom.Document theDom2 = myAxie1.getDom();
			Element reflexies2 = (Element)theDom2.getElementsByTagName(REFLEXIES_TAG).item(0);
			NodeList reflexies2List = reflexies2.getChildNodes();
			for (int i=0; i< reflexies2List.getLength(); i++) {
				Node child = reflexies2List.item(i);
				Node importedChild = theDom1.importNode(child, true);
				reflexies1.appendChild(importedChild);
			}
		
			// updating linked lexies
			Dictionary theDict =  myAxie2.getDictionary();
			Collection srcLanguages = theDict.getSourceLanguagesArray();
			Iterator langIter = srcLanguages.iterator();
			while (langIter.hasNext()) {
				String lang = (String) langIter.next();
				Collection theVolumeCol = VolumesFactory.getVolumesArray(theDict.getName(),lang, null);
				Volume theVolume = (Volume) theVolumeCol.iterator().next();
				String[] lexies = myAxie2.getTranslationsLexieIds(lang);
				if (lexies != null && lexies.length >0) {
					for (int i=0;i< lexies.length;i++) {
						VolumeEntry myEntry = VolumeEntriesFactory.findEntryByEntryId(user, theVolume, lexies[i]);
						updateTranslationLinks(myEntry, "axi", refaxie2, refaxie1, user);
					}
				}
			}
			
			myAxie1.setModification(user.getLogin()," merged with axie " + refaxie2);
			
			// FIXME: if I don't recompute the DOM, it does not take into account some parts like the reflexies
			myAxie1.setDom(XMLServices.buildDOMTree(XMLServices.xmlCode(theDom1)));
			myAxie1.save();
			
			myAxie2.setModification(user.getLogin()," merged with axie " + refaxie1 + " and deleted");
			myAxie2.setStatus(VolumeEntry.DELETED_STATUS);
			myAxie2.save();			
		
		}
	}
	
	protected static void updateTranslationLinks(VolumeEntry myEntry, String lang, String oldRefEntry, String newRefEntry, User user) 
	 throws fr.imag.clips.papillon.business.PapillonBusinessException {
		Document theDom = myEntry.getDom();
		if (lang.equals("axi")) {
			NodeList refaxies = theDom.getElementsByTagName(REFAXIE_TAG);
			if (refaxies != null && refaxies.getLength() >0) {
				for (int i=0;i< refaxies.getLength();i++) {
					Element refAxie = (Element) refaxies.item(i);
					String idrefaxie = refAxie.getAttribute(IDREFAXIE_TAG);
					if (idrefaxie != null && idrefaxie.equals(oldRefEntry)) {
						refAxie.setAttribute(IDREFAXIE_TAG, newRefEntry);
					}
				}
			}
		}
		myEntry.setModification(user.getLogin()," merged target entry " + oldRefEntry + " into " + newRefEntry);
		myEntry.setDom(XMLServices.buildDOMTree(XMLServices.xmlCode(theDom)));
		myEntry.save();
	}

}













