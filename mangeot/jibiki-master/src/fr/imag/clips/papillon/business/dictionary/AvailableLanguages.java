/*
 *  LanguageListsFactory.java
 *  papillon
 *
 *  Created by Mathieu Mangeot on Sat Mar 09 2002.
 *  Copyright (c) 2002 GETA_CLIPS_IMAG. All rights reserved.
 *
 *---------------------------------------------------------
 * $Id$
 *---------------------------------------------------------
 * $Log$
 * Revision 1.6  2007/01/16 13:28:31  serasset
 * Added cache reinitialization when a metadata is modified.
 *
 * Revision 1.5  2006/08/10 22:17:12  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.4  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.3.10.1  2006/02/17 13:21:25  mangeot
 *
 * MM: modified AdvancedQueryForm. getAllTargetLanguages, getAllSourceLanguages and getCdmElementsWithDefaultLanguage are now static in AvailableLanguages.java in order to accelerate the execution.
 *
 * Revision 1.3  2005/01/23 17:21:16  mangeot
 * *** empty log message ***
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *---------------------------------------------------------
 */

package fr.imag.clips.papillon.business.dictionary;

// For the TreMap & TreeSet & Array
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

public class AvailableLanguages {

    protected static TreeMap MapLanguages = null;
    protected static Collection SourceLanguagesArray = null;
    protected static Collection TargetLanguagesArray = null;
    protected static Collection AllLanguagesArray = null;
	
	protected static Set CdmElementsWithDefaultLanguage = null;


	/**
        * return the list of source languages as known by the database
     */
    protected static TreeMap getAvailableLanguagesMap()
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
			if (MapLanguages==null) {
                Collection DictsArray = DictionariesFactory.getDictionariesArray();
                MapLanguages = new TreeMap();
                TreeSet targetSet;
                
                //
                for (Iterator iter = DictsArray.iterator(); iter.hasNext();) {
                    Dictionary dict = (Dictionary)iter.next();
                    
                    //
                    for (Iterator iter2 = dict.getSourceLanguagesArray().iterator(); iter2.hasNext();) {
                        String source = (String)iter2.next();
                        if (null != source && !source.equals("") && !source.equals("axi")) {
                            if (MapLanguages.containsKey(source)) {
                                targetSet=(TreeSet) MapLanguages.get(source);
                            } else {
                                targetSet = new TreeSet();
                            }
                            
                            //
                            for (Iterator iter3 = dict.getTargetLanguagesArray().iterator(); iter3.hasNext();) {
                                String target = (String)iter3.next();
                                
                                if (null != target && !target.equals("") && !target.equals("axi") && !target.equals(source)) {
                                    targetSet.add(target);
                                }
                            }
                            MapLanguages.put(source,targetSet);
                        }
                    }
                }
			}
            
            //
            return MapLanguages;
        }

    /**
        * return the list of source languages as known by the database
     */
    public static Collection getSourceLanguagesArray()
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            if (SourceLanguagesArray == null) {
				SourceLanguagesArray = getAvailableLanguagesMap().keySet();
			}
            return SourceLanguagesArray;
        }


    public static Collection getTargetLanguagesArray(String lang)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            Collection result = null;
            TreeSet mySet = null;
            if (getAvailableLanguagesMap().containsKey(lang)) {
                mySet = (TreeSet)getAvailableLanguagesMap().get(lang);
                result =  mySet;
            }
            return result;
        }


    public static Collection getAllLanguagesArray()
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
			if (AllLanguagesArray == null) {
				TreeSet mySet = new TreeSet();
				mySet.addAll(getAvailableLanguagesMap().keySet());
				for (Iterator iter = getSourceLanguagesArray().iterator(); iter.hasNext();) {
					mySet.addAll((TreeSet)getAvailableLanguagesMap().get((String)iter.next()));
				}
				AllLanguagesArray = mySet;
			}
			return AllLanguagesArray;
        }

    public static Collection getSourceLanguagesArrayForDict(String dict)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
            Dictionary myDict = DictionariesFactory.getDictionaryByName(dict);
            TreeSet mySet = new TreeSet();
            mySet.addAll(myDict.getSourceLanguagesArray());
			// FIXME: Do not remove "axi" any more for the Motamot project
           // mySet.remove("axi");
            mySet.remove("");
			return mySet;
        }
	
    public static Collection getTargetLanguagesArray()
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
			if (TargetLanguagesArray == null) {
				Collection DictsArray = DictionariesFactory.getDictionariesArray();
				TreeSet mySet = new TreeSet();
                for (Iterator iter = DictsArray.iterator(); iter.hasNext();) {
                    Dictionary dict = (Dictionary)iter.next();
					mySet.addAll(dict.getTargetLanguagesArray());
				}
				// FIXME: Do not remove "axi" any more for the Motamot project
			//	mySet.remove("axi");
				mySet.remove("");
				TargetLanguagesArray = mySet;
			}
			return TargetLanguagesArray;
        }
	
	public static Set getCdmElementsWithDefaultLanguage ()
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
		if (CdmElementsWithDefaultLanguage == null) {
			
            //
            CdmElementsWithDefaultLanguage = new java.util.TreeSet();
			
			//
			for (Iterator iter =  VolumesFactory.getVolumesArray().iterator(); iter.hasNext();) {
				// get all cdm-elements with Volume.DEFAULT_LANG language
				Volume volume = (Volume)iter.next();
				HashMap CdmElementsTable = volume.getCdmElements();
				HashMap CdmElementsTableDefaultLang = (HashMap) CdmElementsTable.get(Volume.DEFAULT_LANG);
				
				CdmElementsWithDefaultLanguage.addAll(CdmElementsTableDefaultLang.keySet());
			}
		}
		return CdmElementsWithDefaultLanguage;
	} 
	
	public static boolean resetCache () {
		MapLanguages = null;
		SourceLanguagesArray = null;
		TargetLanguagesArray = null;
		AllLanguagesArray = null;
		
		CdmElementsWithDefaultLanguage = null;

        return true;
	}
	
}
