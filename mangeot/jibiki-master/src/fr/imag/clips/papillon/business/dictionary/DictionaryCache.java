/*
 *  Jibiki
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Francis Brunet-Manquat, Mathieu Mangeot & Gilles Serasset - GETA CLIPS IMAG
 * Projet Jibiki 
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.3  2007/04/05 12:55:54  serasset
 * Added a DBLayer Version management with an auto-update of db layer.
 *
 * Revision 1.2  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.1  2006/08/10 22:17:12  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 *
 */


package fr.imag.clips.papillon.business.dictionary;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;

import java.util.Collection;
import java.util.Hashtable;


public class DictionaryCache {
    
    
    // Dictionary cache
    private static java.util.TreeMap dictionaryCache;
    private static Hashtable dictionaryCacheByHandle; 
    
    /** 
    * Initialize dictionary cache
    *
    * @exception PapillonBusinessException
    * 
    */
    public static void dictionaryCacheInit() {
        dictionaryCache = new java.util.TreeMap();
        dictionaryCacheByHandle = new Hashtable();
    }
    
    
    /** 
        * Put dictionary in cache with dictionary name key
        *
        * @param dictionary name
        * @param xsl sheet
        *
        * @exception PapillonBusinessException
        * 
        */
    public static void putDictionaryInCache(String dictionaryName, Dictionary dict) 
        throws PapillonBusinessException {
            
            try {
                
                //
                PapillonLogger.writeDebugMsg("Dictionary " + dictionaryName +" load");
                dictionaryCache.put(dictionaryName, dict);
                
                // Key is equal to 'handle' (database object id)
                dictionaryCacheByHandle.put(dict.getHandle(), dict);

            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in putDictionaryInCache()", ex);
            }
        }
    
    
    /** 
        * Get dictionaries in cache
        *
        * @return collection of Dictionaries
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Collection getDictionariesInCache() 
        throws PapillonBusinessException {
            
            try {
                
                //
                return dictionaryCache.values();
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getDictionariesInCache()", ex);
            }
        }
    
    
    /** 
        * Get dictionary
        *
        * @return Dictionary
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Dictionary getDictionaryInCache(String key) 
        throws PapillonBusinessException {
            
            try {
                
                //
                return (Dictionary)dictionaryCache.get(key);
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getDictionaryInCache()", ex);
            }
        }

    
    /** 
        * Get dictionary by 'handle' (database object id)
        *
        * @return Dictionary
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Dictionary getDictionaryInCacheByHandle(String handle) 
        throws PapillonBusinessException {
            
            try {
                
                //
                return (Dictionary)dictionaryCacheByHandle.get(handle);
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getDictionaryInCacheByHandle()", ex);
            }
        }
}
