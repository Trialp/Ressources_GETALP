/*
 *  Jibiki
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot & Gilles Serasset - GETA CLIPS IMAG
 * Projet Jibiki 
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
  *
 *
 */


package fr.imag.clips.papillon.business.dictionary;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;

import java.util.*;

public class EntryCache {
    
    
	private static int MAX_CACHE_SIZE = 200;
	
    // Volume cache
    private static Hashtable entryCacheByHandle;
    private static Hashtable entryCacheByEntryId;
    private static Hashtable entryCacheByContributionId;
    
    /** 
    * Initialize volume cache
    *
    * @exception PapillonBusinessException
    * 
    */
    public static void entryCacheInit() {            
                entryCacheByHandle = new Hashtable();
                entryCacheByEntryId = new Hashtable();
                entryCacheByContributionId = new Hashtable();
        }
    
    
    /** 
        * Put volume in caches
        *
        * @param volumeName name
        * @param vol sheet
        *
        * @exception PapillonBusinessException
        * 
        */
    public static void putEntryInCache(VolumeEntry entry) 
        throws PapillonBusinessException {
            
            try {
                // Key is equal to 'handle' (database object id)
				entryCacheByHandle.put(entry.getHandle(), entry);
	
               // PapillonLogger.writeDebugMsg("Entry " + entry.getHeadword() + " (" + entry.getHandle() + ") " + entry.getSourceLanguage() + " -> " + " load");
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in putVolumeInCache()", ex);
            }
        }
    
    
    /** 
        * Put collection of volumes in cache
        *
        * @param Hashtable
        * @param String
        * @param Volume
        *
        * @exception PapillonBusinessException
        * 
        */
    private static void putEntryInCache(Hashtable hash, String key, VolumeEntry entry)
        throws PapillonBusinessException {

        try {
            
            if (hash.containsKey(key)) {
                
                //
                ArrayList collection = (ArrayList)hash.get(key);
                collection.add(entry);
                hash.put(key, collection);
                
            } else {
                
                //
                ArrayList collection = new ArrayList();
                collection.add(entry);
                hash.put(key, collection);
            }
            
        } catch(Exception ex) {
            throw new PapillonBusinessException("Exception in putEntryInCache()", ex);
        }
        
    }
    
    /** 
        * Get volume by 'handle' (database object id)
        *
        * @return Volume
        *
        * @exception PapillonBusinessException
        * 
        */
    public static VolumeEntry getEntryInCacheByHandle(String handle) 
        throws PapillonBusinessException {
            VolumeEntry myEntry = null;
            try {
                
                //
                myEntry = (VolumeEntry)entryCacheByHandle.get(handle);
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getEntryInCacheByHandle()", ex);
            }
			PapillonLogger.writeDebugMsg("getEntryInCacheByHandle: " + handle + myEntry !=null ? myEntry.getHeadword() : "null");
			return myEntry;
        }
    
	public static void cutEntryCaches()  {
		if (entryCacheByHandle.size()> MAX_CACHE_SIZE)  {
			entryCacheByHandle.clear();
		}
	}

}