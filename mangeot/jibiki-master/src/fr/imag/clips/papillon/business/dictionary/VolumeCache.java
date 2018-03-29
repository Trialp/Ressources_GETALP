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
 * Revision 1.4  2007/04/05 12:55:54  serasset
 * Added a DBLayer Version management with an auto-update of db layer.
 *
 * Revision 1.3  2007/01/16 13:28:31  serasset
 * Added cache reinitialization when a metadata is modified.
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

import java.util.*;

public class VolumeCache {
    
    
    // Volume cache
    private static java.util.TreeMap volumeCache;
    private static java.util.TreeMap volumeCacheByIndexDbname;
    private static java.util.TreeMap nonStandardCDMElements;
    private static Hashtable volumeCacheByHandle;
    private static Hashtable volumeCacheByDictionaryName;
    private static Hashtable volumeCacheBySource;
    private static Hashtable volumeCacheByTarget;
    
    /** 
    * Initialize volume cache
    *
    * @exception PapillonBusinessException
    * 
    */
    public static void volumeCacheInit() {            
		volumeCache = new java.util.TreeMap();
		volumeCacheByIndexDbname = new java.util.TreeMap();
		volumeCacheByHandle = new Hashtable();
		volumeCacheByDictionaryName = new Hashtable();
		volumeCacheBySource = new Hashtable();
		volumeCacheByTarget = new Hashtable();
		nonStandardCDMElements = new java.util.TreeMap();
	}
    
    
	
	protected static void putNonStandardCDMElementsInCache(java.util.HashMap CdmElementsTable)  {
        if (CdmElementsTable == null) {
            PapillonLogger.writeDebugMsg("putNonStandardCDMElementsInCache: CdmElementsTable null");
        }
		for (Iterator langKeys = CdmElementsTable.keySet().iterator(); langKeys.hasNext();) {
			String lang = (String) langKeys.next();
			
			java.util.HashMap tmpTable = (java.util.HashMap) CdmElementsTable.get(lang);
			for (Iterator keys = tmpTable.keySet().iterator(); keys.hasNext();) {
				String CdmElement = (String) keys.next();
				
				if (CdmElement.length()<4 || (CdmElement.length()>=4 && !CdmElement.substring(0,4).equals("cdm-"))) {
					ArrayList myArrayList = (ArrayList) tmpTable.get(CdmElement);
					boolean isIndex = false;
					if (myArrayList != null) {
						isIndex = ((Boolean) myArrayList.get(1)).booleanValue();
						if (isIndex) {
							PapillonLogger.writeDebugMsg("putNonStandardCDMElementInCache, key " + CdmElement + ":");
							nonStandardCDMElements.put(CdmElement,lang);
						}
					}
				}
			}
		}		
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
    public static void putVolumeInCache(String volumeName, Volume vol) 
        throws PapillonBusinessException {
            
            try {
                
                // Key is equal to volume name
                volumeCache.put(volumeName, vol);
				putNonStandardCDMElementsInCache(vol.getCdmElements());
                
                // Key is equal to volume name
                volumeCacheByIndexDbname.put(vol.getIndexDbname(), vol);
                
                // Key is equal to 'handle' (database object id)
                volumeCacheByHandle.put(vol.getHandle(), vol);
                
                // Key is equal to dictionary name
                putVolumeInCache(volumeCacheByDictionaryName, vol.getDictname(), vol);
                
                // Key is equal to source
                putVolumeInCache(volumeCacheBySource, vol.getSourceLanguage(), vol);
                
                // Key is equal to target
                Collection targets = vol.getTargetLanguagesArray();
                String targetsList = "";
                for (Iterator iter = targets.iterator(); iter.hasNext();) {
                    String target = (String)iter.next();
                    
                    //
                    putVolumeInCache(volumeCacheByTarget, target, vol);
                    targetsList += target;
					if (iter.hasNext())  {
						targetsList += ",";
					}
                }
                
                //
                PapillonLogger.writeDebugMsg("Volume " + volumeName + " (" + vol.getHandle() + ") [" + vol.getDictname() + "] " + vol.getSourceLanguage() + " -> " + targetsList + " load");
                
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
    private static void putVolumeInCache(Hashtable hash, String key, Volume volume)
        throws PapillonBusinessException {

        try {
            
            if (hash.containsKey(key)) {
                
                //
                ArrayList collection = (ArrayList)hash.get(key);
                collection.add(volume);
                hash.put(key, collection);
                
            } else {
                
                //
                ArrayList collection = new ArrayList();
                collection.add(volume);
                hash.put(key, collection);
            }
            
        } catch(Exception ex) {
            throw new PapillonBusinessException("Exception in putVolumeInCache()", ex);
        }
        
    }
    
    
    /** 
        * Get volumes in cache
        *
        * @return collection of Volumes
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Collection getVolumesInCache() 
        throws PapillonBusinessException {
            Collection collection = null;
            try {
 				collection = volumeCache.values();
                
                if (collection == null) {
                    collection = new ArrayList();
                } 
            } catch(Exception ex) {
                //throw new PapillonBusinessException("Exception in getVolumesInCache()", ex);
            }
			return collection;
        }
    
    
    /** 
        * Get volume
        *
        * @return Volume
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Volume getVolumeInCache(String key) 
        throws PapillonBusinessException {
            
            try {
                
                //
                return (Volume)volumeCache.get(key);
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getVolumeInCache()", ex);
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
    public static Volume getVolumeInCacheByHandle(String handle) 
        throws PapillonBusinessException {
            
            try {
                
                //
                return (Volume)volumeCacheByHandle.get(handle);
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getVolumeInCacheByHandle()", ex);
            }
        }
    
	/** 
	 * Get volume by 'indexdbname' (database object id)
	 *
	 * @return Volume
	 *
	 * @exception PapillonBusinessException
	 * 
	 */
    public static Volume getVolumeInCacheByIndexDbname(String indexDbname) 
	throws PapillonBusinessException {
		
		try {
			
			//
			return (Volume)volumeCacheByIndexDbname.get(indexDbname);
			
		} catch(Exception ex) {
			throw new PapillonBusinessException("Exception in getVolumeInCacheByHandle()", ex);
		}
	}
    

    /** 
        * Get volumes by dictionary name
        *
        * @return Collection
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Collection getVolumesInCacheByDictionaryName(String dictionaryName) 
        throws PapillonBusinessException {
            
            try {
                
                //
                ArrayList collection = (ArrayList)volumeCacheByDictionaryName.get(dictionaryName);
                
                //
                if (collection == null) {
                    return new ArrayList();
                } else {
                    return collection;
                }
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getVolumesInCacheByDictionaryName()", ex);
            }
        }
    
    /** 
        * Get volumes by source language
        *
        * @return Collection
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Collection getVolumesInCacheBySourceLanguage(String source) 
        throws PapillonBusinessException {
            
            try {

                //
                ArrayList collection = (ArrayList)volumeCacheBySource.get(source);
                
                //
                if (collection == null) {
                    return new ArrayList();
                } else {
                    return collection;
                }
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getVolumesInCacheBySource()", ex);
            }
        }
    
    /** 
        * Get all source languages
        *
        * @return Collection
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Enumeration getSourceLanguagesInCache() 
        throws PapillonBusinessException {
            
            try {
                
                //
                return volumeCacheBySource.keys();
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getSourceLanguagesInCache()", ex);
            }
        }
    
    
    
    /** 
        * Get volumes by target language
        *
        * @return Enumeration
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Collection getVolumesInCacheByTargetLanguage(String target) 
        throws PapillonBusinessException {
            
            try {
                
                //
                ArrayList collection = (ArrayList)volumeCacheByTarget.get(target);
                
                //
                if (collection == null) {
                    return new ArrayList();
                } else {
                    return collection;
                }
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getVolumesInCacheByTarget()", ex);
            }
        }
    
    
    /** 
        * Get all target languages
        *
        * @return Enumeration
        *
        * @exception PapillonBusinessException
        * 
        */
    public static Enumeration getTargetLanguagesInCache() 
        throws PapillonBusinessException {
            
            try {
                
                //
                return volumeCacheByTarget.keys();
                
            } catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getTargetLanguagesInCache()", ex);
            }
        }
    
	
	public static Collection getNonStandardCDMElements() 
		throws PapillonBusinessException {
		try {
			
			//
			return nonStandardCDMElements.keySet();
			
		} catch(Exception ex) {
			throw new PapillonBusinessException("Exception in getNonStandardCDMElements()", ex);
		}
	}
	
}

