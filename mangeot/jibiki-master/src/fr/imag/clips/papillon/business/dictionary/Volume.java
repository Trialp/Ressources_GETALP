/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot & Gilles Serasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
*/

package fr.imag.clips.papillon.business.dictionary;

import com.lutris.appserver.server.sql.DBTransaction;
import com.lutris.appserver.server.sql.DatabaseManagerException;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.dods.builder.generator.query.DataObjectException;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.data.VolumeDO;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMResult;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamSource;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

/**
 * Represents a Volume. 
 */
public class Volume {
	
	public final static String LOCAL_LOCATION = "local";
	public final static String REMOTE_LOCATION = "remote";
	
	// All CDM elements
    public static final String CDM_volume = "cdm-volume";
	// This CDM element will not be compiled as an XPath
	// I t is used for matching the entry string in the text file
    public static final  String CDM_entryString = "cdm-entry-string";
    public static final  String CDM_entry = "cdm-entry";
    public static final  String CDM_entryId = "cdm-entry-id";
    public static final  String CDM_headword = "cdm-headword";
    public static final  String CDM_homographNumber = "cdm-homograph-number";
    public static final  String CDM_headwordVariant = "cdm-headword-variant";
    public static final  String CDM_reading = "cdm-reading";
    public static final  String CDM_writing = "cdm-writing";
    public static final  String CDM_pos = "cdm-pos";
    public static final  String CDM_pronunciation = "cdm-pronunciation";
    public static final  String CDM_definition = "cdm-definition";
    public static final  String CDM_domain = "cdm-domain";
    public static final  String CDM_templateEntry = "cdm-template-entry";
    public static final  String CDM_translation = "cdm-translation";
    // FIXME: Added by Gilles for distinction between translations in a bilingual dictionary and
    // FIXME: translation links to target entries in another volume.
    public static final  String CDM_translationReflexie = "cdm-translation-ref";
    public static final  String CDM_example = "cdm-example";
    public static final  String CDM_idiom = "cdm-idiom";
	
	// New CDM elements for links
    public static final  String CDM_link = "link";
    public static final  String CDM_linkElementId = "id";
    public static final  String CDM_linkLabel = "label";
    public static final  String CDM_linkLang = "lang";
    public static final  String CDM_linkType = "type";
    public static final  String CDM_linkValue = "value";
    public static final  String CDM_linkVolume = "volume";
    public static final  String CDM_linkWeight = "weight";
	public final static  String LINK_defaultLinkName = "unknown";
	public final static  String LINK_translationLinkName = "translation";
	
	// For linksTable
	public final static String LINK_ELEMENT_TYPE = "element";
	public final static String LINK_STRING_LANG_TYPE = "langString";
	
	
	// xpaths calculated from previous ones
    public static final  String CDM_headwordElement = "cdm-headword-element";
	public static final  String CDM_entryIdElement = "cdm-entry-id-element";
      
	// history tags
	public static final  String CDM_history = "cdm-history";
	public static final  String CDM_modification = "cdm-modification";
	public static final  String CDM_modificationAuthor = "cdm-modification-author";
    public static final  String CDM_modificationAuthorElement = "cdm-modification-author-element";
	public static final  String CDM_modificationDate = "cdm-modification-date";
    public static final  String CDM_modificationDateElement = "cdm-modification-date-element";
	public static final  String CDM_modificationComment = "cdm-modification-comment";
    public static final  String CDM_modificationCommentElement = "cdm-modification-comment-element";

	// contribution tags
    public static final  String CDM_contribution = "cdm-contribution";
    public static final  String CDM_contributionAuthor = "cdm-contribution-author";
    public static final  String CDM_contributionAuthorElement = "cdm-contribution-author-element";
    public static final  String CDM_contributionCreationDate = "cdm-contribution-creation-date";
    public static final  String CDM_contributionCreationDateElement = "cdm-contribution-creation-date-element";
    public static final  String CDM_contributionGroup = "cdm-contribution-group";
    public static final  String CDM_contributionGroups = "cdm-contribution-groups";
    public static final  String CDM_contributionId = "cdm-contribution-id";
    public static final  String CDM_contributionFinitionDate = "cdm-contribution-finition-date";
    public static final  String CDM_contributionFinitionDateElement = "cdm-contribution-finition-date-element";
    public static final  String CDM_contributionReviewDate = "cdm-contribution-review-date";
    public static final  String CDM_contributionReviewDateElement = "cdm-contribution-review-date-element";
    public static final  String CDM_contributionReviewer = "cdm-contribution-reviewer";
    public static final  String CDM_contributionReviewerElement = "cdm-contribution-reviewer-element";
    public static final  String CDM_contributionValidationDate = "cdm-contribution-validation-date";
    public static final  String CDM_contributionValidationDateElement = "cdm-contribution-validation-date-element";
    public static final  String CDM_contributionValidator = "cdm-contribution-validator";
    public static final  String CDM_contributionValidatorElement = "cdm-contribution-validator-element";
    public static final  String CDM_contributionStatus = "cdm-contribution-status";
    public static final  String CDM_contributionStatusElement = "cdm-contribution-status-element";
    public static final  String CDM_contributionDataElement = "cdm-contribution-data-element";
    public static final  String CDM_originalContributionId = "cdm-original-contribution-id";
    // FBM: Added by francis
    // link to previous classified finished/not-finished contribution
    public static final  String CDM_previousClassifiedFinishedContribution = "cdm-previous-classified-finished-contribution";
    public static final  String CDM_previousClassifiedFinishedContributionElement = "cdm-previous-classified-finished-contribution-element";
    public static final  String CDM_previousClassifiedNotFinishedContribution = "cdm-previous-classified-not-finished-contribution";
    public static final  String CDM_previousClassifiedNotFinishedContributionElement = "cdm-previous-classified-not-finished-contribution-element";
    // link to next contribution author
    public static final  String CDM_nextContributionAuthor = "cdm-next-contribution-author";
    public static final  String CDM_nextContributionAuthorElement = "cdm-next-contribution-author-element";

    // MM: Ajouté par Mathieu: simplifications !
    public static final  String CDM_previousContribution = "cdm-previous-contribution";
    public static final  String CDM_previousContributionElement = "cdm-previous-contribution-element";
	
	
	/* constants added to manage axies, it should be generic...*/
    public static final  String CDM_axiSemanticCat = "axi-semantic-cat";
    public static final  String CDM_axiSynonyms = "axi-synonyms";
    public static final  String CDM_axiLanguage = "axi-language";
    public static final  String CDM_axiLanguageAttribute = "axi-language-attribute";
    public static final  String CDM_axiReflexie = "axi-reflexie";
    public static final  String CDM_axiRefaxie = "axi-refaxie"; 

	// gdef tags
	public static final  String CDM_gdefEstParticule = "gdef-est-particule";
	public static final  String CDM_gdefEstDomaine = "gdef-est-domaine";
	public static final  String CDM_gdefEstFrequence = "gdef-est-frequence";
  
	// other constants
	public final static String DEFAULT_LANG = "#NA";
	
	//FIXME: CDM_translation, CDM_translationReflexie, CDM_linkWeight are linkElements
	public final static String[] indexElements = { 
		CDM_entryId, 
		CDM_headword,
		CDM_headwordVariant,
		CDM_reading,
		CDM_writing,
        CDM_pronunciation,
        CDM_definition,
		CDM_pos,
        CDM_domain,
        CDM_example,
		CDM_idiom,
		CDM_translation,
		CDM_translationReflexie,
		CDM_linkWeight,
		
// contribution CDM elements
		CDM_contributionId,
		CDM_originalContributionId,
		CDM_contributionAuthor,
		CDM_contributionGroup,
		CDM_contributionGroups,
		CDM_contributionCreationDate,
		CDM_contributionFinitionDate,
		CDM_contributionReviewDate,
		CDM_contributionReviewer,
		CDM_contributionValidationDate,
		CDM_contributionValidator,
		CDM_contributionStatus,
        CDM_previousContribution,
        CDM_previousClassifiedFinishedContribution,
        CDM_previousClassifiedNotFinishedContribution,
        CDM_nextContributionAuthor,
		
// gdef-est elements
		CDM_gdefEstParticule,
		CDM_gdefEstFrequence
		
	};
	
	//it does not work. all CDM_Elements are read from the String[] indexElements. Look VolumesFactory.java:createCdmElementsTable, buildCdmElementsTable addCdmElementInTable
	public final static String[] linkElements = {
		CDM_translationReflexie,
		CDM_linkValue,
		CDM_linkWeight,
	};
		
	public final static String[] langElements = { 
		CDM_headword,
		CDM_homographNumber,
		CDM_headwordVariant,
		CDM_reading,
		CDM_writing,
		CDM_pronunciation,
        CDM_definition,
		CDM_pos,
        CDM_domain,
		CDM_example,
		CDM_idiom,
		CDM_translation,
		CDM_translationReflexie,
		// gdef-est elements
		CDM_gdefEstParticule,
		CDM_gdefEstFrequence

	};
	
	public final static String[] defaultLangElements = { 
		CDM_volume, 
		CDM_entry, 
		CDM_entryId, 
		CDM_contributionId,
		CDM_originalContributionId,
		CDM_contributionAuthor,
		CDM_contributionGroup,
		CDM_contributionCreationDate,
		CDM_contributionFinitionDate,
		CDM_contributionReviewDate,
		CDM_contributionReviewer,
		CDM_contributionValidationDate,
		CDM_contributionValidator,
		CDM_contributionStatus,
        CDM_previousContribution,
		CDM_previousClassifiedFinishedContribution,
        CDM_previousClassifiedNotFinishedContribution,
		CDM_nextContributionAuthor
	};
	
	public final static String[] sourceLangElements = { 
		CDM_headword,
		CDM_homographNumber,
		CDM_headwordVariant,
		CDM_reading,
		CDM_writing,
		CDM_pronunciation,
        CDM_definition,
		CDM_pos,
        CDM_domain,
		CDM_example,
		CDM_idiom,
		// gdef-est elements
		CDM_gdefEstParticule,
		CDM_gdefEstFrequence
	};
	
	// local variables
	protected final static String XmlFooterSeparator = "$$$###$$$";

/*
 * table schema:
 * String lang (ISO 639-2/T) -> CDM_element -> ArrayList (String xpath, boolean is_index, XPath xpath_compiled) 
 */
    /*
     * table schema:
     * Hastable with key [String lang (ISO 639-2/T)] 
     * OF HashMap with key [CDM_element] 
     * OF ArrayList (String xpath, boolean is_index, XPath xpath_compiled) 
     */
	protected java.util.HashMap CDM_elements =  null;
	
 
    /*
     * table schema:
     * Hastable with key [name] (translation, xref, etc.)
     * Of HashMap with key [element] (element, value, target_volume, targetId, etc.)
     * Of ArrayList (String xpath, XPath xpath_compiled) 
     */
	protected java.util.HashMap linksTable =  null;
	protected org.apache.xml.utils.PrefixResolver prefixResolver = null;

	/**
	 * The DO of the Volume.
     */
    protected VolumeDO myDO = null;
	
    /**
	 * The public constructor.
     */
    public Volume() throws PapillonBusinessException {
        try {
		    this.CDM_elements = new java.util.HashMap();
		    this.linksTable = new java.util.HashMap();
            this.myDO = VolumeDO.createVirgin(CurrentDBTransaction.get());  
        } catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error creating empty Volume", ex);
        } catch(ObjectIdException ex) {
            throw new PapillonBusinessException("Error creating empty Volume", ex);
        }
    }
	
    /** The protected constructor
		*
		* @param theDisc. The data object of the Volume.
		*/
    protected Volume(VolumeDO theVolumeDO) 
        throws PapillonBusinessException  {
			this.myDO = theVolumeDO;
		}
	
    public boolean isEmpty() {
        return (this.myDO==null) ;
    }
	
	
    /**
	 * Gets the object id for the Volume
     *
     * @return the object id.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getHandle()
        throws PapillonBusinessException {
			try {
				return this.myDO.getHandle();
			} catch(DatabaseManagerException ex) {
				throw new PapillonBusinessException("Error getting Volumes's handle", ex);
			}
		}
	
    /**
	 * Gets the name of the Volume
     *
     * @return the name.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getName()
        throws PapillonBusinessException {
            try {
                return myDO.getName();
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error getting Volume's name", ex);
            }
        }
    public void setName(String volume)
        throws PapillonBusinessException {
            try {
                myDO.setName(volume);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting Volume's name", ex);
            }
        }
    
	
    /**
	 * Gets the dictname of the Volume
     *
     * @return the dictname.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getDictname() 
        throws PapillonBusinessException {
			try {
				return myDO.getDictname();   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting Volume's dictname", ex);
			}
		}
	public void setDictname(String dictname) 
        throws PapillonBusinessException {
			try {
				myDO.setDictname(dictname);   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting Volume's dictname", ex);
			}
		}
	
	
    /**
	 * Gets the dbname of the Volume
     *
     * @return the dbname.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getDbname()
        throws PapillonBusinessException {
            try {
                return myDO.getDbname();
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error getting Volume's Dbname", ex);
            }
        }
    public void setDbname(String dbname)
        throws PapillonBusinessException {
            try {
                myDO.setDbname(dbname);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting Volume's dbname", ex);
            }
        }
	
	
    public String getIndexDbname()
        throws PapillonBusinessException {
			return IndexFactory.INDEX_TABLE_PREFIX + getDbname();
		}
    
    public String getLinkDbname() 
    	throws PapillonBusinessException {
    		return LinkFactory.LINK_TABLE_PREFIX + getDbname();
    }
	
    /**
	 * Gets the location of the Volume
     *
     * @return the dbname.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getLocation()
        throws PapillonBusinessException {
            try {
                return myDO.getLocation();
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error getting Volume's location", ex);
            }
        }
    public void setLocation(String location)
        throws PapillonBusinessException {
            try {
                myDO.setLocation(location);
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error setting Volume's location", ex);
            }
        }
	
	
    /**
	 * Gets the source language of the Volume
     *
     * @return the source language.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getSourceLanguage() 
        throws PapillonBusinessException {
			try {
				return myDO.getSourceLanguage();   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting Volume's source language", ex);
			}
		}
    
    // FIXME: change String to Collection
	public void setSourceLanguage(String lang) 
        throws PapillonBusinessException {
			try {
				myDO.setSourceLanguage(lang);   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting Volume's source language", ex);
			}
		}
	
	
    /**
	 * Gets the target languages of the Volume
     *
     * @return the target languages.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    // FIXME: change String to Collection
    // FIXME: supress getTargetLanguagesArray ? or getTargetLanguages ?
    public String getTargetLanguages() 
        throws PapillonBusinessException {
			try {
				return myDO.getTargetLanguages();   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting Volume's target languages", ex);
			}
		}

	public Collection getTargetLanguagesArray() throws PapillonBusinessException {
			
        //return getTargetLanguages().split("\\s");
        // FIXME: add target collection in volume
        String[] targetsString = getTargetLanguages().split("\\s");
        ArrayList targets = new ArrayList();
        for (int i = 0; i < targetsString.length; i++) {
            targets.add(targetsString[i]);
        }
        
        return targets;
    }

    // FIXME: change String to Collection
	public void setTargetLanguages(String langs) 
        throws PapillonBusinessException {
			try {
				myDO.setTargetLanguages(langs);   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting Volume's target languages", ex);
			}
		}
	
	public String getDefaultTargetVolumeName(String target) 
		throws PapillonBusinessException {
		String volumeName = null;
		Collection Volumes1 = VolumesFactory.getVolumesArray(this.getDictname(), target, null);
		if (!Volumes1.isEmpty() && Volumes1.size()>0) {
			Volume firstVolume1 = (Volume)(Volumes1.iterator()).next();
			volumeName = firstVolume1.getName();
		}
		return volumeName;
	}




	
    /**
	 * Gets the href of the Volume
     *
     * @return the href.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getVolumeRef() 
        throws PapillonBusinessException {
			try {
				return myDO.getVolumeRef();   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting Volume's href", ex);
			}
		}
	public void setVolumeRef(String href) 
        throws PapillonBusinessException {
			try {
				myDO.setVolumeRef(href);   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting Volume's href", ex);
			}
		}
	
    /**
	 * Gets the CDM elements HashMap of the Volume
     *
     * @return the  CDM elements HashMap.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public java.util.HashMap getCdmElements() {
		return this.CDM_elements;
	}
	
    /**
	 * Gets the links HashMap of the Volume
     *
     * @return the  links HashMap.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public java.util.HashMap getLinksTable() {
		return this.linksTable;
	}
	

	/**
	 * Gets a CDM XPath of the Volume
     *
     * @param the name of the CDM XPath as a String.
     * @return the CDM XPath as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    protected String getCdmXPathString(String name) {
		return getCdmXPathString(name,DEFAULT_LANG);
	}
	
    /**
	 * Gets a CDM XPath of the Volume
     *
     * @param the name of the CDM element as a String.
     * @param the language of the CDM element as a ISO 639-2/T 3 letters String.
     * @return the CDM XPath as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    protected String getCdmXPathString(String name, String lang) {
		String res = null;
		java.util.HashMap tmpTable = (java.util.HashMap) this.CDM_elements.get(lang);
		if (tmpTable != null) {
			java.util.ArrayList myArrayList = (java.util.ArrayList) tmpTable.get(name);
			if (myArrayList != null && myArrayList.size()>0) {
				res = (String) myArrayList.get(0);
			}
		}
		else {
			fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Error 1: CDM Element null for name: " + name + " " + lang + " HashMap: " + this.CDM_elements.toString());
		}
		if (res == null || res.equals("")) {
			fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Error 2: CDM Element null for name: " + name + " " + lang + " HashMap: " + this.CDM_elements.toString());
		}
		// fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("getCdmXPathString: " + name + " : " + res);
		return res;
	}
	
	public static String getTagNameFromXPath (String xPathString) {
		return getTagNameFromXPath(xPathString, true);
	}
	
	protected static String getTagNameFromXPath (String xPathString, boolean letNamespacePrefix) {
		// fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("getTagNameFromXPath: " + xPathString);
		String res = xPathString;
		if (res != null) {
			int text = res.indexOf("/text()");
			if (text>0) {
				res = res.substring(0,text);
			}
			int index = res.lastIndexOf("/");
			if (index>=0) {
				res = res.substring(index+1);
			}
			if (!letNamespacePrefix) {
				int colon = res.lastIndexOf(":");
				if (colon>=0) {
					res = res.substring(colon+1);
				}
			}
		}
		return res;
	}
	
    /**
	 * Gets the CDM volume of the Volume
     *
     * @return the CDM volume tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmVolume() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_volume));
	}

    /**
	 * Gets the CDM entry of the Volume
     *
     * @return the  CDM entry tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmEntry() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_entry));
	}

    /**
	 * Gets the CDM entryString of the Volume
     *
     * @return the  CDM entry tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmEntryString() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_entryString));
	}
	
    /**
	 * Gets the local tag name of the CDM entry of the Volume
     *
     * @return the  CDM entry tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmLocalEntry() {
		return Utility.getLocalTagName(getTagNameFromXPath(getCdmXPathString(this.CDM_entry)));
	}

    public String getCdmContribution() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_contribution));
	}

    /**
	 * Gets the CDM headword of the Volume
     *
     * @return the  CDM headword tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmHeadword() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_headword, this.getSourceLanguage()));
	}

    /**
	 * Gets the CDM history of the Volume
     *
     * @return the  CDM history tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmHistory() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_history));
	}

    /**
	 * Gets the CDM modification of the Volume
     *
     * @return the  CDM modification tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmModification() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_modification));
	}

    /**
	 * Gets the CDM author of the Volume
     *
     * @return the  CDM author tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmModificationAuthor() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_modificationAuthor));
	}
	
    public String getCdmModificationAuthorElement() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_modificationAuthorElement));
	}
	
    /**
	 * Gets the CDM date of the Volume
     *
     * @return the  CDM date tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmModificationDate() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_modificationDate));
	}

    public String getCdmModificationDateElement() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_modificationDateElement));
	}
	
    /**
	 * Gets the CDM comment of the Volume
     *
     * @return the CDM comment tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmModificationComment() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_modificationComment));
	}

	public String getCdmModificationCommentElement() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_modificationCommentElement));
	}
	
    /**
	 * Gets the CDM groups element of the Volume
     *
     * @return the CDM comment tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmContributionGroups() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_contributionGroups));
	}
	
    /**
	 * Gets the CDM group of the Volume
     *
     * @return the CDM comment tag name as a String.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getCdmContributionGroup() {
		return getTagNameFromXPath(getCdmXPathString(this.CDM_contributionGroup));
	}
	
	/**
	 * Sets the CDM elements HashMap of the Volume
     *
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
	public void setCdmElements() 
        throws PapillonBusinessException {
			org.w3c.dom.Document myDoc = XMLServices.buildDOMTree(this.getTemplateEntry());
			String dml_prefix = VolumesFactory.getDmlPrefix(myDoc.getDocumentElement());
			this.CDM_elements = VolumesFactory.buildCdmElementsTable(this.getXmlCode(), this.getSourceLanguage(), this.getPrefixResolver(), dml_prefix);
	}
	
	/**
	 * Sets the CDM elements HashMap of the Volume
     *
     * @param the CDM elements as an HashMap.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
	public void setCdmElements(java.util.HashMap elements) {
        this.CDM_elements = elements;
	}
	
    /**
	 * Gets the xml code of the volume metadata file
     *
     * @return the xml code as a string.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */

	/**
	 * Sets the Links HashMap of the Volume
     *
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
	public void setLinksTable() 
	throws PapillonBusinessException {
		org.w3c.dom.Document myDoc = XMLServices.buildDOMTree(this.getTemplateEntry());
		String dml_prefix = VolumesFactory.getDmlPrefix(myDoc.getDocumentElement());
		this.linksTable = VolumesFactory.buildLinksTable(this.getXmlCode(), this.getPrefixResolver(), this.getCdmElements(), dml_prefix);
	}
	
	/**
	 * Sets the Links HashMap of the Volume
     *
     * @param the CDM elements as an HashMap.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
	public void setLinksTable(java.util.HashMap elements) {
        this.linksTable = elements;
	}
	
	
	public String getXmlCode()
        throws PapillonBusinessException {
			try {
				return this.myDO.getXmlCode();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting Volume's xmlcode", ex);
			} 
		}
	public void setXmlCode(String code)
        throws PapillonBusinessException {
			try {
		  		myDO.setXmlCode(code);   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting Volume's xmlcode", ex);
			}
		}	
	
    public String getFormatterClassName() throws PapillonBusinessException {
        NodeList formatters = XMLServices.buildDOMTree(this.getXmlCode()).getElementsByTagName("result-formatter");
        String classname = null;
        if ((null != formatters) && (formatters.getLength() > 0)) {
            // FIXME : no .getFirstChild needs
            classname = ((Element) formatters.item(0).getFirstChild()).getAttribute("class-name");
        }
        return classname;
	}
   
    public String getPreProcessorClassName() throws PapillonBusinessException {
        NodeList preProcessor = XMLServices.buildDOMTree(this.getXmlCode()).getElementsByTagName("result-preprocessor");
        String classname = null;
        if ((null != preProcessor) && (preProcessor.getLength() > 0)) {
            // FIXME : no .getFirstChild needs
            classname = ((Element) preProcessor.item(0).getFirstChild()).getAttribute("class-name");
        }
        return classname;
	}
    
    public String getPostUpdateProcessorClassName() throws PapillonBusinessException {
        NodeList postProcessor = XMLServices.buildDOMTree(this.getXmlCode()).getElementsByTagName("result-postupdateprocessor");
        String classname = null;
        if ((null != postProcessor) && (postProcessor.getLength() > 0)) {
            // FIXME : no .getFirstChild needs
            classname = ((Element) postProcessor.item(0).getFirstChild()).getAttribute("class-name");
        }
        return classname;
	}
    
    public String getPostSaveProcessorClassName() throws PapillonBusinessException {
        NodeList postProcessor = XMLServices.buildDOMTree(this.getXmlCode()).getElementsByTagName("result-postsaveprocessor");
        String classname = null;
        if ((null != postProcessor) && (postProcessor.getLength() > 0)) {
            // FIXME : no .getFirstChild needs
            classname = ((Element) postProcessor.item(0).getFirstChild()).getAttribute("class-name");
        }
        return classname;
	}
    
    
    /**
	 * Gets the xml code of the volume xml schema file
     *
     * @return the xml code as a string.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getXmlSchema() throws PapillonBusinessException {
        try {
			return this.myDO.getXmlSchema();
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting Volume's xmlschema code", ex);
        } 
	}
    
	public void setXmlSchema(String code)
        throws PapillonBusinessException {
			try {
		  		myDO.setXmlSchema(code);   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting Volume's xmlschema code", ex);
			}
		}
	
	/**
	 * Gets the xml code of the edition interface template
     *
     * @return the xml code as a string.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getTemplateInterface()
		throws PapillonBusinessException {
			try {
				return this.myDO.getTemplateItf();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting volume TemplateItf", ex);
			}
		}
    
	/**
	 * Sets the xml code of the edition interface template
     *
     * @param the xml code as a string.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
	public void setTemplateInterface(String code)
        throws PapillonBusinessException {
			try {
		  		myDO.setTemplateItf(code);   
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting volume's TemplateItf code", ex);
			}
		}
	
    /**
	 * Gets the xml code of  template entry of the volume
     *
     * @return the xml code as a string.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String getTemplateEntry()
		throws PapillonBusinessException {
			try {
				return this.myDO.getTemplateEntry();
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting volume template entry code", ex);
			}
		}
	
	public void setTemplateEntry(String code)
		throws PapillonBusinessException {
			try {
				myDO.setTemplateEntry(code);
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting volume template entry code", ex);
			}
		}
	
    /**
	 * Gets the prefix resolver of the volume XML
     *
     * @return the xml prefix resolver.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public org.apache.xml.utils.PrefixResolver getPrefixResolver()
	throws PapillonBusinessException {
			if (this.prefixResolver == null) {
				org.w3c.dom.Document myDoc = XMLServices.buildDOMTree(this.getTemplateEntry());
				this.prefixResolver = new org.apache.xml.utils.PrefixResolverDefault(myDoc.getDocumentElement());

			}
			return this.prefixResolver;
	}

    /**
	 * Gets the xml code of  template entry of the volume
     *
     * @return the xml code as a string.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public boolean IsReverseLookup()
		throws PapillonBusinessException {
			try {
                String rev = this.myDO.getReverse();
				return ((rev != null) && rev.equals("Y"));
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error getting volume reverse lookup", ex);
			}
		}
		
	public void setReverseLookup(boolean reverse)
		throws PapillonBusinessException {
			try {
				myDO.setReverse(reverse ? "Y" : "N");
			} catch(DataObjectException ex) {
				throw new PapillonBusinessException("Error setting volume reverse lookup", ex);
			}
		}
	
    /**
		* IsIndexCDMElement tells if the CDM element has to be indexed in the db
     *
     * @return a boolean.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
						  *   error).
     */
    public static boolean isIndexCDMElement(String eltName) {
		return java.util.Arrays.asList(indexElements).contains(eltName);
	}
    
    
    public static boolean isLinkCDMElement(String eltName) {
    	return java.util.Arrays.asList(linkElements).contains(eltName);
    }
	
    /**
		* IsLangCDMElement tells if the CDM element needs a language 
	 * if yes, we will add the source languageif no one was specified
     *
     * @return a boolean.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
						  *   error).
     */
    public static boolean isLangCDMElement(String eltName) {
		return java.util.Arrays.asList(langElements).contains(eltName);
	}
	
    /**
		* IsSourceLangCDMElement tells if the CDM element needs a language 
	 * if yes, we will add the source language if no one was specified
     *
     * @return a boolean.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
						  *   error).
     */
    public static boolean isSourceLangCDMElement(String eltName) {
		return java.util.Arrays.asList(sourceLangElements).contains(eltName);
	}
	
    /**
		* IsDefaultLangCDMElement tells if the CDM element needs a language 
	 * if yes, we will add the source language if no one was specified
     *
     * @return a boolean.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
						  *   error).
     */
    public static boolean isDefaultLangCDMElement(String eltName) {
		return java.util.Arrays.asList(defaultLangElements).contains(eltName);
	}
	
    /**
	 * Gets the xml code footer of the volume
     *
     * @return the xml code as a string.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public String[] getXmlHeaderAndFooter()
		throws PapillonBusinessException {
			
            String xmlHeader = "<" + this.getCdmVolume() + ">";
            String xmlFooter = "</" + this.getCdmVolume() + ">";
			
			String templateEntry = this.getTemplateEntry();
			
			org.w3c.dom.Document templateDoc = XMLServices.buildDOMTree(templateEntry);
			org.w3c.dom.NodeList volumeNodes = IndexEntry.getCdmElements(templateDoc, Volume.CDM_volume, Volume.DEFAULT_LANG, this.getCdmElements(), this.getPrefixResolver()); 

			if (volumeNodes != null && volumeNodes.getLength() >0) {
				org.w3c.dom.Node volumeNode = volumeNodes.item(0);
                while (volumeNode.hasChildNodes())
                    volumeNode.removeChild(volumeNode.getFirstChild());
				org.w3c.dom.Text myTextNode = templateDoc.createTextNode(Volume.XmlFooterSeparator);
				volumeNode.appendChild(myTextNode);
				templateEntry = XMLServices.xmlCode(templateDoc);
                int XmlHeaderSeparatorIndex = templateEntry.indexOf(Volume.XmlFooterSeparator);
                int XmlFooterSeparatorIndex = templateEntry.lastIndexOf(Volume.XmlFooterSeparator);
                xmlHeader = templateEntry.substring(0,XmlHeaderSeparatorIndex);
				xmlFooter = templateEntry.substring(XmlFooterSeparatorIndex+Volume.XmlFooterSeparator.length());
                //fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("getXmlHeaderAndFooter: volumesNodes not null: " + xmlHeader + " " + xmlFooter);
			}
			else {
				fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("getXmlHeaderAndFooter: volumesNodes null: " + xmlHeader + " " + xmlFooter);
			}
			return new String[] {xmlHeader,xmlFooter};
 		}
    
    public String getXmlFooter() throws PapillonBusinessException {
        return (String) getXmlHeaderAndFooter()[1];
    }

    /**
	 * getCount get entries number of the volume
     *
     * @return the number of entries as an integer.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
	public int getCount()  
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
			return VolumeEntriesFactory.getVolumeEntriesCount(this);
	}

	public int getCount(String status) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
			return VolumeEntriesFactory.getVolumeEntriesCount(this, status);
		}

	/**
	 * getEntries gets entries number of the volume
     *
     * @return the number of entries as an integer.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public int getEntries()
	throws PapillonBusinessException {
		try {
			return myDO.getEntries();
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error getting Volume's entries", ex);
		}
	}
	/**
	 * setEntries sets entries number of the volume
     *
     * @return the number of entries as an integer.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
	 *   error).
     */
    public void setEntries(java.lang.Integer entries)
	throws PapillonBusinessException {
		try {
			myDO.setEntries(entries.intValue());
		} catch(DataObjectException ex) {
			throw new PapillonBusinessException("Error setting Volume's entries", ex);
		}
	}	
	
    public void deleteDefaultXslSheet() throws PapillonBusinessException {
        XslSheetFactory.removeXslSheet("",this.getDictname(),this.getName());
    }

    /**
	 * Saves the Volume into the database.
     *
     * @exception PapillonBusinessException if an error occurs
     *   deleting data (usually due to an underlying data layer
	 *   error).
     */
    public void save() 
        throws PapillonBusinessException {
			try {
				// Reset AvailableLanguages caches before saving because it will be modified
				fr.imag.clips.papillon.business.dictionary.AvailableLanguages.resetCache();
				this.myDO.commit();
			} catch(Exception ex) {
				throw new PapillonBusinessException("Error saving Volume", ex);
			}
		}
    
    /**
	 * Empties the Volume from the database.
     *
     * @exception PapillonBusinessException if an error occurs
     *   deleting data (usually due to an underlying data layer
	 *   error).
     */
	public void empty() 
	throws PapillonBusinessException {
		try {
			if (this.getLocation().equals(LOCAL_LOCATION)) {
				VolumeEntriesFactory.truncateVolumeTables(this);
			}
		} catch(Exception ex) {
			throw new PapillonBusinessException("Error emptying Volume", ex);
		}
	}

    /**
	 * Deletes the Volume from the database.
     *
     * @exception PapillonBusinessException if an error occurs
     *   deleting data (usually due to an underlying data layer
	 *   error).
     */
    public void deleteAll() 
        throws PapillonBusinessException {
			try {
                deleteDefaultXslSheet();
				if (this.getLocation().equals(LOCAL_LOCATION)) {
					VolumeEntriesFactory.dropVolumeTables(this);
				}
				this.myDO.delete();
			} catch(Exception ex) {
				throw new PapillonBusinessException("Error deleting Volume", ex);
			}
		}
	
    public void delete() 
        throws PapillonBusinessException {
			try {
				this.myDO.delete();
			} catch(Exception ex) {
				throw new PapillonBusinessException("Error deleting Volume", ex);
			}
		}



}
