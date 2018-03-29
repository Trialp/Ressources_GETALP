/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.52.2.3  2007/09/06 14:54:11  serasset
 * in lexalp view, non matching entries are now gathered by legal systems and
 * query language now appears first.
 *
 * Revision 1.52.2.2  2007/09/05 15:24:13  serasset
 * Created a page to browse the dictionary index
 * Lexalp formatter now sorts entries correctly within a legal sysem/language group
 *
 * Revision 1.52.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.52  2007/02/09 08:49:10  fbrunet
 * *** empty log message ***
 *
 * Revision 1.51  2007/02/08 15:24:07  fbrunet
 * *** empty log message ***
 *
 * Revision 1.50  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.49  2007/01/15 17:12:18  serasset
 * Several notes added, suppressed the HTMLDOM_CACHE stuff.
 *
 * Revision 1.48  2007/01/08 15:13:42  fbrunet
 * Correction of th xml attribut bug in ContributionHeader (VolumeEntry class)
 *
 * Revision 1.47  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.45  2006/12/14 20:03:26  fbrunet
 * Add method to normalize value into XML structure.
 *
 * Revision 1.44  2006/12/13 09:32:00  fbrunet
 * *** empty log message ***
 *
 * Revision 1.43  2006/11/09 09:04:42  fbrunet
 * *** empty log message ***
 *
 * Revision 1.42  2006/08/10 22:17:12  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.41  2006/05/22 22:45:54  fbrunet
 * LexALP: add merge method in post-save processing (merge axies with same referenced lexies)
 *
 * Revision 1.40  2006/05/05 02:08:23  fbrunet
 * bug correction : url utf8 transfert (in createEntryInit)
 *
 * Revision 1.39  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 * Revision 1.38  2006/03/27 11:47:34  mangeot
 * *** empty log message ***
 *
 * Revision 1.37  2006/03/27 11:46:18  mangeot
 * ReAdded getReviewDate
 *
 * Revision 1.36  2006/03/27 10:56:38  mangeot
 * Syntax fix
 *
 * Revision 1.35  2006/03/27 10:47:34  mangeot
 * Added finition-date in contribution metadata
 *
 * Revision 1.34  2006/03/13 08:48:00  fbrunet
 * bug corrections before merge
 *
 * Revision 1.33  2006/03/12 23:19:18  mangeot
 * Fixed a bug that forbed to retrieve entries by translation ids
 *
 * Revision 1.32  2006/03/06 10:17:53  mangeot
 * Added setDeleted
 *
 * Revision 1.31  2006/03/02 10:58:36  mangeot
 * *** empty log message ***
 *
 * Revision 1.30  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.29  2006/02/22 19:05:56  mangeot
 * MM: Added default status choice when importing entries
 *
 * Revision 1.28  2006/02/21 13:37:54  mangeot
 * *** empty log message ***
 *
 * Revision 1.27  2005/12/04 15:22:39  mangeot
 * Fixed the volume parsing when the volume element is not the root element
 *
 * Revision 1.26  2005/12/01 16:13:42  mangeot
 * *** empty log message ***
 *
 * Revision 1.25  2005/11/23 13:42:27  mangeot
 * Added cdmEntryIdElement for setting the entry id even if it is not an attribute
 *
 * Revision 1.24  2005/11/14 22:49:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.23  2005/11/09 17:38:59  mangeot
 * small bug fixes
 *
 * Revision 1.22.2.5  2006/02/17 10:41:48  fbrunet
 * Change QueryCriteria parameters
 * Add new windows when editing an entry
 *
 * Revision 1.22.2.4  2006/01/25 15:22:23  fbrunet
 * Improvement of QueryRequest
 * Add new search criteria
 * Add modified status
 *
 * Revision 1.22.2.3  2006/01/24 13:39:49  fbrunet
 * Modification view management
 * Modification LexALP postprocessing
 *
 * Revision 1.22.2.2  2005/12/02 10:04:09  fbrunet
 * Add Pre/Post edition processing
 * Add index reconstruction
 * Add new query request
 * Add fuzzy search
 * Add new contribution administration
 * Add xsl transformation volume
 *
 * Revision 1.22.2.1  2005/10/24 16:29:19  fbrunet
 * Added fuzzy search capabilities.
 * Added possibility to rebuild the index DB tables.
 * Added Pre and post processors that could be defined by the user.
 *
 * Revision 1.22  2005/08/01 16:52:45  mangeot
 * Fixed missing reviewer, reviewdate, validator, validator date when setReviewed and setValdiated where called
 *
 * Revision 1.21  2005/08/01 09:27:09  mangeot
 * Bug fix
 *
 * Revision 1.20  2005/08/01 08:34:03  mangeot
 * Added method getCompleteHeadword for VolumeEntry that concatenates the homograph number and the particule to the headword
 *
 * Revision 1.19  2005/07/30 16:10:17  mangeot
 * Bug fix when headword was edited but not changed in the db
 *
 * Revision 1.18  2005/07/16 12:52:37  mangeot
 * *** empty log message ***
 *
 * Revision 1.17  2005/06/23 09:48:17  mangeot
 * Bug fix in xpath completion and creation-date cdm element
 *
 * Revision 1.16  2005/06/22 15:55:53  mangeot
 * Solved an unresolved prefix bug when the dml prefix was not in the template entry.
 * Now we use the DmlPrefixResolver to solve this issue.
 *
 * Revision 1.15  2005/06/17 17:53:39  mangeot
 * *** empty log message ***
 *
 * Revision 1.14  2005/06/17 16:58:11  mangeot
 * *** empty log message ***
 *
 * Revision 1.13  2005/06/17 16:49:47  mangeot
 * *** empty log message ***
 *
 * Revision 1.12  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.11  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.10.4.5  2005/06/01 08:38:43  mangeot
 * Multi bug correction + added the possibility of disabling data edition
 * via the Admin.po page
 *
 * Revision 1.10.4.4  2005/05/19 17:02:22  mangeot
 * Importing entries without the contribution element
 *
 * Revision 1.10.4.3  2005/04/30 09:11:20  mangeot
 * *** empty log message ***
 *
 * Revision 1.10.4.2  2005/04/29 16:04:32  mangeot
 * *** empty log message ***
 *
 * Revision 1.10.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.10  2005/04/26 10:11:32  mangeot
 * *** empty log message ***
 *
 * Revision 1.9  2005/04/20 10:51:14  mangeot
 * Correction de AddDirectTranslations
 *
 * Revision 1.8  2005/04/19 15:47:06  mangeot
 * Fixed a pb with the id
 *
 * Revision 1.7  2005/04/15 15:03:47  mangeot
 * Fixed a bug in setIdIfNull and deleted the empty button on AdminEntries
 *
 * Revision 1.6  2005/04/15 14:36:01  mangeot
 * Added setIdIfNull
 *
 * Revision 1.5  2005/04/15 13:20:08  mangeot
 * Added setIdIfNull
 *
 * Revision 1.4  2005/04/14 13:08:25  mangeot
 * Deleted all references to findContributionByEntryHandle
 *
 * Revision 1.3  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.2.2.11  2005/03/30 11:17:07  mangeot
 * Modified table contributions: replaced originalhandle by originalid
 * Corrected a few bugs when validating an already existing entry
 *
 * Revision 1.2.2.10  2005/03/29 15:27:09  mangeot
 * Bug fix when trying to create a contribution from an existing entry
 *
 * Revision 1.2.2.9  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.2.2.8  2005/03/14 08:47:11  mangeot
 * MemoryOverflow bug resolved. Occured wuth the XPathContext. It stores its context each time an execute is executed so it is better to create a new XPathContext everytime.
 *
 * Revision 1.2.2.7  2005/01/28 23:01:09  mangeot
 * Fixed bugs in the editor. It seems to work now. More testing needed anyway...
 *
 * Revision 1.2.2.6  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
 *
 * Revision 1.2.2.5  2005/01/27 23:55:13  mangeot
 * *** empty log message ***
 *
 * Revision 1.2.2.4  2005/01/27 19:29:21  mangeot
 * Implemented the HtmlDom cache, it increases speed drastically.
 * Still does not compile after an ant clean
 *
 * Revision 1.2.2.3  2005/01/27 18:09:28  mangeot
 * Simple dictionary lookup is now working for GDEF.
 * Does not compile yet but cvs commit for backup
 *
 * Revision 1.2.2.2  2005/01/27 15:56:21  mangeot
 * Able to load a volume with XPointers, cannot lookup the result yet.
 * Does not compile but commit for backup
 *
 * Revision 1.2.2.1  2005/01/25 13:54:54  mangeot
 * changed the volume volumeEntry and index objects. Does not compile but need a backup...
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * Revision 1.12  2004/10/28 10:38:11  mangeot
 * Fixed some bugs that affected the dictd server
 * Modified some methods in order to display a text entry in the dictd server
 *
 * Revision 1.11  2004/03/11 16:12:23  mangeot
 * *** empty log message ***
 *
 * Revision 1.10  2004/02/10 05:27:13  mangeot
 * The version UIGEN_V2 has been merged with the trunk by MM
 * Be careful because the Volumes and contributions database tables have been modified.
 * You have to drop and rebuild them unless you modify them by hands.
 *
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.dictionary;

import com.lutris.appserver.server.sql.DatabaseManagerException;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.dods.builder.generator.query.DataObjectException;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.data.VolumeEntryDO;
import fr.imag.clips.papillon.data.VolumeEntryQuery;
import fr.imag.clips.papillon.papillon_data.ManageDatabase;
import com.lutris.dods.builder.generator.query.QueryBuilder;

import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Locale;
import java.text.Collator;

/**
 * Represents a Dictionary Entry.
 */
// FIXME: All CDM field access should be realized in a superclass to simplify this class.
public class VolumeEntry
        implements IAnswer, Comparable {

    public final static String ORIGINAL_STATUS = "original";    // FIXME: in use ?
    public final static String NOT_FINISHED_STATUS = "not finished";
    public final static String FINISHED_STATUS = "finished";
    public final static String DRAFT_STATUS = "draft";
    public final static String CLASSIFIED_FINISHED_STATUS = "classified";
    public final static String CLASSIFIED_NOT_FINISHED_STATUS = "classified not finished";

    public final static String REVIEWED_STATUS = "revised";
    public final static String VALIDATED_STATUS = "validated";
    public final static String REPLACED_STATUS = "replaced";
    public final static String DELETED_STATUS = "deleted";
    public final static String MODIFIED_STATUS = "modified";

			protected static String DML_PREFIX = "d";
			
			public final static String authorTag = "author";
			public final static String commentTag = "comment";
			public final static String contributionTag = "contribution";
			public final static String contributionIdAttr = "contribid";
			public final static String creationDateTag = "creation-date";
			public final static String dataTag = "data";
			public final static String dateTag = "date";
			public final static String originalContributionIdAttr = "originalcontribid";
			public final static String groupsTag = "groups";
			public final static String groupTag = "group";
			public final static String historyTag = "history";
			public final static String metadataTag = "metadata";
			public final static String modificationTag = "modification";
			public final static String finitionDateTag = "finition-date";
			public final static String reviewDateTag = "review-date";
			public final static String reviewerTag = "reviewer";
			public final static String statusTag = "status";
			public final static String validationDateTag = "validation-date";
			public final static String validatorTag = "validator";
            public final static String previousContributionTag = "previous-contribution";
            public final static String previousClassifiedFinishedContributionTag = "previous-classified-finished-contribution";
			public final static String previousClassifiedNotFinishedContributionTag = "previous-classified-not-finished-contribution";
			public final static String nextContributionAuthorTag = "next-contribution-author";
			
			public final static ArrayList cdmList = new ArrayList();    	
    
    protected static final String ENTRY_ID_SUFFIX = ".e";
    protected static final String CONTRIBUTION_ID_SUFFIX = ".c";

    // by default, the HTML DOM is not cached
    //public static boolean CACHE_HTMLDOM = false;

    protected org.w3c.dom.Document dom = null;
    // protected org.w3c.dom.Document htmldom = null;
    protected Dictionary theDictionary;
    protected Volume theVolume;

    protected VolumeEntryDO myDO = null;


    private Collator collator = null;

    private Collator getCollator() {
        if (null == collator) {
            try {
                String lang = this.getSourceLanguage();
                if (null != lang && ! "".equals(lang)) {
                    collator = Collator.getInstance(new Locale(lang));
                } else {
                    collator = Collator.getInstance();
                }
            } catch (PapillonBusinessException e) {
                collator = Collator.getInstance();
            }
        }
        return collator;
    }

    /**
     * The public constructor.
     * Should find a better method instead of these if elsif elsif
     * How to do it ?
     */

    public VolumeEntry(Dictionary newDict, Volume newVolume)
            throws PapillonBusinessException {
        try {
            this.myDO = VolumeEntryDO.createVirgin(newVolume.getDbname(), CurrentDBTransaction.get());
            this.setVolume(newVolume);
            this.setDictionary(newDict);
        } catch (DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error creating VolumeEntry", ex);
        } catch (ObjectIdException ex) {
            throw new PapillonBusinessException("Error creating VolumeEntry", ex);
        }
    }


    /**
     * The protected constructor
     *
     * @param newDict
     * @param newVolume
     * @param theVolumeEntryDO
     */
    protected VolumeEntry(Dictionary newDict, Volume newVolume, VolumeEntryDO theVolumeEntryDO)
            throws PapillonBusinessException {
        try {
            this.myDO = theVolumeEntryDO;
            //this.dom = Utility.deSerializeDocument(theVolumeEntryDO.getDom());
            this.dom = XMLServices.buildDOMTree(this.myDO.getXmlCode());
            // this.htmldom = Utility.deSerializeDocument(theVolumeEntryDO.getHtmldom());
            this.setVolume(newVolume);
            this.setDictionary(newDict);

            //String xc1 = XMLServices.xmlCode(this.dom);
            //String xc2 = XMLServices.xmlCodePrettyPrinted(this.dom);

//            if (xc1.equals(xc2)) {
//                System.out.println("ok");
//            } else {
            //System.out.println("===================================================");
            //System.out.println(xc1);
            //System.out.println("---------------------------------------------------");
            //System.out.println(xc2);
            //System.out.println("===================================================");
//            }

        } catch (DataObjectException ex) {
            throw new PapillonBusinessException("Error creating VolumeEntry", ex);
        }
    }

    public boolean isEmpty() {
        return (this.myDO == null);
    }


    /**
     * Gets the object id for the Dictionary
     *
     * @return the object id.
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    public String getHandle()
            throws PapillonBusinessException {
        try {
            return this.myDO.getHandle();
        } catch (DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error getting Volume's handle", ex);
        }
    }


    
    /**
     * Gets the subject of the Dictionary
     *
     * @return the subject.
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    public String getTableName() {
        return this.myDO.getTableName();
    }

    /**
     * Gets the headword of the volumeEntry
     *
     * @return the subject.
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    // FIXME: Should not be stored in a column, but should be extracted from the xml using CDM...
    // FIXME: Moreover, an entry may have more than one headword.
    public String getHeadword()
            throws PapillonBusinessException {
        String headword = null;
        try {
            headword = this.myDO.getHeadword();
        } catch (DataObjectException ex) {
            throw new PapillonBusinessException("Error getting volumeEntry's headword", ex);
        }
        if (headword == null || headword.equals("")) {
            headword = getCdmHeadword();
        }
        return headword;
    }

    /**
     * Gets the headword, homograph number and particule of the volumeEntry
     *
     * @return the subject.
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    public String getCompleteHeadword()
            throws PapillonBusinessException {
        String headword = this.getHeadword();
        String particule = this.getParticule();
        if (particule != null && !particule.equals("")) {
            headword = particule + " " + headword;
        }
        String homograph = this.getHomographNumber();
        if (homograph != null && !homograph.equals("")) {
            headword = headword + " " + homograph;
        }
        return headword;
    }


    /**
     * Gets the headword of the volumeEntry extracted from the XML
     *
     * @return the subject.
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    public String getCdmHeadword()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_headword, this.getSourceLanguage());
    }

    public void setHeadword()
            throws PapillonBusinessException {
        String word = IndexEntry.getCdmString(this, Volume.CDM_headword, this.getSourceLanguage());
        if (word != null) {
            word = word.trim();
        }
        try {
            myDO.setHeadword(word);
        } catch (DataObjectException ex) {
            throw new PapillonBusinessException("Error setting volumeEntry's headword", ex);
        }
    }

    public void setHeadword(String word)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_headwordElement, word, this.getSourceLanguage());
        try {
            myDO.setHeadword(word);
        } catch (DataObjectException ex) {
            throw new PapillonBusinessException("Error setting volumeEntry's headword", ex);
        }
    }

    /**
     * Gets the homograph number of the volumeEntry
     *
     * @return the homograph number as a String.
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    public String getHomographNumber()
            throws PapillonBusinessException {
        String hn = "";
        hn = IndexEntry.getCdmString(this, Volume.CDM_homographNumber, this.getSourceLanguage());
        return hn;
    }

    /**
     * Gets the dom of the entry
     *
     * @return the dom of the entry.
     */
    public org.w3c.dom.Document getDom() {
        return this.dom;
    }

    /**
     * Sets the dom of the entry
     */
    public void setDom(org.w3c.dom.Document myDoc) {
        this.dom = myDoc;
    }

//    /**
//     * Gets the html dom of the entry
//     *
//     * @return the html dom of the entry.
//     */
//    public org.w3c.dom.Document getHtmlDom() {
//        if (CACHE_HTMLDOM) {
//            return this.htmldom;
//        } else {
//            return null;
//        }
//    }
//
//    /**
//     * Sets the html dom of the entry
//     *
//     */
//    public void setHtmlDom(org.w3c.dom.Document myDoc) {
//        if (CACHE_HTMLDOM) {
//            this.htmldom = myDoc;
//        } else {
//            this.htmldom = null;
//        }
//    }


    /**
     * Gets the xmlCode of the volumeEntry
     *
     * @return the subject.
     * @throws PapillonBusinessException if an error occurs
     *                                   retrieving data (usually due to an underlying data layer
     *                                   error).
     */
    public String getXmlCode()
            throws PapillonBusinessException {
        try {
            return this.myDO.getXmlCode();
        } catch (DataObjectException ex) {
            throw new PapillonBusinessException("Error getting volume's xml code", ex);
        }
    }

    public void setXmlCode(String code)
            throws PapillonBusinessException {
        try {
            myDO.setXmlCode(code);
        } catch (DataObjectException ex) {
            throw new PapillonBusinessException("Error setting entry xmlcode", ex);
        }
    }

    /* Methods for Entry ID */

    /**
     * setEntryId sets the entry id into the XML code of the entry.
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setId()
            throws PapillonBusinessException {
        this.setEntryIdIfNull();
    }

    public void setEntryId()
            throws PapillonBusinessException {
        this.setEntryId(this.createNewId() + ENTRY_ID_SUFFIX);
    }

    protected void setEntryId(String newId)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_entryIdElement, newId);
    }


    /**
     * setEntryIdIfNull sets the entry id into the XML code of the entry
     * if there is no previous existing entry id
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    protected void setEntryIdIfNull()
            throws PapillonBusinessException {
        if (this.getEntryId() == null || this.getEntryId().equals("")) {
            this.setEntryId(this.createNewId() + ENTRY_ID_SUFFIX);
        }
    }

    /**
     * getEntryId gets the entry id into the XML code of the entry.
     *
     * @return the entry id as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String getId()
            throws PapillonBusinessException {
        return getEntryId();
    }

    public String getEntryId()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_entryId);
    }

    /* methods for the contribution id */

    /**
     * setEntryId sets the entry id into the XML code of the entry.
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setContributionId()
            throws PapillonBusinessException {
        this.setContributionId(this.createNewId() + CONTRIBUTION_ID_SUFFIX);
    }

    /**
     * setContributionIdIfNull sets the entry id into the XML code of the entry
     * if there is no previous existing entry id
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    protected void setContributionIdIfNull()
            throws PapillonBusinessException {
        if (this.getContributionId() == null || this.getContributionId().equals("")) {
            this.setContributionId(this.createNewId() + CONTRIBUTION_ID_SUFFIX);
        }
    }

    // FIXME: public method, use to commute id whenever one save a draft contribution
    public void setContributionId(String newId)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_contributionId, newId);
    }

    /**
     * getContributionId gets the entry id into the XML code of the entry.
     *
     * @return the contribution id as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String getContributionId()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_contributionId);
    }

    /**
     * setOriginalContributionId sets the entry id into the XML code of the entry.
     *
     * @param contribid as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setOriginalContributionId(String newId)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_originalContributionId, newId);
    }
    
    protected void setOriginalContributionIdIfNull()
            throws PapillonBusinessException {
                if (this.getOriginalContributionId() == null || this.getOriginalContributionId().equals("")) {
                    this.setOriginalContributionId(this.createNewId() + CONTRIBUTION_ID_SUFFIX);
                }
            }

    /**
     * getOriginalContributionId gets the original contribution id into the XML code of the entry.
     *
     * @return the contribution id as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String getOriginalContributionId()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_originalContributionId);
    }

/* methods for axi-reflinks */

    /**
     * returns ids of the lexies that are pointed by this by an axi-reflink, for lang
     */
    public String[] getReferencedLexieIds(String lang)
            throws PapillonBusinessException {
    	// utilise table link
    	// verifier ou il est utilise
    	// si c'est plus pratique, renvoyer une collection au lieu d'un tableau de string
        return IndexEntry.getCdmStrings(this, Volume.CDM_axiReflexie, lang);
    }

    /**
     * returns ids of the lexies that are pointed by this by an axi-reflink, for lang
     */
    public String[] getTranslationsLexieIds(String lang)
            throws PapillonBusinessException {
    	// utilise table link
        return IndexEntry.getCdmStrings(this, Volume.CDM_translationReflexie, lang);
    }

    /**
     * returns ids of the axies that are pointed by this by an axi-reflink
     */
    public String[] getReferencedAxieIds()
            throws PapillonBusinessException {
    	// utilise table link

        return IndexEntry.getCdmStrings(this, Volume.CDM_axiRefaxie);
    }

    /* other methods */

    /**
     * getAuthor gets the entry author into the XML code of the entry.
     *
     * @return the author as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String getAuthor()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_contributionAuthor);
    }


    /**
     * setAuthor sets the entry author into the XML code of the entry.
     *
     * @param the author as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setAuthor(String author)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_contributionAuthorElement, author);
    }

    public void setAuthor()
            throws PapillonBusinessException {
        String author = IndexEntry.getCdmString(this, Volume.CDM_contributionAuthor);
        if (author == null || author.equals("")) {
            setAuthor("automatic");
        }
    }

    /**
     * getCreationDate gets the entry date into the XML code of the entry.
     *
     * @return the creation date as a java.util.Date
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public java.util.Date getCreationDate()
            throws PapillonBusinessException {
        java.util.Date resDate = null;
        String dateString = IndexEntry.getCdmString(this, Volume.CDM_contributionCreationDate);
        if (dateString != null && !dateString.equals("")) {
            try {
                resDate = Utility.PapillonCDMDateFormat.parse(dateString);
            } catch (java.text.ParseException ex) {
                throw new PapillonBusinessException("Error parsing a date String", ex);
            }
        }
        return resDate;
    }

    /**
     * setCreationDate sets the entry review date into the XML code of the entry.
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setCreationDate()
            throws PapillonBusinessException {
        String dateString = IndexEntry.getCdmString(this, Volume.CDM_contributionCreationDate);
        if (dateString == null || dateString.equals("")) {
            setCreationDate(Utility.PapillonCDMDateFormat.format(new java.util.Date()));
        }
    }

    public void setCreationDate(java.util.Date myDate)
            throws PapillonBusinessException {
        if (myDate != null) {
            setCreationDate(Utility.PapillonCDMDateFormat.format(myDate));
        }
    }

    protected void setCreationDate(String date)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_contributionCreationDateElement, date);
    }

    /**
     * setFinitionDate sets the entry finition date into the XML code of the entry.
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setFinitionDate()
            throws PapillonBusinessException {
        String dateString = IndexEntry.getCdmString(this, Volume.CDM_contributionFinitionDate);
        if (dateString == null || dateString.equals("")) {
            setFinitionDate(Utility.PapillonCDMDateFormat.format(new java.util.Date()));
        }
    }

    public void setFinitionDate(java.util.Date myDate)
            throws PapillonBusinessException {
        if (myDate != null) {
            setFinitionDate(Utility.PapillonCDMDateFormat.format(myDate));
        }
    }

    protected void setFinitionDate(String date)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_contributionFinitionDateElement, date);
    }

    /**
     * getFinitionDate gets the entry finition date from the XML code of the entry.
     *
     * @return the review date as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public java.util.Date getFinitionDate()
            throws PapillonBusinessException {
        java.util.Date resDate = null;
        String dateString = IndexEntry.getCdmString(this, Volume.CDM_contributionFinitionDate);
        if (dateString != null && !dateString.equals("")) {
            try {
                resDate = Utility.PapillonCDMDateFormat.parse(dateString);
            } catch (java.text.ParseException ex) {
                throw new PapillonBusinessException("Error parsing a date String", ex);
            }
        }
        return resDate;
    }

    /**
     * setReviewDate sets the entry review date into the XML code of the entry.
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setReviewDate()
            throws PapillonBusinessException {
        setReviewDate(Utility.PapillonCDMDateFormat.format(new java.util.Date()));
    }

    public void setReviewDate(java.util.Date myDate)
            throws PapillonBusinessException {
        if (myDate != null) {
            setReviewDate(Utility.PapillonCDMDateFormat.format(myDate));
        }
    }

    protected void setReviewDate(String date)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_contributionReviewDateElement, date);
    }


    /**
     * getReviewDate gets the entry review date from the XML code of the entry.
     *
     * @return the review date as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public java.util.Date getReviewDate()
            throws PapillonBusinessException {
        java.util.Date resDate = null;
        String dateString = IndexEntry.getCdmString(this, Volume.CDM_contributionReviewDate);
        if (dateString != null && !dateString.equals("")) {
            try {
                resDate = Utility.PapillonCDMDateFormat.parse(dateString);
            } catch (java.text.ParseException ex) {
                throw new PapillonBusinessException("Error parsing a date String", ex);
            }
        }
        return resDate;
    }

    /**
     * getReviewer gets the entry author into the XML code of the entry.
     *
     * @return the reviewer as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String getReviewer()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_contributionReviewer);
    }

    /**
     * setReviewer sets the entry reviewer into the XML code of the entry.
     *
     * @param the reviewer as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setReviewer(String reviewer)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_contributionReviewerElement, reviewer);
    }

    /**
     * getValidationDate gets the entry review date from the XML code of the entry.
     *
     * @return the validation date as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public java.util.Date getValidationDate()
            throws PapillonBusinessException {
        java.util.Date resDate = null;
        String dateString = IndexEntry.getCdmString(this, Volume.CDM_contributionValidationDate);
        if (dateString != null && !dateString.equals("")) {
            try {
                resDate = Utility.PapillonCDMDateFormat.parse(dateString);
            } catch (java.text.ParseException ex) {
                throw new PapillonBusinessException("Error parsing a date String", ex);
            }
        }
        return resDate;
    }

    /**
     * setValidationDate sets the entry review date into the XML code of the entry.
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setValidationDate()
            throws PapillonBusinessException {
        setValidationDate(Utility.PapillonCDMDateFormat.format(new java.util.Date()));
    }

    public void setValidationDate(java.util.Date myDate)
            throws PapillonBusinessException {
        if (myDate != null) {
            setValidationDate(Utility.PapillonCDMDateFormat.format(myDate));
        }
    }

    protected void setValidationDate(String date)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_contributionValidationDateElement, date);
    }

    /**
     * getReviewer gets the entry author into the XML code of the entry.
     *
     * @return the reviewer as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String getValidator()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_contributionValidator);
    }

    /**
     * setValidator sets the entry reviewer into the XML code of the entry.
     *
     * @param the reviewer as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setValidator(String validator)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_contributionValidatorElement, validator);
    }

    /**
     * getStatus gets the entry status into the XML code of the entry.
     *
     * @return the status as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String getStatus()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_contributionStatus);
    }

    /**
     * setStatus sets the entry id into the XML code of the entry.
     *
     * @param the status as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setStatus(String status)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_contributionStatusElement, status);
    }

    public void setStatusIfNotNull(String defaultStatus)
            throws PapillonBusinessException {
        String status = IndexEntry.getCdmString(this, Volume.CDM_contributionStatus);
        if (status == null || status.equals("")) {
            setStatus(defaultStatus);
        }
    }

    /**
     * setFinished sets the entry status to finished
     *
     * @param the author as a User
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setFinished(fr.imag.clips.papillon.business.user.User myUser)
            throws PapillonBusinessException {
        if (null != this.getStatus() && this.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {
            this.setFinitionDate();
            this.setModification(myUser.getLogin(), VolumeEntry.FINISHED_STATUS);
            this.setStatus(VolumeEntry.FINISHED_STATUS);
            this.save();
            ContributionLog myContribLog = ContributionsFactory.newContributionLog(myUser, this);
            myContribLog.save();
        }
    }

    /**
     * setReviewed sets the entry status to reviewed
     *
     * @param the reviewer as a User
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setReviewed(fr.imag.clips.papillon.business.user.User myUser)
            throws PapillonBusinessException {
        if (null != this.getStatus() && this.getStatus().equals(VolumeEntry.FINISHED_STATUS)) {
            this.setReviewer(myUser.getLogin());
            this.setReviewDate();
            this.setModification(myUser.getLogin(), VolumeEntry.REVIEWED_STATUS);
            this.setStatus(VolumeEntry.REVIEWED_STATUS);
            this.save();
            ContributionLog myContribLog = ContributionsFactory.newContributionLog(myUser, this);
            myContribLog.save();
        }
    }

    /**
     * setValidated sets the entry status to validated
     *
     * @param the reviewer as a User
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setValidated(fr.imag.clips.papillon.business.user.User myUser)
            throws PapillonBusinessException {
        if (null != this.getStatus() && this.getStatus().equals(VolumeEntry.REVIEWED_STATUS)) {
            this.setValidator(myUser.getLogin());
            this.setValidationDate();
            this.setModification(myUser.getLogin(), VolumeEntry.VALIDATED_STATUS);
            this.setStatus(VolumeEntry.VALIDATED_STATUS);
            this.save();
            String origId = this.getOriginalContributionId();
            if (origId != null && !origId.equals("")) {
                VolumeEntry myEntry = VolumeEntriesFactory.findEntryByEntryId(this.getVolumeName(), origId);
                if (myEntry != null && !myEntry.isEmpty()) {
                    myEntry.setReplaced(myUser);
                }
            }
            ContributionLog myContribLog = ContributionsFactory.newContributionLog(myUser, this);
            myContribLog.save();
        }
    }

    /**
     * setReplaced sets the entry status to replaced
     *
     * @param the validator as a User
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setReplaced(fr.imag.clips.papillon.business.user.User myUser)
            throws PapillonBusinessException {
        if (null != this.getStatus() && this.getStatus().equals(VolumeEntry.VALIDATED_STATUS)) {
            this.setModification(myUser.getLogin(), VolumeEntry.REPLACED_STATUS);
            this.setStatus(VolumeEntry.REPLACED_STATUS);
            this.save();
            //				ContributionLog myContribLog = ContributionsFactory.newContributionLog(myUser, this);
            //				myContribLog.save();
        }
    }


    /**
     * setDeleted sets the entry status to replaced
     *
     * @param the validator as a User
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setDeleted(fr.imag.clips.papillon.business.user.User myUser)
            throws PapillonBusinessException {
        if (null != this.getStatus()) {
            this.setModification(myUser.getLogin(), VolumeEntry.DELETED_STATUS);
            this.setStatus(VolumeEntry.DELETED_STATUS);
            ContributionLog myContribLog = ContributionsFactory.newContributionLog(myUser, this);
            myContribLog.save();
            this.save();
        }
    }

    /**
     * getStatus gets the entry status into the XML code of the entry.
     *
     * @return the status as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String[] getGroups()
            throws PapillonBusinessException {
        return IndexEntry.getCdmStrings(this, Volume.CDM_contributionGroup);
    }

    /**
     * setStatus sets the entry id into the XML code of the entry.
     *
     * @param the status as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public void setGroups(String[] groups)
            throws PapillonBusinessException {
        if (groups != null && groups.length > 0) {
            org.w3c.dom.Document myDocument = this.getDom();
            org.w3c.dom.Node groupsNode = IndexEntry.getCdmElement(this, Volume.CDM_contributionGroups);
            if (groupsNode != null) {
                while (groupsNode.hasChildNodes()) {
                    groupsNode.removeChild(groupsNode.getFirstChild());
                }
                for (int i = 0; i < groups.length; i++) {
                    String group = groups[i];
                    if (group != null && !group.equals("")) {
                        org.w3c.dom.Element myGroup = myDocument.createElement(
                                this.getVolume().getCdmContributionGroup());
                        Utility.setText(myGroup, group);
                        groupsNode.appendChild(myGroup);
                    }
                }
            }
        }
    }

    /**
     * getPos gets the part-of-speech of the entry.
     *
     * @return the part-of-speech as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String getPos()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_pos, this.getSourceLanguage());
    }

    protected void setDictionary(Dictionary dict) {
        theDictionary = dict;
    }

    public Dictionary getDictionary() {
        return theDictionary;
    }

    public Volume getVolume() {
        return theVolume;
    }

    protected void setVolume(Volume volume) {
        theVolume = volume;
    }

    public String getDictionaryName()
            throws PapillonBusinessException {
        return theDictionary.getName();
    }

    public String getDictionaryFullName()
            throws PapillonBusinessException {
        return theDictionary.getFullName();
    }

    public String getVolumeName()
            throws PapillonBusinessException {
        return theVolume.getName();
    }

    public String getSourceLanguage()
            throws PapillonBusinessException {
        return theVolume.getSourceLanguage();
    }

    public int getType() {
        return IAnswer.LocalEntry;
    }

    public void setPreviousContributionId(String contributionId)
      throws PapillonBusinessException {
      IndexEntry.setCdmElement(this, Volume.CDM_previousContributionElement, contributionId);
    }
            
    public String getPreviousContributionId()
        throws PapillonBusinessException {
            return IndexEntry.getCdmString(this, Volume.CDM_previousContribution);
    }
            
    public Collection getClassifiedFinishedContributionIdCollection()
            throws PapillonBusinessException {
        String[] list = IndexEntry.getCdmStrings(this, Volume.CDM_previousClassifiedFinishedContribution);
        ArrayList arrayList = new ArrayList();
        if (list != null) {
            for (int i = 0; i < list.length; i++) {
                String tmp = (String) list[i];
                if ((tmp != null) && !tmp.equals("")) {
                    arrayList.add(tmp);
                }
            }
        }
        return arrayList;
    }

    public void initClassifiedFinishedContribution()
            throws PapillonBusinessException {
        IndexEntry.initCdmElement(this, Volume.CDM_previousClassifiedFinishedContributionElement);
    }

    public void addClassifiedFinishedContribution(VolumeEntry contribution)
            throws PapillonBusinessException {
        addClassifiedFinishedContribution(contribution.getContributionId());
    }

    public void addClassifiedFinishedContribution(String contributionId)
            throws PapillonBusinessException {
        IndexEntry.addCdmElement(this, Volume.CDM_previousClassifiedFinishedContributionElement, contributionId);
    }

    public String getClassifiedNotFinishedContributionId()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_previousClassifiedNotFinishedContribution);
    }

    public void setClassifiedNotFinishedContribution(VolumeEntry contribution)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_previousClassifiedNotFinishedContributionElement,
                contribution.getContributionId());
    }

    public void setClassifiedNotFinishedContribution(String contributionId)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_previousClassifiedNotFinishedContributionElement, contributionId);
    }

    public void initClassifiedNotFinishedContribution()
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_previousClassifiedNotFinishedContributionElement, "");
    }

    public String getNextContributionAuthor()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_nextContributionAuthor);
    }

    public void setNextContributionAuthor(String author)
            throws PapillonBusinessException {
        IndexEntry.setCdmElement(this, Volume.CDM_nextContributionAuthorElement, author);
    }


    protected String createNewId(String headword)
            throws PapillonBusinessException {
        if (headword != null) {
            headword = headword.trim();
        }
        String entryId = this.getSourceLanguage() + "." + headword + "." + this.getHandle();
        entryId = entryId.replace(' ', '_');
        entryId = entryId.replace('\'', '_');
        //System.out.println("getSourceLanguage = " + this.getSourceLanguage() + " headword = " + headword + " getHandle = " + this.getHandle());
        return Utility.encodeXMLEntities(entryId);
    }

    protected String createNewId()
            throws PapillonBusinessException {
        return createNewId(IndexEntry.getCdmString(this, Volume.CDM_headword, this.getSourceLanguage()));
    }

    /**
     * getAuthor gets the modification entry author into the XML code of the entry.
     *
     * @return the author as a String
     * @throws PapillonBusinessException if an error occurs
     *                                   getting data (usually due to an underlying data layer
     *                                   error).
     */
    public String getModificationAuthor()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_modificationAuthor);
    }

    public String getModificationDate()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_modificationDate);
    }

    public String getModificationComment()
            throws PapillonBusinessException {
        return IndexEntry.getCdmString(this, Volume.CDM_modificationComment);
    }


    public void setModification(String author, String comment)
            throws PapillonBusinessException {
        setModification(author, new java.util.Date(), comment);
    }

    private void setModification(String author, java.util.Date date, String comment)
            throws PapillonBusinessException {

        /*
		Volume myVolume = this.getVolume();
		Document myDocument = this.getDom();
		org.w3c.dom.Node myEntry = IndexEntry.getCdmElement(this, Volume.CDM_entry);
		*/
        //PapillonLogger.writeDebugMsg("VolumeEntry.setModification: " + author + " comment: " + comment);

        // ps pour Mathieu : TU TOUCHES PAS A CA :)
        IndexEntry.setCdmElement(this, Volume.CDM_modificationAuthorElement, author);
        IndexEntry.setCdmElement(this, Volume.CDM_modificationDateElement, Utility.PapillonCDMDateFormat.format(date));
        IndexEntry.setCdmElement(this, Volume.CDM_modificationCommentElement, comment);

        /*
          //
          org.w3c.dom.Node myHistory = IndexEntry.getCdmElement(this, Volume.CDM_history);
          if (myHistory == null) {
              fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("setModification: myHistory null");
              myHistory = myDocument.createElement(myVolume.getCdmHistory());
              myEntry.appendChild(myHistory);
          }

          //
          org.w3c.dom.Element myModification = myDocument.createElement(myVolume.getCdmModification());
          org.w3c.dom.Element myAuthor = myDocument.createElement(myVolume.getCdmModificationAuthorElement());
          org.w3c.dom.Element myComment = myDocument.createElement(myVolume.getCdmModificationCommentElement());
          org.w3c.dom.Element myDate = myDocument.createElement(myVolume.getCdmModificationDateElement());
          Utility.setText(myAuthor,author);
          Utility.setText(myDate,Utility.PapillonCDMDateFormat.format(date));
          Utility.setText(myComment,comment);
          myModification.appendChild(myAuthor);
          myModification.appendChild(myComment);
          myModification.appendChild(myDate);
          myHistory.appendChild(myModification);
        */

    }

    /**
     * getDefinition retrieves the definition of the entry
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
    public String getDefinition()
            throws PapillonBusinessException {
        return "   " + IndexEntry.getCdmString(this, Volume.CDM_definition);
    }

    /**
     * replaceData replaces all the data except the handle
     *
     * @exception PapillonBusinessException if an error occurs
     *   replacing data (usually due to an underlying data layer
     *   error).
     *
    public boolean replaceData(VolumeEntry otherEntry) throws PapillonBusinessException {
    boolean res = false;
    if (getVolumeName().equals(otherEntry.getVolumeName()) &&
    getTableName().equals(otherEntry.getTableName())) {
    setHeadword(otherEntry.getHeadword());
    // Do we have to copy the id? Quid of the existing links?
    // setId(otherEntry.getId());
    String oldId = this.getEntryId();
    setDom((org.w3c.dom.Document) otherEntry.getDom().cloneNode(true));
    setEntryId(oldId);
    setHtmlDom(otherEntry.getHtmlDom());
    res= true;
    }
    return res;
    }
     */

			
			public static String getContributionHeader(String templateEntry) 
			throws fr.imag.clips.papillon.business.PapillonBusinessException {
			org.w3c.dom.Document myDoc = XMLServices.buildDOMTree(templateEntry);
			String dmlPrefixOrig = VolumesFactory.getDmlPrefix(myDoc.getDocumentElement());
			String dmlPrefix = dmlPrefixOrig;
			String dmlXmlns = VolumesFactory.XMLNAMESPACE;
			if (dmlPrefixOrig!= null && !dmlPrefixOrig.equals("")) {
				dmlPrefix = dmlPrefixOrig + ":";
				dmlXmlns += ":" +  dmlPrefixOrig;
			}
			
			String result = "<" + dmlPrefix + contributionTag +
			" " + dmlXmlns + "=\"" + VolumesFactory.DML_URI +"\"" + 
				
            " " + dmlPrefix + contributionIdAttr + "=\"\" " + dmlPrefix + originalContributionIdAttr + "=\"\">" +
            "<" + dmlPrefix + metadataTag + ">" +
            "<" + dmlPrefix + authorTag + "/>" +
            "<" + dmlPrefix + groupsTag + "/>" +
            "<" + dmlPrefix + creationDateTag + "/>" +
            "<" + dmlPrefix + finitionDateTag + "/>" +
            "<" + dmlPrefix + reviewDateTag + "/>" +
            "<" + dmlPrefix + reviewerTag + "/>" +
            "<" + dmlPrefix + validationDateTag + "/>" +
            "<" + dmlPrefix + validatorTag + "/>" +
            "<" + dmlPrefix + statusTag + "/>" +
            "<" + dmlPrefix + historyTag + ">" +
            "<" + dmlPrefix + modificationTag + ">" +
            "<" + dmlPrefix + authorTag + "/>" +
            "<" + dmlPrefix + dateTag + "/>" +
            "<" + dmlPrefix + commentTag + "/>" +
            "</" + dmlPrefix + modificationTag + ">" +
            "</" + dmlPrefix + historyTag + ">" +
            "<" + dmlPrefix + previousContributionTag + "/>" +
            "<" + dmlPrefix + previousClassifiedFinishedContributionTag + "/>" +
 //           "<" + dmlPrefix + previousClassifiedNotFinishedContributionTag + "/>" +
 //           "<" + dmlPrefix + nextContributionAuthorTag + "/>" +
            "</" + dmlPrefix + metadataTag + ">" +
            "<" + dmlPrefix + dataTag + ">";
			return result;
		} 			
			
			public static String getContributionFooter(String templateEntry) 
			 throws fr.imag.clips.papillon.business.PapillonBusinessException {
				org.w3c.dom.Document myDoc = XMLServices.buildDOMTree(templateEntry);
				String dmlPrefix = VolumesFactory.getDmlPrefix(myDoc.getDocumentElement());
				
				if (dmlPrefix!= null && !dmlPrefix.equals("")) {
					dmlPrefix += ":";
				}
				return "</" + dmlPrefix + dataTag + ">" + "</" + dmlPrefix + contributionTag + ">";
			}
			
			
    /**
     * getParticule
     * <p/>
     * function specific for the GDEF dictionary
     * retrives the value of the <particule> tag.
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   replacing data (usually due to an underlying data layer
     *                                   error).
     */
	public String getParticule()
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return IndexEntry.getCdmString(this, Volume.CDM_gdefEstParticule, this.getSourceLanguage());
	}

	public String getCdmString(String cdm_element)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			return IndexEntry.getCdmString(this, cdm_element, this.getSourceLanguage());
	}
			
	public Volume getAxieVolume() 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
		Volume resVolume = null;
		java.util.Collection axiVolCol = VolumesFactory.getVolumesArray(this.getDictionary().getName(),"axi",null);
		if (axiVolCol.iterator().hasNext()) {
			resVolume = (Volume) axiVolCol.iterator().next();
		}
		return resVolume;
	}

			/**
     * Saves the volume entry into the database.
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   deleting data (usually due to an underlying data layer
     *                                   error).
     */
    public boolean save() throws PapillonBusinessException {
        return this.save(true);
    }

    public boolean save(boolean index)
            throws PapillonBusinessException {
        boolean res = true;
        //PapillonLogger.writeDebugMsg("VolumeEntry.save()");
       try {
            // Reset caches
            VolumeEntriesFactory.resetCountCache(this.getVolume().getName());

            // Delete index
            IndexFactory.deleteIndexForEntryId(this.getVolume().getIndexDbname(), this.getHandle());
            //
            this.setEntryIdIfNull();
           this.setContributionIdIfNull();
           this.setOriginalContributionIdIfNull();
            // New index
            if (index) {
				res = IndexEntry.indexEntry(this);
			}
            //
            if (null != this.dom) {
                // If this.dom is null, this means that the dom object has not be accessed nor modified...
                this.myDO.setXmlCode(XMLServices.xmlCode(this.dom));
            }

            this.myDO.save();
        } catch (Exception ex) {
            throw new PapillonBusinessException("Error saving volumeEntry", ex);
        }

        //
        return res;
    }

    /**
     * Deletes the volume entry from the database.
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   deleting data (usually due to an underlying data layer
     *                                   error).
     */
    public void delete()
            throws PapillonBusinessException {
        Volume myVolume = this.getVolume();
        if (myVolume == null || myVolume.isEmpty()) {
            //myVolume = VolumesFactory.findVolumeByDbname(this.getTableName());
            myVolume = VolumesFactory.getVolumeByName(this.getVolumeName());
        }
        if (myVolume != null && !myVolume.isEmpty()) {
            // reset caches
            VolumeEntriesFactory.resetCountCache(myVolume.getName());
            this.delete(myVolume.getIndexDbname());
        } else {
            try {
                this.myDO.commit();
            } catch (Exception ex) {
                throw new PapillonBusinessException("Error saving volumeEntry", ex);
            }
            throw new PapillonBusinessException("Error saving volumeEntry, there is no volume so no index entered!");
        }
    }

    public void delete(String indexDbname)
            throws PapillonBusinessException {
        try {
            IndexFactory.deleteIndexForEntryId(indexDbname, this.getHandle());
            this.myDO.delete();
        } catch (Exception ex) {
            throw new PapillonBusinessException("Error deleting VolumeEntry", ex);
        }
    }


    /**
     * String element ...
     *
     * @throws PapillonBusinessException if an error occurs
     *                                   deleting data (usually due to an underlying data layer
     *                                   error).
     */
    public NodeList getNodes(String xpathString)
            throws PapillonBusinessException {
        try {
            Element myRoot = getDom().getDocumentElement();

            return org.apache.xpath.XPathAPI.selectNodeList(getDom(), xpathString);
        } catch (javax.xml.transform.TransformerException e) {
            throw new PapillonBusinessException("javax.xml.transform.TransformerException: ", e);
        }
    }


    /**
     * Undo the entry with the last classified finished contribution.
     *
     * @throws PapillonBusinessException
     */
    // FIXME: call undoMerge ????
    public VolumeEntry undoFinish()
            throws PapillonBusinessException {
        VolumeEntry previousVolumeEntry = null;

        //
        Collection previousFContributionIdCollection = getClassifiedFinishedContributionIdCollection();
        for (Iterator iter = previousFContributionIdCollection.iterator(); iter.hasNext();) {
            String contributionId = (String) iter.next();

            //
            VolumeEntry volumeEntry = VolumeEntriesFactory.findEntryByContributionId(getVolumeName(), contributionId);
            if (volumeEntry.getStatus().equals(VolumeEntry.CLASSIFIED_FINISHED_STATUS)) {
                previousVolumeEntry = volumeEntry;
                volumeEntry.setStatus(VolumeEntry.NOT_FINISHED_STATUS);
                volumeEntry.save();

                //
                Collection contributionIdCollection = volumeEntry.getClassifiedFinishedContributionIdCollection();
                for (Iterator iter2 = contributionIdCollection.iterator(); iter2.hasNext();) {
                    String contributionId2 = (String) iter2.next();

                    //
                    VolumeEntry volumeEntry2 = VolumeEntriesFactory.findEntryByContributionId(getVolumeName(),
                            contributionId2);
                    if (volumeEntry2.getStatus().equals(VolumeEntry.CLASSIFIED_FINISHED_STATUS)) {
                        PapillonLogger.writeDebugMsg("undo finish: setStatus: " + VolumeEntry.MODIFIED_STATUS);
                        volumeEntry2.setStatus(VolumeEntry.MODIFIED_STATUS);
                        volumeEntry2.save();
                    }
                }

            } else if (volumeEntry.getStatus().equals(VolumeEntry.DELETED_STATUS)) { // FIXME: specific to the merge !!
                volumeEntry.setStatus(VolumeEntry.FINISHED_STATUS);
                volumeEntry.save();

            } else {
                throw new PapillonBusinessException("Error in undoFinish()");
            }
        }

        // Delete contribution
        delete();

        //
        return previousVolumeEntry;
    }

    /**
     * Undo the entry with the last classified not finished contribution.
     *
     * @throws PapillonBusinessException
     */
    public VolumeEntry undoNotFinish()
            throws PapillonBusinessException {
        VolumeEntry previousVolumeEntry = null;

        //
        String previousNFContributionId = getClassifiedNotFinishedContributionId();
        if ((previousNFContributionId != null) && (!previousNFContributionId.equals(""))) {

            //
            previousVolumeEntry = VolumeEntriesFactory.findEntryByContributionId(getVolumeName(),
                    previousNFContributionId);
            previousVolumeEntry.setStatus(VolumeEntry.NOT_FINISHED_STATUS);
            previousVolumeEntry.save();

        } else {
            Collection previousFContributionIdCollection = getClassifiedFinishedContributionIdCollection();
            for (Iterator iter = previousFContributionIdCollection.iterator(); iter.hasNext();) {
                String contributionId = (String) iter.next();

                //
                previousVolumeEntry = VolumeEntriesFactory.findEntryByContributionId(getVolumeName(), contributionId);
                if (previousVolumeEntry.getStatus().equals(VolumeEntry.MODIFIED_STATUS)) {
                    previousVolumeEntry.setStatus(VolumeEntry.FINISHED_STATUS);
                    previousVolumeEntry.save();
                }
            }

        }

        // Delete contribution
        delete();

        //
        return previousVolumeEntry;
    }

    /**
     * Compares this object with the specified object for order. Returns a negative integer, zero, or a positive integer as this object is less than, equal to, or greater than the specified object.
     *
     * @param o
     * @return a negative integer, zero, or a positive integer as this object is less than, equal to, or greater than the specified object.
     * @see java.lang.Comparable#compareTo(Object)
     */
    public int compareTo(Object o) {
        try {
            return this.getCollator().compare(this.getCompleteHeadword(), ((VolumeEntry) o).getCompleteHeadword());
        } catch (PapillonBusinessException e) {
            throw new NullPointerException("Could not access to headword elements");
        }
    }

    public String toString() {
        try {
            return this.getCompleteHeadword();
        } catch (PapillonBusinessException e) {
            PapillonLogger.writeErrorMsg("Could not convert VolumeEntry to String: " + e.getLocalizedMessage());
            return "";
        }
    }



}
