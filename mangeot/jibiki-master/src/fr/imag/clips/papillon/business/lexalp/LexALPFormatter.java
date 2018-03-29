/*
 *
 *-----------------------
 * $Id$
 *------------------------
 * $Log$
 * Revision 1.14.2.7  2008/01/24 19:39:10  serasset
 * BUG123: lexies without geographical usage are not displayed in axies
 *
 * Revision 1.14.2.6  2008/01/15 15:23:50  serasset
 * BUG169: Entries from the same axie, with the SAME TERM and the SAME LEGAL SYSTEM are not displayed properly.
 *
 * Revision 1.14.2.5  2007/11/14 16:02:21  serasset
 * Suppressed debug messages.
 *
 * Revision 1.14.2.4  2007/09/06 14:54:11  serasset
 * in lexalp view, non matching entries are now gathered by legal systems and
 * query language now appears first.
 *
 * Revision 1.14.2.3  2007/09/05 15:24:13  serasset
 * Created a page to browse the dictionary index
 * Lexalp formatter now sorts entries correctly within a legal sysem/language group
 *
 * Revision 1.14.2.2  2007/07/25 15:15:43  serasset
 * BUGFIX: process and harmo status values were inverted in advanced search
 * BUGFIX: the source language is now added to the set of target language in simple
 *         search so that synonyms of the source entries are retreived in the result.
 *
 * Revision 1.14.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.14  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.13  2007/01/15 17:12:18  serasset
 * Several notes added, suppressed the HTMLDOM_CACHE stuff.
 *
 * Revision 1.12  2007/01/08 15:13:42  fbrunet
 * Correction of th xml attribut bug in ContributionHeader (VolumeEntry class)
 *
 * Revision 1.11  2006/11/09 09:04:42  fbrunet
 * *** empty log message ***
 *
 * Revision 1.10  2006/09/19 08:15:22  fbrunet
 * *** empty log message ***
 *
 * Revision 1.9  2006/09/18 12:24:54  fbrunet
 * - add xsl cascading to lexalp formatter (in xsl view, add tag nextXsl with dictionaryName, volumeName and xslName attributes)
 *
 * Revision 1.8  2006/09/11 19:57:48  fbrunet
 * - bug correction : interface edition (link axie to another axi)
 *
 * Revision 1.7  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.6  2006/06/01 22:05:05  fbrunet
 * New interface, quick search, new contribution management (the first edition not create new contribution. New contribution is created after add, remove element, update, save, etc. in the interface window)
 *
 * Revision 1.5  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.4.4.3  2006/01/25 15:22:23  fbrunet
 * Improvement of QueryRequest
 * Add new search criteria
 * Add modified status
 *
 * Revision 1.4.4.2  2006/01/24 13:39:49  fbrunet
 * Modification view management
 * Modification LexALP postprocessing
 *
 * Revision 1.4.4.1  2005/08/31 15:01:39  serasset
 * Applied modifications done on the LEXALP_1_0 branch to updated sources of the
 * trunk to create a new updated LEXALP_1_1 branch.
 *
 * Revision 1.4.2.1  2005/07/22 13:28:32  serasset
 * Modified EditEntryInit for Lexalp. It now serves as a main page for db maintenance.
 * Added a function to get url for QueryParameter.
 * Modified the way xslsheets are handled in order to allow several xslsheet with the same name, different dicts.
 *
 * Revision 1.4  2005/07/21 09:37:47  serasset
 * LexALPLinker had a pb with package since MM modification.
 * Lexalp query menu leads to AdvancedSearch.
 * XslSheetFactory's get default xsl for dict and volume now sets the names to "" during fallback.
 *
 * Revision 1.3  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 * Revision 1.2  2005/05/25 13:31:08  serasset
 * Return a monolingual entry even if the lexie is not linked to an axie.
 * LexALP transformer now formats simple monolingual query results.
 *
 * Revision 1.1  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 *------------------------
 */

package fr.imag.clips.papillon.business.lexalp;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.*;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.business.xsl.JibikiXsltExtension;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMResult;
import javax.xml.transform.dom.DOMSource;
import java.util.*;


public class LexALPFormatter
        implements ResultFormatter {

    // Constants
    // Note: I use constants extensively because the XSL transformations are a little
    // bit slow
    protected static Hashtable XslSheetCache = new Hashtable();

    // outils pour les transformation
    protected static final TransformerFactory myTransformerFactory = TransformerFactory.newInstance();
    protected static final DocumentBuilderFactory myDocumentBuilderFactory = DocumentBuilderFactory.newInstance();
    protected static DocumentBuilder myDocumentBuilder;

    protected XslSheet dictXsl = null;

    protected Dictionary currentDictionary;
    protected Volume sourceVolume;

    //
    private final boolean DEBUG = false;

    //
    public void initializeFormatter(Dictionary dictionary, Volume volume, Object parameter, int dialect, String lang)
            throws PapillonBusinessException {

        try {
            //
            String dictionaryName = dictionary.getName();
            String volumeName = volume.getName();

            //
            if (myDocumentBuilder == null) {
                myDocumentBuilder = myDocumentBuilderFactory.newDocumentBuilder();
            }

            //
            dictXsl = getXslSheet(dictionaryName, volumeName, (String) parameter);
			/*
			 System.out.println("Document Builder Factory is: " + myDocumentBuilderFactory.getClass());
			 System.out.println("Transformer Factory is: " + myTransformerFactory.getClass());
			 System.out.println("Document Builder is: " + myDocumentBuilder.getClass());
			 System.out.println("XSLSheet is: " + dictXsl.getName() + " " + dictXsl.getHandle());
			 */
			
        } catch (javax.xml.parsers.ParserConfigurationException e) {
            throw new PapillonBusinessException("CRITICAL: error initializing document builder !", e);
        }

    }

    public Node getFormattedResult(QueryResult qr, User usr) throws PapillonBusinessException {
        // First format the entries that were retrieved by the query, along with their translation
        // on the same Legal system
        /* 2.	always keep LEGAL SYSTEMS TOGETHER and display as a group (together in a box as it is now)
        display order for the legal system groups:
            1.	Legal systems containing search term first
            2.	if more than one hit, then rank legal systems from highest to lowest
             AC- INT - EU - national systems, the latter in the following order:
              1.	a) DE;	b) AT;		c) CH
              2.	a) FR;	b) CH
              3.	a) ITA;	b) CH
              4.	a) SL 
        */
        Document res = myDocumentBuilder.newDocument();
        Element rootdiv = res.createElement("div");
        rootdiv.setAttribute("class", "entry");
        res.appendChild(rootdiv);

        if (qr.getResultKind() == QueryResult.UNIQUE_RESULT) {
            if (null != dictXsl && !dictXsl.isEmpty()) {
                // Format document source
                Node resultNode = formatResult(qr.getSourceEntry().getDom(), dictXsl, usr);
                rootdiv.appendChild(res.importNode(resultNode, true));
            }

        } else if (qr.getResultKind() == QueryResult.AXIE_COLLECTION_RESULT) {

            // Matching entries, legal system by legal system
            // First display the matching entries
            HashMap matchingGroups = new HashMap();

            Iterator it = qr.getSourceEntries().iterator();
            while (it.hasNext()) {
                VolumeEntry ve = (VolumeEntry) it.next();
                // Make sure the source entry is not duplicated in the result set.
                qr.getLexiesHashMap().remove(ve.getEntryId());
                // Then format the group of related entries...
                String ls = IndexEntry.getCdmString(ve, "lexalp-legal-system");
                if (! matchingGroups.containsKey(ls)) {
                    // If the legal system has already been handled, this source entry should be disregarded. 
                    matchingGroups.put(ls, getMatchingResultGroup(res, qr, ve, usr));
                }
            }

            String[] lss = new String[]{"AC", "INT", "EU", "DE", "AT", "FR", "IT", "SL", "CH"};
            for (int i = 0; i < lss.length; i++) {
                Node lsn;
                if (null != (lsn = (Node) matchingGroups.remove(lss[i]))) {
                    rootdiv.appendChild(res.importNode(lsn, true));
                }
            }
            // Add other legal system groups (UNKNOW and/or invalid legal system values).
            if (!matchingGroups.isEmpty()) {
                Node lsn;
                Iterator keys = matchingGroups.keySet().iterator();
                while (keys.hasNext()) {
                    if (null != (lsn = (Node) matchingGroups.remove(keys.next()))) {
                        rootdiv.appendChild(res.importNode(lsn, true));
                    }
                }
            }
            String lg = qr.getFirstSourceEntry().getVolume().getSourceLanguage();
            // now dsplay unmatching groups
            for (int i = 0; i < lss.length; i++) {
                Node lsn = this.getResultGroup(res, qr, usr, lg, lss[i]);
                if (null != lsn) {
                    rootdiv.appendChild(res.importNode(lsn, true));
                }
            }
            if (!qr.getLexiesCollection().isEmpty()) {
                Node lsn = this.getResultGroup(res, qr, usr, lg, null);
                if (null != lsn) {
                    rootdiv.appendChild(res.importNode(lsn, true));
                }
            }
        }
        return rootdiv;
    }

    private Node getMatchingResultGroup(Document res, QueryResult qr, VolumeEntry ve, User user)
            throws PapillonBusinessException {
        return this.getMatchingResultGroup(res, qr, ve, user, ve.getVolume().getSourceLanguage());
    }


    private Node getMatchingResultGroup(Document res, QueryResult qr, VolumeEntry ve, User user, String lg)
            throws PapillonBusinessException {
        Element lsdiv = res.createElement("div");
        lsdiv.setAttribute("class", "lexalp-lsgroup");
        String ls = IndexEntry.getCdmString(ve, "lexalp-legal-system");
        Node resultNode = null;
        if (null != dictXsl && !dictXsl.isEmpty()) {
            // Format document source
            resultNode = formatResult(ve.getDom(), dictXsl, user);
        }
        this.addLegalSystemGroup(lsdiv, res, qr, user, lg, ls);
        // Test if there are associated elements in the lsgroup. If not, skip the lxgroup...
        Node firstChild = lsdiv.getFirstChild();
        if (null != firstChild && null != resultNode) {
            lsdiv.insertBefore(res.importNode(resultNode, true), firstChild);
            return lsdiv;            
        } else {
            return res.importNode(resultNode, true);
        }
    }

    private Node getResultGroup(Document res, QueryResult qr, User user, String lg, String ls)
            throws PapillonBusinessException {
        Element lsdiv = res.createElement("div");
        lsdiv.setAttribute("class", "lexalp-lsgroup");
        this.addLegalSystemGroup(lsdiv, res, qr, user, lg, ls);
        if (lsdiv.hasChildNodes()) {
            if (null != lsdiv.getFirstChild().getNextSibling()) {
                return lsdiv;
            } else {
                return lsdiv.getFirstChild();
            }
        } else {
            return null;
        }
    }

    private void addLegalSystemGroup(Element lsdiv, Document res, QueryResult qr, User user, String lg, String ls)
            throws PapillonBusinessException {
        String[] langs = new String[]{"deu", "fra", "ita", "slv"};
        { // FIXME: Duplicate code everywhere....
            Collection trads = getAndRemoveResultEntriesFor(qr, ls, lg);
            for (Iterator it = trads.iterator(); it.hasNext();) {
                VolumeEntry trad = (VolumeEntry) it.next();
                if (null != dictXsl && !dictXsl.isEmpty()) {
                    // Format document source
                    Element tradNode = (Element) formatResult(trad.getDom(), dictXsl, user);
                    tradNode.setAttribute("class", "translation");
                    lsdiv.appendChild(res.importNode(tradNode, true));
                }
            }
        }
        for (int i = 0; i < langs.length; i++) {
            if (!langs[i].equals(lg)) {
                Collection trads = getAndRemoveResultEntriesFor(qr, ls, langs[i]);
                for (Iterator it = trads.iterator(); it.hasNext();) {
                    VolumeEntry trad = (VolumeEntry) it.next();
                    if (null != dictXsl && !dictXsl.isEmpty()) {
                        // Format document source
                        Element tradNode = (Element) formatResult(trad.getDom(), dictXsl, user);
                        tradNode.setAttribute("class", "translation");
                        lsdiv.appendChild(res.importNode(tradNode, true));
                    }
                }
            }
        }
    }

    private Collection getAndRemoveResultEntriesFor(QueryResult qr, String legalSystem, String lang) {
        Collection lexies = qr.getLexiesCollection();
        Iterator it = lexies.iterator();

        TreeSet results = new TreeSet();
        while (it.hasNext()) {
            VolumeEntry ve = (VolumeEntry) it.next();
            try {
                String ls = IndexEntry.getCdmString(ve, "lexalp-legal-system");
                String lg = ve.getVolume().getSourceLanguage();
                if ((legalSystem == null || legalSystem.equals(ls)) && lang.equals(lg)) {
                    results.add(ve);
                }
            } catch (PapillonBusinessException e) {
                // do nothing
            }
        }
        for(Iterator itResults = results.iterator(); itResults.hasNext();) {
            try {
                qr.getLexiesHashMap().remove(((VolumeEntry)itResults.next()).getEntryId());
            } catch (PapillonBusinessException e) {
                // do nothing... 
            }
        }
        return results;
    }

    /**
     * ...
     * @return Root node representing the
     * @param docSource DOM version of the volume entry to be formatted
     * @param xsl XslSheet used for formatting
     * @param usr User currently requesting the result.
     * @throws fr.imag.clips.papillon.business.PapillonBusinessException
     */
    private Node formatResult(Document docSource, XslSheet xsl, User usr) throws PapillonBusinessException {
        try {

            // init
            boolean isTransform = true;
            Document docCible = null;

            //
            while (isTransform) {

                 // System.out.println("Transforming with " + xsl.getName() + "/" + xsl.getVolumeName() + " :");
                 // System.out.println(XMLServices.xmlCodePrettyPrinted(docSource));
                // Transform
                docCible = Transform(docSource, xsl);

                // System.out.println("Result of transformation :");
                // System.out.println(XMLServices.xmlCodePrettyPrinted(docCible));

                // Replace FindReference element by ResultReference
                // attributes :  termReference, lang, xpath, xslName, dictionaryName, volumeName, ...
                NodeList list = docCible.getElementsByTagName("FindReference");
                while (list.getLength() > 0) {
                    Node node = list.item(0);
                    Node parentNode = node.getParentNode();

                    //
                    Node termRefAttribut = node.getAttributes().getNamedItem("termReference");
                    String termRef = termRefAttribut.getNodeValue();
                    if (DEBUG) {
                        System.out.println("LexALPFormatter: " + termRef);
                    }
                    Node langAttribut = node.getAttributes().getNamedItem("lang");
                    String lang = langAttribut.getNodeValue();
                    if (DEBUG) {
                        System.out.println("LexALPFormatter: " + lang);
                    }
                    Node xpathAttribut = node.getAttributes().getNamedItem("xpath");
                    String xpath = xpathAttribut.getNodeValue();
                    if (DEBUG) {
                        System.out.println("LexALPFormatter: " + xpath);
                    }
                    Node xslNameAttribut = node.getAttributes().getNamedItem("xslName");
                    Node dictionaryNameAttribut = node.getAttributes().getNamedItem("dictionaryName");
                    Node volumeNameAttribut = node.getAttributes().getNamedItem("volumeName");
                    //Node namespaceAttribut = node.getAttributes().getNamedItem("namespace");

                    //
                    String xslName = "";
                    if (xslNameAttribut != null) {
                        xslName = xslNameAttribut.getNodeValue();
                    }

                    //
                    String dictionaryName = "";
                    if (dictionaryNameAttribut != null) {
                        dictionaryName = dictionaryNameAttribut.getNodeValue();
                    }

                    //
                    String volumeName = "";
                    if (volumeNameAttribut != null) {
                        volumeName = volumeNameAttribut.getNodeValue();
                    }

                    //FIXME: search the specified entry, then gets its dictionary and volume to correctly select the XSL
                    //newXsl = XslSheetFactory.findXslSheetByName(xslNameAttribut.getNodeValue());
                    XslSheet newXsl = XslSheetFactory.getXslSheet(dictionaryName, volumeName, xslName);
                    //System.out.println("XslSheetFactory.getXslSheet(" + dictionaryName + ", " + volumeName + ", " + xslName + ") -> " + xpath);

                    /*
                    // Find entry
                    QueryRequest queryReq = new QueryRequest(VolumesFactory.getVolumesArrayName(null, lang, null));
                    QueryCriteria criteria = new QueryCriteria();
                    criteria.add("key", QueryCriteria.EQUAL, Volume.CDM_entryId);
                    criteria.add("value", QueryCriteria.EQUAL, termRef);     // Termref is an entryid
                    queryReq.addCriteria(criteria);
                    QueryCriteria criteriaStatus = new QueryCriteria();
                    criteriaStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);                  // FINISHED_STATUS
                    criteriaStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.FINISHED_STATUS);
                    queryReq.addCriteria(criteriaStatus);

                    //
                    Collection qrset = queryReq.findLexie(usr);

                    // Find not finished entry if no finished entry
                    // FIXME: guess user can view not finished status !!!
                    if ( qrset.isEmpty()) {
                        queryReq = new QueryRequest(VolumesFactory.getVolumesArrayName(null, lang, null));
                        queryReq.addCriteria(criteria);
                        criteriaStatus = new QueryCriteria();
                        criteriaStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);                  // NOT_FINISHED_STATUS
                        criteriaStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.NOT_FINISHED_STATUS);
                        queryReq.addCriteria(criteriaStatus);

                        //
                        qrset = queryReq.findLexie(usr);
                    }
                    */

                    // Find entry
                    // FIXME : Change DB to do quick entry id search !
                    QueryRequest queryReq = new QueryRequest(VolumesFactory.getVolumesArray(null, lang, null));
                    QueryCriteria criteria = new QueryCriteria();
                    criteria.add("key", QueryCriteria.EQUAL, Volume.CDM_entryId);
                    criteria.add("value", QueryCriteria.EQUAL, termRef);     // Termref is an entryid
                    queryReq.addCriteria(criteria);

                    ArrayList listStatus = new ArrayList();
                    QueryCriteria criteriaFinishedStatus = new QueryCriteria();
                    criteriaFinishedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);
                    criteriaFinishedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.FINISHED_STATUS);
                    listStatus.add(criteriaFinishedStatus);
                    QueryCriteria criteriaModifiedStatus = new QueryCriteria();
                    criteriaModifiedStatus.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionStatus);
                    criteriaModifiedStatus.add("value", QueryCriteria.EQUAL, VolumeEntry.MODIFIED_STATUS);
                    listStatus.add(criteriaModifiedStatus);
                    queryReq.addOrCriteriaList(listStatus);

                    //
                    Collection qrset = queryReq.findLexie(usr);

                    //
                    Iterator iter = qrset.iterator();
                    while (iter.hasNext()) {
                        QueryResult relatedQr = (QueryResult) iter.next();

                        // Find nodes with xpath
                        NodeList nodeL = relatedQr.getSourceEntry().getNodes(xpath);

                        // Insert new nodes
                        for (int j = 0; j < nodeL.getLength(); j++) {
                            Document docXpath = myDocumentBuilder.newDocument();
                            Element result = docXpath.createElement("ResultReference");
                            docXpath.appendChild(result);
                            Node nodeXpath = docXpath.importNode(nodeL.item(j), true);
                            result.appendChild(nodeXpath);

                            //
                            if (newXsl != null) {
                                result = (Element) formatResult(docXpath, newXsl, usr);
                            }

                            //
                            if (DEBUG) {
                                System.out.println(
                                        "LexALPFormatter: " + fr.imag.clips.papillon.business.xml.XMLServices.NodeToString(
                                                result));
                            }
                            parentNode.insertBefore(docCible.importNode(result, true), node);
                        }
                    }

                    //
                    parentNode.removeChild(node);

                    // Here because Res change after removeChild and appendChild methods
                    list = docCible.getElementsByTagName("FindReference");
                }

                // Manage xsl cascading !!
                NodeList nextXslList = docCible.getElementsByTagName("nextXsl");
                /// fixme : id !!! supress ... not in xsl file !

                if (nextXslList.getLength() > 0) {
                    Node nextXsl = nextXslList.item(0);

                    //
                    Node nextXslNameAttribut = nextXsl.getAttributes().getNamedItem("xslName");
                    Node nextDictionaryNameAttribut = nextXsl.getAttributes().getNamedItem("dictionaryName");
                    Node nextVolumeNameAttribut = nextXsl.getAttributes().getNamedItem("volumeName");

                    //
                    String nextXslName = "";
                    if (nextXslNameAttribut != null) {
                        nextXslName = nextXslNameAttribut.getNodeValue();
                    }

                    //
                    String nextDictionaryName = "";
                    if (nextDictionaryNameAttribut != null) {
                        nextDictionaryName = nextDictionaryNameAttribut.getNodeValue();
                    }

                    //
                    String nextVolumeName = "";
                    if (nextVolumeNameAttribut != null) {
                        nextVolumeName = nextVolumeNameAttribut.getNodeValue();
                    }

                    //
                    xsl = XslSheetFactory.getXslSheet(nextDictionaryName, nextVolumeName, nextXslName);
                    isTransform = true;
                    docSource = docCible;

                    // erase node nextXsl !!!!

                } else {
                    isTransform = false;
                }

            }

            /*
            // Manage entry actions
            Element actionElement = docCible.getElementById("actions");
            if (true && (actionElement!=null)) {
                // Replace actions
                
                
            } else if (actionElement!=null) {
                // Delete actions ...
                
                actionElement.getParentNode().removeChild(actionElement);
                
            }
            */

            //
            return docCible.getDocumentElement();

        } catch (Exception ex) {
            throw new PapillonBusinessException("Exception in getFormattedResult()", ex);
        }
    }


    /**
     * Transform the xml source by processing it with an xsl sheet.
     */
    protected static Document Transform(Node xmlSource, XslSheet xslSheet) throws
                                                                           fr.imag.clips.papillon.business.PapillonBusinessException,
                                                                           javax.xml.parsers.ParserConfigurationException,
                                                                           javax.xml.transform.TransformerException,
                                                                           java.io.IOException {

        // Find transformer
        // FIXME: if transformer is null ?
        Transformer myTransformer = xslSheet.getTransformer();

        // PapillonLogger.writeDebugMsg("Class: " + myTransformer.getClass());
        // PapillonLogger.writeDebugMsg("Factory: " + TransformerFactory.class.toString());
        // The xalan transformer uses the context class loader instead of the system class loader
        // But in enhydra 5.1, the system class loader is a custom class loader, hence we have
        // to set the context class loader to the enhydra classloader, or the Extension classes
        // would not be visible.
        try {
            Thread currentThread = Thread.currentThread();
            currentThread.setContextClassLoader(JibikiXsltExtension.class.getClassLoader());
        } catch (SecurityException e) {
            PapillonLogger.writeDebugMsg("Could not modify the context class loader.");
        }

        // The source
        if (myDocumentBuilder == null) {
            myDocumentBuilder = myDocumentBuilderFactory.newDocumentBuilder();
        }
        Document newDocument = myDocumentBuilder.newDocument();

        //
        myTransformer.transform(new DOMSource(xmlSource), new DOMResult(newDocument));

        //
        return newDocument;
    }


    /**
     * Return the xsl sheet corresponding to the name, dictionary name and volumename
     *
     * @param dictionaryName
     * @param volumeName
     * @param name
     * @return XslSheet
     * @throws PapillonBusinessException
     */
    public static XslSheet getXslSheet(String dictionaryName, String volumeName, String name)
            throws PapillonBusinessException {

        XslSheet theXsl = null;

        // find specific xsl
        theXsl = XslSheetFactory.getXslSheet(dictionaryName, volumeName, name);

        // find specific xsl
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet(dictionaryName, "", name);
        }

        // find specific xsl
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet("", "", name);
        }

        // find defaut
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet(dictionaryName, volumeName, "");
        }

        // find defaut
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet(dictionaryName, "", "");
        }

        // find defaut
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet("", "", "");
        }

        //
        return theXsl;
    }


}



















