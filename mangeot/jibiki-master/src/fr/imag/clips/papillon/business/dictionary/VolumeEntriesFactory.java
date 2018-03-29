/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.58.2.2  2007/10/02 10:31:21  serasset
 * Modified export of volume to speed up this task.
 *
 * Revision 1.58.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.58  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.57  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.56  2006/12/14 20:03:26  fbrunet
 * Add method to normalize value into XML structure.
 *
 * Revision 1.55  2006/12/13 09:32:00  fbrunet
 * *** empty log message ***
 *
 * Revision 1.54  2006/12/08 14:55:16  fbrunet
 * Add new method in VolumeEntriesFactory.java - getVolumeEntries - to correct transformation xsl bug
 *
 * Revision 1.53  2006/09/12 19:26:10  fbrunet
 * - improve reconstruction index
 *
 * Revision 1.52  2006/08/10 22:17:12  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.51  2006/04/24 13:43:29  fbrunet
 * Add new class ViewQueryResult : allow to use one class to create result display in advancedSearch and EditEntryInit (like advancedQueryForm)
 * Improve result display : view n results per page
 *
 * Revision 1.50  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 * Revision 1.49  2006/03/29 10:18:17  mangeot
 * Fixed a bug when the source language is not defined and the request uses the msort column (> and < comparisons)
 *
 * Revision 1.48  2006/03/15 13:36:39  mangeot
 * Bug fix in queries with pointers with default language
 *
 * Revision 1.47  2006/03/13 08:48:00  fbrunet
 * bug corrections before merge
 *
 * Revision 1.46  2006/03/12 23:31:44  mangeot
 * Added notFinishedStatus in findEntryByEntryId
 * CVS: ----------------------------------------------------------------------
 *
 * Revision 1.45  2006/03/12 23:19:18  mangeot
 * Fixed a bug that forbed to retrieve entries by translation ids
 *
 * Revision 1.44  2006/03/10 16:43:17  mangeot
 * Fix for array.length==0, not satisfied
 *
 * Revision 1.43  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.42  2006/02/28 15:26:22  mangeot
 * Bug fix when creating new tables
 *
 * Revision 1.41  2006/02/27 00:16:57  mangeot
 * *** empty log message ***
 *
 * Revision 1.40  2006/02/26 22:05:02  mangeot
 * *** empty log message ***
 *
 * Revision 1.39  2006/02/26 21:56:05  mangeot
 * *** empty log message ***
 *
 * Revision 1.37  2006/02/26 14:08:16  mangeot
 * Added the multilingual_sort(lang,headword) index on volume tables for speeding up the lookup
 *
 * Revision 1.36  2006/02/21 15:27:49  mangeot
 * *** empty log message ***
 *
 * Revision 1.35  2006/02/21 13:37:54  mangeot
 * *** empty log message ***
 *
 * Revision 1.34  2005/12/01 17:17:25  mangeot
 * *** empty log message ***
 *
 * Revision 1.33  2005/12/01 15:34:28  mangeot
 * MM: I solved the problem of already created tables by creating an sql query for retrieving the table names. If the name already exists, VolumeEntriesFactory.createVolumeTables do not create the tables.
 * It allows the administrator to delete and reload only the metadata files without dropping the whole data.
 * The method is ManageDatabase.getTableNames() and it returns a vector with all the table names created by the database user (usually "papillon").
 *
 * Revision 1.32  2005/11/22 13:21:02  mangeot
 * I moved the VolumeEntriesFactory.createVolumeTables out of the database transactions in AdminDictionaries.java and Adminvolumes.java because otherwise, it is not possible to reload metadata when the data tables already exist (in this case, the transaction does not commit).
 *
 * Revision 1.31  2005/11/14 22:53:26  mangeot
 * *** empty log message ***
 *
 * Revision 1.30  2005/11/14 22:49:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.29  2005/11/14 22:08:07  mangeot
 * *** empty log message ***
 *
 * Revision 1.28  2005/11/14 22:06:07  mangeot
 * *** empty log message ***
 *
 * Revision 1.27  2005/11/09 17:44:05  mangeot
 * Deleted axi volume table special handling, not needed any more
 *
 * Revision 1.26  2005/09/08 15:04:25  mangeot
 * *** empty log message ***
 * Revision 1.25.2.5  2006/02/17 10:41:48  fbrunet
 * Change QueryCriteria parameters
 * Add new windows when editing an entry
 *
 * Revision 1.25.2.4  2006/01/25 15:22:23  fbrunet
 * Improvement of QueryRequest
 * Add new search criteria
 * Add modified status
 *
 * Revision 1.25.2.3  2005/12/02 10:04:09  fbrunet
 * Add Pre/Post edition processing
 * Add index reconstruction
 * Add new query request
 * Add fuzzy search
 * Add new contribution administration
 * Add xsl transformation volume
 *
 * Revision 1.25.2.2  2005/10/24 16:29:19  fbrunet
 * Added fuzzy search capabilities.
 * Added possibility to rebuild the index DB tables.
 * Added Pre and post processors that could be defined by the user.
 *
 * Revision 1.25.2.1  2005/08/31 15:01:39  serasset
 * Applied modifications done on the LEXALP_1_0 branch to updated sources of the
 * trunk to create a new updated LEXALP_1_1 branch.
 *
 * Revision 1.25  2005/08/24 13:59:35  serasset
 * Added FIXME comments.
 *
 * Revision 1.24  2005/08/17 12:58:16  mangeot
 * Fixed a bug when creating an entry from an existing one.
 * From now on, the entry id is the same.
 * Added the links into ReviewContributions.java
 *
 * Revision 1.23  2005/08/16 12:11:18  mangeot
 * Bug fix in findEntryByEntryId method
 *
 * Revision 1.22  2005/08/16 11:25:04  mangeot
 * Modified findEntryByEntryId because it can return several results depending on the status. I take in priority order VALIDATED > REVIEWED > FINISHED > NOT FINISHED
 *
 * Revision 1.21  2005/08/05 18:44:38  mangeot
 * Bug fixes + ProcessVolume.po page creation
 *
 * Revision 1.20  2005/08/01 17:37:33  mangeot
 * Bug fix in sort function
 *
 * Revision 1.19  2005/07/30 13:52:13  mangeot
 * Commit due to some conflicts between directories. Beware !
 *
 * Revision 1.18  2005/07/28 15:34:34  mangeot
 * *** empty log message ***
 *
 * Revision 1.17  2005/07/28 13:06:47  mangeot
 * - Added the possibility to export in PDF format. The conversion into PDF is don
 * e via the fop package that has to be installed (see ToolsForPapillon)
 *
 * Revision 1.16.2.2  2005/07/22 13:28:32  serasset
 * Modified EditEntryInit for Lexalp. It now serves as a main page for db maintenance.
 * Added a function to get url for QueryParameter.
 * Modified the way xslsheets are handled in order to allow several xslsheet with the same name, different dicts.
 *
 * Revision 1.16.2.1  2005/07/21 10:25:28  serasset
 * For LEXALP, the save button in EditEntry.po means that you FINISH the entry.
 *
 * Revision 1.16  2005/07/16 13:41:52  mangeot
 * Added findEntryByContributionId
 *
 * Revision 1.15  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 * Revision 1.14  2005/06/24 10:35:57  mangeot
 * Minor bug fixes
 *
 * Revision 1.13  2005/06/22 15:55:53  mangeot
 * Solved an unresolved prefix bug when the dml prefix was not in the template entry.
 * Now we use the DmlPrefixResolver to solve this issue.
 *
 * Revision 1.12  2005/06/20 16:55:05  mangeot
 * multiple bug fixes
 *
 * Revision 1.11  2005/06/17 15:51:32  mangeot
 * Modified changeAuthor
 *
 * Revision 1.10  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.9  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.8.4.15  2005/06/15 10:08:06  mangeot
 * Removed the AND/OR connector, now only AND criteria can be added for dict lookup
 *
 * Revision 1.8.4.14  2005/06/14 11:56:16  mangeot
 * Added a new page ChangeAuthor for changing the author of a set of previously selected contributions
 *
 * Revision 1.8.4.13  2005/06/10 14:00:40  mangeot
 * Changed Edit to Copy and Edit when a copy of the original entry is done before editing
 *
 * Revision 1.8.4.12  2005/06/09 11:28:24  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.11  2005/06/09 11:07:45  mangeot
 * Deleted the countEntriesCache. entries counts are not cached any more.
 * Fixed a few bugs.
 *
 * Revision 1.8.4.10  2005/06/01 08:38:43  mangeot
 * Multi bug correction + added the possibility of disabling data edition
 * via the Admin.po page
 *
 * Revision 1.8.4.9  2005/05/27 11:53:21  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.8  2005/05/20 17:02:22  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.7  2005/05/20 16:54:53  mangeot
 * Added ExportVolume functionnality
 *
 * Revision 1.8.4.6  2005/05/20 14:43:48  mangeot
 * Repair mismatch in branch tag
 *
 * Revision 1.8.4.5  2005/05/19 09:35:27  mangeot
 * Doing on ly one sql request when querying the volumes
 *
 * Revision 1.8.4.4  2005/05/14 11:56:28  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.3  2005/05/11 15:34:00  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.2  2005/04/29 17:35:12  mangeot
 * *** empty log message ***
 *
 * Revision 1.8.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.8  2005/04/18 10:50:26  mangeot
 * Bug fix when displaying with IExplorer,
 * Bug fixes when seqencial request
 *
 * Revision 1.7  2005/04/15 13:20:08  mangeot
 * Added setIdIfNull
 *
 * Revision 1.6  2005/04/13 14:34:38  mangeot
 * Simplified the expert lookup. Now lookup directly the cdm element name
 *
 * Revision 1.5  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.4.2.10  2005/03/31 08:54:16  mangeot
 * Do not throw an error if the volume tables already exist
 *
 * Revision 1.4.2.9  2005/03/30 11:17:07  mangeot
 * Modified table contributions: replaced originalhandle by originalid
 * Corrected a few bugs when validating an already existing entry
 *
 * Revision 1.4.2.8  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.4.2.7  2005/02/25 10:23:07  mangeot
 * Bug fixes
 *
 * Revision 1.4.2.6  2005/02/06 22:43:49  mangeot
 * Merged the 2 Hashtables CDM Elements and XPaths into one
 * Added a boolean (reverse-lookup) in the volume metadata and functionalities in order to perform a reverse lookup when no direct lookup result is found
 * Added a boolean (index) in the volume metadata for indexing the only specified CDM Elements
 *
 * Revision 1.4.2.5  2005/01/28 23:01:09  mangeot
 * Fixed bugs in the editor. It seems to work now. More testing needed anyway...
 *
 * Revision 1.4.2.4  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
 *
 * Revision 1.4.2.3  2005/01/27 23:55:13  mangeot
 * *** empty log message ***
 *
 * Revision 1.4.2.2  2005/01/27 18:09:28  mangeot
 * Simple dictionary lookup is now working for GDEF.
 * Does not compile yet but cvs commit for backup
 *
 * Revision 1.4.2.1  2005/01/27 15:56:21  mangeot
 * Able to load a volume with XPointers, cannot lookup the result yet.
 * Does not compile but commit for backup
 *
 * Revision 1.4  2005/01/18 12:16:10  mangeot
 * Implemented the SQL LIMIT and OFFSET keywords. It allows us to retrieve the entries as blocks and page them. The LIMIT is the DictionariesFactory.MaxRetrievedEntries constant.
 * The implementation may need further tuning
 *
 * Revision 1.3  2005/01/18 09:41:11  mangeot
 * Recoded the countRows method with a new method that appeared with DODS 5.1
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 *
 */

package fr.imag.clips.papillon.business.dictionary;

import com.lutris.appserver.server.sql.ObjectId;
import com.lutris.dods.builder.generator.query.QueryBuilder;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.CurrentRequestContext;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.UsersFactory;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.papillon_data.ManageDatabase;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.JOptionPane;

/**
* Used to find the instances of xslsheet.
 */
public class VolumeEntriesFactory {
	
	
	public final static String AUTHOR_SORT = "AUTHOR_SORT";
	public final static String CREATION_DATE_SORT = "CREATION_DATE_SORT";
	public final static String HEADWORD_SORT = "HEADWORD_SORT";
	public final static String ORIGINAL_CONTRIBUTION_ID_SORT = "ORIGINAL_CONTRIBUTION_ID_SORT";
	public final static String REVIEW_DATE_SORT = "REVIEW_DATE_SORT";
	public final static String REVIEWER_SORT = "REVIEWER_SORT";
	public final static String STATUS_SORT = "STATUS_SORT";
	
	protected final static String MSORT_FIELD = "msort";
	protected final static String ORDER_DESCENDING = "DESC";
	
	public static final String XMLFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XML_DIALECT);
	public static final String XHTMLFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XHTML_DIALECT);
	public static final String TEXTFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.PLAINTEXT_DIALECT);
	public static final String PDFFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.PDF_DIALECT);
	
	protected static final String XhtmlHeader = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
		+ "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n"
		+ "<html>\n"
		+ "  <head>\n"
		+ "    <meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" />\n"
		+ "    <link href=\"http://www.estfra.ee/css/gdef-home.css\" rel=\"StyleSheet\" type=\"text/css\" />\n"
		+ "    <title>Papillon Generated Dictionary</title>\n"
		+ "  </head>\n"
		+ "  <body>\n"
		+ "    <h1>Papillon Generated Dictionary</h1>\n";
	
	protected static final String XhtmlFooter = "</body>\n"
		+ "</html>\n";
	
	//protected static final String TextHeader = "Dictionary:\n\n";
	protected static final String TextHeader = "";
	protected static final String TextFooter = "";
	
	public static final String FoHeader = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
		+ "<fo:root xmlns:fo=\"http://www.w3.org/1999/XSL/Format\">\n"
		+ "  <fo:layout-master-set>\n"
		+ "    <fo:simple-page-master master-name=\"simpleA4\" page-height=\"29.7cm\" page-width=\"21cm\" margin-top=\"2cm\" margin-bottom=\"2cm\" margin-left=\"2cm\" margin-right=\"2cm\">\n"
		+ "      <fo:region-body/>\n"
		+ "    </fo:simple-page-master>\n"
		+ "  </fo:layout-master-set>\n"
		+ "  <fo:page-sequence master-reference=\"simpleA4\">\n"
		+ "    <fo:flow flow-name=\"xsl-region-body\">";
    
	public static final String FoFooter = "    </fo:flow>\n"
		+ "  </fo:page-sequence>\n"
		+ "</fo:root>\n";
	
	protected static java.util.Hashtable VolumeEntriesCountHashtable = new java.util.Hashtable();
	protected static String NoStatus = "#NoStatus#";

	protected static java.util.regex.Pattern quotePattern = java.util.regex.Pattern.compile("'");

	protected static com.lutris.dods.builder.generator.query.RDBColumn[] Columns = new com.lutris.dods.builder.generator.query.RDBColumn[1];
	
    /**
        * The getVolumeEntriesArray method performs a database query to
     * return an array of Dictionary entries
     * @return
     *     array of discs.
     * @exception PapillonBusinessException
     *   If there is a problem retrieving disc information.
     */
	
	
	public static Vector getVolumeEntriesVector(Dictionary dict, Volume volume, Vector Keys1, Vector Keys2, String any, int offset, int limit) throws PapillonBusinessException {
        Vector MyTable = null;
		//PapillonLogger.writeDebugMsg("getVolumeEntriesVector: " + volume.getName());
        if (null != volume) {
            if (volume.getLocation().equals(Volume.LOCAL_LOCATION)) {
                MyTable = getDbTableEntriesVector(dict, volume,Keys1, Keys2, any, offset, limit);
            }
            else if (volume.getLocation().equals(Volume.REMOTE_LOCATION)) {
                MyTable = getRemoteVolumeEntriesVector(dict, volume, Keys1, Keys2, any);
            }
        }
        return MyTable;
    }
	
	
    public static Vector getRemoteVolumeEntriesVector(Dictionary dict,
                                                      Volume volume,
                                                      Vector Keys1,
                                                      Vector Keys2,
                                                      String any) 
		throws PapillonBusinessException {
			Vector theEntries = new Vector();
			try {
				if (null != volume) {
					fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Remote Volume: " + volume.getName());
					if (volume.getName().equals("Lexico_eng")) {
						Wrapper myWrapper = WrappersFactory.createLexicoWrapper();
						RemoteEntry myEntry = new RemoteEntry();
						myEntry.setDictionary(dict);
						myEntry.setVolume(volume);
						// myEntry.setXmlCode(myWrapper.getResultXmlCode(Headwords));
						//               theEntries.put(myEntry.getHandle(),myEntry);
						//theEntries.add(myEntry);
					}
				}
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in getVolumeEntriesArray()", ex);
			}
			return theEntries;
		}
	
    public static Vector getVolumeNameEntriesVector(String volumeName,
                                                    Vector Keys1,
                                                    Vector Keys2,
													String any)
        throws PapillonBusinessException {
			return getVolumeNameEntriesVector(volumeName, Keys1, Keys2, any, "", 0, 0);
		}
	
	public static Vector getVolumeNameEntriesVector(String volumeName,
                                                    Vector Keys,
                                                    Vector Clauses,
													String any,
													int offset,
                                                    int limit)
        throws PapillonBusinessException {
			return getVolumeNameEntriesVector(volumeName, Keys, Clauses, any, "", offset, limit);
		}
	
    public static Vector getVolumeNameEntriesVector(String volumeName,
                                                    Vector Keys,
                                                    Vector Clauses,
													String any,
													String order,
													int offset,
                                                    int limit)
        throws PapillonBusinessException {
			Vector resultVector = null;
			if (volumeName != null && !volumeName.equals("")) {
				Volume volume;
				Dictionary dict;
				try {
					volume = VolumesFactory.getVolumeByName(volumeName);
					if (volume != null && !volume.isEmpty()) {
						dict = DictionariesFactory.getDictionaryByName(volume.getDictname());
						if (dict != null && !dict.isEmpty()) {
							resultVector = getDbTableEntriesVector(dict, volume, Keys, Clauses, any, order, offset, limit);
						}
					}
				}
				catch(Exception ex) {
					throw new PapillonBusinessException("Exception in getVolumeNameEntriesVector()", ex);
				}
			}
			return resultVector;
        }
    
	
	protected static Vector getDbTableEntriesVector(Dictionary dict, Volume volume, Vector Keys, Vector Clauses, String any, int offset, int limit) throws PapillonBusinessException {
		return getDbTableEntriesVector(dict, volume, Keys, Clauses, any, "", offset, limit);
	}
	
    protected static Vector getDbTableEntriesVector(Dictionary dict, Volume volume, Vector Keys, Vector Clauses, String any, String order, int offset, int limit) throws PapillonBusinessException {
        Vector theEntries = theEntries = new Vector();
		
		String sourceLanguage = volume.getSourceLanguage();
		String volumeTableName = volume.getDbname();
		if (null != volumeTableName) {
			try {
                if (limit==0) {
                    limit = DictionariesFactory.MaxRetrievedEntries;
                }

				com.lutris.dods.builder.generator.query.QueryBuilder myQueryBuilder = null;
				com.lutris.dods.builder.generator.query.RDBColumn entryidColumn = IndexDO.getEntryIdColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBColumn keyColumn = IndexDO.getKeyColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBColumn langColumn = IndexDO.getLangColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBColumn valueColumn = IndexDO.getValueColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBTable volumeEntryTable = new com.lutris.dods.builder.generator.query.RDBTable(volumeTableName);
				VolumeEntryDO myDO = VolumeEntryDO.createVirgin(volumeTableName);
				com.lutris.dods.builder.generator.query.RDBColumn objectidColumn = new com.lutris.dods.builder.generator.query.RDBColumn(volumeEntryTable, myDO.getOIdColumnName());
				// consultation of a local volume
				VolumeEntryQuery query = new VolumeEntryQuery(volumeTableName, CurrentDBTransaction.get());
				
				// distinct avoid to retreive the same entry twice
				// The distinct cannot be combined with an order by...
				// The ORDER BY clause can include items not appearing in the select list. 
				// However, if SELECT DISTINCT is specified, or if the SELECT statement contains a UNION operator,
				// the sort columns must appear in the select list.
				//query.getQueryBuilder().distinct();
                
				// looking for any string on the XML code, thus Sequencial request
				if (any !=null && !any.equals("")) {
					fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Sequencial request table: " + volumeTableName + " any: " + any);
					com.lutris.dods.builder.generator.query.RDBColumn xmlcodeColumn = VolumeEntryDO.getXmlCodeColumn(volume.getDbname());
					query.getQueryBuilder().addWhere(xmlcodeColumn, any, QueryBuilder.CASE_SENSITIVE_CONTAINS);
				}
				else {
					//fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Index request table: " + volumeTableName);
				}
				
				if (Keys != null) {
					for (java.util.Enumeration enumKeys = Keys.elements(); enumKeys.hasMoreElements();) {
						String[] key = (String[]) enumKeys.nextElement();
                        // debug
                        //System.out.println("VolumeEntriesFactory.getDbTableEntriesVector: " + key[0] + " | " + key[1] + " | " + key[2] + " | " + key[3]);
						if (key!=null && key[2] !=null && !key[2].equals("")) {
							myQueryBuilder = new com.lutris.dods.builder.generator.query.QueryBuilder(Columns);
							if (IndexFactory.databaseVendor != null) {
								myQueryBuilder.setDatabaseVendor(IndexFactory.databaseVendor);
							} else {
								myQueryBuilder.setDatabaseVendor();
							}
                            /* clés multiples */
                            String[] keynames = key[0].split("\\|");
                            
                            if (keynames.length>1) {
                                myQueryBuilder.addWhereOpenParen();
                                for (int i=0;i<keynames.length;i++) {
                                    myQueryBuilder.addWhere(keyColumn, keynames[i], QueryBuilder.EQUAL);
                                    if (i<keynames.length-1) {
                                        myQueryBuilder.addWhereOr();
                                    }
                                }
                                myQueryBuilder.addWhereCloseParen();
                            }
                            else {
                                myQueryBuilder.addWhere(keyColumn, keynames[0], QueryBuilder.EQUAL);
                            }
                            
							if ((key[1] ==null || key[1].equals(""))) {
								if (volume.isSourceLangCDMElement(key[0])) {
									key[1] = sourceLanguage;
								}
								if (volume.isDefaultLangCDMElement(key[0])) {
									key[1] = Volume.DEFAULT_LANG;
								}
							}
							if (key[1] !=null && !key[1].equals("")) {
								myQueryBuilder.addWhere(langColumn, key[1], QueryBuilder.EQUAL);
							}
							if (key[1] != null && !key[1].equals("") 
								&& (key[3] == QueryBuilder.LESS_THAN ||
									key[3] == QueryBuilder.LESS_THAN_OR_EQUAL ||
									key[3] == QueryBuilder.GREATER_THAN ||
									key[3] == QueryBuilder.GREATER_THAN_OR_EQUAL)) {
									java.util.regex.Matcher quoteMatcher = quotePattern.matcher(key[2]);
									String newValue = quoteMatcher.replaceAll("''");
									myQueryBuilder.addWhere(MSORT_FIELD + key[3]+ "multilingual_sort('" + key[1] + "','" + newValue + "')");
							}
							else {
								myQueryBuilder.addWhere(valueColumn, key[2],  key[3]);
							}
                            if (offset == 0) {
                                myQueryBuilder.addEndClause(" LIMIT " + limit + " OFFSET " + offset);
                            }

							myQueryBuilder.resetSelectedFields();
							myQueryBuilder.select(entryidColumn);
 							query.getQueryBuilder().addWhereIn(objectidColumn, myQueryBuilder);
						}
                        if (Clauses != null) {
                            com.lutris.dods.builder.generator.query.QueryBuilder clausesQueryBuilder = new com.lutris.dods.builder.generator.query.QueryBuilder(Columns);
                            if (IndexFactory.databaseVendor != null) {
                                clausesQueryBuilder.setDatabaseVendor(IndexFactory.databaseVendor);
                            } else {
                                clausesQueryBuilder.setDatabaseVendor();
                            }
                            for (java.util.Enumeration enumClauses = Clauses.elements(); enumClauses.hasMoreElements();) {
                                String clause = (String) enumClauses.nextElement();
                                clausesQueryBuilder.addWhereOpenParen();
                                clausesQueryBuilder.addWhere(clause);
                                clausesQueryBuilder.addWhereCloseParen();
                            }
                            clausesQueryBuilder.resetSelectedFields();
                            clausesQueryBuilder.select(entryidColumn);
                            query.getQueryBuilder().addWhereIn(objectidColumn, clausesQueryBuilder);
                        }
 					}
				}
				if (order==null || !order.equals(ORDER_DESCENDING)) {
					order = "";
				}				
				query.getQueryBuilder().addOrderByColumn("multilingual_sort('" + sourceLanguage + "',headword)",order);
                query.getQueryBuilder().setMaxRows(limit);
                // seems to be a bug in the queryBuilder, have to put a space before LIMIT or OFFSET
                query.getQueryBuilder().addEndClause(" LIMIT " + limit + " OFFSET " + offset);
				// debug
                // PapillonLogger.writeDebugMsg("getDbTableEntriesVector query debug:");
                //query.getQueryBuilder().debug();
                
				VolumeEntryDO[] DOarray = query.getDOArray();
				if (null != DOarray) {
					// my implementation of the SELECT DISTINCT
					// maybe takes too much time in computation...
					// maybe it isn't needed with a select from where in. To be tested
					/*
                     java.util.List myList = java.util.Arrays.asList(DOarray);
                     for (int j=0; j < DOarray.length; j++) {
                         VolumeEntryDO myTmpDO = DOarray[j];
                         if (j==0 || !myList.subList(0,j-1).contains(myTmpDO)) {
                             VolumeEntry tempEntry = new VolumeEntry(dict, volume,myTmpDO);
                             theEntries.add(tempEntry);
                         }
                     }
					 */
					for (int j=0; j < DOarray.length; j++) {
						VolumeEntry tempEntry = new VolumeEntry(dict, volume, DOarray[j]);
						theEntries.add(tempEntry);
					}
					
				}
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in getDbtableEntriesVector()", ex);
			}
		}
		return theEntries;
	}
	
	public static int getVolumeEntriesCount(Volume theVolume)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			return getVolumeEntriesCount(theVolume, null);
		}
	
	public static int getVolumeEntriesCount(Volume theVolume, String status)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			if (status == null) {
				status = NoStatus;
			}
			Integer count = null;
			java.util.Hashtable volumeTable = (java.util.Hashtable) VolumeEntriesCountHashtable.get(theVolume.getName());
			if (volumeTable != null) {
				count = (Integer) volumeTable.get(status);
				if (count == null) {
					if (status.equals(NoStatus)) {
						count = new Integer(getDbTableEntriesCount(theVolume, null, null, null));
						theVolume.setEntries(count);
						theVolume.save();
					}
					else {
						java.util.Vector Keys = new java.util.Vector();
						String[] statusKey = new String[4];
						statusKey[0] = Volume.CDM_contributionStatus;
						statusKey[1] = Volume.DEFAULT_LANG;
						statusKey[2] = status;
						statusKey[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];					
						Keys.add(statusKey);					
						count = new Integer(getDbTableEntriesCount(theVolume, Keys, null, null));
					}
					volumeTable.put(status, count);
				}
			}
			else {
				volumeTable = new java.util.Hashtable();
				if (status.equals(NoStatus)) {
					count = new Integer(getDbTableEntriesCount(theVolume, null, null, null));
					theVolume.setEntries(count);
					theVolume.save();
				}
				else {
					java.util.Vector Keys = new java.util.Vector();
					String[] statusKey = new String[4];
					statusKey[0] = Volume.CDM_contributionStatus;
					statusKey[1] = Volume.DEFAULT_LANG;
					statusKey[2] = status;
					statusKey[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];					
					Keys.add(statusKey);					
					count = new Integer(getDbTableEntriesCount(theVolume, Keys, null, null));
				}
				volumeTable.put(status, count);
				VolumeEntriesCountHashtable.put(theVolume.getName(),volumeTable);
			}
			return count.intValue();
		}
	
	public static boolean resetCountCache(String volumeName) {
		VolumeEntriesCountHashtable.remove(volumeName);
		return true;
	}
	
	public static void putNbEntriesNumberInCache(String volumeName,int entries) throws PapillonBusinessException {
		java.util.Hashtable volumeTable = (java.util.Hashtable) VolumeEntriesCountHashtable.get(volumeName);
		if (volumeTable == null) {
			volumeTable = new java.util.Hashtable();
			volumeTable.put(NoStatus, new Integer(entries));
			VolumeEntriesCountHashtable.put(volumeName,volumeTable);
		}
		else {
			volumeTable.put(NoStatus, new Integer(entries));
		}
	}
	
    // FIXME: Should the query building code be factorized ?
    public static int getDbTableEntriesCount(Volume volume, Vector Keys, Vector clausesVector, String any) throws PapillonBusinessException {
        int countEntries = 0;
		
		String sourceLanguage = volume.getSourceLanguage();
		String volumeTableName = volume.getDbname();
		if (null != volumeTableName) {
			try {
				com.lutris.dods.builder.generator.query.QueryBuilder myQueryBuilder = null; 
				com.lutris.dods.builder.generator.query.RDBColumn entryidColumn = IndexDO.getEntryIdColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBColumn keyColumn = IndexDO.getKeyColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBColumn langColumn = IndexDO.getLangColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBColumn valueColumn = IndexDO.getValueColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBTable volumeEntryTable = new com.lutris.dods.builder.generator.query.RDBTable(volumeTableName);
				VolumeEntryDO myDO = VolumeEntryDO.createVirgin(volumeTableName);
				com.lutris.dods.builder.generator.query.RDBColumn objectidColumn = new com.lutris.dods.builder.generator.query.RDBColumn(volumeEntryTable, myDO.getOIdColumnName());
				VolumeEntryQuery query = new VolumeEntryQuery(volumeTableName, CurrentDBTransaction.get());
				if (any !=null && !any.equals("")) {
					fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Sequencial count request table: " + volumeTableName + " any: " + any);
					com.lutris.dods.builder.generator.query.RDBColumn xmlcodeColumn = VolumeEntryDO.getXmlCodeColumn(volume.getDbname());
					query.getQueryBuilder().addWhere(xmlcodeColumn, any, QueryBuilder.CASE_SENSITIVE_CONTAINS);
				}
				else {
					fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Index count request table: " + volumeTableName);
				}
				if (Keys !=null) {
					for (java.util.Enumeration enumKeys = Keys.elements(); enumKeys.hasMoreElements();) {
						String[] key = (String[]) enumKeys.nextElement();
						if (key!=null && key[2] !=null && !key[2].equals("")) {
							myQueryBuilder = new com.lutris.dods.builder.generator.query.QueryBuilder(Columns);
							if (IndexFactory.databaseVendor != null) {
								myQueryBuilder.setDatabaseVendor(IndexFactory.databaseVendor);
							} else {
								myQueryBuilder.setDatabaseVendor();
							}
							myQueryBuilder.addWhere(keyColumn, key[0], QueryBuilder.EQUAL);
							if ((key[1] ==null || key[1].equals("")) &&
								volume.isSourceLangCDMElement(key[0])) {
								key[1] = sourceLanguage;
							}
							if (key[1] !=null && !key[1].equals("")) {
								myQueryBuilder.addWhere(langColumn, key[1], QueryBuilder.EQUAL);
							}
							if (key[1] != null && !key[1].equals("") 
								&& (key[3] == QueryBuilder.LESS_THAN ||
									key[3] == QueryBuilder.LESS_THAN_OR_EQUAL ||
									key[3] == QueryBuilder.GREATER_THAN ||
									key[3] == QueryBuilder.GREATER_THAN_OR_EQUAL)) {
									java.util.regex.Matcher quoteMatcher = quotePattern.matcher(key[2]);
									String newValue = quoteMatcher.replaceAll("''");
									myQueryBuilder.addWhere(MSORT_FIELD + key[3]+ "multilingual_sort('" + key[1] + "','" + newValue + "')");
							}
							else {
									myQueryBuilder.addWhere(valueColumn, key[2],  key[3]);
							}
							myQueryBuilder.resetSelectedFields();
							myQueryBuilder.select(entryidColumn);
							query.getQueryBuilder().addWhereIn(objectidColumn, myQueryBuilder);
						}
					}
				}
				// debug
				//query.getQueryBuilder().debug();
				countEntries += query.getCount();
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in getDbtableEntriesCount()", ex);
			}
		}
		return countEntries;
	}
	
    // FIXME: Should the query building code be factorized ?
    protected static void processVolume(Dictionary dict, Volume volume, Vector myKeys, Vector myClauses, IVolumeEntryProcessor myProcessor) throws PapillonBusinessException {
        processVolume(dict, volume, myKeys, myClauses, myProcessor, true);    
    }

    protected static void processVolume(Dictionary dict, Volume volume, Vector myKeys, Vector myClauses, IVolumeEntryProcessor myProcessor, boolean andClauses) throws PapillonBusinessException {
		final int MaxRetrievedEntries = 500;
		int offset = 0;
		boolean stop = false;
		try {
			String volumeTableName = volume.getDbname();
			if (null != volumeTableName) {
				fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Sequencial process request table: " + volumeTableName);
				
				com.lutris.dods.builder.generator.query.QueryBuilder myQueryBuilder = null; 
				com.lutris.dods.builder.generator.query.RDBColumn entryidColumn = IndexDO.getEntryIdColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBColumn keyColumn = IndexDO.getKeyColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBColumn langColumn = IndexDO.getLangColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBColumn valueColumn = IndexDO.getValueColumn(volume.getIndexDbname());
				com.lutris.dods.builder.generator.query.RDBTable volumeEntryTable = new com.lutris.dods.builder.generator.query.RDBTable(volumeTableName);
				VolumeEntryDO myDO = VolumeEntryDO.createVirgin(volumeTableName);
				com.lutris.dods.builder.generator.query.RDBColumn objectidColumn = new com.lutris.dods.builder.generator.query.RDBColumn(volumeEntryTable, myDO.getOIdColumnName());
				
				// consultation of a local volume
				while (!stop) {
					VolumeEntryQuery query = new VolumeEntryQuery(volumeTableName, CurrentDBTransaction.get());
					
					if (myKeys !=null) { 
						for (java.util.Enumeration enumKeys = myKeys.elements(); enumKeys.hasMoreElements();) {
							String[] key = (String[]) enumKeys.nextElement();
							if (key!=null && key[2] !=null && !key[2].equals("")) {
								myQueryBuilder = new com.lutris.dods.builder.generator.query.QueryBuilder(Columns);
								if (IndexFactory.databaseVendor != null) {
									myQueryBuilder.setDatabaseVendor(IndexFactory.databaseVendor);
								} else {
									myQueryBuilder.setDatabaseVendor();
								}
								myQueryBuilder.addWhere(keyColumn, key[0], QueryBuilder.EQUAL);
								if (key[1] !=null && !key[1].equals("")) {
									myQueryBuilder.addWhere(langColumn, key[1], QueryBuilder.EQUAL);
								}
								myQueryBuilder.addWhere(valueColumn, key[2],  key[3]);
								myQueryBuilder.resetSelectedFields();
								myQueryBuilder.select(entryidColumn);
								query.getQueryBuilder().addWhereIn(objectidColumn, myQueryBuilder);
                                if (! andClauses) {
                                    query.getQueryBuilder().addWhereOr();
                                }
                            }
						}
					}
					
					if (myClauses != null) {
						for (java.util.Enumeration enumClauses = myClauses.elements(); enumClauses.hasMoreElements();) {
							String clause = (String) enumClauses.nextElement();
							if (clause!=null && !clause.equals("")) {
								myQueryBuilder = new com.lutris.dods.builder.generator.query.QueryBuilder(Columns);
								if (IndexFactory.databaseVendor != null) {
									myQueryBuilder.setDatabaseVendor(IndexFactory.databaseVendor);
								} else {
									myQueryBuilder.setDatabaseVendor();
								}
								myQueryBuilder.addWhere(clause);
								myQueryBuilder.resetSelectedFields();
								myQueryBuilder.select(entryidColumn);
								query.getQueryBuilder().addWhereIn(objectidColumn, myQueryBuilder);
                                if (! andClauses) {
                                    query.getQueryBuilder().addWhereOr();
                                }
                            }
						}
					}
					
					query.getQueryBuilder().setMaxRows(MaxRetrievedEntries);
					query.getQueryBuilder().addEndClause("OFFSET " + offset);
					//query.getQueryBuilder().distinct();
					query.getQueryBuilder().addOrderByColumn("multilingual_sort('" + volume.getSourceLanguage() + "',headword)","");

                    //debug
                     //System.out.println(query.getQueryBuilder().getSQLwithParms()); // System.exit(2);

                    VolumeEntryDO[] DOarray = query.getDOArray();
					if (null != DOarray) {
						for (int j=0; j < DOarray.length; j++) {
							myProcessor.process(new VolumeEntry(dict, volume,DOarray[j]));
						}
						offset += DOarray.length;
					}
					stop = DOarray == null || DOarray.length<MaxRetrievedEntries;
				}
			}
		}
		catch(Exception ex) {
			throw new PapillonBusinessException("Exception in processVolume()", ex);
		}
	}
	
	public static void changeAuthor(Volume myVolume, User validator, String newAuthorLogin, Vector myKeys, Vector clausesVector) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			
			if (myVolume !=null && !myVolume.isEmpty()) {
				Dictionary myDict = DictionariesFactory.getDictionaryByName(myVolume.getDictname());
				if (myDict !=null && !myDict.isEmpty()) {
					User authorUser = UsersFactory.findUserByLogin(newAuthorLogin);
					if (authorUser == null || authorUser.isEmpty()) {
						authorUser = new User();
						authorUser.setLogin(newAuthorLogin);
					}
					IVolumeEntryProcessor changeAuthorProcessor = new ChangeAuthorProcessor(validator, authorUser);
					processVolume(myDict, myVolume, myKeys, clausesVector, changeAuthorProcessor);
				}
			}
		}
    
    public static void exportVolume(String volumeName, Vector myKeys, Vector clauseVector, String outputFormat, java.io.OutputStream myOutStream)
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
        exportVolume(volumeName, myKeys, clauseVector, true, outputFormat, myOutStream);
    }

    public static void exportVolume(String volumeName, Vector myKeys, Vector clauseVector, boolean andClause, String outputFormat, java.io.OutputStream myOutStream)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			
			Volume myVolume = VolumesFactory.getVolumeByName(volumeName);
			if (myVolume !=null && !myVolume.isEmpty()) {
				Dictionary myDict = DictionariesFactory.getDictionaryByName(myVolume.getDictname());
				if (myDict !=null && !myDict.isEmpty()) {
					
					String tmplEntry = myVolume.getTemplateEntry();
					String contribString = myVolume.getCdmContribution();
					String xmlHeader = "";
					String xmlFooter = "";
					if (tmplEntry.indexOf("<" + contribString)>=0) {
						xmlHeader = tmplEntry.substring(0,tmplEntry.indexOf("<" + contribString));
						String endTag = "</" + contribString + ">";
						xmlFooter = tmplEntry.substring(tmplEntry.indexOf(endTag)+endTag.length()+1);
					}
                    
					PapillonLogger.writeDebugMsg("Start writing volume " + volumeName);
					fr.imag.clips.papillon.business.transformation.FOProcessor myFOProcessor = null;
					
					try {
						if (outputFormat != null && outputFormat.equals(XHTMLFormat)) {
							myOutStream.write(XhtmlHeader.getBytes("UTF-8"));
						}
						else if (outputFormat != null && outputFormat.equals(TEXTFormat)) {
							myOutStream.write(TextHeader.getBytes("UTF-8"));
						}
						else if (outputFormat != null && outputFormat.equals(PDFFormat)) {
							myFOProcessor = new fr.imag.clips.papillon.business.transformation.FOProcessor(myOutStream);
							myOutStream = myFOProcessor.getOutputStreamAsInput();
							myOutStream.write(FoHeader.getBytes("UTF-8"));
						}
						else {
							myOutStream.write(xmlHeader.getBytes("UTF-8"));
						}
                        
						fr.imag.clips.papillon.business.dictionary.IVolumeEntryProcessor myProcessor = new fr.imag.clips.papillon.business.dictionary.ExportVolumeEntryProcessor(outputFormat, myOutStream);
						PapillonLogger.writeDebugMsg("Processor created");
                        
						fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory.processVolume(myDict, myVolume, myKeys, clauseVector, myProcessor, andClause);
						PapillonLogger.writeDebugMsg("Volume processed");
						
						
						if (outputFormat != null && outputFormat.equals(XHTMLFormat)) {
							myOutStream.write(XhtmlFooter.getBytes("UTF-8"));
						}
						else if (outputFormat != null && outputFormat.equals(TEXTFormat)) {
							myOutStream.write(TextFooter.getBytes("UTF-8"));
						}
						else if (outputFormat != null && outputFormat.equals(PDFFormat)) {
							myOutStream.write(FoFooter.getBytes("UTF-8"));
							PapillonLogger.writeDebugMsg("Processing Formating Object to PDF");
							myFOProcessor.renderOutputStream();
						}
						else {
							myOutStream.write(xmlFooter.getBytes("UTF-8"));
						}
					}
					catch (Exception ex) {
						throw new PapillonBusinessException("Error in writing an UTF-8 String: ", ex);
					}
				}
			}
		}

	public static void exportEntryList(String volumeName, String[] Headwords, String outputFormat, java.io.OutputStream myOutStream)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		
		Volume myVolume = VolumesFactory.getVolumeByName(volumeName);
		if (myVolume !=null && !myVolume.isEmpty()) {
			Dictionary myDict = DictionariesFactory.getDictionaryByName(myVolume.getDictname());
			if (myDict !=null && !myDict.isEmpty()) {
				
				String tmplEntry = myVolume.getTemplateEntry();
				String contribString = myVolume.getCdmContribution();
				String xmlHeader = "";
				String xmlFooter = "";
				if (tmplEntry.indexOf("<" + contribString)>=0) {
					xmlHeader = tmplEntry.substring(0,tmplEntry.indexOf("<" + contribString));
					String endTag = "</" + contribString + ">";
					xmlFooter = tmplEntry.substring(tmplEntry.indexOf(endTag)+endTag.length()+1);
				}
				
				PapillonLogger.writeDebugMsg("Start writing volume " + volumeName);
				fr.imag.clips.papillon.business.transformation.FOProcessor myFOProcessor = null;
				
				try {
					if (outputFormat != null && outputFormat.equals(XHTMLFormat)) {
						myOutStream.write(XhtmlHeader.getBytes("UTF-8"));
					}
					else if (outputFormat != null && outputFormat.equals(TEXTFormat)) {
						myOutStream.write(TextHeader.getBytes("UTF-8"));
					}
					else if (outputFormat != null && outputFormat.equals(PDFFormat)) {
						myFOProcessor = new fr.imag.clips.papillon.business.transformation.FOProcessor(myOutStream);
						myOutStream = myFOProcessor.getOutputStreamAsInput();
						myOutStream.write(FoHeader.getBytes("UTF-8"));
					}
					else {
						myOutStream.write(xmlHeader.getBytes("UTF-8"));
					}
					
					fr.imag.clips.papillon.business.dictionary.IVolumeEntryProcessor myProcessor = new fr.imag.clips.papillon.business.dictionary.ExportVolumeEntryProcessor(outputFormat, myOutStream);
					PapillonLogger.writeDebugMsg("Processor created");
					
					for (int i=0; i<Headwords.length; i++) {
						String word = Headwords[i];
						Vector myKeys = new Vector();
						String[] key1 = new String[4];
						key1[0] = Volume.CDM_headword;
						key1[2] = word;
						key1[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];
						myKeys.add(key1);
						fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory.processVolume(myDict, myVolume, myKeys, null, myProcessor, true);
					}
					
					PapillonLogger.writeDebugMsg("Volume processed");
					
					
					if (outputFormat != null && outputFormat.equals(XHTMLFormat)) {
						myOutStream.write(XhtmlFooter.getBytes("UTF-8"));
					}
					else if (outputFormat != null && outputFormat.equals(TEXTFormat)) {
						myOutStream.write(TextFooter.getBytes("UTF-8"));
					}
					else if (outputFormat != null && outputFormat.equals(PDFFormat)) {
						myOutStream.write(FoFooter.getBytes("UTF-8"));
						PapillonLogger.writeDebugMsg("Processing Formating Object to PDF");
						myFOProcessor.renderOutputStream();
					}
					else {
						myOutStream.write(xmlFooter.getBytes("UTF-8"));
					}
				}
				catch (Exception ex) {
					throw new PapillonBusinessException("Error in writing an UTF-8 String: ", ex);
				}
			}
		}
	}
	
	
	public static void convertVolume(String volumeName, Vector myKeys, Vector clauseVector, String stylesheetHandle) throws PapillonBusinessException {
		
        /*
		Volume myVolume = VolumesFactory.getVolumeByName(volumeName);
		if (myVolume !=null && !myVolume.isEmpty()) {
			Dictionary myDict = DictionariesFactory.getDictionaryByName(myVolume.getDictname());
			if (myDict !=null && !myDict.isEmpty()) {
				fr.imag.clips.papillon.business.dictionary.IVolumeEntryProcessor myProcessor = new fr.imag.clips.papillon.business.dictionary.ConvertVolumeEntryProcessor(stylesheetHandle);
				PapillonLogger.writeDebugMsg("Processor created");
				fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory.processVolume(myDict, myVolume, myKeys, clauseVector, myProcessor);
				PapillonLogger.writeDebugMsg("Volume processed");
			}
		}
         */
        Volume myVolume = VolumesFactory.getVolumeByName(volumeName);
		if (myVolume !=null && !myVolume.isEmpty()) {
			Dictionary myDict = DictionariesFactory.getDictionaryByName(myVolume.getDictname());
			if (myDict !=null && !myDict.isEmpty()) {
				fr.imag.clips.papillon.business.dictionary.IVolumeEntryProcessor myProcessor = new fr.imag.clips.papillon.business.dictionary.ConvertVolumeEntryProcessor(stylesheetHandle, myVolume.getDictname(), volumeName);
				PapillonLogger.writeDebugMsg("Processor created");
				fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory.processVolume(myDict, myVolume, myKeys, clauseVector, myProcessor);
				PapillonLogger.writeDebugMsg("Volume processed");
			}
		}
	}
	
	
	public static Vector getFoksEntriesVector(String headword) throws PapillonBusinessException {
        Vector theEntries = new Vector();
        try {
			// consultation of a local volume
			FoksEntryQuery query = new FoksEntryQuery(CurrentDBTransaction.get());
			//set query
			if (headword != null && !headword.equals("")) {
				fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Indexed request table: foksentry word: " + headword);
				query.setQueryReading(headword);
				query.addOrderByScore(false);
				FoksEntryDO[] DOarray = query.getDOArray();
				if (null != DOarray) {
					for (int i=0; i < DOarray.length; i++) {
						theEntries.add(new FoksEntry(DOarray[i]));
					}
				}
			}
		}
		catch(Exception ex) {
			throw new PapillonBusinessException("Exception in getFoksEntriesVector()", ex);
		}
		return theEntries;
	}
	
	
	public static VolumeEntry getJMDictVolumeEntry(String headword) throws PapillonBusinessException {
		// FIXME: special code depending on FoksEdict and JmDict dictionaries
        VolumeEntry myAnswer = null;
		Vector theEntries = null;
		//Headword[0] = key
		//Headword[1] = lang
		//Headword[2] = value
		String[] Headword = new String[4];
		Headword[0] = Volume.CDM_headword;
		Headword[1] = "jpn";
		Headword[2] = headword;
		Headword[3] = "" + IQuery.STRATEGY_EXACT;
		Vector myVector = new Vector();
		myVector.add(Headword);
        try {
			Volume volume = VolumesFactory.getVolumeByName("JMDict_jpn_eng");
			if (volume != null && !volume.isEmpty()) {
				Dictionary myDict = DictionariesFactory.getDictionaryByName(volume.getDictname());
				theEntries = IndexFactory.getEntriesVector(myDict, volume, myVector, null,0);
			}
		}
		catch(Exception ex) {
			throw new PapillonBusinessException("Exception in getJMDictEntriesVector()", ex);
		}
		if (theEntries !=null && theEntries.size()>0) {
			myAnswer = (VolumeEntry) theEntries.elements().nextElement();
		}
		return myAnswer;
	}
	
	
    /**
        * The findEntryByHandle method performs a database query to
     * return a id object
     *
     * @param id, the object id of the entries table.
     * @return
     *     the id
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
    public static VolumeEntry findEntryByHandle(String volumeName, String handle)
        throws PapillonBusinessException {
			Volume volume;
            Dictionary dict;
            try {
                volume = VolumesFactory.getVolumeByName(volumeName);
                dict = DictionariesFactory.getDictionaryByName(volume.getDictname());
            }
            catch(Exception ex) {
				return null;
            }
			return findEntryByHandle(dict, volume, handle);
		}
	
	protected static VolumeEntry findEntryByHandle(Dictionary myDict, Volume myVolume, String handle)
        throws PapillonBusinessException {
			VolumeEntry theEntry = null;
			VolumeEntryDO theVolumeEntryDO = null;
			
			
			int intId = 0;
			try {
				intId = Integer.parseInt(handle);
			}
			catch(NumberFormatException ex) {
				return theEntry;
			}												
			
            try {
                VolumeEntryQuery query = new VolumeEntryQuery(myVolume.getDbname(), CurrentDBTransaction.get());
                //set query
                query.setQueryOId(new ObjectId(intId));
                // Throw an exception if more than one message is found
                query.requireUniqueInstance();
                theVolumeEntryDO = query.getNextDO();
                theEntry = new VolumeEntry(myDict, myVolume,theVolumeEntryDO);
                // Put entry in cache
                // Add the volume entry in the request context.
                CurrentRequestContext.get().set(theEntry.getEntryId(), theEntry);
                CurrentRequestContext.get().set(theEntry.getContributionId(), theEntry);
             //   EntryCache.putEntryInCache(theEntry);

                return theEntry;
            } catch(Exception ex) {
				return theEntry;
            }
        }
	
    /**
		* The findEntryByKey method performs a database query to
     * return a VolumeEntry
     *
     * @param Dictionary, Volume, key, lang, value
     * @return the corresponding VolumeEntry
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
    // FIXME: Shouldn't we use: getDbTableEntriesVector ?
	protected static VolumeEntry findEntryByKey(Dictionary myDict, Volume myVolume, String key, String lang, String value) throws PapillonBusinessException {
		VolumeEntry resEntry = null;
		if (value != null && !value.equals("")) {
			try {
				IndexQuery query = new IndexQuery(myVolume.getIndexDbname(), CurrentDBTransaction.get());
				query.getQueryBuilder().addWhereClause("key", key, QueryBuilder.EQUAL);
				if (lang != null) {
					query.getQueryBuilder().addWhereClause("lang", lang, QueryBuilder.EQUAL);
				}
				query.getQueryBuilder().addWhereClause("value", value, QueryBuilder.EQUAL);
				query.getQueryBuilder().setMaxRows(1);
				IndexDO[] DOarray = query.getDOArray();
				if (null != DOarray && DOarray.length>0) {
					Index myIndex = new Index(DOarray[0]);
					resEntry = findEntryByHandle(myDict, myVolume, ""+myIndex.getEntryId());
				}
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in findEntryByKey()", ex);
			}
		}
		return resEntry;
	}
	
    /**
		* The findEntryByEntryId method performs a database query to
     * return a VolumeEntry
     *
     * @param id, the object id of the entries table.
     * @return the corresponding VolumeEntry
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
    public static VolumeEntry findEntryByEntryId(String volumeName, String entryId)
        throws PapillonBusinessException {
			Volume volume;
            Dictionary dict;
            try {
                volume = VolumesFactory.getVolumeByName(volumeName);
                dict = DictionariesFactory.getDictionaryByName(volume.getDictname());
            }
            catch(Exception ex) {
				return null;
            }
			return findEntryByEntryId(dict, volume, entryId);
		}
	
    
    //
	protected static VolumeEntry findEntryByEntryId(Dictionary myDict, Volume myVolume, String entryId)
        throws PapillonBusinessException {
        // FIXME: should use queries as used in findEntryByEntryId(User user, String entryid)
        // FIXME: Moreover, this is duplicate code...
        //PapillonLogger.writeDebugMsg("Looking for " + entryId + " in volume " + myVolume.getName());


        VolumeEntry resultEntry = (VolumeEntry) CurrentRequestContext.get().get(entryId);
        if (null == resultEntry && null != entryId && !entryId.equals("")) {


            Vector answersVector = null;

            Vector myKeys = new Vector();
            /*			String[] statusReplaced = new String[4];
                   statusReplaced[0] = Volume.CDM_contributionStatus;
                   statusReplaced[1] = Volume.DEFAULT_LANG;
                   statusReplaced[2] = VolumeEntry.REPLACED_STATUS;
                   statusReplaced[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];
                   myKeys.add(statusReplaced);

                   String[] statusDeleted = new String[4];
                   statusDeleted[0] = Volume.CDM_contributionStatus;
                   statusDeleted[1] = Volume.DEFAULT_LANG;
                   statusDeleted[2] = VolumeEntry.DELETED_STATUS;
                   statusDeleted[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];
                   myKeys.add(statusDeleted); */

            String[] entryIdArray = new String[4];
            entryIdArray[0] = Volume.CDM_entryId;
            entryIdArray[1] = Volume.DEFAULT_LANG;
            entryIdArray[2] = entryId;
            entryIdArray[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT + 1];
            myKeys.add(entryIdArray);

            answersVector = getDbTableEntriesVector(myDict, myVolume, myKeys, null, null, 0, 0);

            if (answersVector.size() > 0) {
                if (answersVector.size() == 1) {
                    resultEntry = (VolumeEntry) answersVector.firstElement();
                } else if (answersVector.size() > 1) {
                    VolumeEntry tempEntry = null;
                    VolumeEntry reviewedEntry = null;
                    VolumeEntry finishedEntry = null;
                    VolumeEntry notFinishedEntry = null;
                    VolumeEntry modifiedEntry = null;
                    for (java.util.Enumeration enumEntries = answersVector.elements(); enumEntries.hasMoreElements();) {
                        tempEntry = (VolumeEntry) enumEntries.nextElement();
                        if (tempEntry.getStatus().equals(VolumeEntry.VALIDATED_STATUS)) {
                            resultEntry = tempEntry;
                        }
                        if (tempEntry.getStatus().equals(VolumeEntry.REVIEWED_STATUS)) {
                            reviewedEntry = tempEntry;
                        }
                        if (tempEntry.getStatus().equals(VolumeEntry.FINISHED_STATUS)) {
                            finishedEntry = tempEntry;
                        }
                        // add by Francis to use correctly expandResult !
                        if (tempEntry.getStatus().equals(VolumeEntry.MODIFIED_STATUS)) {
                            modifiedEntry = tempEntry;
                        }
                        if (tempEntry.getStatus().equals(VolumeEntry.NOT_FINISHED_STATUS)) {
                            notFinishedEntry = tempEntry;
                        }
                    }
                    if (resultEntry == null) {
                        if (reviewedEntry != null) {
                            resultEntry = reviewedEntry;
                        } else if (finishedEntry != null) {
                            resultEntry = finishedEntry;
                        } else if (modifiedEntry != null) {
                            resultEntry = modifiedEntry;
                        } else if (notFinishedEntry != null) {
                            resultEntry = notFinishedEntry;
                        }
                    }
                }
            }
    //        CurrentRequestContext.get().set(entryId, resultEntry);
            if (resultEntry!=null) {
                CurrentRequestContext.get().set(resultEntry.getContributionId(), resultEntry);
 //               PapillonLogger.writeDebugMsg("findEntryByEntryId: " + resultEntry.getXmlCode());
            }
            else {
 //               PapillonLogger.writeDebugMsg("findEntryByEntryId: resultentry null");
            }
        } else {
            //PapillonLogger.writeDebugMsg("Found it in request context.");
        }

        //if (resultEntry != null) {
        //	PapillonLogger.writeDebugMsg("findEntryByEntryId selected entry: " + resultEntry.getHeadword() + " status: " + resultEntry.getStatus());
        //}
        return resultEntry;
    }

	/**
	 * The findPreviousEntryByHeadword method performs a database query to
     * return a VolumeEntry
     *
     * @param id, the object id of the entries table.
     * @return the corresponding VolumeEntry
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
	//select value from idxdelafra where key='cdm-headword' and msort<multilingual_sort('fra','essai') order by msort desc limit 2;
	
    public static VolumeEntry findPreviousEntryByHeadword(String volumeName, String headword)
	throws PapillonBusinessException {
		Volume volume;
		Dictionary dict;
		try {
			volume = VolumesFactory.getVolumeByName(volumeName);
			dict = DictionariesFactory.getDictionaryByName(volume.getDictname());
		}
		catch(Exception ex) {
			return null;
		}
		//PapillonLogger.writeDebugMsg("findPreviousIndexEntriesByHeadword: " + headword + " " + entryId);
		Vector theEntries = findPreviousIndexEntriesByHeadword(volume.getIndexDbname(), volume.getSourceLanguage() , headword);
		if (theEntries != null && theEntries.size()>0) {
			int entryId = ((Index)theEntries.firstElement()).getEntryId();
			return findEntryByHandle(dict, volume, ""+entryId);
		}
		else {
			return null;
		}
	}
	
	protected static Vector findPreviousIndexEntriesByHeadword(String indexTableName, String sourceLanguage, String headword) throws PapillonBusinessException {
		Vector theEntries = new Vector();
		if (null != indexTableName) {
			try {
				com.lutris.dods.builder.generator.query.RDBColumn keyColumn = IndexDO.getKeyColumn(indexTableName);
				com.lutris.dods.builder.generator.query.RDBColumn langColumn = IndexDO.getLangColumn(indexTableName);
				com.lutris.dods.builder.generator.query.RDBColumn valueColumn = IndexDO.getValueColumn(indexTableName);
				IndexQuery query = new IndexQuery(indexTableName, CurrentDBTransaction.get());
				//fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Index request table: " + indexTableName);
				
				query.getQueryBuilder().addWhere(keyColumn, Volume.CDM_headword, QueryBuilder.EQUAL);
				query.getQueryBuilder().addWhere(MSORT_FIELD + " < " + "multilingual_sort('" + sourceLanguage + "','" + headword + "')");
				query.getQueryBuilder().setMaxRows(10);
				query.getQueryBuilder().addOrderByColumn(MSORT_FIELD,ORDER_DESCENDING);
				// debug
				//query.getQueryBuilder().debug();
				
				IndexDO[] DOarray = query.getDOArray();
				if (null != DOarray) {
					for (int j=0; j < DOarray.length; j++) {
						Index tempIndex = new Index(DOarray[j]);
						theEntries.add(tempIndex);
					}
				}
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in getIndexEntriesVector()", ex);
			}
		}
		return theEntries;
	}
	
	
    /**
	 * The findNextEntryByHeadword method performs a database query to
     * return a VolumeEntry
     *
     * @param id, the object id of the entries table.
     * @return the corresponding VolumeEntry
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
	//select value from idxdelafra where key='cdm-headword' and msort>multilingual_sort('fra','essai') order by msort limit 2;

    public static VolumeEntry findNextEntryByHeadword(String volumeName, String headword)
	throws PapillonBusinessException {
		Volume volume;
		Dictionary dict;
		try {
			volume = VolumesFactory.getVolumeByName(volumeName);
			dict = DictionariesFactory.getDictionaryByName(volume.getDictname());
		}
		catch(Exception ex) {
			return null;
		}
		//PapillonLogger.writeDebugMsg("findNextIndexEntriesByHeadword: " + headword + " " + entryId);
		Vector theEntries = findNextIndexEntriesByHeadword(volume.getIndexDbname(), volume.getSourceLanguage() , headword);
		if (theEntries != null && theEntries.size()>0) {
			int entryId = ((Index)theEntries.firstElement()).getEntryId();
			return findEntryByHandle(dict, volume, ""+entryId);
		}
		else {
			return null;
		}
	}
	
	protected static Vector findNextIndexEntriesByHeadword(String indexTableName, String sourceLanguage, String headword) throws PapillonBusinessException {
			Vector theEntries = new Vector();
			if (null != indexTableName) {
				try {
					com.lutris.dods.builder.generator.query.RDBColumn keyColumn = IndexDO.getKeyColumn(indexTableName);
					com.lutris.dods.builder.generator.query.RDBColumn langColumn = IndexDO.getLangColumn(indexTableName);
					com.lutris.dods.builder.generator.query.RDBColumn valueColumn = IndexDO.getValueColumn(indexTableName);
					IndexQuery query = new IndexQuery(indexTableName, CurrentDBTransaction.get());
					//fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Index request table: " + indexTableName);
					
					query.getQueryBuilder().addWhere(keyColumn, Volume.CDM_headword, QueryBuilder.EQUAL);
					query.getQueryBuilder().addWhere(MSORT_FIELD + " > " + "multilingual_sort('" + sourceLanguage + "','" + headword + "')");
					query.getQueryBuilder().setMaxRows(10);
					//String order = if (order==null || !order.equals(ORDER_DESCENDING)) {
					query.getQueryBuilder().addOrderByColumn(MSORT_FIELD,"");
					// debug
					//query.getQueryBuilder().debug();
					
					IndexDO[] DOarray = query.getDOArray();
					if (null != DOarray) {
						for (int j=0; j < DOarray.length; j++) {
							Index tempIndex = new Index(DOarray[j]);
							theEntries.add(tempIndex);
						}
					}
				}
				catch(Exception ex) {
					throw new PapillonBusinessException("Exception in getIndexEntriesVector()", ex);
				}
			}
			return theEntries;
		}
			
	
	
/**
* Find Entry by entry Id
 *
 *
 */
public static VolumeEntry findEntryByEntryId(User user, String entryId) 
throws PapillonBusinessException {
    return findEntryByEntryId(user, VolumesFactory.getVolumesArray(), entryId);
}

/**
* Find Entry by entry Id
 *
 *
 */
public static VolumeEntry findEntryByEntryId(User user, Volume volume, String entryId) 
throws PapillonBusinessException {
    ArrayList volumes = new ArrayList();
    volumes.add(volume);
    return findEntryByEntryId(user, volumes, entryId);
}

/**
* Find Entry by entry Id
 *
 *
 */
public static VolumeEntry findEntryByEntryId(User user, Collection volumes, String entryId)
throws PapillonBusinessException {
    //PapillonLogger.writeDebugMsg("Looking for " + entryId + " in " + volumes.size() + " volumes for user " + user );
    //FIXME: an entry id may not be unique externally to a dictionary so the dict must be specified
    VolumeEntry resultEntry = (VolumeEntry) CurrentRequestContext.get().get(entryId);

    if (null == resultEntry && entryId != null && !entryId.equals("")) {
		
        // FIXME ... Add dictionary in QueryRequest class
        QueryRequest queryReq = new QueryRequest(volumes);
        
        // Entry Id
        QueryCriteria criteriaSearch = new QueryCriteria();
        criteriaSearch.add("key", QueryCriteria.EQUAL, Volume.CDM_entryId);  
        criteriaSearch.add("value", QueryCriteria.EQUAL, entryId);
        queryReq.addCriteria(criteriaSearch);
        
        //
        ArrayList qrset = queryReq.findLexie(user);
        
        //
        if (qrset.size() == 1) {
            resultEntry = ((QueryResult) qrset.get(0)).getSourceEntry();
        } else if (qrset.size() < 1) {
            PapillonLogger.writeDebugMsg("Error, 0 entry found: " + entryId);
        } else if (qrset.size() > 1) {
            resultEntry = ((QueryResult) qrset.get(0)).getSourceEntry();
            PapillonLogger.writeDebugMsg("Error, too many entries found: " + entryId);
        }

        CurrentRequestContext.get().set(entryId, resultEntry);            
    } else {
        //PapillonLogger.writeDebugMsg("Found it in request context.");
    }
    
    //FIXME: exception when Qrest < 1 or > 1
    
    return resultEntry;
}

    /**
     * Find Entry by contribution Id
     *
     *
     */
    public static VolumeEntry findEntryByContributionId(User user, String entryId)
    throws PapillonBusinessException {
        return findEntryByContributionId(user, VolumesFactory.getVolumesArray(), entryId);
    }

    protected static VolumeEntry findEntryByContributionId(User user, Collection volumes, String entryId)
    throws PapillonBusinessException {
        //PapillonLogger.writeDebugMsg("findEntryByContributionId " + entryId + " in " + volumes.size() + " volumes");
        //FIXME: an entry id may not be unique externally to a dictionary so the dict must be specified
        VolumeEntry resultEntry = (VolumeEntry) CurrentRequestContext.get().get(entryId);
        
        if (null == resultEntry && entryId != null && !entryId.equals("")) {
            
            // FIXME ... Add dictionary in QueryRequest class
            QueryRequest queryReq = new QueryRequest(volumes);
            
            // Entry Id
            QueryCriteria criteriaSearch = new QueryCriteria();
            criteriaSearch.add("key", QueryCriteria.EQUAL, Volume.CDM_contributionId);
            criteriaSearch.add("value", QueryCriteria.EQUAL, entryId);
            queryReq.addCriteria(criteriaSearch);
            
            //
            ArrayList qrset = queryReq.findLexie(user);
            
            //
            if (qrset.size() == 1) {
                resultEntry = ((QueryResult) qrset.get(0)).getSourceEntry();
            } else if (qrset.size() < 1) {
                PapillonLogger.writeDebugMsg("Error, 0 entry found: " + entryId);
            } else if (qrset.size() > 1) {
                resultEntry = ((QueryResult) qrset.get(0)).getSourceEntry();
                PapillonLogger.writeDebugMsg("Error, too many entries found: " + entryId);
            }
            
            CurrentRequestContext.get().set(entryId, resultEntry);
        } else {
          //  PapillonLogger.writeDebugMsg("Found it in request context.");
        }
        return resultEntry;
    }



/**
* The findEntryByContributionId method performs a database query to
 * return a VolumeEntry
 *
 * @param id, the object id of the entries table.
 * @return the corresponding VolumeEntry
 * @exception PapillonBusinessException
 *    if there is a problem retrieving message.
 */
public static VolumeEntry findEntryByContributionId(String volumeName, String entryId)
throws PapillonBusinessException {
    Volume volume;
    Dictionary dict;
    try {
        volume = VolumesFactory.getVolumeByName(volumeName);
        dict = DictionariesFactory.getDictionaryByName(volume.getDictname());
    }
    catch(Exception ex) {
        return null;
    }
    return findEntryByContributionId(dict, volume, entryId);
}

public static VolumeEntry findEntryByContributionId(Dictionary myDict, Volume myVolume, String entryId)
throws PapillonBusinessException {
    VolumeEntry resultEntry = (VolumeEntry) CurrentRequestContext.get().get(entryId);
    
    if (null == resultEntry && entryId != null && !entryId.equals("")) {
        resultEntry = findEntryByKey(myDict, myVolume, Volume.CDM_contributionId, Volume.DEFAULT_LANG, entryId);
    }
    return resultEntry;
}
    
    /**
     * The findEntriesByOriginalContributionId method performs a database query to
     * return a collection of VolumeEntries
     *
     * @param id, the object id of the entries table.
     * @return the corresponding VolumeEntry
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */


public static java.util.Vector findEntriesByOriginalContributionId(VolumeEntry existingEntry)
throws fr.imag.clips.papillon.business.PapillonBusinessException {
    
    java.util.Vector myKeys = new java.util.Vector();
    String[] origContribId = new String[4];
    origContribId[0] = Volume.CDM_originalContributionId;
    origContribId[1] = Volume.DEFAULT_LANG;
    origContribId[2] = existingEntry.getOriginalContributionId();
    origContribId[3] = QueryBuilder.EQUAL;
    myKeys.add(origContribId);
    
    String[] StatusNotFinished = new String[4];
    StatusNotFinished[0] = Volume.CDM_contributionStatus;
    StatusNotFinished[1] = Volume.DEFAULT_LANG;
    StatusNotFinished[2] = VolumeEntry.NOT_FINISHED_STATUS;
    StatusNotFinished[3] = QueryBuilder.NOT_EQUAL;
    myKeys.add(StatusNotFinished);
    String[] StatusDraft = (String[]) StatusNotFinished.clone();
    StatusDraft[2] = VolumeEntry.DRAFT_STATUS;
    myKeys.add(StatusDraft);
    
    return getDbTableEntriesVector(existingEntry.getDictionary(), existingEntry.getVolume(), myKeys, null, "", "", 0, 0);
    
  }
  
  public static VolumeEntry newEntryFromExisting(VolumeEntry existingEntry) 
throws fr.imag.clips.papillon.business.PapillonBusinessException {
    VolumeEntry resEntry = new VolumeEntry(existingEntry.getDictionary(), existingEntry.getVolume());
    //dom avant toute chose !
    resEntry.setDom((org.w3c.dom.Document) existingEntry.getDom().cloneNode(true));
    //headword
    resEntry.setHeadword(existingEntry.getHeadword());
    
    resEntry.setEntryIdIfNull();
    // FIXME: add by Francis
    resEntry.setContributionId();
    
    return resEntry;
}


public static VolumeEntry createEmptyEntry(String volume)
throws fr.imag.clips.papillon.business.PapillonBusinessException {
    VolumeEntry myEntry = null;
    Volume myVolume = VolumesFactory.getVolumeByName(volume);
    if (myVolume != null && !myVolume.isEmpty()) {
        Dictionary myDict = DictionariesFactory.getDictionaryByName(myVolume.getDictname());
        if (myDict != null && !myDict.isEmpty()) {
            myEntry = new VolumeEntry(myDict, myVolume);
            String templateEntry = myVolume.getTemplateEntry();
            myEntry.setDom(XMLServices.buildDOMTree(templateEntry));
        }
    }
    return myEntry;
}

public static void emptyVolumeEntries(String volume)
throws fr.imag.clips.papillon.business.PapillonBusinessException {
    try {
        //FIXME: no truncate index volume ???
        ManageDatabase.truncateTable(volume);
    }
    catch (Exception e) {
        throw new fr.imag.clips.papillon.business.PapillonBusinessException ("Exception in emptyVolumeEntries with volume: " + volume, e);
    }
}


public static void createVolumeTables(Volume volume)
throws fr.imag.clips.papillon.business.PapillonBusinessException {
	/* Added a sorted index on headword field necessary for order by statements */
	try {
        ManageDatabase.createVolumeTables(volume.getDbname(), volume.getSourceLanguage(), volume.getIndexDbname(), volume.getLinkDbname());
	}
	catch (Exception e) {
		throw new fr.imag.clips.papillon.business.PapillonBusinessException ("Exception in createVolumeTables with volume: " + volume.getName(), e);
		//PapillonLogger.writeDebugMsg("createVolumeTables with volume: " + volume.getName() + ", probably the tables already exist.");
	}
}


public static void dropVolumeTables(Volume volume)
throws fr.imag.clips.papillon.business.PapillonBusinessException {
	try {
		ManageDatabase.dropVolumeTables(volume.getDbname(), volume.getIndexDbname(), volume.getLinkDbname());
		//truncateVolumeTables(volume);
	}
	catch (Exception e) {
		throw new fr.imag.clips.papillon.business.PapillonBusinessException ("Exception in dropVolumeTables with volume: " + volume.getName(), e);
		//PapillonLogger.writeDebugMsg("Exception in dropVolumeTables with volume: " + volume.getName() + ", probably the tables were already deleted.");
	}
}

public static void truncateVolumeTables(Volume volume)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		try {
			java.util.Vector TableNames = ManageDatabase.getTableNames();
			if (TableNames.contains(volume.getDbname())) {				
				ManageDatabase.truncateTable(volume.getDbname());
			}
			if (TableNames.contains(volume.getIndexDbname())) {
				ManageDatabase.truncateTable(volume.getIndexDbname());
			}
			if (TableNames.contains(volume.getLinkDbname())) {
				ManageDatabase.truncateTable(volume.getLinkDbname());
			}
		}
		catch (Exception e) {
			throw new fr.imag.clips.papillon.business.PapillonBusinessException ("Exception in truncateVolumeTables with volume: " + volume.getName(), e);
			//PapillonLogger.writeDebugMsg("Exception in dropVolumeTables with volume: " + volume.getName() + ", probably the tables were already deleted.");
		}
	}
	
	
	
	
public static void sort (Vector EntryVector, String sortBy) {
    if (sortBy != null && !sortBy.equals("")) {
        if (sortBy.equals(AUTHOR_SORT)) {
            Collections.sort(EntryVector, new fr.imag.clips.papillon.business.utility.AuthorComparator());
        }
        else if (sortBy.equals(CREATION_DATE_SORT)) {
            Collections.sort(EntryVector, new fr.imag.clips.papillon.business.utility.CreationDateComparator());
        }
        else if (sortBy.equals(HEADWORD_SORT)) {
            Collections.sort(EntryVector, new fr.imag.clips.papillon.business.utility.HeadwordComparator());
        }
        else if (sortBy.equals(ORIGINAL_CONTRIBUTION_ID_SORT)) {
            Collections.sort(EntryVector, new fr.imag.clips.papillon.business.utility.OriginalContributionIdComparator());
        }
        else if (sortBy.equals(REVIEW_DATE_SORT)) {
            Collections.sort(EntryVector, new fr.imag.clips.papillon.business.utility.ReviewDateComparator());
        }
        else if (sortBy.equals(REVIEWER_SORT)) {
            Collections.sort(EntryVector, new fr.imag.clips.papillon.business.utility.ReviewerComparator());
        }
        else if (sortBy.equals(STATUS_SORT)) {
            Collections.sort(EntryVector, new fr.imag.clips.papillon.business.utility.StatusComparator());
        }
    }
}


// Return the entries of the volume in relation to the offset and the limit
public static Collection getVolumeEntries(Volume volume, int offset, int limit) throws PapillonBusinessException {
    try {
        
        //
        ArrayList result = new ArrayList();
        
        // New query builder
        VolumeEntryQuery veQuery = new VolumeEntryQuery(volume.getDbname(), CurrentDBTransaction.get());
        
        // Limit and offset
        veQuery.getQueryBuilder().addEndClause(" LIMIT " + Integer.toString(limit) + " OFFSET " + Integer.toString(offset));
        
        // Order by objectId
        veQuery.getQueryBuilder().addOrderByColumn("objectId","");
        
        // Debug
        //veQuery.getQueryBuilder().debug();
        
        // 
        VolumeEntryDO[] DOarray = veQuery.getDOArray();
        if (null != DOarray) {
            for (int j=0; j < DOarray.length; j++) {
                
                //
                VolumeEntry tempEntry = new VolumeEntry(DictionariesFactory.getDictionaryByName(volume.getDictname()), volume, DOarray[j]);
                result.add(tempEntry);
            }
        } 
        
        //
        return result;
        
    } catch(Exception ex) {
        throw new PapillonBusinessException("Exception in getVolumeEntries() ", ex);
    }
}

	
	public static void initializeEntryCache() {
		EntryCache.entryCacheInit();
	}
	
	public static void cutEntryCache() {
		EntryCache.cutEntryCaches();
	}
	
	

//
public static String normalizeValue(String value) {
    
    //
    if (value != null) {
    String patternStr = "(\n|\r|\t|[ ])+";
    String replaceStr = " ";
    Pattern pattern = Pattern.compile(patternStr);
    Matcher matcher = pattern.matcher(value);
    value = matcher.replaceAll(replaceStr);
    
    // 
    patternStr = "^ ";
    replaceStr = "";
    pattern = Pattern.compile(patternStr);
    matcher = pattern.matcher(value);
    value = matcher.replaceAll(replaceStr);
    
    // 
    patternStr = " $";
    replaceStr = "";
    pattern = Pattern.compile(patternStr);
    matcher = pattern.matcher(value);
    value = matcher.replaceAll(replaceStr);
    //PapillonLogger.writeDebugMsg("updateElement: normalized value: " + value);
    }
    else {
        value="";
    }
    //
    return value;
}



public static String[] findEntryIdByEntryName (String entryName, Volume myVolume)
throws PapillonBusinessException {
	VolumeEntry theEntry = null;
	String[] myEntryId = null;
	try{
		VolumeEntryQuery query = new VolumeEntryQuery(myVolume.getDbname(), CurrentDBTransaction.get());
		//set query
		query.setQueryHeadword(entryName);
		VolumeEntryDO[] theEntryDO = query.getDOArray();			
		myEntryId = new String[theEntryDO.length];
		for(int i=0; i<theEntryDO.length;i++){	
			String testid = theEntryDO[0].get_Handle();
			myEntryId[i] = theEntryDO[i].get_Handle();
		}
    }catch(Exception ex) {
        throw new PapillonBusinessException("Exception in findEntryIdByEntryName()", ex);
    }
	return myEntryId;
}

static String typeDico = new String();
public static ArrayList findTargetIdByValue(String headword, String keyword, int operator,
		String values) throws PapillonBusinessException{
	try{
		String[] myVolumeName;
		String[] dicName ;
		if (keyword == "dico"){
			myVolumeName = new String[1];
			//myVolumeName[0] = values;
			dicName = new String[1];
			dicName[0] = values;
		}
		else{
			//get all volume name 
			VolumeQuery qr = new VolumeQuery(CurrentDBTransaction.get());
			VolumeDO[] theVolumeDO = qr.getDOArray();
			myVolumeName = new String[theVolumeDO.length];
			dicName = new String[theVolumeDO.length];
			for(int i=0; i<theVolumeDO.length; i++){
				myVolumeName[i] = theVolumeDO[i].getDbname();
				dicName[i] = theVolumeDO[i].getDictname();
			}
		}
		//get all related objectid in volume table by headword
		ArrayList objectId = new ArrayList();
		ArrayList temp = new ArrayList(); 
		//ArrayList arrayTemp = new ArrayList();
		String targetid = new String();
		for(int i=0; i< myVolumeName.length; i++){ //search in each volume
			
			VolumeEntryQuery qrEntry = new VolumeEntryQuery(myVolumeName[i], CurrentDBTransaction.get());
			qrEntry.setQueryHeadword(headword);
			VolumeEntryDO[] theEntryDO = qrEntry.getDOArray();
			DictionaryQuery dicoQr = new DictionaryQuery(CurrentDBTransaction.get());
			dicoQr.setQueryName(dicName[i]);
			DictionaryDO[] dicoDO = dicoQr.getDOArray();
			typeDico =  dicoDO[0].getType();
			//PapillonLogger.writeDebugMsg("type de dico is direct, pivax, bilingue or ... : "+typeDico);

			if (theEntryDO.length!=0){
			
				for(int n=0; n<theEntryDO.length;n++){		
					
					int l = objectId.size();	
					objectId.add(new Integer(Integer.parseInt(theEntryDO[n].get_Handle())));
					int entryId = Integer.parseInt(objectId.get(l).toString());
					//get all related entry in linktable and indextable
					switch (ChooseTableByKeyword(keyword)){
						case 1: 
							//PapillonLogger.writeDebugMsg("in case 1, filter in index table");
							IndexQuery MyqrIndex = new IndexQuery("idx"+myVolumeName[i], CurrentDBTransaction.get());
							MyqrIndex.setQueryKey(keyword);
							MyqrIndex.setQueryValue(values);
							MyqrIndex.setQueryEntryId(entryId);
							if (!MyqrIndex.equals(null)){
								//add entryDO in the return list
							//	temp[arrayTemp.size()] = theEntryDO[n];
							//	arrayTemp.add("");
							}
							break;
						case 2: 
							//PapillonLogger.writeDebugMsg("in case 2, filter in link table");
							//PapillonLogger.writeDebugMsg("is link table");
							LinkQuery MyqrLink = new LinkQuery("link"+myVolumeName[i], CurrentDBTransaction.get());
							if (keyword == "lc"){
								MyqrLink.setQueryLang(values);
							}else if(keyword == "poids"){
								//PapillonLogger.writeDebugMsg("is poids");
								double v = Double.parseDouble(values);
								//PapillonLogger.writeDebugMsg("value is"+v);
								MyqrLink.setQueryWeight(v, QueryBuilder.GREATER_THAN);
								
							}
							MyqrLink.setQueryEntryId(entryId);
							//PapillonLogger.writeDebugMsg("reussi a excuter les query");
							if (!MyqrLink.equals(null)){
								//add tragetid in the return list
								//PapillonLogger.writeDebugMsg("n ="+n);
								//PapillonLogger.writeDebugMsg("theEntryDO[n]="+theEntryDO[n].get_Handle());
								//PapillonLogger.writeDebugMsg("temp.length ="+temp.size());
								//PapillonLogger.writeDebugMsg("query has reslut: n="+n+", theEntryDO[n]="+theEntryDO[n].get_Handle()+", temp.length ="+temp.size());
								//VolumeEntryDO ve = theEntryDO[n];
								//PapillonLogger.writeDebugMsg("ve is existe: ve ="+ve.get_Handle());
								
								int row = MyqrLink.getCount();
								//search target with link table params targetid and lang
								LinkDO[] linkEntry = MyqrLink.getDOArray();
								if (linkEntry.length > 0){
									for (int b = 0; b < row; b++){
										targetid = linkEntry[b].getTargetId();
										//PapillonLogger.writeDebugMsg("+++tragetid ="+targetid);
										//String lang = linkEntry[b].getLang();
										//VolumeEntryDO ve = findTargetEntryByTargetidAndLang(targetid,lang);
										temp.add(targetid);
									}
								}
								
								//zhang ying 11111
								//temp.add(targetid);
								
								
								//PapillonLogger.writeDebugMsg("temp = " + temp[arrayTemp.size()]);
								//arrayTemp.add("");
							}
							break;
						case 3:
							//PapillonLogger.writeDebugMsg("in case 3, error");
							break;
							
						
					}
					
				
				}
				
			}
			
	
			
		}
		//PapillonLogger.writeDebugMsg("return temp");
		return temp;
	
    }catch(Exception ex) {
        throw new PapillonBusinessException("Exception in findEntryByValue()", ex);
    }
 
}

//public static VolumeEntryDO findTargetEntryByTargetidAndLang(String targetid, String lang){
//}


public static int ChooseTableByKeyword(String key){
	ArrayList indexKeyList = new ArrayList();    	
	indexKeyList.add("auteur");
	indexKeyList.add("group");
	indexKeyList.add("comment");
	indexKeyList.add("date");
	ArrayList linkKeyList = new ArrayList();
	linkKeyList.add("poids");
	linkKeyList.add("lc");
	int mykey;
	if (indexKeyList.contains(key)){
		return 1;
	}else if (linkKeyList.contains(key)){
		return 2;
	}else if (key == "dico"){
		return 3;
	}	
	return 0;
}

public static VolumeEntryDO[] findEntry (ArrayList conditions) throws PapillonBusinessException{
	ArrayList targetIdList = new ArrayList();
	if (!conditions.isEmpty()){
		String hw = ((ArrayList)conditions.get(0)).get(0).toString(); // headword : conditions.get(0)
		for (int i=0; i<conditions.size(); i++){
			String keyword = ((ArrayList)conditions.get(i+1)).get(0).toString(); //conditions: conditions.get(>0)
			String o = ((ArrayList)conditions.get(i+1)).get(1).toString();
			int operator = 1;
			if (o == "1"){
				operator = 1;
			}else if (o == "2"){
				operator = 2;
			}else if (o == "3"){
				operator =3;
			}
			String values =  ((ArrayList)conditions.get(i+1)).get(2).toString();
			ArrayList myList = findTargetIdByValue(hw, keyword, operator, values);
//			try{
//				if (myList.size() != 0){
//					for (int a=0; a<myList.size(); a++){
//						//PapillonLogger.writeDebugMsg("--- myList return is :"  +myList.get(a));
//					}
//				}
//		    }catch(Exception ex) {
//		        throw new PapillonBusinessException("Exception in findEntry()", ex);
//		    }
		    		
				for (int j=0; j< myList.size(); j++){
					if (targetIdList.isEmpty()){
						targetIdList.add(myList.get(j).toString()) ;
						
					}else if(!listExist(targetIdList, myList.get(j).toString())){
						targetIdList.add(myList.get(j));
						
					}
				}
			//	XslTransformation.getFormattedResultForLink(typeDico, targetIdList);
				if(myList.size()==0){
					PapillonLogger.writeDebugMsg("no match found");
				}
				for (int j=0; j<myList.size(); j++){
					PapillonLogger.writeDebugMsg("final list:"+targetIdList.get(j));
					
				}
									
			}
		}
	return null;
}



public static boolean listExist (ArrayList targetIdList, String newTarget){

	for (int i=0; i<targetIdList.size(); i++){
		//PapillonLogger.writeDebugMsg("listExist? :"+targetIdList.get(i)+"' new:"+newTarget);
		if (targetIdList.get(i)==newTarget || targetIdList.get(i).equals(newTarget)){
			//PapillonLogger.writeDebugMsg("list equals: targetlist"+targetIdList.get(i)+"' new:"+newTarget);
			return true;
		}
	}
	return false;
	
}


 public static ArrayList findEntriesByHeadword(String volumeName, String headword)
	throws PapillonBusinessException {
    	Volume volume;
		Dictionary dict;
		try {
			volume = VolumesFactory.getVolumeByName(volumeName);
			dict = DictionariesFactory.getDictionaryByName(volume.getDictname());
		}
		catch(Exception ex) {
			return null;
		}
		String[] entriesId = findEntryIdByEntryName (headword, volume);
		ArrayList volumeEntries = new ArrayList();
		for(int i=0; i<entriesId.length; i++){
			VolumeEntry ve = findEntryByHandle(volumeName, entriesId[i]);
			volumeEntries.add(ve);
		}
		return volumeEntries;
    }
}

