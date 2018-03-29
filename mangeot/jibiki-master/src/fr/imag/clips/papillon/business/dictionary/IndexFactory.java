/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.19  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.18  2006/08/10 22:17:12  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.17  2006/03/01 15:20:39  mangeot
 * syntax bug fix
 *
 * Revision 1.16  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.15  2006/02/28 15:26:22  mangeot
 * Bug fix when creating new tables
 *
 * Revision 1.14  2006/02/26 19:21:38  mangeot
 * Work on BrowseVolume
 *
 * Revision 1.13  2006/02/26 14:08:16  mangeot
 * Added the multilingual_sort(lang,headword) index on volume tables for speeding up the lookup
 *
 * Revision 1.12  2005/12/01 15:34:28  mangeot
 * MM: I solved the problem of already created tables by creating an sql query for retrieving the table names. If the name already exists, VolumeEntriesFactory.createVolumeTables do not create the tables.
 * It allows the administrator to delete and reload only the metadata files without dropping the whole data.
 * The method is ManageDatabase.getTableNames() and it returns a vector with all the table names created by the database user (usually "papillon").
 *
 * Revision 1.11  2005/09/17 11:32:51  mangeot
 * *** empty log message ***
 *
 * Revision 1.10.4.3  2006/01/25 15:22:23  fbrunet
 * Improvement of QueryRequest
 * Add new search criteria
 * Add modified status
 *
 * Revision 1.10.4.2  2005/12/02 10:04:09  fbrunet
 * Add Pre/Post edition processing
 * Add index reconstruction
 * Add new query request
 * Add fuzzy search
 * Add new contribution administration
 * Add xsl transformation volume
 *
 * Revision 1.10.4.1  2005/10/24 16:29:19  fbrunet
 * Added fuzzy search capabilities.
 * Added possibility to rebuild the index DB tables.
 * Added Pre and post processors that could be defined by the user.
 *
 * Revision 1.10  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 * Revision 1.9  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.8  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.7.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.7  2005/04/18 13:22:47  mangeot
 * Fixed a bug with the strategy
 *
 * Revision 1.6  2005/04/13 14:34:38  mangeot
 * Simplified the expert lookup. Now lookup directly the cdm element name
 *
 * Revision 1.5  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.4.2.7  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.4.2.6  2005/02/06 22:43:49  mangeot
 * Merged the 2 Hashtables CDM Elements and XPaths into one
 * Added a boolean (reverse-lookup) in the volume metadata and functionalities in order to perform a reverse lookup when no direct lookup result is found
 * Added a boolean (index) in the volume metadata for indexing the only specified CDM Elements
 *
 * Revision 1.4.2.5  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
 *
 * Revision 1.4.2.4  2005/01/27 23:55:13  mangeot
 * *** empty log message ***
 *
 * Revision 1.4.2.3  2005/01/27 18:09:28  mangeot
 * Simple dictionary lookup is now working for GDEF.
 * Does not compile yet but cvs commit for backup
 *
 * Revision 1.4.2.2  2005/01/27 15:56:21  mangeot
 * Able to load a volume with XPointers, cannot lookup the result yet.
 * Does not compile but commit for backup
 *
 * Revision 1.4.2.1  2005/01/25 13:54:54  mangeot
 * changed the volume volumeEntry and index objects. Does not compile but need a backup...
 *
 * Revision 1.4  2005/01/18 12:16:10  mangeot
 * Implemented the SQL LIMIT and OFFSET keywords. It allows us to retrieve the entries as blocks and page them. The LIMIT is the DictionariesFactory.MaxRetrievedEntries constant.
 * The implementation may need further tuning
 *
 * Revision 1.3  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
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

import com.lutris.dods.builder.generator.query.QueryBuilder;
import com.lutris.dods.builder.generator.query.RDBColumn;
import com.lutris.dods.builder.generator.query.RDBTable;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.data.IndexDO;
import fr.imag.clips.papillon.data.IndexQuery;
import fr.imag.clips.papillon.papillon_data.ManageDatabase;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Vector;

/**
* Used to find the instances of xslsheet.
 */
public class IndexFactory {
	
	public final static String INDEX_TABLE_PREFIX = "idx";
	
	public final static String KEY_FIELD = "key";
	public final static String LANG_FIELD = "lang";
	public final static String VALUE_FIELD = "value";
	public final static String ENTRYID_FIELD = "entryid";
	public final static String MSORT_FIELD = "msort";
	
	public final static String ORDER_ASCENDING = "ASC";
	public final static String ORDER_DESCENDING = "DESC";
	
	protected static java.util.regex.Pattern quotePattern = java.util.regex.Pattern.compile("'");

	protected final static String databaseName = IndexDO.get_logicalDBName();
	public static String databaseVendor = null;
	protected static com.lutris.dods.builder.generator.query.RDBColumn[] Columns = new com.lutris.dods.builder.generator.query.RDBColumn[1];

	static {
		try  {
			databaseVendor = org.enhydra.dods.DODS.getDatabaseManager().logicalDatabaseType(databaseName);
		}
		catch (com.lutris.appserver.server.sql.DatabaseManagerException ex) {
			databaseVendor = null;
		}
		catch (java.sql.SQLException ex) {
			databaseVendor = null;
		}
	}
		
    // FIXME: only used by queries on the JM_dict, question: why ?
	protected static Vector getEntriesVector(Dictionary dict, Volume volume, Vector Keys1, Vector Keys2, int offset) throws PapillonBusinessException {
		Vector theEntries = new Vector();
		VolumeEntry myEntry = null;
		
		//myKey[0] = key
		//myKey[1] = lang
		//myKey[2] = value
		//myKey[3] = cmp_op
		
		if (Keys1.size()==0 && Keys2.size()>0) {
			Object myObject = Keys2.firstElement();
			Keys1.add(myObject);
			Keys2.remove(myObject);
		}
		
		com.lutris.dods.builder.generator.query.QueryBuilder myQueryBuilder = null; 
		com.lutris.dods.builder.generator.query.RDBColumn entryidColumn = IndexDO.getEntryIdColumn(volume.getIndexDbname());
		Columns[0] = entryidColumn;
		if (Keys1 !=null) {
		for (java.util.Enumeration enumKeys1 = Keys1.elements(); enumKeys1.hasMoreElements();) {
			//myKey[0] = key
			//myKey[1] = lang
			//myKey[2] = value
			//myKey[3] = cmp_op
			String[] key1 = (String[]) enumKeys1.nextElement();
			if (key1[2] !=null && !key1[2].equals("")) {
				try {
					fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Indexed request table: " + volume.getIndexDbname() + " :: " +  key1[0] + "/" + key1[1] + "/" + key1[2]);
					IndexQuery query = new IndexQuery(volume.getIndexDbname(), CurrentDBTransaction.get());
					query.getQueryBuilder().addWhereClause(KEY_FIELD, key1[0], QueryBuilder.EQUAL);
					if (key1[1] !=null && !key1[1].equals("")) {
						query.getQueryBuilder().addWhereClause(LANG_FIELD, key1[1], QueryBuilder.EQUAL);
					}
					query.getQueryBuilder().addWhereClause(VALUE_FIELD, key1[2], key1[3]);
					if (Keys2 !=null) {
						for (java.util.Enumeration enumKeys2 = Keys2.elements(); enumKeys2.hasMoreElements();) {
						String[] key2 = (String[]) enumKeys2.nextElement();
						if (key2!=null && key2[2] !=null && !key2[2].equals("")) {
							myQueryBuilder = new com.lutris.dods.builder.generator.query.QueryBuilder(Columns);
							if (databaseVendor != null) {
								myQueryBuilder.setDatabaseVendor(databaseVendor);
							} else {
								myQueryBuilder.setDatabaseVendor();
							}
							myQueryBuilder.addWhereClause(KEY_FIELD, key2[0], QueryBuilder.EQUAL);
							if (key2[1] !=null && !key2[1].equals("")) {
								myQueryBuilder.addWhereClause(LANG_FIELD, key2[1], QueryBuilder.EQUAL);
							}
							myQueryBuilder.addWhereClause(VALUE_FIELD, key2[2],  key2[3]);
							myQueryBuilder.resetSelectedFields();
							myQueryBuilder.select(entryidColumn);
							query.getQueryBuilder().addWhereIn(entryidColumn, myQueryBuilder);
							}
						}
					}
					query.getQueryBuilder().setMaxRows(DictionariesFactory.MaxRetrievedEntries);
					query.getQueryBuilder().addEndClause("OFFSET " + offset);
					query.getQueryBuilder().addOrderByColumn("multilingual_sort(lang,value)","");
					PapillonLogger.writeDebugMsg("Query SQL: " + query.getQueryBuilder().getSQL());
					IndexDO[] DOarray = query.getDOArray();
					if (null != DOarray) {
						for (int j=0; j < DOarray.length; j++) {
							Index myIndex = new Index(DOarray[j]);
							myEntry = VolumeEntriesFactory.findEntryByHandle(dict, volume, ""+myIndex.getEntryId());
							theEntries.add(myEntry);
						}
					}
				}
				catch(Exception ex) {
					throw new PapillonBusinessException("Exception in getEntriesVector()", ex);
				}
			}
		}
		}
		return theEntries;
	}
	
		        
	public static Vector getIndexVectorByEntryId(Volume volume, String entryId) throws PapillonBusinessException {
		Vector theIndex = new Vector();
		
		if (entryId != null && !entryId.equals("")) {
			int intId = 0;
			try {
				intId = Integer.parseInt(entryId);
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in getIndexVectorByEntryId()", ex);
			}
			
			try {
				IndexQuery query = new IndexQuery(volume.getIndexDbname(), CurrentDBTransaction.get());
				query.setQueryEntryId(intId);
				IndexDO[] DOarray = query.getDOArray();
				if (null != DOarray) {
					for (int j=0; j < DOarray.length; j++) {
						theIndex.add(new Index(DOarray[j]));
					}
				}
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in getIndexVectorByEntryId()", ex);
			}
		}
		return theIndex;
	}
	
	public static ArrayList getIndexEntriesVector(String indexTableName, Vector Keys, String order, int limit) throws PapillonBusinessException {
		return getIndexEntriesVector(indexTableName, Keys, order, limit,0);
	}

    public static ArrayList getIndexEntriesVector(String indexTableName, Vector Keys, String order, int limit, int offset) throws PapillonBusinessException {
        return getIndexEntriesVector(indexTableName, Keys, null, order, limit, offset);
    }
	public static ArrayList getIndexEntriesVector(String indexTableName, Vector Keys, Vector Clauses, String order, int limit, int offset) throws PapillonBusinessException {
        ArrayList theEntries = new ArrayList();
		
		if (null != indexTableName) {
			try {
				com.lutris.dods.builder.generator.query.RDBColumn keyColumn = IndexDO.getKeyColumn(indexTableName);
				com.lutris.dods.builder.generator.query.RDBColumn langColumn = IndexDO.getLangColumn(indexTableName);
                com.lutris.dods.builder.generator.query.RDBColumn valueColumn = IndexDO.getValueColumn(indexTableName);
                com.lutris.dods.builder.generator.query.RDBColumn entryIdColumn = IndexDO.getEntryIdColumn(indexTableName);
                IndexQuery query = new IndexQuery(indexTableName, CurrentDBTransaction.get());
				//fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Index request table: " + indexTableName);
                if (limit==0) {
                    limit = DictionariesFactory.MaxRetrievedEntries;
                }
                
				if (Keys != null) {
					for (java.util.Enumeration enumKeys = Keys.elements(); enumKeys.hasMoreElements();) {
						String[] key = (String[]) enumKeys.nextElement();
                        //PapillonLogger.writeDebugMsg("keys: 0:" + key[0] + " 1:" + key[1]+ " 2:" + key[2]+ " 3:" + key[3]);
						if (key!=null && key[2] !=null && !key[2].equals("")) {
                           /* clés multiples */
                          String[] keynames = key[0].split("\\|");
                            // Liste des clés qui sont classées par ordre alphabétique
                           if (!keynames[0].equals(Volume.CDM_headword)
                               && !keynames[0].equals(Volume.CDM_headwordVariant)
                               && !keynames[0].equals(Volume.CDM_reading)
                               &&!keynames[0].equals(Volume.CDM_writing)) {
                                query.getQueryBuilder().addWhere(keyColumn,Volume.CDM_headword,QueryBuilder.EQUAL);
                                
                                RDBTable tableIndex = new RDBTable(indexTableName);
                                RDBColumn entryIdRDB = new RDBColumn(tableIndex, "entryId", false );
                                RDBColumn[] tableIndexRDBList = new RDBColumn[1];
                                tableIndexRDBList[0] = entryIdRDB;
                                
                                QueryBuilder querySearch = new QueryBuilder(tableIndexRDBList);
                                if (IndexFactory.databaseVendor != null) {
                                    querySearch.setDatabaseVendor(IndexFactory.databaseVendor);
                                } else {
                                    querySearch.setDatabaseVendor();
                                }

                               if (keynames.length>1) {
                                  querySearch.addWhereOpenParen();
                                    for (int i=0;i<keynames.length;i++) {
                                        querySearch.addWhere(keyColumn, keynames[i], QueryBuilder.EQUAL);
                                        if (i<keynames.length-1) {
                                            query.getQueryBuilder().addWhereOr();
                                        }
                                    }
                                    querySearch.addWhereCloseParen();
                               }
                               else {
                                   querySearch.addWhere(keyColumn, keynames[0], QueryBuilder.EQUAL);
                               }
                               
                                if (key[1] !=null && !key[1].equals("")) {
                                    querySearch.addWhere(langColumn, key[1], QueryBuilder.EQUAL);
                                }
                                if ( key[3] == QueryBuilder.LESS_THAN ||
                                    key[3] == QueryBuilder.LESS_THAN_OR_EQUAL ||
                                    key[3] == QueryBuilder.GREATER_THAN ||
                                    key[3] == QueryBuilder.GREATER_THAN_OR_EQUAL) {
                                    //Replace all apostrophes with double apostrophes
                                    java.util.regex.Matcher quoteMatcher = quotePattern.matcher(key[2]);
                                    String newValue = quoteMatcher.replaceAll("''");
                                    
                                    querySearch.addWhere(MSORT_FIELD + key[3]+ "multilingual_sort('" + key[1] + "','" + newValue + "')");
                                }
                                else {
                                    java.util.regex.Matcher quoteMatcher = quotePattern.matcher(key[2]);
                                   String newValue = quoteMatcher.replaceAll("''");
                                   querySearch.addWhere(valueColumn, key[2],  key[3]);
                                }
                                //querySearch.resetSelectedFields();
                                //querySearch.select(entryIdColumn);
                               query.getQueryBuilder().addWhereIn(entryIdRDB,querySearch);
                             }
                            else {
                                /* clés multiples */
                                if (keynames.length>1) {
                                    query.getQueryBuilder().addWhereOpenParen();
                                    for (int i=0;i<keynames.length;i++) {
                                        query.getQueryBuilder().addWhere(keyColumn, keynames[i], QueryBuilder.EQUAL);
                                        if (i<keynames.length-1) {
                                            query.getQueryBuilder().addWhereOr();
                                        }
                                    }
                                    query.getQueryBuilder().addWhereCloseParen();
                                }
                                else {
                                    query.getQueryBuilder().addWhere(keyColumn, keynames[0], QueryBuilder.EQUAL);
                                }
                                
                                if (key[1] !=null && !key[1].equals("")) {
                                    query.getQueryBuilder().addWhere(langColumn, key[1], QueryBuilder.EQUAL);
                                }
                                if ( key[3] == QueryBuilder.LESS_THAN ||
                                    key[3] == QueryBuilder.LESS_THAN_OR_EQUAL ||
                                    key[3] == QueryBuilder.GREATER_THAN ||
                                    key[3] == QueryBuilder.GREATER_THAN_OR_EQUAL) {
                                    //Replace all apostrophes with double apostrophes
                                    java.util.regex.Matcher quoteMatcher = quotePattern.matcher(key[2]);
                                    String newValue = quoteMatcher.replaceAll("''");
                                    
                                    query.getQueryBuilder().addWhere(MSORT_FIELD + key[3]+ "multilingual_sort('" + key[1] + "','" + newValue + "')");
                                }
                                else {
                                    query.getQueryBuilder().addWhere(valueColumn, key[2],  key[3]);
                                }
                            }
 						}
					}
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
                    clausesQueryBuilder.select(entryIdColumn);
                    query.getQueryBuilder().addWhereIn(entryIdColumn, clausesQueryBuilder);
                }
				if (order==null || !order.equals(ORDER_DESCENDING)) {
					order = "";
				}
                query.getQueryBuilder().addOrderByColumn(MSORT_FIELD,order);
                query.getQueryBuilder().setMaxRows(limit);
                query.getQueryBuilder().addEndClause(" LIMIT " + limit);
                if (offset!=0) {
                    query.getQueryBuilder().addEndClause(" OFFSET " + offset);
                }

				// debug
                //PapillonLogger.writeDebugMsg("getIndexEntriesVector query debug: ");
				//query.getQueryBuilder().debug();
				
				IndexDO[] DOarray = query.getDOArray();
				if (null != DOarray) {
					for (int j=0; j < DOarray.length; j++) {
						Index tempIndex = new Index(DOarray[j]);
						theEntries.add(tempIndex);
                        //PapillonLogger.writeDebugMsg("getIndexEntriesVector entryid: " + tempIndex.getEntryId());
					}
				}
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in getIndexEntriesVector()", ex);
			}
		}
		return theEntries;
	}
	
	public static ArrayList getIndexEntriesVector(String indexTableName, String msort, String operator, String order, int limit) throws PapillonBusinessException {
		return getIndexEntriesVector(indexTableName, msort, operator, order, limit,0);
	}
	
	public static ArrayList getIndexEntriesVector(String indexTableName, String msort, String operator, String order, int limit, int offset) throws PapillonBusinessException {
		return getIndexEntriesVector(indexTableName, Volume.CDM_headword, msort, operator, order, limit, offset);
	}

    public static ArrayList getIndexEntriesVector(String indexTableName, String key, String msort, String operator, String order, int limit, int offset) throws PapillonBusinessException {
        return getIndexEntriesVector(indexTableName, key, null, msort, operator, order, limit, offset);
    }

    public static ArrayList getIndexEntriesVector(String indexTableName, String key, Vector Clauses, String msort, String operator, String order, int limit, int offset) throws PapillonBusinessException {
        ArrayList theEntries = new ArrayList();
		
		if (null != indexTableName) {
			try {
				com.lutris.dods.builder.generator.query.RDBColumn keyColumn = IndexDO.getKeyColumn(indexTableName);
				com.lutris.dods.builder.generator.query.RDBColumn msortColumn = IndexDO.getMsortColumn(indexTableName);
                com.lutris.dods.builder.generator.query.RDBColumn entryIdColumn = IndexDO.getEntryIdColumn(indexTableName);
				IndexQuery query = new IndexQuery(indexTableName, CurrentDBTransaction.get());
				
                if (limit==0) {
                    limit = DictionariesFactory.MaxRetrievedEntries;
                }

                /* clés multiples */
                String[] keynames = key.split("\\|");
                if (keynames.length>1) {
                    query.getQueryBuilder().addWhereOpenParen();
                    for (int i=0;i<keynames.length;i++) {
                        query.getQueryBuilder().addWhere(keyColumn, keynames[i], QueryBuilder.EQUAL);
                        if (i<keynames.length-1) {
                            query.getQueryBuilder().addWhereOr();
                        }
                    }
                    query.getQueryBuilder().addWhereCloseParen();
                }
                else {
                    query.getQueryBuilder().addWhere(keyColumn, keynames[0], QueryBuilder.EQUAL);
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
                    clausesQueryBuilder.select(entryIdColumn);
                    query.getQueryBuilder().addWhereIn(entryIdColumn, clausesQueryBuilder);
                }
             
                
                query.getQueryBuilder().addWhere(msortColumn, msort,  operator);
				
                if (order==null || !order.equalsIgnoreCase(ORDER_DESCENDING)) {
                    order = "";
                }
                query.getQueryBuilder().addOrderByColumn(MSORT_FIELD,order);
                query.getQueryBuilder().setMaxRows(limit);
                query.getQueryBuilder().addEndClause(" LIMIT " + limit);
				if (offset!=0) {
					query.getQueryBuilder().addEndClause(" OFFSET " + offset);
				}
				// debug
                //PapillonLogger.writeDebugMsg("getIndexEntriesVector avec key+msort debug:");
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
	
    
    public static ArrayList getIndexValuesVector(String indexTableName, Vector Keys, String order, int limit, int offset) throws PapillonBusinessException {
        ArrayList theEntries = new ArrayList();
        
        if (null != indexTableName) {
            try {
                com.lutris.dods.builder.generator.query.RDBColumn keyColumn = IndexDO.getKeyColumn(indexTableName);
                com.lutris.dods.builder.generator.query.RDBColumn langColumn = IndexDO.getLangColumn(indexTableName);
                com.lutris.dods.builder.generator.query.RDBColumn valueColumn = IndexDO.getValueColumn(indexTableName);
                com.lutris.dods.builder.generator.query.RDBColumn entryIdColumn = IndexDO.getEntryIdColumn(indexTableName);
                IndexQuery query = new IndexQuery(indexTableName, CurrentDBTransaction.get());
                //fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("Index request table: " + indexTableName);
                if (limit==0) {
                    limit = DictionariesFactory.MaxRetrievedEntries;
                }
                
                if (Keys != null) {
                    for (java.util.Enumeration enumKeys = Keys.elements(); enumKeys.hasMoreElements();) {
                        String[] key = (String[]) enumKeys.nextElement();
                        //PapillonLogger.writeDebugMsg("keys: 0:" + key[0] + " 1:" + key[1]+ " 2:" + key[2]+ " 3:" + key[3]);
                        if (key!=null && key[2] !=null && !key[2].equals("")) {
                            /* clés multiples */
                            String[] keynames = key[0].split("\\|");
                            // Liste des clés qui sont classées par ordre alphabétique
                                /* clés multiples */
                                if (keynames.length>1) {
                                    query.getQueryBuilder().addWhereOpenParen();
                                    for (int i=0;i<keynames.length;i++) {
                                        query.getQueryBuilder().addWhere(keyColumn, keynames[i], QueryBuilder.EQUAL);
                                        if (i<keynames.length-1) {
                                            query.getQueryBuilder().addWhereOr();
                                        }
                                    }
                                    query.getQueryBuilder().addWhereCloseParen();
                                }
                                else {
                                    query.getQueryBuilder().addWhere(keyColumn, keynames[0], QueryBuilder.EQUAL);
                                }
                                
                                if (key[1] !=null && !key[1].equals("")) {
                                    query.getQueryBuilder().addWhere(langColumn, key[1], QueryBuilder.EQUAL);
                                }
                                if ( key[3] == QueryBuilder.LESS_THAN ||
                                    key[3] == QueryBuilder.LESS_THAN_OR_EQUAL ||
                                    key[3] == QueryBuilder.GREATER_THAN ||
                                    key[3] == QueryBuilder.GREATER_THAN_OR_EQUAL) {
                                    //Replace all apostrophes with double apostrophes
                                    java.util.regex.Matcher quoteMatcher = quotePattern.matcher(key[2]);
                                    String newValue = quoteMatcher.replaceAll("''");
                                    
                                    query.getQueryBuilder().addWhere(MSORT_FIELD + key[3]+ "multilingual_sort('" + key[1] + "','" + newValue + "')");
                                }
                                else {
                                    query.getQueryBuilder().addWhere(valueColumn, key[2],  key[3]);
                                }
                        }
                    }
                }
                if (order==null || !order.equals(ORDER_DESCENDING)) {
                    order = "";
                }
                query.getQueryBuilder().addOrderByColumn(MSORT_FIELD,order);
                query.getQueryBuilder().setMaxRows(limit);
                query.getQueryBuilder().addEndClause(" LIMIT " + limit);
                if (offset!=0) {
                    query.getQueryBuilder().addEndClause(" OFFSET " + offset);
                }
                
                // debug
                //PapillonLogger.writeDebugMsg("getIndexValuesVector query debug: ");
                //query.getQueryBuilder().debug();
                
                IndexDO[] DOarray = query.getDOArray();
                if (null != DOarray) {
                    for (int j=0; j < DOarray.length; j++) {
                        Index tempIndex = new Index(DOarray[j]);
                        theEntries.add(tempIndex);
                        //PapillonLogger.writeDebugMsg("getIndexValuesVector entryid: " + tempIndex.getEntryId());
                    }
                }
            }
            catch(Exception ex) {
                throw new PapillonBusinessException("Exception in getIndexEntriesVector()", ex);
            }
        }
        return theEntries;
    }

	
	protected static void deleteIndexForEntryId(String indexDbname, String entryId) throws 	PapillonBusinessException {
		Vector theIndex = new Vector();
		
		String cmp_op = QueryBuilder.EQUAL;
		
		int intId = 0;
		try {
			intId = Integer.parseInt(entryId);
		}
		catch(Exception ex) {
			throw new PapillonBusinessException("Exception in newIndex()", ex);
		}
		
		
		if (entryId != null && !entryId.equals("")) {
			try {
				IndexQuery query = new IndexQuery(indexDbname, CurrentDBTransaction.get());
				query.setQueryEntryId(intId);
				IndexDO[] DOarray = query.getDOArray();
				if (null != DOarray) {
					for (int j=0; j < DOarray.length; j++) {
						Index myIndex = new Index(DOarray[j]);
						myIndex.delete();
					}
				}
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in deleteIndexForEntryId()", ex);
			}
		}
	}
	
	public static Index newIndex(String table, String key, String lang, String value, String handle)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			//
			int intId = 0;
			try {
				intId = Integer.parseInt(handle);
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in newIndex()", ex);
			}
			
			Index newIndex = null;
			if (value != null) {
				value = value.trim();
				if (!value.equals("")) {
					newIndex=new Index(table);
					// name of the key
					newIndex.setKey(key);
					// lang of the key
					if (lang.length()==3) {
						newIndex.setLang(lang);
					}
					else {
						newIndex.setLang(Volume.DEFAULT_LANG);
					}
					// value of the key
					newIndex.setValue(value);
					// entry id
					newIndex.setEntryId(intId);
				}
			}
			//fr.imag.clips.papillon.business.PapillonLogger.writeDebugMsg("NEW INDEX : table=" + table + " key=" + key + " lang=" + lang + " value=" + value + " lang=" + handle);
				return newIndex;
            
		}
	
	public static void emptyIndex(String table)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			try {
				ManageDatabase.truncateTable(table);
			}
			catch (Exception e) {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException ("Exception in emptyIndex with table: " + table, e);
			}
		}
		
	public static void truncateIndexTable(Volume volume) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			truncateIndexTable(volume.getIndexDbname());
		}
	
	public static void truncateIndexTable(String table)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			try {
				ManageDatabase.truncateTable(table);
			}
			catch (Exception e) {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException ("Exception in truncateIndexTable: " + table, e);
			}
		}
	}

