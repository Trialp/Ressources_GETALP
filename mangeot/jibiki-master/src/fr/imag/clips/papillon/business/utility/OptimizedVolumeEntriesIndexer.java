package fr.imag.clips.papillon.business.utility;

import com.lutris.appserver.server.Enhydra;
import com.lutris.appserver.server.sql.DBConnection;
import com.lutris.appserver.server.sql.DBTransaction;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.IndexEntry;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.xml.XMLServices;
import org.w3c.dom.Document;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;


public class OptimizedVolumeEntriesIndexer {

    static long fetchcumul = 0;
    static long parsecumul = 0;
    static long writecumul = 0;

    public static void rebuildIndexes(Volume volume) throws PapillonBusinessException {
        DBConnection myDbConnection = null;
		java.util.HashMap CdmElementsTable = volume.getCdmElements();
		java.util.HashMap linksTable = volume.getLinksTable();
        String sql = "SELECT xmlcode, objectid FROM " + volume.getDbname() + " ORDER BY objectid";

        PapillonLogger.writeDebugMsg("Re-indexing volume " + volume.getDbname() + " which contains " + volume.getCount() + " entries.");

        int offset = 0;
        int resultCount = 100;
        ArrayList indexes = new ArrayList(4000);
        ArrayList links = new ArrayList(4000);
        
        while (resultCount == 100) {
            resultCount = 0;

            long start = System.currentTimeMillis();
            try {
                String actualSql = sql + " OFFSET " + offset + " LIMIT 100";
                myDbConnection = Enhydra.getDatabaseManager().allocateConnection();
                ResultSet myResultSet = myDbConnection.executeQuery(actualSql);


                if (myResultSet != null) {
                    while (myResultSet.next()) {
                        String xmlCode = myResultSet.getString("xmlcode");
                        Document xmlDoc = XMLServices.buildDOMTree(xmlCode);
                        String handle = myResultSet.getString("objectid");
							
						if (xmlDoc != null) {
							org.w3c.dom.Element myRootElt = xmlDoc.getDocumentElement();
								
							indexes.addAll(IndexEntry.indexEntry(CdmElementsTable, myRootElt, volume.getPrefixResolver(), handle));
							links.addAll(IndexEntry.indexEntryLinks(linksTable, volume, myRootElt, handle));
						}						
						
                        resultCount++;
                    }
                    myResultSet.close();
                }
            } catch (SQLException se) {
                se.printStackTrace();
                //very important to throw out bad connections

                if (myDbConnection.handleException(se)) myDbConnection = null;
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                if (myDbConnection != null) {
                    try {
                        myDbConnection.reset();
                        myDbConnection.release();
                    } catch (SQLException se) {
                        se.printStackTrace();
                        //very important to throw out bad connections
                        if (myDbConnection.handleException(se)) myDbConnection = null;
                    }
                }
            }
            long mid = System.currentTimeMillis();

            try {
                CurrentDBTransaction.registerNewDBTransaction();
                IndexEntry.saveIndexes(indexes, volume.getIndexDbname());
                IndexEntry.saveLinks(links);
                CurrentDBTransaction.get().commit();
            } catch (SQLException se) {
                String userMessage = "Problems when re-index entries.";
                PapillonLogger.writeDebugMsg(userMessage);
                se.printStackTrace();
                try {
                    DBTransaction dbt = ((DBTransaction) CurrentDBTransaction.get());
                    if (null != dbt) dbt.rollback();
                } catch (java.sql.SQLException sqle) {
                    PapillonLogger.writeDebugMsg(
                            "reConstructionIndex: SQLException while rolling back failed transaction.");
                    sqle.printStackTrace();
                }
            } finally {
                CurrentDBTransaction.releaseCurrentDBTransaction();
            }

            long end = System.currentTimeMillis();
            
            PapillonLogger.writeDebugMsg("Re indexed entries " + offset + " to " + (offset + resultCount) + " | read: " + (mid-start) + ", write: " + (end-mid) );
            offset += resultCount;
            indexes.clear();
        }

    }

}
