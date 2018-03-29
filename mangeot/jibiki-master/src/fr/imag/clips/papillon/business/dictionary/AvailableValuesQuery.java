package fr.imag.clips.papillon.business.dictionary;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.utility.StringNormalizer;

import java.util.*;
import java.sql.SQLException;
import java.sql.ResultSet;
import java.text.Collator;

import com.lutris.appserver.server.sql.DBConnection;
import com.lutris.appserver.server.Enhydra;

/**
 *
 */
public class AvailableValuesQuery {

    public static Collection getAvailableValues(String field, String prefix, String sourceLanguage, int limit) throws SQLException {

        DBConnection myDBConnection = null;
        ResultSet myResultSet = null;
        TreeSet myTreeSet;
        if (sourceLanguage == null | sourceLanguage.equals(""))
            myTreeSet = new TreeSet();
        else
            myTreeSet = new TreeSet(Collator.getInstance(new Locale(sourceLanguage)));


        try {
            Collection volumes;
            if (sourceLanguage == null | sourceLanguage.equals(""))
                volumes = VolumesFactory.getVolumes();
            else
                volumes = VolumesFactory.getVolumesArray(null, sourceLanguage, null);


            myDBConnection = Enhydra.getDatabaseManager().allocateConnection();

            for (Iterator iter = volumes.iterator(); iter.hasNext();) {
                Volume volume = (Volume) iter.next();

                String sqlQuery = "SELECT DISTINCT value FROM " + volume.getIndexDbname() + " WHERE key='" + field + "' ";
                if (prefix != null & !"".equals(prefix)) {
                    sqlQuery += " AND value ilike '" + escapeForSql(prefix) + "%' ";
                }
                if (sourceLanguage != null & !"".equals(sourceLanguage)) {
                    sqlQuery += " AND lang = '" + sourceLanguage + "' ";
                }
                if (limit != 0) {
                    sqlQuery += " LIMIT " + sourceLanguage;
                }
                sqlQuery += ";";

                // System.out.println(sqlQuery);

                myResultSet = myDBConnection.executeQuery(sqlQuery);
                while (myResultSet.next()) {
                    myTreeSet.add(myResultSet.getString(1));
                }
                myResultSet.close();
            }

        } catch (PapillonBusinessException e) {
            PapillonLogger.writeDebugMsg("Exception in HeadwordListQuery : " + e.toString());

        } catch (SQLException se) {
            se.printStackTrace();
            //very important to throw out bad connections
            if (myDBConnection.handleException(se)) myDBConnection = null;

        } catch (Exception e) {
            e.printStackTrace();

        } finally {
            if (myDBConnection != null) {
                myDBConnection.reset();
                myDBConnection.release();
            }
        }

        //
        return myTreeSet;
    }

    public static String escapeForSql(String str) {
        return str == null ? null : StringNormalizer.normalize(str).replaceAll("'", "''");
    }

}
