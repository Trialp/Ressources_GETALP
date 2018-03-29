/* Jibiki project
 * © jibiki developpers, GETA-CLIPS, IMAG.
 *-----------------------------------------------------------------------------
 * $Id$
 *-----------------------------------------------------------------------------
 * $Log$
 * Revision 1.3  2007/04/05 13:56:23  serasset
 * Corrected UserLanguage (slo -> slv)
 *
 * Revision 1.2  2007/04/05 12:55:54  serasset
 * Added a DBLayer Version management with an auto-update of db layer.
 *
 * Revision 1.1  2007/01/05 13:57:26  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 *-----------------------------------------------------------------------------
*/

package fr.imag.clips.papillon.papillon_data;

import com.lutris.appserver.server.sql.DatabaseManagerException;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.appserver.server.Enhydra;
import com.lutris.dods.builder.generator.query.DataObjectException;
import com.lutris.dods.builder.generator.query.NonUniqueQueryException;
import com.lutris.dods.builder.generator.query.QueryException;
import com.lutris.dods.builder.generator.query.RefAssertionException;
import com.lutris.util.ConfigException;
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.data.JibikiVersionDO;
import fr.imag.clips.papillon.data.JibikiVersionQuery;

import java.sql.SQLException;
import java.util.Collection;
import java.util.Iterator;

/**
 * DataLayerVersion: allows version recognition of the data layer of the current db.
 */
public class DataLayerVersion {

    public static int getDBVersion() {
        // Query the jibikiVersion
        JibikiVersionQuery jvq = new JibikiVersionQuery(CurrentDBTransaction.get());
        jvq.requireUniqueInstance();
        try {
            JibikiVersionDO[] jva = jvq.getDOArray();
            return (jva == null || jva.length == 0) ? 0 : jva[0].getNumber();
        } catch (DataObjectException e) {
            PapillonLogger.writeDebugMsg(e.getLocalizedMessage());
        } catch (NonUniqueQueryException e) {
            PapillonLogger.writeDebugMsg(e.getLocalizedMessage());
        }
        return 0;
    }

    public static int getApplicationVersion() {
        // Retrieve the application version.
        try {
            String version = Enhydra.getApplication().getConfig().getString("jibiki.dblayer.version");
            return Integer.parseInt(version);
        } catch (ConfigException e) {
            return 0;
        }
    }

    public static void setDBVersion(int v) throws PapillonBusinessException {
        JibikiVersionQuery jvq = new JibikiVersionQuery(CurrentDBTransaction.get());
        jvq.requireUniqueInstance();
        try {
            JibikiVersionDO[] jva = jvq.getDOArray();
            JibikiVersionDO jvo;
            if (jva == null || jva.length == 0) {
                jvo = JibikiVersionDO.createVirgin(CurrentDBTransaction.get());
            } else {
                jvo = jva[0];
            }
            jvo.setNumber(v);
            jvo.save();
        } catch (DataObjectException e) {
            PapillonLogger.writeDebugMsg(e.getLocalizedMessage());
        } catch (NonUniqueQueryException e) {
            PapillonLogger.writeDebugMsg(e.getLocalizedMessage());
        } catch (SQLException e) {
            PapillonLogger.writeDebugMsg(e.getLocalizedMessage());
        } catch (DatabaseManagerException e) {
            PapillonLogger.writeDebugMsg(e.getLocalizedMessage());
        } catch (QueryException e) {
            PapillonLogger.writeDebugMsg(e.getLocalizedMessage());
        } catch (ObjectIdException e) {
            PapillonLogger.writeDebugMsg(e.getLocalizedMessage());
        } catch (RefAssertionException e) {
            PapillonLogger.writeDebugMsg(e.getLocalizedMessage());
        }
    }

    // WARNING: This method should be called AFTER CACHE INITIALISATION.
    public static boolean upgradeDB() throws PapillonBusinessException {
		boolean result = false;
        int currentDBVersion = getDBVersion();
        int currentApplicationVersion = getApplicationVersion();
        if (currentDBVersion < currentApplicationVersion) {
            PapillonLogger.writeDebugMsg("Upgrading database from version " + currentDBVersion + " to version " + currentApplicationVersion);
            result = true;
			if (currentDBVersion <= 0) {
                // Check if the version column do exist and create it if necessary
                if (!ManageDatabase.getTableNames().contains("jibikiversion")) {
                    ManageDatabase.executeSql(
                            "create table jibikiversion \n" +
                                    "( /* class JibikiVersion*/ \n" +
                                    "    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY , \n" +
                                    "    ObjectVersion INTEGER  NOT NULL , \n" +
                                    "    number INTEGER NOT NULL \n" +
                                    ");\n");
                    PapillonLogger.writeDebugMsg("jibikiversion table was created in the database");
                }
			}

				
				/*
				 BUG47
				 Deleted the field cdmelements from the volumes table.
				 It was a serialization of an array, thus causing problems when
				 changing the version of java.
				 It is now built at server statup when adding a volume in cache.
				 check if cdmelements column is present in all volumes
				 delete it */
				
				if (ManageDatabase.getColumnNames("volumes").contains("cdmelements")) {
					ManageDatabase.executeSql(
											  "alter TABLE volumes drop COLUMN cdmelements ;\n");
					PapillonLogger.writeDebugMsg("'cdmelements' column dropped in table volumes");                        
				}
				
				
                Collection vols = VolumesFactory.getVolumes();
			if (vols!=null) {
                Iterator volsIter = vols.iterator();
                while (volsIter.hasNext()) {
                    Volume vol = (Volume) volsIter.next();
                    PapillonLogger.writeDebugMsg("Upgrading volume " + vol.getName());
                    // check if htmldom column is present in all volumes
                    // delete it if yes
                    if (ManageDatabase.getColumnNames(vol.getDbname()).contains("htmldom")) {
                        ManageDatabase.executeSql(
                                "alter TABLE " + vol.getDbname() + " drop COLUMN htmldom ;\n");
                        PapillonLogger.writeDebugMsg("'htmldom' column dropped in table " + vol.getDbname());
                    }
                    // check if dom column is present in all volumes
                    // delete it
                    if (ManageDatabase.getColumnNames(vol.getDbname()).contains("dom")) {
                        ManageDatabase.executeSql(
                                "alter TABLE " + vol.getDbname() + " drop COLUMN dom ;\n");
                        PapillonLogger.writeDebugMsg("'dom' column dropped in table " + vol.getDbname());                        
                    }
					
					/*
					 BUG310
					 I recently found that the table field on where a join is performed must be of the same type.
					 We had volume.objectid decimal (19,0) with idxvolume.entryid varchar 255.
					 
					 I changed this and run tests.
					 It appears that the search is much faster on very bug dictionaries when:
					 - all field of the criteria are filled (key, value, lang)
					 - there is only one criterion.
					 
					 If there is more than one criterion and the lang field is missing, the query stucks (I stopped after 30 minutes).
					 If there is more than one criterion and the lang field is filled, the query is just a little bit faster.
					 
					 More tests should be performed in order to ensure that the existing queries will not be stuck. */
					
					
					//if entryid column type is different from objectid column type
					if (!ManageDatabase.getColumnType(vol.getIndexDbname(),"entryid").equals("_numeric")) {
                        PapillonLogger.writeDebugMsg("change entryid  column type in table " + vol.getDbname());                        
                        PapillonLogger.writeDebugMsg("IT TAKES TIME, BE PATIENT.. ");                        
						ManageDatabase.executeSql("alter table "+ vol.getIndexDbname() + " add column entryid2 numeric(19,0) not null default 0;\n");
						ManageDatabase.executeSql("update "+ vol.getIndexDbname() + " set entryid2=int4(entryid);\n");
						ManageDatabase.executeSql("alter table "+ vol.getIndexDbname() + " drop column entryid;\n");
						ManageDatabase.executeSql("alter table "+ vol.getIndexDbname() + " rename column entryid2 to entryid;\n");
						ManageDatabase.executeSql("create index "+ vol.getIndexDbname() + "_entryid_idx on "+vol.getIndexDbname() +"(entryid);\n");
                        PapillonLogger.writeDebugMsg("entryid column type in table " + vol.getDbname() + " changed!");                        
					}
                }
			}
			/*
			 FEAT443
			 Ajout d'un champ crédits dans l'objet utilisateur pour éventuellement gérer
			 une somme de crédits (ils s'acquièrent en contribuant).*/			
			if (!ManageDatabase.getColumnNames("users").contains("credits")) {
				ManageDatabase.executeSql("alter TABLE users add COLUMN credits integer not null default 0;\n");
				PapillonLogger.writeDebugMsg("'entries' column added in table users");                        
			}
			
			/*
			 BUG444
			 The volumes table is very slow to display the first time it is called because we count the entries of each volume every
			 time the server is starting.
			 I added a field in the volumes table in order to store the entries count.*/			
			if (!ManageDatabase.getColumnNames("volumes").contains("entries")) {
				ManageDatabase.executeSql("alter TABLE volumes add COLUMN entries integer not null default 0;\n");
				PapillonLogger.writeDebugMsg("'entries' column added in table volumes");                        
			}
			
			//if entryid column type is different from objectid column type
			if (!ManageDatabase.getColumnType("users","password").equals("_varchar")) {
				PapillonLogger.writeDebugMsg("change password column type in table users");                        
				ManageDatabase.executeSql("alter table users alter column password type VARCHAR(255);\n");
				PapillonLogger.writeDebugMsg("!!!! you must reset all passwords!!!");                        
			}

            if (!ManageDatabase.getColumnNames("users").contains("creationdate")) {
                ManageDatabase.executeSql("alter TABLE users add COLUMN creationdate DATE;\n");
                PapillonLogger.writeDebugMsg("'creationdate' column added in table users");
            }
            
            if (!ManageDatabase.getColumnNames("users").contains("modificationdate")) {
                ManageDatabase.executeSql("alter TABLE users add COLUMN modificationdate DATE;\n");
                PapillonLogger.writeDebugMsg("'modificationdate' column added in table users");
            }
            
            if (!ManageDatabase.getColumnNames("dictionaries").contains("access")) {
                ManageDatabase.executeSql("alter TABLE dictionaries add COLUMN access VARCHAR(255) not null default 'public';\n");
                PapillonLogger.writeDebugMsg("'access' column added in table dictionaries");
            }
            
			
			PapillonLogger.writeDebugMsg("If columns were created or dropped, consider vacuum your database!");
            setDBVersion(currentApplicationVersion);
            PapillonLogger.writeDebugMsg("jibikiversion value incremented.");            
        }
		return result;
    }
}
