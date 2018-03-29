/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.11  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.10  2006/03/27 11:47:24  mangeot
 * Added finitionDate in contributionLog
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
 * Revision 1.7.4.4  2005/05/31 14:22:34  mangeot
 * *** empty log message ***
 *
 * Revision 1.7.4.3  2005/05/25 21:00:36  mangeot
 * Bug fixes
 *
 * Revision 1.7.4.2  2005/05/20 10:27:34  mangeot
 * Added a contributors board in order to count the contribs
 *
 * Revision 1.7.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 * Revision 1.7  2005/04/15 11:38:05  mangeot
 * Fixed a bug, not using entryHandle from contributions table any more
 *
 * Revision 1.6  2005/04/14 13:08:25  mangeot
 * Deleted all references to findContributionByEntryHandle
 *
 * Revision 1.5  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.4.2.3  2005/04/01 15:16:05  mangeot
 * Added validated contributions count on GDEF homepage
 *
 * Revision 1.4.2.2  2005/03/30 11:17:07  mangeot
 * Modified table contributions: replaced originalhandle by originalid
 * Corrected a few bugs when validating an already existing entry
 *
 * Revision 1.4.2.1  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.4  2005/01/15 20:02:19  mangeot
 * Added new search options for ReviewContributions
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
import fr.imag.clips.papillon.CurrentDBTransaction;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.data.ContributionLogDO;
import fr.imag.clips.papillon.data.ContributionLogQuery;

import java.util.Vector;

/**
* Used to find the instances of xslsheet.
 */
public class ContributionsFactory {
				
	/* ContributionLog methods */
	
	public static ContributionLog newContributionLog(User myUser, VolumeEntry myEntry) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
			
			ContributionLog myContribLog = new ContributionLog();
			
			myContribLog.setAuthor(myUser.getLogin());
			myContribLog.setGroups(myUser.getGroups());
			myContribLog.setVolumeName(myEntry.getVolumeName());
			myContribLog.setSourceLanguage(myEntry.getSourceLanguage());
			myContribLog.setHeadword(myEntry.getHeadword());
			myContribLog.setEntryId(myEntry.getEntryId());
			myContribLog.setContributionId(myEntry.getContributionId());
			myContribLog.setStatus(myEntry.getStatus());
			myContribLog.setDate(new java.util.Date());
			
			return myContribLog;
		}
	
	public static void createContributionLogsFromExistingEntry(VolumeEntry myEntry) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			if (myEntry.getStatus().equals(VolumeEntry.FINISHED_STATUS) ||
				myEntry.getStatus().equals(VolumeEntry.REVIEWED_STATUS) ||
				myEntry.getStatus().equals(VolumeEntry.VALIDATED_STATUS) ||
				myEntry.getStatus().equals(VolumeEntry.REPLACED_STATUS)) {
				
				if (myEntry.getAuthor() !=null && !myEntry.getAuthor().equals("")) {
					ContributionLog myContribLog = new ContributionLog();
					
					myContribLog.setAuthor(myEntry.getAuthor());
					myContribLog.setGroups("");
					myContribLog.setVolumeName(myEntry.getVolumeName());
					myContribLog.setSourceLanguage(myEntry.getSourceLanguage());
					myContribLog.setHeadword(myEntry.getHeadword());
					myContribLog.setEntryId(myEntry.getEntryId());
					myContribLog.setContributionId(myEntry.getContributionId());
					if (myEntry.getFinitionDate() !=null) {
						myContribLog.setDate(myEntry.getFinitionDate());
					}
					else if (myEntry.getCreationDate() !=null) {
						myContribLog.setDate(myEntry.getCreationDate());
					}
					else {
						myContribLog.setDate(new java.util.Date());
					}
					myContribLog.setStatus(VolumeEntry.FINISHED_STATUS);
					myContribLog.save();
				}
				
				if (myEntry.getReviewer() !=null && !myEntry.getReviewer().equals("")) {
					ContributionLog myContribLog = new ContributionLog();
					
					myContribLog.setAuthor(myEntry.getReviewer());
					myContribLog.setGroups("");
					myContribLog.setVolumeName(myEntry.getVolumeName());
					myContribLog.setSourceLanguage(myEntry.getSourceLanguage());
					myContribLog.setHeadword(myEntry.getHeadword());
					myContribLog.setEntryId(myEntry.getEntryId());
					myContribLog.setContributionId(myEntry.getContributionId());
					if (myEntry.getReviewDate() !=null) {
						myContribLog.setDate(myEntry.getReviewDate());
					}
					else {
						myContribLog.setDate(new java.util.Date());
					}
					myContribLog.setStatus(VolumeEntry.REVIEWED_STATUS);
					myContribLog.save();
				}

				if (myEntry.getValidator() !=null && !myEntry.getValidator().equals("")) {
					ContributionLog myContribLog = new ContributionLog();
					
					myContribLog.setAuthor(myEntry.getValidator());
					myContribLog.setGroups("");
					myContribLog.setVolumeName(myEntry.getVolumeName());
					myContribLog.setSourceLanguage(myEntry.getSourceLanguage());
					myContribLog.setHeadword(myEntry.getHeadword());
					myContribLog.setEntryId(myEntry.getEntryId());
					myContribLog.setContributionId(myEntry.getContributionId());
					if (myEntry.getValidationDate() !=null) {
						myContribLog.setDate(myEntry.getValidationDate());
					}
					else {
						myContribLog.setDate(new java.util.Date());
					}
					myContribLog.setStatus(VolumeEntry.VALIDATED_STATUS);
					myContribLog.save();
				}
			}
		}
	
	public static ContributionLog newContributionLog(VolumeEntry myEntry, String author, String status, String date) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException {
			
			ContributionLog myContribLog = new ContributionLog();
			User myUser = fr.imag.clips.papillon.business.user.UsersFactory.findUserByLogin(author);
			java.util.Date resDate = null;
			try {
				java.text.DateFormat myDateFormat = new java.text.SimpleDateFormat("EEE MMM dd HH:mm:ss z yyyy");
				resDate = myDateFormat.parse(date);
			}
			catch (java.text.ParseException ex) {
				throw new PapillonBusinessException("Error parsing a date String", ex);
			}
			
			myContribLog.setAuthor(author);
			if (myUser !=null && !myUser.isEmpty()) {
				myContribLog.setGroups(myUser.getGroups());
			}
			else {
				myContribLog.setGroups("");
			}
			myContribLog.setVolumeName(myEntry.getVolumeName());
			myContribLog.setSourceLanguage(myEntry.getSourceLanguage());
			myContribLog.setHeadword(myEntry.getHeadword());
			myContribLog.setEntryId(myEntry.getEntryId());
			myContribLog.setContributionId(myEntry.getContributionId());
			myContribLog.setStatus(status);
			myContribLog.setDate(resDate);
			
			return myContribLog;
		}
	
	public static ContributionLog[] getContributionLogArray(String author, String status, String volume, java.util.Date fromDate, java.util.Date toDate)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
            ContributionLog[] theContribsArray = null;
            try {
				// consultation of a local volume
				ContributionLogQuery query = new ContributionLogQuery(CurrentDBTransaction.get());
				
				if (author != null && !author.equals("")) {
					query.getQueryBuilder().addWhereClause("author", author,
														   QueryBuilder.EQUAL);
				}
				if (status != null && !status.equals("")) {
					query.getQueryBuilder().addWhereClause("status", status,
														   QueryBuilder.EQUAL);
				}
				if (volume != null && !volume.equals("")) {
					query.getQueryBuilder().addWhereClause("volume", volume,
														   QueryBuilder.EQUAL);
				}
				if (fromDate != null) {
					query.getQueryBuilder().addWhereClause("date", new java.sql.Timestamp(fromDate.getTime()),
														   QueryBuilder.GREATER_THAN_OR_EQUAL);
				}
				if (toDate != null) {
					query.getQueryBuilder().addWhereClause("date", new java.sql.Timestamp(toDate.getTime()),
														   QueryBuilder.LESS_THAN_OR_EQUAL);
				}
				query.addOrderByHeadword(true);
				ContributionLogDO[] DOarray = query.getDOArray();
				theContribsArray = new ContributionLog[ DOarray.length ];
				if (null != DOarray) {
					for (int i=0; i < DOarray.length; i++) {
						theContribsArray[i] = new ContributionLog(DOarray[i]);
					}
				}
			}
			catch(Exception ex) {
				throw new PapillonBusinessException("Exception in getContributionLogArray()", ex);
			}
			return theContribsArray;
		}
	
	public static Vector getContributorsBoard(String volume, java.util.Date fromDate, java.util.Date toDate)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			Vector usersVector = new Vector();
			
			User[] UsersArray=fr.imag.clips.papillon.business.user.UsersFactory.getUsersArray("name");
			for (int i=0;i<UsersArray.length;i++) {
				String login = UsersArray[i].getLogin();
				ContributionLog[] myContribsArray = getContributionLogArray(login, null, volume, fromDate, toDate);
				if (myContribsArray != null && myContribsArray.length>0) {
					int finished = 0;
					int reviewed = 0;
					int validated = 0;
					for (int j=0; j< myContribsArray.length; j++) {
						String contribStatus = myContribsArray[j].getStatus();
						if (contribStatus.equals(VolumeEntry.FINISHED_STATUS)) {
							finished++;
						}
						else if (contribStatus.equals(VolumeEntry.REVIEWED_STATUS)) {
							reviewed++;
						}
						else if (contribStatus.equals(VolumeEntry.VALIDATED_STATUS)) {
							validated++;
						}
					}
					if (finished>0 || reviewed >0 || validated>0) {
						Vector myVector = new Vector();
						myVector.add(UsersArray[i]);
						myVector.add("" + finished);
						myVector.add("" + reviewed);
						myVector.add("" + validated);
						usersVector.add(myVector);
					}
				}
			}
			return usersVector;
		}
	
}

