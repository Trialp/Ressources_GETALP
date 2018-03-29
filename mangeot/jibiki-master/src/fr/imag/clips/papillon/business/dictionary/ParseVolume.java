/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.43.2.4  2008/01/09 19:07:44  serasset
 * Better cdm entry detection when parsing volume
 *
 * Revision 1.43.2.3  2007/11/14 15:41:20  serasset
 * Modified indexEntry to seperate index data extraction from index save.
 * This will allow for a more optimized version of reindexation.
 *
 * Revision 1.43.2.2  2007/09/05 15:24:13  serasset
 * Created a page to browse the dictionary index
 * Lexalp formatter now sorts entries correctly within a legal sysem/language group
 *
 * Revision 1.43.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.43  2007/02/08 15:24:07  fbrunet
 * *** empty log message ***
 *
 * Revision 1.42  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 * Revision 1.41  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.40  2006/08/10 22:17:12  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.39  2006/05/22 22:45:54  fbrunet
 * LexALP: add merge method in post-save processing (merge axies with same referenced lexies)
 *
 * Revision 1.38  2006/04/05 20:14:20  mangeot
 * bug fix
 *
 * Revision 1.37  2006/04/05 18:48:12  mangeot
 * bug fix
 *
 * Revision 1.36  2006/04/05 18:23:34  mangeot
 * Bug fix
 *
 * Revision 1.35  2006/04/05 12:40:25  mangeot
 * Fixed a confusion between the import of contribs versus entries
 *
 * Revision 1.34  2006/04/04 21:40:17  mangeot
 * Fixed a bug when importing contributions on one single line. We must test CDM_Contribution befire CDM_Entry
 *
 * Revision 1.33  2006/03/27 10:59:58  mangeot
 * Commented INDEXENTRY logs
 *
 * Revision 1.32  2006/03/06 10:06:23  mangeot
 * Horrible hack sur Home.java pour pouvoir utiliser tout de suite cette version sur le GDEF.
 * Another thing: the fuzzy search takes a lot of time, so I launch it only if the normal search returns no result.
 *
 * Revision 1.31  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.30  2006/02/26 14:08:16  mangeot
 * Added the multilingual_sort(lang,headword) index on volume tables for speeding up the lookup
 *
 * Revision 1.29  2006/02/22 19:05:56  mangeot
 * MM: Added default status choice when importing entries
 *
 * Revision 1.28  2006/02/09 10:49:28  mangeot
 * Added 3 new options when importing entries:
 * - ParseVolume.ReplaceExistingEntry_CopyAnyway
 * - ParseVolume.ReplaceExistingEntry_CopyIfSameStatus
 * - ParseVolume.ReplaceExistingEntry_CopyIfFinished
 * Creates a new entry id before importing the entry in the database
 * And added a log for the ParseVolume.MAX_DISCARDED_ENTRIES_LOGGED = 500
 * first entries that are discarded (not imported)
 *
 * Revision 1.27  2006/01/25 14:05:21  mangeot
 * MM: I modified the import of entries
 * Now, I check if an existing entry with the same id already exists in the database.
 * If yes, the user has to choose with several strategies:
 * - ParseVolume.ReplaceExistingEntry_Ignore
 * 	Do not import the new entry if there is already one in the
 * 	DB with the same entry
 * - ParseVolume.ReplaceExistingEntry_ReplaceAnyway
 * 	Replace the existing entry by the new one
 * - ParseVolume.ReplaceExistingEntry_ReplaceIfSameStatus
 * 	Replace the existing entry by the new one if their status is the same
 * - ParseVolume.ReplaceExistingEntry_ReplaceIfFinished
 * 	Replace the existing entry by the new one if the status of the
 * 	existing entry is 'not_finished' and the one of the new entry is
 * 	'finished'
 *
 * Revision 1.26  2005/12/04 15:22:39  mangeot
 * Fixed the volume parsing when the volume element is not the root element
 *
 * Revision 1.12  2005/09/17 11:15:39  mangeot
 * *** empty log message ***
 *
 * Revision 1.11.4.2  2006/02/24 13:59:55  fbrunet
 * Postprocessing during update and save
 *
 * Revision 1.11.4.1  2005/10/24 16:29:19  fbrunet
 * Added fuzzy search capabilities.
 * Added possibility to rebuild the index DB tables.
 * Added Pre and post processors that could be defined by the user.
 *
 * Revision 1.11  2005/07/16 16:25:26  mangeot
 * Adapted the linker to the GDEF project + bug fixes
 *
 * Revision 1.10  2005/06/24 10:35:57  mangeot
 * Minor bug fixes
 *
 * Revision 1.9  2005/06/23 09:48:17  mangeot
 * Bug fix in xpath completion and creation-date cdm element
 *
 * Revision 1.8  2005/06/22 15:55:53  mangeot
 * Solved an unresolved prefix bug when the dml prefix was not in the template entry.
 * Now we use the DmlPrefixResolver to solve this issue.
 *
 * Revision 1.7  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *-----------------------------------------------
 *
 * Created by Mathieu MANGEOT on Mon Sep 29 2003.
 *  Copyright (c) 2003-2005 mangeot & serasset CLIPS. All rights reserved.
 */

package fr.imag.clips.papillon.business.dictionary;

// internal imports

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.CurrentDBTransaction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;
import java.awt.List;
import java.sql.SQLException;

public class ParseVolume {

    // constants
    // public constants for import
    public static final int ReplaceExistingEntry_Ignore = 0;
    public static final int ReplaceExistingEntry_ReplaceAnyway = 1;
    public static final int ReplaceExistingEntry_ReplaceIfSameStatus = 2;
    public static final int ReplaceExistingEntry_ReplaceIfFinished = 3;
    public static final int ReplaceExistingEntry_CopyAnyway = 4;
    public static final int ReplaceExistingEntry_CopyIfSameStatus = 5;
    public static final int ReplaceExistingEntry_CopyIfFinished = 6;

    public static final int ReplaceExistingContribution_Ignore = 50;
    public static final int ReplaceExistingContribution_ReplaceAnyway = 51;
    public static final int ReplaceExistingContribution_ReplaceIfSameStatus = 52;
    public static final int ReplaceExistingContribution_ReplaceIfFinished = 53;
    
    public static final String ENTRIES_ADDED = " Entries added: ";
    public static final String ENTRIES_DISCARDED = " Entries discarded: ";


    public static final int MAX_DISCARDED_ENTRIES_LOGGED = 500;

    // constants for Foks
    protected static final int FOKS_TOKENS = 4;
    protected static final String FOKS_DELIMITER = "\t";
    protected static final String FOKS_ENCODING = "EUC-JP";
    protected static final String FOKSEDICT_VOLUME = "FoksEdict_jpn";

    // constants
    protected static final String ENCODING = "encoding";
    protected static final String DEFAULT_ENCODING = "UTF-8";
    protected static final String XMLHEADER = "<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>";
    protected static final String XMLDECLSTART = "<?xml ";
    protected static final int MAX_TRY = 5;
    public static final String UTF8_BOM = "\uFEFF";
    
    private static String removeUTF8BOM(String s) {
        if (s.startsWith(UTF8_BOM)) {
            s = s.substring(1);
        }
        return s;
    }

    public static int parseFoksVolume(String url)
            throws PapillonBusinessException {
        return parseFoksVolume(url, FOKS_ENCODING, FOKS_DELIMITER, FOKS_TOKENS);
    }

    protected static int parseFoksVolume(String urlString, String encoding, String delimiter, int tokensNb)
            throws PapillonBusinessException {
        int entry = 0;
        try {
            java.net.URL url = new java.net.URL(urlString);
            java.net.URLConnection connection = url.openConnection();

            java.io.InputStreamReader inReader = new java.io.InputStreamReader(connection.getInputStream(), encoding);
            java.io.BufferedReader buffer = new java.io.BufferedReader(inReader);
            String str = "";
            while (buffer.ready()) {
                str = buffer.readLine();
                java.util.StringTokenizer myTokenizer = new java.util.StringTokenizer(str, delimiter);
                if (myTokenizer.countTokens() == tokensNb) {
                    FoksEntry myEntry = new FoksEntry(myTokenizer.nextToken(), myTokenizer.nextToken(),
                            myTokenizer.nextToken(), myTokenizer.nextToken());
                    if (myEntry != null && !myEntry.isEmpty()) {
                        myEntry.save();
                        entry++;
                    }
                    if ((entry % 1000) == 0) {
                        PapillonLogger.writeDebugMsg("Foks Volume: " + entry + " entries parsed");
                    }
                }
            }
            buffer.close();
            PapillonLogger.writeDebugMsg("Foks Volume Parsed, " + entry + " entries added!");
        } catch (java.io.FileNotFoundException exp) {
            throw new PapillonBusinessException(
                    "ParseVolume.parseFoksVolume, error FileNotFoundException: " + urlString);

        } catch (java.io.IOException exp) {
            throw new PapillonBusinessException("ParseVolume.parseFoksVolume, error IOException");
        }
        return entry;
    }

    public static String parseVolume(String volumeName, java.net.URL myURL, String defaultStatus, int replaceExistingEntries,
                                     int replaceExistingContributions, boolean logContribs, boolean indexEntries)
            throws PapillonBusinessException {
        String message = "";
        Volume volume = VolumesFactory.getVolumeByName(volumeName);
        Dictionary dict = DictionariesFactory.getDictionaryByName(volume.getDictname());
        if (!volume.isEmpty()) {
            message = parseVolume(dict, volume, myURL, defaultStatus, replaceExistingEntries,
                    replaceExistingContributions, logContribs, indexEntries);
        } else {
            message = "The volume does not exist!";
        }
        return message;
    }

    protected static String parseVolume(Dictionary myDict, Volume myVolume, java.net.URL myURL, boolean logContribs, boolean indexEntries)
            throws PapillonBusinessException {
        return parseVolume(myDict, myVolume, myURL, VolumeEntry.FINISHED_STATUS, ReplaceExistingEntry_Ignore,
                ReplaceExistingContribution_Ignore, logContribs, indexEntries);
    }

    protected static String parseVolume(Dictionary myDict, Volume myVolume, java.net.URL myURL, String defaultStatus,
                                        int replaceExistingEntries, int replaceExistingContributions,
                                        boolean logContribs, boolean indexEntries)
    throws PapillonBusinessException {
        String xmlHeader = getXMLHeader(myURL, myVolume.getCdmEntry());
        String encoding = getEncoding(xmlHeader);
        java.io.InputStream is = null;
        try {
            is = myURL.openStream();
        }
        catch (java.io.IOException exp) {
            throw new PapillonBusinessException("ParseVolume.parseEntries, error IOException", exp);
        }
        return parseEntries(myDict, myVolume, is, encoding, defaultStatus, replaceExistingEntries,
                            replaceExistingContributions, logContribs, indexEntries);
    }
    
    public static String parseVolume(Dictionary myDict, Volume myVolume, String volumeString, String defaultStatus,
                                        int replaceExistingEntries, int replaceExistingContributions,
                                        boolean logContribs, boolean indexEntries)
    throws PapillonBusinessException {
        String encoding = getEncoding(volumeString);
        java.io.InputStream is = new java.io.ByteArrayInputStream(volumeString.getBytes());
        return parseEntries(myDict, myVolume, is, encoding, defaultStatus, replaceExistingEntries,
                            replaceExistingContributions, logContribs, indexEntries);
    }
    
    protected static String getXMLHeader(java.net.URL myUrl, String CDM_entry)
            throws PapillonBusinessException {
        String res = XMLHEADER;
        try {
            java.io.InputStream inStream = myUrl.openStream();
            // throws FileNotFoundException
            java.io.InputStreamReader inReader = new java.io.InputStreamReader(inStream, DEFAULT_ENCODING);
            // throws UnsupportedEncodingException
            java.io.BufferedReader buffer = new java.io.BufferedReader(inReader);
            String str = "";
            String header = "";
            int countTry = 0;
            while (buffer.ready() && (countTry <= MAX_TRY)) {
                str = buffer.readLine();
                countTry++;
                if (str.indexOf(XMLDECLSTART) >= 0) {
                    res = str.substring(str.indexOf(XMLDECLSTART));
                    break;
                }

                if (    (str.indexOf("<" + CDM_entry + " ") >= 0) ||
                        (str.indexOf("<" + CDM_entry + "\t") >= 0) ||
                        (str.indexOf("<" + CDM_entry) + 1 + CDM_entry.length() == str.length()) ||
                        (str.indexOf("<" + CDM_entry + ">") >= 0)) {
                    break;
                }
            }
            inStream.close();
        } catch (java.io.IOException exp) {
            throw new PapillonBusinessException("ParseVolume.getXMLHeader, error IOException URL: " + myUrl.toString(), exp);
        }
        return res;
    }

    protected static String getEncoding(String XMLHeader) {
        String res = DEFAULT_ENCODING;
        int start = XMLHeader.indexOf(ENCODING);
        if (start > 0) {
            // looking for the string encoding
            String tmpString = XMLHeader.substring(start + ENCODING.length());
            tmpString = tmpString.trim();
            // looking for the '='
            start = tmpString.indexOf('=');
            if (start >= 0) {
                tmpString = tmpString.substring(start + 1);
                tmpString = tmpString.trim();
                // cut the quotes
                if (tmpString.length() > 2) {
                    String quote = tmpString.substring(0, 1);
                    tmpString = tmpString.substring(1);
                    int end = tmpString.indexOf(quote);
                    if (end > 0) {
                        res = tmpString.substring(0, end);
                    }
                }
            }
        }
        return res;
    }

        /*
	 parseEntries split the input stream into entries. Allows to discard malformed entries
	 
	 Rebuilds the XML header until the entry tag
	 
	 TODO: 
	 - does not work if the entry tag is inside two comments on the same line
	 - does not work if an entry tag is in a CDATA section.
	 
	 */
    protected static String parseEntries(Dictionary myDict, Volume myVolume, java.io.InputStream inStream, String encoding,
                                         String defaultStatus, int replaceExistingEntries,
                                         int replaceExistingContributions, boolean logContribs, boolean indexEntry)
            throws PapillonBusinessException {
        PapillonLogger.writeDebugMsg("parseEntries, encoding: [" + encoding + "]");
        int countEntries = 0;
        String message = "";
        java.util.Vector ParsedEntries = new Vector();
        java.util.Vector DiscardedEntries = new Vector();
        try {
            java.io.InputStreamReader inReader = new java.io.InputStreamReader(inStream, encoding);
            // throws UnsupportedEncodingException
            java.io.BufferedReader buffer = new java.io.BufferedReader(inReader);
            StringBuffer xmlHeaderBuffer = new StringBuffer();
            String CDM_Volume = myVolume.getCdmVolume();
            String CDM_Entry = myVolume.getCdmEntryString();
            String CDM_Contribution = myVolume.getCdmContribution();
            boolean isContributionVolume = false;

            PapillonLogger.writeDebugMsg("parseEntries, logContribs: [" + logContribs + "]");
            PapillonLogger.writeDebugMsg("parseEntries, CDM_Volume: [" + CDM_Volume + "]");
            PapillonLogger.writeDebugMsg("parseEntries, CDM_Entry: [" + CDM_Entry + "]");
            PapillonLogger.writeDebugMsg("parseEntries, CDM_Contribution: [" + CDM_Contribution + "]");
            String bufferLine = "";
			boolean inComment = false;
            int beginComment = -1;
            int endComment = -1;
			boolean inCDATA = false;
            int beginCDATA = -1;
            int endCDATA
			= -1;
            int firstEntryIndex = -1;
            while (buffer.ready() && firstEntryIndex < 0) {
                bufferLine = buffer.readLine();
				if (!inComment) {
					beginComment = bufferLine.lastIndexOf("<!--");
					if (beginComment>=0) {
						endComment = bufferLine.lastIndexOf("-->");
						if (endComment>= 0 && endComment > beginComment) {
							/* The choice for the moment: keeping the comments in the header */
							/*
							bufferLine = bufferLine.substring(0,beginComment) + bufferLine.substring(endComment+3);
							 */
							inComment = false;
						}
						else {
							/* The choice for the moment: keeping the comments in the header */
							/*
							bufferLine = bufferLine.substring(0,beginComment);
							 */
							inComment = true;
						}
					}
				}
				else {
					endComment = bufferLine.lastIndexOf("-->");
					if (endComment>= 0) {
						inComment = false;
						/* The choice for the moment: keeping the comments in the header */
						/*bufferLine = bufferLine.substring(endComment+3);*/
					}
				}
				//PapillonLogger.writeDebugMsg("buffer: [" + bufferLine + "]");
				//PapillonLogger.writeDebugMsg("begin: " + beginComment + " end: " + endComment + " in: " + inComment);
				if (!inComment) {
                firstEntryIndex = bufferLine.indexOf("<" + CDM_Contribution + " ");
                if (firstEntryIndex < 0) {
                    firstEntryIndex = bufferLine.indexOf("<" + CDM_Contribution + "\t");
                }
				if (firstEntryIndex<0) {
					firstEntryIndex = bufferLine.indexOf("<" + CDM_Contribution);
                    int length = bufferLine.length();
                    int filength = firstEntryIndex + 1 + CDM_Contribution.length();
                    if (firstEntryIndex >=0 && filength != length) {
                        firstEntryIndex = -1;
                    }
				}
                if (firstEntryIndex < 0) {
                    firstEntryIndex = bufferLine.indexOf("<" + CDM_Contribution + ">");
                }
                isContributionVolume = (firstEntryIndex >= 0);
               if (firstEntryIndex < 0) {
                    firstEntryIndex = bufferLine.indexOf("<" + CDM_Entry + " ");
                }
                if (firstEntryIndex < 0) {
                    firstEntryIndex = bufferLine.indexOf("<" + CDM_Entry + "\t");
                }
                if (firstEntryIndex<0) {
                    firstEntryIndex = bufferLine.indexOf("<" + CDM_Entry);
                    int length = bufferLine.length();
                    int filength = firstEntryIndex + 1 + CDM_Entry.length();
                    if (firstEntryIndex >=0 && filength != length) {
                        firstEntryIndex = -1;
                    }
                }
                if (firstEntryIndex < 0) {
                    firstEntryIndex = bufferLine.indexOf("<" + CDM_Entry + ">");
                }
				}
                if (firstEntryIndex>=0 && firstEntryIndex>endComment) {
                    if (firstEntryIndex > 0) {
                        xmlHeaderBuffer.append(bufferLine.substring(0, firstEntryIndex));
						xmlHeaderBuffer.append("\n");
                        bufferLine = bufferLine.substring(firstEntryIndex);
                    }
				}
				else if (firstEntryIndex>=0 && firstEntryIndex<=endComment) {
					firstEntryIndex = -1;
				}
				/* The choice for the moment: keeping the comments in the header */
				/*
				else if (inComment) {
					if (beginComment>=0) {
						xmlHeaderBuffer.append(bufferLine);
						beginComment = -1;
					}
                }*/
				else {
                    if (bufferLine.startsWith(UTF8_BOM) && encoding.equalsIgnoreCase(DEFAULT_ENCODING)) {
                        bufferLine = bufferLine.substring(1);
                    }
                    xmlHeaderBuffer.append(bufferLine);
					xmlHeaderBuffer.append("\n");
                }
				endComment = -1;
            }
            StringBuffer xmlFooterBuffer = new StringBuffer();
            if (isContributionVolume) {
                CDM_Entry = CDM_Contribution;
            } else {
                xmlHeaderBuffer.append(VolumeEntry.getContributionHeader(myVolume.getTemplateEntry()));
                xmlFooterBuffer.append(VolumeEntry.getContributionFooter(myVolume.getTemplateEntry()));
            }
			//PapillonLogger.writeDebugMsg("Header [" + xmlHeaderBuffer.toString() + "]");
            PapillonLogger.writeDebugMsg("Will parse [" + CDM_Entry + "]");
            PapillonLogger.writeDebugMsg("XML footer [" + myVolume.getXmlFooter() + "]");
            xmlFooterBuffer.append(myVolume.getXmlFooter());
            StringBuffer entryBuffer = new StringBuffer();
            entryBuffer.append(xmlHeaderBuffer);
            while (buffer.ready() && bufferLine.indexOf("</" + CDM_Volume + ">") < 0) {
                int entryIndex = bufferLine.indexOf("<" + CDM_Entry + " ");
                entryIndex = (entryIndex<0) ? bufferLine.indexOf("<" + CDM_Entry + "\t") : entryIndex;
                if (entryIndex<0) {
                    entryIndex = (entryIndex<0) ? bufferLine.indexOf("<" + CDM_Entry) : entryIndex;
                    if (entryIndex >=0 && bufferLine.length() != entryIndex + 1 + CDM_Entry.length()) {
                        entryIndex = -1;
                    }
                }
                entryIndex = (entryIndex<0) ? bufferLine.indexOf("<" + CDM_Entry + ">") : entryIndex;
                
                while (entryIndex >= 0) {
                    if (entryIndex > 0) {
                        entryBuffer.append(bufferLine.substring(0, entryIndex));
                        entryBuffer.append("\n");
                        bufferLine = bufferLine.substring(entryIndex);
                    }
                    if (entryBuffer.length() > xmlHeaderBuffer.length()) {
                        // PapillonLogger.writeDebugMsg("Middle call: parseEntry " + entryBuffer.toString());
                       if (parseEntry(myDict, myVolume, entryBuffer.append(xmlFooterBuffer), defaultStatus,
                                isContributionVolume, replaceExistingEntries, replaceExistingContributions, logContribs,
                                       indexEntry, ParsedEntries, DiscardedEntries)) {
                            countEntries++;
                        }
                        entryBuffer = new StringBuffer();
                        entryBuffer.append(xmlHeaderBuffer);
                    }
                    entryIndex = bufferLine.indexOf("<" + CDM_Entry + " ", entryIndex + CDM_Entry.length() + 1);
					entryIndex = (entryIndex<0) ? bufferLine.indexOf("<" + CDM_Entry + "\t", entryIndex + CDM_Entry.length() + 1) : entryIndex;
                    if (entryIndex<0) {
                        entryIndex = bufferLine.indexOf("<" + CDM_Entry, entryIndex + CDM_Entry.length() + 1);
                        if (entryIndex >=0 && bufferLine.length() != entryIndex + 1 + CDM_Entry.length()) {
                            entryIndex = -1;
                        }
                    }
					entryIndex = (entryIndex<0) ? bufferLine.indexOf("<" + CDM_Entry + ">", entryIndex + CDM_Entry.length() + 1) : entryIndex;
                }
                entryBuffer.append(bufferLine);
                bufferLine = buffer.readLine() + "\n";
            }
            buffer.close();
            inStream.close();
            int closeVolume = bufferLine.indexOf("</" + CDM_Volume + ">");
            if (closeVolume > 0) {
                entryBuffer.append(bufferLine.substring(0, closeVolume));
            }
           // PapillonLogger.writeDebugMsg("Final call: parseEntry " + entryBuffer.toString());
            if (parseEntry(myDict, myVolume, entryBuffer.append(xmlFooterBuffer), defaultStatus, isContributionVolume,
                    replaceExistingEntries, replaceExistingContributions, logContribs, indexEntry, ParsedEntries, DiscardedEntries)) {
                countEntries++;
            }
            message = "volume parsed, " + countEntries + " entries added. ";
            if (ParsedEntries.size()>0) {
                message += ENTRIES_ADDED + new org.json.JSONArray((java.util.Collection)ParsedEntries).toString();
            }
            if (DiscardedEntries.size()>0) {
                message += DiscardedEntries.size() + ENTRIES_DISCARDED + new org.json.JSONArray((java.util.Collection)DiscardedEntries).toString();
            }
            PapillonLogger.writeDebugMsg(message);
        } catch (java.io.FileNotFoundException exp) {
            throw new PapillonBusinessException("FileNotFoundException: " + myVolume.getVolumeRef(), exp);

        } catch (java.io.IOException exp) {
            throw new PapillonBusinessException("ParseVolume.parseEntries, error IOException", exp);
        } 
        return message;
    }

    protected static boolean parseEntry(Dictionary myDict, Volume myVolume, StringBuffer entryBuffer,
                                        String defaultStatus, boolean isContributionVolume, int replaceExistingEntries,
                                        int replaceExistingContributions, boolean logContribs, boolean indexEntry,
                                        java.util.Vector ParsedEntries, java.util.Vector DiscardedEntries)
             throws PapillonBusinessException {
        boolean result = false;
				org.w3c.dom.Document myDoc = null;
				try {
					myDoc = XMLServices.buildDOMTree(entryBuffer.toString());
				}
				catch (Exception e) {
                    myDoc = null;
				}
        if (myDoc != null) {
            VolumeEntry newEntry = new VolumeEntry(myDict, myVolume);
            newEntry.setDom(myDoc);
            newEntry.setAuthor();
            newEntry.setCreationDate();
            newEntry.setHeadword();
            newEntry.setStatusIfNotNull(defaultStatus);
            String entryId = "";
            if (isContributionVolume) {
                entryId = newEntry.getContributionId();
//                PapillonLogger.writeDebugMsg("Entry " + entryBuffer.toString() + "New contributionId "+ entryId);
                if (entryId != null && !entryId.equals("")) {
                    VolumeEntry existingEntry = VolumeEntriesFactory.findEntryByContributionId(myDict, myVolume,
                            entryId);
                    if (existingEntry != null && !existingEntry.isEmpty()) {
                        switch (replaceExistingContributions) {
                            case ReplaceExistingContribution_Ignore:
                                break;
                            case ReplaceExistingContribution_ReplaceAnyway:
                                result = newEntry.save(indexEntry);
                                existingEntry.delete();
                                break;
                            case ReplaceExistingContribution_ReplaceIfSameStatus:
                                if (newEntry.getStatus().equals(existingEntry.getStatus())) {
                                    result = newEntry.save(indexEntry);
                                    existingEntry.delete();
                                }
                                break;
                            case ReplaceExistingContribution_ReplaceIfFinished:
                                if (existingEntry.getStatus().equals(
                                        VolumeEntry.NOT_FINISHED_STATUS) && newEntry.getStatus().equals(
                                        VolumeEntry.FINISHED_STATUS)) {
                                    result = newEntry.save(indexEntry);
                                    existingEntry.delete();
                                }
                                break;
                            default:
                                break;
                        }
                    } else {
                        result = newEntry.save(indexEntry);
                    }
                } else {
                    result = newEntry.save(indexEntry);
                }
            } else {
                entryId = newEntry.getEntryId();
                if (entryId != null && !entryId.equals("")) {
                    VolumeEntry existingEntry = VolumeEntriesFactory.findEntryByEntryId(myDict, myVolume, entryId);
                    if (existingEntry != null && !existingEntry.isEmpty()) {
                        switch (replaceExistingEntries) {
                            case ReplaceExistingEntry_Ignore:
                                break;
                            case ReplaceExistingEntry_ReplaceAnyway:
                                result = newEntry.save(indexEntry);
                                existingEntry.delete();
                                break;
                            case ReplaceExistingEntry_ReplaceIfSameStatus:
                                if (newEntry.getStatus().equals(existingEntry.getStatus())) {
                                    result = newEntry.save(indexEntry);
                                    existingEntry.delete();
                                }
                                break;
                            case ReplaceExistingEntry_ReplaceIfFinished:
                                if (existingEntry.getStatus().equals(
                                        VolumeEntry.NOT_FINISHED_STATUS) && newEntry.getStatus().equals(
                                        VolumeEntry.FINISHED_STATUS)) {
                                    result = newEntry.save(indexEntry);
                                    existingEntry.delete();
                                }
                                break;
                            case ReplaceExistingEntry_CopyAnyway:
                                newEntry.setEntryId();
                                result = newEntry.save(indexEntry);
                                break;
                            case ReplaceExistingEntry_CopyIfSameStatus:
                                if (newEntry.getStatus().equals(existingEntry.getStatus())) {
                                    newEntry.setEntryId();
                                    result = newEntry.save(indexEntry);
                                }
                                break;
                            case ReplaceExistingEntry_CopyIfFinished:
                                if (existingEntry.getStatus().equals(
                                        VolumeEntry.NOT_FINISHED_STATUS) && newEntry.getStatus().equals(
                                        VolumeEntry.FINISHED_STATUS)) {
                                    newEntry.setEntryId();
                                    result = newEntry.save(indexEntry);
                                }
                                break;
                            default:
                                break;
                        }
                    } else {
                        result = newEntry.save(indexEntry);
                    }
                } else {
                    result = newEntry.save(indexEntry);
                }
            }
            if (result) {
                if (logContribs) {
                    ContributionsFactory.createContributionLogsFromExistingEntry(newEntry);
                }
                PapillonLogger.writeDebugMsg("Adding entry " + newEntry.getHeadword() + " // " + newEntry.getId());
                if (indexEntry) {
                    if (ParsedEntries.size() < MAX_DISCARDED_ENTRIES_LOGGED) {
                        ParsedEntries.add(newEntry.getContributionId());
                    }
                }
                
            } else {
                PapillonLogger.writeDebugMsg("Discarding entry " + newEntry.getHeadword() + " // " + newEntry.getId());
                if (DiscardedEntries.size() < MAX_DISCARDED_ENTRIES_LOGGED) {
                    DiscardedEntries.add(entryId);
                }
            }
        } else {
            PapillonLogger.writeDebugMsg("Entry not valid:[" + entryBuffer.toString() + "]");
        }
        return result;
    }
    
}
