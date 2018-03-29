/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.15  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.14  2006/02/26 14:04:56  mangeot
 * Corrected a bug: the content was a static variable, thus there were problems when two users wanted to aces the same page at the same time
 *
 * Revision 1.13  2006/01/25 10:13:06  mangeot
 * *** empty log message ***
 *
 * Revision 1.12  2005/08/05 19:55:39  mangeot
 * *** empty log message ***
 *
 * Revision 1.11  2005/08/05 18:49:17  mangeot
 * *** empty log message ***
 *
 * Revision 1.10  2005/08/05 18:44:38  mangeot
 * Bug fixes + ProcessVolume.po page creation
 *
 * Revision 1.9  2005/08/02 08:27:16  mangeot
 * Now, the display of an entry with EditEntryInit is done in the page itself.
 *
 * Revision 1.8  2005/07/30 13:52:13  mangeot
 * Commit due to some conflicts between directories. Beware !
 *
 * Revision 1.7  2005/07/28 16:40:33  mangeot
 * *** empty log message ***
 *
 * Revision 1.6  2005/07/28 15:34:34  mangeot
 * *** empty log message ***
 *
 * Revision 1.5  2005/07/28 14:59:07  mangeot
 * *** empty log message ***
 *
 * Revision 1.4  2005/07/28 13:06:47  mangeot
 * - Added the possibility to export in PDF format. The conversion into PDF is don
 * e via the fop package that has to be installed (see ToolsForPapillon)
 *
 * Revision 1.3  2005/07/21 15:09:20  mangeot
 * Bug fixes and corrections mainly for the GDEF project
 *
 * Revision 1.2  2005/06/15 16:48:28  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.1.2.4  2005/05/20 16:54:53  mangeot
 * Added ExportVolume functionnality
 *
 * Revision 1.1.2.3  2005/05/20 15:56:47  mangeot
 * *** empty log message ***
 *
 * Revision 1.1.2.2  2005/05/20 14:43:48  mangeot
 * Repair mismatch in branch tag
 *
 * Revision 1.1.2.1  2005/05/20 10:31:46  mangeot
 * Added 2 new classes 1 contributors board and one for exporting a volume
 *
 *
 *-----------------------------------------------
 * Papillon Export Volume page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;
import com.lutris.appserver.server.session.Session;


// manipulations HTML
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

import fr.imag.clips.papillon.business.message.MessageDBLoader;
import fr.imag.clips.papillon.presentation.PapillonSessionData;


// Standard imports
import java.io.IOException;
import java.util.Date;
import java.util.Iterator;
import java.text.DateFormat;
import java.io.*;

import fr.imag.clips.papillon.presentation.xhtml.orig.ExportVolumeTmplXHTML;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.IQuery;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;


public class ExportVolume extends PapillonBasePO {
	
	protected final static String ALL="*ALL*";
	protected final static String DEFAULT="*default*";
	protected final static String SORTBY_PARAMETER="SortBy";
	protected final static String AnyContains_PARAMETER="AnyContains";
    protected final static String BASE_DIR_CONFIG = "Papillon.Informations.baseDir";
    protected final static String RELATIVE_DIR_CONFIG = "Papillon.Informations.relativeDir";
    protected final static String MEDIA_DIR_CONFIG = "Papillon.Informations.mediaDir";
	protected final static String EXPORT_VOLUME_DIR="export";
	
	protected static final String XMLFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XML_DIALECT);
	protected static final String XHTMLFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XHTML_DIALECT);
	protected static final String TEXTFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.PLAINTEXT_DIALECT);
	protected static final String PDFFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.PDF_DIALECT);

	protected static java.util.regex.Pattern quotePattern = java.util.regex.Pattern.compile("'");

    protected ExportVolumeTmplXHTML content;

    protected boolean loggedInUserRequired() {
        return true;
    }

    protected boolean userMayUseThisPO() {
        try {
            return this.getUser().isSpecialist();
        } catch (fr.imag.clips.papillon.business.PapillonBusinessException ex) {
            this.getSessionData().writeUserMessage("Error getting the authorisation to use this PO.");
        }
        return false;
    }
	
    protected  int getCurrentSection() {
        return NO_SECTION;
    }


    public Node getContent()
        throws javax.xml.parsers.ParserConfigurationException,
			HttpPresentationException,
		    IOException, org.xml.sax.SAXException,
			javax.xml.transform.TransformerException,
			fr.imag.clips.papillon.presentation.PapillonPresentationException {
        
        // Création du contenu
        content = (ExportVolumeTmplXHTML)MultilingualXHtmlTemplateFactory.createTemplate("ExportVolumeTmplXHTML", this.getComms(), this.getSessionData());
	  
        HttpPresentationRequest req = this.getComms().request;

		String export = myGetParameter(content.NAME_EXPORT);
		String exportList = myGetParameter(content.NAME_EXPORT_LIST);
		String entryList = myGetParameter(content.NAME_ENTRY_LIST);
		String volume = myGetParameter(content.NAME_VOLUME);
		String outputFormat = myGetParameter(content.NAME_FORMAT);

        // If the page is called with parameters, take the requested action
		if (export != null && volume != null) {
					
			// search1
			String search1 = myGetParameter(content.NAME_search1);
			String search1text = myGetParameter(content.NAME_search1text);
			
			// search2
			String search2 = myGetParameter(content.NAME_search2);
			String search2text = myGetParameter(content.NAME_search2text);
			
			// search3
			String search3 = myGetParameter(content.NAME_search3);
			String search3text = myGetParameter(content.NAME_search3text);
			
			// search4
			String search4 = myGetParameter(content.NAME_search4);
			String search4text = myGetParameter(content.NAME_search4text);
			
			String anyContains = null;
			
			
			if (null != search1 && null != search1text && !search1text.equals("")) {
				if (search1.equals(AnyContains_PARAMETER)) {
					anyContains = search1text;
				}
			}			
			if (null != search2 && null != search2text && !search2text.equals("")) {
				if (search2.equals(AnyContains_PARAMETER)) {
					anyContains = search2text;
				}
			}
			if (null != search3 && null != search3text && !search3text.equals("")) {
				if (search3.equals(AnyContains_PARAMETER)) {
					anyContains = search3text;
				}
			}
			if (null != search4 && null != search4text && !search4text.equals("")) {
				if (search4.equals(AnyContains_PARAMETER)) {
					anyContains = search4text;
				}
			}
			
			// strategy1
			String strategyString1 = myGetParameter(content.NAME_Strategy1);
			int strategy1 = IQuery.STRATEGY_NONE;
			if (null != strategyString1 && !strategyString1.equals("")) {
				strategy1 = Integer.parseInt(strategyString1);
			}
			// strategy2
			String strategyString2 = myGetParameter(content.NAME_Strategy2);
			int strategy2 = IQuery.STRATEGY_NONE;
			if (null != strategyString2 && !strategyString2.equals("")) {
				strategy2 = Integer.parseInt(strategyString2);
			}
			// strategy3
			String strategyString3 = myGetParameter(content.NAME_Strategy3);
			int strategy3 = IQuery.STRATEGY_NONE;
			if (null != strategyString3 && !strategyString3.equals("")) {
				strategy3 = Integer.parseInt(strategyString3);
			}
			// strategy4
			String strategyString4 = myGetParameter(content.NAME_Strategy4);
			int strategy4 = IQuery.STRATEGY_NONE;
			if (null != strategyString4 && !strategyString4.equals("")) {
				strategy4 = Integer.parseInt(strategyString4);
			}
			
			// status
			String status = myGetParameter(content.NAME_STATUS);
			if (status != null && status.equals(ALL)) {
				status = null;
			}
			
			java.util.Vector myKeys = new java.util.Vector();
			java.util.Vector myClauses = new java.util.Vector();
			Volume myVolume = VolumesFactory.getVolumeByName(volume);
			String source = "eng";
			if (myVolume !=null && !myVolume.isEmpty()) {
				source = myVolume.getSourceLanguage();
			}
			if (search1 !=null && !search1.equals("")  &&
				search1text != null && !search1text.equals("")) {
				if (strategy1 == IQuery.STRATEGY_GREATER_THAN ||
					strategy1 == IQuery.STRATEGY_GREATER_THAN_OR_EQUAL ||
					strategy1 == IQuery.STRATEGY_LESS_THAN ||
					strategy1 == IQuery.STRATEGY_LESS_THAN_OR_EQUAL) {
					String clause = "key='" + search1 + "'";
					java.util.regex.Matcher quoteMatcher = quotePattern.matcher(search1text);
					String newValue = quoteMatcher.replaceAll("''");
					clause += " and multilingual_sort('" + source + "',value)" + IQuery.QueryBuilderStrategy[strategy1+1] + " multilingual_sort('" + source + "','" + newValue +"') "; 
					myClauses.add(clause);
				}
				else {
					String[] key1 = new String[4];
					key1[0] = search1;
					key1[2] = search1text;
					key1[3] = IQuery.QueryBuilderStrategy[strategy1+1];
					myKeys.add(key1);
				}
			}
			if (search2 !=null && !search2.equals("") &&
				search2text != null && !search2text.equals("")) {
				if (strategy2 == IQuery.STRATEGY_GREATER_THAN ||
					strategy2 == IQuery.STRATEGY_GREATER_THAN_OR_EQUAL ||
					strategy2 == IQuery.STRATEGY_LESS_THAN ||
					strategy2 == IQuery.STRATEGY_LESS_THAN_OR_EQUAL) {
					String clause = "key='" + search2 + "'";
					java.util.regex.Matcher quoteMatcher = quotePattern.matcher(search2text);
					String newValue = quoteMatcher.replaceAll("''");
					clause += " and multilingual_sort('" + source + "',value)" + IQuery.QueryBuilderStrategy[strategy2+1] + " multilingual_sort('" + source + "','" + newValue +"') "; 
					myClauses.add(clause);
				}
				else {
					String[] key2 = new String[4];
					key2[0] = search2;
					key2[2] = search2text;
					key2[3] = IQuery.QueryBuilderStrategy[strategy2+1];
					myKeys.add(key2);
				}
			}
			if (search3 !=null && !search3.equals("") &&
				search3text != null && !search3text.equals("")) {
				if (strategy3 == IQuery.STRATEGY_GREATER_THAN ||
					strategy3 == IQuery.STRATEGY_GREATER_THAN_OR_EQUAL ||
					strategy3 == IQuery.STRATEGY_LESS_THAN ||
					strategy3 == IQuery.STRATEGY_LESS_THAN_OR_EQUAL) {
					String clause = "key='" + search3 + "'";
					java.util.regex.Matcher quoteMatcher = quotePattern.matcher(search3text);
					String newValue = quoteMatcher.replaceAll("''");
					clause += " and multilingual_sort('" + source + "',value)" + IQuery.QueryBuilderStrategy[strategy3+1] + " multilingual_sort('" + source + "','" + newValue +"') "; 
					myClauses.add(clause);
				}
				else {
					String[] key3 = new String[4];
					key3[0] = search3;
					key3[2] = search3text;
					key3[3] = IQuery.QueryBuilderStrategy[strategy3+1];
					myKeys.add(key3);
				}
			}
			if (search4 !=null && !search4.equals("") &&
				search4text != null && !search4text.equals("")) {
				if (strategy4 == IQuery.STRATEGY_GREATER_THAN ||
					strategy4 == IQuery.STRATEGY_GREATER_THAN_OR_EQUAL ||
					strategy4 == IQuery.STRATEGY_LESS_THAN ||
					strategy4 == IQuery.STRATEGY_LESS_THAN_OR_EQUAL) {
					String clause = "key='" + search4 + "'";
					java.util.regex.Matcher quoteMatcher = quotePattern.matcher(search4text);
					String newValue = quoteMatcher.replaceAll("''");
					clause += " and multilingual_sort('" + source + "',value)" + IQuery.QueryBuilderStrategy[strategy3+1] + " multilingual_sort('" + source + "','" + newValue +"') "; 
					myClauses.add(clause);
				}
				else {
					String[] key4 = new String[4];
					key4[0] = search4;
					key4[2] = search4text;
					key4[3] = IQuery.QueryBuilderStrategy[strategy4+1];
					myKeys.add(key4);
				}
			}
			if (status !=null && !status.equals("")) {
				String[] key2 = new String[4];
				key2[0] = Volume.CDM_contributionStatus;
				key2[1] = Volume.DEFAULT_LANG;
				key2[2] = status;
				key2[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];			
				myKeys.add(key2);
			}
			exportVolume(volume, myKeys, myClauses, outputFormat);
		}
        
		if (exportList != null && volume != null && entryList != null) {
			String[] EntryList = entryList.split("[\\t\\n\\r]+");
			exportEntryList(volume, EntryList, outputFormat);
		}
				
		addConsultForm(volume);
		
        //On rend le contenu correct
        return content.getElementFormulaire();
    }
	
	protected void addConsultForm(String selectedVolume) 
        throws fr.imag.clips.papillon.business.PapillonBusinessException, 
                HttpPresentationException {
					
		XHTMLAnchorElement repositoryAnchor = content.getElementRepositoryAnchor();
		repositoryAnchor.setHref(this.getExportRelativeDir());
					
           // Adding the appropriate source languages to the source list
        XHTMLOptionElement volumeOptionTemplate = content.getElementVolumeOptionTemplate();
        Node volumeSelect = volumeOptionTemplate.getParentNode();
        volumeOptionTemplate.removeAttribute("id");
        // We assume that the option element has only one text child 
        // (it should be this way if the HTML is valid...)
        Text volumeTextTemplate = (Text)volumeOptionTemplate.getFirstChild(); 
                
		//      
        for (Iterator iter = VolumesFactory.getVolumesArray().iterator(); iter.hasNext();) {
            String volumeName = ((Volume)iter.next()).getName();
            
            //
            volumeOptionTemplate.setValue(volumeName);
            volumeOptionTemplate.setLabel(volumeName);
            // Je dois ici mettre un text dans l'OPTION, car les browser PC ne sont pas conformes aux 
            // specs W3C.
            volumeTextTemplate.setData(volumeName);
            volumeOptionTemplate.setSelected(volumeName.equals(selectedVolume));
            volumeSelect.appendChild(volumeOptionTemplate.cloneNode(true));
        }
        volumeSelect.removeChild(volumeOptionTemplate);

       /* XHTMLOptionElement xslOptionTemplate = content.getElementXslOptionTemplate();
        Node xslSelect = xslOptionTemplate.getParentNode();
        xslOptionTemplate.removeAttribute("id");
        Text xslTextTemplate = (Text)xslOptionTemplate.getFirstChild(); 
		
		fr.imag.clips.papillon.business.xsl.XslSheet[] AllXsls = fr.imag.clips.papillon.business.xsl.XslSheetFactory.getXslSheetsArray();
		
        for (int i = 0; i < AllXsls.length; i++) {
            String xslName = AllXsls[i].getName();
            String xslDescription = AllXsls[i].getDescription();
			if (xslDescription != null && !xslDescription.equals("")) {
				xslDescription = " (" + xslDescription + ")";
			}
			else {
				xslDescription = "";
			}
			xslName += xslDescription;
            xslOptionTemplate.setValue(AllXsls[i].getHandle());
            xslOptionTemplate.setLabel(xslName);
            xslTextTemplate.setData(xslName);
            xslSelect.appendChild(xslOptionTemplate.cloneNode(true));
        }
        xslSelect.removeChild(xslOptionTemplate); */
				
	}
	
	protected void exportVolume(String volume, java.util.Vector myKeys, java.util.Vector myClauses, String outputFormat) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException,
		java.io.IOException,
		PapillonPresentationException {
		java.io.File fileDir = new java.io.File(getExportAbsoluteDir());
		fileDir.mkdirs();
		
		String filename = createFileName(volume, this.getUser().getLogin(), outputFormat);
		java.io.File exportFile = new java.io.File(fileDir.getCanonicalPath() + File.separator + filename);
		exportFile.createNewFile();
		
		FileOutputStream fileOutStream = new FileOutputStream(exportFile.getCanonicalFile());
		
		java.util.zip.GZIPOutputStream myGZipOutStream = new java.util.zip.GZIPOutputStream(fileOutStream);
		
		fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory.exportVolume(volume, myKeys, myClauses, outputFormat, myGZipOutStream);
		
		PapillonLogger.writeDebugMsg("Compressing volume");
		myGZipOutStream.close();
		
		String userMessage = "Volume " + volume + " exported";
		
		if (userMessage != null) {
			this.getSessionData().writeUserMessage(userMessage);
			PapillonLogger.writeDebugMsg(userMessage);
		}
		PapillonLogger.writeDebugMsg("ClientPageRedirectException: " + getExportRelativeDir() + filename);			
		throw new ClientPageRedirectException(getExportRelativeDir() + filename); 
	}

	protected void exportEntryList(String volume, String[] Headwords,String outputFormat) 
	throws fr.imag.clips.papillon.business.PapillonBusinessException,
	java.io.IOException,
	PapillonPresentationException {
		// PapillonLogger.writeDebugMsg("absoluteDir:"+getExportAbsoluteDir());
		java.io.File fileDir = new java.io.File(getExportAbsoluteDir());
		fileDir.mkdirs();
		
		String filename = createFileName(volume, this.getUser().getLogin(), outputFormat);
		java.io.File exportFile = new java.io.File(fileDir.getCanonicalPath() + File.separator + filename);
		exportFile.createNewFile();
		
		FileOutputStream fileOutStream = new FileOutputStream(exportFile.getCanonicalFile());
		
		java.util.zip.GZIPOutputStream myGZipOutStream = new java.util.zip.GZIPOutputStream(fileOutStream);
		
		fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory.exportEntryList(volume, Headwords, outputFormat, myGZipOutStream);
		
		PapillonLogger.writeDebugMsg("Compressing volume");
		myGZipOutStream.close();
		
		String userMessage = "Volume " + volume + " exported";
		
		if (userMessage != null) {
			this.getSessionData().writeUserMessage(userMessage);
			PapillonLogger.writeDebugMsg(userMessage);
		}
		PapillonLogger.writeDebugMsg("ClientPageRedirectException: " + getExportRelativeDir() + filename);			
		throw new ClientPageRedirectException(getExportRelativeDir() + filename); 
	}
	
	
	protected String getExportAbsoluteDir() throws PapillonPresentationException {            
		String baseDir = "";
		String mediaDir = "";
		try {
			baseDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(BASE_DIR_CONFIG);
			mediaDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(MEDIA_DIR_CONFIG);
			if (! baseDir.endsWith(File.separator)) {
				baseDir = baseDir + File.separator;
			}
			if (! mediaDir.endsWith(File.separator)) {
				mediaDir = mediaDir + File.separator;
			}
		}
		catch (com.lutris.util.ConfigException ex) {
			throw new PapillonPresentationException("Error in Papillon Configuration File: ", ex);
		}
		baseDir = baseDir + mediaDir + EXPORT_VOLUME_DIR + File.separator;
		return baseDir;
	}

	protected String getExportRelativeDir() throws PapillonPresentationException {            
		String baseDir = "";
		String mediaDir = "";
		try {
			baseDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(RELATIVE_DIR_CONFIG);
			mediaDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(MEDIA_DIR_CONFIG);
			if (! baseDir.endsWith(File.separator)) {
				baseDir = baseDir + File.separator;
			}
			if (! mediaDir.endsWith(File.separator)) {
				mediaDir = mediaDir + File.separator;
			}
		}
		catch (com.lutris.util.ConfigException ex) {
			throw new PapillonPresentationException("Error in Papillon Configuration File: ", ex);
		}
		baseDir = baseDir + mediaDir + EXPORT_VOLUME_DIR + File.separator;
		return baseDir;
	}

	       
	protected String createFileName(String volume, String login, String format) {
		String timestamp = "" + new java.util.Date().getTime();
		String extension = "";
		if (format !=null && format.equals(XHTMLFormat)) {
			extension = ".html";
		}
		else if (format !=null && format.equals(TEXTFormat)) {
			extension = ".txt";
		}
		else if (format !=null && format.equals(PDFFormat)) {
			extension = ".pdf";
		}
		else {
			extension = ".xml";
		}
		
		String fileName = volume + "-" + login + "-" + timestamp + extension + ".gz";
		return fileName;
	}
					      


}
