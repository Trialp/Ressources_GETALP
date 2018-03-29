/*
 *  jibiki platform
 *
 * Jibiki/Papillon project © GETA-CLIPS-IMAG
 * Francis Brunet-Manquat, Mathieu Mangeot & Gilles Sérasset
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.5.2.1  2007/10/02 10:31:21  serasset
 * Modified export of volume to speed up this task.
 *
 * Revision 1.5  2007/02/28 09:27:07  fbrunet
 * Added ajax method to AdvancedQueryForm page
 *
 * Revision 1.4  2007/02/07 13:58:57  fbrunet
 * added message before axies are merged and undo process if the merge is not correct.
 *
 *
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

import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.message.MessageDBLoader;
import fr.imag.clips.papillon.presentation.PapillonSessionData;


// Standard imports
import java.io.IOException;
import java.util.Date;
import java.util.Iterator;
import java.util.Collection;
import java.text.DateFormat;
import java.io.*;

import fr.imag.clips.papillon.presentation.xhtml.orig.ExportXHTML;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.IQuery;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.system.Saxon8bTransformer;
import fr.imag.clips.papillon.business.system.FOPTransformer;


public class Export extends PapillonBasePO {
	
	protected final static String ALL="*ALL*";
	protected final static String DEFAULT="*default*";
	protected final static String SORTBY_PARAMETER="SortBy";
	protected final static String AnyContains_PARAMETER="AnyContains";
    protected final static String BASE_FO_XSL_CONFIG = "Papillon.Fo.Xsl";
    protected final static String BASE_DIR_CONFIG = "Papillon.Export.baseDir";
    protected final static String RELATIVE_DIR_CONFIG = "Papillon.Export.relativeDir";
    protected final static String MEDIA_DIR_CONFIG = "Papillon.Export.mediaDir";
    protected final static String BASE_URL_CONFIG = "Papillon.Export.baseurl";
	protected final static String EXPORT_VOLUME_DIR="export";
    protected final static String FORMAT_XML="4";
	
	protected static final String XMLFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XML_DIALECT);
	protected static final String XHTMLFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XHTML_DIALECT);
	protected static final String TEXTFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.PLAINTEXT_DIALECT);
	protected static final String PDFFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.PDF_DIALECT);
    
    protected ExportXHTML content;
    
    protected boolean loggedInUserRequired() {
        return true;
    }
    
    protected boolean userMayUseThisPO() {
        return true;
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
            content = (ExportXHTML)MultilingualXHtmlTemplateFactory.createTemplate("ExportXHTML", this.getComms(), this.getSessionData());
            
            HttpPresentationRequest req = this.getComms().request;
            
            String export = myGetParameter(content.NAME_EXPORT);
            String export_myfinished = myGetParameter(content.NAME_EXPORTMYFINISHED);
            String export_myunfinished = myGetParameter(content.NAME_EXPORTMYUNFINISHED);
            String volume = myGetParameter(content.NAME_VOLUME);
            String exportType = ""; // used for filename
            
            // If the page is called with parameters, take the requested action
            if (export != null && volume != null) {
                
                // Initialize
                java.util.Vector myKeys = new java.util.Vector();
                java.util.Vector myClauses = new java.util.Vector();
                Volume myVolume = VolumesFactory.getVolumeByName(volume);
                
                /*
                // AUTHOR
                String author = myGetParameter(content.NAME_AUTHOR);
                if (author !=null && !author.equals("")) {
                        String[] key1 = new String[4];
                        key1[0] = Volume.CDM_modificationAuthor;
                        key1[2] = author;
                        key1[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL+1];
                        myKeys.add(key1);
                }
                */
                
                // STATUS
//                String[] CNFS = new String[4];
//                CNFS[0] = Volume.CDM_contributionStatus;
//                CNFS[1] = Volume.DEFAULT_LANG;
//                CNFS[2] = VolumeEntry.CLASSIFIED_NOT_FINISHED_STATUS;
//                CNFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];
//                myKeys.add(CNFS);
//
//                String[] CFS = new String[4];
//                CFS[0] = Volume.CDM_contributionStatus;
//                CFS[1] = Volume.DEFAULT_LANG;
//                CFS[2] = VolumeEntry.CLASSIFIED_FINISHED_STATUS;
//                CFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];
//                myKeys.add(CFS);
//
//                String[] DS = new String[4];
//                DS[0] = Volume.CDM_contributionStatus;
//                DS[1] = Volume.DEFAULT_LANG;
//                DS[2] = VolumeEntry.DELETED_STATUS;
//                DS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];
//                myKeys.add(DS);
//
//                String[] NFS = new String[4];
//                NFS[0] = Volume.CDM_contributionStatus;
//                NFS[1] = Volume.DEFAULT_LANG;
//                NFS[2] = VolumeEntry.NOT_FINISHED_STATUS;
//                NFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_NOT_EQUAL+1];
//                myKeys.add(NFS);

                String[] CNFS = new String[4];
                CNFS[0] = Volume.CDM_contributionStatus;
                CNFS[1] = Volume.DEFAULT_LANG;
                CNFS[2] = VolumeEntry.FINISHED_STATUS;
                CNFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL+1];
                myKeys.add(CNFS);

                String[] CFS = new String[4];
                CFS[0] = Volume.CDM_contributionStatus;
                CFS[1] = Volume.DEFAULT_LANG;
                CFS[2] = VolumeEntry.MODIFIED_STATUS;
                CFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL+1];	
                myKeys.add(CFS);
                //
                exportType = "complete";
                exportVolume(volume, myKeys, myClauses, false, exportType);

            } else if (export_myfinished != null && volume != null) {
                
                // Initialize
                java.util.Vector myKeys = new java.util.Vector();
                java.util.Vector myClauses = new java.util.Vector();
                Volume myVolume = VolumesFactory.getVolumeByName(volume);
                
                
                // AUTHOR
                String[] author = new String[4];
                author[0] = Volume.CDM_modificationAuthor;
                author[2] = this.getUser().getLogin();
                author[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL+1];
                myKeys.add(author);
                
                // STATUS
                String[] CNFS = new String[4];
                CNFS[0] = Volume.CDM_contributionStatus;
                CNFS[1] = Volume.DEFAULT_LANG;
                CNFS[2] = VolumeEntry.FINISHED_STATUS;
                CNFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL+1];			
                myKeys.add(CNFS);
                
                //
                exportType = this.getUser().getLogin() + "-finished-contributions";
                exportVolume(volume, myKeys, myClauses, exportType);
                
            } else if (export_myunfinished != null && volume != null) {
                
                // Initialize
                java.util.Vector myKeys = new java.util.Vector();
                java.util.Vector myClauses = new java.util.Vector();
                Volume myVolume = VolumesFactory.getVolumeByName(volume);
                
                
                // AUTHOR
                String[] author = new String[4];
                author[0] = Volume.CDM_modificationAuthor;
                author[2] = this.getUser().getLogin();
                author[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL+1];
                myKeys.add(author);
                
                // STATUS
                String[] CNFS = new String[4];
                CNFS[0] = Volume.CDM_contributionStatus;
                CNFS[1] = Volume.DEFAULT_LANG;
                CNFS[2] = VolumeEntry.NOT_FINISHED_STATUS;
                CNFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL+1];			
                myKeys.add(CNFS);
                
                //
                exportType = this.getUser().getLogin() + "-unfinished-contributions";
                exportVolume(volume, myKeys, myClauses, exportType);
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

    protected void exportVolume(String volume, java.util.Vector myKeys, java.util.Vector myClauses, String exportType)
        throws fr.imag.clips.papillon.business.PapillonBusinessException,
        java.io.IOException,
        PapillonPresentationException {
        exportVolume(volume, myKeys, myClauses, true, exportType);
    }
    
    protected void exportVolume(String volume, java.util.Vector myKeys, java.util.Vector myClauses, boolean andClauses, String exportType)
		throws fr.imag.clips.papillon.business.PapillonBusinessException,
		java.io.IOException,
		PapillonPresentationException {
            
            //
            if (!volume.equals("*ALL*")) {  
                
                //
                java.io.File fileDir = new java.io.File(getExportAbsoluteDir());
                fileDir.mkdirs();
                String filename = createFileName(volume, exportType);
                
                // -> XML
                String xmlFilename = filename + ".xml";
                java.io.File xmlFile = new java.io.File(fileDir.getCanonicalPath() + File.separator + xmlFilename);
                xmlFile.createNewFile();
                FileOutputStream xmlFileOutStream = new FileOutputStream(xmlFile.getCanonicalFile());
                fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory.exportVolume(volume, myKeys, myClauses, andClauses, FORMAT_XML, xmlFileOutStream);
                
                /*
                 // Compress -> GZ
                 java.util.zip.GZIPOutputStream myGZipOutStream = new java.util.zip.GZIPOutputStream(fileOutStream);
                 fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory.exportVolume(volume, myKeys, myClauses, FORMAT_XML, myGZipOutStream);
                 PapillonLogger.writeDebugMsg("Compressing volume");
                 myGZipOutStream.close();
                 */
                /*
                 // XML -> FO
                 String foFilename = filename + ".fo";
                 Saxon8bTransformer.doSaxonTransformation(getAbsoluteXslFo(), fileDir.getCanonicalPath() + File.separator + xmlFilename, fileDir.getCanonicalPath() + File.separator + foFilename);
                 
                 // FO -> PDF
                 String pdfFilename = filename + ".pdf";
                 FOPTransformer.doFOPTransformation(fileDir.getCanonicalPath() + File.separator + foFilename, fileDir.getCanonicalPath() + File.separator + pdfFilename);
                 
                 // Delete files
                 //xmlFile.delete();
                 //java.io.File foFile = new java.io.File(fileDir.getCanonicalPath() + File.separator + foFilename);
                 //foFile.delete();
                 */
                
                //
                String userMessage = "Volume " + volume + " exported";
                
                //
                if (userMessage != null) {
                    this.getSessionData().writeUserMessage(userMessage);
                    PapillonLogger.writeDebugMsg(userMessage);
                }
                
                /*
                 //
                 PapillonLogger.writeDebugMsg("ClientPageRedirectException: " + getExportRelativeDir() + pdfFilename);			
                 throw new ClientPageRedirectException(getExportRelativeDir() + pdfFilename); 
                 */
                
            } else {
             
                for (Iterator iter = ((Collection)VolumesFactory.getVolumes()).iterator(); iter.hasNext();) {
                    Volume vol = (Volume)iter.next();
                    exportVolume(vol.getName(), myKeys, myClauses, andClauses, exportType);
                }
            }
           
        }
    
        protected String getAbsoluteXslFo() throws PapillonPresentationException {            
            String baseDir = "";
            String mediaDir = "";
            try {
                baseDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(BASE_FO_XSL_CONFIG);
            }
            catch (com.lutris.util.ConfigException ex) {
                throw new PapillonPresentationException("Error in Papillon Configuration File: ", ex);
            }

            //
            return baseDir;
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
			baseDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(BASE_URL_CONFIG);
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
    
    
	protected String createFileName(String volume, String exportType) {
		String timestamp = "" + new java.util.Date().getTime();
		String fileName = volume + "-" + exportType + "-" + timestamp;
		return fileName;
	}
    
    
    
}
