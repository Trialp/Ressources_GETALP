/*
 *  papillon
 *
 *  Enhydra super-servlet
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id$
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.4  2006/08/10 16:26:13  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.3  2006/08/10 09:55:21  mangeot
 *  Added a bunzipper for .bz2 compressed files
 *
 *  Revision 1.2  2006/08/07 09:32:21  mangeot
 *  Bug fix: when the method was finished, the file was deleted. I make a copy of the file before exiting.
 *
 *  Revision 1.1  2006/08/03 15:22:47  mangeot
 *  This page is used in order to upload a unique file on the server. The file can be zipped or gzipped. This page must be called via a javascript with window.open.
 *  See the AdminXsl.po for an example.
 *
 *
 *  -----------------------------------------------
 *  Abstract class implementing small windows used to upload documents.
 */
package fr.imag.clips.papillon.presentation;

// Papillon import
import fr.imag.clips.papillon.presentation.xhtml.orig.*;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;

import org.enhydra.xml.xhtml.dom.XHTMLFormElement;
import org.enhydra.xml.xhtml.dom.XHTMLInputElement;
import com.lutris.appserver.server.Enhydra;

//pour le debogage
import fr.imag.clips.papillon.business.PapillonLogger;

/**
*  Description of the Class
 *
 * @author     mangeot
 * @created    August 8, 2006
 *
 * Imports a file from a multipart request into the local temp dir, decompress it and yelds a local file:/ URL.
 */
public class UploadFile extends AbstractPO {
	
    protected final static String TMP_DIR_CONFIG="Papillon.Informations.tmpDir";
    protected final static String TMP_NAME_PREFIX="jibiki-tmp-"; 

	/**
    *  This method should be implemented in the subclass so that it returns
     *  true if this particular request requires the user to be logged in,
     *  otherwise false.
     *
     * @return    Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return true;
    }
    
    /**
    *  This method should be implemented in the subclass so that it returns
     *  true if this particular request requires the user to be logged in,
     *  otherwise false.
     *
     * @return    Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        return true;
    }
	
	
	public org.w3c.dom.Node getDocument() throws javax.xml.parsers.ParserConfigurationException,
	PapillonPresentationException,
	HttpPresentationException,
	java.io.IOException,
	IllegalArgumentException ,
	com.lutris.mime.MimeException,
	javax.xml.transform.TransformerException,
	org.xml.sax.SAXException {
				
		/*********************************/
		// Reading the request
		/*********************************/
		// Avant toute chose, on récupère les paramêtres selon la méthode que le client a utilisé
		// Ce peut-être du GET (c'est standard), du POST en url-encoded (c'est standard aussi dans enhydra)
		// ou bien du POST en multipart (non géré par enhydra, mais par le multipart request).
		HttpPresentationComms comms=this.getComms();
		
		String contentType = comms.request.getContentType();
		if (contentType == null) {
			contentType = "";
		}
		com.lutris.mime.ContentHeader contentHdr = new com.lutris.mime.ContentHeader("Content-Type: " + contentType);
		javax.servlet.http.HttpServletRequest req = null;
		
		if (contentHdr.getValue().toLowerCase().startsWith("multipart/")) {
			// On a affaire à un multipart request, on construit donc la requête appropriée.
			// ATTENTION: Pour simplifier la gestion du fichier chargé plus tard, je choisis le constructeur
			// qui force la sauvegarde des données dans un fichier temporaire.
			PapillonLogger.writeDebugMsg("Multipart request");
			String tmpDir="/tmp/";      // Dangereux ???
			try {
				tmpDir = Enhydra.getApplication().getConfig().getString(TMP_DIR_CONFIG);
				if (! tmpDir.endsWith(java.io.File.separator)) {
					tmpDir = tmpDir + java.io.File.separator;
				}
			} catch (com.lutris.util.KeywordValueException kve) {
				PapillonLogger.writeDebugMsg("Unexpected Error: Incorrect configuration file for group: "
											 + TMP_DIR_CONFIG);
			}
			req = new de.opus5.servlet.MultipartRequest(comms.request.getHttpServletRequest(), 30000000, tmpDir,TMP_NAME_PREFIX, true);
		}
		else {
			// PapillonLogger.writeDebugMsg("Normal request");
			req = comms.request.getHttpServletRequest();
		}
		
		/*********************************/
		// Managing the request
		/*********************************/
		String[] result = getFileNameAndURL(req);
		String fileNameValue = result[0];
		String resultUrlValue = result[1];
		
		/*********************************/
		// Creating the document Layout
		/*********************************/		
		return getForm(resultUrlValue, fileNameValue);
		
	}
	
	
	/**
	*  Returns the complete document.
     *
     * @exception  Exception
     * @exception  HttpPresentationException  Description of the Exception
     * @exception  IOException                Description of the Exception
     */
    protected org.w3c.dom.Node getForm(String resultURLValue, String fileName)
	throws HttpPresentationException, java.io.IOException {
		
		UploadFileXHTML uploaderLayout;
				
		// Création du document
		uploaderLayout = (UploadFileXHTML) MultilingualXHtmlTemplateFactory.createTemplate("UploadFileXHTML", this.myComms, this.sessionData);
		
		// Add the form address
		XHTMLFormElement formElement = (XHTMLFormElement) uploaderLayout.getElementForm();
		formElement.setAttribute("action",this.getUrl());
		
		// Add the resultURL as a value of an input
		XHTMLInputElement urlInputElement = (XHTMLInputElement) uploaderLayout.getElementResultUrl();
		urlInputElement.setValue(resultURLValue);
		// Add the original file name as a value of an input
		XHTMLInputElement fileNameInputElement = (XHTMLInputElement) uploaderLayout.getElementFileName();
		fileNameInputElement.setValue(fileName);
		
		return (org.w3c.dom.Node) uploaderLayout;
	}
	
	
	protected String[] getFileNameAndURL(javax.servlet.http.HttpServletRequest mReq)
	throws 	javax.xml.parsers.ParserConfigurationException,
	HttpPresentationException,
	java.io.IOException,
	IllegalArgumentException ,
	com.lutris.mime.MimeException,
	javax.xml.transform.TransformerConfigurationException,
	javax.xml.transform.TransformerException,
	org.xml.sax.SAXException  {
		String fileName = "";
		String theURL = "";
		
		if (mReq.getClass().getName().equals("de.opus5.servlet.MultipartRequest")) {
			
			de.opus5.servlet.UploadedFile theFile = ((de.opus5.servlet.MultipartRequest)mReq).getUploadedFile(UploadFileXHTML.NAME_file);
			if (null != theFile && !theFile.getName().equals("")) {
				fileName = theFile.getName();				
				String prefix = TMP_NAME_PREFIX+fileName;
				String newFilePath = theFile.getFile().getCanonicalPath();
				String number = "0";
				if (newFilePath.lastIndexOf(prefix)>0) {
					number = newFilePath.substring(newFilePath.lastIndexOf(prefix) + prefix.length());
					newFilePath = newFilePath.substring(0,newFilePath.lastIndexOf(prefix));
				}
				if (number.lastIndexOf(".tmp")>0) {
					number = number.substring(0,number.lastIndexOf(".tmp"));				
				}
				newFilePath += TMP_NAME_PREFIX + number + "-"; 
				String contentType = theFile.getContentType();
				String type = "";
				String extension = "";
				java.io.File newFile = null;
				if (null != contentType) {
					try {
						int i = contentType.indexOf("/");
						String category = contentType.substring(0, i).replace('-', '_');
						type = contentType.substring(i+1).replace('-', '_');
					} catch (IndexOutOfBoundsException e) {
						PapillonLogger.writeDebugMsg("Unexpected Error: malformed content type: " + contentType);
					}
				} 
				try {
					int i = fileName.lastIndexOf(".");
					if (i>0) {
						extension = fileName.substring(i+1).toLowerCase();
					}
				} catch (IndexOutOfBoundsException e) {
					PapillonLogger.writeDebugMsg("Unexpected Error: Can't determine file extension for file: " + fileName);
				}
				if (type.equals("gzip") || type.equals("x_gzip") ||  extension.equals("gz")) {
					if (extension.equals("gz")) {
						fileName = fileName.substring(0,fileName.indexOf("."+extension));
					}
					newFilePath += fileName;
					newFile = ungzipFile(theFile, newFilePath);
				}
				// It does not work, it abruptly crashes when creating the CBZip2InputStream without any error message
				/*else if (type.equals("bzip") || type.equals("x_bzip") || extension.equals("bz2")) {
					if (extension.equals("bz2")) {
						fileName = fileName.substring(0,fileName.indexOf("."+extension));
					}
					newFilePath += fileName;
					newFile = unbzipFile(theFile, newFilePath);
				} */
				else if (type.equals("zip") || extension.equals("zip")) {
					if (extension.equals("zip")) {
						fileName = fileName.substring(0,fileName.indexOf("."+extension));
					}
					newFilePath += fileName;
					newFile = unzipFile(theFile, newFilePath);
				}
				else {
					// The pb was the following: when exiting the method, the file was deleted
					// So I copy it.
					newFilePath += fileName;
					newFile = new java.io.File(newFilePath);
					copyFile(theFile.getFile(), newFile);
					theFile.getFile().deleteOnExit();
				}
				theURL = newFile.toURL().toExternalForm();
			}
			
		}
		String[] resArray = new String[2];
		resArray[0] = fileName;
		resArray[1] = theURL;
		return resArray;
	}	
	
	protected static java.io.File ungzipFile (de.opus5.servlet.UploadedFile file, String newPath) throws 
		java.io.IOException,
		fr.imag.clips.papillon.business.PapillonBusinessException,
		fr.imag.clips.papillon.business.PapillonImportException{
			// First, computing the output name
			String source = file.getFile().getCanonicalPath();
			String dest = newPath;		
			java.io.File newFile = null;
			java.io.InputStream is = file.getInputStream();
			
			if (is.available() == 0) {
				dest = "";
				PapillonLogger.writeDebugMsg("ImportGZippedFile: Encountered an empty GZipped file.");
				throw new fr.imag.clips.papillon.business.PapillonImportException("the GZipped file is empty.");
			} 
			else {
				// gunzip the file, then apply the appropriate action, based on gunzipped file type.
				
				java.io.FileOutputStream out = null;
				java.util.zip.GZIPInputStream zIn = null;
				try {
					newFile = new java.io.File(dest);
					out = new java.io.FileOutputStream(dest);
					zIn = new java.util.zip.GZIPInputStream(is);
					byte[] buffer = new byte[8 * 1024];
					int count = 0;
					do {
						out.write(buffer, 0, count);
						count = zIn.read(buffer, 0, buffer.length);
					} while (count != -1);
				} catch (java.io.IOException ioe) {
					String msg = "Problem expanding gzip " + ioe.getMessage();
					PapillonLogger.writeDebugMsg(msg);
					throw new fr.imag.clips.papillon.business.PapillonBusinessException(msg, ioe);
				} finally {
					if (out != null) {
						try {
							out.close();
						} catch (java.io.IOException ioex) {}
					}
					if (zIn != null) {
						try {
							zIn.close();
						} catch (java.io.IOException ioex) {}
					}
				}
				file.getFile().deleteOnExit();
			}
			return newFile;
		}
	
	// It does not work, it abruptly crashes when creating the CBZip2InputStream without any error message
	protected static java.io.File unbzipFile (de.opus5.servlet.UploadedFile file, String newFilePath) throws 
		java.io.IOException,
		fr.imag.clips.papillon.business.PapillonBusinessException,
		fr.imag.clips.papillon.business.PapillonImportException{
			// First, computing the output name
			String source = file.getFile().getCanonicalPath();
			String dest = newFilePath;		
			java.io.File newFile = null;
			java.io.InputStream is = file.getInputStream();
			
			if (is.available() == 0) {
				dest = "";
				PapillonLogger.writeDebugMsg("ImportBZippedFile: Encountered an empty BZipped file.");
				throw new fr.imag.clips.papillon.business.PapillonImportException("the BZipped file is empty.");
			} 
			else {
				// bunzip the file, then apply the appropriate action, based onbgunzipped file type.
				
				java.io.FileOutputStream out = null;
				org.apache.tools.bzip2.CBZip2InputStream zIn = null;
				try {
					PapillonLogger.writeDebugMsg("bunzipping the file...");
					newFile = new java.io.File(dest);
					PapillonLogger.writeDebugMsg("file created: " + dest);
					out = new java.io.FileOutputStream(dest);
					PapillonLogger.writeDebugMsg("out created ");
					//crashes here
					zIn = new org.apache.tools.bzip2.CBZip2InputStream(is);
					PapillonLogger.writeDebugMsg("zIn created ");

					byte[] buffer = new byte[8 * 1024];
					PapillonLogger.writeDebugMsg("buffer created ");
					int count = 0;
					do {
						out.write(buffer, 0, count);
						count = zIn.read(buffer, 0, buffer.length);
						PapillonLogger.writeDebugMsg("Count: " + count);
					} while (count != -1);
					PapillonLogger.writeDebugMsg("Stream end");
					zIn.close();
					//out.close();
				} catch (java.io.IOException ioe) {
					String msg = "Problem expanding bzip2 " + ioe.getMessage();
					PapillonLogger.writeDebugMsg(msg);
					throw new fr.imag.clips.papillon.business.PapillonBusinessException(msg, ioe);
				} 
				catch (Exception e) {
					String msg = "Problem expanding bzip2 " + e.getMessage();
					PapillonLogger.writeDebugMsg(msg);
					throw new fr.imag.clips.papillon.business.PapillonBusinessException(msg, e);
				} 
				//finally {
				//	PapillonLogger.writeDebugMsg("finalizing the bunzipping");
					
					//if (zIn != null) {
					//	try {
					//		PapillonLogger.writeDebugMsg("in != null");
					//		zIn.close();
					//	} catch (java.io.IOException ioex) {
					//		String msg = "Problem expanding bzip2 " + ioex.getMessage();
					//		PapillonLogger.writeDebugMsg(msg);
					//	}
					//}
					//if (out != null) {
					//	PapillonLogger.writeDebugMsg("out != null");
					//	try {
					//		out.close();
					//	} catch (java.io.IOException ioex) {
					//		String msg = "Problem expanding bzip2 " + ioex.getMessage();
					//		PapillonLogger.writeDebugMsg(msg);
					//	}
					//}
				//}
				PapillonLogger.writeDebugMsg("file bunzipped");
				file.getFile().deleteOnExit();
			}
			return newFile;
		} 
	
	protected static java.io.File unzipFile (de.opus5.servlet.UploadedFile file, String newFilePath) throws 
		java.io.IOException,
		fr.imag.clips.papillon.business.PapillonBusinessException,
		fr.imag.clips.papillon.business.PapillonImportException{
			// First, computing the output name
			String source = file.getFile().getCanonicalPath();
			String dest = newFilePath;	
			java.io.File newFile = null;
			java.io.InputStream is = file.getInputStream();
			
			if (is.available() == 0) {
				dest = "";
				PapillonLogger.writeDebugMsg("ImportZippedFile: Encountered an empty Zipped file.");
				throw new fr.imag.clips.papillon.business.PapillonImportException("the Zipped file is empty.");
			} 
			else {
				// Here, we deal with archives containing one unique file
				// unzip the first file of the archive
				java.util.zip.ZipInputStream zis = new java.util.zip.ZipInputStream(is);
				java.util.zip.ZipEntry ze = zis.getNextEntry();
				if (ze != null &&   !ze.isDirectory()) {
					newFile = new java.io.File(dest);            
					byte[] buffer = new byte[1024];
                    int length = 0;
                    java.io.FileOutputStream fos = new java.io.FileOutputStream(newFile);
                    while ((length = zis.read(buffer)) >= 0) {
                        fos.write(buffer, 0, length);
                    }
                    fos.close();
					zis.close();
				}
				file.getFile().deleteOnExit();
			}
			return newFile;
		}	
	
	protected static void copyStream(java.io.InputStream input,
							java.io.OutputStream output )
		throws java.io.IOException
	{
		final byte[] buffer = new byte[ 8024 ];
		int n = 0;
		while( -1 != ( n = input.read( buffer ) ) )
		{
			output.write( buffer, 0, n );
		}
	}
	
	   // Copies src file to dst file.
	   // If the dst file does not exist, it is created
    protected static void copyFile(java.io.File src, java.io.File dst) throws java.io.IOException {
        java.io.InputStream in = new java.io.FileInputStream(src);
        java.io.OutputStream out = new java.io.FileOutputStream(dst);
		
        // Transfer bytes from in to out
        byte[] buf = new byte[1024];
        int len;
        while ((len = in.read(buf)) > 0) {
            out.write(buf, 0, len);
        }
        in.close();
        out.close();
    }
}
