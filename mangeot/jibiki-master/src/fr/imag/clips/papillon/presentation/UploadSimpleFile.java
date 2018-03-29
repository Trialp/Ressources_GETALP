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
 *  Revision 1.2  2006/08/10 16:26:13  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.1  2006/08/10 09:55:21  mangeot
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
 * Same as UploadFile but do not unzip if it is a zipped archive. It may be unzipped later. Used by AdminInformations.java
 */
public class UploadSimpleFile extends UploadFile {
	
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
				// do not unzip zip archives, it will be done later
				/*else if (type.equals("zip") || extension.equals("zip")) {
					if (extension.equals("zip")) {
						fileName = fileName.substring(0,fileName.indexOf("."+extension));
					}
					newFilePath += fileName;
					newFile = unzipFile(theFile, newFilePath);
				} */
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
}
