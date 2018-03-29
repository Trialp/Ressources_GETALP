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
 * Revision 1.2  2007/01/05 13:57:26  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.1  2005/07/28 14:36:56  mangeot
 * Added a News presentation page that takes a static xhtml page and displays it.
 * People can edit this static page on the server by accessing it via ftp or ssh.
 * There is a cache that can be reset via the Admin.po page.
 *
 *
 *-----------------------------------------------
 * Papillon News page.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import fr.imag.clips.papillon.business.xml.XMLServices;

public class News extends PapillonBasePO {
	
    protected final static String BASE_DIR_CONFIG = "Papillon.Informations.baseDir";
    protected final static String MEDIA_DIR_CONFIG = "Papillon.Informations.mediaDir";
	protected final static String NEWS_VOLUME_DIR="news";
	protected final static String NEWS_FILE="news.xhtml";
	protected final static String NewsContentIdString = "NewsContent";
	
	protected static org.w3c.dom.Node NewsDOMCache = null;
	

    protected boolean loggedInUserRequired() {
        return false;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }

    protected  int getCurrentSection() {
        return NO_SECTION;
    }


    public org.w3c.dom.Node getContent()
    throws HttpPresentationException, java.io.IOException {

		if (NewsDOMCache == null) {
			org.xml.sax.InputSource newsInputSource = getInputSource(getNewsFileAbsolutePath());
			if (newsInputSource != null) {
				org.w3c.dom.Document myNewsDocument = XMLServices.buildDOMTree(newsInputSource);
				if (myNewsDocument != null) {
					NewsDOMCache = myNewsDocument.getElementById(NewsContentIdString);
				}
			}
		}
		
        //On rend le contenu correct
        return NewsDOMCache;
    }
	
	protected String getNewsFileAbsolutePath() throws PapillonPresentationException {            
		String baseDir = "";
		String mediaDir = "";
		String filePath = "";
		try {
			baseDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(BASE_DIR_CONFIG);
			mediaDir = com.lutris.appserver.server.Enhydra.getApplication().getConfig().getString(MEDIA_DIR_CONFIG);
			if (! baseDir.endsWith(java.io.File.separator)) {
				baseDir = baseDir + java.io.File.separator;
			}
			if (! mediaDir.endsWith(java.io.File.separator)) {
				mediaDir = mediaDir + java.io.File.separator;
			}
		}
		catch (com.lutris.util.ConfigException ex) {
			throw new PapillonPresentationException("Error in Papillon Configuration File: ", ex);
		}
		filePath = baseDir + mediaDir + NEWS_VOLUME_DIR + java.io.File.separator + NEWS_FILE;
		return filePath;
	}
	
	protected org.xml.sax.InputSource getInputSource(String filePath) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {	
			org.xml.sax.InputSource myInputSource = null;
			try {
				java.io.FileInputStream myFileInputStream = new java.io.FileInputStream(filePath);
				myInputSource = new org.xml.sax.InputSource(myFileInputStream);
			}
			catch(java.io.IOException ioex) {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in renderOutputStream()", ioex);
			}
			return myInputSource;
		}
	
	protected static void resetCache() {
		NewsDOMCache = null;
	}
}
