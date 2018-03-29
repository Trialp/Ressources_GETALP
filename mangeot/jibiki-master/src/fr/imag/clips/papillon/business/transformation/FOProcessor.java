/*
 *
 *-----------------------
 * $Id$
 *------------------------
 * $Log$
 * Revision 1.1  2005/07/28 13:06:47  mangeot
 * - Added the possibility to export in PDF format. The conversion into PDF is don
 * e via the fop package that has to be installed (see ToolsForPapillon)
 *
 *
 *------------------------
 */

package fr.imag.clips.papillon.business.transformation;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;


public class FOProcessor {

    protected final static String BASE_DIR_CONFIG = "Papillon.Informations.baseDir";
    protected final static String MEDIA_DIR_CONFIG = "Papillon.Informations.mediaDir";
	protected final static String EXPORT_VOLUME_DIR="export";
	
	protected static org.apache.fop.apps.Driver myFopDriver = new org.apache.fop.apps.Driver();
	protected java.io.OutputStream myTempOutputStream = null;
	protected java.io.File myTempFile = null;

	static {
		myFopDriver.setRenderer(org.apache.fop.apps.Driver.RENDER_PDF);
	}
	
	public FOProcessor() {
		;
	}
	
	public FOProcessor(java.io.OutputStream outStream) {
		myFopDriver.setOutputStream(outStream);
	}

	public FOProcessor(java.io.InputStream inStream, java.io.OutputStream outStream) {
		myFopDriver.setInputSource(new org.xml.sax.InputSource(inStream));
		myFopDriver.setOutputStream(outStream);
	}
	
		
	public java.io.OutputStream getOutputStreamAsInput() 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
		java.io.File fileDir = new java.io.File(getExportAbsoluteDir());
		fileDir.mkdirs();
		
		try {
			myTempFile = java.io.File.createTempFile("tmp-formatingObject-",".xml.fo",fileDir);
			myTempOutputStream = new java.io.FileOutputStream(myTempFile.getCanonicalFile());
		}
		catch(java.io.IOException ioex) {
			throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in getOutputStreamAsInput()", ioex);
		}
		return myTempOutputStream;
	}
	
	public void render (org.w3c.dom.Document domFoDocument) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
		try {
			myFopDriver.render(domFoDocument);
		}
		catch(org.apache.fop.apps.FOPException fopex) {
			throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in render()", fopex);
		}
	}
	    		
	public void render(java.io.InputStream inStream)
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			try {
				myFopDriver.setInputSource(new org.xml.sax.InputSource(inStream));
				myFopDriver.run();
			}
			catch(java.io.IOException ioex) {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in render()", ioex);
			}
			catch(org.apache.fop.apps.FOPException fopex) {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in render()", fopex);
			}
		}

	public void renderOutputStream()
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
			try {
				java.io.FileInputStream myFileInputStream = new java.io.FileInputStream(myTempFile.getCanonicalFile());
				myFopDriver.setInputSource(new org.xml.sax.InputSource(myFileInputStream));
				myTempOutputStream.close();
				
				myFopDriver.run();
				
				myFileInputStream.close();
				myTempFile.delete();
			}
			catch(java.io.IOException ioex) {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in renderOutputStream()", ioex);
			}
			catch(org.apache.fop.apps.FOPException fopex) {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in renderOutputStream()", fopex);
			}
		}
	
			
	protected String getExportAbsoluteDir() throws PapillonBusinessException {            
		String baseDir = "";
		String mediaDir = "";
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
			throw new PapillonBusinessException("Error in Papillon Configuration File: ", ex);
		}
		baseDir = baseDir + mediaDir + EXPORT_VOLUME_DIR + java.io.File.separator;
		return baseDir;
	}	
}


















