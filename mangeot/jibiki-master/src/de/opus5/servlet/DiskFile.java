/*
    de.opus5.servlet - Handles multipart requests
    Copyright (C) 1999, Frederik Dahlke opus 5 interaktive medien gmbh
    email: dahlke@opus5.de http://www.opus5.de

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
                                                                                                                           
    opus 5 interaktive medien gmbh hereby disclaims all copyright interest
    in the library de.opus5.servlet (multipart request handling library)
    written by Frederik Dahlke.
    
*/

/*
 * DiskFile.java
 */

package de.opus5.servlet;

import java.io.*;

/**
 * Stores filedata on disk. Implements UploadedFile
 * interface.
 * @author Frederik Dahlke, opus 5 interaktive medien gmbh
 * @version $Id$
 */

public class DiskFile extends UploadedFile {

	private File file;

    private boolean keepFile = false;

	public DiskFile (String fieldname, String filename,
		String tmpDir, String uniqueFilename, boolean keep) throws IOException 
	{
		super(fieldname,filename);
        String ext = ".tmp";
/*        int index = uniqueFilename.lastIndexOf(".");
        if (index > -1) {
            ext = uniqueFilename.substring(index,uniqueFilename.length());
            uniqueFilename = uniqueFilename.substring(0,index);
        } */
		file = File.createTempFile(uniqueFilename,ext,new File(tmpDir));
        if (!keep) 
            file.deleteOnExit();
    }

	/**
	 * returns the corresponding java.io.File object.
	 * @return the file object.
	 */
	public File getFile() {
		return file;
	}

	/**
	 * Returns an InputStream for reading the
	 * filedata.
	 * @return an InputStream
	 */
	public InputStream getInputStream () throws IOException {
		return (InputStream) 
			(new FileInputStream(file));
	};
	
	
	/**
	 * Returns an OutputStream for writing the
	 * filedata
	 */
	public OutputStream getOutputStream () 
		throws IOException {
		return (OutputStream) 
			(new FileOutputStream(file));
	}

    protected void finalize() {
        // deleteOnExit does not work allways
        if (!keepFile) {
            file.delete();
        }
    }
}

