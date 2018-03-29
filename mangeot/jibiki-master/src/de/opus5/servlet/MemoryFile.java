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
 * MemoryFile.java
 */

package de.opus5.servlet;

import java.io.*;

/**
 * Stores filedata in memory. Implements UploadedFile
 * interface.
 * @author Frederik Dahlke, opus 5 interaktive medien gmbh
 * @version $Id$
 */

public class MemoryFile extends UploadedFile {

	private ByteArrayOutputStream byteArrayOS;
	

	public MemoryFile (String fieldname, String filename) {
		super (fieldname, filename);
		byteArrayOS = new ByteArrayOutputStream();
	}

	public MemoryFile (String fieldname, String filename, int size) {
		super (fieldname, filename);
		byteArrayOS = new ByteArrayOutputStream(size);
	}

	
	/**
	 * Returns an InputStream for reading the
	 * filedata.
	 * @return an InputStream
	 */
	public InputStream getInputStream () throws IOException {
		return (InputStream) 
			(new ByteArrayInputStream(this.byteArrayOS.toByteArray()));
	};
	
	
	/**
	 * Returns an OutputStream for writing the
	 * filedata.
	 * @return an OutputStream
	 */
	public OutputStream getOutputStream () throws IOException {
		return (OutputStream) this.byteArrayOS;
	}
}

