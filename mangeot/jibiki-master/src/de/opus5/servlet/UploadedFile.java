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
 * UploadedFile.java
 */

package de.opus5.servlet;

import java.io.*;

/**
 * Abstract base class for Uploaded files.
 * Descendants must implement getInputStream() and
 * getOutputStream().
 * You should set the filesize.
 * @author Frederik Dahlke, opus 5 interaktive medien gmbh
 * @version $Id$
 */

public abstract class UploadedFile {

	//
	// keeps the file's contentType
	//
	private String contentType;

	// 
	// the file's name
	//
	private String filename;

	//
	// the form parameter name
	//
	private String formParameterName;

	//
	// filesize
	//
	private int size;


	/**
	 * Create a new UploadedFile
	 * @param paraname the form field name
	 * @param filename the file's name
	 * @param size the file's size in bytes
	 */
	public UploadedFile (String paraname, String filename) {
		this.formParameterName = paraname;
		this.filename = filename;
	}

	/**
	 * Retruns file's contentType
	 * @retrun contentType
	 */
	public String getContentType() {
		return contentType;
	}

	/**
	 * Classes extending UploadedFile can override this
	 * method and return the appropriate file object.
	 * By default this Method allways returns null.
	 */
	public File getFile() {
		return null;
	}

	/**
	 * returns the name of the file upload
	 * form field.
	 * @return form parameter name
	 */
	public String getFormParameterName() {
		return this.formParameterName;
	}

	
	/**
	 * Returns an InputStream for reading the
	 * filedata.
	 * @return an InputStream
	 */
	public abstract InputStream getInputStream () throws IOException;
	
	/**
	 * Returns the file's name.
	 * @return the filename
	 */
	public String getName() {
		return this.filename;
	}
	
	/**
	 * Returns an OutputStream for reading the
	 * filedata.
	 * @return an OutputStream
	 */
	public abstract OutputStream getOutputStream () throws IOException;

	/**
	 * Returns the file's size.
	 * @return the filesize in bytes 
	 */
	public int getSize() {
		return this.size;
	}

	/**
	 * set file's contentType
	 * @param contentType
	 */
	public void setContentType(String ct) {
		contentType = ct;
	}
	
	/**
	 * set file's size
	 * @param filesize
	 */
	public void setSize(int size) {
		this.size = size;
	}

}

