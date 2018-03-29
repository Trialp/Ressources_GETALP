/*
    de.opus5.servlet - Handles multipart requests
    Copyright (C) 1999, Frederik Dahlke opus 5 interaktive medien gmbh
    email: dahlke@opus5.de http://www.opus5.de

    Contributions by 
    Peter Speck, email: speck@joe.ing.dk

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
 * The original multipart request has been modified to better handle multilingual input
 * in multipart form-data. If the charset is specified by the client, it is used to read the parameters.
 *--------------------------------------------------------------
 * $Id$
 *--------------------------------------------------------------
 * $Log$
 * Revision 1.3  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.2  2006/02/26 19:21:38  mangeot
 * Work on BrowseVolume
 *
 * Revision 1.1.1.1  2004/12/06 16:38:24  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 * Revision 1.1.1.1  2004/12/06 16:38:24  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * Revision 1.1.1.1  2002/10/28 16:49:11  serasset
 * Creation of the papillon CVS repository for enhydra 5.0
 *
 * Revision 1.2  2001/11/16 13:56:49  serasset
 * Correction d'un probleme d'encodage de chaine dans la consultation d'un fichier d'information.
 *  
 */

//
// MultipartRequest.java
//


package de.opus5.servlet;

import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.util.*;

// This has been added to Multipart original code in order to better take into account the
// content-type's charset parameter.
import com.lutris.mime.ContentHeader;

/**
 * A class for handling multipart form uploads.
 * This version is for JSDK 2.2.
 * Unlike com.oreilly.servlet.MultipartRequest this
 * class returns a InputStream for reading the
 * file's content, so you can also write the data
 * into magic crystals instead of writing it to
 * the file system. ;) 
 * There is also the option to save files to
 * disk and return a java.io.File object.
 * You can also access the file data via the 
 * getUploadedFile() method, which returns an
 * file object implementing the 
 * de.opus5.servlet.UploadedFile interface.
 * The class can decide based on the total request size
 * if the file data is temporary stored in memory
 * or in the file system.
 * <p>
 * Example:
 * <blockquote><pre>
 * MulipartRequest req;
 * ..
 * try {
 *    req = new MultipartRequest(servletRequest,maxMemoryStoredRequestSize,
 *			tmpdir,fileprefix);
 * } catch (Exception e) {
 *   ...
 * }
 *
 * ....
 * filename = req.getUploadedFile("Document").getName();
 * is = req.getInputStreamForFile("Document");
 * while (is.read()...) {
 * ...
 * }
 * login = req.getParameter("Login");
 * ...
 * UploadedFile adressFile = mReq.getUploadedFile(Global.MPADRESSF_KEY);
 * UploadedFile docFile = mReq.getUploadedFile(Global.MPDOCFILE_KEY);
 * myObject.setAdressFile(adressFile); 
 * ....
 * ....
 * </pre></blockquote>
 * </p>
 * @author Frederik Dahlke, opus 5 interaktive medien gmbh
 * @version $Id$
 */


public class MultipartRequest implements HttpServletRequest {

	private static final int DEFAULT_MAX_REQUEST_SIZE = 1048576; // 1 MB
	private static final int DEFAULT_MAX_MEMORY_STORED_REQUEST_SIZE = 0;

	//
	// the parameter boundary
	//
	private String boundary;

	//
	// totaly bytes read so far
	//
	int bytesRead = 0;

	//
	// Store files and data
	//
	private Hashtable files = new Hashtable();

	// 
	// if set all files must be saved to
	// disk and therefore maxMemoryStoredRequestSize
	// must be set to 0.
	// Temporary files must not be deleted.
	//
	private boolean keepFiles = false;

	//
	// if the request exceeds this size, all files
	// are stored on disk, they are kept in memory
	// otherwise. Setting this to 0, means all files
	// will be stored on disk, setting it to 
	// maxRequestSize, will keep all files in memory.
	//
	private int maxMemoryStoredRequestSize; 

	//
	// maximum size of request (bytes)
	//
	private int maxRequestSize; 

	//
	// Store parameter - values in this hash
	//
	private BucketHash parameterValues = new BucketHash();

	//
	// request size
	//
	int requestSize;

	//
	// the input stream 
	//
	private ServletInputStream servletInputStream;
	
	//
	// The original HttpServletRequest
	//
	private HttpServletRequest servletRequest;

	//
	// temporary directory for saving uploaded files
	// 
	private String tmpDir;

	// 
	// unique prefix for files, to prevent overwriting
	// files with the same name
	//
	private String uniquePrefix;


	/**
	 * Simple class for storing parameters
	 */
    public class BucketHash extends Hashtable {
        
        public Object put(Object key, Object value) {
            Vector values;
            // is this key in here already ?
            if (containsKey(key)) {
                // get vector 
                values = (Vector) super.get(key);
            } else {
                // create Vector
                values = new Vector();
                super.put(key,values);
            }
            
            values.addElement(value);
            return null;
        }

        public Object getFirst (Object key) {
            Vector values = (Vector) super.get(key);
            if (values != null) {
                return values.firstElement();
            }
            return null;
        }
        
        public int elementCount (Object key) {
            Vector values = (Vector) super.get(key);
            if (values != null) {
                return values.size();
            }
            return 0;
        }
        
        public Enumeration getElements (Object key) {
            Vector values = (Vector) super.get(key);
            if (values != null) {
                return values.elements();
            }
            return null;
        }
    }
    
    /**
     * This constructor is intentionaly private to prevent
     * developers calling the class with obscure settings.
     * All desired behaviours can be obtained by calling
     * one of the public constructors.
     * @param sR - the HttpServletRequest
     */
    private MultipartRequest (HttpServletRequest sR, int maxMemStoredReqSiz,
                              int maxReqSiz, String tmpDir,
                              String uniquePrefix, boolean keepFiles) 
		throws IllegalArgumentException, IOException {
        this.tmpDir = tmpDir;
        this.uniquePrefix = uniquePrefix;
        this.maxMemoryStoredRequestSize = maxMemStoredReqSiz;
        this.maxRequestSize = maxReqSiz;
        this.keepFiles = keepFiles;
        
        // Checks the request object, throws 
        // IllegalArgumentException, if anything is wrong.
        this.checkServletRequest(sR);
        // debug
        
        this.servletRequest = sR;
        
        this.readRequest();
    }

    /**
     * Create a MultipartRequest, 
     * temporary files will be deleted.
     * @param sR the HttpServletRequest
     * @param maxMemStoredReqSiz If the request size in bytes exceeds
     * this value, the uploaded files will be stored on disk.
     * @param maxReqSiz Maximum size of request.
     * @param tmpDir Directory where temporary files are saved.
     * @param uniquePrefix Prefix added to the filenames to prevent
     * overwriting of files with the same name.
     */
    public MultipartRequest (HttpServletRequest sR, int maxMemStoredReqSiz,
                             int maxReqSiz, String tmpDir,
                             String uniquePrefix) 
        throws IllegalArgumentException, IOException {
        this(sR,maxMemStoredReqSiz,maxReqSiz,
             tmpDir,uniquePrefix,false);
    }       
	
    /**
     * Create a MultipartRequest,
     * temporary files will be deleted, the maximum request size
     * is set to 1MB.
     * @param sR the HttpServletRequest
     * @param maxMemStoredReqSiz If the request size in bytes exceeds
     * this value, the uploaded files will be stored on disk.
     * @param tmpDir Directory where temporary files are saved.
     * @param uniquePrefix Prefix added to the filenames to prevent
     * overwriting of files with the same name.
     */
    public MultipartRequest (HttpServletRequest sR, int maxMemStoredReqSiz,
                             String tmpDir, String uniquePrefix) 
        throws IllegalArgumentException, IOException {
        this(sR,maxMemStoredReqSiz,DEFAULT_MAX_REQUEST_SIZE,
             tmpDir,uniquePrefix,false);
    }

    /**
     * Create a MultipartRequest,
     * files will be saved to disk allways.
     * @param sR the HttpServletRequest
     * @param maxReqSiz Maximum size of request.
     * @param tmpDir Directory where temporary files are saved.
     * @param uniquePrefix Prefix added to the filenames to prevent
     * overwriting of files with the same name.
     * @param keep If set, the uploaded files will not be deleted.
     */
    public MultipartRequest (HttpServletRequest sR,int maxReqSiz, String tmpDir, 
                             String uniquePrefix, boolean keep) 
        throws IllegalArgumentException, IOException 
	{
            this(sR,0,maxReqSiz,tmpDir,
                 uniquePrefix,keep);
	}

    /**
     * Create a MultipartRequest using the HttpServletRequest object,
     * temporary files will be deleted.
     * Maximum request size is 1MB, files will be saved to disk.
     * @param sR the HttpServletRequest
     * @param tmpDir Directory where temporary files are saved.
     * @param uniquePrefix Prefix added to the filenames to prevent
     * overwriting of files with the same name.
     */
    public MultipartRequest (HttpServletRequest sR,String tmpDir, 
                             String uniquePrefix) 
        throws IllegalArgumentException, IOException 
	{
            this(sR,DEFAULT_MAX_MEMORY_STORED_REQUEST_SIZE,DEFAULT_MAX_REQUEST_SIZE,
                 tmpDir,
                 uniquePrefix,false);
	}

    /**
     * Check if the request is not null and has the
     * appropriate content type.
     */
    private void checkServletRequest(HttpServletRequest sR) 
        throws IllegalArgumentException, IOException
	{
            //
            // Check null...
            //
            if (sR == null)
                throw new IllegalArgumentException("Request object is null.");
            
            
            //
            // maximum request size
            //
            this.requestSize = sR.getContentLength();
            if (this.requestSize > maxRequestSize && maxRequestSize != -1) {
                throw new IllegalArgumentException(
                    "Request size of " + this.requestSize
                    +" bytes exceeds maximum request size ( "
                    + maxRequestSize +")");
            }	
            
            //
            // content type and boundary
            //
            String contentType = sR.getContentType();
            if (!(contentType.toLowerCase().startsWith("multipart/form-data")))
                throw new IllegalArgumentException("Unknown content type.");

            int index = contentType.indexOf("boundary=");
            if (index == -1)
                throw new IllegalArgumentException ("No boundary specified.");

            index += 9;
            
            this.boundary = "--" + contentType.substring(index);
	            
            // 
            // construct input stream
            //
            this.servletInputStream = sR.getInputStream();
	}

	/**
	 * Read File Data
	 */
	private UploadedFile readFileData(String filename, 
		String fieldname, String contentType) 
		throws IOException 
	{
		byte [] buffer = new byte[8 * 1024];
		boolean ret = false;
		boolean checkBoundary = false;
		UploadedFile upfile;
		OutputStream os;
		BufferedOutputStream bos;
		int result, filesize,buffPos;
		
		// debug

		if (requestSize > maxMemoryStoredRequestSize) {
			upfile = new DiskFile(fieldname,filename, 
			tmpDir,uniquePrefix + filename,
			keepFiles);
			// debug
		} else {
			upfile = new MemoryFile (fieldname, filename);
			// debug
		}
		upfile.setContentType(contentType);
		
		os = upfile.getOutputStream();
		bos = new BufferedOutputStream(os);
		
		filesize = 0;
		result =0;
		buffPos = 0;

		// debug
		while((bytesRead < requestSize) 
			&& ((result = servletInputStream.read()) != -1 )) {
			// servletInputStream.readline() sucks
			// this implementation sucks,too 
			// the whole class has to be reimplemented using
			// an BufferedInputStream for reading from
			// the ServletInputStream

			bytesRead++;
			
			if (checkBoundary) {
				// try to read bytes up to boundary length
				// break if \r occours
				buffer[buffPos] = (byte) result;
				buffPos++;
				if (result == '\r') {
						bos.write('\r'); bos.write('\n');
						bos.write(buffer,0,buffPos-1);
						filesize += buffPos+2;
						buffPos = 0;
						checkBoundary = false;
				} else if (buffPos == boundary.length()) {
					// check for boundary
					String line = new String(buffer,0,buffPos,"ISO-8859-1");
					if (line.startsWith(boundary)) {
						// read \r\n at end of boundary as well
						if (servletInputStream.read() != -1) bytesRead++;
						if (servletInputStream.read() != -1) bytesRead++;
						// EOF, there's no more data to read
						break;
					} else {
						// insert buffer we read
						bos.write('\r'); bos.write('\n');
						bos.write(buffer,0,buffPos);
						filesize += buffPos+2;
						buffPos = 0;
						checkBoundary = false;
						ret = false;
					}
				}
			} else if (ret) {
				if (result == '\n') {
					checkBoundary = true;
				} else {
					bos.write('\r');bos.write(result);
					filesize += 2;
					ret = false;
				}
			} else if (result == '\r') {
				ret = true;
			} else {
				bos.write(result);
				filesize++;
			}
		}
		
		bos.flush();
		bos.close();
		os.close();
		upfile.setSize(filesize);
		// debug
		return upfile;
	}		


	/**
	 * Read next line of input.
  	 * @return byte array, null at end of stream
     */
    private byte [] readLine() throws IOException {

		byte [] line = this.readLineSimple();
                
		if (line == null)
			return null;
		// cut \r\n
                byte [] strippedLine = new byte [line.length - 2];
                
		for (int i = 0; i < strippedLine.length; i++) 
                    strippedLine[i]=line[i];
		return strippedLine;
	}
	
    /**
	 * Read next line of input, do not chop \r\n .
  	 * @return byte array, null at end of stream
     */
    private byte[] readLineSimple() throws IOException {
		byte [] buffer = new byte [8 * 1024];
                ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
                // StringBuffer stringBuffer = new StringBuffer();
		int result;
                //String line;
		
		if (this.bytesRead >= this.requestSize) {
			return null;
		}

		do {
			result = servletInputStream.readLine(buffer,0,buffer.length);
			if (result != -1) {
				this.bytesRead += result;
                                //this.showLineBytes(buffer, 0, result);
				byteBuffer.write(buffer,0,result);
			}
		} while (result == buffer.length);

		if (byteBuffer.size() == 0)
			return null;

		return byteBuffer.toByteArray();
	}

// Debug only !!!
private static void showLineBytes(byte[] buffer, int start, int end) {
    for(int i = start; i < end; i++) {
        if ((i % 18) == 0) System.out.println();
        int v = new Byte(buffer[i]).intValue();
        
        System.out.print(Integer.toHexString(v));
        System.out.print(" ");
    }
    System.out.println();
}

	
	/**
	 * read the parameter's data.
	 * @return the parameter data
	 */
	private byte[] readParameterData() throws IOException {
                ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
		byte [] bytes;
                //StringBuffer stringBuffer = new StringBuffer();
		String line;
                
                // The first line is treated independently, as it is not preceded by a crlf
                if (((bytes = this.readLine()) != null) && (! (new String(bytes,"ISO-8859-1")).startsWith(boundary))) {
                    byteBuffer.write(bytes, 0, bytes.length);
                    while (((bytes = this.readLine()) != null) && (! (new String(bytes,"ISO-8859-1")).startsWith(boundary))) {
                        byteBuffer.write('\r'); //cr
                        byteBuffer.write('\n'); //lf
                        byteBuffer.write(bytes, 0, bytes.length);
                    }
                }
			
		if (byteBuffer.size() == 0) return null;

		return byteBuffer.toByteArray();
	}


	
	/**
	 * read all paramters and filenames from request
	 * and store them in the hashes.
	 */
	private void readRequest() throws IOException {
                byte [] bytes;
		//String line = "";

		// first line is boundary
		// we do not need this
		bytes = this.readLine();

		while ((bytes = this.readLine()) != null) {
                    if (bytes.length == 0) continue;
                    
                        // The first line should be a Mime content-disposition header with form-data value.
                        String line = new String(bytes, "ISO-8859-1");
                        ContentHeader cont = new ContentHeader(line);
                        
                        if (! (cont.getHeaderType().toLowerCase().equals("content-disposition") &&
                               cont.getValue().toLowerCase().equals("form-data")))
                           throw new IOException(
					"Unknown or no content-disposition found.");
 
			//
			// get field name
			//
                        String fieldname = cont.getParameter("name");
                        if (null == fieldname)
                            throw new IOException (
					"No field name found in content-disposition.");
			
			//
			// get file name
			//
			String filename = cont.getParameter("filename");
                        if (null != filename) {
                            int basename = Math.max(filename.lastIndexOf("/"),
                                    filename.lastIndexOf("\\"));
                            if (basename != -1)
                                    filename = filename.substring(basename+1);
			}
			

			// 
			// next line will be emty or content-type definition
			//
			String contentType = null;
                        String charset = null;
			if ((bytes = this.readLine()) == null) {
				throw new IOException("Malformed multipart-request.");
			}
                        line = new String(bytes, "ISO-8859-1");
                        cont = new ContentHeader(line);
                        String htype = cont.getHeaderType();
                        if (htype != null && htype.toLowerCase().equals("content-type")) {
                            // Get the content type
                            contentType = cont.getValue();
                            // AND the eventuel charset...
                            charset = cont.getParameter("charset");
                            // eat all lines until first empty line
                            do {
                                bytes = this.readLine();
                                    if (bytes == null) 
                                            throw new IOException("Malformed multipart-request.");
                            } while (bytes.length != 0);
			}		
			
			//
			// handle file upload
			// 
			if (filename != null) {
				UploadedFile file = this.readFileData(filename,fieldname,
					contentType);
				this.files.put(fieldname,file);
				continue;
			} 
			
			//
			// this is no file, handle parameter
			//
                        if (null == charset) charset = "ISO-8859-1";
                        byte [] pdata = this.readParameterData();
                        if (null != pdata)
                            this.parameterValues.put(fieldname,new String(pdata,charset));
		}	
	}



	/**
	 * returns a java.io.File object if the file
	 * was saved on disk, null otherwise. 
	 * This this method should only be called, if
	 * keepFiles is set to true.
	 * @param fieldname the form parameter field name 
	 * @return the java.io.File object
	 */
	public File getFile(String fieldname) {
		if (keepFiles)
			return ((UploadedFile)files.get(fieldname)).getFile();
		else
			return null;
	}

	/**
	 * return an InputStream for reading an uploaded
	 * file.
	 * @param filename the form parameter field name
	 * @return InputStream for reading the filedata.
	 */
	public InputStream getInputStreamForFile(String fieldname) throws
		IOException {
		return ((UploadedFile)files.get(fieldname)).getInputStream();
	}

	/**
	 * return the names of all parameters as Enumeration
	 * @return Enumeration of parameter names
	 */
	public Enumeration getParameterNames() {
		return parameterValues.keys();
	}

	/** 
	 * return the value of parameter with specified name
	 * @param name the parameter name
	 * @return the parameter value
	 */
	public String getParameter(String name) {
		return (String) parameterValues.getFirst(name);
	}

	/**
	 * return the value of the specified parameter as
	 * an array of strings.
	 */
	public String [] getParameterValues (String name) {
		int size = parameterValues.elementCount(name);
		String [] values = new String [size];
		int c = 0;
		for (Enumeration e = parameterValues.getElements(name);
			e.hasMoreElements();c++) {
			values[c] = (String) e.nextElement();
		}
		return values;
	}

	
	/**
	 * Get the UploadedFile object for the
	 * uploaded file with given name.
	 * @param fieldname the form parameter field name
	 */
	public UploadedFile getUploadedFile(String fieldname) {
			return (UploadedFile)files.get(fieldname);
	}

    /**
     * Get an enumeration of the UploadedFile objects.
     */
    public Enumeration getUploadedFiles() {
        return files.elements();
    }

    


    // All methods below are simply passed through to the wrapped object.
    

    /**
     * Returns the size of the request entity data, or -1 if not known. Same
     * as the CGI variable CONTENT_LENGTH.
     */
    public int getContentLength() {
        return servletRequest.getContentLength();
    }
    
    /*
     * Returns the Internet Media Type of the request entity data, or null if
     * not known. Same as the CGI variable CONTENT_TYPE.
     */
    public String getContentType() {
        return servletRequest.getContentType();
    }

    /**
     * Returns the protocol and version of the request as a string of
     * the form <code>&lt;protocol&gt;/&lt;major version&gt;.&lt;minor
     * version&gt</code>.  Same as the CGI variable SERVER_PROTOCOL.
     */
    public String getProtocol() {
        return servletRequest.getProtocol();
    }

    /**
     * Returns the scheme of the URL used in this request, for example
     * "http", "https", or "ftp".  Different schemes have different
     * rules for constructing URLs, as noted in RFC 1738.  The URL used
     * to create a request may be reconstructed using this scheme, the
     * server name and port, and additional information such as URIs.
     */
    public String getScheme() {
        return servletRequest.getScheme();
    }

    /**
     * Returns the host name of the server that received the request.
     * Same as the CGI variable SERVER_NAME.
     */
    public String getServerName() {
        return servletRequest.getServerName();
    }

    /**
     * Returns a boolean indicating whether this
     * request was made using a secure channel, such as
     * HTTPS.
     */
    public boolean isSecure() {
        return servletRequest.isSecure();
    }

    /**
     * Removes an attribute from this request.
     */
    public void removeAttribute(String name) {
        servletRequest.removeAttribute(name);
    }

    /**
     * Stores an attribute in this request.
     */
    public void setAttribute(String name, Object o) {
        servletRequest.setAttribute(name,o);
    }

    /**
     * Returns the port number on which this request was received.
     * Same as the CGI variable SERVER_PORT.
     */
    public int getServerPort() {
        return servletRequest.getServerPort();
    }

    /**
     * Returns the IP address of the agent that sent the request. Same as the
     * CGI variable REMOTE_ADDR.
     */
    public String getRemoteAddr() {
        return servletRequest.getRemoteAddr();
    }

    /**
     * Returns the fully qualified host name of the agent that sent the
     * request. Same as the CGI variable REMOTE_HOST.
     */
    public String getRemoteHost() {
        return servletRequest.getRemoteHost();
    }

    /**
     * Returns a RequestDispatcher object that acts
     * as a wrapper for the resource located at the given
     * path.
     */
    public RequestDispatcher getRequestDispatcher(String path) {
        return servletRequest.getRequestDispatcher(path);
    }

    /**
     * Returns the method with which the request was made. The returned
     * value can be "GET", "HEAD", "POST", or an extension method. Same
     * as the CGI variable REQUEST_METHOD.
     */
    public String getMethod() {
        return servletRequest.getMethod();
    }

    /**
     * Returns the request URI.
     */
    public String getRequestURI() {
        return servletRequest.getRequestURI();
    }

    /**
     * Returns extra path information translated to a real path. Returns
     * null if no extra path information specified. Same as the CGI variable
     * PATH_TRANSLATED.  
     */
    public String getPathTranslated() {
        return servletRequest.getPathTranslated();
    }

    /**
     * Returns the query string part of the presentation URI, or null if none.
     * Same as the CGI variable QUERY_STRING.
     */
    public String getQueryString() {
        return servletRequest.getQueryString();
    }

    /**
     * Returns the name of the user making this request, or null if not
     * known.  The user name is set with HTTP authentication.  Whether
     * the user name will continue to be sent with each subsequent
     * communication is browser-dependent.  Same as the CGI variable
     * REMOTE_USER.
     */
    public String getRemoteUser() {
        return servletRequest.getRemoteUser();
    }

    /**
     * Returns the authentication scheme of the request, or null if none.
     * Same as the CGI variable AUTH_TYPE.
     */
    public String getAuthType() {
        return servletRequest.getAuthType();
    }

    /**
     * Returns the portion of the request URI that
     * indicates the context of the request.
     */
    public String getContextPath() {
        return servletRequest.getContextPath();
    }

    /**
     * Gets the array of cookies found in this request.
     *
     * @return The array of cookies found in this request.
     */
    public Cookie[] getCookies() {
        return servletRequest.getCookies();
    }
  
    
    /**
     * Returns the value of a header field, or null if not known.
     * The case of the header field name is ignored.
     * @param name the case-insensitive header field name
     */
    public String getHeader(String name) {
        return servletRequest.getHeader(name);
    }

    /**
     * Returns the value of an integer header field, or -1 if not found.
     * The case of the header field name is ignored.
     * @param name the case-insensitive header field name
     */
    public int getIntHeader(String name) {
        return servletRequest.getIntHeader(name);
    }

    /**
     * Returns the value of a date header field, or -1 if not found.
     * The case of the header field name is ignored.
     * @param name the case-insensitive header field name
     */
    public long getDateHeader(String name) {
        return servletRequest.getDateHeader(name);
    }

    /**
     * Returns an enumeration of strings representing the header names
     * for this request. Some server implementations do not allow headers
     * to be accessed in this way, in which case this method will return null.
     */
    public Enumeration getHeaderNames() {
        return servletRequest.getHeaderNames();
    }

    /**
     * Returns all the values of the specified request
     *  header as an Enumeration of String objects.
     */
    public Enumeration getHeaders(String name) {
        return servletRequest.getHeaders(name);
    }

    /**
     * Calls the corresponding method in ServletRequest-object.
     */
    public Object getAttribute(String name) {
        return servletRequest.getAttribute(name);
    }
    
    /**
     * Calls the corresponding method in ServletRequest-object.
     */
    public Enumeration getAttributeNames() {
        return servletRequest.getAttributeNames();
    }

    /**
     * Calls the corresponding method in ServletRequest-object.
     */
    public String getRealPath(String path) {
        return servletRequest.getRealPath(path);
    }
    
    /**
     * Calls the corresponding method in ServletRequest-object.
     */
    public ServletInputStream getInputStream() throws IOException {
        return servletRequest.getInputStream();
    }
    
    /**
     * Calls the corresponding method in ServletRequest-object.
     */
    public Locale getLocale() {
        return servletRequest.getLocale();
    }

    /**
     * Calls the corresponding method in ServletRequest-object.
     */
    public Enumeration getLocales() {
        return servletRequest.getLocales();
    }

    /**
     * Calls the corresponding method in ServletRequest-object.
     */
    public BufferedReader getReader() throws IOException {
        return servletRequest.getReader();
    }
    
    /**
     * Calls the corresponding method in ServletRequest-object.
     */
    public String getCharacterEncoding() {
        return servletRequest.getCharacterEncoding();
    }

    
    
    /** @see HttpPresentationRequest#getRequestedSessionId */
    public String getRequestedSessionId() {
        return servletRequest.getRequestedSessionId();
    }

    
    /** @see HttpPresentationRequest#getPathInfo */
    public String getPathInfo() {
        return servletRequest.getPathInfo();
    }
    
    
    /** @see HttpPresentationRequest#isRequestedSessionIdFromCookie */
    public boolean isRequestedSessionIdFromCookie() {
        return servletRequest.isRequestedSessionIdFromCookie();
    }

    
    /** @see HttpPresentationRequest#isRequestedSessionIdFromUrl */
    public boolean isRequestedSessionIdFromUrl() {
        return servletRequest.isRequestedSessionIdFromUrl();
    }

    /** @see HttpPresentationRequest#isRequestedSessionIdFromURL */
    public boolean isRequestedSessionIdFromURL() {
        return servletRequest.isRequestedSessionIdFromURL();
    }


    
    /** @see HttpPresentationRequest#getSession */
    public HttpSession getSession() {
        return servletRequest.getSession();
    }

     /** @see HttpPresentationRequest#getSession */
    public HttpSession getSession(boolean create) {
        return servletRequest.getSession(create);
    }
    
     /** @see HttpPresentationRequest#getUserPrinicpal */
    public java.security.Principal getUserPrincipal() {
        return servletRequest.getUserPrincipal();
    }

     /** @see HttpPresentationRequest#isRequestedSessionIdValid */
    public boolean isRequestedSessionIdValid() {
        return servletRequest.isRequestedSessionIdValid();
    }

    /** @see HttpPresentationRequest#isUserInRole */
    public boolean isUserInRole(String role) {
        return servletRequest.isUserInRole(role);
    }

    /** @see HttpPresentationRequest#getServletPath */
    public String getServletPath() {
        return servletRequest.getServletPath();
    }
}


