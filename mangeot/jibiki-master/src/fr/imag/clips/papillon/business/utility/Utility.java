/*
*
*----------------------------------------
*$Id$
*------------------------------------------
*$Log$
*Revision 1.19  2007/01/05 13:57:26  serasset
*multiple code cleanup.
*separation of XMLServices from the Utility class
*added an xml parser pool to allow reuse of parser in a multithreaded context
*added a new field in the db to identify the db layer version
*added a new system property to know which db version is known by the current app
*
*Revision 1.18  2006/12/14 20:03:26  fbrunet
*Add method to normalize value into XML structure.
*
*Revision 1.17  2006/12/13 09:32:00  fbrunet
**** empty log message ***
*
*Revision 1.16  2006/08/13 14:42:20  mangeot
*Re-added the ArrayIntersection method with String[] arrays for retro-compatibility
*-------------------------------------------------------------------
*
*Revision 1.15  2006/08/10 22:55:09  mangeot
*Added a method for printong a DOM tree into a file
*
*Revision 1.14  2006/08/10 22:17:13  fbrunet
*- Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
*- Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
*- Bug correction : +/- in advanced search
*
*Revision 1.13  2006/08/10 21:27:49  mangeot
*Added a boolean for printing the doctype when bulding a DOM tree
*
*Revision 1.12  2006/08/10 19:21:58  mangeot
**** empty log message ***
*
*Revision 1.11  2006/04/10 12:22:19  mangeot
*Added XMl indeted or not when doing serialization in Node2String
*
*Revision 1.10  2006/02/26 14:08:16  mangeot
*Added the multilingual_sort(lang,headword) index on volume tables for speeding up the lookup
*
*Revision 1.9  2005/11/21 17:41:36  mangeot
**** empty log message ***
*
*Revision 1.8  2005/11/20 18:03:22  mangeot
**** empty log message ***
*
*Revision 1.7  2005/07/16 12:58:31  serasset
*Added limit parameter to query functions
*Added a parameter to Formater initializations
*Developped a new Advanced search functionality with reusable code for the query form handling...
*
*Revision 1.6  2005/06/20 16:55:05  mangeot
*multiple bug fixes
*
*Revision 1.5  2005/06/17 17:53:39  mangeot
**** empty log message ***
*
*Revision 1.4  2005/06/15 16:48:28  mangeot
*Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
*
*Revision 1.3.4.4  2005/06/01 08:38:43  mangeot
*Multi bug correction + added the possibility of disabling data edition
*via the Admin.po page
*
*Revision 1.3.4.3  2005/05/19 17:02:22  mangeot
*Importing entries without the contribution element
*
*Revision 1.3.4.2  2005/04/29 17:30:30  mangeot
**** empty log message ***
*
*Revision 1.3.4.1  2005/04/29 14:50:25  mangeot
*New version with contribution infos embedded in the XML of the entries
*
*Revision 1.3  2005/04/11 12:29:59  mangeot
*Merge between the XPathAndMultipleKeys branch and the main trunk
*
*Revision 1.2.2.2  2005/01/27 15:56:21  mangeot
*Able to load a volume with XPointers, cannot lookup the result yet.
*Does not compile but commit for backup
*
*Revision 1.2.2.1  2005/01/25 13:54:54  mangeot
*changed the volume volumeEntry and index objects. Does not compile but need a backup...
*
*Revision 1.2  2004/12/24 14:31:28  mangeot
*I merged the latest developments of Papillon5.0 with this version 5.1.
*Have to be tested more ...
*
*Revision 1.1.1.1  2004/12/06 16:38:32  serasset
*Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
*There are still bugs in the code.
*
*--------------------------------------
*/
package fr.imag.clips.papillon.business.utility;

import java.io.*;
import java.util.Collection;
import java.util.ArrayList;
import java.util.Iterator;

//pour parser le document avec le DOM
import org.w3c.dom.*;

//for URLs

//pour utiliser ant: untar
//import org.apache.tools.tar.*;


//pour parser les doc html
import org.w3c.tidy.Tidy;

// pour les sax

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.xml.XMLServices;

public class Utility {


	private static final java.util.Locale PapillonLocale = new java.util.Locale("C");
	public static final java.text.DateFormat PapillonPrintDateFormat = new java.text.SimpleDateFormat("EEE MMM dd HH:mm:ss z yyyy", PapillonLocale);
	public static final java.text.DateFormat PapillonCDMDateFormat = new java.text.SimpleDateFormat("yyyy/MM/dd HH:mm:ss", PapillonLocale);
	public static final java.text.DateFormat PapillonShortDateFormat = new java.text.SimpleDateFormat("yyyy/MM/dd", PapillonLocale);
	public static final java.text.DateFormat LogDateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss,S", PapillonLocale);

	protected static java.util.Random theRandomGenerator = null;
	protected static java.util.regex.Pattern starPattern = java.util.regex.Pattern.compile("^[★]+$");


	/**
     * DOM convenience method getTextChild of an element
     *
     * @return a String
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */	
		public static String getText(Element myElt) {
		String res = "";
		if (myElt.hasChildNodes()) {
			NodeList childNodes = myElt.getChildNodes();
			for (int i=0; i< childNodes.getLength();i++) {
				res += childNodes.item(i).getNodeValue();
			}
		}
		return res;
	}
	
	/**
     * DOM convenience method setTextChild of an element
     *
     * @return a String
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */	
	public static void setText(Element myElt, String myValue) {
		Node myTextNode = myElt.getOwnerDocument().createTextNode(myValue);
		removeChildNodes(myElt);
		myElt.appendChild(myTextNode);
	}

	public static void removeChildNodes(Element myElt) {
		while (myElt.hasChildNodes()) {
			myElt.removeChild(myElt.getFirstChild());
		}
	}
    
    public static String encodeXMLEntities(String theString) {
        if (null != theString && !theString.equals("")) {
            String tmpString = "";
            while (null != theString && theString.indexOf("&") >= 0) {
                tmpString = theString.substring(0,theString.indexOf("&"))
                        + "&amp;";
                theString = theString.substring(theString.indexOf("&") + 1);
            }
            theString = tmpString + theString;

            while (theString.indexOf("'") >= 0 || theString.indexOf("\"") >= 0 
                || theString.indexOf("<") >= 0 ||theString.indexOf(">") >= 0) {
                if (theString.indexOf("'") >= 0) {
                    theString = theString.substring(0,theString.indexOf("'"))
                        + "&#39;" + theString.substring(theString.indexOf("'") + 1);
                }
                if (theString.indexOf("\"") >= 0) {
                    theString = theString.substring(0,theString.indexOf("\""))
                        + "&quot;" + theString.substring(theString.indexOf("\"") + 1);
                }
                if (theString.indexOf("<") >= 0) {
                    theString = theString.substring(0,theString.indexOf("<"))
                        + "&lt;" + theString.substring(theString.indexOf("<") + 1);
                }
                if (theString.indexOf(">") >= 0) {
                    theString = theString.substring(0,theString.indexOf(">"))
                        + "&gt;" + theString.substring(theString.indexOf(">") + 1);
                } 
            }
        }
    return theString;
    }                

		/**
     * Trims the XML header of an XML String
     *
     * @return a String
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	public static String trimXMLHeader(String xmlString) {
		String region = "?>";
		if (xmlString.indexOf(region) > 0) {
			xmlString = xmlString.substring(xmlString.indexOf(region)+region.length());
		}
		return xmlString;
	}
	
		/**
     * converts a String from an encoding to another
     *
     * @return a String
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public static String convertFromEncoding(String theString, String encoding)
        throws java.io.UnsupportedEncodingException {
            byte[] pbytes = theString.getBytes();
            return new String(pbytes, encoding);
        }

		/**
     * converts a String for writing an URL
     *
     * @return a String
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public static String convertToUrlForEncoding(String str, String encoding) throws java.io.UnsupportedEncodingException {

        return java.net.URLEncoder.encode(str,encoding);
    }

	/**
		* converts a String from the local encoding to UTF-8
     *
     * @return a String
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
						  *   error).
     */
	public static String convertFromLocalEncoding(String theString) {
		String resultString = theString;
		if (!fr.imag.clips.papillon.business.locales.Languages.getDefaultEncoding().equals("UTF-8")) {
			try {
				resultString = Utility.convertFromEncoding(theString,fr.imag.clips.papillon.business.locales.Languages.getDefaultEncoding());
			}
			catch (java.io.UnsupportedEncodingException e) {
				PapillonLogger.writeDebugMsg("Error: UnsupportedEncodingException:"+e.toString());
			}
		}
		return resultString;
	}				
	
    //
    public static boolean intersect(Collection col1, Collection col2) {
        Iterator it1 = col1.iterator();
        while(it1.hasNext()) {
            boolean found = col2.contains(it1.next());
            if (found) return true;
        }
        return false;
    }
    
	//public static String[] ArrayIntersection(String[] myArray1, String[] myArray2) {
	//	return (String[]) ArrayIntersection((java.util.Collection)java.util.Arrays.asList(myArray1), (java.util.Collection) java.util.Arrays.asList(myArray2));
	//}
	public static Collection ArrayIntersection(Collection myArray1, Collection myArray2) {
        /*java.util.Vector myRes = new java.util.Vector();
        if (myArray1 != null && myArray1.length>0) {
            for (int i=0; i<myArray1.length;i++) {
                String tempString = myArray1[i];
                if (IsInArray(tempString,myArray2)) {
                    myRes.add(tempString);
                }
            }
        }
        return (String[]) myRes.toArray(new String[0]);*/
        
        //
        ArrayList newArray = new ArrayList(myArray1);
        newArray.retainAll(myArray2);
        
        //
        return newArray;
    }
	
	public static String[] ArrayUnion(String[] myArray1, String[] myArray2) {
		//java.util.ArrayList list = java.util.Arrays.asList(myArray1);
		java.util.HashSet set = new java.util.HashSet();
		if (myArray1 != null && myArray1.length>0) {
			set.addAll(java.util.Arrays.asList(myArray1));
		}
 		if (myArray2 != null && myArray2.length>0) {
			set.addAll(java.util.Arrays.asList(myArray2));
		}
		//list.clear();
		//list.addAll(set);
	
		return (String[]) set.toArray(new String[0]);
	}
	
	public static Collection ArrayUnion(Collection myArray1, Collection myArray2) {
        /*java.util.Vector myRes = new java.util.Vector();
        if (myArray1 != null && myArray1.length>0) {
            for (int i=0; i<myArray1.length;i++) {
                String tempString = myArray1[i];
                if (IsInArray(tempString,myArray2)) {
                    myRes.add(tempString);
                }
            }
        }
        return (String[]) myRes.toArray(new String[0]);*/
        
        //
		if (myArray1 != null && myArray1.size()>0) {
			if (myArray2 != null && myArray2.size()>0) {
				myArray1.addAll(myArray2);
			}
		}
		else {
			if (myArray2 != null && myArray2.size()>0) {
				myArray1 = myArray2;
			}
		}
			
		if (myArray1 != null) {
			myArray1 = (Collection) new java.util.HashSet(myArray1);
		}
        //
        return myArray1;
    }
   
   public static void removeElement(Element elem) {
		if (elem != null) {
			Node myParent = elem.getParentNode();
			if (myParent != null)
				myParent.removeChild(elem);
		}
	}
	
	
	/**
     * getLocalTagName
     *
     * @return a String
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	public static String getLocalTagName(String tagname) {
		if (tagname != null) {
			if (tagname.indexOf(":") >0) {
				tagname = tagname.substring(tagname.indexOf(":")+1);
			}
		}
		return tagname;
	}

	/**
     * getPrefix
     *
     * @return a String
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	public static String getPrefix(String tagname) {
		String prefix = "";
		if (tagname != null) {
			if (tagname.indexOf(":") >0) {
				prefix = tagname.substring(0,tagname.indexOf(":"));
			}
		}
		return prefix;
	}

		/**
     * Tests if a String array is in another String array
     *
     * @return a String
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	
	 
	public static byte[] serializeHashtable(java.util.Hashtable myTable) throws fr.imag.clips.papillon.business.PapillonBusinessException {
		byte[] resultArray = null;
		try {
			ByteArrayOutputStream myByteArrayOutputStream = new ByteArrayOutputStream();
			ObjectOutputStream myObjectOutputStream = new ObjectOutputStream(myByteArrayOutputStream);
			myObjectOutputStream.writeObject (myTable);
			resultArray = myByteArrayOutputStream.toByteArray();
			myByteArrayOutputStream.flush();
		}
		catch(Exception ex) {
            throw new fr.imag.clips.papillon.business.PapillonBusinessException("Error in Utility.deSerializeHashtable", ex);
        }
		return resultArray;
	}
	
	/**
     * deSerialize a Hashtable from a byte array
     *
     * @return a hashtable
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	public static java.util.Hashtable deSerializeHashtable(byte[] myByteArray) throws fr.imag.clips.papillon.business.PapillonBusinessException {
		java.util.Hashtable myRes = null;
		if (myByteArray != null) {
		try {
			ObjectInputStream myOIStream = new ObjectInputStream(new ByteArrayInputStream(myByteArray));
			myRes = (java.util.Hashtable) myOIStream.readObject();
		}
		catch(Exception ex) {
            throw new fr.imag.clips.papillon.business.PapillonBusinessException("Error in Utility.deSerializeHashtable", ex);
        }
		}
		return myRes;
	}

	/**
     * Serialize a DOM Document into a byte array
     *
     * @return a byte array
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	public static byte[] serializeDocument(org.w3c.dom.Document myDoc) throws fr.imag.clips.papillon.business.PapillonBusinessException {
		byte[] resultArray = null;
		try {
			ByteArrayOutputStream myByteArrayOutputStream = new ByteArrayOutputStream();
			ObjectOutputStream myObjectOutputStream = new ObjectOutputStream(myByteArrayOutputStream);
			myObjectOutputStream.writeObject (myDoc);
			resultArray = myByteArrayOutputStream.toByteArray();
			myByteArrayOutputStream.flush();
		}
		catch(Exception ex) {
            throw new fr.imag.clips.papillon.business.PapillonBusinessException("Error in Utility.deSerializeHashtable", ex);
        }
		return resultArray;
	}
	
	/**
     * deSerialize a DOM Document  from a byte array
     *
     * @return a DOM Document 
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	public static org.w3c.dom.Document deSerializeDocument(byte[] myByteArray) throws fr.imag.clips.papillon.business.PapillonBusinessException {
		org.w3c.dom.Document myRes = null;
		try {
			ObjectInputStream myOIStream = new ObjectInputStream(new ByteArrayInputStream(myByteArray));
			myRes = (org.w3c.dom.Document) myOIStream.readObject();
		}
		catch(Exception ex) {
            throw new fr.imag.clips.papillon.business.PapillonBusinessException("Error in Utility.deSerializeHashtable", ex);
        }
		return myRes;
	}
	/**
     * returns a secure random generator
	 *
	 * @return random generator 
	 * @exception PapillonBusinessException if an error occurs
	 *   retrieving data (usually due to an underlying data layer
	 *   error).
	 */
	public static java.util.Random getRandomGenerator() {
		if (theRandomGenerator == null) {
			theRandomGenerator = initializeRandomGenerator();
		}
		return theRandomGenerator;
	}
	/**
     * initialize a random generator
	 *
	 * @return a random generator 
	 * @exception PapillonBusinessException if an error occurs
	 *   retrieving data (usually due to an underlying data layer
	 *   error).
	 */
	protected static java.util.Random initializeRandomGenerator() {
		java.util.Random rand = null;
		try {
			rand = java.security.SecureRandom.getInstance("SHA1PRNG");
		}
		catch (java.security.NoSuchAlgorithmException nsae) {
			PapillonLogger.writeDebugMsg("Utility Error: NoSuchAlgorithmException: ");
			nsae.printStackTrace();
		}
		return rand;
	}
	
	public static String getStars(String[] stringArray) {
		String res = "";
		if (stringArray !=null &&  stringArray.length>0) {
			for (int i=0; i<stringArray.length; i++) {
				String theString = stringArray[i];
				java.util.regex.Matcher starMatcher = starPattern.matcher(theString);
				if (starMatcher.matches() && theString.length() > res.length()) {
					res = theString;
				}
			}
		}
		return res;
	}	

	public static String serializeStringArray(String[] StringArray, String sepString) {
		String stringRes = sepString;
		if (null != StringArray && StringArray.length >0) {
			for (int i=0; i< StringArray.length; i++) {
				if (StringArray[i]!=null) {
					stringRes += StringArray[i] + sepString;
				}
			}
		}
		return stringRes;
	}

	public static String[] deserializeStringIntoArray(String theString, String sepString) {
		String[] resArray = new String[0];
		if (null != theString && !theString.equals("")){
			// delete the first separator in order to avoid an empty group
			if (theString.indexOf(sepString) ==0) {
				theString = theString.substring(sepString.length());
			}
			resArray = theString.split(sepString);
		}
		return resArray;
	}

}
