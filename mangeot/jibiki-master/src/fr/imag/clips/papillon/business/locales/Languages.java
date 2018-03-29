/*
 *  LanguageListsFactory.java
 *  papillon
 *
 *  Created by Mathieu Mangeot on Sat Mar 09 2002.
 *  Copyright (c) 2002 GETA_CLIPS_IMAG & NII. All rights reserved.
 *
 *---------------------------------------------------------
 * $Id$
 *---------------------------------------------------------
 * $Log$
 * Revision 1.6  2007/04/05 12:55:54  serasset
 * Added a DBLayer Version management with an auto-update of db layer.
 *
 * Revision 1.5  2006/08/10 19:21:45  mangeot
 * *** empty log message ***
 *
 * Revision 1.4  2006/08/10 19:12:37  mangeot
 * Added getDefaultEncoding
 *
 * Revision 1.3  2005/01/23 17:21:16  mangeot
 * *** empty log message ***
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *---------------------------------------------------------
 */

package fr.imag.clips.papillon.business.locales;

import java.util.Hashtable;

import java.util.Locale;
import java.util.ResourceBundle;

import com.lutris.appserver.server.Enhydra;
import com.lutris.util.ConfigException;


/* The java bundles files work with ISO-639-1 two letters codes,
* but Papillon plateform uses ISO-639-2T three letters codes.
* Thus, we do not use the locales dynamically. Instead, we load statically
* all the available language lists.
*
* If you want to add a language list, you have to write a Languages.properties
* file and to modify the following code. */

public class Languages {

	/* Constants */
	protected final static String PropertiesPath = "fr.imag.clips.papillon.properties.";
	protected final static String LanguagesFile = "Languages";
	protected final static String LanguagesASCIIFile = "LanguagesASCII";
	protected static Hashtable LanguagesResbundles = new Hashtable();
	protected static Hashtable LanguagesASCIIResbundles = new Hashtable();
	protected final static Locale defaultLocale = Locale.getDefault();
	protected final static java.io.InputStreamReader myInputStreamReader = new java.io.InputStreamReader(System.in);


	protected static final String[] LanguagesArray = {
		"deu","eng","esp","est","fra","hun","ita",
		"jpn","kor","lao","msa","tha","vie",
		"zho"};

	public static String localizeName(String langLoc, String langName) {
		String res = langName;
		try {
			/* lookup the cache */
			ResourceBundle myBundle = (ResourceBundle) LanguagesResbundles.get(langLoc);
			if (myBundle==null) {
				/* if not in the cache, create */
				try {
                    myBundle = ResourceBundle.getBundle(PropertiesPath + LanguagesFile + "-" + langLoc, defaultLocale);
				}
				catch (Exception e) {
					/* language bundle was not found */
					myBundle=ResourceBundle.getBundle(PropertiesPath + LanguagesFile, defaultLocale);
				}				
				LanguagesResbundles.put(langLoc,myBundle);
			}
			res=myBundle.getString(langName);
		}
		catch (Exception e) {
			/* language name was not found */
			res=langName;
		}
		return res;
	}

	public static String localizeLabel(String langLoc, String langName) {
		String res = langName;
		try {
			/* lookup the cache */
			ResourceBundle myBundle = (ResourceBundle) LanguagesASCIIResbundles.get(langLoc);
			if (myBundle==null) {
				/* if not in the cache, create */
				try {
					myBundle = ResourceBundle.getBundle(PropertiesPath + LanguagesASCIIFile + "-" + langLoc, defaultLocale);
				}
				catch (Exception e) {
					/* language bundle was not found */
					myBundle=ResourceBundle.getBundle(PropertiesPath + LanguagesASCIIFile, defaultLocale);
				}
				LanguagesASCIIResbundles.put(langLoc,myBundle);
			}
			res=myBundle.getString(langName);
		}
		catch (Exception e) {
			/* language name was not found */
			res=langName;
		}
		return res;
	}

	public static String[] getArray() {
		return LanguagesArray;
	}

	public static boolean resetCache() {
		LanguagesResbundles = new Hashtable();
		LanguagesASCIIResbundles = new Hashtable();
		return true;
	}
	
	public static String getDefaultEncoding () {
		String systemEncoding = myInputStreamReader.getEncoding();
		if (systemEncoding.equals("ASCII")) {
			systemEncoding = "ISO-8859-1";
		}
		return systemEncoding;
	}
}
