/*
 * papillon
 *
 * LanguageFactory.java
 *
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.3  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * Language Name Factory.
 */

package fr.imag.clips.papillon.business.locales;

import java.util.ArrayList;
import java.util.Locale;
import java.util.ResourceBundle;

import fr.imag.clips.papillon.business.PapillonLogger;

public class LanguageFactory {

		public static final ResourceBundle ISO6391to2TBundle = ResourceBundle.getBundle("fr.imag.clips.papillon.properties.ISO-639-1to2T", Locale.getDefault());
		public static final ResourceBundle ISO6392to1Bundle = ResourceBundle.getBundle("fr.imag.clips.papillon.properties.ISO-639-2to1", Locale.getDefault());

    /**
			* input: a String representing the ISO-639-1 two letter code of a language
		 * output: a String representing the ISO-639/T 3 letter code representing the language
		 * or a String with max 3 characters if the code is not found.
		 *
		 * To add a new language code, please edit the fr.imag.clips.papillon.properties.ISO-639-1to2T.properties
		 *    and fr.imag.clips.papillon.properties.ISO-639-2to1.properties file
		 *
		 * @return ArrayList containing the accepted languages.
		 */
    public static String getISO3LanguageCode(String lang) {
			String res = lang;
			if (lang.length() > 3) {
				lang=lang.substring(0,3);
			}
			if (lang.length() >= 2) {
				try {
					res = ISO6391to2TBundle.getString(lang.substring(0,2));
				}
				catch (Exception e) {
					/* code was not found */
					res=lang;
				}
			}
			return res;
    }

    /**
			* input: a String representing the ISO-639-2 3 letter code of a language
		 * output: a String representing the ISO-639/T 2 letter code representing the language
		 * or a String with max 2 characters if the code is not found.
		 *
		 * To add a new language code, please edit the fr.imag.clips.papillon.properties.ISO-639-2to1.properties file
		 *    and fr.imag.clips.papillon.properties.ISO-639-1to2T.properties file
		 *
		 * @return ArrayList containing the accepted languages.
		 */
    public static String getISO2LanguageCode(String lang) {
			String res = lang;
			if (lang.length() >= 3) {
				lang=lang.substring(0,3);
				try {
					res = ISO6392to1Bundle.getString(lang.substring(0,3));
				}
				catch (Exception e) {
					/* code was not found */
					res=lang.substring(0,2);
				}
			}
			return res;
    }

    /**
    * Returns a List of String representing the languages accepted by the user, in
    * the user's preference order.
    *
    * @return ArrayList containing the accepted languages.
    */
    public static ArrayList decodeAcceptLanguages(String languages) {
        ArrayList result = new ArrayList(); 

        if (null == languages) return result;
        String[] languagesArray = languages.split(",");

        if (null == languagesArray) return result;

        for (int i=0; i<languagesArray.length; i++) {
            String lang =  languagesArray[i];
            String[] langSplit = lang.split(";");
            lang = (null == langSplit || 0 == langSplit.length) ? "" : langSplit[0];

            result.add(getISO3LanguageCode(lang));
        }
        return result;
    }        
}
