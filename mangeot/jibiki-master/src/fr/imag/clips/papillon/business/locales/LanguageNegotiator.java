package fr.imag.clips.papillon.business.locales;

import java.util.ArrayList;
import java.util.Vector;


/**
 * This class is for language negotiation. Used to choose the user's preferred language 
 * between the available ones.
 */
public class LanguageNegotiator {


    //Constructor
    public LanguageNegotiator() {
	//Nothing...
    }


    /**
     * Finds the language that must be used to display a document's title
     *
     * @param acceptedLangs the languages accepted by the user (including browser languages)
     * @param availLangs a Vector of languages available for the document
     * @param default the default language if no available language is accepted
     * @return the language in wich the document's title must be displayed
     */
    public static String negotiateLanguage(ArrayList acceptedLangs, Vector availLangs, String defLang) {

	boolean langFound = false;
	String res = null;

	int i = 0;
	while (i < acceptedLangs.size() && !langFound) {
	    if (availLangs.contains((String)acceptedLangs.get(i))) {
		langFound = true;
		res = (String)acceptedLangs.get(i);
	    }
	    i++;
	}
    
	// available languages don't match with accepted language: return default language
	if (!langFound) {
	    res = defLang;
	}
	return res;	  
    }	


}
