/*
 *  InformationSectionFactory.java
 *  papillon
 *
 *  Created by Gilles Sérasset on Wed Jan 30 2002.
 *  Copyright (c) 2002 GETA_CLIPS_IMAG. All rights reserved.
 *
 *---------------------------------------------------------
 * $Id$
 *---------------------------------------------------------
 * $Log$
 * Revision 1.2  2006/08/10 18:30:26  mangeot
 * *** empty log message ***
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * Revision 1.2  2003/09/03 10:15:45  mangeot
 * reorganizing imports and using eclipse
 *
 * Revision 1.1.1.1  2002/10/28 16:49:14  serasset
 * Creation of the papillon CVS repository for enhydra 5.0
 *
 * Revision 1.3  2002/08/01 12:40:01  tache
 * Cleaned up a lot of code in import classes. Moved HTML parsing methods from
 * ImportHTMLFile to new class HTMLParser to reuse them in ImportArchive.
 * Moved XMLTitleParser to fr.imag.clips.papillon.presentation.
 *
 * Revision 1.2  2002/06/10 11:07:57  mangeot
 * I began to implement the localization but very rapidly ...
 *
 * Revision 1.1  2002/02/06 15:03:03  serasset
 * Consultation and Administration of the information files now correctly use the Sections.
 * We do not display the files directly anymore, but display a document. This allows the display of
 * images as embedded docs in the default layout.
 *
 *---------------------------------------------------------
 */
package fr.imag.clips.papillon.business.informationfile;

import java.util.ArrayList;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;

import com.lutris.appserver.server.Enhydra;
import com.lutris.util.ConfigException;
 

/* Beware: the strings retrieved from the enhydra config files are encoded in local encoding, 
they must be converted before use */
public class InformationSections {
    public static String SECTION_GROUP = "Papillon.Informations.sections";
    public static String DEFAULT_GROUP = ".default";
    public static String DEFAULT_LANGUAGE = ".eng";
    public String[] defaultLanguageSections = null;
    public String[] sections = null;
    
    // constructor
    public InformationSections() throws PapillonBusinessException {
        try {
            sections =  Enhydra.getApplication().getConfig().getStrings(SECTION_GROUP + DEFAULT_GROUP);
        } catch (ConfigException e) {
            throw new PapillonBusinessException("InformationFile found no section list. Check the application config file.", e);
        }
        try {
            defaultLanguageSections =  Enhydra.getApplication().getConfig().getStrings(SECTION_GROUP + DEFAULT_LANGUAGE);
        } catch (ConfigException e) {
             PapillonLogger.writeDebugMsg("InformationFile found no section list for default language. " +
                                          "Check the application config file.");
            defaultLanguageSections = sections;
        }
    }
    
    /**
     * return the list of sections as known by the database
     */
    public String[] getInformationSectionArray() {
        return sections;
    }
    

    /**
     * returns the list of sections as they will be presented to the user (in the user's preferred language)
     */
    public String[] getAcceptedLocalizedSectionArray(ArrayList prefLang) {
	boolean langFound = false;
	int i = 0;
	String[] res = null;
	while (i < prefLang.size() && !langFound) {
	    String[] sections = null;
	    try {
		sections = Enhydra.getApplication().getConfig().getStrings(SECTION_GROUP + "." + prefLang.get(i), 
										defaultLanguageSections);
		if (sections[0]!="DEV") {
		    // the array is not the dafault array so prefLang.get(i) is the lang to use
		    res = sections;
		    langFound = true;
		}
	    
	    } catch (ConfigException e) {
		PapillonLogger.writeDebugMsg("The section list for language : " + prefLang.get(i) + "is incorrect.");
	    }
		
	    
	    i++;
	}
	if (!langFound) {
	    PapillonLogger.writeDebugMsg("InformationFile found no section list for accepted languages");
	    res = defaultLanguageSections;
	}
	return res;
    }

    /**
     * returns the list of sections as they will be presented to the user (in language lang)
     */
    public String[] getLocalizedSectionArray(String lang) {
        try {
            return Enhydra.getApplication().getConfig().getStrings(SECTION_GROUP + "." + lang);
        } catch (ConfigException e) {
             PapillonLogger.writeDebugMsg("InformationFile found no section list for language : " + lang);
             return defaultLanguageSections;
        }
    }
    
    /**
     * returns the list of sections as they will be presented to the user (in the default language: english)
     */
    public String[] getLocalizedSectionArray() {
        return defaultLanguageSections;
    }
    
    /**
     * returns the name of the section corresponding to "section" (in language lang).
     */
    public String getLocalizedSectionName(String section, String lang) {
        String[] locsections = getLocalizedSectionArray(lang);
        int l = sections.length;
        int pos = 0;
        while (pos != l && section.equals(sections[pos])) pos++;
        if (pos != l) {
            return locsections[pos];
        } else {
            return "";
        }
        
    }
    
}
