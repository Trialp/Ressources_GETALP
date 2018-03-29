/*
 * papillon
 *
 * MultilingualWmlTemplateFactory.java
 *
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *
 *-----------------------------------------------
 * Multilingual WML Template Factory.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import java.util.ArrayList;

import fr.imag.clips.papillon.business.locales.LanguageFactory;

import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;

public class MultilingualWmlTemplateFactory {
    private final static String PACKAGE = "fr.imag.clips.papillon.presentation.wml.";
    private final static String ORIG_PACKAGE = PACKAGE + "orig.";

    private final static String DEFAULT_LANGUAGE = "eng";
    
    /**
     * Return an HTML Template object that is an localized instance of the
     * requested class. The language of the returned object depends on
     * the HTTPRequest (which contains the user's preferred languages).
     *
     * @return The localized HTML Template Object
     */
    public static java.lang.Object createTemplate(String wmlClass,
                                                  HttpPresentationComms comms,
                                                  PapillonSessionData sessiondata)
        throws HttpPresentationException
    {
        ArrayList languages = sessiondata.getUserAcceptLanguages();

        int i = 0;
        int languagesSize = 0;
        java.lang.Object template = null;

        if (languages != null) {
            languagesSize = languages.size();
        }
        while (i != languagesSize &&
               (template = getTemplateForLanguage(wmlClass, (String)languages.get(i), comms)) == null) {
            i++;
        }
        if (i == languagesSize) {
            try {
                template = comms.xmlcFactory.create(Class.forName(ORIG_PACKAGE + wmlClass));
            } catch (java.lang.ClassNotFoundException e) {
                throw new HttpPresentationException("ERREUR:", e);
            }
        }
        return template;

    }

    public static java.lang.Object getTemplateForLanguage(String wmlClass, String lang, HttpPresentationComms comms) {
        String className = PACKAGE + lang + "." + wmlClass;
        java.lang.Object template = null;

        try {
            template = comms.xmlcFactory.create(Class.forName(className));
        } catch (java.lang.ClassNotFoundException e) {
        }
        return template;
    }

    public static ArrayList getAcceptLanguages (HttpPresentationRequest req)  {
        ArrayList defaultAccept = new ArrayList();
        defaultAccept.add(new String(DEFAULT_LANGUAGE));

        ArrayList res = null;

        try {
            String languages = null;
            languages = req.getHeader("Accept-Language");
            res = LanguageFactory.decodeAcceptLanguages(languages);
        } catch (com.lutris.appserver.server.httpPresentation.HttpPresentationException e) {
            res = defaultAccept;
        }
        return (null == res || 0 == res.size()) ? defaultAccept : res;
    }
    
}
