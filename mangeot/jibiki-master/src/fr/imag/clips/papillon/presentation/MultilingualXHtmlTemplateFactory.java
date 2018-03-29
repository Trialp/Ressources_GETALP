/*
 * papillon
 *
 * MultilingualXHtmlTemplateFactory.java
 *
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.6  2007/01/15 17:12:18  serasset
 * Several notes added, suppressed the HTMLDOM_CACHE stuff.
 *
 * Revision 1.5  2006/08/10 21:28:57  mangeot
 * error treatment
 *
 * Revision 1.4  2005/06/16 10:42:15  mangeot
 * Added and modified files for the GDEF project
 *
 * Revision 1.3  2005/06/15 16:48:28  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.2  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.1.6.1  2005/05/27 11:53:34  mangeot
 * *** empty log message ***
 *
 * Revision 1.1  2004/12/24 08:57:44  serasset
 * Premiere version de l'interface avec fond papillon et transparence.
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *-----------------------------------------------
 * Multilingual XHTML Template Factory.
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.Enhydra;
import fr.imag.clips.papillon.Papillon;

import java.util.ArrayList;

import fr.imag.clips.papillon.business.locales.LanguageFactory;

import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;

public class MultilingualXHtmlTemplateFactory {
    private final static String PACKAGE = "fr.imag.clips.papillon.presentation.xhtml";
    private final static String ORIG_PACKAGE = PACKAGE + "orig.";
	
    private final static String DEFAULT_LANGUAGE = "eng";
	private static java.lang.ClassNotFoundException myClassNotFoundException = null;
    
    /**
		* Return an HTML Template object that is a localized instance of the
     * requested class. The language of the returned object depends on
     * the HTTPRequest (which contains the user's preferred languages).
     *
     * @return The localized HTML Template Object
     //* @deprecated Use createTemplate with package specifier.
     */
    public static java.lang.Object createTemplate(String xhtmlClass,
                                                  HttpPresentationComms comms,
                                                  PapillonSessionData sessiondata)
        throws HttpPresentationException {
			java.lang.Object resObject = null;
			String priorityPackage = ((Papillon)Enhydra.getApplication()).getPriorityPackage();
			if (priorityPackage != null && !priorityPackage.equals("")) {
				resObject = createTemplate(priorityPackage,xhtmlClass,comms,sessiondata);
			}
			if (resObject==null) {
				resObject = createTemplate(PACKAGE,xhtmlClass,comms,sessiondata);
			}
			if (resObject==null) {
				throw new HttpPresentationException("ERREUR:", myClassNotFoundException);
			}
			
			return resObject;
		}
    
    /**
        * Return an HTML Template object that is an localized instance of the
     * requested class. The language of the returned object depends on
     * the HTTPRequest (which contains the user's preferred languages).
     *
     * @return The localized HTML Template Object
     */
    protected static java.lang.Object createTemplate(String xhtmlPackage,
													 String xhtmlClass,
													 HttpPresentationComms comms,
													 PapillonSessionData sessiondata)
        throws HttpPresentationException
    {
        
		int i = 0;
		java.lang.Object template = null;
        ArrayList languages = sessiondata.getUserAcceptLanguages();
		if (languages!=null &&languages.size()>0) {
			while (i != languages.size() &&
				   (template = getTemplateForLanguage(xhtmlPackage, xhtmlClass, (String)languages.get(i), comms)) == null) {
				i++;
			}
		}
		boolean tmpl = (template != null);
		if (languages == null || languages.size()==0 || i == languages.size()) {
			try {
				template = comms.xmlcFactory.create(Class.forName(xhtmlPackage + ".orig." + xhtmlClass));
			} catch (java.lang.ClassNotFoundException e) {
				myClassNotFoundException = e;
			}
		}
		return template;
	}
	
	public static java.lang.Object getTemplateForLanguage(String xhtmlPackage, String htmlClass, String lang, HttpPresentationComms comms) {
		String className = xhtmlPackage + "." + lang + "." + htmlClass;
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
