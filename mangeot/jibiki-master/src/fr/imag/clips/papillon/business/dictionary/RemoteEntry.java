/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.4  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.3  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.2  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.1.1.1.2.1  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * Revision 1.5  2004/02/10 05:27:13  mangeot
 * The version UIGEN_V2 has been merged with the trunk by MM
 * Be careful because the Volumes and contributions database tables have been modified.
 * You have to drop and rebuild them unless you modify them by hands.
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.dictionary;

import fr.imag.clips.papillon.business.PapillonBusinessException;

/**
 * Represents a Mailing list Dictionary. 
 */
public class RemoteEntry implements IAnswer {

    protected Dictionary theDictionary;
    protected Volume theVolume;
    protected String xmlCode;
    protected String theHeadword;
    protected String theId;

    public RemoteEntry() throws PapillonBusinessException {
    }

    public boolean isEmpty() {
        return (this.xmlCode==null || this.xmlCode.equals(""));
    }

    public String getXmlCode() throws PapillonBusinessException {
        return xmlCode;
    }

    public void setXmlCode(String code)
        throws PapillonBusinessException {
            this.xmlCode = code;
    }

    public Dictionary getDictionary() throws PapillonBusinessException {
        return this.theDictionary;
    }

    public void setDictionary(Dictionary dict) {
        this.theDictionary = dict;
    }

    public Volume getVolume() throws PapillonBusinessException {
        return this.theVolume;
    }

    public void setVolume(Volume volume) {
        this.theVolume = volume;
    }

    public String getDictionaryName() throws PapillonBusinessException {
        return this.theDictionary.getName();
    }

    public String getDictionaryFullName() throws PapillonBusinessException {
        return this.theDictionary.getFullName();
    }

    public String getVolumeName() throws PapillonBusinessException {
        return this.theVolume.getName();
    }

    public String getSourceLanguage() throws PapillonBusinessException {
        return this.theVolume.getSourceLanguage();
    }

    public String getHandle() throws PapillonBusinessException {
        return this.theId;
    }

// compatibility with IAnswer interface
    public String getId() throws PapillonBusinessException {
        return this.theId;
    }

	public void setId(String newId) throws PapillonBusinessException {
        this.theId = newId;
    }

	public void setId() throws PapillonBusinessException {
        this.theId = "";
    }

    public org.w3c.dom.Document getDom() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}

    public void setDom(org.w3c.dom.Document myDoc) throws fr.imag.clips.papillon.business.PapillonBusinessException {
		;
	}

//    public org.w3c.dom.Document getHtmlDom() throws fr.imag.clips.papillon.business.PapillonBusinessException {
//		return null;
//	}
//
//    public void setHtmlDom(org.w3c.dom.Document myDoc) throws fr.imag.clips.papillon.business.PapillonBusinessException {
//		;
//	}
			
    public void setHeadword(String headword) {
        this.theHeadword = headword;
    }

    public String getHeadword() {
			return this.theHeadword;
    }

    public int getType() {
        return IAnswer.RemoteEntry;
    }
	
	public boolean save() {
		return true;
	}
	public void delete() {
		this.xmlCode =null;
	}
}
