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
 * Revision 1.1.1.1.2.3  2005/01/28 19:45:55  mangeot
 * First version that runs basically.
 * Should compile after an ant clean.
 * XPath loading and virtual volumes for terminological lexicons are OK.
 * Bugs remain, needs more testings like the editor for example.
 *
 * Revision 1.1.1.1.2.2  2005/01/27 19:29:21  mangeot
 * Implemented the HtmlDom cache, it increases speed drastically.
 * Still does not compile after an ant clean
 *
 * Revision 1.1.1.1.2.1  2005/01/25 13:54:54  mangeot
 * changed the volume volumeEntry and index objects. Does not compile but need a backup...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * Revision 1.7  2004/10/28 10:38:11  mangeot
 * Fixed some bugs that affected the dictd server
 * Modified some methods in order to display a text entry in the dictd server
 *
 * Revision 1.6  2004/02/10 05:27:12  mangeot
 * The version UIGEN_V2 has been merged with the trunk by MM
 * Be careful because the Volumes and contributions database tables have been modified.
 * You have to drop and rebuild them unless you modify them by hands.
 *
 *-----------------------------------------------
 */

package fr.imag.clips.papillon.business.dictionary;

/**
 * Insert the type's description here.
 * Creation date: (10.08.01 20:55:06)
 * @author: Administrator
 */
public interface IAnswer {
/**
 * Insert the method's description here.
 * Creation date: (10.08.01 20:55:40)
 * @return org.dict.IDatabase
 */
    public static final int Error = -1;
    public static final int Default = 1;
    public static final int LocalEntry = 1;
    public static final int RemoteEntry = 2;
    public static final int Contribution = 3;
    public static final int LocalAxie = 4;

    int getType();

    boolean isEmpty();
    
    Dictionary getDictionary() throws fr.imag.clips.papillon.business.PapillonBusinessException;

    String getDictionaryName() throws fr.imag.clips.papillon.business.PapillonBusinessException;
    
    String getDictionaryFullName() throws fr.imag.clips.papillon.business.PapillonBusinessException;
    
    String getVolumeName() throws fr.imag.clips.papillon.business.PapillonBusinessException;

    Volume getVolume() throws fr.imag.clips.papillon.business.PapillonBusinessException;

    String getSourceLanguage() throws fr.imag.clips.papillon.business.PapillonBusinessException;
		    
    String getXmlCode() throws fr.imag.clips.papillon.business.PapillonBusinessException;
    
    void setXmlCode(String code) throws fr.imag.clips.papillon.business.PapillonBusinessException;

    String getId() throws fr.imag.clips.papillon.business.PapillonBusinessException;

    void setId() throws fr.imag.clips.papillon.business.PapillonBusinessException;

    String getHeadword() throws fr.imag.clips.papillon.business.PapillonBusinessException;

    String getHandle() throws fr.imag.clips.papillon.business.PapillonBusinessException;

    org.w3c.dom.Document getDom() throws fr.imag.clips.papillon.business.PapillonBusinessException;

    void setDom(org.w3c.dom.Document myDoc) throws fr.imag.clips.papillon.business.PapillonBusinessException;

//    org.w3c.dom.Document getHtmlDom() throws fr.imag.clips.papillon.business.PapillonBusinessException;
//
//    void setHtmlDom(org.w3c.dom.Document myDoc) throws fr.imag.clips.papillon.business.PapillonBusinessException;
		
    void delete() throws fr.imag.clips.papillon.business.PapillonBusinessException;

    boolean save() throws fr.imag.clips.papillon.business.PapillonBusinessException;

}
