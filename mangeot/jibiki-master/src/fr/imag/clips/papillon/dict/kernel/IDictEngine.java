package fr.imag.clips.papillon.dict.kernel;

import fr.imag.clips.papillon.business.dictionary.IAnswer;
/**
* Insert the type's description here.
* Creation date: (7/31/01 3:35:05 PM)
* @author: Administrator
*/
public interface IDictEngine {

	public static final String TEXT_view = "TEXT";
	public static final String XML_view = "XML";
	public static final String HTML_view = "HTML";

    /**
        * Insert the method's description here.
        * Creation date: (7/31/01 3:35:55 PM)
        * @return IDefinition[]
        * @param db java.lang.String
        * @param word java.lang.String
        */
    IAnswer[] lookup(String dict, String lang, String word, int strategy);
    /**
        * Insert the method's description here.
        * Creation date: (7/31/01 3:35:55 PM)
        * @return IDefinition[]
        * @param db java.lang.String
        * @param word java.lang.String
        */
    String[] getDictionariesInfo();
    /**
        * Register the application
        * Creation date: (7/31/01 3:35:55 PM)
        * 
        */
    String getDictionaryInfo(String dict);
    /**
        * Register the application
        * Creation date: (7/31/01 3:35:55 PM)
        * 
        */

    String[] getDatabaseInfo();
    /**
        * Register the application
        * Creation date: (7/31/01 3:35:55 PM)
        * 
        */
    void initialize();
    /**
        * unRegister the application
        * Creation date: (7/31/01 3:35:55 PM)
        * 
        */
    void finalize();
    /**
        * authenticate the user
        * Creation date: (7/31/01 3:35:55 PM)
        * 
        */
    String authenticate(String login, String password);
	
    /**
        * creates a new dictEngine
        * Creation date: (10/27/04 3:35:55 PM)
        * 
        */
    IDictEngine copy();
    /**
        * set mime header 
        * Creation date: (10/27/04 3:35:55 PM)
        * 
        */
    void setMime(boolean newMime);
    /**
        * get mime header 
        * Creation date: (10/27/04 3:35:55 PM)
        * 
        */
    boolean IsMime();
    /**
        * set view type
        * Creation date: (10/27/04 3:35:55 PM)
        * 
        */
    void setView(String newView);
    /**
        * process view
        * Creation date: (10/27/04 3:35:55 PM)
        * 
        */
    String processView(IAnswer answer);
}
