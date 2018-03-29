package fr.imag.clips.papillon.dict.kernel;

import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;

import fr.imag.clips.papillon.business.dictionary.AvailableLanguages;
import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.ErrorAnswer;
import fr.imag.clips.papillon.business.dictionary.IAnswer;
import fr.imag.clips.papillon.business.dictionary.IQuery;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.transformation.XslTransformation;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.UsersFactory;

import com.lutris.appserver.server.Application;
import com.lutris.appserver.server.Enhydra;

/**
* Insert the type's description here.
* Creation date: (29.07.01 12:47:32)
* @author: Administrator
*/
public class DictEngine implements IDictEngine {

    protected Application enhydraApp;
    protected User theUser = null;
	protected boolean mime = false;
	protected String view = "";
	protected static final IAnswer errorAnswer = new ErrorAnswer();
	protected static final IAnswer[] errorAnswers = {errorAnswer};
	protected static final int DEFAULT_LIMIT = 100;

    /**
        * DatabaseFactory constructor comment.
        */
    public DictEngine(Application myApplication) {
        super();
        enhydraApp = myApplication;
    }
    public void initialize() {
        Enhydra.register(enhydraApp);
    }

    public void finalize() {
        Enhydra.unRegister();
    }
	
	public void setMime(boolean newMime) {
		mime = newMime;
	}
	public boolean IsMime() {
		return mime;
	}

	public void setView(String newView) {
		view = newView;
	}
	public String processView (IAnswer entry) {
		String res = null;
		try {
		// disabled
			// is there a way to obtain a dom result which is a text string?
			if (view!=null && !view.equals("")) {
				if (view.equals(IDictEngine.TEXT_view)) {
					res = XslTransformation.applyXslSheetsForText(entry);
				}
				else if (view.equals(IDictEngine.HTML_view)) {
					res = XslTransformation.applyXslSheetsAndSerialize(entry);
				}
				else {
					res = entry.getXmlCode();
				}
			}
			else { 
				res = entry.getXmlCode();
			}
		}
		catch (Exception e) {
            throw new RuntimeException("Error in processView");
        }
		return res;
	}
	
    public String[] getDictionariesInfo() {
        String[] Infos = null;
        try {
            Collection Dicts = DictionariesFactory.getDictionariesArray();
            Infos = new String[Dicts.size()];
            int i = 0;
            for (Iterator iter = Dicts.iterator(); iter.hasNext();) {
                Dictionary dict = (Dictionary)iter.next();
                Infos[i] = dict.getName() + " \"" + dict.getFullName() + "\"";
                i++;
            }
        }
        catch (Exception e) {
            throw new RuntimeException("Error in getDictionariesInfo");
        }
        return Infos;
    }

    public String[] getDatabaseInfo() {
        String[] Infos = null;
        try {
            Collection Dicts = DictionariesFactory.getDictionariesArray();
            Infos = new String[Dicts.size()];
            int i = 0;
            for (Iterator iter = Dicts.iterator(); iter.hasNext();) {
                Dictionary dict = (Dictionary)iter.next();
                
                //
                int count = 0;
                for (Iterator iter2 = VolumesFactory.getVolumesArray(dict.getName()).iterator(); iter2.hasNext();) {
                    count += ((Volume)iter2.next()).getCount();
                }
                
                //
                Infos[i] = dict.getName() + " has " + count + " entries";
                i++;
            }
        }
        catch (Exception e) {
            throw new RuntimeException("Error in getDatabaseInfo");
        }
        return Infos;
    }

    public String getDictionaryInfo(String dictName) {
        String info = null;
        try {
            Dictionary dict = DictionariesFactory.getDictionaryByName(dictName);
			if (dict != null && !dict.isEmpty()) {
				info = dict.getName() + " \"" + dict.getFullName() + "\"\n";
				
				String languages = "";
                for (Iterator iter = AvailableLanguages.getSourceLanguagesArrayForDict(dict.getName()).iterator(); iter.hasNext();) {
                    languages = languages + " " + (String)iter.next();
                }
				
				info = info + "Available source languages: " + languages + "\n";
				
                //
                int count = 0;
                for (Iterator iter =  VolumesFactory.getVolumesArray(dict.getName()).iterator(); iter.hasNext();) {
                    count = count + ((Volume)iter.next()).getCount();
                }
				
                //
				info = info + dict.getName() + " has " + count + " entries";
			}
			else {
				info = null;
			}
            
        }
        catch (Exception e) {
            throw new RuntimeException("Error in getDictionaryInfo with dict: "+dictName);
        }
        return info;
    }

    /**
        * defineMatch method comment.
        */
    public IAnswer[] lookup(String dictName, String lang, String word, int strategy) {
        Collection Answers = null;
		IAnswer[] AnswerArray = null;
        try {
            if (dictName.equals("*")) {
				Answers = lookupAll(lang, word, strategy);
			}
            if (dictName.equals("!")) {
				Answers = lookupAny(lang, word, strategy);
			}
            Dictionary dict = DictionariesFactory.getDictionaryByName(dictName);
            if (dict != null && !dict.isEmpty()) {
                Answers = lookupDict(dict, lang, word, strategy);
            }
        }
        catch (Exception e) {
			System.out.println("Error in Lookup with Dictionary: " + e.toString());
			e.printStackTrace();
//            throw new RuntimeException("Error in Lookup with Dictionary: " + dictName);
        }
        if (null != Answers) {
			try {
			AnswerArray = new IAnswer[Answers.size()];
			int ind=0;
			for (Iterator i = Answers.iterator(); i.hasNext(); ) {
				QueryResult oneAnswer = (QueryResult) i.next();
				AnswerArray[ind++] = (IAnswer)oneAnswer.getSourceEntry();
			}
			}
			catch (fr.imag.clips.papillon.business.PapillonBusinessException pbe) {
				throw new RuntimeException("Error in lookup with dict: "+ dictName);
			}
			return AnswerArray;
        }
        else {
            return null;
        }
    }

    public Collection lookupAll(String lang, String word, int strategy) {
  		//System.out.println("lookupAll word: " + word + " lang: " + lang + " strat: " + strategy + " user: " + this.getUser());
      Collection Answers = null;
		Vector theEntries = null;
			//Headword[0] = key
			//Headword[1] = lang
			//Headword[2] = value
			//Headword[3] = strategy
		String[] Headword = new String[4];
		Headword[0] = Volume.CDM_headword;
		Headword[1] = lang;
		Headword[2] = word;
		Headword[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];
		Vector myVector = new Vector();
		myVector.add(Headword);
		
        try {
            Answers = DictionariesFactory.getAllDictionariesEntriesCollection(myVector,
																	null,
                                                                     null,
                                                                     this.getUser(),
																			  0,
																			  DEFAULT_LIMIT);
        }
        catch (Exception e) {
			System.out.println("Error in LookupAll: " + e.toString());
			e.printStackTrace();
            //throw new RuntimeException("Error in Lookup All");
        }
        return Answers;
    }

    public Collection lookupAny(String lang, String word, int strategy) {
	   //System.out.println("lookupAny word: " + word + " lang: " + lang + " strat: " + strategy + " user: " + this.getUser());
       Vector Answers = new Vector();
        try {
            Collection Dictionaries = DictionariesFactory.getDictionariesArray();
            Iterator iter = Dictionaries.iterator();
            
            boolean answersIsEmpty = true;
            while ( answersIsEmpty && iter.hasNext() ) {
                Dictionary dict = (Dictionary)iter.next();
                
                Answers.addAll(lookupDict(dict, lang, word, strategy));
                answersIsEmpty = !(Answers.size()>0);
            }
            
        } catch (Exception e) {
			System.out.println("Error in Lookup Any: " + e.toString());
			e.printStackTrace();
            throw new RuntimeException("Error in Lookup Any");
        }
        return (Collection) Answers;
    }

    public Collection lookupDict(Dictionary dict, String lang, String word, int strategy) 
		throws fr.imag.clips.papillon.business.PapillonBusinessException {
		//System.out.println("lookupDict word: " + word + " lang: " + lang + " strat: " + strategy + " user: " + this.getUser());
		Vector theEntries = null;
			//Headword[0] = key
			//Headword[1] = lang
			//Headword[2] = value
			//Headword[3] = strategy
		String[] Headword = new String[4];
		Headword[0] = Volume.CDM_headword;
		Headword[1] = lang;
		Headword[2] = word;
		Headword[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EXACT+1];
		Vector myVector = new Vector();
		myVector.add(Headword);
		
            return DictionariesFactory.getDictionaryEntriesCollection(dict,
                                                                     lang,
																	 null,
                                                                     myVector,
                                                                     null,
																	 null,
                                                                     this.getUser(),
																	 0,DEFAULT_LIMIT);
    }

    public String authenticate(String login, String password) {
        String userMessage = "User unknown";
        try {
            if (null != login && !login.equals("") &&
                null != password && !password.equals("")) {
                User myUser = UsersFactory.findUserByLogin(login);
                if (null != myUser && !myUser.isEmpty()) {
                    if (myUser.HasCorrectPassword(password)) {
                        setUser(myUser);
                        userMessage = "User authenticated";
                    } else {
                        userMessage = "Wrong password";
                    }
                }
            }
        }
        catch (Exception ex) {
            userMessage = ex.toString();
        }

        return userMessage;
    }
	
	protected Object clone() {
		return new DictEngine(enhydraApp);
	}

	public IDictEngine copy() {
		return (IDictEngine)this.clone();
	}

    private void setUser(User aUser) {
        theUser = aUser;
    }
    private User getUser() {
        return theUser;
    }
}
