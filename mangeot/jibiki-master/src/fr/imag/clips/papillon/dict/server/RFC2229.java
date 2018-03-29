package fr.imag.clips.papillon.dict.server;

//import com.sun.java.util.collections.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Date;

import fr.imag.clips.papillon.business.dictionary.IAnswer;
import fr.imag.clips.papillon.business.dictionary.IQuery;
import fr.imag.clips.papillon.dict.kernel.IDictEngine;

/**
* Insert the type's description here.
* Creation date: (28.07.01 11:45:40)
* @author: Administrator
*/
public class RFC2229 implements Runnable {
    protected static java.io.PrintStream myOutStream = new java.io.PrintStream(System.out, true);
	{
		try {
			myOutStream = new java.io.PrintStream(System.out, true, "UTF-8");
		}
		catch (java.io.UnsupportedEncodingException ex) {
			;
		}
	}

    protected final static String CRLF = "\r\n";
    protected final static String MimeHeader = "Content-type: text/plain; charset=utf-8" +
		CRLF +
		"Content-transfer-encoding: 8bit" + CRLF;
    private java.net.Socket fSocket;
    private IDictEngine fEngine;
    /**
        * RFC2229 constructor comment.
        */
    public RFC2229(IDictEngine engine, java.net.Socket s) {
        super();
        this.fEngine = engine.copy();
        this.fSocket = s;
    }
    IDictEngine getEngine() {
        return fEngine;
    }
    /**
        * Insert the method's description here.
        * Creation date: (28.07.01 16:08:45)
        * @return java.io.OutputStream
        */
    public java.io.OutputStream getOut() throws IOException {
        return fSocket.getOutputStream();
    }
    int processBanner() throws IOException {
        String myString = "220 " + JDictd.SERVER_NAME + " " 
			+ JDictd.SERVER_VERSION + " <" 
			+ System.currentTimeMillis()
			+ "@papillon-dictionary.org>";
        return sendLine(myString);
    }
    int processClient() throws IOException {
        return sendLine("250 ok");
    }
    public int processCommand(String cmd) throws IOException {
        // RFC2229 3.1 - Initial Connection
        if (cmd == null) {
            return processBanner();
        }
		if (getEngine().IsMime()) {
			sendLine(MimeHeader);
		}

        cmd = cmd.trim();
        // RFC2229 3.2 - DEFINE
        if (cmd.toUpperCase().startsWith("DEFINE")) {
            return processDefine(cmd);
        }
        // RFC2229 3.3 - MATCH
        if (cmd.toUpperCase().startsWith("MATCH")) {
            return processMatch(cmd);
        }
        // RFC2229 3.5 - SHOW command
        if (cmd.toUpperCase().startsWith("SHOW")) {
            return processShow(cmd);
        }
        // RFC2229 3.6
        if (cmd.toUpperCase().startsWith("CLIENT")) {
            return processClient();
        }
        // RFC2229 3.7
        if (cmd.equalsIgnoreCase("STATUS")) {
            return processStatus();
        }
        // RFC2229 3.8
        if (cmd.equalsIgnoreCase("HELP")) {
            return processHelp();
        }
        // RFC2229 3.9
        if (cmd.equalsIgnoreCase("QUIT") ||
			cmd.equalsIgnoreCase("EXIT") ) {
            return processQuit();
        }
        // RFC2229 3.10
        if (cmd.toUpperCase().startsWith("OPTION")) {
            return processOption(cmd);
        }
        // RFC2229 3.11 - AUTH
        if (cmd.toUpperCase().startsWith("AUTH")) {
            return processAuth(cmd);
        }
        // NOT IN RFC2229, used for convenience
        if (cmd.toUpperCase().startsWith("LOOKUP")) {
            return processLookup(cmd);
        }
        // NOT IN RFC2229, used for convenience
        if (cmd.toUpperCase().startsWith("LOOKM")) {
            return processLookm(cmd);
        }

        return sendLine("500 Syntax error, command not recognized");
    }
    int processAuth(String cmd) throws IOException {

        java.util.StringTokenizer st = new java.util.StringTokenizer(cmd);
        st.nextToken();
        String user = st.nextToken();
        if (user == null) {
            sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
            return 0;
        }
        String info = st.nextToken();
        if (info == null) {
            sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
            return 0;
        }

        sendLine("110 " + getEngine().authenticate(user,info));
        sendLine(".");
        sendLine("250 Command complete");
        return 0;
    }
    int processDefine(String cmd) throws IOException {
        java.util.StringTokenizer st = new java.util.StringTokenizer(cmd);
		if (st.countTokens() < 3) {
            sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
            return 0;
		}
        String def = st.nextToken();
        String db = st.nextToken();
		String what = getWhat(st);
		return myProcessQuery(db, null, what, IQuery.STRATEGY_NONE, cmd);
    }
    int processHelp() throws IOException {
        sendLine("113 Help text follows");
        sendLine("LOOKUP dictionary src word   -- lookup word in language src in dictionary");
        sendLine("                             -- src = 3 letter language codes ISO 639/2-T");
        sendLine("LOOKM dict src strat word    -- lookup word in language src in dictionary");
        sendLine("                             -- dict using strategy strat");
        sendLine("DEFINE database word         -- look up word in database");
        sendLine("MATCH database strategy word -- match word in database using strategy");
        sendLine("                             -- The MATCH command returns up to 20 matches");
        sendLine("OPTION MIME                  -- use MIME headers");
        sendLine("OPTION VIEW (TEXT|XML|HTML)  -- toggle between output formats");
        sendLine("SHOW DB                      -- list all accessible databases");
        sendLine("SHOW DATABASES               -- list all accessible databases");
        sendLine("SHOW STRAT                   -- list available matching strategies");
        sendLine("SHOW STRATEGIES              -- list available matching strategies");
        sendLine("SHOW INFO database           -- provide information about the database");
        sendLine("SHOW SERVER                  -- provide site-specific information");
        sendLine("CLIENT info                  -- identify client to server");
        sendLine("AUTH user string             -- provide authentication information");
        sendLine("STATUS                       -- display timing information");
        sendLine("HELP                         -- display this help information");
        sendLine("QUIT                         -- terminate connection");
        sendLine(".");
        sendLine("250 Command complete");
        return 0;
    }
    int processLookup(String cmd) throws IOException {
        java.util.StringTokenizer st = new java.util.StringTokenizer(cmd);
		if (st.countTokens() < 4) {
            sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
            return 0;
		}
        String lookup = st.nextToken();
        String dict = st.nextToken();
        String src = st.nextToken();
        if (src.length() !=3) {
            sendLine("550 Invalid language name, use a 3 letter ISO 639-2/T language code");
            return 0;
        }
		String what = getWhat(st);
        return myProcessQuery(dict, src, what, IQuery.STRATEGY_NONE, cmd);
    }

    int processLookm(String cmd) throws IOException {
        java.util.StringTokenizer st = new java.util.StringTokenizer(cmd);
		if (st.countTokens() < 5) {
            sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
            return 0;
		}
        String lookup = st.nextToken();
        String dict = st.nextToken();
        String src = st.nextToken();
        if (src.length() !=3) {
            sendLine("550 Invalid language name, use a 3 letter ISO 639-2/T language code");
            return 0;
        }
        String strat = st.nextToken();
        int strategy = myProcessStrategy(strat);
        if (strategy == IQuery.STRATEGY_NONE) {
            sendLine("551 Invalid strategy, use \"SHOW STRAT\" for a list of strategies");
            return 0;
        }		
		String what = getWhat(st);
		return myProcessQuery(dict, src, what, strategy, cmd);
    }
    int processMatch(String cmd) throws IOException {
         java.util.StringTokenizer st = new java.util.StringTokenizer(cmd);
		if (st.countTokens() < 4) {
            sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
            return 0;
		}
        String lookup = st.nextToken();
        String db = st.nextToken();
        String strat = st.nextToken();
        int strategy = myProcessStrategy(strat);
        if (strategy == IQuery.STRATEGY_NONE) {
            sendLine("551 Invalid strategy, use \"SHOW STRAT\" for a list of strategies");
            return 0;
        }		
		String what = getWhat(st);
        IAnswer[] m = getEngine().lookup(db, null, what, strategy);
        if (m == null) {
            sendLine("550 Invalid database, use \"SHOW DB\" for list of databases");
            return 0;
        }		
        int count = m.length;
       if (m == null) {
            sendLine("550 Invalid database, use \"SHOW DB\" for list of databases");
            return 0;
        }
        if (count == 0) {
            sendLine("552 No match");
            return 0;
        }
        if (count == 1 && m[0].getType()==IAnswer.Error) {
            sendLine("553 Internal Error ProcessMatch: " + cmd);
            return -1;
        }
        sendLine("152 "+count+" matches found: list follows");
		try {
			for (int i = 0; i < m.length; i++){
				String id = m[i].getDictionaryName();
				String res = m[i].getHeadword();
				sendLine(id + " " + res);
			}
        }
        catch (Exception e) {
            sendLine("553 Internal Error ProcessMatch: " + cmd);
            return -1;
        }
        sendLine(".");
        sendLine("250 Command complete");
        return 0;
    }
    int processOption(String cmd) throws IOException {
        java.util.StringTokenizer st = new java.util.StringTokenizer(cmd);
		if (st.countTokens() < 2) {
            sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
            return 0;
		}
        String option = st.nextToken();
        String what = st.nextToken().toUpperCase();
		
        if (what.equals("MIME")) {
			this.getEngine().setMime(true);
			return sendLine("250 ok");
        }
         if (what.equals("VIEW")) {
			if (st.countTokens() != 1) {
				sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
				return 0;
			}
			String view = st.nextToken().toUpperCase();
			if (view.equals(IDictEngine.TEXT_view)) {
				this.getEngine().setView(view);
			}
			else if (view.equals(IDictEngine.XML_view)) {
				this.getEngine().setView(view);
			}
			else if (view.equals(IDictEngine.HTML_view)) {
				this.getEngine().setView(view);
			}
			else {
				sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
				return 0;
			}
			return sendLine("250 ok");
        }
        return sendLine("210 Not supported");
    }
    int myProcessQuery(String dict, String lang, String what, int strategy, String cmd) throws IOException {
        IAnswer[] m = getEngine().lookup(dict, lang, what, strategy);
        if (m == null) {
            sendLine("550 Invalid database, use \"SHOW DB\" for list of databases");
            return 0;
        }
        if (m.length == 0) {
            sendLine("552 No match");
            return 0;
        }
        if (m.length == 1 && m[0].getType()==IAnswer.Error) {
            sendLine("553 Internal Error myProcessQuery: " + cmd);
            return -1;
        }
        sendLine("150 "+m.length+" definitions retrieved: definitions follow");
        try {
            for (int i = 0; i < m.length; i++){
                String id = m[i].getDictionaryName();
                String name = m[i].getDictionaryFullName();
                sendLine("151 \""+what+"\" "+id+" \""+name+"\": text follows");
				String res = this.getEngine().processView(m[i]);
                sendLine(res);
                sendLine(".");
            }
        }
        catch (Exception e) {
            sendLine("553 Internal Error myProcessQuery: " + cmd);
            return -1;
        }       
        sendLine("250 Command complete ");
        return 0;
    }
    int processQuit() throws IOException {
        return sendLine("221 Closing connection");
    }
    int processShow(String cmd) throws IOException {
        java.util.StringTokenizer st = new java.util.StringTokenizer(cmd);
        String show = st.nextToken();
        String what = st.nextToken().toUpperCase();
        if (what.equals("DB") || what.equals("DATABASES")) {
            return processShowDB();
        }
        if (what.equals("INFO")) {
            String db = st.nextToken();
            return processShowInfo(db);
        }
        if (what.equals("STRAT") || what.equals("STRATEGIES")) {
            return processShowStrat();
        }
        if (what.equals("SERVER")) {
            return processShowServer();
        }
        return 0;
    }
    int processShowDB() throws IOException {
        String[] all = getEngine().getDictionariesInfo();
        sendLine("110 "+all.length+" databases present - list follows");
        for (int i = 0; i < all.length; i++){
            sendLine(all[i]);
        }
        sendLine(".");
        sendLine("250 Command complete");
        return 0;
    }
    int processShowInfo(String db) throws IOException {
        if (null == db || db.equals("")) {
            sendLine("550 Invalid syntax, use \"HELP\" for syntax help");
            return 0;
        }
        
        String it = getEngine().getDictionaryInfo(db);
        sendLine("112 database information follows");
        if (it != null) {
            sendLine(it);
        } else {
            sendLine("Invalid database, use \"SHOW DB\" for list of databases");
        }
        sendLine(".");
        sendLine("250 Command complete");
        return 0;
    }
    int processShowServer() throws IOException {
        String[] all = getEngine().getDatabaseInfo();
        sendLine("114 server information follows");
        sendLine("Java implementation of the DICT protocol (RFC2229)");
        sendLine("Copyright © 2002 Mathieu Mangeot Papillon project");
        for (int i = 0; i < all.length; i++){
            sendLine(all[i]);
        }
        sendLine(".");
        sendLine("250 Command complete");
        return 0;
    }
    int processShowStrat() throws IOException {
            // sendLine("555 No strategies available");
            sendLine("111 9 strategies present: list follows");
            sendLine("exact \"Match words exactly\"");
            sendLine("prefix \"Match prefixes\"");
            sendLine("substring \"Match substring occurring anywhere in word\"");
            sendLine("suffix \"Match word suffixes\"");
            sendLine("insensitive \"Match words case-insensitively\"");
            sendLine("inprefix \"Match case-insensitive word prefixes\"");
            sendLine("insuffix \"Match case-insensitive word suffixes\"");
            sendLine("insubstring \"Match case-insensitive substring occurring anywhere in word\"");
            sendLine("lemmatize \"Using XRCE lemmatizer module, for eng, fra, esp, ita\"");
            sendLine("foks \"Using Slaven BILAC's FOKS module, only for Japanese\"");
            sendLine(".");
            sendLine("250 Command complete");
            return 0;
    }
    int processStatus() throws IOException {
        return sendLine("210 Not supported");
    }
    int myProcessStrategy(String strat) throws IOException {
        int strategy = IQuery.STRATEGY_NONE;
        if ("exact".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_EXACT;
        }
        else if ("prefix".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_PREFIX;
        }
        else if ("suffix".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_SUFFIX;
        }
        else if ("substring".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_SUBSTRING;
        }
        else if ("insensitive".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_INSENSITIVE_EXACT;
        }
        else if ("inprefix".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_INSENSITIVE_PREFIX;
        }
        else if ("insuffix".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_INSENSITIVE_SUFFIX;
        }
        else if ("insubstring".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_INSENSITIVE_SUBSTRING;
        }
        else if ("lemmatize".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_LEMMATIZE;
        }
        else if ("foks".equalsIgnoreCase(strat)) {
            strategy = IQuery.STRATEGY_FOKS;
        }
		return strategy;
    }
	
    public void run() {
        try {
            this.fEngine.initialize();
            InputStreamReader myReader = new InputStreamReader(fSocket.getInputStream(), "UTF-8");
            BufferedReader in = new BufferedReader(myReader);
            String inputLine;
            processCommand(null);
            while ((inputLine = in.readLine()) != null) {
                myOutStream.println(new Date().toString() + ": JDictd user command: " + inputLine);
                processCommand(inputLine);
                if ("QUIT".equalsIgnoreCase(inputLine) ||
					"EXIT".equalsIgnoreCase(inputLine)) {
                    break;
                }
            }
        } catch (Exception e) {
            trace(e.toString());
        } finally {
            this.fEngine.finalize();
            try {
                fSocket.close();
            } catch (Throwable t) {
                trace(t.toString());
            }
        }
    }
    public int sendLine(String line) throws IOException {
        if (line == null) return 0;
        sendLine(line, getOut());
        return line.length() + 2;
    }
    public void sendLine(String line, OutputStream out) 
		throws IOException, 
				java.io.UnsupportedEncodingException {
        out.write(line.getBytes("UTF-8"));
        out.write(CRLF.getBytes("UTF-8"));
    }
    void trace(String s) {
        myOutStream.println(s);
    }
	protected String getWhat(java.util.StringTokenizer st) {
		String what = st.nextToken();
		if (what.startsWith("\"")) {
			what = what.substring(1, what.length());
		}
		while (st.countTokens()>0) {
			what += " " + st.nextToken();
		}
		if (what.endsWith("\"")) {
			what = what.substring(0, what.length()-1);
		}
		return what;
	}
}
