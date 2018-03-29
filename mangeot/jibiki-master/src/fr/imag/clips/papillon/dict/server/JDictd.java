package fr.imag.clips.papillon.dict.server;

import java.net.*;
import java.io.*;
import java.util.*;
import fr.imag.clips.papillon.dict.kernel.*;

// Special Papillon imports
import com.lutris.appserver.server.Application;

/**
 * Start with: java ...JDictd iniFile<p>
 * For the format of the ini file see DatabaseFactory.
 */
public class JDictd implements Runnable {

    protected static java.io.PrintStream myOutStream = new java.io.PrintStream(System.out, true);
	{
		try {
			myOutStream = new java.io.PrintStream(System.out, true, "UTF-8");
		}
		catch (java.io.UnsupportedEncodingException ex) {
			;
		}
	}

	public static final String SERVER_NAME = "Papillon DICT server";
	public static final String SERVER_VERSION = "1.2";
        //	private static boolean fSilent = Boolean.getBoolean("silent");
        public static final int DEFAULT_PORT = 2628;
        public static final int DEFAULT_TIMEOUT = 500000;

        public static final String LISTEN_STRING = "Dictd.Listen";
        public static final String PORT_STRING = "Dictd.Port";
        public static final String TIMEOUT_STRING = "Dictd.Timeout";

        private static boolean fSilent = false;
        private IDictEngine fEngine;
        private ServerSocket fServerSocket;
        private int fTimeout = DEFAULT_TIMEOUT;

        public JDictd(ServerSocket dictdSocket, IDictEngine engine, int timeout) {
            fServerSocket = dictdSocket;
            fEngine = engine;
            fTimeout = timeout;
        }
        public static void listen(int port, Application enhydraApp, int timeout) {
    try {        
        IDictEngine engine = new DictEngine(enhydraApp);
        ServerSocket socket = new ServerSocket(port);
        Runnable r = new JDictd(socket, engine, timeout);
        new Thread(r, SERVER_NAME).start();
        trace(new Date().toString() + ": JDictd started at " + port);
    } catch (Exception e) {
        myOutStream.println("Cannot init: " + e);
    }
}
public static void log(String msg, String file) {
	if (!fSilent) {
		myOutStream.println(msg);
		return;
	}
	try {
		RandomAccessFile out = new RandomAccessFile(file, "rw");
		out.seek(out.length());
		out.writeBytes(msg);
		out.writeByte('\n');
		out.close();
	} catch (Throwable t) {
	}
}
public static void main(String args[]) {
    if (args.length == 0) {
        myOutStream.println("Usage: java ...JDictd configFile");
        System.exit(0);
    }
    listen(DEFAULT_PORT,null, DEFAULT_TIMEOUT);
}

public void run() {
    if (fServerSocket != null) {
        runRFC2229();
    }
}

protected void runRFC2229() {
	try {
		do {
			java.net.Socket s = fServerSocket.accept();
			s.setSoTimeout(fTimeout);
			RFC2229 r = new RFC2229(fEngine, s);
			new Thread(r).start();
		} while (true);
	} catch (IOException e) {
		trace("Error while running: " + e.getMessage());
	}
}
static void trace(String s) {
	log(s, "jdictd.log");
}
}
