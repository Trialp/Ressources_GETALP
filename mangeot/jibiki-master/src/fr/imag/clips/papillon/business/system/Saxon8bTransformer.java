/*
 *  jibiki platform
 *
 * Jibiki/Papillon project © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id$
 *-----------------------------------------------------------
 * $Log *
 */
package fr.imag.clips.papillon.business.system;

import java.util.Vector;
import java.util.Hashtable;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.Enumeration;

import java.io.File;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;

import java.io.InputStream;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.BufferedOutputStream;
import java.io.OutputStream;
import java.io.Writer;
import java.io.InputStreamReader;

import com.lutris.util.Config;
import com.lutris.appserver.server.Enhydra;
import com.lutris.util.ConfigException;

import fr.imag.clips.papillon.business.PapillonBusinessException;

public class Saxon8bTransformer extends SystemCall {
    String xsl; // name of the xsl (local file)
    String xml; // name of the xml (local file)
    String out; // name of the output file (local file)
    String saxonJar;
    
    static String javahome;
    static String fileseparator;
    static String pathseparator;
    
    
    static {
        javahome = System.getProperty("java.home");
        fileseparator = System.getProperty("file.separator");
        pathseparator = System.getProperty("path.separator");
        
    }
    
    public Saxon8bTransformer(String xsl, String xml, String out) throws PapillonBusinessException {
        this.xsl = xsl; this.xml = xml; this.out=out;
        try {
            saxonJar =  Enhydra.getApplication().getConfig().getString("Papillon.Saxon.Classpath");
        } catch (ConfigException e) {
            throw new PapillonBusinessException("Could not get the Saxon Classpath, check the application config file.", e);
        }
        
    }
    
    /** @returns the command line as a String array where the first element is the command, followed by each argument.
    */
    public  String [] commandLine() {
        String command = javahome + fileseparator + "bin/java";
        String classpathOption = "-cp";
        String classpathValue = saxonJar;
        
        String [] cmd = {command, classpathOption, classpathValue, 
            "net.sf.saxon.Transform", "-o", out, xml, xsl};
        
        //
        for (int i = 0; i < cmd.length; i++ ) {
            System.out.println(cmd[i]);
        }
        
        return cmd;
    }
    
    /** writes the data to the process input stream. 
    * Do nothing if you do not have any input data to be given to the process input stream
    *
    */
    public void writeInputToProcess(OutputStream pin) {
    }
    
    /** This method is called back if the process returned a non 0 error code. 
        */
    public void displayErrorOutput(int resCode, String errorMessage) {
        
        System.out.println("java returned " + String.valueOf(resCode));
        System.out.println("--> " + errorMessage);
    }
    
    public static void doSaxonTransformation(String xsl, String xml, String out) 
    throws PapillonBusinessException {
        Saxon8bTransformer transf = new Saxon8bTransformer(xsl, xml, out);
        transf.call();
    }
    
}
