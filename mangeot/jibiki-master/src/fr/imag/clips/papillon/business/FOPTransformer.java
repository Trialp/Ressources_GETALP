/*
 *  jibiki platform
 *
 * Jibiki/Papillon project © GETA-CLIPS-IMAG
 * Francis Brunet-Manquat
 *-----------------------------------------------------------
 * $Id$
 *-----------------------------------------------------------
 * $Log$
 * Revision 1.2  2007/01/05 12:57:49  fbrunet
 * Add undo draft method (bug in EditEntry.java : undo after last finish contribution)
 * Modify transformation method
 * *
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

public class FOPTransformer extends SystemCall {
    String fo; // name of the fo file (local file)
    String pdf; // name of the pdf file (local file)
    String fopJar;
    String batikJar;
    String xalanJar;
    String xercesJar;
    String avalonJar;
    String xmlapiJar;
    
    static String javahome;
    static String fileseparator;
    static String pathseparator;
    
    
    static {
        javahome = System.getProperty("java.home");
        fileseparator = System.getProperty("file.separator");
        pathseparator = System.getProperty("path.separator");
        
    }
    
    public FOPTransformer(String fo, String pdf) throws PapillonBusinessException {
        this.fo = fo; this.pdf = pdf;
        try {
            fopJar =  Enhydra.getApplication().getConfig().getString("Papillon.Fop.Classpath");
            batikJar =  Enhydra.getApplication().getConfig().getString("Papillon.Batik.Classpath");
            xalanJar =  Enhydra.getApplication().getConfig().getString("Papillon.Xalan.Classpath");
            xercesJar =  Enhydra.getApplication().getConfig().getString("Papillon.Xerces.Classpath");
            avalonJar =  Enhydra.getApplication().getConfig().getString("Papillon.Avalon.Classpath");
            xmlapiJar =  Enhydra.getApplication().getConfig().getString("Papillon.XmlApi.Classpath");
            
        } catch (ConfigException e) {
            throw new PapillonBusinessException("Could not get the Saxon Classpath, check the application config file.", e);
        }
        
    }
    
    /** @returns the command line as a String array where the first element is the command, followed by each argument.
        */
    public  String [] commandLine() {
        String command = javahome + fileseparator + "bin/java";
        String classpathOption = "-cp";
        String classpathValue = fopJar + ":" + batikJar  + ":" + xalanJar  + ":" + xercesJar  + ":" + avalonJar  + ":" + xmlapiJar;
        
        String [] cmd = {command, classpathOption, classpathValue, 
            "org.apache.fop.apps.Fop", "-fo", fo, "-pdf", pdf};
        
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
    
    public static void doFOPTransformation(String fo, String pdf) 
        throws PapillonBusinessException {
            FOPTransformer transf = new FOPTransformer(fo, pdf);
            transf.call();
        }
    
}
