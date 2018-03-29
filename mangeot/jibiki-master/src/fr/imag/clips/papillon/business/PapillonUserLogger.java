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
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 */
package fr.imag.clips.papillon.business;
import fr.imag.clips.papillon.business.PapillonLogger;

public class PapillonUserLogger {
    
    protected final int INITIAL_LOG_SIZE = 1024;

    protected StringBuffer userLog = null;

    public PapillonUserLogger() {
        this.userLog = new StringBuffer(INITIAL_LOG_SIZE);
    }
    
    public String getUserLog() {
        String log = this.userLog.toString();
        this.flushUserLog();
        return log;
    }
    
    /**
     * Appends the given String to the Message that will be provided to the user
     * after the completion of the action.
     */
    public void writeUserLog(String msg) {
        if (null != msg) {
            this.userLog.append(msg);
            if (! msg.endsWith("\n")) {
                this.userLog.append('\n');
            }
            // Log all output to the debug log...
            PapillonLogger.writeDebugMsg(msg);
        }
    }
    
    public void flushUserLog() {
        this.userLog.setLength(0);
    } 
}
