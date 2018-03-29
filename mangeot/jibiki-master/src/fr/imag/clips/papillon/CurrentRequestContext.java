/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.1.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 *
 *-----------------------------------------------
 * Papillon enhydra Thread Local for current Transaction.
 */

package fr.imag.clips.papillon;

import fr.imag.clips.papillon.business.PapillonBusinessException;

import fr.imag.clips.papillon.facelets.util.JibikiContext;

/**
 * The CurrentRequestContext object is basically a HashMap (encapsulated in a JibikiContext).
 * This context is available anytime during the handling of a request. This means that you can use it to store
 * objects tht you will retreive during the handling of the SAME REQUEST.
 * WARNING: A redirect will generate another request, so it will generate a new context.
 * WARNING: Each AJAX call is a request on its own, each of these will be assigned a different context.
 */

public class CurrentRequestContext {
    
    private static ThreadLocal currentContext = new ThreadLocal() {
        protected synchronized Object initialValue() {
            return null;
        }
    };
    
    public static JibikiContext get() {
        return ((JibikiContext) currentContext.get());
    }
    
    public static void registerNewPresentationContext(JibikiContext context) throws PapillonBusinessException {
        if ( null != currentContext.get() ) throw new PapillonBusinessException("Attempt to register an already existing Context.");
        currentContext.set(context);
    }
    
    public static void releasePresentationContext() throws PapillonBusinessException {
        JibikiContext context = ((JibikiContext) currentContext.get());
        if (null == context ) throw new PapillonBusinessException("Attempt to release a null transaction.");
        currentContext.set(null);
    }
}