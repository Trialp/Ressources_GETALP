/*
 *  Jibiki plateform
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
 */
package fr.imag.clips.papillon.facelets.util;

import java.util.HashMap;
import java.util.Map;

public class JibikiContext {
    private HashMap context;

    public JibikiContext() {
        this.context = new HashMap();
    }

    public JibikiContext(Map map) {
        this.context = new HashMap(map);
    }

    public JibikiContext(JibikiContext ctxt) {
        this.context = new HashMap(ctxt.context);
    }

    public Object get(String key) {
        return this.context.get(key);
    }

    public Object set(String key, Object value) {
        return this.context.put(key, value);
    }

}
