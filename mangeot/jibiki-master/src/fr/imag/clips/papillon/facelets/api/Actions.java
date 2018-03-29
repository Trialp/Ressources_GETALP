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
package fr.imag.clips.papillon.facelets.api;
import org.w3c.dom.Node;
import fr.imag.clips.papillon.business.PapillonBusinessException;


public interface Actions {
    
    /**
     * returns the menu that contains the internal status/action menus for a specific entry.
     */
    public Node getActions(String entryid) throws PapillonBusinessException;
    
    
    
}
