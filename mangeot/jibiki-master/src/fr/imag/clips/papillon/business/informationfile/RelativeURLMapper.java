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
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.informationfile;
import java.util.Hashtable;

// Enhydra SuperServlet imports

public class RelativeURLMapper {
    protected Hashtable myTable = null;
    
    /**
     * The public constructor.
     */
    public RelativeURLMapper() {
            this.myTable = new Hashtable();  
    }

    /**
     * Gets the file id for the relative URL
     *
     * @return the file id.
     */
	  
    public String getFileIDForURL(String url) {
        return (String)this.myTable.get(url);
    }

    /**
     * Sets the file id for the relative URL
     *
     */
	  
    public void setFileIDForURL(String url, String fileId) {
        try {
            this.myTable.put(url, fileId);
        } catch (NullPointerException e) {
            // nothing.
        }
    }


}
