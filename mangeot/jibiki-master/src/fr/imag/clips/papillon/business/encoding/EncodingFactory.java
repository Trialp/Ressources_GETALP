 /*
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.1  2004/12/06 16:38:31  serasset
 * Initial revision
 *
 * Revision 1.1.1.1  2002/10/28 16:49:13  serasset
 * Creation of the papillon CVS repository for enhydra 5.0
 *
 * Revision 1.3  2001/10/17 13:02:03  serasset
 * Distinction entre Doucmuent (abstrait) et fichier (qui constituent concretement un doucument)
 *
 * Revision 1.2  2001/09/07 06:55:04  serasset
 * *** empty log message ***
 *
 * Revision 1.1  2001/08/07 13:13:46  salvati
 * Added to cvs entries.
 *
 */
 
package fr.imag.clips.papillon.business.encoding;
import org.w3c.tidy.Configuration;

public class EncodingFactory{

    public static String[] getEncodingList() {
        String[] theList= {
            "--guess--",
            "UTF-8",
            "ISO-latin-1",
            "Mac Roman",
            "ISO-2022-JP",
            "US-ASCII"
        };
        return theList;
    }
    
    // These two lists should correspond !
    public static int[] getTidyEncodingList() {
        int[] theList= {
            Configuration.UTF8,
            Configuration.LATIN1,
            Configuration.MACROMAN,
            Configuration.ISO2022,
            Configuration.ASCII
        };
        return theList;
    }
    
    public static boolean haveToGuessEncoding(int formEncoding) {
        // Assumes that the first encoding is a special "guess" value...
        return ((formEncoding == 0) || (formEncoding > getEncodingList().length));
    }
    
    public static int getTidyEncodingFromFormEncoding(int formEncoding) {
        // Returns an out of bound exception if called with "--guess--".
        // You should use haveToGuessEncoding before calling this...
        return getTidyEncodingList()[formEncoding-1];
    } 
 }
