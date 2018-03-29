 /*
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.1  2004/12/06 16:38:31  serasset
 * Initial revision
 *
 * Revision 1.1.1.1  2002/10/28 16:49:14  serasset
 * Creation of the papillon CVS repository for enhydra 5.0
 *
 * Revision 1.2  2001/08/20 15:57:28  salvati
 * Conversion de lien marche corectement.
 *
 * Revision 1.1  2001/08/07 13:12:57  salvati
 * Added in cvs entries.
 *
 */
 
package fr.imag.clips.papillon.business.filetype;
 
public class FileTypeFactory{
    
    public static String[] getFileTypeList()
    {
    String[] theList=new String[4];
    theList[0]="texte";
    theList[1]="html";
    theList[2]="xml";
    theList[3]="tar";    
    return theList; 
 
 
    }
 
 
 }
