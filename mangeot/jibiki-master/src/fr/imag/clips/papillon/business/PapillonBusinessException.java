/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * (c) Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.1  2004/12/06 16:38:31  serasset
 * Initial revision
 *
 * Revision 1.2  2003/09/03 10:08:30  mangeot
 * reorganizing imports and using eclipse
 *
 * Revision 1.1.1.1  2002/10/28 16:49:12  serasset
 * Creation of the papillon CVS repository for enhydra 5.0
 *
 * Revision 1.4  2002/10/25 14:10:29  mangeot
 * merge between PAPILLON_1_4 and trunk
 * CVg: 	papillon-enh-src/docs/Bugs-papillon.rtf
 *
 * Revision 1.3.2.2  2002/10/23 09:51:06  serasset
 * Clean up of the source tree,
 *     Every source file is now encoded in ISO-Latin-1,
 *     Every html file is still encoded in UTF8.
 *
 * Revision 1.3.2.1  2002/10/09 06:18:05  mangeot
 * password encoding for protection
 *
 * Revision 1.3  2002/07/26 16:41:00  tache
 * *** empty log message ***
 *
 * Revision 1.2  2001/07/04 12:50:43  serasset
 * Creation du dossier CVS pour Papillon, Mise a jour de fichiers pour inclusion du log
 * et de l'Id, suppression du dossier enh-deme introduit par erreur.
 *
 *-----------------------------------------------
 * 
 */
 

package fr.imag.clips.papillon.business;

import com.lutris.appserver.server.httpPresentation.HttpPresentationException;

/**
 * Papillon exception
 *
 * @author
 * @version
 */
public class PapillonBusinessException extends HttpPresentationException {

    /**
    * Public constructor to initialize an exception with a user message.
     */
    public PapillonBusinessException(String msg) {
        super(msg);
    }

    /**
     * Public constructor to initialize an exception with a user message
     * and an exception chain to follow
     */
    public PapillonBusinessException(String msg, Exception ex) {
        super(msg, ex);   
    }
}
