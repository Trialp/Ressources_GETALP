/*
 *  papillon
 * 
 * (c) Francis Brunet-Manquat - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------
 * $Id$
 *------------------------
 * $Log$
 * Revision 1.1  2006/05/05 02:08:23  fbrunet
 * bug correction : url utf8 transfert (in createEntryInit)
 *
 *
 *
 *------------------------
 */
package fr.imag.clips.papillon.business.transformation;


import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.user.User;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;


/** Interface PostSaveProcessor.
* PostSaveProcessor Modifies the volume entry after the entry editing
*
*/
public interface ResultPostSaveProcessor {
    
    /** 
    transform the volume entry in post-save-preocessing.
    */
    public abstract void transformation(VolumeEntry ve, User user) throws PapillonBusinessException;
    
    
}