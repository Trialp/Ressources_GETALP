/*
 *  papillon
 * 
 * (c) Francis Brunet-Manquat - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------
 * $Id$
 *------------------------
 * $Log$
 * Revision 1.3  2006/05/05 02:08:23  fbrunet
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


/** Interface PreProcessor.
* PreProcessor Modifies the volume entry before the entry editing
*
*/
public interface ResultPreProcessor {
        
    /** 
    transform the volume entry in pre-preocessing.
    */
    public abstract void transformation(VolumeEntry ve, User user) throws PapillonBusinessException;
    
    
}