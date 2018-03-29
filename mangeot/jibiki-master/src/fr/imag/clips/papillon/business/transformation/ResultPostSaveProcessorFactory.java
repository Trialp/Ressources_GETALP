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
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.Volume;

import fr.imag.clips.papillon.business.PapillonBusinessException;

public class ResultPostSaveProcessorFactory {
    
     // returns the postSaveProcessor for the appropriate volume or dictionary. 
    public static  ResultPostSaveProcessor getPostSaveProcessor(VolumeEntry volumeEntry)
    throws PapillonBusinessException {
        
        Volume volume = volumeEntry.getVolume();
        Dictionary dictionary = volumeEntry.getDictionary();
        
        String postSaveProcessorClassName = null;
        postSaveProcessorClassName = volume.getPostSaveProcessorClassName();
        
        if (postSaveProcessorClassName == null) {
            postSaveProcessorClassName = dictionary.getPostSaveProcessorClassName();
        }
        
        // If not, use the generic PostSaveProcessor
        if (postSaveProcessorClassName == null) {
            postSaveProcessorClassName = "fr.imag.clips.papillon.business.transformation.PostSaveProcessor";
        }

        // Instanciate the postSaveProcessor
        ResultPostSaveProcessor postSaveProcessor = null;
        try {
            postSaveProcessor = (ResultPostSaveProcessor) Class.forName(postSaveProcessorClassName).newInstance();
        } catch (java.lang.ClassNotFoundException e) {
            throw new PapillonBusinessException("Could not initialize postSaveProcessor. [class: " + postSaveProcessorClassName + "]", e);
        } catch (java.lang.InstantiationException e) {
            throw new PapillonBusinessException("Could not initialize postSaveProcessor. [class: " + postSaveProcessorClassName + "]", e);
        } catch (java.lang.IllegalAccessException e) {
            throw new PapillonBusinessException("Could not initialize postSaveProcessor. [class: " + postSaveProcessorClassName + "]", e);
        }
        
        return postSaveProcessor;
    }
    
}
