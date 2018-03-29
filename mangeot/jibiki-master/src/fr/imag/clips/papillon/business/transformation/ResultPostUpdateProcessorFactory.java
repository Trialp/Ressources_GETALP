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

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.Volume;

import fr.imag.clips.papillon.business.PapillonBusinessException;

public class ResultPostUpdateProcessorFactory {
    
    // returns the postUpdateProcessor for the appropriate volume or dictionary. 
    public static  ResultPostUpdateProcessor getPostUpdateProcessor(VolumeEntry volumeEntry)
    throws PapillonBusinessException {
        
        Volume volume = volumeEntry.getVolume();
        Dictionary dictionary = volumeEntry.getDictionary();
        
        String postUpdateProcessorClassName = null;
        postUpdateProcessorClassName = volume.getPostUpdateProcessorClassName();
        
        if (postUpdateProcessorClassName == null) {
            postUpdateProcessorClassName = dictionary.getPostUpdateProcessorClassName();
        }
        
        // If not, use the generic PostUpdateProcessor
        if (postUpdateProcessorClassName == null) {
            postUpdateProcessorClassName = "fr.imag.clips.papillon.business.transformation.PostUpdateProcessor";
        }
        //PapillonLogger.writeDebugMsg("postUpdateProcessor: " + postUpdateProcessorClassName);
        
        // Instanciate the postUpdateProcessor
        ResultPostUpdateProcessor postUpdateProcessor = null;
        try {
            postUpdateProcessor = (ResultPostUpdateProcessor) Class.forName(postUpdateProcessorClassName).newInstance();
        } catch (java.lang.ClassNotFoundException e) {
            throw new PapillonBusinessException("Could not initialize postUpdateProcessor. [class: " + postUpdateProcessorClassName + "]", e);
        } catch (java.lang.InstantiationException e) {
            throw new PapillonBusinessException("Could not initialize postUpdateProcessor. [class: " + postUpdateProcessorClassName + "]", e);
        } catch (java.lang.IllegalAccessException e) {
            throw new PapillonBusinessException("Could not initialize postUpdateProcessor. [class: " + postUpdateProcessorClassName + "]", e);
        }
        
        return postUpdateProcessor;
    }
    
}
