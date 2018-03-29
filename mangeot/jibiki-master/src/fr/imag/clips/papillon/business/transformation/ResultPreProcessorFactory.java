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
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.Volume;

import fr.imag.clips.papillon.business.PapillonBusinessException;

public class ResultPreProcessorFactory {
        
    // returns the preProcessor for the appropriate volume or dictionary. 
    public static  ResultPreProcessor getPreProcessor(VolumeEntry volumeEntry)
        throws PapillonBusinessException {

            Volume volume = volumeEntry.getVolume();
            Dictionary dictionary = volumeEntry.getDictionary();
            
            String preProcessorClassName = null;
            preProcessorClassName = volume.getPreProcessorClassName();
            
            if (preProcessorClassName == null) {
                preProcessorClassName = dictionary.getPreProcessorClassName();
            }
            
            // If not, use the generic PreProcessor
            if (preProcessorClassName == null) {
                preProcessorClassName = "fr.imag.clips.papillon.business.transformation.PreProcessor";
            }
            
            // Instanciate the preProcessor
            ResultPreProcessor preProcessor = null;
            try {
                preProcessor = (ResultPreProcessor) Class.forName(preProcessorClassName).newInstance();
            } catch (java.lang.ClassNotFoundException e) {
                throw new PapillonBusinessException("Could not initialize preProcessor. [class: " + preProcessorClassName + "]", e);
            } catch (java.lang.InstantiationException e) {
                throw new PapillonBusinessException("Could not initialize preProcessor. [class: " + preProcessorClassName + "]", e);
            } catch (java.lang.IllegalAccessException e) {
                throw new PapillonBusinessException("Could not initialize preProcessor. [class: " + preProcessorClassName + "]", e);
            }
            
            return preProcessor;
        }
    
}
