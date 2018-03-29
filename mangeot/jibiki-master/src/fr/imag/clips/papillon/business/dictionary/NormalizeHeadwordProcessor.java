/*
-----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.1.2.1  2007/10/29 15:11:03  serasset
 * NEW: lexalp css now defines different forms for HARMONISED/REJECTED entries
 * NEW: added new db url/user/password configuration keys in papillon.properties file
 * BUG158: headwords are now harmonised at edition and search time, added a "normalise headword" admin action
 *
 *
 */
package fr.imag.clips.papillon.business.dictionary;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.utility.StringNormalizer;

public class NormalizeHeadwordProcessor
        implements IVolumeEntryProcessor {

    public NormalizeHeadwordProcessor() {
        super();
    }

    public void process(VolumeEntry myEntry)
            throws PapillonBusinessException {
        try {
            String newhw = StringNormalizer.normalize(myEntry.getHeadword());
            if (! newhw.equals(myEntry.getHeadword())) {
                PapillonLogger.writeInfoMsg("Normalizing " + myEntry.getHeadword() + " to " + newhw);
                myEntry.setHeadword(newhw);
                myEntry.save();
            }
        } catch (Exception ex) {
            throw new PapillonBusinessException("Error in Normalize Headword Process: ", ex);
        }
    }

}