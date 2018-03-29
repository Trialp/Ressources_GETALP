/*
 *  Created by Mathieu Mangeot on 28/04/05.
 *  Copyright 2005 Mangeot-Sérasset. All rights reserved.
 ------------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.2  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.1.2.1  2005/04/29 15:29:09  mangeot
 * Added classes for new version
 *
 *
 *-----------------------------------------------
 */ 

package fr.imag.clips.papillon.business.dictionary;

import fr.imag.clips.papillon.business.PapillonBusinessException;

public interface IVolumeEntryProcessor {

	void process(VolumeEntry myEntry)  throws PapillonBusinessException;

}
