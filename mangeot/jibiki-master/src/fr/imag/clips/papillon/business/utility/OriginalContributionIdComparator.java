/*
 * HeadwordComparator.java
 *  Created by Mathieu Mangeot on 28/04/05.
 *  Copyright 2005 Mangeot-Sérasset. All rights reserved.
 *
 * Cascade call of comparators when equality:
 *  StatusComparator -> AuthorComparator -> HeadwordComparator -> HomographNumberComparator -> ParticuleComparator
 *  ReviewerComparator -> AuthorComparator -> HeadwordComparator -> HomographNumberComparator -> ParticuleComparator
 *  ReviewDateComparator -> CreationDateComparator -> HeadwordComparator -> HomographNumberComparator -> ParticuleComparator
 *  OriginalContributionIdComparator -> HeadwordComparator -> HomographNumberComparator -> ParticuleComparator
 *
 *----------------------------------------
 *$Id$
 *------------------------------------------
 *$Log$
 *Revision 1.3  2005/07/16 09:15:58  mangeot
 *Added HomographNumberComparator and ParticuleComparator
 *
 *Revision 1.2  2005/06/15 16:48:28  mangeot
 *Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *Revision 1.1.2.3  2005/06/09 13:29:23  mangeot
 *Sorting bug fixes
 *
 *Revision 1.1.2.2  2005/04/29 18:43:13  mangeot
 **** empty log message ***
 *
 *Revision 1.1.2.1  2005/04/29 15:29:09  mangeot
 *Added classes for new version
 *
 *
 *--------------------------------------
 */
package fr.imag.clips.papillon.business.utility;

import fr.imag.clips.papillon.business.dictionary.VolumeEntry;

public class OriginalContributionIdComparator implements java.util.Comparator {
	// Compare two Strings. Callback for sort.
	// effectively returns a-b;
	// e.g. +1 (or any +ve number) if a > b
	// 0 if a == b
	// -1 (or any -ve number) if a < b
	public final int compare ( Object a, Object b) {
		int res = 0;
		try {
			String origIdA = ((VolumeEntry)a).getOriginalContributionId();
			String origIdB = ((VolumeEntry)b).getOriginalContributionId();
			if (origIdA != null && !origIdB.equals("")) {
			// a et b ont un originalId
				if (origIdB != null && !origIdB.equals(""))  {
					res = origIdA.compareTo(origIdB);
					if (res == 0) {
						res = new HeadwordComparator().compare(a,b);
					}
				}
			// a seulement a un originalId
				else {
					res = 1;
				}
			} 
			else {
			// b seulement un originalId
				if (origIdB != null && !origIdB.equals(""))  {
					res = -1;
				}
			// ni a ni b n'ont d'originalId
				else {
					res = new HeadwordComparator().compare(a,b);
				}
			}
		}
		catch (Exception e) {
			res = 0;
		}
		return res;
	}
} 
