/*
 *  Created by Mathieu Mangeot on 28/04/05.
 *  Copyright 2005 Mangeot-Sérasset. All rights reserved.
 *
 * Cascade call of comparators when equality:
 *  StatusComparator -> AuthorComparator -> HeadwordComparator -> HomographNumberComparator -> ParticuleComparator
 *  ReviewerComparator -> AuthorComparator -> HeadwordComparator -> HomographNumberComparator -> ParticuleComparator
 *  ReviewDateComparator -> CreationDateComparator -> HeadwordComparator -> HomographNumberComparator -> ParticuleComparator
 *  OriginalContributionIdComparator -> HeadwordComparator -> HomographNumberComparator -> ParticuleComparator
 *
 * Note: the HeadwordComparator might be modified in order to sort following local alphabetical orders
 *
 *----------------------------------------
 *$Id$
 *------------------------------------------
 *$Log$
 *Revision 1.2  2005/07/30 16:03:27  mangeot
 *Bug fixes
 *
 *Revision 1.1  2005/07/16 09:15:58  mangeot
 *Added HomographNumberComparator and ParticuleComparator
 *
 *
 *
 *--------------------------------------
 */
package fr.imag.clips.papillon.business.utility;

import fr.imag.clips.papillon.business.dictionary.VolumeEntry;

public class HomographNumberComparator implements java.util.Comparator {
	// Compare two Strings. Callback for sort.
	// effectively returns a-b;
	// e.g. +1 (or any +ve number) if a > b
	// 0 if a == b
	// -1 (or any -ve number) if a < b
	public final int compare (Object a, Object b) {
		int res = 0;
		try {
			String hnA = ((VolumeEntry)a).getHomographNumber();
			String hnB = ((VolumeEntry)b).getHomographNumber();
			if (hnA != null && hnB != null) {
				res = hnA.compareTo(hnB);
			}
		}
		catch (Exception e) {
			res = 0;
		}
		// FIXME: special sort for estonian!
		if (res == 0) {
			return new ParticuleComparator().compare(a,b);
		}
		else {
			return res;
		}
	} 
} 
