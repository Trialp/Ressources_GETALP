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
 */
package fr.imag.clips.papillon.business.utility;

public class StringNormalizer {

        public static String normalize(String str) {
        // remove redundant spaces
        if (null != str) {
            // left/right/internal trim
            String trimmed = str.replaceAll("^\\s+", "").replaceAll("\\s+$", "").replaceAll("\\b\\s{2,}\\b", " ");
            // change ’ chars to '
            trimmed = trimmed.replace('’', '\'');

            return trimmed;
        } else {
            return str;
        }
    }
}
