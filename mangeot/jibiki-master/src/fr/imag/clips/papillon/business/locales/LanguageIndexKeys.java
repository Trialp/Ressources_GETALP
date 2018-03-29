package fr.imag.clips.papillon.business.locales;

import java.util.Locale;

/**
 * Provide the set of characters traditionaly used as index keys for dictionary access.
 */

public class LanguageIndexKeys {

    private String isoLang;

    private String[] keys;


    public String getMyLanguage() {
        return isoLang;
    }

    public String[] getKeys() {
        return keys;
    }

    public LanguageIndexKeys(String language) {
        isoLang = language;
        setup();
    }

    protected void setup() {
        if ("fra".equals(isoLang) ) {
            keys = new String[] {
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            };
        } else if ("deu".equals(isoLang) ) {
            keys = new String[] {
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            };
        } else if ("ita".equals(isoLang) ) {
            keys = new String[] {
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            };
        } else if ("slv".equals(isoLang) ) {
            keys = new String[] {
                "A", "B", "C", "Č", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "R", "S", "Š", "T", "U", "V", "Z", "Ž"
            };
        } else {
            keys = new String[] {
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            };
        }
    }
}
