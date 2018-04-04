package getalp.wsd.ufsac.utils;

import java.util.regex.Pattern;

public class Util
{        
    private static final Pattern nonLetterPattern = Pattern.compile("[^\\p{IsAlphabetic}]");

    /**
     * Trims, Lowercases, and Removes all non-ASCII and non-letters characters
     */
    public static String normalize(String str)
    {
        str = str.trim().toLowerCase();
        str = nonLetterPattern.matcher(str).replaceAll("");
        return str;
    }
    
    public static String getWordKeyOfSenseKey(String senseKey)
    {
        String lemma = senseKey.substring(0, senseKey.indexOf("%"));
        int pos = Integer.valueOf(senseKey.substring(senseKey.indexOf("%") + 1, senseKey.indexOf("%") + 2));
        return lemma + "%" + POSHelper.processPOS(pos);
    }
}
