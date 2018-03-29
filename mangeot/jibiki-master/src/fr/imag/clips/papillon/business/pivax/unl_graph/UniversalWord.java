/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: UniversalWord.java,v 1.2 2003/10/30 14:17:03 serasset Exp $
 *-----------------------------------------------------------
 * $Log: UniversalWord.java,v $
 * Revision 1.2  2003/10/30 14:17:03  serasset
 * The dictionary manager now handles errors in uploaded dictionary in a safer way.
 * Modifications on the dictionary are done only when there is NO error.
 *
 * Revision 1.1.1.1  2003/10/13 14:01:13  serasset
 * Creation of unldeco CVS repository.
 *
 */
package fr.imag.clips.papillon.business.pivax.unl_graph;

import java.util.SortedSet;
import java.util.TreeSet;
import java.util.Set;

import java.lang.StringBuffer;
import java.util.Iterator;

// TODO: perhaps we should cache the toString representation that's used in equal and hashCode
public class UniversalWord {

    private String headword;
    private SortedSet restrictions;

    // CONSTRUCTORS
    public UniversalWord() {
        this.headword = "";
        this.restrictions = new TreeSet();
    }

    public UniversalWord(String headword) {
        this.headword = headword;
        this.restrictions = new TreeSet();
    }

    public UniversalWord(String headword, Set restrictions) {
        this.headword = headword;
        this.restrictions = (SortedSet) new TreeSet(restrictions);
    }
    
    public UniversalWord(String headword, SortedSet restrictions) {
        this.headword = headword;
        this.restrictions = restrictions;
    }
    
    // ACCESSORS
    public String getHeadword() {
        return this.headword;
    }

    public SortedSet getRestrictions() {
        return restrictions;
    }

    public boolean add(Restriction restr) {
        return this.restrictions.add(restr);
    }
    
    public String toString() {
        StringBuffer out = new StringBuffer();
        this.outputTo(out);
        return out.toString();
    }
    
    public String toStringTree() {
        StringBuffer out = new StringBuffer();
        this.outputToTree(out);
        return out.toString();
    }
    
    public String toStringTree(String transcription) {
        StringBuffer out = new StringBuffer();
        this.outputToTree(out, transcription);
        return out.toString();
    }
    
    public String toStringTreeContent() {
        StringBuffer out = new StringBuffer();
        this.outputToTreeContent(out);
        return out.toString();
    }

    public void outputTo(StringBuffer out) {
        out.append(this.headword);
        if (this.restrictions != null && ! this.restrictions.isEmpty()) {
            out.append('(');
            Iterator iter = this.restrictions.iterator();
            while(iter.hasNext()) {
                ((Restriction) iter.next()).outputTo(out);
                out.append(',');
            }
            // Replace the last ',' by a ')'
            out.setCharAt(out.length()-1,')');
        }
    }
    
    /**
     * Print headword
     * @param out
     */
    public void outputToTree(StringBuffer out) {
        out.append("$UW(");
        outputToTreeContent(out);
        out.append(")");
    }
    

    /**
     * Print headword
     * @param out
     */
    public void outputToTree(StringBuffer out, String transcription) {
        out.append("$UW(");
        outputToTreeContent(out, transcription);
        out.append(")");
    }
    
    /**
     * Print headword
     * @param out
     */
    public void outputToTreeContent(StringBuffer out) {
    	out.append(parseHW(this.headword));
        if (this.restrictions != null && ! this.restrictions.isEmpty()) {
            out.append(',');
            Iterator iter = this.restrictions.iterator();
            if (this.restrictions.size() > 0) {
            	out.append("$REST(");
	            while(iter.hasNext()) {
	                ((Restriction) iter.next()).outputToTree(out);
	                if (iter.hasNext())
	                	out.append(',');
	                else
	                	out.append(')');
	            }
            }
        }
    }
    
    /**
     * Print headword
     * @param out
     */
    public void outputToTreeContent(StringBuffer out, String transcription) {
    	out.append("$UW( " + transcription + ")");
        if (this.restrictions != null && ! this.restrictions.isEmpty()) {
            out.append(',');
            Iterator iter = this.restrictions.iterator();
            if (this.restrictions.size() > 0) {
            	out.append("$REST(");
	            while(iter.hasNext()) {
	                ((Restriction) iter.next()).outputToTree(out);
	                if (iter.hasNext())
	                	out.append(',');
	                else
	                	out.append(')');
	            }
            }
        }
    }
    
    /**
     * Separate words in headword
     * @param headword
     * @return
     */
    public String parseHW(String headword) {
    	String hw = headword;
    	int pos =  hw.indexOf("-");
    	boolean isTransformed = false;
    	if (pos >= 0) {
    		hw = hw.replaceAll("-", ",-,");
    		isTransformed = true;
    	}
    	pos =  hw.indexOf("_");
    	if (pos >= 0) {
    		hw = hw.replaceAll("_", ",#sp,");
    		isTransformed = true;
    	}
    	if (isTransformed)
    		hw = "$MOTCP(" + hw + ")";
    	/*
    	if (pos >= 0) {
    		String prefix = hw.substring(0, pos);
    		String suffix = hw.substring(pos +1, hw.length());
    		hw = "$MOTCP(" + prefix + ",-," + suffix + ")";
    	}
    	
    	pos =  hw.indexOf("_");
    	if (pos >= 0) {
    		String prefix = hw.substring(0, pos);
    		String suffix = hw.substring(pos +1, hw.length());
    		hw = "$MOTCP(" + prefix + ",#sp," + suffix + ")";
    	}
    	*/
    	hw = "$HW(" + hw + ")";
    	return hw;
    }

    // Static method to compute headword from a string (to be dropped after use...)
    public static String getHeadword(String uw) {
        // Pour l'instant, on se contente de chercher la première parenthèse ouvrante...
        int parpos = uw.indexOf("(");
        if (parpos > 0) {
            return uw.substring(0, parpos);
        } else {
            return uw;
        }
    }

    public boolean isANumber() {
        // True if the UW represent a number.
        try {
            int val = Integer.parseInt(this.getHeadword(), 10);
            return true;
        } catch (java.lang.NumberFormatException e) {
            return false;
        }
    }

    
    public boolean equals(Object anObject) {
        if (anObject == this) return true;
        if (anObject == null) return false;
        if (anObject.getClass() != this.getClass()) return false;
        return this.toString().equals(((UniversalWord) anObject).toString());
    }

    public int hashCode() {
        return this.toString().hashCode();
    }
    
}
