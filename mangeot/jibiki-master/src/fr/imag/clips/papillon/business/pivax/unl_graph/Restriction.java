/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: Restriction.java,v 1.1.1.1 2003/10/13 14:01:13 serasset Exp $
 *-----------------------------------------------------------
 * $Log: Restriction.java,v $
 * Revision 1.1.1.1  2003/10/13 14:01:13  serasset
 * Creation of unldeco CVS repository.
 *
 */
package fr.imag.clips.papillon.business.pivax.unl_graph;

import java.lang.StringBuffer;

public class Restriction implements Comparable {
    private String label;
    private char direction;
    private UniversalWord uw;

    // CONSTRUCTORS
    public Restriction() {
        this("", ' ', null);
    }

    public Restriction(String lbl, char dir, UniversalWord uw) {
        this.label = lbl;
        this.direction = dir;
        this.uw = uw;
    }
    
    // ACCESSORS
    public String getLabel() {
        return label;
    }

    public void setLabel(String lbl) {
        this.label = lbl;
    }
    
    public char getDirection() {
        return direction;
    }

    public void setDirection(char chr) {
        this.direction = chr;
    }

    public UniversalWord getUniversalWord() {
        return uw;
    }

    public void setUniversalWord(UniversalWord uw) {
        this.uw = uw;
    }

    // TO STRING

    public String toString() {
        if(uw == null) {
            return "";
        } else {
            StringBuffer out = new StringBuffer();
            this.outputTo(out);
            return out.toString();
        }
    }

    public void outputTo(StringBuffer out) {
        if(uw != null) {
            out.append(this.label);
            out.append(this.direction);
            this.uw.outputTo(out);
        }
    }
    
    /**
     * Add by HT Nguyen
     * Print UW in stree format
     * @param out
     */
    public void outputToTree(StringBuffer out) {
        if(uw != null) {
            out.append(this.label);
            out.append("(");
            this.uw.outputToTree(out);
            out.append(")");
        }
    }
    
    // COMPARATORS

    public int compareTo(Object o) {
        return this.compareTo((Restriction) o);
    }

    public int compareTo(Restriction restr) {
        String thisStr = this.toString();
        String otherStr = restr.toString();
        return thisStr.compareTo(otherStr);
    }
}
