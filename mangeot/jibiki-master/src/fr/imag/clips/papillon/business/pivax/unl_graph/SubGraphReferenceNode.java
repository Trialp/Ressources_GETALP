/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: SubGraphReferenceNode.java,v 1.2 2003/11/14 17:53:55 serasset Exp $
 *-----------------------------------------------------------
 * $Log: SubGraphReferenceNode.java,v $
 * Revision 1.2  2003/11/14 17:53:55  serasset
 * Handling of subgraphs is now correct.
 *
 * Revision 1.1.1.1  2003/10/13 14:01:13  serasset
 * Creation of unldeco CVS repository.
 *
 */
package fr.imag.clips.papillon.business.pivax.unl_graph;

import java.util.TreeSet;
import java.util.Set;
import java.lang.StringBuffer;
import java.io.Writer;


public class SubGraphReferenceNode extends GraphNode {
    private String subGraphReferenceNumber;

    public SubGraphReferenceNode(String refnum) {
        this(refnum, new TreeSet());
    }

    public SubGraphReferenceNode(String refnum, Set attributes) {
        super(attributes);
        this.subGraphReferenceNumber = refnum;
    }

    public String getSubGraphReferenceNumber() {
        return this.subGraphReferenceNumber;
    }

    public String getNodeId() {
        return this.subGraphReferenceNumber;
    }

    public String toString() {
        StringBuffer out = new StringBuffer();
        this.outputTo(out);
        return out.toString();
    }

    public void outputTo(StringBuffer out) {
        out.append(this.subGraphReferenceNumber);
        this.outputAttributesTo(out);
    }


    public boolean equals(Object anObject) {
        if (anObject == this) return true;
        if (anObject == null) return false;
        if (anObject.getClass() != this.getClass()) return false;
        return this.toString().equals(((SubGraphReferenceNode) anObject).toString());
    }

    public int hashCode() {
        return this.toString().hashCode();
    }

    public void writeDot(Writer out) throws java.io.IOException {
    }
}
