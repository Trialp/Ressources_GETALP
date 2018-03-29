/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: UWNode.java,v 1.2 2003/11/14 17:53:55 serasset Exp $
 *-----------------------------------------------------------
 * $Log: UWNode.java,v $
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
import java.util.Iterator;

public class UWNode extends GraphNode {
    private UniversalWord uw;
    private String instanceNumber;

    public UWNode(UniversalWord uw) {
        this(uw, "", new TreeSet());
    }

    public UWNode(UniversalWord uw, String instanceNumber) {
        this(uw, instanceNumber, new TreeSet());
    }
    
    public UWNode(UniversalWord uw, Set attributes) {
        this(uw, "", attributes);
    }
    
    public UWNode(UniversalWord uw, String instanceNumber, Set attributes) {
        super(attributes);
        this.uw = uw;
        this.instanceNumber = instanceNumber;
    }

    public UniversalWord getUW() {
        return this.uw;
    }

    public void setUW(UniversalWord uw) {
        this.uw = uw;
    }

    public String getInstanceNumber() {
        return this.instanceNumber;
    }

    public String getNodeId() {
        return this.uw.toString() + this.instanceNumber;
    }
    
    public String toString() {
        StringBuffer out = new StringBuffer();
        this.outputTo(out);
        return out.toString();
    }

    public void outputTo(StringBuffer out) {
        this.uw.outputTo(out);
        out.append(this.instanceNumber);
        this.outputAttributesTo(out);
    }

    public boolean equals(Object anObject) {
        if (anObject == this) return true;
        if (anObject == null) return false;
        if (anObject.getClass() != this.getClass()) return false; 
        return this.toString().equals(((UWNode) anObject).toString());
    }

    public int hashCode() {
        return this.toString().hashCode();
    }

    public void writeDot(Writer out) throws java.io.IOException {
        out.write(String.valueOf(this.getNodeNumber()));
        out.write(" [\n");
        out.write("label=\"" + this.getNodeId() + "\\n");
        
        Set attrs = this.getAttributes();
        if (attrs != null && ! attrs.isEmpty()) {
            Iterator iter = attrs.iterator();
            while(iter.hasNext()) {
                out.write((String) iter.next());
            }
        }
        
        out.write("\"\n");
        out.write("shape=\"box\"\n");
        if (this.isEntryNode()) out.write("fontname=\"courb\"\n");
        out.write("];\n");
    }
    
    
}
