/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: GraphNode.java,v 1.1.1.1 2003/10/13 14:01:13 serasset Exp $
 *-----------------------------------------------------------
 * $Log: GraphNode.java,v $
 * Revision 1.1.1.1  2003/10/13 14:01:13  serasset
 * Creation of unldeco CVS repository.
 *
 */
package fr.imag.clips.papillon.business.pivax.unl_graph;

import java.util.TreeSet;
import java.util.Set;
import java.util.Iterator;
import java.lang.StringBuffer;
import java.io.Writer;

public abstract class GraphNode {
    private Set attributes;
    private int nodeNumber;

    public GraphNode(Set attributes) {
        this.attributes = attributes;
    }

    public GraphNode(Set attributes, int nb) {
        this(attributes);
        this.nodeNumber = nb;
    }

    public void setNodeNumber(int nb) {
        this.nodeNumber = nb;
    }

    public int getNodeNumber() {
        return this.nodeNumber;
    }
    
    public abstract String getNodeId() ;

    public Set getAttributes() {
        return this.attributes;
    }

    public boolean isEntryNode() {
        return this.attributes.contains(".@entry");
    }
    
    public abstract String toString() ;
    public abstract void outputTo(StringBuffer out) ;

    public void outputAttributesTo(StringBuffer out) {
        if (this.attributes != null && ! this.attributes.isEmpty()) {
            Iterator iter = this.attributes.iterator();
            while(iter.hasNext()) {
                out.append((String) iter.next());
            }
        }
    }

    public abstract void writeDot(Writer out) throws java.io.IOException;

}
