/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: GraphRelation.java,v 1.3 2003/12/10 16:50:02 serasset Exp $
 *-----------------------------------------------------------
 * $Log: GraphRelation.java,v $
 * Revision 1.3  2003/12/10 16:50:02  serasset
 * We now draw a complete document.
 *
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
import java.io.Writer;


public class GraphRelation {
    private GraphNode node1;
    private GraphNode node2;
    private String relationLabel;
    private String subGraphReferenceNumber;

    public GraphRelation(String relationLabel, GraphNode n1, GraphNode n2) {
        this(relationLabel, "", n1, n2);
    }

    public GraphRelation(String relationLabel, String refnum, GraphNode n1, GraphNode n2) {
        this.relationLabel = relationLabel;
        this.subGraphReferenceNumber = refnum;
        this.node1 = n1;
        this.node2 = n2;
    }

    public String getRelationLabel() {
        return this.relationLabel;
    }

    public String getSubGraphReferenceNumber() {
        return this.subGraphReferenceNumber;
    }

    public GraphNode getNode1() {
        return this.node1;
    }

    public GraphNode getNode2() {
        return this.node2;
    }
    
    public void setNode1(GraphNode n) {
        this.node1 = n;
    }

    public void setNode2(GraphNode n) {
        this.node2 = n;
    }
    
    public String toString() {
        StringBuffer out = new StringBuffer();
        this.outputTo(out);
        return out.toString();
    }

    public void outputTo(StringBuffer out) {
        out.append(this.relationLabel);
        out.append(this.subGraphReferenceNumber);
        out.append('(');
        out.append(this.node1);
        out.append(',');
        out.append(this.node2);
        out.append(')');
    }

    public void writeDot(Writer out, Graph gr) throws java.io.IOException, NoEntryNodeException {
        GraphNode n1 = this.getNode1();
        GraphNode n2 = this.getNode2();

        if (n1 instanceof SubGraphReferenceNode) {
            out.write(String.valueOf(gr.getEntryNode(((SubGraphReferenceNode) n1).getSubGraphReferenceNumber()).getNodeNumber()));
        } else {
            out.write(String.valueOf(n1.getNodeNumber()));
        }
        out.write(" -> ");
        if (n2 instanceof SubGraphReferenceNode) {
            out.write(String.valueOf(gr.getEntryNode(((SubGraphReferenceNode) n2).getSubGraphReferenceNumber()).getNodeNumber()));
        } else {
            out.write(String.valueOf(n2.getNodeNumber()));
        }
        
        out.write(" [label=\"" + this.getRelationLabel() + "\" ");
        if (n1 instanceof SubGraphReferenceNode) {
            out.write(" ltail=\"cluster_" + ((SubGraphReferenceNode) n1).getSubGraphReferenceNumber().substring(1) + "\"");
        }
        if (n2 instanceof SubGraphReferenceNode) {
            out.write(" lhead=\"cluster_" + ((SubGraphReferenceNode) n2).getSubGraphReferenceNumber().substring(1) + "\"");
        }
        out.write("];\n");
    }
    
}
