/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: Graph.java,v 1.7 2003/12/10 16:50:02 serasset Exp $
 *-----------------------------------------------------------
 * $Log: Graph.java,v $
 * Revision 1.7  2003/12/10 16:50:02  serasset
 * We now draw a complete document.
 *
 * Revision 1.6  2003/11/25 16:08:28  serasset
 * Relations in scope are not duplicated in graphe drawing anymore.
 * Lexical errors in dictionary entries are caught and displayed to the user.
 *
 * Revision 1.5  2003/11/21 12:48:21  serasset
 * Collect more exceptions to get a result when deconverting
 *
 * Revision 1.4  2003/11/20 16:04:56  serasset
 * Suppressed a bug when an unknown unl attribute was used.
 * Added support for UNL docs without [D] ... [/D] labels.
 * Added support for isolated entry UW in graph [W] UW [W] syntax.
 *
 * Revision 1.3  2003/11/14 17:53:55  serasset
 * Handling of subgraphs is now correct.
 *
 * Revision 1.2  2003/10/15 16:39:22  serasset
 * Detection of non connected graphs.
 * Resulting tree is now sent to the user.
 *
 * Revision 1.1.1.1  2003/10/13 14:01:13  serasset
 * Creation of unldeco CVS repository.
 *
 */
package fr.imag.clips.papillon.business.pivax.unl_graph;

import fr.imag.clips.papillon.business.pivax.unl_graph.NoEntryNodeException;

import com.lutris.dods.builder.generator.query.DataObjectException;
import com.lutris.dods.builder.generator.query.QueryException;

import java.util.Collection;
import java.util.Vector;
import java.util.Hashtable;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeSet;

import java.lang.StringBuffer;
import java.io.StringWriter;
import java.io.Writer;

public class Graph {

    private Vector nodes;
    private Vector relations;
    private int nodeCounter = 0;
    
    public Graph() {
        this(new Vector(), new Vector());
    }
    
    public Graph(Collection nodes, Collection relations) {
        this(new Vector(nodes), new Vector(relations));
    }

    // WARNING uws will be wrong...
    public Graph(Vector nodes, Vector relations) {
        this.nodes = nodes;
        this.relations = relations;
    }

    public Collection getNodes() {
        return (Collection) this.nodes;
    }
    
    public Collection getRelations() {
        return (Collection) this.relations;
    }

    public GraphNode getEntryNode() throws NoEntryNodeException {
        return getEntryNode("");
    }
    
    public GraphNode getEntryNode(String graphNumber) throws NoEntryNodeException {
        int i = 0;
        GraphNode gr = null;
        while ((i < relations.size()) && ((gr = getEntryNode((GraphRelation) relations.get(i), graphNumber)) == null)) {
            i++;
        }
        // we may not have found it, because, the entry node is isolated (no relation...)
        if ((gr == null) && (1 == this.nodes.size()) && ((GraphNode) this.nodes.get(0)).isEntryNode()) {
            gr = (GraphNode) this.nodes.get(0);
        }
            
        // Will make a particular exception
        if (gr == null) throw new NoEntryNodeException("No Entry node for subgraph " + graphNumber);
        return gr;
    }

    public static GraphNode getEntryNode(GraphRelation rel, String graphNumber) {
        String str = rel.getSubGraphReferenceNumber();
        GraphNode n = null;
        if (str.equals(graphNumber)) {
            if (rel.getNode1().isEntryNode()) n = rel.getNode1();
            if (rel.getNode2().isEntryNode()) n = rel.getNode2();
        }
        return n;
    }

    
    public static GraphNode getEntryNode(GraphRelation rel) {
        return getEntryNode(rel, "");
    }

    public ArrayList getRemainingRelations(HashSet alreadyProcessedRelations, String scope) {
        GraphRelation gr = null;
        ArrayList rels = new ArrayList(relations.size());
        
        for (int i=0; i < relations.size(); i++) {
            GraphRelation crel = (GraphRelation) relations.get(i);
            if (crel.getSubGraphReferenceNumber().equals(scope)
                && ! alreadyProcessedRelations.contains(crel)) {
                rels.add(crel);
            }
        }
        return rels;
    }
    
    public GraphRelation getUnprocessedRelation(HashSet alreadyProcessedRelations,
                                                Hashtable graphNode2treeNode,
                                                String scope) {
        int i = 0;
        GraphRelation gr = null;
        while ((i < relations.size())) {
            GraphRelation crel = (GraphRelation) relations.get(i);
            GraphNode orig = crel.getNode1();
            if (crel.getSubGraphReferenceNumber().equals(scope) 	// isInCorrectScope
                && (! alreadyProcessedRelations.contains(crel)) 	//(! alreadyProcessed)
                && graphNode2treeNode.containsKey(orig) 		//origIsInTree
                ) break; // We found a relation...
            i++;
        }
        if (i == relations.size()) {
            return null;
        } else {
            return (GraphRelation) relations.get(i);
        }
    }

    public GraphRelation getUnprocessedInverseRelation (HashSet alreadyProcessedRelations,
                                                        Hashtable graphNode2treeNode,
                                                        String scope) {
        int i = 0;
        GraphRelation gr = null;
        while ((i < relations.size())) {
            GraphRelation crel = (GraphRelation) relations.get(i);
            GraphNode dest = crel.getNode2();
            if (crel.getSubGraphReferenceNumber().equals(scope) 	// isInCorrectScope
                && (! alreadyProcessedRelations.contains(crel)) 	//(! alreadyProcessed)
                && graphNode2treeNode.containsKey(dest) 		//destIsInTree
                ) break; // We found a relation...
            i++;
        }
        if (i == relations.size()) {
            return null;
        } else {
            return (GraphRelation) relations.get(i);
        }
    }
    
    
    /**
     * Adds the given relation to the set of relations, adding its nodes if necessary.
     * When a node of the relation is already in the graph, the relation is modified
     * in order to avoid node duplication.
     * <br><b>i.e.</b> rn being a relation node, if a node gn exists such that rn.equals(gn),
     * the relation now refers to gn and rn is lost.
     * <p><b> this may be a mistake if nodes bearing different attributes are to be considered as equal...
     */
    public void addRelation(GraphRelation r) {
        int i1 = nodes.indexOf(r.getNode1());
        int i2 = nodes.indexOf(r.getNode2());

        if (-1 == i1) {
            this.addNode(r.getNode1());
            nodeCounter++;
            r.getNode1().setNodeNumber(nodeCounter);
        } else {
            r.setNode1((GraphNode) nodes.get(i1));
        }
        if (-1 == i2) {
            this.addNode(r.getNode2());
            nodeCounter++;
            r.getNode2().setNodeNumber(nodeCounter);
        } else {
            r.setNode2((GraphNode) nodes.get(i2));
        }
        relations.add(r);
    }

    /**
        * Adds the given node in the graph
     */
    public void addNode(GraphNode n) {
        nodes.add(n);
    }

    public String toString() {
        StringBuffer out = new StringBuffer();
        this.outputTo(out);
        return out.toString();
    }

    public void outputTo(StringBuffer out) {
        if ((relations.size() == 0) && (nodes.size() != 0)) {
            out.append("[W]");
            outputNodesTo(out);
            out.append("[/W]");
        } else {
            outputRelationsTo(out);
        }
    }

    public void outputRelationsTo(StringBuffer out) {
        for (int i=0; i<relations.size(); i++) {
            ((GraphRelation) relations.get(i)).outputTo(out);
            out.append('\n');
        }
    }
    
    public void outputNodesTo(StringBuffer out) {
        for (int i=0; i<nodes.size(); i++) {
            ((GraphNode) nodes.get(i)).outputTo(out);
            out.append('\n');
        }
    }

    /////////////////////////////////////
    // DOT Format
    /////////////////////////////////////
    public String toDotString() throws NoEntryNodeException {
        StringWriter out = new StringWriter();
        try {
            this.writeDot(out);
        } catch (java.io.IOException e) {
            System.out.println("IOException while exporting Graph to dot format.");
        }
        return out.toString();
    }

    public void writeDot(Writer out) throws java.io.IOException, NoEntryNodeException {
        out.write("digraph G {\n");
        out.write("graph [fontname=\"Courier\", compound=\"true\"];\n");
        
        for (int i=0; i<nodes.size(); i++) {
            GraphNode n = (GraphNode) nodes.get(i);
            if (n instanceof UWNode) n.writeDot(out);
            out.write('\n');
        }
        
        // get all scopes in the graph
        TreeSet scopes = new TreeSet();
        for (int i=0; i<relations.size(); i++) {
            scopes.add(((GraphRelation) relations.get(i)).getSubGraphReferenceNumber());
        }
        scopes.remove("");
        
        Iterator iter = scopes.iterator();
        while (iter.hasNext()) {
            writeClusterDot(out, (String) iter.next());
        }
        
        // Relations for the main graph
        for (int i=0; i<relations.size(); i++) {
            GraphRelation gr = (GraphRelation) relations.get(i);
            if (gr.getSubGraphReferenceNumber().equals("")) {
                gr.writeDot(out, this);
                //out.write('\n');
            }
        }
        out.write("}\n");

    }

    public void writeClusterDot(Writer out, String scope) throws java.io.IOException, NoEntryNodeException {
        out.write("subgraph cluster_" + scope.substring(1) + "{\n");
        out.write("   color = black;\n");
        out.write("   label = \"" + scope + "\";\n");
        // write this subgraph relations...
        for (int i=0; i<relations.size(); i++) {
            GraphRelation rel =  (GraphRelation) relations.get(i);
            if (rel.getSubGraphReferenceNumber().equals(scope)) rel.writeDot(out, this);
            out.write('\n');
        }
        
        out.write("}\n");
        
    }
    
}
