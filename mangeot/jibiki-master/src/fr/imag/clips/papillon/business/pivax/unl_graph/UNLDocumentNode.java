/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: UNLDocumentNode.java,v 1.1.1.1 2003/10/13 14:01:13 serasset Exp $
 *-----------------------------------------------------------
 * $Log: UNLDocumentNode.java,v $
 * Revision 1.1.1.1  2003/10/13 14:01:13  serasset
 * Creation of unldeco CVS repository.
 *
 */
package fr.imag.clips.papillon.business.pivax.unl_graph;

import java.lang.StringBuffer;

public class UNLDocumentNode {
    public static final int PARAGRAPH_START = 0;
    public static final int PARAGRAPH_END = 1;
    public static final int TITLE = 2;
    public static final int SENTENCE = 3;

    private int kindOfNode;
    private String docNodeLabel;
    private Graph graph;
    
    private String error;

    public UNLDocumentNode(int kind) {
        this(kind, "", null);
    }
    
    public UNLDocumentNode(String error) {
    	this.error = error;
    }

    public UNLDocumentNode(int kind, String label) {
        this(kind, label, null);
    }

    public UNLDocumentNode(int kind, String label, Graph graph) {
        this.kindOfNode = kind;
        this.docNodeLabel = label;
        this.graph = graph;
    }

    public int getKind() {
        return this.kindOfNode;
    }
    
    public String getNodeLabel() {
        return this.docNodeLabel;
    }
    
    public Graph getGraph() {
        return this.graph;
    }
    
    public void setGraph(Graph graph) {
        this.graph = graph;
    }
    
    public String getError() {
    	return this.error;
    }
    
    public void setError(String error) {
    	this.error = error;
    }

    public String toString() {
        StringBuffer out = new StringBuffer();
        this.outputTo(out);
        return out.toString();
    }

    public void outputTo(StringBuffer out) {
        if (PARAGRAPH_START == this.kindOfNode) {
            out.append(this.docNodeLabel);
            out.append('\n');
        } else if (PARAGRAPH_END == this.kindOfNode) {
            out.append("[/P]");
            out.append('\n');
        } else if (TITLE == this.kindOfNode) {
            out.append(this.docNodeLabel);
            out.append('\n');
            if (this.graph != null) { this.graph.outputTo(out); }
            out.append("[/T]");
            out.append('\n');
        } else if (SENTENCE == this.kindOfNode) {
            out.append(this.docNodeLabel);
            out.append('\n');
            if (this.graph != null) { this.graph.outputTo(out); }
            out.append("[/S]");
            out.append('\n');
        } else if (this.error != null && !this.error.equals("")) {
        	out.append("[S]");
        	out.append("[ERROR]");
        	out.append(this.error);
        	out.append("[/ERROR]");
        	out.append("[/S]");
        	out.append('\n');
        }
    }
}
