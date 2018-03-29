/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: UNLDocument.java,v 1.1.1.1 2003/10/13 14:01:13 serasset Exp $
 *-----------------------------------------------------------
 * $Log: UNLDocument.java,v $
 * Revision 1.1.1.1  2003/10/13 14:01:13  serasset
 * Creation of unldeco CVS repository.
 *
 */
package fr.imag.clips.papillon.business.pivax.unl_graph;

import java.lang.StringBuffer;
import java.util.Vector;
import java.util.Collection;
import java.util.Hashtable;


public class UNLDocument {
    private String docLabel;
    private Vector docElements;
    private String error;
    
    public UNLDocument() {
        this("[D]", new Vector());
    }

    public UNLDocument(String docLabel) {
        this(docLabel, new Vector());
    }

    public UNLDocument(String docLabel, Vector elements) {
        this.docLabel = docLabel;
        this.docElements = elements;
    }
    
    public UNLDocument(String docLabel, Collection elements) {
        this(docLabel, new Vector(elements));
    }


    public Vector getElements() {
        return this.docElements;
    }

    public String getDocumentLabel() {
        return this.docLabel;
    }
    
    public String getError() {
    	return this.error;
    }
    
    public void setError(String error) {
    	this.error = error;
    }

    public void addElement(UNLDocumentNode node) {
        this.docElements.add(node);
    }

    public String toString() {
        StringBuffer out = new StringBuffer();
        this.outputTo(out);
        return out.toString();
    }

    public void outputTo(StringBuffer out) {
        out.append(this.docLabel);
        if (this.error != null && !this.error.equals("")) {
        	out.append("[ERROR]");
        	out.append(this.error);
        	out.append("[/ERROR]");
        }
        out.append('\n');
        for (int i = 0; i < this.docElements.size(); i++) {
            ((UNLDocumentNode) this.docElements.get(i)).outputTo(out);
        }
        out.append("[/D]");
        out.append('\n');
    }

   /* public Hashtable getDocumentDictionary() {
        Hashtable dic = new Hashtable();
        for (int i = 0; i < this.docElements.size(); i++) {
            System.out.println("Docnode:" + this.docElements.get(i));
            UNLDocumentNode currentNode = (UNLDocumentNode) this.docElements.get(i);
            if (currentNode.getKind() == UNLDocumentNode.SENTENCE ||
                currentNode.getKind() == UNLDocumentNode.TITLE ) {
                currentNode.getGraph().completeDictionary(dic);
            }
        }
        return dic;
    }
   */
    
}