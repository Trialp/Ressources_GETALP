package fr.imag.clips.papillon.presentation;

import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Element;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;

public class HTMLTransformFactory {

    // Public construcutor
    public HTMLTransformFactory() {
	// Nothing for the moment..
    }
    
    /** 
     * Creates a new <DIV> node wich has the same childs and attributes that the given <BODY> node.
     * The created node is owned by the given Document.
     * Some attributes that are defined for <BODY> and not for <DIV> tags are removed:
     * For example onload or bgcolor
     *
     * @param layout the Document
     * @param bodyNode the <BODY> node wich content is copied into the new <DIV> node
     *
     * @return the created <DIV> node
     */
    public Node mkDivFromBody(Document layout, Element docBody) {
	Element bodyDiv = layout.createElement("div");
	// remove attributes not defined for a <DIV> tag
	docBody.removeAttribute("bgcolor");
     	  // following attributes should be added to layout <body> tag in order to execute scripts
	  // we may have to correct URIs in the values of these attributes...
	docBody.removeAttribute("onload");
	docBody.removeAttribute("onunload");
	NodeList bodyChilds = docBody.getChildNodes();
	for (int i = 0; i < bodyChilds.getLength(); i++) {
	    Node bodyNode = layout.importNode(bodyChilds.item(i), true);
	    bodyDiv.appendChild(bodyNode);
	}
	// add attributes
	NamedNodeMap bodyAttrs = docBody.getAttributes();
	for (int i = 0; i < bodyAttrs.getLength(); i++) {
	    Attr attr = (Attr)layout.importNode(bodyAttrs.item(i), true);
	    bodyDiv.setAttributeNode(attr);
	}
	return bodyDiv;
    }
}
