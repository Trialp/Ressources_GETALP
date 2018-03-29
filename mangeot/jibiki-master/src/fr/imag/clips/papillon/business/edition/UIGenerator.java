/*
 *  papillon
 *
 *  Created by Mathieu Mangeot on Thu Sep 16 2004.
 *  Copyright (c) 2002 GETA_CLIPS_IMAG. All rights reserved.
 *
 *---------------------------------------------------------
 * $Id$
 *---------------------------------------------------------
 * $Log$
 * Revision 1.28  2006/12/14 20:03:26  fbrunet
 * Add method to normalize value into XML structure.
 *
 * Revision 1.27  2006/12/13 09:32:00  fbrunet
 * *** empty log message ***
 *
 * Revision 1.26  2006/11/21 22:51:55  fbrunet
 * Correct UIGenerator bug and another minor bugs
 *
 * Revision 1.25  2006/11/09 09:04:42  fbrunet
 * *** empty log message ***
 *
 * Revision 1.24  2006/09/11 19:57:48  fbrunet
 * - bug correction : interface edition (link axie to another axi)
 *
 * Revision 1.23  2006/08/10 22:17:12  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.22  2006/05/05 02:08:23  fbrunet
 * bug correction : url utf8 transfert (in createEntryInit)
 *
 * Revision 1.21  2006/04/24 13:43:29  fbrunet
 * Add new class ViewQueryResult : allow to use one class to create result display in advancedSearch and EditEntryInit (like advancedQueryForm)
 * Improve result display : view n results per page
 *
 * Revision 1.20  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 * Revision 1.19  2006/03/27 10:23:03  mangeot
 * Fixed the moveElements method
 *
 * Revision 1.18  2006/03/27 09:26:44  mangeot
 * Bug fix in moveElements
 *
 * Revision 1.17  2006/03/13 08:48:00  fbrunet
 * bug corrections before merge
 *
 * Revision 1.16  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.15  2006/02/18 17:57:21  mangeot
 * MM: Fixed a problem when, in the editor, the last element of a list was deleted. 
 * The user still could enter data but it was not saved because the XML structure was modified.
 * Now, I check if a list still has an element.
 *
 * Revision 1.12  2005/10/04 10:21:24  mangeot
 * Experimental implementation of moving up and down blocks
 *
 * Revision 1.11.2.2  2006/01/24 13:39:49  fbrunet
 * Modification view management
 * Modification LexALP postprocessing
 *
 * Revision 1.14  2005/11/21 14:16:29  mangeot
 * *** empty log message ***
 *
 * Revision 1.13  2005/11/16 16:52:43  mangeot
 * *** empty log message ***
 *
 * Revision 1.12  2005/10/04 10:21:24  mangeot
 * Experimental implementation of moving up and down blocks
 *
 * Revision 1.11  2005/08/01 15:03:41  mangeot
 * Corrected an important bug in the editor that forbidded to change a boolean value from true to false.
 * Beware, you have to edit the existing interface templates by hands:
 * 1- duplicate all the input elements with name='boolean' and type='checkbox'.
 * - for each input element pair,
 *  2- change one input element name into name='booleantrue'
 *  3- change the other input element type to type='hidden'
 *
 * Revision 1.10  2005/07/16 12:58:31  serasset
 * Added limit parameter to query functions
 * Added a parameter to Formater initializations
 * Developped a new Advanced search functionality with reusable code for the query form handling...
 *
 * Revision 1.9  2005/06/16 16:09:17  mangeot
 * *** empty log message ***
 *
 * Revision 1.8  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *
 *---------------------------------------------------------
 */

package fr.imag.clips.papillon.business.edition;

// utils
import java.util.Vector;

// internal imports
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.xml.XMLServices;


// DOM elements
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

//
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class UIGenerator {
	
	
	// constants
	
	public static final String NEW_BLOCK_ANCHOR = "NewBlock";
	public static final String TYPE_SEP_ANCHOR = ".";
	public static final String TYPE_PARENT_ANCHOR = "parent";

	public static final String ID_SEPARATOR = ".";
	public static final String ATTR_SEPARATOR = ".@";
	public static final String PARAMETERS_SEPARATOR = "+";
	public static final String ITF_ELT_BLOCK_NAME = "div";
	public static final String ITF_ELT_INLINE_NAME = "span";
	public static final String ITF_ELT_DUPLICATE_NAME = "tr";
	public static final String ITF_ATTR_NAME = "class";
	public static final String ITF_HIDDEN_STYLE = "hidden";
	public static final String SELECT_ATTR_NAME = "select";
	public static final String CHOOSE_ATTR_NAME = "choose";
	public static final String BOOLEAN_ATTR_NAME = "boolean";
	public static final String BOOLEAN_TRUE_ATTR_NAME = "booleantrue";
	public static final String TYPE_ATTR_NAME = "type";
	
	public static final String CHOICE_NODE_NAME = "xsd:choice";
    
    // WARN: patterns depend on id and attr separators !
    protected static final String attributePatternString = "^([^\\.]+)\\.(\\d+)\\.@(.+)$";
    protected static Pattern attributePattern = Pattern.compile(attributePatternString);
    protected static Matcher attributeMatcher = attributePattern.matcher("");
	
    protected static final String nodePatternString = "^([^\\.]+)\\.(\\d+)$";
    protected static Pattern nodePattern = Pattern.compile(nodePatternString);
    protected static Matcher nodeMatcher = nodePattern.matcher("");
    
	
	/******************************************************/
	/* Public methods                                     */
	/******************************************************/
	
	public static void fillInterfaceTemplate(Element entryElt, Element rootItfElt, Element itfTemplate) {
		Element correspItf = UIGenerator.findCorrespondingElement(entryElt, rootItfElt);
		UIGenerator.fillTemplate(entryElt, correspItf, itfTemplate);
	}
	
	public static String addElement(String elementName, String parentId, Element entryElt, Element entryTemplate, String[] siblingIds) {
		//PapillonLogger.writeDebugMsg("addElement: " + elementName + " parent: " + parentId + " entry: " + entryElt + " entryTemplate: " + entryTemplate);
        
        //
        if (siblingIds != null) {
            for (int i=0; i < siblingIds.length; i++) {
                //PapillonLogger.writeDebugMsg("siblingIds: " + siblingIds[i]);
            }
        }
        
        //
		Element siblingElement = null;
		if (siblingIds != null && siblingIds.length>0) {
			String siblingId = siblingIds[siblingIds.length-1]; 
            nodeMatcher.reset(siblingId);
            if (nodeMatcher.find()) {
                String tagName = nodeMatcher.group(1);
                String tagNum = nodeMatcher.group(2);
				if (tagName.equals(elementName)) {
					siblingElement = findElementInEntry(tagName, tagNum, entryElt);
				}
			}
		}
        
		nodeMatcher.reset(parentId);
        if (nodeMatcher.find()) {
            String parentName = nodeMatcher.group(1);
            String parentNb = nodeMatcher.group(2);
			
			Element parentElt = findElementInEntry(parentName, parentNb, entryElt);
			if (parentElt!=null) {
				Element myElement = getTemplateEntryElement(elementName,parentName,entryTemplate);
				return insertEntryElement(parentElt,myElement,siblingElement);
			}
		}
		return null;
	}
	
	public static String chooseElement(String elementId, String parentId, Element entryElt, Element entryTemplate) {
		//PapillonLogger.writeDebugMsg("chooseElement: " + elementId + " parent: " + parentId);	
		String found = null;
        
		nodeMatcher.reset(parentId);
        if (nodeMatcher.find()) {
            String parentName = nodeMatcher.group(1);
            String parentNb = nodeMatcher.group(2);
            		
			Element resultParent = findElementInEntry(parentName, parentNb, entryElt);
			if (resultParent!=null) {
                nodeMatcher.reset(elementId);
                if (nodeMatcher.find()) {
                    String elementName = nodeMatcher.group(1);
                    String elementNb = nodeMatcher.group(2);
					Element templateElt = getTemplateEntryElement(elementName,parentName,entryTemplate);
					if (templateElt!=null) {
						Node chooseNode = getChildNode(resultParent,CHOICE_NODE_NAME);
						Node resNode = resultParent.getOwnerDocument().importNode(templateElt,true);
						resultParent.replaceChild(resNode,chooseNode);
						found = createId(resNode,null);
					}
				}
			}
		}
		return found;
	}
	
	
	// Update an element in the XML entry entryElt with the new value value
	public static boolean updateElement(String elementId, String value, Element entryElt)
    throws PapillonBusinessException {

        // Normalized value
        value = VolumeEntriesFactory.normalizeValue(value);
        
        //convert element name into UTF-8
        try {
            byte[] pbytes = elementId.getBytes("ISO-8859-1");
            elementId = new String(pbytes, "UTF-8");
          } catch (java.io.UnsupportedEncodingException e) {
            throw new PapillonBusinessException("UTF-8 encoding is not supported on this plateform.", e);
        }

 
        //
        attributeMatcher.reset(elementId);
        nodeMatcher.reset(elementId);
        Element res = null;
        if (attributeMatcher.find()) {
            String tagName = attributeMatcher.group(1);
            String tagNum = attributeMatcher.group(2);
            String attributeName = attributeMatcher.group(3);
            Element resultElt = findElementInEntry(tagName, tagNum, entryElt);
            if (resultElt!=null) {
                // PapillonLogger.writeDebugMsg("updateElement: " + elementId + " call setAttribute " + attributeName + " value: [" + value + "]");
                resultElt.setAttribute(attributeName,value);
                return true;
            }
        } else if (nodeMatcher.find()) {
            String tagName = nodeMatcher.group(1);
            String tagNum = nodeMatcher.group(2);
            Element resultElt = findElementInEntry(tagName, tagNum, entryElt);
            if (resultElt!=null) { 
                // remove all children.
                while(resultElt.hasChildNodes()) {
                    resultElt.removeChild(resultElt.getFirstChild());
                }
                Document documentValue = null;
                if (value.indexOf('<')>=0 && value.indexOf('/')>0 && value.indexOf('>')>1) {
                    try {
                        documentValue = XMLServices.buildDOMTree("<?xml version='1.0' ?><root>"+value+"</root>");
                    }
                    catch (Exception ex) {
                    }
                }
                if (documentValue != null) {
                    NodeList myNodeList = documentValue.getDocumentElement().getChildNodes();
                    for (int i = 0; i < myNodeList.getLength (); i++) {
                        Node nodeItem = myNodeList.item(i);
                        resultElt.appendChild(resultElt.getOwnerDocument().importNode(nodeItem,true));
                    }
                }
                else {
                    Node textNode = resultElt.getOwnerDocument().createTextNode(value);
                    resultElt.appendChild(textNode);
                }
                return true;
            }
        }
        return false;
	}
	
	public static String deleteElements(String elementName, String parentId, String[] elementIds, Element entryElt, Element entryTemplate) {
		String result = null;
		NodeList myNodeList = entryElt.getElementsByTagName (elementName);
		Vector removeNodes = new Vector();
		for (int i=0; i<elementIds.length;i++) {
			String elementId = elementIds[i];
			//PapillonLogger.writeDebugMsg("deleteElement: " + elementName + " eltId: " + elementIds[0]);            
			nodeMatcher.reset(elementId);
            if (nodeMatcher.find()) {
                String currentName = nodeMatcher.group(1);
                String currentNb = nodeMatcher.group(2);
                
                if (elementName!=null && elementName.equals(currentName)) {
                    int itemNb = Integer.parseInt(currentNb);
                    Element resultElt = (Element) myNodeList.item(itemNb);
                    removeNodes.add(resultElt);
                }
            }
        }
		Node parentNode = null;
		Node lastSibling = null;
		for (int i=0; i<removeNodes.size();i++) {
			Element removeElt = (Element) removeNodes.elementAt(i);
			parentNode = removeElt.getParentNode();
			lastSibling = removeElt.getPreviousSibling();
			parentNode.removeChild(removeElt);
		}
		if (parentNode !=null && parentNode.getNodeType()==Node.ELEMENT_NODE) {
			// If the parent has no more children, add a new child.
			// There must always be at least one item in a list 
			int i=0;
			boolean oneChild = false;
			NodeList childNodes = parentNode.getChildNodes();
			while (i<childNodes.getLength() && !oneChild) {
				Node tmpNode = childNodes.item(i);
				if (tmpNode.getNodeType() == Node.ELEMENT_NODE) {
					Element currentElt = (Element) tmpNode;
					oneChild = currentElt.getTagName().equals(elementName);
				}
				i++;
			}
			if (!oneChild) {
				result = addElement(elementName, parentId, entryElt, entryTemplate, null);
			}
			else {
				result = createId(lastSibling,null);
			}
		}
		return result;
	}
	
	public static String moveElementsUp(String elementName, String parentId, String[] elementIds, Element entryElt) {
		return moveElements(elementName, parentId, elementIds, entryElt, true);
	}
	
	public static String moveElementsDown(String elementName, String parentId, String[] elementIds, Element entryElt) {
		return moveElements(elementName, parentId, elementIds, entryElt, false);
	}
	
	public static String moveElements(String elementName, String parentId, String[] elementIds, Element entryElt, boolean up) {
		String result = null;
		//PapillonLogger.writeDebugMsg("moveElement: up " + up + " name: "  + elementName + " parentId: " + parentId + " eltid1: " + elementIds[0]);
		
		/* looking for the targeted elements */
		NodeList myNodeList = entryElt.getElementsByTagName (elementName);
		Vector moveElements = new Vector();
		for (int i=0; i<elementIds.length; i++) {
			String elementId = elementIds[i];
			nodeMatcher.reset(elementId);
			if (nodeMatcher.find()) {
				String currentName = nodeMatcher.group(1);
				String currentNb = nodeMatcher.group(2);
				if (elementName!=null && elementName.equals(currentName)) {
					int itemNb = Integer.parseInt(currentNb);
					Element resultElt = (Element) myNodeList.item(itemNb);
					moveElements.add(resultElt);
				}
			}
		}
		/* looking for the parent element */
		nodeMatcher.reset(parentId);
        if (nodeMatcher.find()) {
            String parentName = nodeMatcher.group(1);
            String parentNb = nodeMatcher.group(2);
			
			Element parentElt = findElementInEntry(parentName, parentNb, entryElt);
			/* moving up targeted element */
			if (parentElt!=null && moveElements !=null) {
				for (java.util.Iterator moveIterator = moveElements.iterator(); moveIterator.hasNext();) {
					Element moveElement = (Element) moveIterator.next();
					NodeList myChildrenList = parentElt.getChildNodes();
					Element beforeElement = null;
					Element afterElement = null;
					int i=0;
					result = null;
					while (result == null && myChildrenList != null && i<myChildrenList.getLength()) {
						Node tmpNode = myChildrenList.item(i);
						if (tmpNode.getNodeType() == Node.ELEMENT_NODE) {
							Element currentElt = (Element) tmpNode;
							if (currentElt.getTagName().equals(elementName)) {
								if (currentElt.equals(moveElement)) {
									if (up) {
										afterElement = currentElt; 
									}
									else {
										beforeElement = currentElt; 
										afterElement = null;
									}
								}
								else {
									if (up) {
										beforeElement = currentElt;
										afterElement = null;
									}
									else {
										afterElement = currentElt; 
									}
								}
								if (result == null && beforeElement != null && afterElement != null) {
									//PapillonLogger.writeDebugMsg("swapElements: " + afterElement.getTagName());
									parentElt.removeChild(afterElement);
									parentElt.insertBefore(afterElement,beforeElement);
									result = createId(afterElement,null);
								}
							}
						}
						i++;
					}
				}
			}
		}
		return result;
	}
	
	public static boolean setValueInput(Element itfElt, String correspName, String value) {
		//PapillonLogger.writeDebugMsg("setValueInput: " + correspName);
		boolean found = false;
        if (null != itfElt) {
            NodeList myNodeList = itfElt.getElementsByTagName ("input");
            int i=0;
            while (i<myNodeList.getLength () && !found) {
                Element currentElt = (Element) myNodeList.item(i);
                String name = currentElt.getAttribute("name");
                if (name !=null && name.equals(correspName)) {
                    String type = currentElt.getAttribute("type");
                    currentElt.setAttribute("value", value);
                    found = true;
                }	
                i++;
            }
        }
		return found;
	}
	
	
	
	/******************************************************/
	/* protected methods								  */
	/******************************************************/
	
	protected static void fillTemplate(Node entryNode, Element itfElt,Element itfTemplate) {
		fillTemplate(entryNode, itfElt, itfTemplate, "");
	}
	
	protected static void fillTemplate(Node entryNode, Element itfElt, Element itfTemplate, String oldId) {
		if (itfElt !=null) {
			String newId = createId(entryNode, oldId);
			String entryNodeName = entryNode.getNodeName();
			if (entryNode.getNodeType()==Node.ATTRIBUTE_NODE) {
				entryNodeName = ((Attr) entryNode).getOwnerElement().getNodeName() + ATTR_SEPARATOR + entryNodeName;
			}
            //PapillonLogger.writeDebugMsg("fillTemplate Node: " + entryNodeName + " ID: " + newId);
			setIdCorrespondingAnchor(entryNodeName, itfElt, newId);
			setIdCorrespondingSubmitInputs(entryNodeName, itfElt, newId);
			setIdCorrespondingLabel(entryNodeName, itfElt, newId);
			setIdCorrespondingSelectCheckbox(entryNodeName,itfElt, newId);
            setClassCorrespondingLinkers(entryNodeName,itfElt, newId);

            
			// special handling of the xsd:choice schema directive
			Node parentNode = entryNode.getParentNode();
			if (parentNode!= null && !parentNode.getNodeName().equals(CHOICE_NODE_NAME)) {
				setClassCorrespondingSpan(entryNodeName,"",itfElt);
			}
			Node choiceChild = getChildNode(entryNode, CHOICE_NODE_NAME);
			if (choiceChild==null) {
				setClassCorrespondingSpan(CHOICE_NODE_NAME + PARAMETERS_SEPARATOR + entryNodeName,ITF_HIDDEN_STYLE, itfElt);
			}
			
			// computing attributes
			NamedNodeMap myNodeMap = entryNode.getAttributes();
			if (myNodeMap!=null) {
				for (int i=0; i< myNodeMap.getLength();i++) {
					Attr myAttribute = (Attr) myNodeMap.item(i);
					fillTemplate(myAttribute, itfElt, itfTemplate, newId);
				}
			}
			
			// computing child elements
			String inputValue = "";
            boolean found = findCorrespondingTextInput(entryNodeName,itfElt);
            if (found) {
                inputValue = serializeChildren(entryNode);
            }
            else if (entryNode.hasChildNodes ()) {
				NodeList myNodeList = entryNode.getChildNodes ();
				Node previousNode = null;
				Element correspItf;
				for (int i = 0; i < myNodeList.getLength (); i++) {
					Node nodeItem = myNodeList.item(i);
					switch (nodeItem.getNodeType()) {
						// an attribute note is not a child node
						// so, this case should not be present 
						// attributes are computed later
						case  Node.ATTRIBUTE_NODE:
							fillTemplate((Attr) nodeItem, itfElt, itfTemplate, newId);
							break;
							// this case case should not be present.
						case  Node.DOCUMENT_NODE:
							fillTemplate(((Document) nodeItem).getDocumentElement() ,itfElt, itfTemplate, newId);
							break;
						case  Node.ELEMENT_NODE:
							// If the previous node is of the same type, then duplicate the previous corresponding interface node 
							if (previousNode!=null && nodeItem.getNodeName().equals(previousNode.getNodeName())) {
								correspItf = findCorrespondingElement((Element) previousNode,itfElt);
								correspItf = duplicateInterfaceElement((Element) nodeItem,correspItf,itfTemplate);
							}
							else {
								correspItf = findCorrespondingElement((Element) nodeItem,itfElt);
							}
							previousNode = nodeItem;
							fillTemplate(nodeItem,correspItf, itfTemplate, newId);
							break;
						case Node.TEXT_NODE:
							inputValue = nodeItem.getNodeValue();
							// virer tous les blancs de nodevalue
							break;
						default:
							break;
					}
				}
			}
			boolean valueSet = setIdValueCorrespondingTextInput(entryNodeName,itfElt, newId, inputValue);
			if (!valueSet) {
				setIdValueCorrespondingSelect(entryNodeName,itfElt, newId, inputValue);
			}
			if (!valueSet) {
				setIdValueCorrespondingBooleanCheckbox(entryNodeName,itfElt, newId, inputValue);
			}
		}
	}
    
    protected static String serializeChildren(Node entryNode) {
        String result = "";
        
        if (entryNode.hasChildNodes ()) {
            NodeList myNodeList = entryNode.getChildNodes ();
            Node previousNode = null;
            Element correspItf;
            for (int i = 0; i < myNodeList.getLength (); i++) {
                Node nodeItem = myNodeList.item(i);
                switch (nodeItem.getNodeType()) {
                        // an attribute note is not a child node
                        // so, this case should not be present
                    case  Node.ATTRIBUTE_NODE:
                        break;
                        // this case case should not be present.
                    case  Node.DOCUMENT_NODE:
                        break;
                    case  Node.ELEMENT_NODE:
                        String contentString = serializeChildren(nodeItem);
                        if (contentString.equals("")) {
                            result += "<" + nodeItem.getNodeName() + "/>";
                        }
                        else {
                             result += "<" + nodeItem.getNodeName() + ">" + serializeChildren(nodeItem) + "</" + nodeItem.getNodeName() + ">";
                        }
 
                        break;
                    case Node.TEXT_NODE:
                        result += nodeItem.getNodeValue();
                        break;
                    default:
                        break;
                }
            }
        }
        
        return result;
    }
	
	protected static int getElementNumber(Element entryElt) {
		Document myDoc = entryElt.getOwnerDocument();
		NodeList myNodeList = myDoc.getElementsByTagName (entryElt.getNodeName());
		int idCount=0;
		int i=0;
		boolean found = false;
		while (i<myNodeList.getLength() && !found) {
			Element currentElt = (Element) myNodeList.item(i);
			if (currentElt==entryElt) {
				idCount=i;
				found= true;
			}	
			i++;
		}
		return idCount;
	}
	
	protected static String createId(Node entryNode, String eltId) {
		String idString = ""; 
		switch (entryNode.getNodeType()) {
			// this case case should not be present. It is solved in the next section
			case  Node.ATTRIBUTE_NODE:
				idString = eltId + ATTR_SEPARATOR + entryNode.getNodeName();
				break;
			case  Node.DOCUMENT_NODE:
				idString = entryNode.getNodeName() + ID_SEPARATOR + getElementNumber(((Document) entryNode).getDocumentElement());
				break;
			case  Node.ELEMENT_NODE:
				idString = entryNode.getNodeName() + ID_SEPARATOR + getElementNumber((Element) entryNode);
				break;
			case Node.TEXT_NODE:
				// this case case should not be present.
				idString = entryNode.getNodeName() + ID_SEPARATOR + "0";
				break;
			default:
				// this case case should not be present.
				idString = entryNode.getNodeName() + ID_SEPARATOR + "0";
				break;
		}
        
		return idString;
	}
	
	protected static Element findCorrespondingElement(Element entryElt, Element itfElt) {
		return findCorrespondingElement(entryElt, itfElt, false);
	}
	
	protected static Element findCorrespondingElement(Element entryElt, Element itfElt, boolean template) {
		// if template is true, it means that if nothing is found, the result is null
		// instead of the parent interface element.
		//PapillonLogger.writeDebugMsg("findCorrespondingElement " + entryElt.getNodeName());
        //PapillonLogger.writeDebugMsg(" itfelt: " + itfElt.getNodeName());
        //PapillonLogger.writeDebugMsg(" class: " + itfElt.getAttribute(ITF_ATTR_NAME));
		Element resultElt = null;
		if (entryElt != null && itfElt!= null) {
		String entryEltName = entryElt.getNodeName();
		NodeList myNodeList = itfElt.getElementsByTagName (ITF_ELT_DUPLICATE_NAME);
		if (myNodeList != null &&myNodeList.getLength ()>0) {
			int i=0;
			while (i<myNodeList.getLength () && resultElt==null) {
				Element currentElt = (Element) myNodeList.item(i);
				String myAttr = currentElt.getAttribute(ITF_ATTR_NAME);
				if (myAttr !=null) {
					if (myAttr.equals(entryEltName)) {
						// In any case, we take the first corresponding child
						// because after we duplicate from an empty template 
						resultElt = currentElt;
					}
				}	
				i++;	
			}
		}
		if (resultElt==null) {
			myNodeList = itfElt.getElementsByTagName (ITF_ELT_BLOCK_NAME);
			if (myNodeList != null &&myNodeList.getLength ()>0) {
				int i=0;
				while (i<myNodeList.getLength () && resultElt==null) {
					Element currentElt = (Element) myNodeList.item(i);
					String myAttr = currentElt.getAttribute(ITF_ATTR_NAME);
					if (myAttr !=null) {
						if (myAttr.equals(entryEltName)) {
							// In any case, we take the first corresponding child
							// because after we duplicate from an empty template 
							resultElt = currentElt;
						}
					}	
					i++;	
				}
			}			
		}
		if (resultElt==null) {
			myNodeList = itfElt.getElementsByTagName (ITF_ELT_INLINE_NAME);
			if (myNodeList != null &&myNodeList.getLength ()>0) {
				int i=0;
				while (i<myNodeList.getLength () && resultElt==null) {
					Element currentElt = (Element) myNodeList.item(i);
					String myAttr = currentElt.getAttribute(ITF_ATTR_NAME);
					if (myAttr !=null) {
						if (myAttr.equals(entryEltName)) {
							// In any case, we take the first corresponding child
							// because after we duplicate from an empty template 
							resultElt = currentElt;
						}
					}	
					i++;	
				}
			}			
		}
		}

		// if template is true, it means that if nothing is found, the result is null
		// instead of the parent interface element.
		if (resultElt==null && !template) {
			resultElt = itfElt;
		} 
		return resultElt;
	}
	
	protected static boolean setIdCorrespondingAnchor(String entryNodeName, Element itfElt, String newId) {
		//PapillonLogger.writeDebugMsg("setIdCorrespondingAnchor: " + entryNodeName);
		boolean found = false;
		NodeList myNodeList = itfElt.getElementsByTagName ("a");
		int i=0;
		while (i<myNodeList.getLength () && !found) {
			Element currentElt = (Element) myNodeList.item(i);
			String name = currentElt.getAttribute("name");
			if (name !=null && name.equals(entryNodeName)) {
				currentElt.setAttribute("name", newId);
				found = true;
			}
		i++;	
		}
		return found;
	}
	
	
	protected static boolean setIdCorrespondingLabel(String label, Element itfElt, String newId) {
		//PapillonLogger.writeDebugMsg("setIdCorrespondingLabel: " + label);
		boolean found = false;
		NodeList myNodeList = itfElt.getElementsByTagName ("label");
		int i=0;
		while (i<myNodeList.getLength () && !found) {
			Element currentElt = (Element) myNodeList.item(i);
			String name = currentElt.getAttribute("for");
			if (name !=null && name.equals(label)) {
				currentElt.setAttribute("for", newId);
				found = true;
			}	
			i++;	
		}
		return found;
	}
	
	protected static boolean setClassCorrespondingSpan(String spanTitle, String classType, Element itfElt) {
        //PapillonLogger.writeDebugMsg("setClassCorrespondingSpan: " + spanTitle + " class: " + classType);
		boolean found = false;
		NodeList myNodeList = itfElt.getElementsByTagName ("span");
		int i=0;
		while (i<myNodeList.getLength () && !found) {
			Element currentElt = (Element) myNodeList.item(i);
			String name = currentElt.getAttribute("title");
			if (name !=null && name.equals(spanTitle)) {
				currentElt.setAttribute("class", classType);
				found = true;
			}	
			i++;	
		}
		return found;
	}
	
	protected static boolean setIdCorrespondingSubmitInputs(String correspName, Element itfElt, String newId) {
		//PapillonLogger.writeDebugMsg("setIdCorrespondingSubmitInputs: " + correspName);
		boolean found = false;
		NodeList myNodeList = itfElt.getElementsByTagName ("input");
		for (int i=0;i<myNodeList.getLength ();i++) {
			Element currentElt = (Element) myNodeList.item(i);
			String name = currentElt.getAttribute("name");
			if (name !=null && name.equals(correspName)) {
				String type = currentElt.getAttribute("type");
				if (type !=null && type.equals("submit")) {
					currentElt.setAttribute("onclick", currentElt.getAttribute("onclick") + newId +"'");
					found = true;
				}
			}	
		}
		return found;
	}
    
    protected static boolean setClassCorrespondingLinkers(String correspName, Element itfElt, String newId) {
		//PapillonLogger.writeDebugMsg("setIdCorrespondingLinkers: " + correspName + ", " + newId);
		boolean found = false;
		NodeList myNodeList = itfElt.getElementsByTagName ("img");
		for (int i=0;i<myNodeList.getLength ();i++) {
			Element currentElt = (Element) myNodeList.item(i);
			String classValue = currentElt.getAttribute("class");
            
            //
            if (classValue !=null) {
                
                // Compile regular expression
                // Replace all correspName between = and (", . or &) with newId
                String patternStr = "(=)" + correspName + "([.&]|$)";
                String replaceStr = "$1" + newId + "$2";
                Pattern pattern = Pattern.compile(patternStr);
            
                // Match and replace
                Matcher matcher = pattern.matcher(classValue);
                found = matcher.find();
                String output = matcher.replaceAll(replaceStr);
                currentElt.setAttribute("class", output);
            }
		}
		return found;
	}
    
    protected static boolean setIdValueCorrespondingTextInput(String correspName, Element itfElt, String newId, String value) {
        //PapillonLogger.writeDebugMsg("setIdValueCorrespondingTextInput: " + correspName + " (newId = " + newId + ") (value = " + value + ")");
        boolean found = false;
        NodeList myNodeList = itfElt.getElementsByTagName ("input");
        int i=0;
        while (i<myNodeList.getLength() && !found) {
            Element currentElt = (Element) myNodeList.item(i);
            String name = currentElt.getAttribute("name");
            if (name !=null && name.equals(correspName)) {
                String type = currentElt.getAttribute("type");
                if (type !=null && (type.equals("text") || type.equals("textarea"))) {
                    currentElt.setAttribute("name", newId);
                    currentElt.setAttribute("id", newId);
                    currentElt.setAttribute("value", value);
                    found = true;
                }
            }
            i++;
        }
        if (!found) {
            myNodeList = itfElt.getElementsByTagName ("textarea");
            i=0;
            while (i<myNodeList.getLength() && !found) {
                Element currentElt = (Element) myNodeList.item(i);
                String name = currentElt.getAttribute("name");
                if (name !=null && name.equals(correspName)) {
                    currentElt.setAttribute("name", newId);
                    currentElt.setAttribute("id", newId);
                    NodeList children = currentElt.getChildNodes();
                    int j=0;
                    while (j<children.getLength()) {
                        currentElt.removeChild(children.item(j));
                        j++;
                    }
                    org.w3c.dom.Text textElt = currentElt.getOwnerDocument().createTextNode(value);
                    currentElt.appendChild(textElt);
                    found = true;
                }
                i++;	
            }		
        }
        return found;
    }
    

    
	
	protected static boolean findCorrespondingTextInput(String correspName, Element itfElt) {
		//PapillonLogger.writeDebugMsg("setIdValueCorrespondingTextInput: " + correspName + " (newId = " + newId + ") (value = " + value + ")");
		boolean found = false;
		NodeList myNodeList = itfElt.getElementsByTagName ("input");
		int i=0;
		while (i<myNodeList.getLength() && !found) {
			Element currentElt = (Element) myNodeList.item(i);
			String name = currentElt.getAttribute("name");
			if (name !=null && name.equals(correspName)) {
				String type = currentElt.getAttribute("type");
				if (type !=null && (type.equals("text") || type.equals("textarea"))) {
					found = true;
				}
			}
			i++;	
		}
		if (!found) {
			myNodeList = itfElt.getElementsByTagName ("textarea");
			i=0;
			while (i<myNodeList.getLength() && !found) {
                Element currentElt = (Element) myNodeList.item(i);
				String name = currentElt.getAttribute("name");
				if (name !=null && name.equals(correspName)) {
 					found = true;
				}
				i++;	
			}		
		}
		return found;
	}
	
	protected static boolean setIdCorrespondingSelectCheckbox(String correspName, Element itfElt, String newId) {
		//PapillonLogger.writeDebugMsg("setIdCorrespondingCheckbox: " + correspName);
		boolean found = false;
		NodeList myNodeList = itfElt.getElementsByTagName ("input");
		int i=0;
		while (i<myNodeList.getLength () && !found) {
			Element currentElt = (Element) myNodeList.item(i);
			String name = currentElt.getAttribute("name");
			String type = currentElt.getAttribute("type");
			if ((name.equals(SELECT_ATTR_NAME) || name.equals(CHOOSE_ATTR_NAME))
				&& type.equals("checkbox")) {
				String value = currentElt.getAttribute("value");
				if (value.equals(correspName)) {
					currentElt.setAttribute("id","chkb-"+newId);
					currentElt.setAttribute("value",newId);
					found = true;
				}
			}	
			i++;	
		}		
		return found;
	}
	
	protected static boolean setIdValueCorrespondingBooleanCheckbox(String correspName, Element itfElt, String newId, String newValue) {
		//PapillonLogger.writeDebugMsg("setIdValueCorrespondingBooleanCheckbox: " + correspName + " id: " + newId + " value: " + newValue);
		boolean found = false;
		int foundInt = 0;
		NodeList myNodeList = itfElt.getElementsByTagName ("input");
		int i=0;
		while (i<myNodeList.getLength () && !found) {
			Element currentElt = (Element) myNodeList.item(i);
			String name = currentElt.getAttribute("name");
			String type = currentElt.getAttribute("type");
			if (name.equals(BOOLEAN_TRUE_ATTR_NAME) && type.equals("checkbox")) {
				String value = currentElt.getAttribute("value");
				if (value.equals(correspName)) {
					if (newValue!=null && newValue.equals("true")) {
						currentElt.setAttribute("checked",newValue);
					}
					currentElt.setAttribute("id",newId);
					currentElt.setAttribute("value",newId);
					foundInt++;
				}
			}	
			else if (name.equals(BOOLEAN_ATTR_NAME) && type.equals("hidden")) {
				String value = currentElt.getAttribute("value");
				if (value.equals(correspName)) {
					currentElt.setAttribute("id",newId);
					currentElt.setAttribute("value",newId);
					foundInt++;
				}
			}	
			i++;	
			found = (foundInt == 2);
		}		
		return found;
	}
	
	
	protected static boolean setIdValueCorrespondingSelect(String correspName, Element itfElt, String newId, String value) {
		//PapillonLogger.writeDebugMsg("setIdValueCorrespondingSelect: " + correspName);
		boolean found = false;
		NodeList myNodeList = itfElt.getElementsByTagName ("select");
		int i=0;
		while (i<myNodeList.getLength () && !found) {
			Element currentElt = (Element) myNodeList.item(i);
			String myAttr = currentElt.getAttribute("name");
			if (myAttr !=null) {
				if (myAttr.equals(correspName)) {
					setSelected(currentElt,value);
					currentElt.setAttribute("name",newId);
					currentElt.setAttribute("id",newId);
					found = true;
				}
			}	
			i++;	
		}		
		return found;
	}
	
	protected static boolean setSelected(Element itfElt, String value) {
		//PapillonLogger.writeDebugMsg("setSelected: " + value);
		NodeList myNodeList = itfElt.getElementsByTagName("option");
		boolean found = false;
		int i=0;
		while (i<myNodeList.getLength () && !found) {
			Element currentElt = (Element) myNodeList.item(i);
			String myAttr = currentElt.getAttribute("value");
			if (myAttr !=null) {
				if (myAttr.equals(value)) {
					currentElt.setAttribute("selected","selected");
					found = true;
				}
			}	
			i++;	
		}		
		return found;
	}
	
	
	protected static Element duplicateInterfaceElement(Element entryElt, Element itfElt, Element itfTemplate) {
		//PapillonLogger.writeDebugMsg("duplicateInterfaceElement: " + entryElt.getNodeName() + " itf: " + itfElt.getNodeName() + " class: " + itfElt.getAttribute(ITF_ATTR_NAME));	
		Vector nodeVector = new Vector();
		boolean template = true;
		Element resElt = (Element) findCorrespondingElement(entryElt,itfTemplate, template);
		if (resElt !=null) {
			
			Document itfDoc = itfElt.getOwnerDocument();
			resElt = (Element) itfDoc.importNode(resElt,true);
			Element parentElement = (Element) itfElt.getParentNode();
			
			NodeList childNodes = parentElement.getChildNodes();
			boolean found = false;
			int i=0;
			while (i<childNodes.getLength () && !found) {
				Node nodeItem = childNodes.item(i);
				if (nodeItem.getNodeType()== Node.ELEMENT_NODE) {
					Element currentElt = (Element) nodeItem;		
					String myAttr = currentElt.getAttribute(ITF_ATTR_NAME);
					if (myAttr !=null) {
						found = (myAttr.equals(entryElt.getNodeName()));				
					}
				}	
				i++;	
			}					
			boolean other = false;
			while (i<childNodes.getLength () && !other) {
				Node nodeItem = childNodes.item(i);
				if (nodeItem.getNodeType()== Node.ELEMENT_NODE) {
					Element currentElt = (Element) nodeItem;		
					String myAttr = currentElt.getAttribute(ITF_ATTR_NAME);
					if (myAttr !=null) {
						other = (!myAttr.equals(entryElt.getNodeName()));
					}
					else {
						other=true;
					}
				}	
				i++;	
			}		
			i--;
			if (i<childNodes.getLength ()) {
				parentElement.insertBefore(resElt,childNodes.item(i));
			}
			else {
				parentElement.appendChild(resElt);
			}
		}
		return resElt;
	}
	
	
	protected static String insertEntryElement(Element parentElement, Element newElement, Element siblingElement) {
//		PapillonLogger.writeDebugMsg("insertEntryElement: " + newElement.toString() + " parent: " + parentElement.toString());
		Vector nodeVector = new Vector();
		
		Element resElt = (Element) parentElement.getOwnerDocument().importNode(newElement,true);
		
		if (siblingElement != null) {
			parentElement.insertBefore(resElt,siblingElement);
		}
		else {
			NodeList childNodes = parentElement.getChildNodes();
			boolean found = false;
			int i=0;
			while (i<childNodes.getLength () && !found) {
				Node nodeItem = childNodes.item(i);
				if (nodeItem.getNodeType()== Node.ELEMENT_NODE) {
					Element currentElt = (Element) nodeItem;	
					found = (currentElt.getNodeName().equals(resElt.getNodeName()));
				}	
				i++;	
			}		
		
			boolean other = false;
			while (i<childNodes.getLength () && !other) {
				Node nodeItem = childNodes.item(i);
				if (nodeItem.getNodeType()== Node.ELEMENT_NODE) {
					Element currentElt = (Element) nodeItem;		
					other = (!currentElt.getNodeName().equals(resElt.getNodeName()));
				}
				i++;	
			}
            if (other) {
                i--;
            }
			if (i<childNodes.getLength()) {
 //               PapillonLogger.writeDebugMsg("insertEntryElement: insert before: " + i + "<" + childNodes.getLength());

				parentElement.insertBefore(resElt,childNodes.item(i));
			}
			else {
 //               PapillonLogger.writeDebugMsg("insertEntryElement: append");
				parentElement.appendChild(resElt);
			}
		}
		return createId(resElt,null);
	}
	
	protected static Element getTemplateEntryElement(String elementName, String parentName, Element entryTemplate) {
		//PapillonLogger.writeDebugMsg("getTemplateEntryElement: " + elementName + " parent: " + parentName);
		Element resElt = null;
		Element myParent = null;
		if (entryTemplate.getTagName().equals(parentName)) {
			myParent = entryTemplate;
		}
		else {
			NodeList parentNodes = entryTemplate.getElementsByTagName(parentName);
			if (parentNodes.getLength()>0) {
				myParent = (Element) parentNodes.item(0);
			}
		}
		if (myParent !=null) {
			NodeList myNodes = myParent.getElementsByTagName(elementName);
			if (myNodes.getLength()>0) {
				resElt = (Element)myNodes.item(0);
			}
		}
		return resElt;
	}
	
    
    // FIXED: this should not return an element if the eltNb is not a number...
    // FIXED: The eltNb is now parsed as an int.
	protected static Element findElementInEntry(String eltName, String eltNb, Element entryElt) {
		if (entryElt.getTagName().equals(eltName)) {
            // Assumes that eltNb == 0...
			return entryElt;
		} else {
            try {
                NodeList myNodeList = entryElt.getElementsByTagName (eltName);
                int nb = Integer.parseInt(eltNb);;
                return (Element) myNodeList.item(nb);
            } catch (Exception e) {
                return null;
            }
		}
	}
	
	protected static Node getChildNode(Node parentNode, String childName) {
		Node resultNode = null;
		boolean found = false;
		NodeList myNodeList = parentNode.getChildNodes ();
		int i=0;
		while (i<myNodeList.getLength() && !found) {
			Node tempNode = myNodeList.item(i);
			if (tempNode.getNodeName().equals(childName)) {
				found = true;
				resultNode= tempNode;
			}
			i++;
		}
		return resultNode;
	}
	
}
