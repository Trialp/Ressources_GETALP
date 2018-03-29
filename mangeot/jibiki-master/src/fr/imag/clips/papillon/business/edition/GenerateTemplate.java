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
 * Revision 1.20.6.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.20  2006/03/29 13:54:10  mangeot
 * Fixed a bug when managing a list with prefixed elements
 *
 * Revision 1.19  2006/03/13 08:48:00  fbrunet
 * bug corrections before merge
 *
 * Revision 1.18  2006/03/01 16:11:01  mangeot
 * The edit form is now in an XHTML file
 *
 * Revision 1.17  2006/03/01 15:41:13  mangeot
 * bug fixes
 *
 * Revision 1.16  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.10.2.1  2006/01/24 13:39:49  fbrunet
 * Modification view management
 * Modification LexALP postprocessing
 *
 * Revision 1.10  2005/08/01 15:03:41  mangeot
 * Corrected an important bug in the editor that forbidded to change a boolean value from true to false.
 * Beware, you have to edit the existing interface templates by hands:
 * 1- duplicate all the input elements with name='boolean' and type='checkbox'.
 * - for each input element pair,
 *  2- change one input element name into name='booleantrue'
 *  3- change the other input element type to type='hidden'
 *
 * Revision 1.9  2005/07/05 09:21:59  serasset
 * Template interface generator now correctly generates attribute names (with an @).
 * Target languages are now correctly handled when querying a pivot multilingual dictionary.
 *
 * Revision 1.8  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 *
 *---------------------------------------------------------
 */

package fr.imag.clips.papillon.business.edition;

import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.presentation.EditEntry;

// to parse the XML schema
import org.apache.xerces.xni.parser.XMLInputSource;
import org.apache.xerces.impl.xs.*;
import org.apache.xerces.xs.*;
//import org.apache.xerces.impl.xs.psvi.*;

// io imports
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.StringReader;


public class GenerateTemplate {
	
	protected final static String DefaultEncoding = "UTF-8";
	
	protected final static String xmlHeader = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" +
		"<html xmlns=\"http://www.w3.org/1999/xhtml\" xmlns:d=\"http://www-clips.imag.fr/geta/services/dml\">\n" +
		"<head>\n" +
		"	<meta http-equiv=\"content-type\" content=\"text/html; charset=utf-8\" />\n" +
		"	<title>Interface Template</title>\n" +
		"</head>\n" +
		"<body>\n" +
		"<form  action=\"HandleEntryModifications.po\" method=\"post\" accept-charset=\"utf-8\" enctype=\"application/x-www-form-urlencoded\" xml:lang=\"en\" lang=\"en\" d:lang=\"eng\">\n" +
		"<div name=\"" + UITemplates.EDIT_ENTRY_INTERFACE + "\" id=\"" + UITemplates.DEFAULT_FORM  + "\"  type=\"" + UITemplates.DEFAULT_TYPE  + "\" >\n" +
		"<h1 style=\"text-align:center\">Edition interface</h1>\n";
	
	protected final static String xmlFooter = "</div>\n" +
		"</form>\n" +
		"</body>\n" +
		"</html>\n";
	
	protected final static int DEFAULT_ELEMENT = 0;
	protected final static int LIST_ELEMENT = 1;
	protected final static int CHOICE_ELEMENT = 2;
	
	// public methods
	public static void generateInterfaceTemplate(Volume myVolume)  
		throws PapillonBusinessException {
			
			BufferedWriter bufferInterface;
			
			try {
				ByteArrayOutputStream interfaceOS = new ByteArrayOutputStream();
				OutputStreamWriter interfaceWriter =
					new OutputStreamWriter(interfaceOS, DefaultEncoding);
				// throws UnsupportedEncodingException
				bufferInterface = new BufferedWriter(interfaceWriter);
				PapillonLogger.writeDebugMsg("Generating interface template from XML schema");
				produceInterfaceTemplate(bufferInterface, myVolume.getXmlSchema(), myVolume.getCdmEntry(), myVolume.getName());
				bufferInterface.close();
				interfaceOS.flush();
				String itfString = interfaceOS.toString(DefaultEncoding);
				myVolume.setTemplateInterface(itfString);
				myVolume.save();
				//	PapillonLogger.writeDebugMsg("interface template generated: ");
			} catch (java.io.IOException ex) {
				throw new PapillonBusinessException ("IOException in generateInterfaceTemplate ", ex);
			} 
		}
	
	// protected methods
	protected static void produceInterfaceTemplate(BufferedWriter interfaceBuffer, 
												   String schemaString, 
												   String rootElement, 
												   String volumeName) 
		throws PapillonBusinessException {
			
			XMLInputSource	grammarSource = new XMLInputSource ("", "", "", new StringReader(schemaString), "");
			
			XMLSchemaLoader loader;
			SchemaGrammar schemaGrammar = null;
			
			try {
				loader = new XMLSchemaLoader();
				schemaGrammar = (SchemaGrammar) loader.loadGrammar(grammarSource);
				PapillonLogger.writeDebugMsg("produceInterfaceTemplate: Schema parsed ===============================");
			} catch (Exception ex) {
				throw new PapillonBusinessException("Exception in XmlSchemaFactory.produceInterfaceTemplate with " +  schemaString, ex);
			}
			produceInterfaceTemplate(interfaceBuffer, schemaGrammar, rootElement, volumeName);
		}
	
	protected static void produceInterfaceTemplate(BufferedWriter interfaceBuffer, 
												   SchemaGrammar schemaGrammar, String rootElement, String volumeName)  
		throws PapillonBusinessException {
			
			XSElementDeclaration myDeclaration = null;
			XSNamedMap listComponent;
			XSObject object;
			
			// FIXME: Attention, pour la génération, j'utilise les noms de balises locaux, non préfixés par un espace de noms
			// Il faudrait corriger ça en ajoutant le préfixe à toutes les balises.
			// Si on modifie la génération, alors il faut aussi modifier UIGenerator.fillInterfaceTemplate
			String prefix = fr.imag.clips.papillon.business.utility.Utility.getPrefix(rootElement);
			if (prefix != null && !prefix.equals("")) {
				prefix += ":";
			}
			
			String localRootName = fr.imag.clips.papillon.business.utility.Utility.getLocalTagName(rootElement);
			PapillonLogger.writeDebugMsg("produceInterfaceTemplate: prefix: " + prefix + " local root name: " + localRootName);
			
			try {
				interfaceBuffer.write(xmlHeader);
				myDeclaration = schemaGrammar.getElementDeclaration(localRootName);
				if (myDeclaration !=null) {
					parseElementDeclaration(interfaceBuffer, volumeName, myDeclaration, prefix);
				}
				interfaceBuffer.write(xmlFooter);
			} catch (Exception ex) {
				throw new PapillonBusinessException("Exception in GenerateTemplate.produceInterfaceTemplate with " +  volumeName, ex);
			}
		}
	
	protected static void parseElementDeclaration(BufferedWriter interfaceBuffer,
												  String volumeName, XSElementDeclaration declaration, String prefix) {
		parseElementDeclaration(interfaceBuffer, volumeName, declaration, prefix, DEFAULT_ELEMENT);
	}
	
	
	protected static void parseElementDeclaration(BufferedWriter interfaceBuffer,
												  String volumeName, XSElementDeclaration declaration, String prefix, int elementType) {
		
		XSObject object;
		XSTypeDefinition baseType;
		XSComplexTypeDefinition complexType;
		
		XSAttributeDeclaration attributePSVI;
		XSParticle particle = null;
		XSTerm term = null;
		XSModelGroup modelGroup = null;
		
		try {
			// FIXME: il faudrait utiliser declaration.getNamespace au lieu de construire le préfixe...
			String elementName = prefix + declaration.getName();
			String elementNameDisplay = declaration.getName();
			if (elementType==GenerateTemplate.LIST_ELEMENT) {
				interfaceBuffer.write("  <tr " +  UIGenerator.ITF_ATTR_NAME  + "='" + elementName + "' style='background-color:#ffebdc'>\n");
				interfaceBuffer.write("    <td style='text-align:center' valign='top' width='25'>\n");
				interfaceBuffer.write("		 <a name='" + elementName + "'> </a>");
				interfaceBuffer.write("<input name='" + UIGenerator.SELECT_ATTR_NAME + "' type='checkbox' value='" + elementName + "' />");
				interfaceBuffer.write("\n");
				interfaceBuffer.write("    </td>\n");
				interfaceBuffer.write("    <td  style='text-align:center' valign='top'  class='block'>\n");
				interfaceBuffer.write("      <h4 class='blockTitle'>" + elementNameDisplay + "</h4>\n");
			}
			else if (elementType==GenerateTemplate.CHOICE_ELEMENT) {
				interfaceBuffer.write("<" + UIGenerator.ITF_ELT_INLINE_NAME + " " + UIGenerator.ITF_ATTR_NAME  + "='" + elementName + "'>" + "\n");
				interfaceBuffer.write("<label for='" + elementName + "'>"+ elementNameDisplay + ":</label>\n");
				interfaceBuffer.write("<input name='" + UIGenerator.CHOOSE_ATTR_NAME + "' type='checkbox' value='" + elementName + "' />\n");
				interfaceBuffer.write("<br />\n");
				interfaceBuffer.write("<span title='" + elementName + "' class='" + UIGenerator.ITF_HIDDEN_STYLE + "'>\n");
			}
			else {
				interfaceBuffer.write("<" + UIGenerator.ITF_ELT_INLINE_NAME + " " + UIGenerator.ITF_ATTR_NAME  + "='" + elementName + "'>" + "\n");
			}
			baseType = declaration.getTypeDefinition ();
			String typeName = declaration.getName() + "Type";
			if (baseType.getName()!=null) {
				typeName =  baseType.getName();
			}
			
			//PapillonLogger.writeDebugMsg("Element: " + elementName);
			//PapillonLogger.writeDebugMsg("Type Name: " + typeName);
			
			if (baseType instanceof XSSimpleTypeDefinition) {
				parseSimpleTypeDeclaration (interfaceBuffer,(XSSimpleTypeDefinition) baseType,elementName);
			}
			else if (baseType instanceof XSComplexTypeDefinition) {
				//	interfaceBuffer.write("  <blockquote>\n");	
				//PapillonLogger.writeDebugMsg("Complex type definition");
				complexType = (XSComplexTypeDefinition) baseType; 
				
				/* Attributes */
				XSObjectList listObject = complexType.getAttributeUses ();
				if (listObject != null) {
					for (int i = 0; i < listObject.getLength (); i++) {
						object = listObject.item (i);
						if ((object != null) && (object instanceof XSAttributeUse)) {
							attributePSVI = ((XSAttributeUse) object).getAttrDeclaration ();
							String attrName = attributePSVI.getName();
							// avoid special attributes
							if (!attrName.equals("id")) {
								attrName = elementName + UIGenerator.ATTR_SEPARATOR + attrName;
								parseSimpleTypeDeclaration (interfaceBuffer,attributePSVI.getTypeDefinition(),attrName);
							}
						}
					}
				}
				XSSimpleTypeDefinition simpleTypeDefinition;
				switch (complexType.getContentType ()) {
					case XSComplexTypeDefinition.CONTENTTYPE_SIMPLE:
						//			PapillonLogger.writeDebugMsg ("************************CONTENTTYPE_SIMPLE" );
						/* SimpleContent */
						simpleTypeDefinition = complexType.getSimpleType ();
						if (simpleTypeDefinition != null) {
							parseSimpleTypeDeclaration (interfaceBuffer,simpleTypeDefinition,elementName);
							//PapillonLogger.writeDebugMsg ("PB in constructComplexType: a simpleTypeDefinition not managed: " + simpleTypeDefinition);
						}
                        break;
						
					case XSComplexTypeDefinition.CONTENTTYPE_MIXED:
						//		PapillonLogger.writeDebugMsg ("************************CONTENTTYPE_MIXED" );
						particle = complexType.getParticle ();
						/* Case where there are no child elements */
						if (particle == null) {
							simpleTypeDefinition = complexType.getSimpleType ();
							if (simpleTypeDefinition != null) {
								parseSimpleTypeDeclaration (interfaceBuffer,simpleTypeDefinition,elementName);
							}
							else {
								// I simulate the content as a string
								addStringTypeDefinition(interfaceBuffer, elementName);
							}
						}
                        else {
                            // I simulate the content as a string
                            addStringTypeDefinition(interfaceBuffer, elementName);
                        }
                        break;
						
					case XSComplexTypeDefinition.CONTENTTYPE_ELEMENT:
						//	PapillonLogger.writeDebugMsg ("************************CONTENTTYPE_ELEMENT" );
						/* Complex type definied with a {choice, sequence or all) elements */
						particle = complexType.getParticle ();
						if (particle != null) {
							term = particle.getTerm();
							if (term instanceof XSModelGroup) {
								// PapillonLogger.writeDebugMsg ("XSModelGroup ");
								modelGroup = (XSModelGroup) term;
								if (modelGroup.getCompositor()==XSModelGroup.COMPOSITOR_SEQUENCE) {
									//PapillonLogger.writeDebugMsg ("Compositor sequence");
									XSObjectList myList = modelGroup.getParticles();
									for (int j=0;j<myList.getLength();j++) {
										XSParticle tempParticle = (XSParticle)myList.item(j);
										term = tempParticle.getTerm();
										int newTypeElement = DEFAULT_ELEMENT;
										String termNameDisplay = term.getName();
										if (tempParticle.getMaxOccursUnbounded() && termNameDisplay!=null) {
											String termName = prefix+termNameDisplay;
											// list	
											interfaceBuffer.write("<table border='0' cellpadding='5' cellspacing='2' summary='List of " + termNameDisplay + "s' width='100%'>\n");
											interfaceBuffer.write("  <tr style='background-color: #fbbe78'>\n");
											interfaceBuffer.write("    <td style='text-align:center' width='25'>\n");
											interfaceBuffer.write("      <input name='" + elementName + "' onclick=\"this.form.AddCall.value='" + termName + UIGenerator.PARAMETERS_SEPARATOR + "\" type='submit' value='+' />\n");
											interfaceBuffer.write("      <input name='" + elementName + "' onclick=\"this.form.DelCall.value='" + termName + UIGenerator.PARAMETERS_SEPARATOR + "\" type='submit' value='-' />\n");
											interfaceBuffer.write("    </td>\n");
											interfaceBuffer.write("    <th align='center'>List of " +  termNameDisplay + "s:</th>\n");
											interfaceBuffer.write("  </tr>\n");
											PapillonLogger.writeDebugMsg(" list element: " + termName);
											newTypeElement = LIST_ELEMENT;
										}
										if (term instanceof XSElementDeclaration) {
											parseElementDeclaration(interfaceBuffer, volumeName, 
																	(XSElementDeclaration) term, prefix, newTypeElement);
										}	
										if (newTypeElement == LIST_ELEMENT) {
											interfaceBuffer.write("</table>\n");
										}
									}																					
								}
								else if (modelGroup.getCompositor()==XSModelGroup.COMPOSITOR_CHOICE) {
									PapillonLogger.writeDebugMsg ("Compositor choice");
									// choice
									interfaceBuffer.write("  <span title='" +  
														  UIGenerator.CHOICE_NODE_NAME + UIGenerator.PARAMETERS_SEPARATOR + elementName
														  + "'>\n");
									interfaceBuffer.write("  Choice for " +  elementNameDisplay + ":\n");
									interfaceBuffer.write("  <input name='" + elementName + "' onclick=\"this.form.ChooseCall.value='" + elementName + UIGenerator.PARAMETERS_SEPARATOR + "'\" type='submit' value='||' />\n");
									interfaceBuffer.write("  <br />\n</span>\n");
									XSObjectList myList = modelGroup.getParticles();
									for (int j=0;j<myList.getLength();j++) {
										XSParticle tempParticle = (XSParticle)myList.item(j);
										term = tempParticle.getTerm();
										if (term instanceof XSElementDeclaration) {
											PapillonLogger.writeDebugMsg(" choice element: " + ((XSElementDeclaration) term).getName());
											parseElementDeclaration(interfaceBuffer, volumeName, 
																	(XSElementDeclaration) term, prefix, CHOICE_ELEMENT);
										}	
									}
								}
							}
							if (!(particle instanceof XSParticleDecl)) {
								PapillonLogger.writeDebugMsg ("-------- PB in constructComplexType: particle have a bad type.");
							}
							else {
								//		attribute = constructAttributeFromParticle ((XSParticleDecl) particle);
								//		if (attribute != null)
								//			concept.addAttributeRelationID (attribute.getID (), currentContainer.version.getID ());
							}
						}
							break;
						
					case XSComplexTypeDefinition.CONTENTTYPE_EMPTY:
						PapillonLogger.writeDebugMsg ("************************CONTENTTYPE_EMPTY" );
						/* Complex type definied only with attibutes */
						break;
		}
				//				interfaceBuffer.write("  </blockquote>\n");	
	}
			if (elementType==GenerateTemplate.CHOICE_ELEMENT) {
				interfaceBuffer.write("</" + UIGenerator.ITF_ELT_INLINE_NAME + ">\n");	
				interfaceBuffer.write("</" + UIGenerator.ITF_ELT_INLINE_NAME + ">\n");	
			}
			else if (elementType==GenerateTemplate.LIST_ELEMENT) {
				interfaceBuffer.write("    </td>\n  </tr>\n");
			}
			else  {
				interfaceBuffer.write("</" + UIGenerator.ITF_ELT_INLINE_NAME + ">\n");	
			}
		}
		catch (java.io.IOException IOEx) {
			IOEx.printStackTrace();
		}		
	}
	
	protected static void parseSimpleTypeDeclaration (BufferedWriter interfaceBuffer, 
													  XSSimpleTypeDefinition simpleTypeDefinition, String elementName) 
		throws java.io.IOException {
			String typeName = simpleTypeDefinition.getName();
			String elementNameDisplay = fr.imag.clips.papillon.business.utility.Utility.getLocalTagName(elementName);
			//PapillonLogger.writeDebugMsg("Simple type definition: " + elementName + " type: " + typeName);
			interfaceBuffer.write("  <label for='" + elementName + "' >" + elementNameDisplay + ":</label>\n");	
			// we do not manage all the different types, we can only use a checkbox, a text box and a select
			// for a closed list, we use a select with a list of options
			if (simpleTypeDefinition.isDefinedFacet(XSSimpleTypeDefinition.FACET_ENUMERATION)) {
				interfaceBuffer.write("    <select name=\"" + elementName + "\" >" + "\n");			
				StringList stringList = simpleTypeDefinition.getLexicalEnumeration();
				for (int i=0; i < stringList.getLength (); i++) {
					interfaceBuffer.write("      <option value='" + stringList.item(i) + "'>" + stringList.item(i) + "</option>\n");		
				}
				interfaceBuffer.write("    </select>" + "\n");	
				interfaceBuffer.write("<br />\n");					
			}
			// for a boolean, we use a checkbox.
			else if (typeName!= null && typeName.equals("boolean")) {
				interfaceBuffer.write("    <input name='" + UIGenerator.BOOLEAN_ATTR_NAME + "' type='hidden' value='" + elementName + "' />\n");			
				interfaceBuffer.write("    <input name='" + UIGenerator.BOOLEAN_TRUE_ATTR_NAME + "' type='checkbox' value='" + elementName + "' />\n");			
				interfaceBuffer.write("<br />\n");					
			}
			else if (typeName!= null && typeName.equals("integer")) {
				interfaceBuffer.write("    <input name='" + elementName + "' type='text' onkeyup=\"this.value=this.value.replace(/\\D/,'')\" onchange=\"this.value=this.value.replace(/\\D/g,'')\" value=\"\" />\n");			
				interfaceBuffer.write("<br />\n");			
			}
			// for all the other cases, we use a text box.
			else {
				//	if (typeName!= null && typeName.equals("string")) {
				interfaceBuffer.write("    <input name='" + elementName + "' type='text' value='' />" + "\n");			
				interfaceBuffer.write("<br />\n");			
				}
			}
	
	
	protected static void addStringTypeDefinition(BufferedWriter interfaceBuffer, String elementName) 
		throws java.io.IOException {
			interfaceBuffer.write("  <label for='" + elementName + "' >" + elementName + ":</label>\n");	
			interfaceBuffer.write("    <input name='" + elementName + "' type='text' value='' />" + "\n");			
			interfaceBuffer.write("<br />\n");			
		}
	
}
