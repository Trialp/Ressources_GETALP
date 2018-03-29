/*
 *  DmlPrefixResolver.java
 *  papillon
 *
 *  Created by Mathieu Mangeot on 22/06/05.
 *  Copyright (c) 2005 GETA_CLIPS_IMAG. All rights reserved.
 *
 *---------------------------------------------------------
 * $Id$
 *---------------------------------------------------------
 * $Log$
 * Revision 1.2  2005/07/05 09:21:59  serasset
 * Template interface generator now correctly generates attribute names (with an @).
 * Target languages are now correctly handled when querying a pivot multilingual dictionary.
 *
 * Revision 1.1  2005/06/22 16:05:49  mangeot
 * *** empty log message ***
 *
 *
 *---------------------------------------------------------
 */

package fr.imag.clips.papillon.business.dictionary;

import fr.imag.clips.papillon.business.dictionary.VolumesFactory;

import org.apache.xml.utils.PrefixResolver;
import org.apache.xml.utils.PrefixResolverDefault;
import org.w3c.dom.Node;

public class DmlPrefixResolver implements PrefixResolver  {
	
	//FIXME: le préfixe de DML est d. Il faudrait choisir un préfixe plus élaboré genre
	// papillon_dml pour éviter les conflits avec des préfixes définis
	// dans d'autres dictionnaires
	
	public final static String DML_URI = "http://www-clips.imag.fr/geta/services/dml";
    public final static String XLINK_URI = "http://www.w3.org/1999/xlink";
    public final static String XMLSCHEMA_URI = "http://www.w3.org/2001/XMLSchema";
    public final static String XSLT_URI = "http://www.w3.org/1999/XSL/Transform";
	
	
	protected static PrefixResolver defaultPrefixResolver = null;
	
	public DmlPrefixResolver(Node rootElement) {
		defaultPrefixResolver = new PrefixResolverDefault(rootElement);
	}
	
	public java.lang.String getNamespaceForPrefix(java.lang.String prefix) {
		if (prefix.equals(VolumesFactory.DEFAULT_DML_PREFIX)) {
			return VolumesFactory.DML_URI;
		}
		else 
			return defaultPrefixResolver.getNamespaceForPrefix(prefix);
	}
	
	public java.lang.String getNamespaceForPrefix(java.lang.String prefix, Node context) {
		if (prefix.equals(VolumesFactory.DEFAULT_DML_PREFIX)) {
			return VolumesFactory.DML_URI;
		}
		else 
			return defaultPrefixResolver.getNamespaceForPrefix(prefix, context);
	}
	
	public java.lang.String getBaseIdentifier() {
		return defaultPrefixResolver.getBaseIdentifier();
	}
	
	public boolean handlesNullPrefixes() {
		return defaultPrefixResolver.handlesNullPrefixes();
	}
	
}
