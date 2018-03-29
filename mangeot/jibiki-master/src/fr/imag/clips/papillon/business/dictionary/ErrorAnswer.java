//
//  ErrorAnswer.java
//  PapillonStable
//
//  Created by Mathieu Mangeot on 27/10/04.
//  Copyright 2004 __MyCompanyName__. All rights reserved.
//

package fr.imag.clips.papillon.business.dictionary;

public class ErrorAnswer implements IAnswer {

    public int getType() {
		return IAnswer.Error;
	}

   public  boolean isEmpty() {
	return true;
   }
    
    public Dictionary getDictionary() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}

    public String getDictionaryName() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}
    
    public String getDictionaryFullName() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}
    
    public String getVolumeName() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}

    public Volume getVolume() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}

    public String getSourceLanguage() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}
		    
    public String getXmlCode() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}
    
    public void setXmlCode(String code) throws fr.imag.clips.papillon.business.PapillonBusinessException {
		;
	}

    public String getId() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}

    public void setId(String newId) throws fr.imag.clips.papillon.business.PapillonBusinessException {
		;
	}

    public void setId() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		;
	}

    public String getHeadword() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}
	
    public org.w3c.dom.Document getDom() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}

    public void setDom(org.w3c.dom.Document myDoc) throws fr.imag.clips.papillon.business.PapillonBusinessException {
		;
	}

//    public org.w3c.dom.Document getHtmlDom() throws fr.imag.clips.papillon.business.PapillonBusinessException {
//		return null;
//	}
//
//    public void setHtmlDom(org.w3c.dom.Document myDoc) throws fr.imag.clips.papillon.business.PapillonBusinessException {
//		;
//	}
		

    public String getHandle() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return null;
	}
		
    public void delete() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		;
	}

    public boolean save() throws fr.imag.clips.papillon.business.PapillonBusinessException {
		return false;
	}

}
