/*
 *  Created by Mathieu Mangeot on 28/04/05.
 *  Copyright 2005 Mangeot-Sérasset. All rights reserved.
 ------------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.5  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.4  2006/04/10 12:19:17  mangeot
 * XML indented or not
 *
 * Revision 1.3  2005/07/28 13:06:47  mangeot
 * - Added the possibility to export in PDF format. The conversion into PDF is don
 * e via the fop package that has to be installed (see ToolsForPapillon)
 *
 * Revision 1.2  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.1.2.2  2005/05/20 16:54:53  mangeot
 * Added ExportVolume functionnality
 *
 * Revision 1.1.2.1  2005/05/20 14:44:29  mangeot
 * *** empty log message ***
 *
 *
 *-----------------------------------------------
 */ 

package fr.imag.clips.papillon.business.dictionary;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.xml.XMLServices;

public class ExportVolumeEntryProcessor implements IVolumeEntryProcessor {

	protected static final String XMLFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XML_DIALECT);
	protected static final String XMLIndentFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XMLIndent_DIALECT);
	protected static final String XHTMLFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.XHTML_DIALECT);
	protected static final String TEXTFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.PLAINTEXT_DIALECT);
	protected static final String PDFFormat = Integer.toString(fr.imag.clips.papillon.business.transformation.ResultFormatterFactory.PDF_DIALECT);
	protected static java.io.OutputStream outStream = null;
	protected static String outputFormat = "";

	public ExportVolumeEntryProcessor(java.io.OutputStream newOutStream) {
		outStream = newOutStream;
	}
	
	public ExportVolumeEntryProcessor(String outFormat, java.io.OutputStream newOutStream) {
		outputFormat = outFormat;
		outStream = newOutStream;
	}
	
	public void process(VolumeEntry myEntry) throws PapillonBusinessException {
		Volume myVolume = myEntry.getVolume();
		String resultString = "";
		
		if (outputFormat != null && outputFormat.equals(XHTMLFormat)) {
			resultString = fr.imag.clips.papillon.business.transformation.XslTransformation.applyXslSheetsAndSerialize((IAnswer) myEntry);
		}
		else if (outputFormat != null && outputFormat.equals(TEXTFormat)) {
			resultString = fr.imag.clips.papillon.business.transformation.XslTransformation.applyXslSheetsForText((IAnswer) myEntry);
		}
		else if (outputFormat != null && outputFormat.equals(PDFFormat)) {
			org.w3c.dom.Element myElement = fr.imag.clips.papillon.business.transformation.XslTransformation.applyXslSheetsForFo((IAnswer) myEntry);
			resultString = XMLServices.ElementToString(myElement, false, false);
		}
		else { // (outputFormat == null || outputFormat.equals("") || outputFormat.equals(XMLFormat))
			String contribString = myVolume.getCdmContribution();
			String xmlCode;
			if (outputFormat.equals(XMLIndentFormat)) {
				 xmlCode = XMLServices.DocumentToString(myEntry.getDom(), false, true);
			}
			else {
				xmlCode = XMLServices.DocumentToString(myEntry.getDom(), false, false);
			}
			if (xmlCode.indexOf("<" + contribString)>0) {
				xmlCode= xmlCode.substring(xmlCode.indexOf("<" + contribString)); 
				String endTag = "</" + contribString + ">";
				xmlCode = xmlCode.substring(0,xmlCode.indexOf(endTag) + endTag.length());
				xmlCode += "\n";
			}
			resultString = xmlCode;
		}
		try {
			resultString = Utility.trimXMLHeader(resultString);
			outStream.write(resultString.getBytes("UTF-8"));	
		}
		catch (Exception ex) {
			throw new PapillonBusinessException("Error in writing an UTF-8 String: ", ex);
		}
	}

}
