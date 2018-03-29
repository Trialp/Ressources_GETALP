/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: UniversalWordFactory.java,v 1.1.1.1 2003/10/13 14:01:13 serasset Exp $
 *-----------------------------------------------------------
 * $Log: UniversalWordFactory.java,v $
 * Revision 1.1.1.1  2003/10/13 14:01:13  serasset
 * Creation of unldeco CVS repository.
 *
 */
package fr.imag.clips.papillon.business.pivax.unl_graph;

import java.io.*;

public class UniversalWordFactory {
    
    public static UniversalWord createUniversalWord(String str)
    throws fr.imag.clips.papillon.business.pivax.unl_graph.ParseException {
        // We have to parse the uw in order to standardize it.
        UNLParser parser = new UNLParser(new BufferedReader(new StringReader(str)));
        UniversalWord uw = parser.universalWord();
        return uw;
    }

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String filedicpath= args[0];
		File file = new File(filedicpath);
		try {
			BufferedReader input =  new BufferedReader(new FileReader(file));
	        String line = null; 
	        while (( line = input.readLine()) != null){
	        	UniversalWord uw = createUniversalWord(line);
	        	System.out.println(uw.toStringTree());
	        }
		} catch (IOException ex){
			ex.printStackTrace();
		} catch (ParseException ex) {
			ex.printStackTrace();
		}
	}
    
}
