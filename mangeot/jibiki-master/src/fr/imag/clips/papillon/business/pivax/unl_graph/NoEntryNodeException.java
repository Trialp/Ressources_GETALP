/*
 *  unl_deconv_server
 *
 * UNL deconversion server © GETA-CLIPS-IMAG
 * Gilles Sérasset
 *-----------------------------------------------------------
 * $Id: NoEntryNodeException.java,v 1.1 2003/11/21 12:48:21 serasset Exp $
 *-----------------------------------------------------------
 * $Log: NoEntryNodeException.java,v $
 * Revision 1.1  2003/11/21 12:48:21  serasset
 * Collect more exceptions to get a result when deconverting
 *
 *
 */
package fr.imag.clips.papillon.business.pivax.unl_graph;

public class NoEntryNodeException extends java.lang.Exception {
    
    /**
    * Public constructor to initialize an exception with a user message.
     */
    public NoEntryNodeException(String msg) {
        super(msg);
    }
}

