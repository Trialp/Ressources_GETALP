package fr.imag.clips.papillon.business;

import com.lutris.appserver.server.httpPresentation.HttpPresentationException;

/**
 * Papillon exception : error during file import
 *
 * @author Olivier Tache
 * @version 1.0
 */
public class PapillonImportException extends HttpPresentationException {
    
    /**
     * Public constructor to initialize an exception with a user message.
     */
    public PapillonImportException(String msg) {
        super(msg);   
    }

    /**
     * Public constructor to initialize an exception with a user message
     * and an exception chain to follow
     */
    public PapillonImportException(String msg, Exception e) {
        super(msg, e);   
    }

}
