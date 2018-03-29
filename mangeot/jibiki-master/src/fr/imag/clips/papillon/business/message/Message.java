/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.3  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.2.2.1  2005/03/29 09:41:33  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.message;

import fr.imag.clips.papillon.CurrentDBTransaction;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.PapillonBusinessException;

import com.lutris.appserver.server.sql.DatabaseManagerException;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.dods.builder.generator.query.DataObjectException;

/**
 * Represents a Mailing list message. 
 */
public class Message {
    /**
     * The DO of the message.
     */
    protected MessageDO myDO = null;

    /**
     * The public constructor.
     */
    public Message() throws PapillonBusinessException {
        try {
            this.myDO = MessageDO.createVirgin(CurrentDBTransaction.get());  
        } catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error creating empty Message", ex);
        } catch(ObjectIdException ex) {
            throw new PapillonBusinessException("Error creating empty Message", ex);
        }
    }

    /** The protected constructor
     *
     * @param theDisc. The data object of the message.
     */
    protected Message(MessageDO theMessage) 
        throws PapillonBusinessException  {
        this.myDO = theMessage;
    }

    /**
     * Gets the object id for the message
     *
     * @return the object id.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getHandle()
        throws PapillonBusinessException {
        try {
            return this.myDO.getHandle();
        } catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error getting messages's handle", ex);
        }
    }

    /**
     * Gets the subject of the message
     *
     * @return the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getSubject() 
        throws PapillonBusinessException {
        try {
            return myDO.getSubject();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting message's subject", ex);
        }
    }
    
    /**
     * Gets the msgid for the message
     *
     * @return the msgid.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getMsgid()
        throws PapillonBusinessException {
        try {
            return myDO.getMsgid();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting message's Msgid", ex);
        }
    }
    
    /**
     * Gets the body for the message
     *
     * @return the body.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getMsg()
        throws PapillonBusinessException {
        try {
            //byte [] buf = this.myDO.getMsg().getBytes();
            //return new String(buf, "UTF-8");
	    return this.myDO.getMsg();
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting message's body", ex);
        } //catch(java.io.UnsupportedEncodingException ex) {
          //  throw new PapillonBusinessException("Error getting Dictionary XmlCode", ex);
        //}
    }

    /**
     * Gets the date for the message
     *
     * @return the Date.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public java.sql.Date getDate()
        throws PapillonBusinessException {
        try {
            return myDO.getDate();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting message's date", ex);
        }
    }
    
    /**
     * Gets the author of the message
     *
     * @return the Author.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getAuthor()
        throws PapillonBusinessException {
        try {            
            return myDO.getAuthor();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting message's author", ex);
        }
    }

    /**
     * Gets the author address of the message
     *
     * @return the Author.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getAuthorAddress()
        throws PapillonBusinessException {
        try {            
            return myDO.getAuthorAddress();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting message's author", ex);
        }
    }
    
    /**
     * Gets the thread of the message
     *
     * @return the Thread.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public String getThread()
        throws PapillonBusinessException {
        try {            
            return myDO.getThread();   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting message's thread", ex);
        }
    }

    // Setting the values
    /**
     * Set the subject of the message
     *
     * @param the subject.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public void setSubject(String subject) 
        throws PapillonBusinessException {
        try {
            myDO.setSubject(subject);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting message's subject", ex);
        }
    }
    
    /**
     * Sets the msgid for the message
     *
     * @param the msgid.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public void setMsgid(String msgid)
        throws PapillonBusinessException {
        try {
            myDO.setMsgid(msgid);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting message's Msgid", ex);
        }
    }
    
    /**
     * Sets the body for the message
     *
     * @param the body.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public void setMsg(String body)
        throws PapillonBusinessException {
        try {
             myDO.setMsg(body);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting message's body", ex);
        }
    }

    /**
     * Gets the date for the message
     *
     * @param the Date.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public void setDate(java.sql.Date date)
        throws PapillonBusinessException {
        try {
             myDO.setDate(date);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting message's date", ex);
        }
    }
    
    /**
     * Gets the author for the message
     *
     * @param the author.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public void setAuthor(String author)
        throws PapillonBusinessException {
        try {
             myDO.setAuthor(author);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting message's author", ex);
        }
    }
    
     /**
     * Sets the author address for the message
     *
     * @param the author address.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public void setAuthorAddress(String address)
        throws PapillonBusinessException {
        try {
             myDO.setAuthorAddress(address);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting message's author", ex);
        }
    }
    
   
   /**
     * Gets the thread for the message
     *
     * @param the thread.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public void setThread(String thread)
        throws PapillonBusinessException {
        try {
             myDO.setThread(thread);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting message's thread", ex);
        }
    }


    /**
     * Commits all changes to the database.
     *
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
    public void save() 
        throws PapillonBusinessException {
        try {
            this.myDO.commit();
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error saving message", ex);
        }
    }
    
    /**
     * Deletes the message from the database.
     *
     * @exception PapillonBusinessException if an error occurs
     *   deleting data (usually due to an underlying data layer
     *   error).
     */
    public void delete() 
        throws PapillonBusinessException {
        try {
            this.myDO.delete();
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error deleting message", ex);
        }
    }
}
