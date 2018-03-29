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
 * Revision 1.4  2005/05/24 12:51:21  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.3  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.2.2.1  2005/03/29 09:41:32  serasset
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

package fr.imag.clips.papillon.business.informationfile;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.CurrentDBTransaction;

import com.lutris.appserver.server.sql.DatabaseManagerException;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.dods.builder.generator.query.DataObjectException;

// Enhydra SuperServlet imports

public class InformationDocument {
    protected InformationDocumentDO myDO = null;
    
    /**
     * The public constructor.
     */
    public InformationDocument() throws PapillonBusinessException {
        try {
            this.myDO = InformationDocumentDO.createVirgin(CurrentDBTransaction.get());  
        } catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error creating empty InformationDocument", ex);
        } catch(ObjectIdException ex) {
            throw new PapillonBusinessException("Error creating empty InformationDocument", ex);
        }
    }

    /** The protected constructor
     *
     * @param theDisc. The data object of the message.
     */
    protected InformationDocument(InformationDocumentDO theFile) 
        throws PapillonBusinessException  {
        this.myDO = theFile;
    }

    /**
     * Gets the object id for the message
     *
     * @return the object id.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	  
    public boolean isEmpty() {
        return (this.myDO==null) ;
    }
	  
    public String getHandle()
        throws PapillonBusinessException {
        try {
            return this.myDO.getHandle();
        } catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error getting InformationDocument's handle", ex);
        }
    }
////////////////////////// data member Title


   /**
    * Get title of the InformationDocument
    *
    * @return title of the InformationDocument
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
    public String getTitle () throws PapillonBusinessException {
	try {
            // I have to convert it back to UTF8 as it seems that either enhydra, or the JDBC driver is
            // wrong.
	    //    byte [] buf = this.myDO.getTitle().getBytes();
	    //return new String(buf, "UTF-8");

	    return this.myDO.getTitle();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting InformationDocument's title", ex);
        }// catch(java.io.UnsupportedEncodingException ex) {
	 //   throw new PapillonBusinessException("Error getting InformationDocument's title", ex);
 	//}
    }

   
   /**
    * Set title of the InformationDocument
    *
    * @param title of the InformationDocument
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
    public void setTitle ( String title ) throws PapillonBusinessException {
        try {
            myDO.setTitle(title);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's filename", ex);
        }
    }

////////////////////////// data member author

    /**
        * Get author(s) of the InformationDocument
     *
     * @return author(s) of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public String getAuthor () throws PapillonBusinessException {
        try {
	    //            byte [] buf = this.myDO.getAuthor().getBytes();
	    //            return new String(buf, "UTF-8");
            return this.myDO.getAuthor();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting InformationDocument's Author", ex);
        } //catch(java.io.UnsupportedEncodingException ex) {
          //  throw new PapillonBusinessException("Error getting InformationDocument's Author", ex);
        //}
    }


    /**
        * Set author of the InformationDocument
     *
     * @param author of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setAuthor ( String author ) throws PapillonBusinessException {
        try {
            myDO.setAuthor(author);
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's Author", ex);
        }
    }

    /**
     * Get owner of the InformationDocument
     *owner     
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public String getOwner()
        throws PapillonBusinessException {
            try {
                return myDO.getOwner();
            } catch(DataObjectException ex) {
                throw new PapillonBusinessException("Error getting user's owner", ex);
            }
        }
    /**
        * Set owner of the InformationDocument
     *owner     * @param author of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setOwner ( String owner ) throws PapillonBusinessException {
        try {
            myDO.setOwner(owner);
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's owner", ex);
        }
    }

    /**
        * Get reference of the InformationDocument
     *
     * @return reference of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public String getReference () throws PapillonBusinessException {
        try {
            //byte [] buf = this.myDO.getReference().getBytes();
            //return new String(buf, "UTF-8");
	    return this.myDO.getReference();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting InformationDocument's Reference", ex);
        } //catch(java.io.UnsupportedEncodingException ex) {
          //  throw new PapillonBusinessException("Error getting InformationDocument's Reference", ex);
        //}
    }


    /**
        * Set reference of the InformationDocument
     *
     * @param reference of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setReference ( String reference ) throws PapillonBusinessException {
        try {
            myDO.setReference(reference);
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's Reference", ex);
        }
    }

    ////////////////////////// data member CreationDate

   /**
    * Get CreationDate of the InformationDocument
    *
    * @return CreationDate of the InformationDocument
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
    public java.sql.Date getCreationDate () throws PapillonBusinessException {
        try {
            return this.myDO.getCreationDate();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting InformationDocument's FileType", ex);
        }
    }

   
   /**
    * Set CreationDate of the InformationDocument
    *
    * @param CreationDate of the InformationDocument
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
    public void setCreationDate ( java.sql.Date date ) throws PapillonBusinessException {
        try {
            myDO.setCreationDate(date);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's date", ex);
        }
    }

    /**
        * Set CreationDate of the InformationDocument
     *
     * @param CreationDate (as a java.util.Date) of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setCreationDate ( java.util.Date date ) throws PapillonBusinessException {
        try {
            myDO.setCreationDate(new java.sql.Date(date.getTime()));
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's date", ex);
        }
    }

    /**
        * Set CreationDate of the InformationDocument
     *
     * @param CreationDate (as a String) of the InformationDocument
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setCreationDate (String date) throws PapillonBusinessException {
        try {
            java.text.SimpleDateFormat myDateFormat = new java.text.SimpleDateFormat("dd/MM/yyyy");
            java.util.Date myDate = myDateFormat.parse(date);
            myDO.setCreationDate(new java.sql.Date(myDate.getTime()));
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's date", ex);
        }
    }

    
////////////////////////// data member Section

   /**
    * Get Section of the InformationDocument
    *
    * @return Section of the InformationDocument
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
    public String getSection () throws PapillonBusinessException {
        try {
            return this.myDO.getSection();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting InformationDocument's FileCode", ex);
        }
    }

   
   /**
    * Set Section of the InformationDocument
    *
    * @param Section of the InformationDocument
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
    public void setSection ( String section ) throws PapillonBusinessException {
        try {
            myDO.setSection(section);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting InformationDocument's FileCode", ex);
        }
    }

 
    public void save() 
        throws PapillonBusinessException {
        try {
            this.myDO.commit();
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error saving InformationDocument", ex);
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
            throw new PapillonBusinessException("Error deleting InformationDocument", ex);
        }
    }
     
   

}
