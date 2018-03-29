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
 * Revision 1.8.4.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.8  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.7  2006/03/01 15:12:31  mangeot
 * Merge between maintrunk and LEXALP_1_1 branch
 *
 * Revision 1.6.2.1  2006/02/17 15:16:42  mangeot
 * Do not display the list of all XSL on the search form any more. Displays only a list of XSL descriptions
 *
 * Revision 1.6  2005/07/28 13:06:47  mangeot
 * - Added the possibility to export in PDF format. The conversion into PDF is don
 * e via the fop package that has to be installed (see ToolsForPapillon)
 *
 * Revision 1.5  2005/07/14 13:48:53  serasset
 * Added columns dictionaryname and volumename to the xslsheets.
 * Modified the XslSheet and XslSheetFactory accordingly.
 * Modified AdminXsl interface accordingly.
 * Modified DictionariesFactory and VolumesFactory to allow several xsl-sheets and to
 * correctly provide dictionaryName/volumeName to XslSheetFactory.
 * Cleanup of several errors in papillon_static doml file.
 *
 * Revision 1.4  2005/05/24 12:51:22  serasset
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
 * Revision 1.2.2.2  2005/03/29 09:41:33  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.2.2.1  2005/03/16 13:24:31  serasset
 * Modified all boolean fields in table to CHAR(1) in order to be more db independant.
 * Suppressed ant.jar from class path, informationfiles (which rely on it) should be corrected.
 * The version of Xerces is now displayed on application init.
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:38  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.xsl;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.CurrentDBTransaction;

import java.io.StringReader;

import javax.xml.transform.Transformer;
import javax.xml.transform.Templates;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.stream.StreamSource;

import com.lutris.appserver.server.sql.DatabaseManagerException;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.dods.builder.generator.query.DataObjectException;

// Enhydra SuperServlet imports

public class XslSheet {
	
	public final static String DEFAULT_view = "DEFAULT";
	public final static String FO_view = "FormatingObject";
	public final static String TEXT_view = "TEXT";
	public final static String TEXT_suffix = "-text";
	public final static String XML_view = "XML";
    
	protected XslSheetDO myDO = null;
    protected Templates myTemplate = null;

    
    /**
     * The public constructor.
     */
    public XslSheet() 
        throws PapillonBusinessException {
            try {
                this.myDO = XslSheetDO.createVirgin(CurrentDBTransaction.get());  
            } catch(DatabaseManagerException ex) {
                throw new PapillonBusinessException("Error creating empty XslSheet", ex);
            } catch(ObjectIdException ex) {
                throw new PapillonBusinessException("Error creating empty XslSheet", ex);
            }
        }

    /** The protected constructor
     *
     * @param theDisc. The data object of the message.
     */
    protected XslSheet(XslSheetDO theSheet) 
        throws PapillonBusinessException  {
            try {
                
                //
                this.myDO = theSheet;
                try {
                    Thread currentThread = Thread.currentThread();
                    currentThread.setContextClassLoader(JibikiXsltExtension.class.getClassLoader());
                } catch (SecurityException e) {
                    PapillonLogger.writeDebugMsg("Could not modify the context class loader.");
                }
                // Create template
                TransformerFactory myTransformerFactory = TransformerFactory.newInstance();
                myTemplate = myTransformerFactory.newTemplates(new StreamSource(new StringReader (this.getCode())));
				//PapillonLogger.writeDebugMsg("template created: " + this.getName() + " " + this.getHandle());

				if (myTemplate==null) {PapillonLogger.writeDebugMsg("template null");}
                
            } catch(javax.xml.transform.TransformerConfigurationException ex) {
                throw new PapillonBusinessException("Error creating XslSheet", ex);
            }
        }

    /**
     * Gets the object id for the message
     *
     * @return the object id.
     * @exception PapillonBusinessException if an error occurs
     *   retrieving data (usually due to an underlying data layer
     *   error).
     */
	  
	 public boolean isEmpty()
	 {
	 return (this.myDO==null) ;
	 }
	  
    public String getHandle()
        throws PapillonBusinessException {
        try {
            return this.myDO.getHandle();
        } catch(DatabaseManagerException ex) {
            throw new PapillonBusinessException("Error getting XslSheet's handle", ex);
        }
    }
    
    
////////////////////////// data member Name


   /**
    * Get name of the xslsheets
    *
    * @return name of the xslsheets
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
    public String getName () throws PapillonBusinessException {
        try {
            return this.myDO.getName();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting XslSheet's name", ex);
        }
    }

   
   /**
    * Set name of the xslsheets
    *
    * @param name of the xslsheets
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
    public void setName ( String name ) throws PapillonBusinessException {
        try {
            myDO.setName(name);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting XslSheet's name", ex);
        }
    }
 
    
    /**
        * Get dictionaryName of the xslsheets
     *
     * @return dictionaryName of the xslsheets
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public String getDictionaryName () 
        throws PapillonBusinessException 
    {
        try {
            return this.myDO.getDictionaryName();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting XslSheet's dictionary name", ex);
        }
    }
    
    
    /**
        * Set dictionaryName of the xslsheets
     *
     * @param dictionaryName of the xslsheets
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setDictionaryName ( String dictionaryName )
        throws PapillonBusinessException 
    {
        try {
            myDO.setDictionaryName(dictionaryName);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting XslSheet's dictionary name", ex);
        }
    }
    
    
    ////////////////////////// data member VolumeName
    
    /**
        * Get volumeName of the xslsheets
     *
     * @return volumeName of the xslsheets
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public String getVolumeName () 
        throws PapillonBusinessException
    {
        try {
            return this.myDO.getVolumeName();
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting XslSheet's volume name", ex);
        }
    }
    
    
    /**
        * Set volumeName of the xslsheets
     *
     * @param volumeName of the xslsheets
     *
     * @exception DataObjectException
     *   If the object is not found in the database.
     */
    public void setVolumeName ( String volumeName )
        throws PapillonBusinessException 
    {
        try {
            myDO.setVolumeName(volumeName);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting XslSheet's volume name", ex);
        }
    }
    
    
    
////////////////////////// data member Default
 public boolean isDefaultxsl() throws PapillonBusinessException {
        try {
            String def =  this.myDO.getDefaultxsl();
            return ((def != null) && (def.equals("Y")));
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting XslSheet's defaultxsl", ex);
        }
    }

 
    public void setDefaultxsl (boolean Defaultxsl) throws PapillonBusinessException {
        try {
            myDO.setDefaultxsl(Defaultxsl ? "Y" : "N");   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting XslSheet's defaultxsl", ex);
        }
    }
   
   
    ////////////////////////// data member userxsl
    public boolean isExternalxsl() throws PapillonBusinessException {
        try {
            String isExternalxsl =  this.myDO.getExternalxsl();
            return ((isExternalxsl != null) && (isExternalxsl.equals("Y")));
        } catch(DataObjectException  ex) {
            throw new PapillonBusinessException("Error getting XslSheet's externalxsl", ex);
        }
    }
    
    
    public void setExternalxsl(boolean externalxsl) throws PapillonBusinessException {
        try {
            myDO.setExternalxsl(externalxsl ? "Y" : "N");   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting XslSheet's userxsl", ex);
        }
    }
    
    
////////////////////////// data member Description


   /**
    * Get description of the xslsheets
    *
    * @return description of the xslsheets
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
   public String getDescription () throws PapillonBusinessException {
        try {
            return this.myDO.getDescription();
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting XslSheet's description", ex);
        }
    }


   
   /**
    * Set description of the xslsheets
    *
    * @param description of the xslsheets
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
   public void setDescription (String description) throws PapillonBusinessException {
        try {
            myDO.setDescription(description);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting XslSheet's description", ex);
        }
    }
 
   
////////////////////////// data member Code


   /**
    * Get code of the xslsheets
    *
    * @return code of the xslsheets
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
   public String getXmlCode () throws PapillonBusinessException {
        try {
            return this.myDO.getCode();
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting XslSheet's code", ex);
        }
    }

   public String getCode () throws PapillonBusinessException {
        try {
            return this.myDO.getCode();
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error getting XslSheet's code", ex);
        }
    }

   
   /**
    * Set code of the xslsheets
    *
    * @param code of the xslsheets
    *
    * @exception DataObjectException
    *   If the object is not found in the database.
    */
   public void setXmlCode ( String code )
   throws PapillonBusinessException {
        try {
            myDO.setCode(code);   
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting XslSheet's code", ex);
        }
    }
   public void setCode ( String code )
   throws PapillonBusinessException {
        try {
            
            //
            myDO.setCode(code);   
            
            // Modify template
            TransformerFactory myTransformerFactory = TransformerFactory.newInstance();
            myTemplate = myTransformerFactory.newTemplates(new StreamSource(new StringReader (this.getCode())));
            
        } catch(DataObjectException ex) {
            throw new PapillonBusinessException("Error setting XslSheet's code", ex);
        } catch(javax.xml.transform.TransformerConfigurationException ex) {
            throw new PapillonBusinessException("Error modifying xsl template", ex);
        }
    }
 
       public void save() 
        throws PapillonBusinessException {
			// reset XslSheetFactory cache
			//XslSheetFactory.resetCache();
        try {
            this.myDO.commit();
        } catch(Exception ex) {
            throw new PapillonBusinessException("Error saving XslSheet", ex);
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
            throw new PapillonBusinessException("Error deleting XslSheet", ex);
        }
    }
     
    
    ////////////////////////// Transformer
    
    
    /**
     * Get transformer of the xslsheets
     *
     * @return transformer of the xslsheets
     * @throws PapillonBusinessException if any problem arise during transformer creation.
     */
    public Transformer getTransformer () 
        throws PapillonBusinessException {
            try {
                
                //
				//PapillonLogger.writeDebugMsg("create new transformer: " + this.getName() + " " + this.getHandle());
                return this.myTemplate.newTransformer();
            
            } catch(Exception ex) {
                throw new PapillonBusinessException("Error getting transformer", ex);
            }
        }

    
    
    // A FAIRE method transform !!
}
