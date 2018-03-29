/*
 *  Jibiki
 *
 *  Enhydra super-servlet presentation object
 *
 * © Francis Brunet-Manquat, Mathieu Mangeot et Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.2  2007/02/28 16:19:51  fbrunet
 * *** empty log message ***
 *
 * Revision 1.1  2007/02/28 09:27:07  fbrunet
 * Added ajax method to AdvancedQueryForm page
 *
 */
 
package fr.imag.clips.papillon.presentation;

//
import java.util.Iterator;
import java.util.Collection;
import java.util.TreeSet;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;

//import org.enhydra.xml.xmlc.XMLObject;
import org.enhydra.xml.xhtml.dom.XHTMLOptionElement;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

// internal imports
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.presentation.xhtml.orig.*;

import java.sql.ResultSet;
import java.sql.SQLException;
import com.lutris.appserver.server.sql.DBConnection;
import com.lutris.appserver.server.Enhydra;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;


public class AdvancedQueryFormAjax extends AbstractPO {
    
    //
    private final boolean DEBUG = false;
        
    //
    public static String ID_PARAMETER = "id";
    public static String VALUE_PARAMETER = "value";
    
    //
    protected boolean loggedInUserRequired() {
        return false;
    }
	
    //
    protected boolean userMayUseThisPO() {
        return true;
    }
    
	//
    public Node getDocument() throws java.io.UnsupportedEncodingException, HttpPresentationException {
			
        //
        if (DEBUG) PapillonLogger.writeDebugMsg ("AdvancedQueryFormAjax : getContent");   
    
        //
        String id = myGetParameter(ID_PARAMETER);
        String value = myGetParameter(VALUE_PARAMETER);
        if (DEBUG) PapillonLogger.writeDebugMsg ("AdvancedQueryFormAjax : id=" + id + " & value=" + value);
        
        //
        AdvancedQueryFormAjaxXHTML content = (AdvancedQueryFormAjaxXHTML) MultilingualXHtmlTemplateFactory.createTemplate("AdvancedQueryFormAjaxXHTML", this.myComms, this.sessionData);

        //
        Node selectNode = null;
        try {
            
            // Modify XHTML
            selectNode = (Node)content.getElementSelectNode();
            Node nameAttribut = selectNode.getAttributes().getNamedItem("name");
            nameAttribut.setNodeValue(id);
            
            //
            XHTMLOptionElement templateOption = (XHTMLOptionElement)content.getElementTemplate();
            for (Iterator iter = getSelectElementCollection(value).iterator(); iter.hasNext();) {
                String myString = (String)iter.next();
                
                //
                templateOption.setValue(myString);
                templateOption.setLabel(myString);
                Text txt = (Text) templateOption.getFirstChild();
                txt.setData(myString);
                templateOption.getParentNode().appendChild(templateOption.cloneNode(true));
            }
            templateOption.getParentNode().removeChild(templateOption);
            
        } catch(SQLException e) {
            e.printStackTrace();            
        }        
       
        //
        return selectNode;
    }
    
    
    //
    private Collection getSelectElementCollection(String CDM_value) throws SQLException {
        
        //
        DBConnection myDBConnection = null;
        ResultSet myResultSet = null;
        TreeSet myTreeSet = new TreeSet();
        
        try {

            myDBConnection = Enhydra.getDatabaseManager().allocateConnection();
            
            for (Iterator iter = VolumesFactory.getVolumes().iterator(); iter.hasNext();) {
                Volume volume = (Volume)iter.next();
                String sqlQuery = "SELECT DISTINCT value FROM " + volume.getIndexDbname() + " WHERE key='" + CDM_value + "';";
                
                myResultSet = myDBConnection.executeQuery(sqlQuery);
                while(myResultSet.next()){
                    myTreeSet.add(myResultSet.getString(1));
                }
                myResultSet.close();
            }
            
        }  catch(PapillonBusinessException e) {
            PapillonLogger.writeDebugMsg("Exception in AdvancedQueryFormAjax : " + e.toString());
            
        }  catch(SQLException se) {
            se.printStackTrace();
            //very important to throw out bad connections
            if(myDBConnection.handleException(se)) myDBConnection=null;
            
        } catch(Exception e) {
            e.printStackTrace();
            
        } finally {
            if(myDBConnection!=null) {
                myDBConnection.reset();
                myDBConnection.release();
            }
        }
        
        //
        return myTreeSet;
    }
    
}
