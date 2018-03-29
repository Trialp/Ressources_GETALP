/*
 *  Jibiki
 *
 *  Enhydra super-servlet
 *
 *  © Francis Brunet-Manquat & Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *
 *  Abstract class implementing windows used to edit entries.
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.4  2006/04/06 15:06:39  fbrunet
 * New class 'creationEditInit' : create new entry
 * Modify LexALPEditEntry : only edit entry
 *
 * Revision 1.3  2006/03/13 08:48:00  fbrunet
 * bug corrections before merge
 *
 */

package fr.imag.clips.papillon.presentation;

// Papillon import
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.presentation.xhtml.orig.*;
import fr.imag.clips.papillon.presentation.HTMLTransformFactory;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
import com.lutris.appserver.server.httpPresentation.HttpPresentationRequest;

import com.lutris.appserver.server.httpPresentation.ClientPageRedirectException;
import com.lutris.appserver.server.Enhydra;
//import org.enhydra.xml.xmlc.XMLObject;
//import org.w3c.dom.html.*;

import org.w3c.dom.Node;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.html.HTMLCollection;
import org.w3c.dom.html.HTMLOptionElement;
import org.w3c.dom.html.HTMLSelectElement;

import org.enhydra.xml.xhtml.dom.*;
import org.enhydra.xml.xmlc.*;
//import org.enhydra.xml.xmlc.html.*;
import com.lutris.logging.*;
import com.lutris.util.*;

// For the cookies
import javax.servlet.http.Cookie;
import fr.imag.clips.papillon.business.user.UsersFactory;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;
import java.text.DateFormat;

//pour le debogage
import fr.imag.clips.papillon.business.utility.*;
import fr.imag.clips.papillon.business.PapillonLogger;

import org.enhydra.xml.io.OutputOptions;
import org.enhydra.xml.io.DOMFormatter;

/**
*  Description of the Class
 *
 * @author     francis
 * @created    January 31, 2005
 */
public abstract class EditingBasePO extends AbstractPO {
    
    /**
    *  Description of the Field
     */
    protected final static String LANG_PARAMETER = "lang";
    
    
    /**
    *  Description of the Field
     */
    public static int NO_SECTION = 0;
    

    /**
    *  Description of the Field  JavaScript to add in the header
    */
    protected XHTMLScriptElement headerScript = null;
    
    
    /**
     *  This is the procedure that is called when an HTML request occurs.
     *
     * @return                XMLObject The XMLObject (in HTML format) that is
     *      to be included in the standard layout.
     * @exception  Exception  Description of the Exception
     */
    public abstract Node getContent() throws Exception;
    
    
    /**
        *  This is the procedure that is called to know in what section we
     *  currently are in order for the header to be modified appropriately.
     *
     * @return    XMLObject The XMLObject (in HTML fomat) that is to be included
     *      in the standard layout.
     */
    protected abstract int getCurrentSection();
    
    
    /**
     *  Reference to the user logged in to the session
     */
    //protected User myUser = null;
    
    
    /**
     *  Sets the headerScript attribute of the EditingBasePO object
     *
     * @param  newScript                          The new headerScript value
     * @exception  PapillonPresentationException  Description of the Exception
     */
    public void setHeaderScript(XHTMLScriptElement newScript)
        throws PapillonPresentationException {
            this.headerScript = newScript;
        }
    
    
    /**
     *  Returns the complete document.
     *
     * @exception  Exception
     * @exception  HttpPresentationException  Description of the Exception
     * @exception  IOException                Description of the Exception
     */
    public Node getDocument()
        throws HttpPresentationException, IOException, Exception {
            try {
                // Creation du contenu
                //String lang = myGetParameter(LANG_PARAMETER);
                //if (null != lang) this.setUserPreferredLanguage(lang);
                
                // I put getContent before stdLayout because I can change dynamically the Lang
                Node content = getContent();
                
                // Creation du document vide
                //StdLayout stdLayout = StandardLayoutFactory.createStandardLayout(getComms(), this.getSessionData(), this.getUrl(), this.headerScript);
                //StdLayout stdLayout = new StdLayout(getComms(), this.getSessionData(), this.getUrl(), this.headerScript);
                        
            
                //String layoutClassName = ((Papillon)Enhydra.getApplication()).getLayoutClassName();
                EditingLayout layout = new EditingLayout();
                layout.initLayout(getComms(), this.getSessionData(), this.getUrl(), this.headerScript);
                
                // Affichage en debut de document d'un eventuel message administatif a l'utilisateur.
                handleAdminMessage(layout);
                
                // Affichage en debut de document d'un eventuel message a l'utilisateur.
                handleUserMessage(layout);
                
                //Insertion du contenu dans le document vide.
                layout.getContentPlaceHolder().appendChild(layout.getLayout().importNode(content, true));
                
                return (Node) layout.getLayout();
                
            } catch (java.lang.ClassNotFoundException e) {
                throw new HttpPresentationException("Could not find a suitable Layout. Please contact administrator ASAP.", e);
            } catch (java.lang.InstantiationException e) {
                throw new HttpPresentationException("Could not instanciate the Layout. Please contact administrator ASAP.", e);
            } catch (java.lang.IllegalAccessException e) {
                throw new HttpPresentationException("Illegal access exception while instanciating the Layout. Please contact administrator ASAP.", e);
            } catch (java.io.UnsupportedEncodingException e) {
                throw new HttpPresentationException("UnsupportedEncodingException while instanciating the Layout. Please contact administrator ASAP.", e);
            }
        }
    
    
    /**
     *  Method to put an eventual message for the user into the layout
     *
     * @param  stdLayout                      Description of the Parameter
     * @exception  HttpPresentationException  Description of the Exception
     */
    protected void handleUserMessage(StdLayout stdLayout) throws HttpPresentationException {
        // don't do anything if there is no message...
        String userMessageText = this.getSessionData().getUserMessage();
        if (userMessageText.length() != 0) {
            UserMessageXHTML messageBlock = (UserMessageXHTML) MultilingualXHtmlTemplateFactory.createTemplate("UserMessageXHTML", this.getComms(), this.getSessionData());
            messageBlock.setTextUserMessageText(userMessageText);
            this.getSessionData().flushUserMessage();
            stdLayout.getContentPlaceHolder().appendChild(stdLayout.getLayout().importNode(messageBlock.getElementUserMessage(), true));
        }
    }
    
    
    /**
     *  Description of the Method
     *
     * @param  stdLayout                      Description of the Parameter
     * @exception  HttpPresentationException  Description of the Exception
     */
    protected void handleAdminMessage(StdLayout stdLayout) throws HttpPresentationException {
        if (this.adminMessage != null && !this.adminMessage.equals("")) {
            AdminMessageXHTML messageBlock = (AdminMessageXHTML) MultilingualXHtmlTemplateFactory.createTemplate("AdminMessageXHTML", this.getComms(), this.getSessionData());
            messageBlock.setTextAdminMessageText(this.adminMessage);
            stdLayout.getContentPlaceHolder().appendChild(stdLayout.getLayout().importNode(messageBlock.getElementAdminMessage(), true));
        }
    }
    
    
    /**
        *  Gets the referrer request-header field that allows the client to specify,  
	 *  for the server's benefit, the address (URI) of the resource from  which the
	 *  Request-URI was obtained (the "referrer", although the  header field is misspelled.)
     *
     * @return  String  The referrer value
     */
    public String getReferrer() throws
        fr.imag.clips.papillon.business.PapillonBusinessException {
			String referrer = "";
			try {
				referrer = this.getComms().request.getHeader("Referer");
			}
			catch (com.lutris.appserver.server.httpPresentation.HttpPresentationException ex) {
				throw new fr.imag.clips.papillon.business.PapillonBusinessException ("Error in EditingBasePO.getReferrer: ", ex);
			}
            return referrer;
        }    
}