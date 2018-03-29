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
 * Revision 1.4  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.3  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.2  2004/12/24 08:57:44  serasset
 * Premiere version de l'interface avec fond papillon et transparence.
 *
 * Revision 1.1.1.1  2004/12/06 16:38:42  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;
import com.lutris.dods.builder.generator.query.DataObjectException;
// Standard imports
import java.io.IOException;
import java.util.Date;
import java.text.DateFormat;


// Other imports
import java.util.*;
import java.io.*;

//import com.lutris.xml.xmlc.*;
import com.lutris.appserver.server.httpPresentation.*;
//import org.w3c.dom.html.*;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.*;

import org.enhydra.xml.io.OutputOptions;
import org.enhydra.xml.io.DOMFormatter;
// Imported JAVA API for XML Parsing classes
//import javax.xml.parsers.DocumentBuilder;
//import javax.xml.parsers.DocumentBuilderFactory;
//import javax.xml.parsers.ParserConfigurationException;
//import org.xml.sax.InputSource;
//import org.xml.sax.SAXException;

import fr.imag.clips.papillon.business.message.*;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.PapillonBusinessException;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

import org.w3c.dom.CDATASection;

public class MailingList extends PapillonBasePO {
    protected final static String ID="Id";
    protected final static String BODYPLACE="@@MESSAGE_BODY@@";
    protected final static String SORT_PARAMETER="sort";
    protected final static String THREAD_PARAMETER="thread";
    protected final static String AUTHOR_CONTAINS_PARAMETER="authorcontains";
    protected final static String SUBJECT_CONTAINS_PARAMETER="subjectcontains";
    protected final static String MESSAGE_CONTAINS_PARAMETER="messagecontains";
    protected final static String OFFSET_PARAMETER="offset";
    protected final static String LIMIT_PARAMETER="limit";
    protected final static String NEXT_BUTTON_PARAMETER="next.x";
    protected final static String PREVIOUS_BUTTON_PARAMETER="previous.x";

    protected final static int DEFAULT_LIMIT = 20;
    protected final static int DEFAULT_OFFSET = 0;

    protected  int getCurrentSection() {
        return INFORMATIONS_SECTION;
    }

    protected boolean loggedInUserRequired() {
        return false;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }
    
    public Node getMessageList() throws HttpPresentationException, 
                                        IOException, 
                                        DataObjectException {
                                        
        // On regarde d'abord les paramètres qui nous sont demandés.
        String sort = this.getComms().request.getParameter(SORT_PARAMETER);
        String thread = myGetParameter(THREAD_PARAMETER);
        
        String authorcontains = myGetParameter(AUTHOR_CONTAINS_PARAMETER);
        String subjectcontains = myGetParameter(SUBJECT_CONTAINS_PARAMETER);
        String messagecontains = myGetParameter(MESSAGE_CONTAINS_PARAMETER);
        String offsetString = myGetParameter(OFFSET_PARAMETER);
        String limitString = myGetParameter(LIMIT_PARAMETER);
        String nextButton = myGetParameter(NEXT_BUTTON_PARAMETER);
        String previousButton = myGetParameter(PREVIOUS_BUTTON_PARAMETER);

        int offset = DEFAULT_OFFSET;
        if (offsetString != null) {
            try {
                offset = Integer.parseInt(offsetString);
            } catch (java.lang.NumberFormatException e) {
                // nothing...
            }
        }
        
        int limit = DEFAULT_LIMIT;
        if (limitString != null) {
            try {
                limit = Integer.parseInt(limitString);
            } catch (java.lang.NumberFormatException e) {
                // nothing...
            }
        }
        
        if (null != nextButton) {
            // if the next button has been pushed, get the next page...
            offset = offset + limit;
        } else if (null != previousButton) {
            // if the previous button has been pushed, get the previous page...
            if (offset > limit) offset = offset-limit; else offset=0;
        } else {
            // else, it's a new request => give the first page
            offset = 0;
        }

        Message[] theMessageArray;
        
        theMessageArray = MessageFactory.getMessagesArray(authorcontains, subjectcontains, messagecontains,
                                                          thread, sort, offset, limit);
        int nbMsg = MessageFactory.getMessagesCount(authorcontains, subjectcontains, messagecontains,
                                                    thread, sort);
        
        //Pass the array to the formatter
        MessageListXHTML messageList = getMessageList(theMessageArray);
        
        //Then, set the appropriate values in the form elements
        XHTMLInputElement authorContainsInput = messageList.getElementAuthorContains();
        XHTMLInputElement subjectContainsInput = messageList.getElementSubjectContains();
        XHTMLInputElement messageContainsInput = messageList.getElementMessageContains();
        XHTMLOptionElement dateOption= messageList.getElementSortByDate();
        XHTMLOptionElement subjectOption= messageList.getElementSortBySubject();
        XHTMLOptionElement authorOption= messageList.getElementSortByAuthor();

        XHTMLInputElement currentLimitInput = messageList.getElementCurrentLimit();
        XHTMLInputElement currentOffsetInput = messageList.getElementCurrentOffset();
        
        XHTMLInputElement buttonPrevious = messageList.getElementButtonPrevious();
        XHTMLInputElement buttonNext = messageList.getElementButtonNext();
        
        authorContainsInput.setValue(authorcontains);
        subjectContainsInput.setValue(subjectcontains);
        messageContainsInput.setValue(messagecontains);
        currentLimitInput.setValue(limitString);
        currentOffsetInput.setValue(String.valueOf(offset));
        
        if ((null != sort) && (sort.equals("author"))) {
            authorOption.setSelected(true);
        } else if ((null != sort) && (sort.equals("subject"))) {
            subjectOption.setSelected(true);
        } else {
            dateOption.setSelected(true);
        }
        
        // Remove the previous button if we are already presenting the first message
        if (0 >= offset) {
            Node parent=buttonPrevious.getParentNode();
            parent.removeChild(buttonPrevious);
        }
        
        // remove the next button if we are already presenting the last message.
        if (offset+limit >= nbMsg) {
            Node parent=buttonNext.getParentNode();
            parent.removeChild(buttonNext);
        }
        
        // show the range of the currently displayed messages
        int lastMsg = (offset+limit >= nbMsg) ? nbMsg : (offset + limit);
        int firstMsg = offset + 1;
        messageList.setTextMessageRange(firstMsg + "-" + lastMsg);
        
        return messageList.getElementDiscussion();
    }
    
    public MessageListXHTML getMessageList(Message[] theMessageArray)
        throws HttpPresentationException, IOException, DataObjectException
    {
        MessageListXHTML messageList;
                
        // On initialise le layout
        messageList = (MessageListXHTML)MultilingualXHtmlTemplateFactory.createTemplate("MessageListXHTML", this.getComms(), this.getSessionData());

        // On récupère les éléments du layout
        XHTMLTableRowElement templateRow = messageList.getElementTemplateRow();
        XHTMLElement dateCellTemplate = messageList.getElementDate();
        XHTMLInputElement autInput = messageList.getElementAuthorInput();
        XHTMLElement auteurCellTemplate = messageList.getElementAuthor();
        XHTMLAnchorElement sujAnchor = messageList.getElementSubjectAnchor();
        XHTMLElement sujetCellTemplate = messageList.getElementSubject();
        XHTMLAnchorElement threadAnchor = messageList.getElementThreadAnchor();
        XHTMLElement threadCellTemplate = messageList.getElementThreads();
	
        // On supprime les identificateurs de ces éléments, car on va les dupliquer
        // et il ne peut y avoir 2 éléments de même id dans un document XML
        templateRow.removeAttribute("id");
        dateCellTemplate.removeAttribute("id");
        autInput.removeAttribute("id");
        auteurCellTemplate.removeAttribute("id");
        sujetCellTemplate.removeAttribute("id");
        sujAnchor.removeAttribute("id");
        threadCellTemplate.removeAttribute("id");
        threadAnchor.removeAttribute("id");
	
        // On récupère le noeud contenant la table...
        Node messageTable = templateRow.getParentNode();
	
        for(int i = 0; i < theMessageArray.length; i++) {                
            //la date
            java.sql.Date date=theMessageArray[i].getDate();
            DateFormat df = DateFormat.getDateInstance(DateFormat.LONG, Locale.US);
            messageList.setTextDate(df.format(date));
            
            // L'auteur
            String authorName = theMessageArray[i].getAuthor();
            String authorAddress = theMessageArray[i].getAuthorAddress();
            messageList.setTextAuthor(this.maskAddress(authorName));
	    // Split the address in 2 parts
	    int arobasPos=authorAddress.indexOf("@", 0);
	    if (arobasPos != -1) {
		autInput.setName(authorAddress.substring(0, arobasPos));
		autInput.setValue(authorAddress.substring(arobasPos+1));
	    }
            //autAnchor.setHref("mailto:"+authorAddress);
            
            // Le sujet
            String subject = theMessageArray[i].getSubject();
            if(null == subject || "" == subject.trim()) {
                messageList.setTextSubject("no subject");
            } else {
                messageList.setTextSubject(subject);
            }
            sujAnchor.setHref("MailingList.po?Id="+theMessageArray[i].getHandle());

            // Le nombre de réponses
            int nbth = MessageFactory.numberOfMessagesWithThread(theMessageArray[i].getThread());
            
            messageList.setTextThreads(String.valueOf(nbth));
            threadAnchor.setHref("MailingList.po?thread="+myUrlEncode(theMessageArray[i].getThread()));

            Node clone=templateRow.cloneNode(true);
            messageTable.appendChild(clone);
        }
        messageTable.removeChild(templateRow);
        
        
        return messageList;
    }
    
    public Node getMessage() throws HttpPresentationException, 
                                    IOException, 
                                    DataObjectException
                                    {
        MessageXHTML messageLayout;
        HttpPresentationRequest req = this.getComms().request;

        // Récupération du message demandé.
        Message theMessage = MessageFactory.findMessageByID(req.getParameter(ID));
        
        // Création du layout
        messageLayout = (MessageXHTML)MultilingualXHtmlTemplateFactory.createTemplate("MessageXHTML", this.getComms(), this.getSessionData());

        //la date
        java.sql.Date date = theMessage.getDate();
        DateFormat df = DateFormat.getDateInstance(DateFormat.LONG, Locale.US);
        messageLayout.setTextDate(df.format(date));
            
        // L'auteur
        XHTMLInputElement autInput = messageLayout.getElementAuthorInput();

        String authorName = theMessage.getAuthor();
        String authorAddress = theMessage.getAuthorAddress();
        messageLayout.setTextAuthor(this.maskAddress(authorName));
        // Split the address in 2 parts
        int arobasPos=authorAddress.indexOf("@", 0);
        if (arobasPos != -1) {
            autInput.setName(authorAddress.substring(0, arobasPos));
            autInput.setValue(authorAddress.substring(arobasPos+1));
        }
        //messageLayout.setTextAuthor(authorName);
        //autAnchor.setHref("mailto:"+authorAddress);
        
        String subject = theMessage.getSubject();
        messageLayout.setTextSubject(subject);
        
        String msgText = theMessage.getMsg();
        CDATASection cds = messageLayout.createCDATASection(msgText);
        messageLayout.getElementMessageBody().appendChild(cds);
        
        return messageLayout.getElementMessage();
    }
 
    /**
     * This implements the run method in MailingList.
     * I overrided the PapillonBasePO.run method because I don't want to parse the
     * message body (as many of then are ill-formed).
     *
     * @param HttpPresentationComms
     * @exception Exception
     */
    public Node getContent()
        throws HttpPresentationException, IOException, Exception 
    {
        
        Node content;
        
        HttpPresentationRequest req = this.getComms().request;
        // If the page is called with the Id parameter, present the requested message,
        // else present the message list.
        if (null != req.getParameter(ID)) { 
            //Insertion du contenu sans corps de message dans le document vide.
            return this.getMessage();
            
        } else {
            //Insertion du contenu dans le document vide.
            return this.getMessageList();
        }
    }
    
    public static String maskAddress(String address) {
	int arobasPos = address.indexOf("@", 0);
	if (arobasPos != -1) {
	    // Suppress the address, only keep the login id
	    return address.substring(0, arobasPos);
	} else {
	    return address;
	}
    }

}
