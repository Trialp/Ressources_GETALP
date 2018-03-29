
package fr.imag.clips.papillon.presentation;

// Enhydra SuperServlet imports
import com.lutris.appserver.server.httpPresentation.HttpPresentation;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
//import org.enhydra.xml.xmlc.XMLObject;
import org.w3c.dom.html.HTMLElement;
import org.w3c.dom.Node;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
//import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

// Standard imports
import java.io.IOException;
import java.util.Date;
import java.text.DateFormat;

import fr.imag.clips.papillon.presentation.xhtml.orig.*;

public class ImportHelp extends PapillonBasePO {


    protected boolean loggedInUserRequired() {
        return false;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }
    
    protected  int getCurrentSection() {
        return NO_SECTION;
    }
    

    public Node getContent()
    throws HttpPresentationException, IOException {

        Node contentNode;

        ImportHelpContentXHTML content = (ImportHelpContentXHTML)MultilingualXHtmlTemplateFactory.createTemplate("ImportHelpContentXHTML", this.getComms(), this.getSessionData());
        contentNode = content.getElementImportHelpContent();
        
        //On rend le contenu correct
        return contentNode;
    }
    
}
