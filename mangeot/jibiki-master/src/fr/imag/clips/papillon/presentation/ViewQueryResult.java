/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Francis Brunet-Manquat - GETA CLIPS IMAG
 * Projet Jibiki
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.12.2.1  2007/07/23 14:23:50  serasset
 * Commiting most changes done for the XALAN27_NEWDISPLAY on the branch
 *  - Added XSL extensions callable during xsl transformations
 *  - Implemented new display of query results as requested by EURAC team
 *  - Modified edition interface generator to adapt it to xalan 2.7.0
 *  - Added autocompletion feature to simple search fields
 *  - Moved some old pages to "deprecated" folder (this will forbid direct use of this code for papillon/GDEF)
 *
 * Revision 1.12  2007/04/09 15:27:19  serasset
 * Modified xhtml files and applications layout, because the "Targets" id was duplicated
 * between QueryMenu and AdvancedSearch.
 * Corrected problems with the Lexalp italian home page and advanced query form.
 * Added a link to axie when searching for a lexie and user is logged in.
 *
 * Revision 1.11  2007/03/25 22:00:57  fbrunet
 * improved avancedqueryform javascript
 * bug correction: in ViewQueryResult class, encode url criteria in UTF-8
 *
 * Revision 1.10  2007/01/05 12:57:49  fbrunet
 * Add undo draft method (bug in EditEntry.java : undo after last finish contribution)
 * Modify transformation method
 *
 * Revision 1.9  2006/12/13 09:32:00  fbrunet
 * *** empty log message ***
 *
 * Revision 1.8  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
 *
 * Revision 1.7  2006/06/19 15:27:01  fbrunet
 * Jibiki : improvement of the search result display
 * Lexalp : add help menu (link to wiki and bug tracker)
 *
 * Revision 1.6  2006/06/06 09:15:10  fbrunet
 * Bug correction : view action in advanced search page if user is registered.
 *
 * Revision 1.5  2006/06/01 22:05:05  fbrunet
 * New interface, quick search, new contribution management (the first edition not create new contribution. New contribution is created after add, remove element, update, save, etc. in the interface window)
 *
 * Revision 1.4  2006/05/22 22:45:54  fbrunet
 * LexALP: add merge method in post-save processing (merge axies with same referenced lexies)
 *
 * Revision 1.3  2006/05/05 02:08:23  fbrunet
 * bug correction : url utf8 transfert (in createEntryInit)
 *
 * Revision 1.2  2006/04/24 13:50:37  fbrunet
 * *** empty log message ***
 *
 * Revision 1.1  2006/04/24 13:43:29  fbrunet
 * Add new class ViewQueryResult : allow to use one class to create result display in advancedSearch and EditEntryInit (like advancedQueryForm)
 * Improve result display : view n results per page
 */

package fr.imag.clips.papillon.presentation;

//General java imports

import java.util.Collection;
import java.util.Iterator;

// DOM imports
import org.w3c.dom.Node;
import org.enhydra.xml.xhtml.dom.*;

// Enhydra SuperServlet imports
import fr.imag.clips.papillon.presentation.xhtml.orig.ViewQueryResultXHTML;
import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;

//local imports
import fr.imag.clips.papillon.business.dictionary.ParseVolume;
import fr.imag.clips.papillon.business.dictionary.QueryResult;
import fr.imag.clips.papillon.business.dictionary.QueryParameter;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.utility.Utility;
import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.transformation.ResultFormatterFactory;
import fr.imag.clips.papillon.business.xml.XMLServices;
import fr.imag.clips.papillon.CurrentRequestContext;


public class ViewQueryResult {

    protected final static String EditEntryInitURL = "LexalpEditEntryInit.po";
    protected final static String HistoryURL = "History.po";
    // protected final static String Action = "action";

    protected static Node createNodeResult(HttpPresentationComms comms, PapillonSessionData sessionData, String url,
                                           User user, Collection qrset, QueryParameter qp, boolean viewActions)
            throws HttpPresentationException, java.io.UnsupportedEncodingException, java.io.IOException,
                   fr.imag.clips.papillon.business.PapillonBusinessException {

        //
        ViewQueryResultXHTML content = (ViewQueryResultXHTML) MultilingualXHtmlTemplateFactory.createTemplate(
                "ViewQueryResultXHTML", comms, sessionData);

        // Entry
        XHTMLElement contributionNode = content.getElementContributionNode();

        //
        XHTMLAnchorElement previousAnchorTop = content.getElementPreviousAnchorTop();
        XHTMLAnchorElement nextAnchorTop = content.getElementNextAnchorTop();
        XHTMLAnchorElement previousAnchorBottom = content.getElementPreviousAnchorBottom();
        XHTMLAnchorElement nextAnchorBottom = content.getElementNextAnchorBottom();

        // remove ID
        contributionNode.removeAttribute("id");
        previousAnchorTop.removeAttribute("id");
        nextAnchorTop.removeAttribute("id");
        previousAnchorBottom.removeAttribute("id");
        nextAnchorBottom.removeAttribute("id");

        // Set result number
        content.setTextNbResults(Integer.toString(qrset.size()));

        // post the result size in the current context
        CurrentRequestContext.get().set("result_size", new Integer(qrset.size()));

        // Create display
        Node parentNode = contributionNode.getParentNode();
        for (Iterator iter = qrset.iterator(); iter.hasNext();) {

            // FIXME: create a original template AND tempory template
            // -> don't set attribute class here (setAttribute("class","action")), only if jibiki hidden anchor (setAttribute("class","hidden"))
            // use NAME (like AdminXslTmplXHTML.NAME_defaultxsl) and duplicate node !

            //
            QueryResult qr = (QueryResult) iter.next();

            //System.out.println("ViewQueryResult : size: " + qrset.size() + " xsl: " + qp.getXsl());
            ResultFormatter rf = ResultFormatterFactory.getFormatter(qr, qp.getXsl(),
                    ResultFormatterFactory.XHTML_DIALECT, null);

            Node entryDOM = (Node) rf.getFormattedResult(qr, user);
            XHTMLElement cloneContribution = (XHTMLElement) contributionNode.cloneNode(true);
            cloneContribution.appendChild(content.importNode(entryDOM, true));

            parentNode.appendChild(cloneContribution);
            //entryNode.getParentNode().insertBefore(entryNode.cloneNode(true), entryNode);
        }

        // the previous button
        if (qp.getOffset() != 0) {

            // ATTENTION NO DEEP COPY in QueryParameter duplicate method
            QueryParameter previousQp = (QueryParameter) qp.duplicate();
            previousQp.setOffset(qp.getOffset() - qp.getLimit());
            // Top
            previousAnchorTop.setHref(
                    url + "?" + EditEntryInitFactory.ACTION_PARAMETER + "=lookup&" + AdvancedQueryForm.getEncodedUrlForParameter(
                            previousQp));
            previousAnchorTop.setAttribute("class", "pagination");
            // Bottom
            previousAnchorBottom.setHref(
                    url + "?" + EditEntryInitFactory.ACTION_PARAMETER + "=lookup&" + AdvancedQueryForm.getEncodedUrlForParameter(
                            previousQp));
            previousAnchorBottom.setAttribute("class", "pagination");

        } else {

            // Top
            previousAnchorTop.setHref("");
            previousAnchorTop.setAttribute("class", "hidden");
            // Bottom
            previousAnchorBottom.setHref("");
            previousAnchorBottom.setAttribute("class", "hidden");
        }

        // the next button
        if (qrset.size() >= qp.getLimit()) {

            QueryParameter nextQp = (QueryParameter) qp.duplicate();
            nextQp.setOffset(qp.getOffset() + qp.getLimit());
            // Top
            nextAnchorTop.setHref(
                    url + "?" + EditEntryInitFactory.ACTION_PARAMETER + "=lookup&" + AdvancedQueryForm.getEncodedUrlForParameter(
                            nextQp));
            nextAnchorTop.setAttribute("class", "pagination");
            // Bottom
            nextAnchorBottom.setHref(
                    url + "?" + EditEntryInitFactory.ACTION_PARAMETER + "=lookup&" + AdvancedQueryForm.getEncodedUrlForParameter(
                            nextQp));
            nextAnchorBottom.setAttribute("class", "pagination");

        } else {

            // Top
            nextAnchorTop.setHref("");
            nextAnchorTop.setAttribute("class", "hidden");
            // Bottom
            nextAnchorBottom.setHref("");
            nextAnchorBottom.setAttribute("class", "hidden");
        }

        //
        parentNode.removeChild(contributionNode);
        //removeEntryListTemplate(content);

        //
        XHTMLElement queryResultForm = content.getElementQueryResultForm();
        queryResultForm.removeAttribute("id");
        return queryResultForm;
    }
/*
private static void removeEntryListTemplate(ViewQueryResultXHTML content) {
    Element myElement = content.getElementEntryListRow();
    Node myParent = myElement.getParentNode();
    if (myParent != null)
        myParent.removeChild(myElement);
}
*/
}
