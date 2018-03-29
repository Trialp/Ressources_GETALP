package fr.imag.clips.papillon.presentation;

import com.lutris.appserver.server.httpPresentation.HttpPresentation;
import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
import org.w3c.dom.Node;
import org.json.JSONArray;

import java.util.Collection;

import fr.imag.clips.papillon.business.dictionary.HeadwordListQuery;

/**
 *
 */
public class JSONHeadwordList
        extends AbstractPO {


    protected boolean loggedInUserRequired() {
        return false;
    }


    protected boolean userMayUseThisPO() {
        return true;
    }

    public Node getDocument() throws Exception {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public void run(HttpPresentationComms comms) throws Exception {
        this.myComms = comms;
        initSessionData();

        // Check if the user needs to be logged in for this request.
        if (this.loggedInUserRequired()) {
            checkForUserLogin();                  // This will redirect the user to the login page if necessary
        }

        // After this point, user is logged in if required...

        if (!this.userMayUseThisPO()) {
            userIsNotAuthorized();                // This will redirect the user to the login page if necessary
        }

        // System.out.println(this.getComms().request.getQueryString());
        String hw = this.myGetParameter("headword");
        String lang = this.myGetParameter("lang");

        Collection hws = HeadwordListQuery.getHeadwordListForLanguage(lang, hw, 10);
        JSONArray ja = new JSONArray(hws);

        // System.out.println(ja.toString());
        HttpPresentationOutputStream out = this.getComms().response.getOutputStream();
        byte[] buffer = ja.toString().getBytes("UTF-8");
        out.write(buffer);
        out.flush();
    }
}
