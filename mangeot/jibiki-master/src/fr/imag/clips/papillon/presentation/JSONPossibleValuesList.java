package fr.imag.clips.papillon.presentation;

import com.lutris.appserver.server.httpPresentation.HttpPresentationComms;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.AvailableValuesQuery;
import org.json.JSONArray;
import org.w3c.dom.Node;

import java.util.Collection;

/**
 *
 */
public class JSONPossibleValuesList
        extends AbstractPO {


    protected boolean loggedInUserRequired() {
        return false;
    }


    protected boolean userMayUseThisPO() {
        return true;
    }

    public Node getDocument() throws Exception {
        return null;
    }

    public void run(HttpPresentationComms comms) throws Exception {
        this.myComms = comms;
        initSessionData();

        // System.out.println(this.getComms().request.getQueryString());
        String field = this.myGetParameter("field");
        String prefix = this.myGetParameter("value");
        String lang = this.myGetParameter("lang");
        String limitString = this.myGetParameter("limit");
        int limit = 0;
        try {
            if (limitString != null & !"".equals(limitString)) {
                limit = Integer.parseInt(limitString);
            }
        } catch (NumberFormatException e) {
            PapillonLogger.writeDebugMsg("JSONPossibleValuesList: limit variable is not a number.");
        }

        Collection hws = AvailableValuesQuery.getAvailableValues(field, prefix, lang, limit);
        JSONArray ja = new JSONArray(hws);

        // System.out.println(ja.toString());
        HttpPresentationOutputStream out = this.getComms().response.getOutputStream();
        byte[] buffer = ja.toString().getBytes("UTF-8");
        out.write(buffer);
        out.flush();
    }
}
