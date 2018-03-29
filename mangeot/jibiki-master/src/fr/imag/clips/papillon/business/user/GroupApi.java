/*
-----------------------------
 * $Id$
 *-----------------------------------------------
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.user;

import com.lutris.appserver.server.httpPresentation.HttpPresentationResponse;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.xml.XMLServices;


public class GroupApi {

    
    protected static final String GROUPLIST_TAG = "group-list";
    protected static final String GROUP_TAG = "group";

    protected static final String GROUPLIST_XMLSTRING_HEADER = "<?xml version='1.0' encoding='UTF-8'?><"+GROUPLIST_TAG+" xmlns='http://www-clips.imag.fr/geta/services/dml'>\n";
    protected static final String GROUPLIST_XMLSTRING_FOOTER = "</"+GROUPLIST_TAG+">";
    
    protected static final String GROUP_XMLSTRING_HEADER = "<?xml version='1.0' encoding='UTF-8'?><"+GROUP_TAG+" xmlns='http://www-clips.imag.fr/geta/services/dml'>\n";
    
    protected static final String GROUP_XMLSTRING_FOOTER = "</"+GROUP_TAG+">";

    public static java.util.Vector getGroupsList(User theUser)
    throws PapillonBusinessException {
        java.util.Vector responseVector = new java.util.Vector(3);
        Integer status = new Integer(HttpPresentationResponse.SC_OK);
        org.w3c.dom.Document content = null;
        String errorMsg = "";

        
        String answer = GROUPLIST_XMLSTRING_HEADER;
        Group[] groupsArray = GroupsFactory.getAllGroupsArray();
        
        for (int i = 0; i < groupsArray.length; i++) {
            Group  aGroup = groupsArray[i];
            
            answer += "<"+GROUP_TAG+"><name>" + aGroup.getName() + "</name></"+GROUP_TAG+">\n";
        }
        answer += GROUPLIST_XMLSTRING_FOOTER;
        content = XMLServices.buildDOMTree(answer);
        
        responseVector.addElement(content);
        responseVector.addElement(status);
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    public static java.util.Vector getGroup(String name, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";

        Group aGroup = GroupsFactory.findGroupByName(name);
        
        if (aGroup != null && !aGroup.isEmpty()) {
            String answer = GROUP_XMLSTRING_HEADER + "<name>" + aGroup.getName() + "</name>";
            if (theUser != null && (theUser.isAdmin() || aGroup.isAdmin(theUser.getLogin()))) {
                String[] usersArray = aGroup.getUsersArray();
                answer += "<members>";
                for (int i = 0; i < usersArray.length; i++) {
                    String  userLogin = usersArray[i];
                    answer += "<user-ref>" + userLogin + "</user-ref>\n";
                }
                answer += "</members>\n";
                answer += "<admins>";
                String[] adminsArray = aGroup.getAdminsArray();
                for (int i = 0; i < adminsArray.length; i++) {
                    String  adminLogin = adminsArray[i];
                    answer += "<user-ref>" + adminLogin + "</user-ref>\n";
                }
                answer += "</admins>";
            }
            answer += GROUP_XMLSTRING_FOOTER;
            content = XMLServices.buildDOMTree(answer);
        }
 /*       else if (name !=null && name.equals(Group.ADMIN_GROUP)) {
            
        }*/
        else {
            errorMsg = "Error: group: " + name + " does not exist!";
            status = HttpPresentationResponse.SC_NOT_FOUND;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    public static java.util.Vector postGroup(String groupName, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_CREATED;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        if (theUser != null && !theUser.isEmpty()) {
            Group aGroup = GroupsFactory.findGroupByName(groupName);
            if (aGroup!=null && !aGroup.isEmpty()) {
                errorMsg = "Error: conflict, group " + groupName + " already exists!";
                status = 409;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Conflict</h1><p>" + errorMsg + "</p></html>");
            }
            else {
                Group myGroup = new Group();
                myGroup.setName(groupName);
                myGroup.setPassword(theUser.getPassword());
                myGroup.addUser(theUser.getLogin());
                myGroup.addAdmin(theUser.getLogin());
                myGroup.save();
                String answerMessage =  "Group: "+ myGroup.getName() + " created";
                PapillonLogger.writeDebugMsg(answerMessage);
            }
        }
        else {
            errorMsg = "Error: anonymous user: not authorized to post user!";
            status = HttpPresentationResponse.SC_UNAUTHORIZED;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    public static java.util.Vector deleteGroup(String groupName, User theUser)
    throws PapillonBusinessException {
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        Group theGroup = GroupsFactory.findGroupByName(groupName);
        if (theGroup != null && !theGroup.isEmpty()) {
            if (theUser != null && (theUser.isAdmin() || theGroup.isAdmin(theUser.getLogin()))) {
                theGroup.delete();
            }
            else {
                String login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():"";
                errorMsg = "Error: user: " + login +" not authorized to delete group!";
                status = HttpPresentationResponse.SC_UNAUTHORIZED;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            errorMsg = "Error: group: " + groupName + " does not exist!";
            status = HttpPresentationResponse.SC_NOT_FOUND;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }


}
