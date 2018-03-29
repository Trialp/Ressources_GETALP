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

import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.dictionary.DmlPrefixResolver;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.xml.XMLServices;


public class UserApi {

    protected static String JSON_CONTENTTYPE = "text/json";
    protected static String XML_CONTENTTYPE = "text/xml";

    protected static final String USERLIST_TAG = "user-list";
    protected static final String USER_TAG = "user";
    
    protected static final String USERLIST_XMLSTRING_HEADER = "<?xml version='1.0' encoding='UTF-8'?><"+USERLIST_TAG+" "
    + "xmlns='http://www-clips.imag.fr/geta/services/dml'>";
    protected static final String USERLIST_XMLSTRING_FOOTER = "</"+USERLIST_TAG+">";
    protected static final String USER_XMLSTRING_HEADER = "<?xml version='1.0' encoding='UTF-8'?><"+USER_TAG+" xmlns='http://www-clips.imag.fr/geta/services/dml'>";
    protected static final String USER_XMLSTRING_FOOTER = "</"+USER_TAG+">";
 
    protected static final String ROLELIST_XMLSTRING_HEADER = "<?xml version='1.0' encoding='UTF-8'?><role-list xmlns='http://www-clips.imag.fr/geta/services/dml'>\n";
    protected static final String ROLELIST_XMLSTRING_FOOTER = "</role-list>";
    
    protected static final String sortByDefault = "login";

	
    public static java.util.Vector getUserList(User theUser)
        throws PapillonBusinessException {
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        
        if (theUser != null && theUser.isAdmin()) {
            content = getUsersListForAdmin();
        }
        else {
            content = getUsersListForPublic();
        }
        
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    private static org.w3c.dom.Document getUsersListForPublic()
        throws PapillonBusinessException {
        
        String resultString = USERLIST_XMLSTRING_HEADER;
        
        User[] usersArray = UsersFactory.getUsersArray(sortByDefault);
        
        for (int i = 0; i < usersArray.length; i++) {
            User  aUser = usersArray[i];
            
            resultString += "<"+USER_TAG+"><name>" + aUser.getName() + "</name><login>" + aUser.getLogin() + "</login></"+USER_TAG+">";
        }
        resultString += USERLIST_XMLSTRING_FOOTER;
        org.w3c.dom.Document resultDoc = XMLServices.buildDOMTree(resultString);
        return resultDoc;
    }
    
    private static org.w3c.dom.Document getUsersListForAdmin() throws PapillonBusinessException {

        org.w3c.dom.Document resultDoc = XMLServices.buildDOMTree(USERLIST_XMLSTRING_HEADER+USERLIST_XMLSTRING_FOOTER);
        
        User[] usersArray = UsersFactory.getUsersArray(sortByDefault);
        
        for (int i = 0; i < usersArray.length; i++) {
            User  aUser = usersArray[i];
            
            org.w3c.dom.Document aUserDom = XMLServices.buildDOMTree(aUser.getXmlCode());
            org.w3c.dom.Node aUserImported = resultDoc.importNode(aUserDom.getDocumentElement(), true);
            resultDoc.getDocumentElement().appendChild(aUserImported);
        }
        return resultDoc;
    }
    
    public static java.util.Vector getUser(String login, User theUser)
        throws PapillonBusinessException {

        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";

        if (theUser != null && (theUser.isAdmin() || theUser.getLogin().equals(login))) {
            content = getUserForAdmin(login);
        }
        else {
            content = getUserForPublic(login);
        }
        if (content == null) {
            errorMsg = "Error: user: " + login + " does not exist!";
            status = HttpPresentationResponse.SC_NOT_FOUND;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    
    private static org.w3c.dom.Document getUserForPublic(String login) throws PapillonBusinessException {

        
        User aUser = UsersFactory.findUserByLogin(login);
        org.w3c.dom.Document resultDoc = null;
        
        if (aUser != null && !aUser.isEmpty()) {
            String resultString = USER_XMLSTRING_HEADER + "<name>" + aUser.getName() + "</name><login>" + aUser.getLogin() + "</login>" + USER_XMLSTRING_FOOTER;
            resultDoc = XMLServices.buildDOMTree(resultString);
        }
        return resultDoc;
    }
    
    private static org.w3c.dom.Document getUserForAdmin(String login)
        throws PapillonBusinessException {
        org.w3c.dom.Document resultDoc = null;
        
        User aUser = UsersFactory.findUserByLogin(login);
        
        if (aUser != null && !aUser.isEmpty()) {
            resultDoc = XMLServices.buildDOMTree(aUser.getXmlCode());
        }
        return resultDoc;
    }
    
    public static java.util.Vector postUser(String authlogin, String authpassword, String login, String userString, String contentType, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_CREATED;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
                
        if (authlogin != null && authlogin != "" && authpassword != null && authpassword != "") {
            User aUser = UsersFactory.findUserByLogin(authlogin);
            if (aUser != null && !aUser.isEmpty()) {
                errorMsg = "Error: conflict, user " + authlogin + " already exists!";
                status = 409;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Conflict</h1><p>" + errorMsg + "</p></html>");
            }
            else {
                String userName = "";
                String email = "";
                String lang = "";
                if (contentType.equals(JSON_CONTENTTYPE)) {
                    org.json.JSONObject userObject = null;
                    try {
                        userObject = new org.json.JSONObject(userString).getJSONObject("user");
                    }
                    catch (org.json.JSONException jsone) {
                        errorMsg = "Error: user data: <![CDATA["+ userString +"]]> JSON is malformed! " + jsone.getMessage();
                        status = HttpPresentationResponse.SC_BAD_REQUEST;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                    }
                    try {
                        email = userObject.getString("email");
                    }
                    catch (Exception e) {
                    }
                    try {
                        userName = userObject.getString("name");
                    }
                    catch (Exception e) {
                    }
                    try {
                        lang = userObject.getString("lang");
                    }
                    catch (Exception e) {
                    }
               }
                else {
                    org.w3c.dom.Document userDom = null;
                    try {
                        userDom = XMLServices.buildDOMTree(userString);
                    }
                    catch (Exception se) {
                        errorMsg = "Error: user data: <![CDATA["+ userString +"]]> XML is malformed! " + se.getMessage();
                        status = HttpPresentationResponse.SC_BAD_REQUEST;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");

                    }
                    if (content == null) {
                        try {
                        org.w3c.dom.NodeList nodeList = userDom.getElementsByTagNameNS(DmlPrefixResolver.DML_URI,"user");
                        org.w3c.dom.Element userElement = (org.w3c.dom.Element) nodeList.item(0);
                        lang = userElement.getAttributes().getNamedItem("lang").getNodeValue();
                        userName = userElement.getAttributes().getNamedItem("name").getNodeValue();
                        login = ((org.w3c.dom.Element) userElement.getElementsByTagName("login").item(0)).getTextContent();
                        email = ((org.w3c.dom.Element) userElement.getElementsByTagName("email").item(0)).getTextContent();
                        }
                        catch (Exception e) {
                            errorMsg = "Error: user data: " + userString + " is not semantically correct! " + e.getMessage();;
                            status = 422;
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                        }
                    }
                }
                if (content == null) {
                    User myUser=new User();
                    myUser.setName(userName);
                    myUser.setLogin(authlogin);
                    myUser.setPassword(authpassword);
                    myUser.setEmail(email);
                    myUser.setLang(lang);
                    myUser.save();
                    content = XMLServices.buildDOMTree(myUser.getXmlCode());
                    String answerMessage =  "User: "+ myUser.getName() + " // login: " + myUser.getLogin() + " added";
                    PapillonLogger.writeDebugMsg(answerMessage);
                }
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
    
    public static java.util.Vector putUser(String authlogin, String authpassword, String login, String userString, String contentType, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        if (theUser == null || theUser.isEmpty() || (!theUser.getLogin().equals(login) && !theUser.isAdmin())) {
            login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():login;
            errorMsg = "Error: user: " + login +" not authorized to post dict!";
            status = HttpPresentationResponse.SC_UNAUTHORIZED;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            User aUser = UsersFactory.findUserByLogin(login);
            if (aUser == null || aUser.isEmpty()) {
                errorMsg = "Error: user: " + login + " does not exist!";
                status = HttpPresentationResponse.SC_NOT_FOUND;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
            else {
                String userName = "";
                String email = "";
                String lang = "";
                if (contentType.equals(JSON_CONTENTTYPE)) {
                    try {
                        org.json.JSONObject userObject = new org.json.JSONObject(userString).getJSONObject("user");
                        email = userObject.getString("email");
                        userName = userObject.getString("name");
                        lang = userObject.getString("lang");
                    }
                    catch (org.json.JSONException jsone) {
                        errorMsg = "Error: user data: <![CDATA["+ userString +"]]> JSON is malformed! " + jsone.getMessage();
                        status = HttpPresentationResponse.SC_BAD_REQUEST;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                    }
                }
                else {
                    org.w3c.dom.Document userDom = null;
                    try {
                        userDom = XMLServices.buildDOMTree(userString);
                    }
                    catch (Exception se) {
                        errorMsg = "Error: user data: <![CDATA["+ userString +"]]> XML is malformed! " + se.getMessage();
                        status = HttpPresentationResponse.SC_BAD_REQUEST;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                        
                    }
                    if (content == null) {
                        try {
                            org.w3c.dom.NodeList nodeList = userDom.getElementsByTagNameNS(DmlPrefixResolver.DML_URI,"user");
                            org.w3c.dom.Element userElement = (org.w3c.dom.Element) nodeList.item(0);
                            lang = userElement.getAttributes().getNamedItem("lang").getNodeValue();
                            userName = userElement.getAttributes().getNamedItem("name").getNodeValue();
                            login = ((org.w3c.dom.Element) userElement.getElementsByTagName("login").item(0)).getTextContent();
                            email = ((org.w3c.dom.Element) userElement.getElementsByTagName("email").item(0)).getTextContent();
                        }
                        catch (Exception e) {
                            errorMsg = "Error: user data: " + userString + " is not semantically correct! " + e.getMessage();;
                            status = 422;
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                        }
                    }
                }
                if (userName!=null && !userName.equals("")) {
                    aUser.setName(userName);
                }
                if (email!=null && !email.equals("")) {
                    aUser.setEmail(email);
                }
                if (lang!=null && !lang.equals("")) {
                    aUser.setEmail(email);
                }
                if (authpassword!=null) {
                    aUser.setPassword(authpassword);
                }
                aUser.save();
                String answerMessage =  "User: "+ aUser.getName() + " // login: " + aUser.getLogin() + " modified";
                PapillonLogger.writeDebugMsg(answerMessage);
            }
        }
        else {
            login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():login;
            errorMsg = "Error: user: " + login +" not authorized to put/modify user!";
            status = HttpPresentationResponse.SC_UNAUTHORIZED;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    

    public static java.util.Vector deleteUser(String login, User theUser)
    throws PapillonBusinessException {
        
        //TODO supprimer l'utilisateur des groupes auxquels il appartient
        // pb : pour les groupes admin dict où il est le seul utilisateur ?
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_NO_CONTENT;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        if (theUser != null && (theUser.isAdmin() || theUser.getLogin().equals(login))) {
            User aUser = UsersFactory.findUserByLogin(login);
            if (aUser != null && !aUser.isEmpty()) {
                String[] Groups = aUser.getGroupsArray();
                for(int i = 0; i < Groups.length; i++)
                {
                    Group theGroup = GroupsFactory.findGroupByName(Groups[i]);
                    if (theGroup!= null && !theGroup.isEmpty()) {
                        theGroup.removeUser(login);
                    }
                }
                aUser.delete();
                content = XMLServices.buildDOMTree(aUser.getXmlCode());
            }
            else {
                errorMsg = "Error: user: " + login + " does not exist!";
                status = HttpPresentationResponse.SC_NOT_FOUND;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():login;
            errorMsg = "Error: user: " + login +" not authorized to delete user!";
            status = HttpPresentationResponse.SC_UNAUTHORIZED;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }

    public static java.util.Vector getGroupsForUser(String login, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        
        if (theUser != null && (theUser.isAdmin() || theUser.getLogin().equals(login))) {
            User aUser = UsersFactory.findUserByLogin(login);
            
            if (aUser != null && !aUser.isEmpty()) {
                String[] groupsArray = aUser.getGroupsArray();
                String resultString = GroupApi.GROUPLIST_XMLSTRING_HEADER;
                for (int i = 0; i < groupsArray.length; i++) {
                    String groupName = groupsArray[i];
                    resultString += "<"+GroupApi.GROUP_TAG+"><name>" + groupName + "</name>";
                    String dictionary = "";
                    if (groupName.startsWith(Group.ADMIN_DICT_GROUP_PREFIX)) {
                        resultString += "<role>" + Group.ADMIN_GROUP + "</role>";
                        resultString += "<dictionary>" + groupName.substring(Group.ADMIN_DICT_GROUP_PREFIX.length()) + "</dictionary>";
                    }
                    else if (groupName.startsWith(Group.VALIDATOR_DICT_GROUP_PREFIX)) {
                        resultString += "<role>" + Group.VALIDATOR_GROUP + "</role>";
                        resultString += "<dictionary>" + groupName.substring(Group.VALIDATOR_DICT_GROUP_PREFIX.length()) + "</dictionary>";
                    }
                    else if (groupName.startsWith(Group.SPECIALIST_DICT_GROUP_PREFIX)) {
                        resultString += "<role>" + Group.SPECIALIST_GROUP + "</role>";
                        resultString += "<dictionary>" + groupName.substring(Group.SPECIALIST_DICT_GROUP_PREFIX.length()) + "</dictionary>";
                    }
                    else if (groupName.startsWith(Group.READER_DICT_GROUP_PREFIX)) {
                        resultString += "<role>" + Group.READER_GROUP + "</role>";
                        resultString += "<dictionary>" + groupName.substring(Group.READER_DICT_GROUP_PREFIX.length()) + "</dictionary>";
                    }
                    resultString += "</"+GroupApi.GROUP_TAG+">";
                }
                resultString += GroupApi.GROUPLIST_XMLSTRING_FOOTER;
                content = XMLServices.buildDOMTree(resultString);
            }
            else {
                errorMsg = "Error: user: " + login + " does not exist!";
                status = HttpPresentationResponse.SC_NOT_FOUND;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            String uname = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():"";
            errorMsg = "Error: user "+ uname + " not authorized to see user groups!";
            status = HttpPresentationResponse.SC_UNAUTHORIZED;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    public static java.util.Vector getRolesForDictionary(String dictName, String role, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        Dictionary theDict = DictionariesFactory.getDictionaryByName(dictName);
        if (theDict !=null && !theDict.isEmpty()) {
            if (theUser != null && (theUser.isAdmin() || theDict.isAdmin(theUser.getLogin()))) {
                String resultString = UserApi.ROLELIST_XMLSTRING_HEADER;
                
                if (role == null || role.equals("") || role.equals(Group.ADMIN_GROUP)) {
                resultString += "<role><name>" + Group.ADMIN_GROUP + "</name>";
                resultString += "<group-name>" + Group.ADMIN_DICT_GROUP_PREFIX + dictName + "</group-name>";
                resultString += "<members>";
                String[] usersArray = theDict.getAdminArray();
                for (int i = 0; i < usersArray.length; i++) {
                    resultString += "<user-ref>" + usersArray[i] + "</user-ref>\n";
                }
                resultString += "</members></role>";
                }
                
                if (role == null || role.equals("") || role.equals(Group.VALIDATOR_GROUP)) {
                resultString += "<role><name>" + Group.VALIDATOR_GROUP + "</name>";
                resultString += "<group-name>" + Group.VALIDATOR_DICT_GROUP_PREFIX + dictName + "</group-name>";
                resultString += "<members>";
                String[] usersArray = theDict.getValidatorArray();
                for (int i = 0; i < usersArray.length; i++) {
                    resultString += "<user-ref>" + usersArray[i] + "</user-ref>\n";
                }
                resultString += "</members></role>";
                }
                
                    if (role == null || role.equals("") || role.equals(Group.SPECIALIST_GROUP)) {
                resultString += "<role><name>" + Group.SPECIALIST_GROUP+ "</name>";
                resultString += "<group-name>" + Group.SPECIALIST_DICT_GROUP_PREFIX + dictName + "</group-name>";
                resultString += "<members>";
                String[] usersArray = theDict.getSpecialistArray();
                for (int i = 0; i < usersArray.length; i++) {
                    resultString += "<user-ref>" + usersArray[i] + "</user-ref>\n";
                }
                resultString += "</members></role>";
                    }
                
                if (role == null || role.equals("") || role.equals(Group.READER_GROUP)) {
                resultString += "<role><name>" + Group.READER_GROUP + "</name>";
                resultString += "<group-name>" + Group.READER_DICT_GROUP_PREFIX + dictName + "</group-name>";
                resultString += "<members>";
                String[] usersArray = theDict.getReaderArray();
                for (int i = 0; i < usersArray.length; i++) {
                    resultString += "<user-ref>" + usersArray[i] + "</user-ref>\n";
                }
                resultString += "</members></role>";
                }
                
                resultString += UserApi.ROLELIST_XMLSTRING_FOOTER;
                content = XMLServices.buildDOMTree(resultString);
            }
            else {
                String uname = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():"";
                errorMsg = "Error: user "+ uname + " not authorized to see dictionary groups!";
                status = HttpPresentationResponse.SC_UNAUTHORIZED;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
        }
        else {
            errorMsg = "Error: dictionary: " + dictName + " does not exist!";
            status = HttpPresentationResponse.SC_NOT_FOUND;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }

    
    public static java.util.Vector putUserInGroup(String login, String groupName, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        if (theUser == null || theUser.isEmpty()) {
            errorMsg = "Error: user: not authorized to add user " + login +" in group " + groupName +"!";
            status = HttpPresentationResponse.SC_UNAUTHORIZED;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        else {
            User aUser = UsersFactory.findUserByLogin(login);
            if (aUser == null || aUser.isEmpty()) {
                errorMsg = "Error: user: " + login + " does not exist!";
                status = HttpPresentationResponse.SC_NOT_FOUND;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
            else {
                //search for an existing special group
                String refGroupPassword = aUser.getGroupPassword(groupName);
                if (refGroupPassword!=null) {
                    if (theUser.isAdmin()) {
                        if (aUser.isInGroup(groupName)) {
                            errorMsg = "Error: conflict, user " + login + " already in group "+groupName+"!";
                            status = 409;
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Conflict</h1><p>" + errorMsg + "</p></html>");
                        }
                        else {
                            aUser.addGroup(groupName);
                            aUser.save();
                            String answerMessage =  "User: " + login + " added in special group " + groupName + "; ";
                            PapillonLogger.writeDebugMsg(answerMessage);
                        }
                    }
                    else {
                        errorMsg = "Error: user: not authorized to add user " + login +" in special group " + groupName +"!";
                        status = HttpPresentationResponse.SC_UNAUTHORIZED;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                    }
                }
                else {
                    Group myGroup = GroupsFactory.findGroupByName(groupName);
                    if (myGroup != null && !myGroup.isEmpty()) {
                        if (theUser.isAdmin() || myGroup.isAdmin(theUser.getLogin())) {
                            if (aUser.isInGroup(groupName)) {
                                errorMsg = "Error: conflict, user " + login + " already in group "+groupName+"!";
                                status = 409;
                                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Conflict</h1><p>" + errorMsg + "</p></html>");
                            }
                            else {
                                myGroup.addUser(aUser.getLogin());
                                aUser.addGroup(groupName);
                                myGroup.save();
                                aUser.save();
                                String answerMessage =  "User: " + login + " added in group " + groupName + "; ";
                                PapillonLogger.writeDebugMsg(answerMessage);
                            }
                        }
                        else {
                            errorMsg = "Error: user: not authorized to add user " + login +" in special group " + groupName +"!";
                            status = HttpPresentationResponse.SC_UNAUTHORIZED;
                            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                        }
                    }
                    else {
                        errorMsg = "Error: group: " + groupName + " does not exist!";
                        status = HttpPresentationResponse.SC_NOT_FOUND;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                    }
                }
            }
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
    
    
    /*
     
     put UsersInGroup
     
     207 Multi-Status
     
     <?xml version="1.0" encoding="utf-8" ?>
     <D:multistatus xmlns:D='DAV:'>
     <D:response>
     <D:user>user-123</D:user>
     <D:status>success</D:status>
     </D:response>
     <D:response>
     <D:user>user-789</D:user>
     <D:status>failure</D:status>
     </D:response>
     </D:multistatus>
     
     */
    
    public static java.util.Vector putUsersInGroup(String usersString, String groupName, String contentType, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_OK;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        if (theUser == null || theUser.isEmpty()) {
            errorMsg = "Error: user: not authorized to add users in group " + groupName +"!";
            status = HttpPresentationResponse.SC_UNAUTHORIZED;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        else {
            String login = "";
            String responseString = "<?xml version='1.0' encoding='utf-8' ?><multistatus xmlns='DAV::'>";
            if (contentType.equals(JSON_CONTENTTYPE)) {
                try {
                    org.json.JSONObject usersObject = new org.json.JSONObject(usersString).getJSONObject(USERLIST_TAG);
                    org.json.JSONArray usersArray = (org.json.JSONArray) usersObject.get(USER_TAG);
                    for (int i = 0 ; i < usersArray.length(); i++) {
                        responseString += "<response>";
                        org.json.JSONObject obj = usersArray.getJSONObject(i);
                        login  = obj.getString("login");
                        responseString += "<"+USER_TAG+">" + login + "</"+USER_TAG+">";
                        java.util.Vector partialResponseVector = putUserInGroup(login, groupName, theUser);
                        int partialStatus = ((Integer)partialResponseVector.elementAt(1)).intValue();
                        if (partialStatus != HttpPresentationResponse.SC_OK) {
                            status = 207;
                            responseString += "<status>failure</status>";
                        }
                        else {
                            responseString += "<status>success</status>";
                        }
                        responseString += "</response>";
                    }
                    responseString += "</multistatus>";
                    content = XMLServices.buildDOMTree(responseString);
                }
                catch (org.json.JSONException jsone) {
                    errorMsg = "Error: user data: <![CDATA["+ usersString +"]]> JSON is malformed! " + jsone.getMessage();
                    status = HttpPresentationResponse.SC_BAD_REQUEST;
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                }
            }
            else {
                org.w3c.dom.Document userDom = null;
                try {
                    userDom = XMLServices.buildDOMTree(usersString);
                }
                catch (Exception se) {
                    errorMsg = "Error: user data: <![CDATA["+ usersString +"]]> XML is malformed! " + se.getMessage();
                    status = HttpPresentationResponse.SC_BAD_REQUEST;
                    content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                    
                }
                if (content == null) {
                    try {
                        org.w3c.dom.NodeList userList = userDom.getElementsByTagNameNS(DmlPrefixResolver.DML_URI,USER_TAG);
                        for (int i = 0; i < userList.getLength(); i++) {
                            responseString += "<response>";
                            org.w3c.dom.Element userElement = (org.w3c.dom.Element) userList.item(i);
                            login = ((org.w3c.dom.Element) userElement.getElementsByTagName("login").item(0)).getTextContent();
                            responseString += "<"+USER_TAG+">" + login + "</"+USER_TAG+">";
                            java.util.Vector partialResponseVector = putUserInGroup(login, groupName, theUser);
                            int partialStatus = ((Integer)responseVector.elementAt(1)).intValue();
                            if (partialStatus != HttpPresentationResponse.SC_OK) {
                                status = 207;
                                responseString += "<status>failure</status>";
                            }
                            else {
                                responseString += "<status>success</status>";
                            }

                            responseString += "</response>";
                        }
                        responseString += "</multistatus>";
                        content = XMLServices.buildDOMTree(responseString);
                    }
                    catch (Exception e) {
                        errorMsg = "Error: user data: " + usersString + " is not semantically correct! " + e.getMessage();;
                        status = 422;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + " Unprocessable entity</h1><p>" + errorMsg + "</p></html>");
                    }
                }
            }
            
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }
   

    public static java.util.Vector removeUserFromGroup(String login, String groupName, User theUser)
    throws PapillonBusinessException {
        
        java.util.Vector responseVector = new java.util.Vector(3);
        int status = HttpPresentationResponse.SC_NO_CONTENT;
        org.w3c.dom.Document content = null;
        String errorMsg = "";
        
        if (theUser != null && !theUser.isEmpty() && (theUser.getLogin().equals(login) || theUser.isAdmin())) {
            User aUser = UsersFactory.findUserByLogin(login);
            if (aUser == null || aUser.isEmpty()) {
                errorMsg = "Error: user: " + login + " does not exist!";
                status = HttpPresentationResponse.SC_NOT_FOUND;
                content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
            }
            else {
                //search for an existing special group
                String refGroupPassword = aUser.getGroupPassword(groupName);
                if (refGroupPassword!=null) {
                        aUser.removeGroup(groupName);
                        aUser.save();
                }
                //search for an existing created group
                else {
                    Group myGroup = GroupsFactory.findGroupByName(groupName);
                    if (myGroup != null && !myGroup.isEmpty()) {
                        myGroup.removeUser(aUser.getLogin());
                        aUser.removeGroup(groupName);
                        myGroup.save();
                        aUser.save();
                   }
                    else {
                        errorMsg = "Error: group: " + groupName + " does not exist!";
                        status = HttpPresentationResponse.SC_NOT_FOUND;
                        content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
                    }
                }
                String answerMessage =  "User: " + login + " removed from group " + groupName;
                PapillonLogger.writeDebugMsg(answerMessage);
            }
        }
        else {
            login = (theUser!=null && !theUser.isEmpty())?theUser.getLogin():login;
            errorMsg = "Error: user: not authorized to remove user " + login +" from group " + groupName +"!";
            status = HttpPresentationResponse.SC_UNAUTHORIZED;
            content = XMLServices.buildDOMTree("<?xml version='1.0'?><html><h1>Error : " + status + "</h1><p>" + errorMsg + "</p></html>");
        }
        responseVector.addElement(content);
        responseVector.addElement(new Integer(status));
        responseVector.addElement(errorMsg);
        return responseVector;
    }

    
    
                

}
