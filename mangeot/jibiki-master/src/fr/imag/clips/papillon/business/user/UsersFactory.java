/*
-----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.5  2006/08/10 22:17:13  fbrunet
 * - Add caches to manage Dictionaries, Volumes and Xsl sheets (improve efficiency)
 * - Add export contibutions to pdf file base on exportVolume class and, Saxon8b & FOP transformations (modify papillon.properties to specify XML to FO xsl)
 * - Bug correction : +/- in advanced search
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
 * Revision 1.2.2.1  2005/03/29 09:41:33  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.2  2004/12/24 14:31:28  mangeot
 * I merged the latest developments of Papillon5.0 with this version 5.1.
 * Have to be tested more ...
 *
 * Revision 1.1.1.1  2004/12/06 16:38:32  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.user;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.CurrentDBTransaction;

import com.lutris.dods.builder.generator.dataobject.GenericDO;
import com.lutris.dods.builder.generator.query.QueryBuilder;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;

import com.lutris.appserver.server.*;
import com.lutris.appserver.server.sql.ObjectId;

import java.util.*;

import fr.imag.clips.papillon.business.utility.*;

/**
 * Used to find the instances of xslsheet.
 */
public class UsersFactory {
	
	private static Hashtable userCacheById = new Hashtable();
    private static Hashtable userCacheByLogin = new Hashtable();


   public static User findUserByName(String name)
    throws PapillonBusinessException {
        User theUser = null;

        try {
            UserQuery query = new UserQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryName(name);
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            UserDO theUserDO = query.getNextDO();
            theUser = new User(theUserDO);
        }catch(Exception ex) {
            throw new PapillonBusinessException("Exception in getVolumeByName()", ex);
        }
        return theUser;
    }

    public static User findUserByLogin(String login)
    throws PapillonBusinessException {
        User theUser = (User) userCacheByLogin.get(login);
		if (theUser==null) {
        try {
            UserQuery query = new UserQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryLogin(login);
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            UserDO theUserDO = query.getNextDO();
            theUser = new User(theUserDO);
        }catch(Exception ex) {
            throw new PapillonBusinessException("Exception in findUserByLogin()", ex);
        }
			if (!theUser.isEmpty()) {
				userCacheByLogin.put(login, theUser);
			}
		}
		return theUser;
    }

    public static User findUserByEmail(String email)
    throws PapillonBusinessException {
        User theUser = null;
		
        try {
            UserQuery query = new UserQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryEmail(email);
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            UserDO theUserDO = query.getNextDO();
            theUser = new User(theUserDO);
            return theUser;
        }catch(Exception ex) {
            ex.printStackTrace();
            throw new PapillonBusinessException("Exception in findUserByEmail()", ex);
        }
    }
	
    public static UserAnswer createUniqueUser(String name,
                                              String login,
                                              String password,
                                              String password2,
                                              String email,
											  String lang)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        User myUser = null;
        String answerMessage = "";

        if (name==null || name.equals("")) {
            answerMessage = "User name empty";
        }
        else if (login==null || login.equals("")) {
            answerMessage = "User login empty";
        }
        else if (password==null || password.equals("") &&
                 password2==null || password2.equals("")) {
            answerMessage = "User password empty";
        }
        else if (!password.equals(password2)) {
            answerMessage = "Different passwords";
        }
        else if (email==null || email.equals("")) {
            answerMessage = "User email empty";
        }
        else {
            //search for an existing user
            User Existe=UsersFactory.findUserByName(name);
            if (!Existe.isEmpty()) {
                answerMessage = "User " + name + " already in the database";
            }
            else {
                Existe=UsersFactory.findUserByLogin(login);
                if (!Existe.isEmpty()) {
                    answerMessage = "Login " + login + " already in the database";
                }
                else {//doesn't exist, create:
                    myUser=new User();
                    myUser.setName(name);
                    myUser.setLogin(login);
                    myUser.setPassword(password);
                    myUser.setEmail(email);
                    myUser.setLang(lang);
                    myUser.save();
                    answerMessage =  "User: "+ myUser.getName() + " // login: " + myUser.getLogin() + " added";
                }
            }
        }
        return new UserAnswer(myUser,answerMessage);
    }

    public static UserAnswer changeUserPassword(String login,
                                                String password,
                                                String newPassword,
                                                String newPassword2)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        User myUser = null;
        String answerMessage = "";

        if (login==null || login.equals("")) {
            answerMessage = "User login empty";
        }
        else if (password==null || password.equals("")) {
            answerMessage = "User old password empty";
        }
        else if (newPassword==null || newPassword.equals("")) {
            answerMessage = "User new password empty";
        }
        else if (!newPassword.equals(newPassword2)) {
            answerMessage = "Different new passwords";
        }
        else {
            //search for an existing user
            myUser = UsersFactory.findUserByLogin(login);
            if (myUser.isEmpty()) {
                answerMessage = "Login " + login + " not in the database";
            }
            else {
                if (myUser.HasCorrectPassword(password)) {
                    if (newPassword.equals(newPassword2)) {
                        myUser.setPassword(newPassword);
                        myUser.save();
                        answerMessage = "Password changed for user: "+ myUser.getName() + " // login: " + myUser.getLogin();
                    }
                    else {
                        answerMessage = "Wrong new password";
                    }

                }
                else {
                    answerMessage = "Wrong old password";
                }
            }
        }
        return new UserAnswer(myUser,answerMessage);
    }

    public static UserAnswer addUserInGroup(User myUser,
                                            String group,
                                            String groupPassword)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        String answerMessage = "";

        if (myUser==null || myUser.isEmpty()) {
            answerMessage = "Non existing user";
        }
        else if (group==null || group.equals("")) {
            answerMessage = "Group empty";
        }
        else if (groupPassword==null || groupPassword.equals("")) {
            answerMessage = "Group password empty";
        }
        else {
            //search for an existing special group
			String refGroupPassword = myUser.getGroupPassword(group);
			if (refGroupPassword!=null && !refGroupPassword.equals("")) {
					if (groupPassword.equals(refGroupPassword)) {
						myUser.addGroup(group);
						myUser.save();
						answerMessage = "User "+ myUser.getName() + " is in group " + group;
					}
					else {
						answerMessage = "Wrong group password";
					}
			}
			//search for an existing created group
			else {
				Group myGroup = GroupsFactory.findGroupByName(group);
				if (!myGroup.isEmpty()) {
					if (myGroup.HasCorrectPassword(groupPassword)) {
						myGroup.addUser(myUser.getLogin());
						myUser.addGroup(group);
						myGroup.save();
						myUser.save();
						answerMessage = "User "+ myUser.getName() + " is in group " + group;
					}
					else {
						answerMessage = "Wrong group password";
					}
				}
				else {
					answerMessage = "Group " + group + " does not exist!";
				}
			}
        }
        return new UserAnswer(myUser,answerMessage);
    }

    public static UserAnswer removeUserFromGroup(User myUser,
                                            String group,
                                            String groupPassword)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        String answerMessage = "";

        if (myUser==null || myUser.isEmpty()) {
            answerMessage = "Non existing user";
        }
        else if (group==null || group.equals("")) {
            answerMessage = "Group empty";
        }
        else if (groupPassword==null || groupPassword.equals("")) {
            answerMessage = "Group password empty";
        }
        else {
            //search for an existing special group
			String refGroupPassword = myUser.getGroupPassword(group);
			if (refGroupPassword!=null && !refGroupPassword.equals("")) {
					if (groupPassword.equals(refGroupPassword)) {
						myUser.removeGroup(group);
						myUser.save();
						answerMessage = "User "+ myUser.getName() + " has been removed from group " + group;
					}
					else {
						answerMessage = "Wrong group password";
					}
			}
			//search for an existing created group
			else {
				Group myGroup = GroupsFactory.findGroupByName(group);
				if (!myGroup.isEmpty()) {
					if (myGroup.HasCorrectPassword(groupPassword)) {
						myGroup.removeUser(myUser.getLogin());
						myUser.removeGroup(group);
						myGroup.save();
						myUser.save();
						answerMessage = "User "+ myUser.getName() + " has been removed from group " + group;
					}
					else {
						answerMessage = "Wrong group password";
					}
				}
				else {
					answerMessage = "Group " + group + " does not exist!";
				}
			}
        }
        return new UserAnswer(myUser,answerMessage);
    }

	public static UserAnswer deleteUser(User myUser, String password)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		String answerMessage = "";

		//search for an existing user
		if (myUser == null || myUser.isEmpty()) {
			answerMessage = "User not in the database";
		}
		else {
			if (myUser.HasCorrectPassword(password)) {
				answerMessage = "User " + myUser.getName() + " deleted";
				myUser.delete();
			}
			else {
				answerMessage = "Wrong user password";
			}
		}
		return new UserAnswer(myUser,answerMessage);
		}
   
//ajouter pour la page profil

 public static UserAnswer ChangeNameAndEmail(User myUser,
                                             String name, String email)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        String answerMessage = "";

        if (email==null || email.equals("")) {
					answerMessage = "Current email empty";
        }
        else if (name==null || name.equals("")) {
					answerMessage = "Current name empty";
        }
        else {
            //search for an existing user
            if (myUser.isEmpty()) {
                answerMessage = "User not in the database";
            }
            else {
							myUser.setName(name);
							myUser.setEmail(email);
							myUser.save();
							answerMessage = "Email changed for user: "+ myUser.getName() + " // login: " + myUser.getLogin();
						}
        }
        
        return new UserAnswer(myUser,answerMessage);
    }
     
    public static User findUserById(String id) 
        throws PapillonBusinessException {
			User theUser = (User) userCacheById.get(id);
			if (theUser==null) {
        try {
           UserQuery query = new UserQuery(CurrentDBTransaction.get());  
             //set query
            query.setQueryOId(new ObjectId(id));
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            UserDO theUserDO = query.getNextDO();
			theUser = new User(theUserDO);
        }
        catch(Exception ex) {
            throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in findUserById()", ex);
        }
				if (!theUser.isEmpty()) {
					userCacheById.put(id, theUser);
				}
			}
		return theUser;
    }
	
	public static User[] getUsersArray(String sortBy) 
		throws PapillonBusinessException {
		return getUsersArray("","","","",sortBy);
	}
    
	public static User[] getUsersArrayForGroup(String group)
		throws PapillonBusinessException {
		User[] theUsersArray = null;
        try {
            UserQuery query = new UserQuery(CurrentDBTransaction.get());
            
            if ((null != group) && (!group.trim().equals(""))) {
				group  = User.GROUPS_SEPARATOR_STRING + group;
                query.getQueryBuilder().addWhereClause("groups", group, 
                    QueryBuilder.CASE_SENSITIVE_CONTAINS);
            }
            UserDO[] DOarray = query.getDOArray();
            theUsersArray = new User[ DOarray.length ];
            for ( int i = 0; i < DOarray.length; i++ )
	    	theUsersArray[i] = new User(DOarray[i]);

        }catch(Exception ex) {
            throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in getUsersArray()", ex);
        }
         
        return theUsersArray;
	}
	
    protected static User[] getUsersArray(String name, String login, String password, String email, String sortBy) 
        throws PapillonBusinessException {
        User[] theDictArray = null;
        
        try {
            UserQuery query = new UserQuery(CurrentDBTransaction.get());
            
            if ((null != name) && (!name.trim().equals(""))) {
                query.getQueryBuilder().addWhereClause("name", name, 
                    QueryBuilder.EQUAL);
            }
            if ((null != login) && (!login.trim().equals(""))) {
                query.getQueryBuilder().addWhereClause("login", login, 
                    QueryBuilder.EQUAL);
            }
            if ((null != password) && (!password.trim().equals(""))) {
                query.getQueryBuilder().addWhereClause("password", password,
                                                       QueryBuilder.EQUAL);
            }
            if ((null != email) && (!email.trim().equals(""))) {
                query.getQueryBuilder().addWhereClause("email", email,
                                                       QueryBuilder.EQUAL);
            }
			if (sortBy!=null && !sortBy.equals("")) {
				if (sortBy.equals("name")) {
					query.addOrderByName(true);
				}
				else if (sortBy.equals("login")) {
					query.addOrderByLogin(true);
				}
                else if (sortBy.equals("email")) {
                    query.addOrderByEmail(true);
                }
                else if (sortBy.equals("creationDate")) {
                    query.addOrderByCreationDate(true);
                }
                else if (sortBy.equals("modificationDate")) {
                    query.addOrderByModificationDate(true);
                }
			}
            UserDO[] DOarray = query.getDOArray();
            theDictArray = new User[ DOarray.length ];
            for ( int i = 0; i < DOarray.length; i++ )
	    	theDictArray[i] = new User(DOarray[i]);

        }catch(Exception ex) {
            throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in getUsersArray()", ex);
        }
         
        return theDictArray;
    }
    
    public static void removeUserFromCache(User theUser) throws PapillonBusinessException {
        userCacheById.remove(theUser.getHandle());
        userCacheByLogin.remove(theUser.getLogin());
    }

	public static String generateRandomPassword () {
		java.util.Random rand = Utility.getRandomGenerator();
		int length = rand.nextInt(6) + 6;
		char[] password = new char[length];
		
		for (int x = 0; x < length; x++) {
			int randDecimalAsciiVal = rand.nextInt(93) + 33;
			password[x] = (char) randDecimalAsciiVal;
		}
		return String.valueOf(password);
	}


}
