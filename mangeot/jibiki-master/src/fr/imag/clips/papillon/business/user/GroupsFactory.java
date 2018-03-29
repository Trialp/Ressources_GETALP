/*
-----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.5  2005/08/17 14:27:42  mangeot
 * Added groups lists in user profile and from now on, displays the groups as a listinstead of a text box
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
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.user;

import fr.imag.clips.papillon.data.*;
import fr.imag.clips.papillon.CurrentDBTransaction;

//for URLs
import java.net.*;

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
public class GroupsFactory {

    public static GroupAnswer createUniqueGroup(String name,
                                              String password,
                                              String password2,
											  String admin)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        Group myGroup = null;
        String answerMessage = "";

        if (name==null || name.equals("")) {
            answerMessage = "Group name empty";
        }
        else if (password==null || password.equals("") &&
                 password2==null || password2.equals("")) {
            answerMessage = "Group password empty";
        }
        else if (!password.equals(password2)) {
            answerMessage = "Different passwords";
        }
        else {
            //search for an existing user
            Group Existe=GroupsFactory.findGroupByName(name);
            if (!Existe.isEmpty()) {
                answerMessage = "Group " + name + " already in the database";
            }
			else {//doesn't exist, create:
				myGroup=new Group();
				myGroup.setName(name);
				myGroup.setPassword(password);
				myGroup.setUsers("");
				myGroup.setAdmins(admin);
				myGroup.save();
				answerMessage =  "Group: "+ myGroup.getName() + " added";
			}
        }
        return new GroupAnswer(myGroup,answerMessage);
    }
   
   public static Group findGroupByName(String name)
    throws PapillonBusinessException {
        Group theGroup = null;

        try {
            GroupQuery query = new GroupQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryName(name);
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            GroupDO theGroupDO = query.getNextDO();
            theGroup = new Group(theGroupDO);
        }catch(Exception ex) {
            throw new PapillonBusinessException("Exception in findGroupByName()", ex);
        }
        return theGroup;
    }

    public static Group findGroupByHandle(String handle)
    throws PapillonBusinessException {
        Group theGroup = null;

        try {
            GroupQuery query = new GroupQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryOId(new ObjectId(handle));
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            GroupDO theGroupDO = query.getNextDO();
            theGroup = new Group(theGroupDO);
            return theGroup;
        }catch(Exception ex) {
            ex.printStackTrace();
            throw new PapillonBusinessException("Exception in findGroupByHandle()", ex);
        }
    }

    public static GroupAnswer changeGroupPassword(String name,
                                                String password,
                                                String newPassword,
                                                String newPassword2)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        Group myGroup = null;
        String answerMessage = "";

        if (name==null || name.equals("")) {
            answerMessage = "Group name empty";
        }
        else if (password==null || password.equals("")) {
            answerMessage = "Group old password empty";
        }
        else if (newPassword==null || newPassword.equals("")) {
            answerMessage = "Group new password empty";
        }
        else if (!newPassword.equals(newPassword2)) {
            answerMessage = "Different new passwords";
        }
        else {
            //search for an existing group
            myGroup = GroupsFactory.findGroupByName(name);
            if (myGroup.isEmpty()) {
                answerMessage = "Group " + name + " not in the database";
            }
            else {
                if (myGroup.HasCorrectPassword(password)) {
                    if (newPassword.equals(newPassword2)) {
                        myGroup.setPassword(newPassword);
                        myGroup.save();
                        answerMessage = "Password changed for group: "+ myGroup.getName();
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
        return new GroupAnswer(myGroup,answerMessage);
    }

	public static GroupAnswer deleteGroup(Group myGroup, String password)
	throws fr.imag.clips.papillon.business.PapillonBusinessException {
		String answerMessage = "";

		//search for an existing group
		if (myGroup.isEmpty()) {
			answerMessage = "Group not in the database";
		}
		else {
			if (myGroup.HasCorrectPassword(password)) {
				answerMessage = "Group " + myGroup.getName() + " deleted";
				myGroup.delete();
			}
			else {
				answerMessage = "Wrong group password";
			}
		}
		return new GroupAnswer(myGroup,answerMessage);
		}

    public static GroupAnswer deleteGroup	(String name)
    throws fr.imag.clips.papillon.business.PapillonBusinessException {
        Group myGroup = null;
        String answerMessage = "";

        if (name==null || name.equals("")) {
            answerMessage = "Group name empty";
        } else {
            //search for an existing group
            myGroup = GroupsFactory.findGroupByName(name);
            if (null != myGroup && !myGroup.isEmpty()) {
                myGroup.delete();
                answerMessage = "Group: "+ myGroup.getName() + " deleted";
            }
        }
        return new GroupAnswer(myGroup,answerMessage);
    }
        
	public static Group[] getGroupsArray()  
	throws PapillonBusinessException {
		return getGroupsArray(null,null);
	}
    
	public static Group[] getAllGroupsArray()  
	throws PapillonBusinessException {
		Group[] tmpGroup = getGroupsArray(null,null);
		Vector myGroupsVector = new Vector(java.util.Arrays.asList(tmpGroup));
		Group specialistGroup = new Group();
		specialistGroup.setName(Group.SPECIALIST_GROUP);
		myGroupsVector.add(specialistGroup);

		Group validatorGroup = new Group();
		validatorGroup.setName(Group.VALIDATOR_GROUP);
		myGroupsVector.add(validatorGroup);

		Group adminGroup = new Group();
		adminGroup.setName(Group.ADMIN_GROUP);
		myGroupsVector.add(adminGroup);
		
		return (Group[]) myGroupsVector.toArray(new Group[0]);
	}
    
    public static Group[] getGroupsArray(String name, String password) 
        throws PapillonBusinessException {
        Group[] theDictArray = null;
        
        try {
            GroupQuery query = new GroupQuery(CurrentDBTransaction.get());
            
            if ((null != name) && (!name.trim().equals(""))) {
                query.getQueryBuilder().addWhereClause("name", name, 
                    QueryBuilder.CASE_INSENSITIVE_CONTAINS);
            }
            if ((null != password) && (!password.trim().equals(""))) {
                query.getQueryBuilder().addWhereClause("password", password,
                                                       QueryBuilder.CASE_INSENSITIVE_CONTAINS);
            }
           query.addOrderByName(true);
          //         query.addOrderByLogin(true);  
            GroupDO[] DOarray = query.getDOArray();
            theDictArray = new Group[ DOarray.length ];
            for ( int i = 0; i < DOarray.length; i++ )
	    	theDictArray[i] = new Group(DOarray[i]);

        }catch(Exception ex) {
            throw new fr.imag.clips.papillon.business.PapillonBusinessException("Exception in getGroupsArray()", ex);
        }
         
        return theDictArray;
    }
	
}
