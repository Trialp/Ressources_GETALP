/*
 *  papillon
 *
 *  Enhydra super-servlet presentation object
 * 
 * © Mathieu Mangeot - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.3  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.2  2005/05/24 12:51:22  serasset
 * Updated many aspect of the Papillon project to handle lexalp project.
 * 1. Layout is now parametrable in the application configuration file.
 * 2. Notion of QueryResult has been defined to handle mono/bi and multi lingual dictionary requests
 * 3. Result presentation may be done by way of standard xsl or with any class implementing the appropriate interface.
 * 4. Enhanced dictionary edition management. The template interfaces has to be revised to be compatible.
 * 5. It is now possible to give a name to the cookie key in the app conf file
 * 6. Several bug fixes.
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * Revision 1.2  2004/02/10 05:27:13  mangeot
 * The version UIGEN_V2 has been merged with the trunk by MM
 * Be careful because the Volumes and contributions database tables have been modified.
 * You have to drop and rebuild them unless you modify them by hands.
 *
 * Revision 1.1.2.1  2004/01/08 15:18:36  mangeot
 * New object from database, do not forget to run the sql script create_tables.sql
 * To manage groups
 *
 * Revision 1.1  2002/11/26 06:38:47  mangeot
 * Work on RegisterUser: fixed bugs and clarified !
 *
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.user;

/* FIXME: this class is a convenient class. It is used only to return a user and
a message to the RegisterUser.java class. If the createNewUser does not work,
a message explains it.

We should find a generic system for all the answers and multilingual !
*/
 

public class GroupAnswer {


    /**
         */
    protected Group myGroup = null;
    protected String myAnswer = "";

        /**
            * The public constructor.
         */
    public GroupAnswer(Group aGroup, String anAnswer) {
        myGroup = aGroup;
        myAnswer = anAnswer;
    }

    public String getMessage() {
        return myAnswer;
    }
    public Group getGroup() {
        return myGroup;
    }
    public boolean isEmpty() {
        return (myGroup == null || myGroup.isEmpty());
    }
}