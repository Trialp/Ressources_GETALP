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
 * Revision 1.3  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.2.2.1  2005/03/29 09:41:33  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:31  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 *-----------------------------------------------
 * 
 */

package fr.imag.clips.papillon.business.message;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.CurrentDBTransaction;

import fr.imag.clips.papillon.data.MessageDO;
import fr.imag.clips.papillon.data.MessageQuery;
import fr.imag.clips.papillon.business.PapillonLogger;

import com.lutris.appserver.server.sql.ObjectId;
import com.lutris.appserver.server.sql.DBConnection;
import com.lutris.dods.builder.generator.query.QueryBuilder;
import com.lutris.appserver.server.Enhydra;
import com.lutris.appserver.server.sql.DBQuery;
import com.lutris.appserver.server.sql.ObjectIdException;
import com.lutris.appserver.server.sql.DatabaseManagerException;

import java.sql.ResultSet;
/**
 * Used to find the instances of message.
 */
public class MessageFactory {
    public final static String SORT_BY_DATE = "date";
    public final static String SORT_BY_AUTHOR = "author";
    public final static String SORT_BY_SUBJECT = "subject";
    
    /** 
     * The getMessagesArray method performs a database query to
     * return an array of <CODE>Messages</CODE> objects
     * representing all the rows in the <CODE>message</CODE> table. 
     *
     * @return
     *     array of discs. 
     * @exception PapillonBusinessException
     *   If there is a problem retrieving disc information.
     */
    public static Message[] getMessagesArray(String authorcontains, 
                                             String subjectcontains, 
                                             String messagecontains, 
                                             String thread, 
                                             String sort,
                                             int offset,
                                             int limit) 
        throws PapillonBusinessException {
        Message[] theMessageArray = null;
        
        try {
            MessageQuery query = new MessageQuery(CurrentDBTransaction.get());
            
            if ((null != thread) && ("" != thread.trim())) {
                query.setQueryThread(thread);
            } else {
                if ((null != authorcontains) && (!authorcontains.trim().equals(""))) {
                    query.getQueryBuilder().addWhereOpenParen();
                    query.getQueryBuilder().addWhereClause("author", authorcontains, 
                        QueryBuilder.CASE_INSENSITIVE_CONTAINS);
                    query.getQueryBuilder().addWhereOr();
                    query.getQueryBuilder().addWhereClause("authoraddress", authorcontains, 
                        QueryBuilder.CASE_INSENSITIVE_CONTAINS);
                    query.getQueryBuilder().addWhereCloseParen();
                }
                if ((null != subjectcontains) && (!subjectcontains.trim().equals(""))) {
                    query.getQueryBuilder().addWhereClause("subject", subjectcontains, 
                        QueryBuilder.CASE_INSENSITIVE_CONTAINS);
                }
                if ((null != messagecontains) && (!messagecontains.trim().equals(""))) {
                    query.getQueryBuilder().addWhereClause("msg", messagecontains, 
                        QueryBuilder.CASE_INSENSITIVE_CONTAINS);
                }
            }
            
            // Order 
            if ((null != sort) && (sort.equals(SORT_BY_DATE))) {
                query.addOrderByDate(false);  // Or true for ascending ?
            } else if ((null != sort) && (sort.equals(SORT_BY_AUTHOR))) {
                query.addOrderByAuthor();  // ascending, what about descending ?
            } else if ((null != sort) && (sort.equals(SORT_BY_SUBJECT))) {
                query.addOrderByThread();  // ascending, what about descending ?
            } else {
                query.addOrderByDate(false);  // Or true for ascending ?
            }
        
            QueryBuilder qb = query.getQueryBuilder();
            qb.addEndClause(" LIMIT " + limit + " OFFSET " + offset);
            
            //query.getQueryBuilder().debug();
            MessageDO[] DOarray = query.getDOArray();
            theMessageArray = new Message[ DOarray.length ];
            for ( int i = 0; i < DOarray.length; i++ )
	    	theMessageArray[i] = new Message(DOarray[i]);
        }catch(Exception ex) {
            throw new PapillonBusinessException("Exception in getMessagesArray()", ex);
        }
        
        return theMessageArray;
    }
    
    
    /** 
     * The getMessagesCount method performs a database query to
     * return the number of <CODE>Messages</CODE> objects
     * representing all the rows matching criteria in the <CODE>message</CODE> table. 
     *
     * @return
     *     number of messages. 
     * @exception PapillonBusinessException
     *   If there is a problem retrieving information.
     */
    public static int getMessagesCount(String authorcontains, 
                                  String subjectcontains, 
                                  String messagecontains, 
                                  String thread, 
                                  String sort) 
        throws PapillonBusinessException {
        int nbmsg = 0;
        
        DBConnection c = null;
        try {
            // Preparing the Query :

            QueryBuilder qb = new QueryBuilder("messages", "count(objectid)");
            qb.setDatabaseVendor( "PostgreSQL" );
            qb.setStringMatchDetails( "LIKE", "%" );
//            qb.reset();
            
            if ((null != thread) && ("" != thread.trim())) {
                qb.addWhere( MessageDO.Thread, thread, QueryBuilder.EQUAL );
            } else {
                if ((null != authorcontains) && (!authorcontains.trim().equals(""))) {
                    
                    qb.addWhereOpenParen();
                    qb.addWhereClause("author", authorcontains, 
                        QueryBuilder.CASE_INSENSITIVE_CONTAINS);
                    qb.addWhereOr();
                    qb.addWhereClause("authoraddress", authorcontains, 
                        QueryBuilder.CASE_INSENSITIVE_CONTAINS);
                    qb.addWhereCloseParen();
                }
                if ((null != subjectcontains) && (!subjectcontains.trim().equals(""))) {
                    qb.addWhereClause("subject", subjectcontains, 
                        QueryBuilder.CASE_INSENSITIVE_CONTAINS);
                }
                if ((null != messagecontains) && (!messagecontains.trim().equals(""))) {
                    qb.addWhereClause("msg", messagecontains, 
                        QueryBuilder.CASE_INSENSITIVE_CONTAINS);
                }
            }
                        
            //qb.debug();
            
            c = Enhydra.getDatabaseManager().allocateConnection();
            ResultSet rs = qb.executeQuery(c);
            if (rs.next()) {
                nbmsg = rs.getInt("count");
            }
            c.reset();
        } catch (Exception e) {
            throw new PapillonBusinessException( "Database connection Error", e );
        } finally {
            if ( null != c )
                c.release();
        }

        return nbmsg;
    }


    /** 
     * The findMessageByID method performs a database query to
     * return a <CODE>Message</CODE> object
     * representing the row in the <CODE>message</CODE> table
     * that matches the object id. 
     *
     * @param id, the object id of the message table.
     * @return
     *    the message. null if there isn't a message associated
     *    to the id
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
    public static Message findMessageByID(String id) 
        throws PapillonBusinessException {
        Message theMessage = null;
        
        try {
            MessageQuery query = new MessageQuery(CurrentDBTransaction.get());
            //set query
            query.setQueryOId(new ObjectId(id));
            // Throw an exception if more than one message is found
            query.requireUniqueInstance();
            MessageDO theMessageDO = query.getNextDO();
            theMessage = new Message(theMessageDO);
            return theMessage;
        }catch(Exception ex) {
            throw new PapillonBusinessException("Exception in findMessageByID()", ex);
        }
    }
    
    /** 
     * The numberOfMessageWithThread method performs a database query to
     * return a <CODE>int</CODE>
     * representing the number of <CODE>message</CODE>
     * that matches the thread. 
     *
     * @param thread, the thread.
     * @return
     *    the number of messages
     * @exception PapillonBusinessException
     *    if there is a problem retrieving message.
     */
    public static int numberOfMessagesWithThread(String thread) 
        throws PapillonBusinessException {
        int result = 0;
        try {
            QueryBuilder qb;
            DBConnection conn = Enhydra.getDatabaseManager().allocateConnection();
            qb = new QueryBuilder( "messages", "count(Date)" );
            qb.addWhereClause( "Thread", thread, QueryBuilder.CASE_INSENSITIVE_EQUAL);
            //qb.debug();
            java.sql.ResultSet rs = qb.executeQuery(conn);
            if (rs.next()) result= rs.getInt(1);
            conn.release();
            return result;
        }catch(Exception ex) {
            throw new PapillonBusinessException("Exception in findMessageByID()", ex);
        }
    }

     
}

