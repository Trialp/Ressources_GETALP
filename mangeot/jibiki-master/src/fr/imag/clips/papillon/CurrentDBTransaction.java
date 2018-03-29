/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.2  2005/04/11 12:29:59  mangeot
 * Merge between the XPathAndMultipleKeys branch and the main trunk
 *
 * Revision 1.1.2.1  2005/03/29 09:41:32  serasset
 * Added transaction support. Use CurrentDBTransaction class to define a transaction
 * context in which all db commands will be executed.
 *
 *
 *-----------------------------------------------
 * Papillon enhydra Thread Local for current Transaction.
 */

package fr.imag.clips.papillon;

import fr.imag.clips.papillon.business.PapillonBusinessException;

import com.lutris.appserver.server.sql.DBTransaction;
import org.enhydra.dods.DODS;

/**
 * the CurrentTransaction class is a Thread Local that returns the db transaction
 * associated to the current thread if any.
 */
public class CurrentDBTransaction {
    
    private static ThreadLocal currentDBTransaction = new ThreadLocal() {
        protected synchronized Object initialValue() {
            return null;
        }
    };
    
    public static DBTransaction get() {
        return ((DBTransaction) currentDBTransaction.get());
    }
    
    public static void registerNewDBTransaction() throws PapillonBusinessException {
        if ( null != currentDBTransaction.get() ) throw new PapillonBusinessException("Attempt to register an already existing Transaction.");
        DBTransaction dbt = null;
        try {
            dbt = DODS.getDatabaseManager().createTransaction();
            currentDBTransaction.set(dbt);
        } catch (com.lutris.appserver.server.sql.DatabaseManagerException e) {
            throw new PapillonBusinessException("Could not allocate database transaction.", e);
        } catch (java.sql.SQLException e) {
            throw new PapillonBusinessException("Could not allocate database transaction.", e);
        } 
    }
    
    public static void releaseCurrentDBTransaction() throws PapillonBusinessException {
        DBTransaction dbt = ((DBTransaction) currentDBTransaction.get());
        if (null == dbt ) throw new PapillonBusinessException("Attempt to release a null transaction.");
        dbt.release();
        currentDBTransaction.set(null);
    }
}