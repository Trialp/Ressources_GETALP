//
//  PapillonSessionManager.java
//  PapillonStable
//
//  Created by Mathieu Mangeot on 22/10/04.
//  Copyright 2004 __MyCompanyName__. All rights reserved.
//
package fr.imag.clips.papillon.presentation;

import com.lutris.appserver.server.session.Session;

public class PapillonSessionManager {

	// Note: I had to reimplement a sesisonManager because there is a possible bug in
	// SessionManager.getSession(sessionKey)

	protected static java.util.Hashtable SessionsTable = new java.util.Hashtable();
	// methods for the SessionManager
	
	public static void addNewSession(Session newSession) 
		throws fr.imag.clips.papillon.presentation.PapillonPresentationException {
		addNewSession(newSession, null);
	}
	
    public static void addNewSession(Session newSession, com.lutris.appserver.server.user.User myUser)
    throws PapillonPresentationException {
        if (myUser!=null) {
            try {
                newSession.setUser(myUser);
            }
            catch (com.lutris.appserver.server.session.SessionException SesEx) {
                throw new PapillonPresentationException("PapillonSessionManager:Session Error for setUser: ",SesEx);
            }
        }
        SessionsTable.put(newSession.getSessionKey(), newSession);
        updateActiveSessions();
    }
    
    public static Session getSession(String sessionKey)
    throws PapillonPresentationException {
        return (Session) SessionsTable.get(sessionKey);
    }
    
	protected static java.util.Vector getActiveSessionsVector()
		throws PapillonPresentationException {
		java.util.Vector ActiveSessions = new java.util.Vector();
		updateActiveSessions();
		for (java.util.Enumeration keys = SessionsTable.keys(); keys.hasMoreElements();) {					
			String key = (String) keys.nextElement();
			ActiveSessions.add(SessionsTable.get(key));
		}		
		return ActiveSessions;
	}

	public static java.util.Vector getActiveUsersVector() 
		throws PapillonPresentationException {
		java.util.Vector ActiveSessions = getActiveSessionsVector();
		java.util.Vector ActiveUsers = new java.util.Vector();
		for (int i=0; i< ActiveSessions.size();i++) {
			Session mySession = (Session) ActiveSessions.elementAt(i);
			fr.imag.clips.papillon.business.user.User myUser = (fr.imag.clips.papillon.business.user.User) mySession.getUser();
			if (!ActiveUsers.contains(myUser)) {
				ActiveUsers.add(myUser);
			}
		}		
		return ActiveUsers;
	}

	protected static void updateActiveSessions() 
		throws PapillonPresentationException {
			com.lutris.appserver.server.session.SessionManager mySessionManager = com.lutris.appserver.server.Enhydra.getSessionManager();
			try {
				if (mySessionManager.activeSessionCount()<SessionsTable.size()) {
					java.util.Vector KeysVector = new java.util.Vector();
					for (java.util.Enumeration SessionKeys = mySessionManager.getSessionKeys(); SessionKeys.hasMoreElements();) {					
						KeysVector.add((String) SessionKeys.nextElement());
					}
					for (java.util.Enumeration keys = SessionsTable.keys(); keys.hasMoreElements();) {					
						String key = (String) keys.nextElement();
						if (!KeysVector.contains(key)) {
							SessionsTable.remove(key);
						}
					}
				}
			}
			catch (com.lutris.appserver.server.session.SessionException SesEx) {
				throw new PapillonPresentationException("PapillonSessionManager: Error in getSessionKeys: ", SesEx);
			}	
		}
	}
