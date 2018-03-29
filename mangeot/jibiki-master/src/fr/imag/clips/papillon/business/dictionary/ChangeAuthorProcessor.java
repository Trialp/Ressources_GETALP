/*
 *  Created by Mathieu Mangeot on 13/06/05.
 *  Copyright 2005 Mangeot-Sérasset. All rights reserved.
 ------------------------------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.5  2007/01/05 13:57:25  serasset
 * multiple code cleanup.
 * separation of XMLServices from the Utility class
 * added an xml parser pool to allow reuse of parser in a multithreaded context
 * added a new field in the db to identify the db layer version
 * added a new system property to know which db version is known by the current app
 *
 * Revision 1.4  2005/06/18 23:01:43  mangeot
 * changed the modif comment
 *
 * Revision 1.3  2005/06/17 15:51:32  mangeot
 * Modified changeAuthor
 *
 * Revision 1.2  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.1.2.1  2005/06/14 11:56:16  mangeot
 * Added a new page ChangeAuthor for changing the author of a set of previously selected contributions
 *
 *
 *-----------------------------------------------
 */ 

package fr.imag.clips.papillon.business.dictionary;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.user.User;

public class ChangeAuthorProcessor implements IVolumeEntryProcessor {

	protected static User newAuthor = null;
	protected static User validator = null;

	public ChangeAuthorProcessor(User theValidator, User author) {
		validator = theValidator;
		newAuthor = author;
	}

	public void process(VolumeEntry myEntry) throws PapillonBusinessException {
		try {
			if (newAuthor != null && ! newAuthor.equals("")) {
				String oldAuthor = myEntry.getAuthor();
				myEntry.setAuthor(newAuthor.getLogin());	
				myEntry.setGroups(newAuthor.getGroupsArray());	
				myEntry.setModification(validator.getLogin(), "Changed author: from " + oldAuthor + " to " + newAuthor.getLogin());
				myEntry.save();
			}
		}
		catch (Exception ex) {
			throw new PapillonBusinessException("Error in ChangeAuthorProcessor: ", ex);
		}
	}

}
