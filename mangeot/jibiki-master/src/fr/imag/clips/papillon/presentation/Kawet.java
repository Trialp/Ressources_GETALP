/*
 *  papillon
 *
 *  Enhydra super-servlet
 *
 *  © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 *  Projet Papillon
 *  -----------------------------------------------
 *  $Id: GetXmlDocument.java 700 2006-09-10 09:18:25Z mangeot $
 *  -----------------------------------------------
 *  $Log$
 *  Revision 1.1.2.1  2006/09/10 09:18:25  mangeot
 *  Resoures modified and added for playing with authentic xml plugin
 *
 *  Revision 1.7  2006/07/15 08:55:14  mangeot
 *  The BrowseVolumePage opens an HTML form that is used to lookup a volume in alphabetical order.
 *  The BrowseVolume is the server side of the AJAX script for retrieving the entries in alphabetical order
 *
 *  Revision 1.6  2006/02/27 00:04:01  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.5  2006/02/26 22:05:02  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.4  2006/02/26 20:24:30  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.3  2006/02/26 19:58:18  mangeot
 *  *** empty log message ***
 *
 *  Revision 1.2  2006/02/26 19:21:38  mangeot
 *  Work on BrowseVolume
 *
 *  Revision 1.1  2006/02/26 14:09:32  mangeot
 *  *** empty log message ***
 *
 *
 *  -----------------------------------------------
 *  beta version
 */
package fr.imag.clips.papillon.presentation;

import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.dods.builder.generator.query.QueryBuilder;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.user.User;
import fr.imag.clips.papillon.business.user.UsersFactory;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;

import fr.imag.clips.papillon.business.xml.XMLServices;

/**
*  Description of the Class
 *
 * @author     mangeot
 * @created    February 24, 2006
 */
public class Kawet extends XmlBasePO {
	
	protected static final String LOGIN_PARAMETER = "LOGIN";
	protected static final String EMAIL_PARAMETER = "EMAIL";
	protected static final String PASSWORD_PARAMETER = "PASSWORD";
	protected static final String CREDITS_PARAMETER = "CREDITS";
    	
    /**
    *  This method should be implemented in the subclass so that it returns
     *  true if this particular request requires the user to be logged in,
     *  otherwise false.
     *
     * @return    Description of the Return Value
     */
    protected boolean loggedInUserRequired() {
        return false;
    }
    
    /**
    *  This method should be implemented in the subclass so that it returns
     *  true if this particular request requires the user to be logged in,
     *  otherwise false.
     *
     * @return    Description of the Return Value
     */
    protected boolean userMayUseThisPO() {
        return true;
    }
	
    /**
        *  Returns the complete document.
     *
     * @exception  Exception
     * @exception  HttpPresentationException  Description of the Exception
     * @exception  IOException                Description of the Exception
     */
    public org.w3c.dom.Document getContent()
        throws HttpPresentationException, java.io.IOException, Exception {
			
			String errorDocument = "<?xml version='1.0' ?><error>";
			org.w3c.dom.Document resultDoc = null;
			
			String login  = myGetParameter(LOGIN_PARAMETER);
			String email = myGetParameter(EMAIL_PARAMETER);
			String password = myGetParameter(PASSWORD_PARAMETER);
			String credits = myGetParameter(CREDITS_PARAMETER);
			
			PapillonLogger.writeDebugMsg("Kawet: login: " + login+ " montant: "+ credits);
			
			if (login !=null && !login.equals("") && email !=null && !email.equals("")) {
				User myUser = UsersFactory.findUserByLogin(login);
				if (myUser != null && !myUser.isEmpty()) {
					if (myUser.getEmail().equals(email)) {
						if (password !=null && !password.equals("")
							&& credits !=null && !credits.equals("")) {
							if (myUser.HasCorrectPassword(password)) {
								int kawet = Integer.parseInt(credits)+myUser.getCredits();
								myUser.setCredits(kawet);
								myUser.save();
							}
							else {
								errorDocument += "wrong password</error>";
								resultDoc = XMLServices.buildDOMTree(errorDocument);
							}
						}
						resultDoc = XMLServices.buildDOMTree(myUser.getXmlCode());
					}
					else {
						errorDocument += "wrong email</error>";
						resultDoc = XMLServices.buildDOMTree(errorDocument);
					}
				}
				else {
					errorDocument += "user unknown</error>";
					resultDoc = XMLServices.buildDOMTree(errorDocument);
				}
			}			
			return resultDoc;			
        }
}
