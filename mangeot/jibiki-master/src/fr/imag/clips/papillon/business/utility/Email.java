/*
 * Email.java
 *------------------------------------------
 *$Id: Email.java 685 2006-08-27 20:52:57Z mangeot $
 *------------------------------------------
 *$Log$
 *Revision 1.1.2.1  2006/08/27 20:52:57  mangeot
 *Added email functionalities.
 *javamail 1.4 and java activation framework must be added
 *
 *
 *------------------------------------------
 *
 * Code copied from Javamail smtpsend.java example
 *
 * Requires javamail 1.4 and java activation framework 1.1
 *
 */
package fr.imag.clips.papillon.business.utility;

import fr.imag.clips.papillon.business.PapillonLogger;
import com.lutris.appserver.server.Enhydra;

import javax.mail.*;
import javax.mail.internet.*;

import com.sun.mail.smtp.*;

/*
 Given a certificate for the server as used in case #2 above, you can
 import this certificate into your Java keystore file using a command
 such as:
 
	keytool -import -alias imap-server -file imap.cer
 */ 

public class Email {
	
	protected static final String JAVAMAIL_FROM_ADDRESS = "Papillon.javamail.from";
	protected static final String JAVAMAIL_MAILHOST_ADDRESS = "Papillon.javamail.mailhost.address";
	protected static final String JAVAMAIL_MAILHOST_USER = "Papillon.javamail.mailhost.user";
	protected static final String JAVAMAIL_MAILHOST_PASSWORD = "Papillon.javamail.mailhost.password";
	protected static final String JAVAMAIL_MAILHOST_AUTH = "Papillon.javamail.mailhost.auth";
	protected static final String JAVAMAIL_MAILHOST_SSL = "Papillon.javamail.mailhost.ssl";
	protected static final String javax_net_ssl_trustStore = "javax.net.ssl.trustStore";
	protected static final String javax_net_ssl_trustStorePassword = "javax.net.ssl.trustStore";
	protected static final String JAVAMAIL_RECORDHOST_ADDRESS = "Papillon.javamail.recordhost.address";
	
	protected static final String mailer = "smtpsend";
	
	protected static String from = null;
	protected static String mailhost = null;
	protected static String user = null;
	protected static String password = null;
	protected static String recordhost = null;
	protected static boolean auth =  true;
	protected static boolean ssl =  true;
	
	protected static boolean init =  false;
	
	public static boolean send(String to, String subject, String cc, String text, boolean debug, boolean verbose) {
		return send(to, subject, cc, null, text, null, 
			 null, null, null,
			 debug, verbose);
	}
	
	public static boolean send(String to, String subject, String cc, String text) {
		return send(to, subject, cc, null, text, null, 
			 null, null, null,
			 false, false);
	}
	
	public static boolean send(String to, String subject, String cc, String bcc, String text, String file) {
		return send(to, subject, cc, bcc, text, file, 
			 null, null, null,
			 false, false);
	}
	
	public static boolean send(String to, String subject, String cc, String bcc, String text, String file,
							boolean debug, boolean verbose) {
		return send(to, subject, cc, bcc, text, file, 
			 null, null, null,
			 debug, verbose);
	}
	
	public static boolean send(String to, String subject, String cc, String bcc, String text, String file,
							String record, String url, String protocol, 
							boolean debug, boolean verbose) {
		
		boolean sent = false;
		try {
			
			if (!init) {
				init = initialize();
			}
			java.util.Properties props = System.getProperties();
			if (mailhost != null) {
				props.put("mail.smtp.host", mailhost);
			}
			if (auth) {
				props.put("mail.smtp.auth", "true");
				props.put("mail.smtps.auth", "true");
			}
			
			// Get a Session object
			Session session = Session.getInstance(props, null);
			if (debug) {
				session.setDebug(true);
			}
			// construct the message
			Message msg = new MimeMessage(session);
			if (from != null) {
				msg.setFrom(new InternetAddress(from));
			}
			else {
				msg.setFrom();
			}
			
			msg.setRecipients(Message.RecipientType.TO,
							  InternetAddress.parse(to, false));
			if (cc != null)
				msg.setRecipients(Message.RecipientType.CC,
								  InternetAddress.parse(cc, false));
			if (bcc != null)
				msg.setRecipients(Message.RecipientType.BCC,
								  InternetAddress.parse(bcc, false));
			
			msg.setSubject(subject);
			
			if (file != null) {
				// Attach the specified file.
				// We need a multipart message to hold the attachment.
				MimeBodyPart mbp1 = new MimeBodyPart();
				mbp1.setText(text);
				MimeBodyPart mbp2 = new MimeBodyPart();
				// Error when  compiling inside Enhydra: no such method error
				// No error when compiling smtpsend.java alone!!!
				//mbp2.attachFile(file);
				MimeMultipart mp = new MimeMultipart();
				mp.addBodyPart(mbp1);
				mp.addBodyPart(mbp2);
				msg.setContent(mp);
			} else {
				// If the desired charset is known, you can use
				// setText(text, charset)
				//msg.setText(text, "UTF-8");
				msg.setText(text);
			}
			
			msg.setHeader("X-Mailer", mailer);
			msg.setSentDate(new java.util.Date());
						
			// send the thing off
			/*
			 * The simple way to send a message is this:
			 *
			 Transport.send(msg);
			 *
			 * But we're going to use some SMTP-specific features for
			 * demonstration purposes so we need to manage the Transport
			 * object explicitly.
			 */
			SMTPTransport t =
				(SMTPTransport)session.getTransport(ssl ? "smtps" : "smtp");
			try {
				if (auth) {
					t.connect(mailhost, user, password);
				}
				else
					t.connect();
				t.sendMessage(msg, msg.getAllRecipients());
			} finally {
				if (verbose)
					PapillonLogger.writeDebugMsg("Email Response: " +
												 t.getLastServerResponse());
				t.close();
			}
			PapillonLogger.writeDebugMsg("Email To: " + to + " Subject: '" + subject + "' was sent successfully.");
			sent = true;
			
			// Keep a copy, if requested.
			
			if (record != null) {
				// Get a Store object
				Store store = null;
				if (url != null) {
					URLName urln = new URLName(url);
					store = session.getStore(urln);
					store.connect();
				} else {
					if (protocol != null)		
						store = session.getStore(protocol);
					else
						store = session.getStore();
					
					// Connect
					if (recordhost != null || user != null || password != null)
						store.connect(recordhost, user, password);
					else
						store.connect();
				}
				
				// Get record Folder.  Create if it does not exist.
				Folder folder = store.getFolder(record);
				if (folder == null) {
					PapillonLogger.writeDebugMsg("Email Error: Can't get record folder.");
					//System.exit(1);
				}
				if (!folder.exists())
					folder.create(Folder.HOLDS_MESSAGES);
				
				Message[] msgs = new Message[1];
				msgs[0] = msg;
				folder.appendMessages(msgs);
				
				PapillonLogger.writeDebugMsg("Email To: " + to + " Subject: " + subject + " was recorded successfully.");
			}
			
		} catch (Exception e) {
			if (e instanceof SendFailedException) {
				MessagingException sfe = (MessagingException)e;
				if (sfe instanceof SMTPSendFailedException) {
					SMTPSendFailedException ssfe =
					(SMTPSendFailedException)sfe;
					PapillonLogger.writeDebugMsg("Email Error: SMTP SEND FAILED:");
					if (verbose)
						PapillonLogger.writeDebugMsg(ssfe.toString());
					PapillonLogger.writeDebugMsg("Email Error:  Command: " + ssfe.getCommand());
					PapillonLogger.writeDebugMsg("Email Error:  RetCode: " + ssfe.getReturnCode());
					PapillonLogger.writeDebugMsg("Email Error:  Response: " + ssfe.getMessage());
				} else {
					if (verbose)
						PapillonLogger.writeDebugMsg("Email Error: Send failed: " + sfe.toString());
				}
				Exception ne;
				while ((ne = sfe.getNextException()) != null &&
					   ne instanceof MessagingException) {
					sfe = (MessagingException)ne;
					if (sfe instanceof SMTPAddressFailedException) {
						SMTPAddressFailedException ssfe =
						(SMTPAddressFailedException)sfe;
						PapillonLogger.writeDebugMsg("Email Error: ADDRESS FAILED:");
						if (verbose) {
							PapillonLogger.writeDebugMsg(ssfe.toString());
						}
						PapillonLogger.writeDebugMsg("Email Error:  Address: " + ssfe.getAddress());
						PapillonLogger.writeDebugMsg("Email Error:  Command: " + ssfe.getCommand());
						PapillonLogger.writeDebugMsg("Email Error:  RetCode: " + ssfe.getReturnCode());
						PapillonLogger.writeDebugMsg("Email Error:  Response: " + ssfe.getMessage());
					} else if (sfe instanceof SMTPAddressSucceededException) {
						PapillonLogger.writeDebugMsg("Email Error: ADDRESS SUCCEEDED:");
						SMTPAddressSucceededException ssfe =
							(SMTPAddressSucceededException)sfe;
						if (verbose)
							PapillonLogger.writeDebugMsg(ssfe.toString());
						PapillonLogger.writeDebugMsg("Email Error:   Address: " + ssfe.getAddress());
						PapillonLogger.writeDebugMsg("Email Error:   Command: " + ssfe.getCommand());
						PapillonLogger.writeDebugMsg("Email Error:   RetCode: " + ssfe.getReturnCode());
						PapillonLogger.writeDebugMsg("Email Error:   Response: " + ssfe.getMessage());
					}
				}
			} else {
				PapillonLogger.writeDebugMsg("Email Error: Got Exception: " + e);
				if (verbose) {
					e.printStackTrace();
				}
			}
		}
		return sent;
	}
	
	public static String getFromAddress() {
		return from;
	}
	
	protected static boolean initialize() {
		try {
			from = Enhydra.getApplication().getConfig().getString(JAVAMAIL_FROM_ADDRESS);
			mailhost = Enhydra.getApplication().getConfig().getString(JAVAMAIL_MAILHOST_ADDRESS);
			user = Enhydra.getApplication().getConfig().getString(JAVAMAIL_MAILHOST_USER);
			password = Enhydra.getApplication().getConfig().getString(JAVAMAIL_MAILHOST_PASSWORD);
			//recordhost = Enhydra.getApplication().getConfig().getString(JAVAMAIL_RECORDHOST_ADDRESS);
			auth =  Enhydra.getApplication().getConfig().getBoolean(JAVAMAIL_MAILHOST_AUTH);
			ssl =  Enhydra.getApplication().getConfig().getBoolean(JAVAMAIL_MAILHOST_SSL);
			if (ssl) {
				System.setProperty(javax_net_ssl_trustStore, Enhydra.getApplication().getConfig().getString(javax_net_ssl_trustStore));
				System.setProperty(javax_net_ssl_trustStorePassword, Enhydra.getApplication().getConfig().getString(javax_net_ssl_trustStorePassword));
			}
		}
		catch (com.lutris.util.ConfigException ce) {
			PapillonLogger.writeDebugMsg("Email com.lutris.util.ConfigException: ");
			ce.printStackTrace();
		}
		return true;
	}
	
}
