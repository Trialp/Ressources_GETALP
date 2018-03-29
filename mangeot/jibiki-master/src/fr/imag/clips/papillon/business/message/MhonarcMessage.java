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

import java.io.*;
import java.util.*;
import java.text.*;

/* Cette classe a pour fonction de traiter un fichier html produit par mhonarc et en retire les informations
  interessantes, à savoir: le texte du message, l'auteur, la date, le sujet */

public class MhonarcMessage { 
    /*Ces constantes sont des strings qui serviront à reconnaitre les differentes régions du fichier*/
    final private static String DEBUT_MSG_ID = "<!--X-Message-Id:";
    final private static String FIN_MSG_ID = "-->";
    final private static String DEBUT_MSG ="<!--X-MsgBody-->";
    final private static String FIN_MSG ="<!--X-MsgBody-End-->" ;
    final private static String DEBUT_ENTETE = "<!--X-Head-of-Message-->" ;
    // final private static String rec_fin_entete ="<!--X-Head-of-Message-End-->" ;
    final private static String DEBUT_CORPS ="<!--X-Body-of-Message-->" ;
    final private static String FIN_CORPS ="<!--X-Body-of-Message-End-->" ;
    final private static String DEBUT_AUTEUR ="<li><em>From</em>:" ;
    final private static String DEBUT_AUTEUR_MAJ ="<li><em>FROM</em>:" ;
    final private static String FIN_AUTEUR ="</li>" ;
    final private static String DEBUT_ADRESSE_AUTEUR ="<a href=\"mailto:" ;
    final private static String FIN_ADRESSE_AUTEUR ="\">" ;
    final private static String FIN_LIEN_ADRESSE_AUTEUR ="</a>" ;
    final private static String DEBUT_DATE="<li><em>Date</em>:" ;    
    final private static String DEBUT_DATE_MAJ="<li><em>DATE</em>:" ;    
    final private static String FIN_DATE="</li>" ;
    final private static String DEBUT_SUJET ="<li><em>Subject</em>:" ;
    final private static String DEBUT_SUJET_MAJ ="<li><em>SUBJECT</em>:" ;
    final private static String FIN_SUJET ="</li>" ;
    final private static String QUOTE ="&quot;" ;
    final private static String AROBAS = "%40";

    // This should be UTF-8 when mhonarc will be correctly configured.
    final private static String MESSAGE_ENCODING = "UTF-8";
    
    public String corps = null;   //corps du message
    public String auteur = null;  // nom de l'auteur du message
    public String adresseAuteur = null;  // adresse mail de l'auteur
    public String sujet= null;     // sujet du message
    public Date date = null;    // date de la reception du message
    public String msgId = null;

    /*************************************************************
     ** MhonarcMessage (String nom) 
     **     throws NotAMhonarcMessageException, 
     **		   FileNotFoundException,
     **	           IOException
     ** Constructeur: La construction est faite à partir des infos contenues dans
     ** le fichier de nom "nom"
     **/
    MhonarcMessage (String nom) throws NotAMhonarcMessageException, 
				       FileNotFoundException,
				       IOException, 
                                       NotAValidUTF8Message  {
	int startPosition = -1;           // la position du début d'un élément 
	int spos;                         // la position de départ de la recherche de l'élément suivant
	int endPosition = -1;             // la position de fin d'un élément
	
	//	filename = new String(nom);
	
	// Création du reader avec le bon encodage
	InputStreamReader reader = new InputStreamReader(new FileInputStream(nom), MESSAGE_ENCODING);
	
	// Extraction du contenu du message
	String message = extractionMessage(reader);
	
	// Extraction des parties du message qui nous intéressent
	// Les infos de l'entête, destinataire, sujet, auteur, date (! dans cet ordre !)
	// Position de l'entête
	startPosition = message.indexOf(DEBUT_ENTETE);
	if (startPosition == -1) throw new NotAMhonarcMessageException("No message header found");
	
	// On cherche le sujet
	spos = endPosition;
	startPosition = message.indexOf(DEBUT_SUJET, spos);
	if (startPosition == -1) startPosition = message.indexOf(DEBUT_SUJET_MAJ, spos);
	if (startPosition == -1) throw new NotAMhonarcMessageException("Can't find \"Subject:\" line");
	endPosition = message.indexOf(FIN_SUJET, startPosition);
	if (endPosition == -1) throw new NotAMhonarcMessageException("Can't find \"Subject:\" line");
	sujet = message.substring(startPosition+DEBUT_SUJET.length(), endPosition).trim();

	// On cherche l'auteur et son adresse
	spos = endPosition;
	startPosition = message.indexOf(DEBUT_AUTEUR, spos);
	if (startPosition == -1) startPosition = message.indexOf(DEBUT_AUTEUR_MAJ, spos);
	if (startPosition == -1) throw new NotAMhonarcMessageException("Can't find \"From:\" line");
	endPosition = message.indexOf(FIN_AUTEUR, startPosition);
	if (endPosition == -1) throw new NotAMhonarcMessageException("Can't find \"From:\" line");
	String mhonarcAdresse = message.substring(startPosition+DEBUT_AUTEUR.length(), endPosition).trim();
	// L'adresse Mhonarc peut avoir plusieurs formes:
	// Louis Autret &lt;<A HREF="mailto:autretATidris.fr">autretATidris.fr</A>&gt;
	// <A HREF="mailto:terjeATin-progress.com">terjeATin-progress.com</A> (Terje Norderhaug)
	// <A HREF="mailto:VincentBMTATaol.com">VincentBMTATaol.com</A>
	int debutAdresse=-1, 
	    finAdresse = -1,
	    finLien = -1;
	debutAdresse = mhonarcAdresse.indexOf(DEBUT_ADRESSE_AUTEUR, 0);
	if (debutAdresse != -1) finAdresse = mhonarcAdresse.indexOf(FIN_ADRESSE_AUTEUR, debutAdresse);
	if (finAdresse != -1) finLien = mhonarcAdresse.indexOf(FIN_LIEN_ADRESSE_AUTEUR, finAdresse);
	if (debutAdresse == -1 || finAdresse == -1 || finLien == -1) {
	    adresseAuteur = "unknown";
	    auteur = mhonarcAdresse;
	} else {
	    adresseAuteur = mhonarcAdresse.substring(debutAdresse+DEBUT_ADRESSE_AUTEUR.length(), finAdresse).trim();
	    String avantAdresse = mhonarcAdresse.substring(0, debutAdresse).trim();
	    String apresAdresse = mhonarcAdresse.substring(finLien+FIN_LIEN_ADRESSE_AUTEUR.length(), mhonarcAdresse.length()).trim();
	    // Change %40 into @
	    int arobasPos = adresseAuteur.indexOf(AROBAS ,0);
	    if (arobasPos != -1) {
		adresseAuteur = adresseAuteur.substring(0, arobasPos) + "@" + adresseAuteur.substring(arobasPos + AROBAS.length());
	    }
	    // Si la partie qui précède l'adresse contient &lt;, on est dans le premier cas
	    int ltpos = avantAdresse.indexOf("&lt;", 0);
	    if (ltpos != -1) {
		auteur = avantAdresse.substring(0, ltpos).trim();
		if (auteur.startsWith(QUOTE) && auteur.endsWith(QUOTE)) 
		    auteur = auteur.substring(QUOTE.length(), auteur.length()-QUOTE.length());
	    } else {
		auteur = apresAdresse;
		if (auteur.startsWith("(") && auteur.endsWith(")")) 
		    auteur = auteur.substring(1, auteur.length()-1);
	    }
	}
	// On cherche la date
	spos = endPosition;
	startPosition = message.indexOf(DEBUT_DATE, spos);
	if (startPosition == -1) startPosition = message.indexOf(DEBUT_DATE_MAJ, spos);
	if (startPosition == -1) throw new NotAMhonarcMessageException("Can't find \"Date:\" line");
	endPosition = message.indexOf(FIN_DATE, startPosition);
	if (endPosition == -1) throw new NotAMhonarcMessageException("Can't find \"Date:\" line");
	
	// On parse la date
	SimpleDateFormat df = new SimpleDateFormat("EEE, dd MMM yyyy HH:mm:ss zzz");
	ParsePosition parsePos = new ParsePosition(0);
	date = df.parse(message.substring(startPosition+DEBUT_DATE.length(), endPosition).trim(), parsePos);
	// Si la date n'a pas été reconnue, on met la date du jour...
	if (date==null) date = new Date();
	
	//Le corps du message
	spos = endPosition;
	startPosition = message.indexOf(DEBUT_CORPS, spos);
	if (startPosition == -1) throw new NotAMhonarcMessageException("Can't find the body of the message");
	endPosition = message.indexOf(FIN_CORPS, startPosition);
	if (endPosition == -1) throw new NotAMhonarcMessageException("Can't find the body of the message");
	corps = message.substring(startPosition+DEBUT_CORPS.length(), endPosition).trim();
	
    }
    
    /*-----------------------------------------------------
     * extractionMessage(InputStreamReader reader)
     *-----------------------------------------------------
     * Le fichier lu contient des infos propres à Mhonarc. 
     * La seule chose qui nous intéresse est la message proprement dit. 
     * cette methode renvoie la partie du fichier qui nous intéresse. 
     */
    
    private String extractionMessage (InputStreamReader reader) throws NotAMhonarcMessageException,
								       IOException, 
                                                                       NotAValidUTF8Message {
	String message;
	
	try {
	    BufferedReader lect = new BufferedReader (reader);
	    StringBuffer messagebuff = new StringBuffer();
	    String temp;
	    while (! (temp = lect.readLine().trim()).startsWith(DEBUT_MSG_ID)) ;
	    // On récupère le msgId en passant...
	    if (temp.endsWith(FIN_MSG_ID))
		msgId = temp.substring(DEBUT_MSG_ID.length(), temp.length()-FIN_MSG_ID.length()).trim();
	    else {
		Date now = new Date();
		Long time = new Long(now.getTime());
		msgId = new String(time.toString()) +"@papillon.imag.fr";
	    }
	    while (! lect.readLine().equals(DEBUT_MSG)) ;	
	    temp = lect.readLine();
	    while (!temp.equals(FIN_MSG)) {
		messagebuff = messagebuff.append(temp).append("\n");
		temp = lect.readLine();
	    }
	    message = messagebuff.toString();    
	} catch (NullPointerException n) { 
	    // Cette exception ne peut survenir que si on n'a pas trouvé l'élément de fin du body
	    throw new NotAMhonarcMessageException();
	} catch (java.io.CharConversionException ex) {
            throw new NotAValidUTF8Message();
        }

	return message;
    }

    /*************************************************************
     ** public String sujetCannonique()
     ** Renvoie le sujet canonique du message.
     ** Fwd: toto --> toto
     ** Re: toto --> toto
     */

    public String sujetCannonique() {
	return sujetCannonique(sujet);
    }

    public static String sujetCannonique(String str) {
	if (str.toUpperCase().startsWith("RE:")) {
	    return sujetCannonique(str.substring(3).trim());
	} else if (str.toUpperCase().startsWith("FWD:")) {
	    return sujetCannonique(str.substring(4).trim());
	} else if (str.toString().toUpperCase().startsWith("FW:")) {
	    return sujetCannonique(str.substring(3).trim());
	} else if (str.startsWith("[") && str.endsWith("]")) {
	    return sujetCannonique(str.substring(1,str.length()-1).trim());
	} else return str;

    }
    
    // Programme de test...
    public static void main (String args []) throws Exception {
	MhonarcMessage monFichier;

	for (int i = 0; i < args.length; i++) {
	    monFichier = new MhonarcMessage(args[i]);
	    System.out.println("["+i+"]================================================================");
	    System.out.println(monFichier.sujet);
	    System.out.println(monFichier.msgId);
	    
	    System.out.println("Par " +monFichier.auteur + ", le " + monFichier.date);
	    System.out.println();

	    System.out.println("Adresse de départ :"+monFichier.adresseAuteur);
	    //  System.out.println("Adresse d'arrivée :"+monFichier.adresseDest);
	    
	    System.out.println("Date :"+monFichier.date);
	}

    }
    
}
