package fr.imag.clips.papillon.business.lexalp;

import fr.imag.clips.papillon.business.transformation.ResultFormatter;
import fr.imag.clips.papillon.business.xsl.XslSheet;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.business.xsl.JibikiXsltExtension;
import fr.imag.clips.papillon.business.dictionary.*;
import fr.imag.clips.papillon.business.dictionary.Dictionary;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.user.User;

import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.dom.DOMResult;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import java.util.*;

import org.w3c.dom.Node;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class LexALPSimpleFormatter
        implements ResultFormatter {

    // Constants
    // Note: I use constants extensively because the XSL transformations are a little
    // bit slow
    protected static Hashtable XslSheetCache = new Hashtable();

    //outils pour les transformation
    protected static final TransformerFactory myTransformerFactory = TransformerFactory.newInstance();
    protected static final DocumentBuilderFactory myDocumentBuilderFactory = DocumentBuilderFactory.newInstance();
    protected static DocumentBuilder myDocumentBuilder;

    protected XslSheet dictXsl = null;

    protected Dictionary currentDictionary;
    protected Volume sourceVolume;

    //
    private final boolean DEBUG = false;

    //
    public void initializeFormatter(Dictionary dictionary, Volume volume, Object parameter, int dialect, String lang)
            throws PapillonBusinessException {

        try {

            //
            String dictionaryName = dictionary.getName();
            String volumeName = volume.getName();

            //
            if (myDocumentBuilder == null) {
                myDocumentBuilder = myDocumentBuilderFactory.newDocumentBuilder();
            }

            // FIXME: specific name for the dictionary in order to have 2 different formatters in parallel
            String dicName = (null == dictionaryName) ? "_2" : dictionaryName + "_2";
            dictXsl = getXslSheet(dictionaryName, volumeName, (String) parameter);

        } catch (javax.xml.parsers.ParserConfigurationException e) {
            throw new PapillonBusinessException("CRITICAL: error initializing document builder !", e);
        }

    }

    public Node getFormattedResult(QueryResult qr, User usr)
            throws PapillonBusinessException {

        Document res = myDocumentBuilder.newDocument();
        Element rootdiv = res.createElement("div");
        rootdiv.setAttribute("class", "entry");
        res.appendChild(rootdiv);

        if (qr.getResultKind() == QueryResult.UNIQUE_RESULT) {
            if (null != dictXsl && !dictXsl.isEmpty()) {
                // Format document source
                Node resultNode = formatResult(qr.getSourceEntry().getDom(), dictXsl, usr);
                rootdiv.appendChild(res.importNode(resultNode, true));
            }

        } else if (qr.getResultKind() == QueryResult.AXIE_COLLECTION_RESULT) {
            if (null != dictXsl && !dictXsl.isEmpty()) {
                // Format document source
                Node resultNode = formatResult(qr.getResultAxie().getDom(), dictXsl, usr);
                rootdiv.appendChild(res.importNode(resultNode, true));
            }
        }
        return rootdiv;
    }

    /**
     * ...
     *
     * @param docSource DOM version of the volume entry to be formatted
     * @param xsl       XslSheet used for formatting
     * @param usr       User currently requesting the result.
     * @return Root node representing the
     * @throws fr.imag.clips.papillon.business.PapillonBusinessException
     *
     */
    private Node formatResult(Document docSource, XslSheet xsl, User usr)
            throws PapillonBusinessException {
        try {

            // init
            boolean isTransform = true;
            Document docCible = null;

            docCible = Transform(docSource, xsl);

            return docCible.getDocumentElement();

        } catch (Exception ex) {
            throw new PapillonBusinessException("Exception in getFormattedResult()", ex);
        }
    }


    /**
     * Transform the xml source by processing it with an xsl sheet.
     */
    protected static Document Transform(Node xmlSource, XslSheet xslSheet)
            throws PapillonBusinessException, javax.xml.parsers.ParserConfigurationException,
                   javax.xml.transform.TransformerException, java.io.IOException {

        // Find transformer
        // FIXME: if transformer is null ?
        Transformer myTransformer = xslSheet.getTransformer();

        // PapillonLogger.writeDebugMsg("Class: " + myTransformer.getClass());
        // PapillonLogger.writeDebugMsg("Factory: " + TransformerFactory.class.toString());
        // The xalan transformer uses the context class loader instead of the system class loader
        // But in enhydra 5.1, the system class loader is a custom class loader, hence we have
        // to set the context class loader to the enhydra classloader, or the Extension classes
        // would not be visible.
        try {
            Thread currentThread = Thread.currentThread();
            currentThread.setContextClassLoader(JibikiXsltExtension.class.getClassLoader());
        } catch (SecurityException e) {
            PapillonLogger.writeDebugMsg("Could not modify the context class loader.");
        }

        // The source
        if (myDocumentBuilder == null) {
            myDocumentBuilder = myDocumentBuilderFactory.newDocumentBuilder();
        }
        Document newDocument = myDocumentBuilder.newDocument();

        //
        myTransformer.transform(new DOMSource(xmlSource), new DOMResult(newDocument));

        //
        return newDocument;
    }


    /**
     * Return the xsl sheet corresponding to the name, dictionary name and volumename
     *
     * @param dictionaryName
     * @param volumeName
     * @param name
     * @return XslSheet
     * @throws fr.imag.clips.papillon.business.PapillonBusinessException
     *
     */
    public static XslSheet getXslSheet(String dictionaryName, String volumeName, String name)
            throws PapillonBusinessException {

        XslSheet theXsl = null;

        // find specific xsl
        theXsl = XslSheetFactory.getXslSheet(dictionaryName, volumeName, name);

        // find specific xsl
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet(dictionaryName, "", name);
        }

        // find specific xsl
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet("", "", name);
        }

        // find defaut
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet(dictionaryName, volumeName, "");
        }

        // find defaut
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet(dictionaryName, "", "");
        }

        // find defaut
        if (theXsl == null) {
            theXsl = XslSheetFactory.getXslSheet("", "", "");
        }

        //
        return theXsl;
    }


}
