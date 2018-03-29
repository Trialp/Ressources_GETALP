package fr.imag.clips.papillon.presentation;

import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.dictionary.AvailableLanguages;
import fr.imag.clips.papillon.business.dictionary.HeadwordListQuery;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.locales.LanguageIndexKeys;
import fr.imag.clips.papillon.business.locales.Languages;
import fr.imag.clips.papillon.presentation.xhtml.orig.DictionaryIndexXHTML;
import org.enhydra.xml.xhtml.dom.*;
import org.w3c.dom.Node;

import java.sql.SQLException;
import java.text.MessageFormat;
import java.util.Collection;
import java.util.Iterator;

public class DictionaryIndexBrowser extends PapillonBasePO {

    private static final String ENTRY_URL = "Home.po?FACET.0=cdm-headword&OPERATOR.0=2&FACETVALUE.0={0}&SOURCE.0={1}&TARGETS=*ALL*&XSL=Default&LIMIT=5&OFFSET=&action=lookup";
    private static final String KEY_INDEX_URL = "DictionaryIndexBrowser.po?SAMPLESIZE=0&KEY={0}&LANGUAGE={1}";
    private static final String LANG_INDEX_URL = "DictionaryIndexBrowser.po?LANGUAGE={0}";

    protected boolean loggedInUserRequired() {
        return false;
    }

    protected boolean userMayUseThisPO() {
        return true;
    }


    protected int getCurrentSection() {
        return NO_SECTION;
    }

    private static class Params {
        protected String vol;
        protected String dict;
        protected String lang;
        protected String key;
        protected int sampleSize;
        protected int nbChars; // Number of chars to be displayed for each volumes
    }

    public Node getContent()
            throws Exception {
        // Retrieve parameters
        Params params = new Params();

        params.vol = myGetParameter("VOLUME");
        if (params.vol != null && VolumesFactory.getVolumeByName(params.vol).getSourceLanguage().equals("axi")) {
            params.vol = null;
        }
        params.dict = myGetParameter("DICT");
        params.lang = myGetParameter("LANGUAGE");
        params.key = myGetParameter("KEY");
        String sampleSize = myGetParameter("SAMPLESIZE");
        params.sampleSize = 20;
        if (null != sampleSize) {
            try {
                params.sampleSize = Integer.parseInt(sampleSize);
            } catch (NumberFormatException e) {
                // Do nothing
            }
        }
        params.nbChars = 0;
        if ((null == params.lang || "".equals(params.lang)) &&
                ((null == params.vol) || "".equals(params.vol)) &&
                ((null == params.key) || "".equals(params.key))) {
            params.nbChars = 5;
        }

        DictionaryIndexXHTML content = (DictionaryIndexXHTML) MultilingualXHtmlTemplateFactory.createTemplate("DictionaryIndexXHTML", this.getComms(), this.getSessionData());

        XHTMLDivElement indexList = content.getElementIndexList();
        XHTMLDivElement indexKey = content.getElementIndexKey();
        XHTMLSpanElement keyTitle = content.getElementKeyTitle();
        XHTMLAnchorElement sectionAnchor = content.getElementSectionAnchor();
        XHTMLDivElement languageDiv = content.getElementLanguageDiv();
        XHTMLSpanElement languageName = content.getElementLanguageName();
        XHTMLDivElement languageTitle = content.getElementLanguageTitle();
        XHTMLAnchorElement anchor = content.getElementHeadwordAnchor();
        XHTMLSpanElement hwText = content.getElementHeadwordText();
        XHTMLLIElement li = content.getElementHeadwordListItem();
        XHTMLUListElement ul = content.getElementHeadwordList();
        XHTMLDivElement keyToc = content.getElementKeyToc();
        XHTMLSpanElement tocSeparator = content.getElementTocSeparator();
        XHTMLAnchorElement tocAnchor = content.getElementTocAnchor();
        XHTMLSpanElement tocAnchorText = content.getElementTocAnchorText();
        XHTMLAnchorElement tocAnchorLast = content.getElementTocAnchorLast();
        XHTMLDivElement keyTocTop = content.getElementKeyTocTop();
        XHTMLDivElement keyTocBottom = content.getElementKeyTocBottom();

        anchor.removeAttribute("id");
        sectionAnchor.removeAttribute("id");
        hwText.removeAttribute("id");
        li.removeAttribute("id");
        ul.removeAttribute("id");
        languageDiv.removeAttribute("id");
        languageName.removeAttribute("id");
        languageTitle.removeAttribute("id");
        indexList.removeAttribute("id");
        indexKey.removeAttribute("id");
        keyTitle.removeAttribute("id");
        keyToc.removeAttribute("id");
        tocSeparator.removeAttribute("id");
        tocAnchor.removeAttribute("id");
        tocAnchorText.removeAttribute("id");
        keyTocTop.removeAttribute("id");
        keyTocBottom.removeAttribute("id");
        tocAnchorLast.removeAttribute("id");


        indexKey.getParentNode().removeChild(indexKey);
        keyToc.getParentNode().removeChild(keyToc);
        tocSeparator.getParentNode().removeChild(tocSeparator);
        tocAnchor.getParentNode().removeChild(tocAnchor);
        tocAnchorLast.getParentNode().removeChild(tocAnchorLast);
        li.getParentNode().removeChild(li);

        fillInPage(content, params);

        return content.getElementDictionaryIndexContent();
    }

    private void fillInPage(DictionaryIndexXHTML content, Params params) throws SQLException, PapillonBusinessException {
        XHTMLDivElement languageDiv = content.getElementLanguageDiv();
        Node lgParent = languageDiv.getParentNode();
        lgParent.removeChild(languageDiv);

        if (null != params.vol && !"".equals(params.vol)) {
            Volume vol = VolumesFactory.getVolumeByName(params.vol);
            Node index = getIndexForLanguage(content, params, vol.getSourceLanguage());
            if (null != index) lgParent.appendChild(index);
        } else if (null != params.dict && !"".equals(params.dict)) {
            Collection vols;
            vols = VolumesFactory.getVolumesArray(params.dict, params.lang, null);
            Iterator it = vols.iterator();
            while (it.hasNext()) {
                Volume vol = (Volume) it.next();
                if (!vol.getSourceLanguage().equals("axi")) {
                    Node index = getIndexForLanguage(content, params, vol.getSourceLanguage());
                    if (null != index) lgParent.appendChild(index);
                }
            }
        } else if (null != params.lang && !"".equals(params.lang)) {
            Node index = getIndexForLanguage(content, params, params.lang);
            if (null != index) lgParent.appendChild(index);
        } else {
            Collection langs = AvailableLanguages.getSourceLanguagesArray();
            Iterator it = langs.iterator();
            while (it.hasNext()) {
                String lang = (String) it.next();
                Node index = getIndexForLanguage(content, params, lang);
                if (null != index) lgParent.appendChild(index);
            }
        }
    }


    private Node getToc(DictionaryIndexXHTML content, LanguageIndexKeys indexKeys, int climit) {
        String lang = indexKeys.getMyLanguage();
        XHTMLDivElement keyToc = content.getElementKeyToc();
        XHTMLSpanElement tocSeparator = content.getElementTocSeparator();
        XHTMLAnchorElement tocAnchor = content.getElementTocAnchor();
        XHTMLSpanElement tocAnchorText = content.getElementTocAnchorText();
        XHTMLAnchorElement tocAnchorLast = content.getElementTocAnchorLast();

        String others = tocAnchorText.getNodeValue();
        String[] keys = indexKeys.getKeys();
        int i;
        for ( i = 0; i < keys.length && i < climit; i++) {
            String key = keys[i];
            content.setTextTocAnchorText(key);
            tocAnchor.setAttribute("href", "#" + lang + "_" + key);
            keyToc.appendChild(content.importNode(tocAnchor.cloneNode(true), true));
            keyToc.appendChild(content.importNode(tocSeparator.cloneNode(true), true));
        }
        if (i == keys.length) {
            tocAnchorLast.setAttribute("href", "#" + lang + "_NA");
            keyToc.appendChild(content.importNode(tocAnchorLast.cloneNode(true), true));
        } else {
            tocAnchorLast.setAttribute("href", MessageFormat.format(LANG_INDEX_URL, new String[]{lang}));
            content.setTextLastTocAnchorText("...");
            keyToc.appendChild(content.importNode(tocAnchorLast.cloneNode(true), true));
        }

        Node result = content.importNode(keyToc.cloneNode(true), true);

        while (keyToc.hasChildNodes()) {
            keyToc.removeChild(keyToc.getFirstChild());
        }

        return result;
    }


    public Node getIndexForLanguage(DictionaryIndexXHTML content, Params params, String language) throws SQLException, PapillonBusinessException {
        content.setTextLanguageName(Languages.localizeName((String) this.getUserAcceptLanguages().get(0), language));
        Node currentCell = content.getElementIndexList();
        Node result = null;
        XHTMLDivElement keyTocTop = content.getElementKeyTocTop();
        XHTMLDivElement keyTocBottom = content.getElementKeyTocBottom();

        boolean hasAnIndex = false;
        if (null == params.key || params.key.equals("")) {
            LanguageIndexKeys indexKeys = new LanguageIndexKeys(language);

            String[] keys = indexKeys.getKeys();
            int climit = (params.nbChars == 0) ? keys.length + 1 : params.nbChars;

            Node toc = getToc(content, indexKeys, climit);
            keyTocTop.appendChild(content.importNode(toc.cloneNode(true), true));
            keyTocBottom.appendChild(toc);

            int i = 0;
            for (i = 0; i < keys.length && i < climit; i++) {
                String key = keys[i];
                // size is the number of entries for this index key.
                Node indexForKey = getIndexDataForLanguage(content, language, key, params.sampleSize);
                if (null != indexForKey) {
                    hasAnIndex = true;
                    currentCell.appendChild(indexForKey);
                }
            }
            if (i == keys.length) {
                Node indexForKey = getIndexDataForLanguage(content, language, "*", params.sampleSize);
                if (null != indexForKey) {
                    hasAnIndex = true;
                    currentCell.appendChild(indexForKey);
                }
            } else {
                Node moreChars = createEmptySection(content, language);
                if (null != moreChars) {
                    hasAnIndex = true;
                    currentCell.appendChild(moreChars);
                }
            }
        } else {
            Node indexForKey = getIndexDataForLanguage(content, language, params.key, params.sampleSize);
            if (null != indexForKey) {
                hasAnIndex = true;
                currentCell.appendChild(indexForKey);
            }
        }

        if (hasAnIndex) {
            result = content.importNode(content.getElementLanguageDiv().cloneNode(true), true);
            // Reinitialize language Section
            while (currentCell.hasChildNodes()) {
                currentCell.removeChild(currentCell.getFirstChild());
            }
        }
        while (keyTocTop.hasChildNodes()) {
            keyTocTop.removeChild(keyTocTop.getFirstChild());
        }
        while (keyTocBottom.hasChildNodes()) {
            keyTocBottom.removeChild(keyTocBottom.getFirstChild());
        }
        return result;
    }

    public Node getIndexDataForLanguage(DictionaryIndexXHTML content, String language, String key, int limit) throws SQLException, PapillonBusinessException {
        Collection headwords = HeadwordListQuery.getHeadwordListForLanguageIgnoringAccents(language, key, limit == 0 ? 0 : limit + 1);
        return createIndexDataForCollection(content, language, key, headwords, limit);
    }

    public Node createIndexDataForCollection(DictionaryIndexXHTML content, String language, String key, Collection headwords, int limit)
            throws SQLException {
        XHTMLAnchorElement anchor = content.getElementHeadwordAnchor();
        XHTMLUListElement ul = content.getElementHeadwordList();
        XHTMLLIElement li = content.getElementHeadwordListItem();
        Node indexKey = content.getElementIndexKey();
        XHTMLAnchorElement sectionAnchor = content.getElementSectionAnchor();

        Node newSection = null;
        if (headwords.size() != 0) {
            // Set the key
            content.setTextKeyTitle(key.toUpperCase());
            sectionAnchor.setAttribute("name", language + "_" + key);
            int nb = 0;
            for (Iterator it = headwords.iterator(); it.hasNext();) {
                nb++;
                String hw = (String) it.next();
                if (limit > 0 & nb > limit) {
                    content.setTextHeadwordText("...");
                    anchor.setHref(MessageFormat.format(KEY_INDEX_URL, new String[]{myUrlEncode(key), language}));
                    // li.setAttribute("class", "alt_li" + nb%2);
                    ul.appendChild(content.importNode(li.cloneNode(true), true));
                } else {
                    content.setTextHeadwordText(hw);
                    anchor.setHref(MessageFormat.format(ENTRY_URL, new String[]{myUrlEncode(hw), language}));
                    // li.setAttribute("class", "alt_li" + nb%2);
                    ul.appendChild(content.importNode(li.cloneNode(true), true));
                }
            }

            newSection = content.importNode(indexKey.cloneNode(true), true);
            // Clean up the cell
            while (ul.hasChildNodes()) {
                ul.removeChild(ul.getFirstChild());
            }
        }
        return newSection;
    }


    public Node createEmptySection(DictionaryIndexXHTML content, String language)
            throws SQLException {
        XHTMLUListElement ul = content.getElementHeadwordList();
        Node indexKey = content.getElementIndexKey();
        XHTMLAnchorElement sectionAnchor = content.getElementSectionAnchor();

        Node newSection = null;

        content.setTextKeyTitle("...");
        sectionAnchor.setHref(MessageFormat.format(LANG_INDEX_URL, new String[]{language}));


        newSection = content.importNode(indexKey.cloneNode(true), true);
        // Clean up the cell
        while (ul.hasChildNodes()) {
            ul.removeChild(ul.getFirstChild());
        }

        return newSection;
    }

}
