package getalp.wsd.conversion;

import org.apache.commons.lang3.StringUtils;
import org.xml.sax.*;
import org.xml.sax.helpers.XMLReaderFactory;
import getalp.wsd.corpus.Document;
import getalp.wsd.corpus.Paragraph;
import getalp.wsd.corpus.Sentence;
import getalp.wsd.corpus.Word;
import getalp.wsd.corpus.xml.writer.XMLCorpusSequentialWriterDocument;
import getalp.wsd.xml.SAXEntityResolverIgnoringDTD;
import org.xml.sax.helpers.DefaultHandler;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Semeval2007Task7Converter extends DefaultHandler
{
	private XMLCorpusSequentialWriterDocument out;

	private Document currentDocument;

	private Paragraph currentParagraph;

	private Sentence currentSentence;
	
	private Word currentWord;

	private boolean saveCharacters;

	private String currentCharacters;

	private String currentPos;

	private String currentLemma;
	
	private String currentWordId;

    private Map<String, String> sensesById;
    
	@Override
	public void startElement(String uri, String localName, String qName, Attributes atts) throws SAXException
	{
		if (localName.equals("text"))
		{
			currentDocument = new Document();
			currentDocument.setAnnotation("id", atts.getValue("id"));
			currentParagraph = new Paragraph(currentDocument);
		}
		else if (localName.equals("sentence"))
		{
			currentSentence = new Sentence(currentParagraph);
			currentSentence.setAnnotation("id", atts.getValue("id"));
			saveCharacters = true;
			currentCharacters = "";
		}
		else if (localName.equals("instance"))
		{
	        addNonAnnotatedWords();
			saveCharacters = true;
			currentCharacters = "";
			currentPos = atts.getValue("pos");
			currentLemma = atts.getValue("lemma");
			currentLemma = currentLemma.toLowerCase();
			currentWordId = atts.getValue("id");
		}
	}

	@Override
	public void endElement(String uri, String localName, String qName) throws SAXException
	{
	    if (localName.equals("text"))
        {
	        out.writeDocument(currentDocument);
        }
        else if (localName.equals("sentence"))
        {
            addNonAnnotatedWords();
        }
        else if (localName.equals("instance"))
		{
			currentWord = new Word(currentSentence);
			currentWord.setValue(currentCharacters);
			currentWord.setAnnotation("lemma", currentLemma);
			currentWord.setAnnotation("pos", currentPos);
			currentWord.setAnnotation("id", currentWordId);
			currentWord.setAnnotation("wn21_key", sensesById.get(currentWordId));
			saveCharacters = true;
			currentCharacters = "";
		}
	}
	
	private void addNonAnnotatedWords()
	{
	    String[] words = currentCharacters.split("\\s+");
	    for (String surfaceForm : words)
	    {
	        if (surfaceForm.isEmpty()) continue;
	        Word word = new Word(currentSentence);
	        word.setValue(surfaceForm);
	    }
	    saveCharacters = false;
	    currentCharacters = "";
	}

	@Override
	public void characters(char[] ch, int start, int length) throws SAXException
	{
		if (saveCharacters)
		{
			currentCharacters += new String(ch, start, length);
		}
	}

    public void convert(String inpath, String outpath)
	{
        out = new XMLCorpusSequentialWriterDocument(outpath);
	    loadSenses(inpath + "/key/dataset21.test.key");
	    loadCorpus(inpath + "/test/eng-coarse-all-words.xml");
	}
	
	private void loadSenses(String path)
	{
	    sensesById = new HashMap<>();
        try
        {
            Scanner sc = new Scanner(new File(path));
            while (sc.hasNextLine())
            {
                String line = sc.nextLine();
                String[] tokens = line.split("\\s+");
                String id = tokens[1];
                List<String> senses = new ArrayList<>();
                for (int i = 2 ; i < tokens.length - 2 ; i++)
                {
                    senses.add(tokens[i].toLowerCase());
                }
                String sense = StringUtils.join(senses, ";");
                sensesById.put(id, sense);
            }
            sc.close();
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }

	private void loadCorpus(String path)
	{
        try
        {
            XMLReader saxReader = XMLReaderFactory.createXMLReader();
            saxReader.setContentHandler(this);
            saxReader.setEntityResolver(new SAXEntityResolverIgnoringDTD());
            out.writeHeader();
            saxReader.parse(path);
            out.writeFooter();
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
	}
}
