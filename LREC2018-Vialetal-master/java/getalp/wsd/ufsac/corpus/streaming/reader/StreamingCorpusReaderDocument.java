package getalp.wsd.ufsac.corpus.streaming.reader;

import getalp.wsd.ufsac.corpus.*;

public class StreamingCorpusReaderDocument
{
	public void readDocument(Document document)
	{
	    
	}

    public void load(String path)
    {
        reader.load(path);
    }
 
    private StreamingCorpusReader reader = new StreamingCorpusReader()
    {
        private Document currentDocument;

        private Paragraph currentParagraph;
        
        private Sentence currentSentence;
        
        @Override
        public void readBeginDocument(Document document)
        {
            currentDocument = document;
        }

        @Override
        public void readBeginParagraph(Paragraph paragraph)
        {
            currentParagraph = paragraph;
            currentParagraph.setParentDocument(currentDocument);
        }

        @Override
        public void readBeginSentence(Sentence sentence)
        {
            currentSentence = sentence;
            currentSentence.setParentParagraph(currentParagraph);
        }

        @Override
        public void readWord(Word word)
        {
            Word currentWord = word;
            currentWord.setParentSentence(currentSentence);
        }

        @Override
        public void readEndDocument()
        {
            readDocument(currentDocument);
        }
    };
}
