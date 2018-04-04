package getalp.wsd.ufsac.corpus.streaming.reader;

import getalp.wsd.ufsac.corpus.Paragraph;
import getalp.wsd.ufsac.corpus.Sentence;
import getalp.wsd.ufsac.corpus.Word;

public class StreamingCorpusReaderParagraph
{
	public void readParagraph(Paragraph paragraph)
	{
	    
	}

    public void load(String path)
    {
        reader.load(path);
    }
 
    private StreamingCorpusReader reader = new StreamingCorpusReader()
    {
        private Paragraph currentParagraph;
        
        private Sentence currentSentence;
        
        @Override
        public void readBeginParagraph(Paragraph paragraph)
        {
            currentParagraph = paragraph;
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
        public void readEndParagraph()
        {
            StreamingCorpusReaderParagraph.this.readParagraph(currentParagraph);
        }
    };
}
