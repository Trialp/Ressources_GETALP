package getalp.wsd.ufsac.corpus.streaming.reader;

import getalp.wsd.ufsac.corpus.Word;

public class StreamingCorpusReaderWord
{
    public void readWord(Word word)
    {

    }
    
    public void load(String path)
    {
        reader.load(path);
    }
    
    private StreamingCorpusReader reader = new StreamingCorpusReader()
    {
        @Override
        public void readWord(Word word)
        {
            StreamingCorpusReaderWord.this.readWord(word);
        }
    };
}
