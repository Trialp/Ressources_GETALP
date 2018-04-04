package getalp.wsd.ufsac.corpus.streaming.writer;

import getalp.wsd.ufsac.corpus.Word;

public class StreamingCorpusWriterWord
{
    public StreamingCorpusWriterWord()
    {

    }
    
    public void open(String path)
    {
        out.open(path);
        out.writeBeginCorpus();
        out.writeBeginDocument();
        out.writeBeginParagraph();
        out.writeBeginSentence();
    }
    
    public void writeWord(Word word)
    {
        out.writeWord(word);
    }
    
    public void close()
    {
        out.writeEndSentence();
        out.writeEndParagraph();
        out.writeEndDocument();
        out.writeEndCorpus();
        out.close();
    }
    
    private StreamingCorpusWriter out = new StreamingCorpusWriter();
}
