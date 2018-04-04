package getalp.wsd.ufsac.corpus.streaming.writer;

import getalp.wsd.ufsac.corpus.Sentence;
import getalp.wsd.ufsac.corpus.Word;

public class StreamingCorpusWriterSentence
{
    public StreamingCorpusWriterSentence()
    {

    }
    
    public void open(String path)
    {
        out.open(path);
        out.writeBeginCorpus();
        out.writeBeginDocument();
        out.writeBeginParagraph();
    }
    
    public void writeSentence(Sentence sentence)
    {
        out.writeBeginSentence(sentence);
        for (Word word : sentence.getWords())
        {
            out.writeWord(word);
        }
        out.writeEndSentence();
    }
    
    public void close()
    {
        out.writeEndParagraph();
        out.writeEndDocument();
        out.writeEndCorpus();
        out.close();
    }
    
    private StreamingCorpusWriter out = new StreamingCorpusWriter();
}
