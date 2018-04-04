package getalp.wsd.ufsac.corpus.streaming.writer;

import getalp.wsd.ufsac.corpus.Corpus;
import getalp.wsd.ufsac.corpus.Document;
import getalp.wsd.ufsac.corpus.Paragraph;
import getalp.wsd.ufsac.corpus.Sentence;
import getalp.wsd.ufsac.corpus.Word;

public class StreamingCorpusFullWriter
{
    public void open(String path)
    {
        out.open(path);
    }
    
    public void writeCorpus(Corpus corpus)
    {
        out.writeBeginCorpus(corpus);
        for (Document document : corpus.getDocuments())
        {
            out.writeBeginDocument(document);
            for (Paragraph paragraph : document.getParagraphs())
            {
                out.writeBeginParagraph(paragraph);
                for (Sentence sentence : paragraph.getSentences())
                {
                    out.writeBeginSentence(sentence);
                    for (Word word : sentence.getWords())
                    {
                        out.writeWord(word);
                    }
                    out.writeEndSentence();
                }
                out.writeEndParagraph();
            }
            out.writeEndDocument();
        }
        out.writeEndCorpus();
    }
    
    public void close()
    {
        out.close();
    }
    
    private StreamingCorpusWriter out = new StreamingCorpusWriter();
}
