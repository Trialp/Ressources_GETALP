package getalp.wsd.ufsac.corpus.streaming.modifier;

import getalp.wsd.ufsac.corpus.*;
import getalp.wsd.ufsac.corpus.streaming.reader.StreamingCorpusReader;
import getalp.wsd.ufsac.corpus.streaming.writer.StreamingCorpusWriter;

public class StreamingCorpusModifier extends StreamingCorpusReader
{
    private StreamingCorpusWriter out = new StreamingCorpusWriter();

    public void load(String inputPath, String outputPath)
    {
        out.open(outputPath);
        super.load(inputPath);
    }
    
    @Override
    public void readBeginCorpus(Corpus corpus)
    {
        out.writeBeginCorpus(corpus);
    }

    @Override
    public void readBeginDocument(Document document)
    {
        out.writeBeginDocument(document);
    }

    @Override
    public void readBeginParagraph(Paragraph paragraph)
    {
        out.writeBeginParagraph(paragraph);
    }

    @Override
    public void readBeginSentence(Sentence sentence)
    {
        out.writeBeginSentence(sentence);
    }

    @Override
    public void readWord(Word word)
    {
        out.writeWord(word);
    }

    @Override
    public void readEndSentence()
    {
        out.writeEndSentence();
    }

    @Override
    public void readEndParagraph()
    {
        out.writeEndParagraph();
    }

    @Override
    public void readEndDocument()
    {
        out.writeEndDocument();
    }

    @Override
    public void readEndCorpus()
    {
        out.writeEndCorpus();
    }
}
