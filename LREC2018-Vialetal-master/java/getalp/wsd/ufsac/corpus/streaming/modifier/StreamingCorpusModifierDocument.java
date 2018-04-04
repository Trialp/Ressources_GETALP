package getalp.wsd.ufsac.corpus.streaming.modifier;

import getalp.wsd.ufsac.corpus.*;

public class StreamingCorpusModifierDocument
{
    public void modifyDocument(Document document)
    {
        
    }
    
    public void load(String inputPath, String outputPath)
    {
        modifier.load(inputPath, outputPath);
    }
    
    private StreamingCorpusModifier modifier = new StreamingCorpusModifier()
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
            word.setParentSentence(currentSentence);
        }

        @Override
        public void readEndSentence()
        {

        }

        @Override
        public void readEndParagraph()
        {

        }

        @Override
        public void readEndDocument()
        {
            modifyDocument(currentDocument);
            super.readBeginDocument(currentDocument);
            for (Paragraph paragraph : currentDocument.getParagraphs())
            {
                super.readBeginParagraph(paragraph);
                for (Sentence sentence : paragraph.getSentences())
                {
                    super.readBeginSentence(sentence);
                    for (Word word : sentence.getWords())
                    {
                        super.readWord(word);
                    }
                    super.readEndSentence();
                }
                super.readEndParagraph();
            }
            super.readEndDocument();
        }
    };
}
