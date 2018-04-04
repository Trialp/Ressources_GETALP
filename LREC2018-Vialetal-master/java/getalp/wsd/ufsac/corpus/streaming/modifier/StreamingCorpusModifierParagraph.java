package getalp.wsd.ufsac.corpus.streaming.modifier;

import getalp.wsd.ufsac.corpus.*;

public class StreamingCorpusModifierParagraph
{
    public void modifyParagraph(Paragraph paragraph)
    {
        
    }
    
    public void load(String inputPath, String outputPath)
    {
        modifier.load(inputPath, outputPath);
    }
    
    private StreamingCorpusModifier modifier = new StreamingCorpusModifier()
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
            word.setParentSentence(currentSentence);
        }

        @Override
        public void readEndSentence()
        {
            
        }

        @Override
        public void readEndParagraph()
        {
            modifyParagraph(currentParagraph);
            super.readBeginParagraph(currentParagraph);
            for (Sentence sentence : currentParagraph.getSentences())
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
    };
}
