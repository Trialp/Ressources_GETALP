package getalp.wsd.ufsac.corpus.streaming.modifier;

import getalp.wsd.ufsac.corpus.*;

public class StreamingCorpusModifierSentence
{
    public void modifySentence(Sentence sentence)
    {
        
    }
    
    public void load(String inputPath, String outputPath)
    {
        modifier.load(inputPath, outputPath);
    }
    
    private StreamingCorpusModifier modifier = new StreamingCorpusModifier()
    {
        private Sentence currentSentence;
        
        @Override
        public void readBeginSentence(Sentence sentence)
        {
            currentSentence = sentence;
        }

        @Override
        public void readWord(Word word)
        {
            word.setParentSentence(currentSentence);
        }

        @Override
        public void readEndSentence()
        {
            modifySentence(currentSentence);
            super.readBeginSentence(currentSentence);
            for (Word word : currentSentence.getWords())
            {
                super.readWord(word);
            }
            super.readEndSentence();
        }
    };
}
