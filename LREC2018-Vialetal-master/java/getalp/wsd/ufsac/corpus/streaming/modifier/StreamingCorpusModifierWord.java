package getalp.wsd.ufsac.corpus.streaming.modifier;

import getalp.wsd.ufsac.corpus.*;

public class StreamingCorpusModifierWord 
{
    public void modifyWord(Word word)
    {
        
    }
    
    public void load(String inputPath, String outputPath)
    {
        modifier.load(inputPath, outputPath);
    }
    
    private StreamingCorpusModifier modifier = new StreamingCorpusModifier()
    {
        @Override
        public void readWord(Word word)
        {
            modifyWord(word);
            super.readWord(word);
        }
    };
}
