package getalp.wsd.ufsac.corpus.modifier;

import getalp.wsd.ufsac.corpus.Word;
import getalp.wsd.ufsac.corpus.streaming.modifier.StreamingCorpusModifierWord;
import getalp.wsd.ufsac.utils.POSHelper;
import getalp.wsd.ufsac.wordnet.WordnetHelper;

public class CorpusLemmaTagger extends StreamingCorpusModifierWord
{
    private WordnetHelper wn;
    
    public CorpusLemmaTagger(WordnetHelper wn)
    {
        this.wn = wn;
    }
    
    public void modifyWord(Word word)
    {
        if (!word.hasAnnotation("lemma"))
        {
            if (word.hasAnnotation("pos"))
            {
                String pos = POSHelper.processPOS(word.getAnnotationValue("pos"));
                word.setAnnotation("lemma", wn.morphy(word.getValue(), pos));
            }
            else
            {
                word.setAnnotation("lemma", wn.morphy(word.getValue()));
            }
        }
    }
}
