package getalp.wsd.ufsac.corpus.modifier;

import getalp.wsd.ufsac.corpus.Word;
import getalp.wsd.ufsac.corpus.streaming.reader.StreamingCorpusReaderWord;
import getalp.wsd.ufsac.utils.POSHelper;
import getalp.wsd.ufsac.wordnet.WordnetHelper;

public class CorpusLemmaPOSTagChecker extends StreamingCorpusReaderWord
{
    private WordnetHelper wn;
    
    public CorpusLemmaPOSTagChecker(WordnetHelper wn)
    {
        this.wn = wn;
    }
    
    @Override
    public void readWord(Word w)
    {
        String lemma = w.getAnnotationValue("lemma");
        String pos = w.getAnnotationValue("pos");
        pos = POSHelper.processPOS(pos);
        if (!lemma.isEmpty() && !pos.equals("x"))
        {
            String wordKey = lemma + "%" + pos;
            if (!wn.isWordKeyExists(wordKey))
            {
                System.out.println("Warning : " + wordKey + " is not in WN" + wn.getVersion() + " vocabulary");
            }
        }
    }
}
