package getalp.wsd.ufsac.corpus.modifier;

import java.util.ArrayList;
import java.util.List;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import getalp.wsd.ufsac.corpus.Sentence;
import getalp.wsd.ufsac.corpus.Word;
import getalp.wsd.ufsac.corpus.streaming.modifier.StreamingCorpusModifierSentence;

public class CorpusPOSTagger extends StreamingCorpusModifierSentence
{
    private MaxentTagger tagger;  
    
    public CorpusPOSTagger()
    {
        tagger = new MaxentTagger("data/stanford/model/english.tagger");
    }
    
    public void modifySentence(Sentence sentence)
    {
        List<TaggedWord> stanfordSentence = tagger.tagSentence(toStanfordSentence(sentence));
        assert(stanfordSentence.size() != sentence.getWords().size());
        for (int i = 0 ; i < stanfordSentence.size() ; i++)
        {
            Word word = sentence.getWords().get(i);
            if (!word.hasAnnotation("pos"))
            {
                TaggedWord stanfordWord = stanfordSentence.get(i);
                word.setAnnotation("pos", stanfordWord.tag());
            }
        }
    }
    
    private static List<HasWord> toStanfordSentence(Sentence sentence)
    {
        List<HasWord> stanfordSentence = new ArrayList<>();
        for (Word word : sentence.getWords())
        {
            stanfordSentence.add(new edu.stanford.nlp.ling.Word(word.getValue()));
        }
        return stanfordSentence;
    }
}
