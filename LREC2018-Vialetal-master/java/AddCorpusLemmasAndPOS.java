import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.HasWord;
import java.util.ArrayList;
import java.util.List;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import getalp.wsd.ufsac.corpus.Sentence;
import getalp.wsd.ufsac.corpus.Word;
import getalp.wsd.ufsac.corpus.streaming.modifier.StreamingCorpusModifierSentence;
import getalp.wsd.ufsac.utils.POSHelper;
import getalp.wsd.ufsac.utils.Wrapper;
import getalp.wsd.ufsac.wordnet.WordnetHelper;

public class AddCorpusLemmasAndPOS
{
    public static void main(String[] args)
    {
    	String inputPath = "data/corpus/dso.xml";
    	String outputPath = "data/corpus/omsti_plus_lemma.xml";
    	int wordnetVersion = 30;
    	
        doIt(inputPath, outputPath, wordnetVersion);
    }
    
    public static void doIt(String inPath, String outPath, int wordnetVersion)
    {
        Wrapper<Integer> countTotal = new Wrapper<Integer>(0);
        Wrapper<Integer> countFailed = new Wrapper<Integer>(0);
        
        MaxentTagger tagger = new MaxentTagger("data/stanford/model/english.tagger");        
        
        StreamingCorpusModifierSentence inout = new StreamingCorpusModifierSentence()
        {
            public void modifySentence(Sentence sentence)
            {
                List<TaggedWord> stanfordSentence = tagger.tagSentence(toStanfordSentence(sentence));
                if (stanfordSentence.size() != sentence.getWords().size()) throw new RuntimeException();
                for (int i = 0 ; i < stanfordSentence.size() ; i++)
                {
                    Word word = sentence.getWords().get(i);
                    TaggedWord stanfordWord = stanfordSentence.get(i);
                    String stanfordPostag = stanfordWord.tag();
                    if (word.hasAnnotation("pos"))
                    {
                        String stanfordPostagProcessed = POSHelper.processPOS(stanfordPostag);
                        String inplacePostagProcessed = POSHelper.processPOS(word.getAnnotationValue("pos"));
                        if (stanfordPostagProcessed.equals(inplacePostagProcessed))
                        {
                            word.setAnnotation("pos", stanfordPostag);
                        }
                    }
                    else
                    {
                        word.setAnnotation("pos", stanfordPostag);
                    }
                    if (!word.hasAnnotation("lemma"))
                    {
                        String pos = POSHelper.processPOS(word.getAnnotationValue("pos"));
                        if (!pos.equals("x"))
                        {
                            countTotal.obj++;
                            String surfaceForm = word.getValue();
                            String lemma = WordnetHelper.wn(wordnetVersion).morphy(surfaceForm, pos);
                            if (WordnetHelper.wn(wordnetVersion).isWordKeyExists(lemma + "%" + pos))
                            {
                                word.setAnnotation("lemma", lemma);
                            }
                            else
                            {
                                //System.out.println("Failed : " + surfaceForm);
                                countFailed.obj++;
                            }
                        }
                    }
                }
            }
        };
        
        inout.load(inPath, outPath);
        
        System.out.println("Info : " + countTotal.obj + " total missing tags");
        System.out.println("Info : " + (countTotal.obj - countFailed.obj) + " suceed to tag");
        System.out.println("Info : " + countFailed.obj + " failed to tag");
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
