import getalp.wsd.ufsac.corpus.Word;
import getalp.wsd.ufsac.corpus.streaming.modifier.StreamingCorpusModifierWord;
import getalp.wsd.ufsac.utils.POSHelper;
import getalp.wsd.ufsac.utils.Wrapper;
import getalp.wsd.ufsac.wordnet.WordnetHelper;

public class AddCorpusLemmas
{
    public static void main(String[] args)
    {
    	String inputPath = "data/corpus/omsti.xml";
    	String outputPath = "data/corpus/omsti_plus_lemma.xml";
    	int wordnetVersion = 30;
    	
        doIt(inputPath, outputPath, wordnetVersion);
    }
    
    public static void doIt(String inPath, String outPath, int wordnetVersion)
    {
        Wrapper<Integer> countTotal = new Wrapper<Integer>(0);
        Wrapper<Integer> countFailed = new Wrapper<Integer>(0);
        
        StreamingCorpusModifierWord inout = new StreamingCorpusModifierWord()
        {
            public void modifyWord(Word word)
            {
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
        };
        
        inout.load(inPath, outPath);
        
        System.out.println("Info : " + countTotal.obj + " total missing tags");
        System.out.println("Info : " + (countTotal.obj - countFailed.obj) + " suceed to tag");
        System.out.println("Info : " + countFailed.obj + " failed to tag");
    }
}
