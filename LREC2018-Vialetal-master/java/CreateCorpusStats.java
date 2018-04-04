
import getalp.wsd.ufsac.corpus.Sentence;
import getalp.wsd.ufsac.corpus.Word;
import getalp.wsd.ufsac.corpus.streaming.reader.StreamingCorpusReader;
import getalp.wsd.ufsac.utils.POSHelper;
import getalp.wsd.ufsac.utils.Wrapper;

public class CreateCorpusStats
{
    public static void main(String[] args) throws Exception
    {
        System.out.println("\\begin{tabular}{l|ccccccc}");
        System.out.println("\\toprule");
        System.out.println("\\multirow{2}{*}{Ressource} & \\multirow{2}{*}{Phrases} & \\multicolumn{6}{c}{Mots} \\\\");
        System.out.println(" & & total & annotés & noms & verbes & adj. & adv. \\\\");
        System.out.println("\\midrule");           

        //String[][] corpuses = new String[][]{{"semcor"}, {"dso"}, {"wngt"}, {"masc"}, {"omsti_part0", "omsti_part1", "omsti_part2", "omsti_part3"}, 
        //    {"ontonotes"}, {"sem7"}, {"sem13"}, {"semeval15task13"}, {"senseval2"}, {"senseval3task1"}};

        String[][] corpuses = new String[][]{{"semeval2007task17"}};
                
    	int i = 0;
    	    
	    for (String[] corpus : corpuses)
	    {
	        String resourceName = corpus[0];
	        Wrapper<Integer> sentenceCount = new Wrapper<>(0);
            Wrapper<Integer> wordCount = new Wrapper<>(0);
            Wrapper<Integer> annotatedWordCount = new Wrapper<>(0);
            Wrapper<Integer> nounWordCount = new Wrapper<>(0);
            Wrapper<Integer> verbWordCount = new Wrapper<>(0);
            Wrapper<Integer> adjectiveWordCount = new Wrapper<>(0);
            Wrapper<Integer> adverbWordCount = new Wrapper<>(0);
	        StreamingCorpusReader reader = new StreamingCorpusReader()
            {
                public void readBeginSentence(Sentence sentence)
                {
                    sentenceCount.obj++;
                }
                
                public void readWord(Word word)
                {
                    wordCount.obj++;
                    if (word.hasAnnotation("wn30_key"))
                    {
                        annotatedWordCount.obj++;
                        if (word.hasAnnotation("pos"))
                        {
                            if (POSHelper.processPOS(word.getAnnotationValue("pos")).equals("n"))
                            {
                                nounWordCount.obj++;
                            }
                            else if (POSHelper.processPOS(word.getAnnotationValue("pos")).equals("v"))
                            {
                                verbWordCount.obj++;
                            }
                            else if (POSHelper.processPOS(word.getAnnotationValue("pos")).equals("a"))
                            {
                                adjectiveWordCount.obj++;
                            }
                            else if (POSHelper.processPOS(word.getAnnotationValue("pos")).equals("r"))
                            {
                                adverbWordCount.obj++;
                            }
                        }
                    }
                }
            };
            for (String corpusPart : corpus)
            {
                reader.load("data/corpus/" + corpusPart + ".xml");
            }
            
            if ((i % 2) == 1)
            {
                System.out.print("\\rowcolor{gray!10} ");
            }
            i++;
	        
	        System.out.println(resourceName.replaceAll("_", "") + " & " + sentenceCount.obj + " & " + wordCount.obj + " & " + annotatedWordCount.obj + " & " + 
	        nounWordCount.obj + " & " + verbWordCount.obj + " & " + adjectiveWordCount.obj + " & " + adverbWordCount.obj + " \\\\");
	    }
	    System.out.println("\\bottomrule");
        System.out.println("\\end{tabular}");
    }
}
