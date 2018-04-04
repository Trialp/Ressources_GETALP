import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import getalp.wsd.ufsac.corpus.Annotation;
import getalp.wsd.ufsac.corpus.Sentence;
import getalp.wsd.ufsac.corpus.Word;
import getalp.wsd.ufsac.corpus.conversion.*;
import getalp.wsd.ufsac.corpus.streaming.modifier.StreamingCorpusModifierSentence;
import getalp.wsd.ufsac.corpus.streaming.modifier.StreamingCorpusModifierWord;
import getalp.wsd.ufsac.corpus.streaming.reader.StreamingCorpusReaderSentence;
import getalp.wsd.ufsac.corpus.streaming.reader.StreamingCorpusReaderWord;
import getalp.wsd.ufsac.corpus.streaming.writer.StreamingCorpusWriterSentence;
import getalp.wsd.ufsac.utils.Data;
import getalp.wsd.ufsac.utils.File;
import getalp.wsd.ufsac.utils.POSHelper;
import getalp.wsd.ufsac.utils.StdOutStdErr;
import getalp.wsd.ufsac.utils.StringUtils;
import getalp.wsd.ufsac.utils.Wrapper;
import getalp.wsd.ufsac.wordnet.WordnetHelper;
import getalp.wsd.ufsac.wordnet.WordnetMapping;

public class CreateCorpus
{
    public static MaxentTagger tagger = null;

    public static void main(String[] args) throws Exception
    {
        convertCorporaToNewFormat();
        postProcessCorpora();
    }

    public static void convertCorporaToNewFormat()
    {
        convertSemcor();
        convertDSO();
        convertWNGT();
        convertMASC();
        convertOMSTI();
        convertOntonotes();
        convertSenseval2();
        convertSenseval3Task1();
        convertSemeval2007Task7();
        convertSemeval2007Task17();
        convertSemeval2013Task12();
        convertSemeval2015Task13();
    }

    public static void postProcessCorpora()
    {
        String[] corporaPath = { 
                Data.semcorPath, 
                Data.dsoPath, 
                Data.wngtPath, 
                Data.mascPath, 
                Data.omstiPath, 
                Data.ontonotesPath,
                Data.senseval2Path, 
                Data.senseval3task1Path, 
                Data.semeval2007task7Path, 
                Data.semeval2007task17Path, 
                Data.semeval2013task12Path, 
                Data.semeval2015task13Path 
        };

        String[] sentenceBasedCorporaPath = { 
                Data.dsoPath, 
                Data.mascPath, 
                Data.omstiPath,
                Data.ontonotesPath
        };

        Arrays.stream(corporaPath).forEach(corpusPath -> postProcessCorpus(corpusPath));
        Arrays.stream(sentenceBasedCorporaPath).forEach(corpusPath -> postProcessSentenceBasedCorpus(corpusPath));
    }

    public static void convertSemcor()
    {
        System.out.println("Converting corpus " + Data.semcorPath);
        new SemcorConverter().convert("data/corpus/original/semcor", Data.semcorPath, 16);
    }

    public static void convertDSO()
    {
        System.out.println("Converting corpus " + Data.dsoPath);
        new DSOConverter().convert("data/corpus/original/dso/", Data.dsoPath, 16);
    }

    public static void convertWNGT()
    {
        System.out.println("Converting corpus " + Data.wngtPath);
        new WNGTConverter().convert("data/wordnet/30/glosstag", Data.wngtPath, 30);
    }

    public static void convertMASC()
    {
        System.out.println("Converting corpus " + Data.mascPath);
        new MASCConverter().convert("data/corpus/original/google/masc", Data.mascPath, 30);
    }

    public static void convertOMSTI()
    {
        System.out.println("Converting corpus " + Data.omstiPath);
        new OMSTIConverter().convert("data/corpus/original/omsti/30", Data.omstiPath, 30);
    }

    public static void convertOntonotes()
    {
        System.out.println("Converting corpus " + Data.ontonotesPath);
        new OntonotesConverter().convert("data/corpus/original/ontonotes/5.0/data/files/data/english", Data.ontonotesPath, 30);
    }

    public static void convertSenseval2()
    {
        System.out.println("Converting corpus " + Data.senseval2Path);
        new Senseval2Converter().convert("data/corpus/original/mihalcea/senseval2", Data.senseval2Path, 171);
    }
    
    public static void convertSenseval3Task1()
    {
        System.out.println("Converting corpus " + Data.senseval3task1Path);
        new Senseval3Task1Converter().convert("data/corpus/original/mihalcea/senseval3", Data.senseval3task1Path, 171);
    }

    public static void convertSemeval2007Task7()
    {
        System.out.println("Converting corpus " + Data.semeval2007task7Path);
        new Semeval2007Task7Converter().convert("data/semeval/2007/task7", Data.semeval2007task7Path);
    }

    public static void convertSemeval2007Task17()
    {
        System.out.println("Converting corpus " + Data.semeval2007task17Path);
        new Semeval2007Task17Converter().convert("data/semeval/2007/task17", Data.semeval2007task17Path);
    }

    public static void convertSemeval2013Task12()
    {
        System.out.println("Converting corpus " + Data.semeval2013task12Path);
        new Semeval2013Task12Converter().convert("data/semeval/2013/task12", Data.semeval2013task12Path);
    }

    public static void convertSemeval2015Task13()
    {
        System.out.println("Converting corpus " + Data.semeval2015task13Path);
        new Semeval2015Task13Converter().convert("data/semeval/2015/task13", Data.semeval2015task13Path);
    }

    public static void postProcessCorpus(String corpusPath)
    {
        trimWordsAndRemoveInvisibleWords(corpusPath);
        convertWordnetAnnotations(corpusPath, 30);
        setLemmaAndPOSAnnotationsFromFirstSenseAnnotations(corpusPath, 30);
        addLemmasAndPOSAnnotations(corpusPath, 30);
        removeSenseTagsWhereLemmaOrPOSDiffers(corpusPath, 30);
        checkWordnetAnnotations(corpusPath);
    }

    public static void postProcessSentenceBasedCorpus(String corpusPath)
    {
        mergeDuplicatedSentences(corpusPath);
        checkDuplicatedSentences(corpusPath);
    }

    public static void trimWordsAndRemoveInvisibleWords(String corpusPath)
    {
        System.out.println("Trimming and Removing invisible words from corpus " + corpusPath);

        String corpusPathTmp = corpusPath + ".tmp.xml";
        final Pattern nonVisiblePattern = Pattern.compile("[^\\p{Graph}]");
        StreamingCorpusModifierSentence inout = new StreamingCorpusModifierSentence()
        {
            @Override
            public void modifySentence(Sentence sentence)
            {
                List<Word> sentenceWordsCopy = new ArrayList<>(sentence.getWords());
                for (Word w : sentenceWordsCopy)
                {
                    String wordValue = w.getValue();
                    wordValue = wordValue.trim();
                    Matcher matcher = nonVisiblePattern.matcher(wordValue);
                    wordValue = matcher.replaceAll("");
                    w.setValue(wordValue);
                    if (wordValue.equals(""))
                    {
                        sentence.removeWord(w);
                    }
                }
            }
        };

        inout.load(corpusPath, corpusPathTmp);
        File.moveFile(corpusPathTmp, corpusPath);
    }

    public static void mergeDuplicatedSentences(String inputPath)
    {
        System.out.println("Merging duplicated sentences of " + inputPath);

        String inputPathTmp = inputPath + ".tmp.xml";
        Wrapper<Integer> total = new Wrapper<>(0);
        Wrapper<Integer> failed = new Wrapper<>(0);
        Wrapper<Integer> annotationsDifferentReplaced = new Wrapper<>(0);
        StreamingCorpusWriterSentence out = new StreamingCorpusWriterSentence();
        Map<String, Sentence> realSentences = new LinkedHashMap<>();
        StreamingCorpusReaderSentence in = new StreamingCorpusReaderSentence()
        {
            @Override
            public void readSentence(Sentence s)
            {
                String sentenceAsString = s.toString();
                if (realSentences.containsKey(sentenceAsString))
                {
                    total.obj++;
                    Sentence realSentence = realSentences.get(sentenceAsString);
                    if (s.getWords().size() != realSentence.getWords().size())
                    {
                        System.out.println("Error for sentence 1 : " + sentenceAsString + " (" + s.getWords().size() + " words)");
                        System.out.println("Error for sentence 2 : " + realSentence.toString() + " (" + realSentence.getWords().size() + " words)");
                        failed.obj++;
                        return;
                    }
                    for (int i = 0; i < s.getWords().size(); i++)
                    {
                        Word w = s.getWords().get(i);
                        Word realWord = realSentence.getWords().get(i);
                        List<Annotation> wAnnotationsCopy = new ArrayList<>(w.getAnnotations());
                        for (Annotation a : wAnnotationsCopy)
                        {
                            if (realWord.hasAnnotation(a.getAnnotationName()) && !realWord.getAnnotationValue(a.getAnnotationName()).equals(a.getAnnotationValue()))
                            {
                                if (a.getAnnotationName().startsWith("wn") && a.getAnnotationName().endsWith("_key"))
                                {
                                    String newAnnotationValue = a.getAnnotationValue() + ";" + realWord.getAnnotationValue(a.getAnnotationName());
                                    newAnnotationValue = removeDuplicateInAnnotationValue(newAnnotationValue);
                                    w.setAnnotation(a.getAnnotationName(), newAnnotationValue);
                                }
                                else
                                {
                                    annotationsDifferentReplaced.obj++;
                                }
                            }
                        }
                        w.transfertAnnotationsToCopy(realWord);
                    }
                }
                else
                {
                    realSentences.put(sentenceAsString, s);
                }
            }
        };

        in.load(inputPath);
        out.open(inputPathTmp);
        for (Map.Entry<String, Sentence> realSentencesEntry : realSentences.entrySet())
        {
            out.writeSentence(realSentencesEntry.getValue());
        }
        out.close();
        File.moveFile(inputPathTmp, inputPath);
        
        System.out.println("Found " + total.obj + " duplicated sentences in " + inputPath + " - failed to delete " + failed.obj);
        System.out.println("Replaced " + annotationsDifferentReplaced.obj + " different annotations");
    }

    public static String removeDuplicateInAnnotationValue(String annotationValue)
    {
        return StringUtils.join(new HashSet<>(Arrays.asList(annotationValue.split(";"))), ";");
    }

    public static void convertWordnetAnnotations(String inputPath, int wnVersionOut)
    {
        System.out.println("Converting WN annotations of " + inputPath + " to WN " + wnVersionOut);

        String inputPathTmp = inputPath + ".tmp.xml";

        Wrapper<Integer> count = new Wrapper<>(0);
        Wrapper<Integer> failed = new Wrapper<>(0);

        String wnVersionOutTag = "wn" + wnVersionOut + "_key";

        StreamingCorpusModifierWord inout = new StreamingCorpusModifierWord()
        {
            public void modifyWord(Word word)
            {
                List<Annotation> annotationsCopy = new ArrayList<>(word.getAnnotations());
                for (Annotation annotation : annotationsCopy)
                {
                    String annotationName = annotation.getAnnotationName();
                    if (!annotationName.startsWith("wn")) continue;
                    if (!annotationName.endsWith("_key")) continue;
                    count.obj++;
                    String wnVersion = annotation.getAnnotationName().substring(2, annotationName.indexOf("_key"));
                    if (wnVersion.equals("" + wnVersionOut)) continue;
                    String wnKey = annotation.getAnnotationValue();
                    word.setAnnotation(wnVersionOutTag, getNewSenseKey(wnKey, WordnetMapping.wnXtoY(Integer.valueOf(wnVersion), wnVersionOut)));
                    if (!word.hasAnnotation(wnVersionOutTag))
                    {
                        failed.obj++;
                    }
                }
            }
        };

        inout.load(inputPath, inputPathTmp);
        File.moveFile(inputPathTmp, inputPath);

        System.out.println(count.obj + " total WN annotations - failed to convert " + failed.obj + " to WN " + wnVersionOut);
    }

    public static String getNewSenseKey(String rawSenseKey, WordnetMapping mapping)
    {
        String[] senseKeys = rawSenseKey.split(";");
        List<String> newSenseKeys = new ArrayList<>();
        for (String senseKey : senseKeys)
        {
            String mappedSenseKey = mapping.fromXtoY(senseKey);
            if (!(mappedSenseKey == null || mappedSenseKey.isEmpty() || newSenseKeys.contains(mappedSenseKey)))
            {
                newSenseKeys.add(mappedSenseKey);
            }
        }
        if (newSenseKeys.isEmpty()) return "";
        else return StringUtils.join(newSenseKeys, ";");
    }

    public static void setLemmaAndPOSAnnotationsFromFirstSenseAnnotations(String inputPath, int wordnetVersion)
    {
        System.out.println("Setting Lemma and POS annotations from first sense annotation for " + inputPath);

        String inputPathTmp = inputPath + ".tmp.xml";
        String senseTag = "wn" + wordnetVersion + "_key";
        StreamingCorpusModifierWord inout = new StreamingCorpusModifierWord()
        {
            public void modifyWord(Word word)
            {
                if (word.hasAnnotation(senseTag))
                {
                    String senseKey = word.getAnnotationValue(senseTag);
                    String senseKeyPOS = POSHelper.processPOS(Integer.valueOf(senseKey.substring(senseKey.indexOf("%") + 1, senseKey.indexOf("%") + 2)));
                    String senseKeyLemma = senseKey.substring(0, senseKey.indexOf("%"));
                    String currentPOS = POSHelper.processPOS(word.getAnnotationValue("pos"));
                    if (!currentPOS.equals(senseKeyPOS))
                    {
                        word.setAnnotation("pos", senseKeyPOS);
                    }
                    String currentLemma = word.getAnnotationValue("lemma");
                    if (!currentLemma.equals(senseKeyLemma))
                    {
                        word.setAnnotation("lemma", senseKeyLemma);
                    }
                }
            }
        };
        inout.load(inputPath, inputPathTmp);
        File.moveFile(inputPathTmp, inputPath);
    }

    public static void addLemmasAndPOSAnnotations(String corpusPath, int wnVersion)
    {
        System.out.println("Adding Lemma and POS annotations with Stanford Tagger for " + corpusPath);

        StdOutStdErr.stfu();
        if (tagger == null) tagger = new MaxentTagger("data/stanford/model/english.tagger");
        StdOutStdErr.speak();
        
        WordnetHelper wn = WordnetHelper.wn(wnVersion);
        StreamingCorpusModifierSentence inout = new StreamingCorpusModifierSentence()
        {
            public void modifySentence(Sentence sentence)
            {
                List<TaggedWord> stanfordSentence = tagger.tagSentence(toStanfordSentence(sentence));
                assert (stanfordSentence.size() != sentence.getWords().size());
                for (int i = 0; i < stanfordSentence.size(); i++)
                {
                    Word word = sentence.getWords().get(i);
                    if (word.hasAnnotation("lemma") && word.hasAnnotation("pos"))
                    {
                        continue;
                    }
                    String pos = word.getAnnotationValue("pos");
                    if (pos.isEmpty() || POSHelper.processPOS(pos).equals(POSHelper.processPOS(stanfordSentence.get(i).tag())))
                    {
                        pos = stanfordSentence.get(i).tag();
                    }
                    if (!pos.isEmpty())
                    {
                        String lemma = word.getAnnotationValue("lemma");
                        if (lemma.isEmpty() && !POSHelper.processPOS(pos).equals("x"))
                        {
                            lemma = wn.morphy(word.getValue(), POSHelper.processPOS(pos));
                        }
                        if (word.hasAnnotation("pos") && !lemma.isEmpty())
                        {
                            word.removeAnnotation("pos");
                        }
                        word.setAnnotation("lemma", lemma);
                        word.setAnnotation("pos", pos);
                    }
                }
            }
        };
        String corpusPathTmp = corpusPath + ".tmp.xml";
        inout.load(corpusPath, corpusPathTmp);
        File.moveFile(corpusPathTmp, corpusPath);
    }

    public static List<HasWord> toStanfordSentence(Sentence sentence)
    {
        List<HasWord> stanfordSentence = new ArrayList<>();
        for (Word word : sentence.getWords())
        {
            stanfordSentence.add(new edu.stanford.nlp.ling.Word(word.getValue()));
        }
        return stanfordSentence;
    }

    public static void removeSenseTagsWhereLemmaOrPOSDiffers(String corpusPath, int wordnetVersion)
    {
        System.out.println("Removing sense tags where POS differs for " + corpusPath);

        String inputPathTmp = corpusPath + ".tmp.xml";
        String senseTag = "wn" + wordnetVersion + "_key";
        StreamingCorpusModifierWord inout = new StreamingCorpusModifierWord()
        {
            public void modifyWord(Word word)
            {
                if (word.hasAnnotation(senseTag))
                {
                    String lemma = word.getAnnotationValue("lemma");
                    String pos = POSHelper.processPOS(word.getAnnotationValue("pos"));
                    String newSenseKey = "";
                    String[] senses = word.getAnnotationValue(senseTag).split(";");
                    for (String sense : senses)
                    {
                        String senseLemma = sense.substring(0, sense.indexOf("%"));
                        String sensePos = POSHelper.processPOS(Integer.valueOf(sense.substring(sense.indexOf("%") + 1, sense.indexOf("%") + 2)));
                        if (senseLemma.equals(lemma) && sensePos.equals(pos))
                        {
                            newSenseKey += sense + ";";
                        }
                    }
                    if (!newSenseKey.isEmpty())
                    {
                        newSenseKey = newSenseKey.substring(0, newSenseKey.length() - 1);
                        word.setAnnotation(senseTag, newSenseKey);
                    }
                    else
                    {
                        word.removeAnnotation(senseTag);
                    }
                }
            }
        };
        inout.load(corpusPath, inputPathTmp);
        File.moveFile(inputPathTmp, corpusPath);
    }

    public static void checkWordnetAnnotations(String corpusPath)
    {
        System.out.println("Checking WN annotations for " + corpusPath);

        Wrapper<Integer> count = new Wrapper<>(0);
        Wrapper<Integer> fail = new Wrapper<>(0);

        StreamingCorpusReaderWord corpus = new StreamingCorpusReaderWord()
        {
            @Override
            public void readWord(Word word)
            {
                for (Annotation annotation : word.getAnnotations())
                {
                    String annotationName = annotation.getAnnotationName();
                    if (!annotationName.startsWith("wn")) continue;
                    if (!annotationName.endsWith("_key")) continue;
                    count.obj++;
                    String wnVersion = annotation.getAnnotationName().substring(2, annotationName.indexOf("_key"));
                    WordnetHelper wn = WordnetHelper.wn(Integer.valueOf(wnVersion));
                    String wnKey = annotation.getAnnotationValue();
                    String[] senseKeys = wnKey.split(";");
                    for (String senseKey : senseKeys)
                    {
                        if (!wn.isSenseKeyExists(senseKey))
                        {
                            fail.obj++;
                        }
                    }
                }
            }
        };

        corpus.load(corpusPath);

        System.out.println(count.obj + " total WN annotations - " + fail.obj + " incorrect");
    }

    public static void checkDuplicatedSentences(String corpusPath)
    {
        System.out.println("Checking duplicated sentences in " + corpusPath);

        Wrapper<Integer> i = new Wrapper<>(0);
        Set<String> realSentences = new HashSet<>();
        
        StreamingCorpusReaderSentence in = new StreamingCorpusReaderSentence()
        {
            @Override
            public void readSentence(Sentence s)
            {
                if (realSentences.contains(s.toString()))
                {
                    i.obj = i.obj + 1;
                }
                else
                {
                    realSentences.add(s.toString());
                }
            }
        };
        
        in.load(corpusPath);
        System.out.println("Found " + i.obj + " duplicate sentences in corpus " + corpusPath);
    }

    public static void cutInPieces(String inputPath, String outputPath, int piecesCount)
    {
        Wrapper<Integer> totalSentenceCount = new Wrapper<>(0);
        
        StreamingCorpusReaderSentence in = new StreamingCorpusReaderSentence()
        {
            @Override
            public void readSentence(Sentence s)
            {
                totalSentenceCount.obj += 1;
            }
        };

        in.load(inputPath);

        int sentencePerPieces = totalSentenceCount.obj / piecesCount;
        int remainingSentences = totalSentenceCount.obj % piecesCount;
        int finalSentencePerPieces = sentencePerPieces + ((remainingSentences == 0) ? 0 : 1);

        Wrapper<Integer> currentSentenceCount = new Wrapper<>(finalSentencePerPieces);
        Wrapper<Integer> currentPart = new Wrapper<>(0);
        Wrapper<StreamingCorpusWriterSentence> out = new Wrapper<>(null);
        
        in = new StreamingCorpusReaderSentence()
        {
            @Override
            public void readSentence(Sentence s)
            {
                if (currentSentenceCount.obj >= finalSentencePerPieces)
                {
                    if (out.obj != null) out.obj.close();
                    currentSentenceCount.obj = 0;
                    out.obj = new StreamingCorpusWriterSentence();
                    out.obj.open(outputPath + currentPart.obj + ".xml");
                    currentPart.obj += 1;
                }
                out.obj.writeSentence(s);
                currentSentenceCount.obj += 1;
            }
        };
        
        in.load(inputPath);
        out.obj.close();

        File.removeFile(inputPath);
    }
}
