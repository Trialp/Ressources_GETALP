package getalp.wsd.ufsac.corpus.conversion;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import getalp.wsd.ufsac.corpus.Sentence;
import getalp.wsd.ufsac.corpus.Word;
import getalp.wsd.ufsac.corpus.streaming.writer.StreamingCorpusWriterSentence;
import getalp.wsd.ufsac.wordnet.WordnetHelper;

public class DSOConverter
{
    private WordnetHelper wn;
    
    private StreamingCorpusWriterSentence out;

    public void convert(String inputPath, String outputPath, int wnVersion)
    {
        wn = WordnetHelper.wn(wnVersion);
        out = new StreamingCorpusWriterSentence();
        out.open(outputPath);
        try
        {
            processList(inputPath, inputPath + "/vlist.txt", "v");
            processList(inputPath, inputPath + "/nlist.txt", "n");
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
        out.close();
    }

    private void processList(String rootPath, String listPath, String pos) throws IOException
    {
        BufferedReader br = new BufferedReader(new FileReader(listPath));
        for (String line = br.readLine() ; line != null ; line = br.readLine())
        {
            processWordInList(rootPath + "/" + line + "." + pos, line, pos);
        }
        br.close();
    }
    
    private void processWordInList(String wordPath, String lemma, String pos) throws IOException
    {
        //System.out.println("[DSO] Processing word " + word + "...");
        BufferedReader br = new BufferedReader(new FileReader(wordPath));
        for (String line = br.readLine() ; line != null ; line = br.readLine())
        {
            if (line.isEmpty()) continue;
            processSentenceInWord(line, lemma, pos);
        }
        br.close();
    }
    
    private void processSentenceInWord(String sentence, String wordLemma, String wordPOS)
    {
        String[] tokens = sentence.split("\\s+");
        Sentence resultingSentence = new Sentence();
        for (int i = 2 ; i < tokens.length ; i++)
        {
            if (tokens[i].equals(">>") && i + 3 < tokens.length && tokens[i + 3].equals("<<"))
            {
                String surfaceForm = tokens[i + 1];
                int senseNumber = Integer.valueOf(tokens[i + 2]);
                i += 3;
                if (senseNumber <= 0)
                {
                    return;
                }
                String senseKey = wn.getSenseKeyFromSenseNumber(wordLemma + "%" + wordPOS + "#" + senseNumber);
                if (senseKey == null)
                {
                    //System.out.println("Warning : cannot find sense key for " + wordLemma + "%" + wordPOS + "#" + senseNumber);
                    return;
                }
                Word currentWord = new Word(resultingSentence);
                currentWord.setValue(surfaceForm);
                currentWord.setAnnotation("lemma", wordLemma);
                currentWord.setAnnotation("pos", wordPOS);
                currentWord.setAnnotation("wn" + wn.getVersion() + "_key", senseKey);
            } 
            else 
            {
                Word currentWord = new Word(resultingSentence);
                currentWord.setValue(tokens[i]);
            }
        } 
        out.writeSentence(resultingSentence);
    }

}