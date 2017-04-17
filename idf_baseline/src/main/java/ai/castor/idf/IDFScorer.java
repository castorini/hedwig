/**
 * Anserini: An information retrieval toolkit built on Lucene
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.castor.idf;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.kohsuke.args4j.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


public class IDFScorer {

  public static class Args {
    // required arguments
    @Option(name = "-index", metaVar = "[path]", required = true, usage = "Lucene index")
    public String index;

    @Option(name = "-config", metaVar = "[string]", required = true, usage = "type of the dataset")
    public String config;

    @Option(name = "-output", metaVar = "[path]", required = true, usage = "path of the file to be created")
    public String output;

    // optional arguments
    @Option(name = "-analyze", usage = "passage scores")
    public boolean analyze = false;
  }

  private final IndexReader reader;
  private final FSDirectory directory;
  private final List<String> stopWords;

  public static final String FIELD_BODY = "contents";

  public IDFScorer(IDFScorer.Args args) throws Exception {
    Path indexPath = Paths.get(args.index);
    if (!Files.exists(indexPath) || !Files.isDirectory(indexPath) || !Files.isReadable(indexPath)) {
      throw new IllegalArgumentException(args.index + " does not exist or is not a directory.");
    }
    this.directory = FSDirectory.open(indexPath);
    this.reader = DirectoryReader.open(directory);

    stopWords = new ArrayList<>();

    //Get file from resources folder
    InputStream is = getClass().getResourceAsStream("/ai/castor/qa/english-stoplist.txt");
    BufferedReader bRdr = new BufferedReader(new InputStreamReader(is));
    String line;
    while ((line = bRdr.readLine()) != null) {
      if (!line.contains("#")) {
        stopWords.add(line);
      }
    }
  }

  public double calcIDF(String query, String answer, boolean analyze) throws ParseException {
    Analyzer analyzer;
    if (analyze) {
      analyzer = new EnglishAnalyzer(StopFilter.makeStopSet(stopWords));
    } else {
      analyzer = new WhitespaceAnalyzer();
    }

    QueryParser qp = new QueryParser(FIELD_BODY, analyzer);
    ClassicSimilarity similarity = new ClassicSimilarity();

    String escapedQuery = qp.escape(query);
    Query question = qp.parse(escapedQuery);
    HashSet<String> questionTerms = new HashSet<>(Arrays.asList(question.toString().trim().split("\\s+")));

    double idf = 0.0;
    HashSet<String> seenTerms = new HashSet<>();

    String[] terms = answer.split("\\s+");
    for (String term : terms) {
      try {
        TermQuery q = (TermQuery) qp.parse(term);
        Term t = q.getTerm();

        if (questionTerms.contains(t.toString()) && !seenTerms.contains(t.toString())) {
          idf += similarity.idf(reader.docFreq(t), reader.numDocs());
          seenTerms.add(t.toString());
        } else {
          idf += 0.0;
        }
      } catch (Exception e) {
        continue;
      }
    }
    return idf;
  }

  public  void writeToFile(IDFScorer.Args args) throws IOException, ParseException {
    BufferedReader questionFile = new BufferedReader(new FileReader(args.config + "/a.toks"));
    BufferedReader answerFile = new BufferedReader(new FileReader(args.config + "/b.toks"));
    BufferedReader idFile = new BufferedReader(new FileReader(args.config + "/id.txt"));

    BufferedWriter outputFile = new BufferedWriter(new FileWriter(args.output));
    int i = 0;

    String old_id = "0";    
    while (true) {
      String question = questionFile.readLine();
      String answer = answerFile.readLine();
      String id = idFile.readLine();

      if (question == null || answer == null || id == null) {
        break;
      }

      // we need new lines here
      if (args.config.contains("WikiQA") && !old_id.equals(id)) {
        old_id = id;
        i = 0;
      }

      // 32.1 0 0 0 0.6212325096130371 smmodel
      // 32.1 0 1 0 0.13309887051582336 smmodel
      outputFile.write(id + " 0 " + i + " 0 " + calcIDF(question, answer, args.analyze) + " idfbaseline\n");
      i++;
    }
    outputFile.close();

  }

  public static void main(String[] args) throws Exception {
    Args qaArgs = new Args();
    CmdLineParser parser = new CmdLineParser(qaArgs, ParserProperties.defaults().withUsageWidth(90));

    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.err.println("Example: IDFScorer" + parser.printExample(OptionHandlerFilter.REQUIRED));
      return;
    }

    IDFScorer g = new IDFScorer(qaArgs);
    g.writeToFile(qaArgs);
  }
}
