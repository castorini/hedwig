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
import org.apache.lucene.analysis.CharArraySet;
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

public class FetchTermIDF {
    
    public static class Args {
        @Option(name = "-index", metaVar="[path]", required = true, usage = "path to Lucene index")
        public String index;

        @Option(name = "-terms", usage = "space separated list of terms to get idf for.")
        public String terms;

        @Option(name = "-vocabFile", usage = "file with one term per line")
        public String vocabFile;
    }

    private final IndexReader reader;
    private final FSDirectory directory;

    public static final String FIELD_BODY = "contents";

    public FetchTermIDF(FetchTermIDF.Args args) throws Exception {
        Path indexPath = Paths.get(args.index);
        if (!Files.exists(indexPath) || !Files.isDirectory(indexPath) || !Files.isReadable(indexPath)) {
            throw new IllegalArgumentException(args.index + " does not exist, is not a directory, or is not readable");
        }
        this.directory = FSDirectory.open(indexPath);
        this.reader = DirectoryReader.open(this.directory); // #changed
    }   

    public double getTermIDF(String term) throws ParseException {
        Analyzer analyzer = new EnglishAnalyzer(CharArraySet.EMPTY_SET);
        QueryParser qp = new QueryParser(FIELD_BODY, analyzer);
        ClassicSimilarity similarity = new ClassicSimilarity();

        String esTerm = qp.escape(term);
        double termIDF = 0.0;
        try {
            TermQuery q = (TermQuery) qp.parse(esTerm);
            Term t = q.getTerm();
            termIDF = similarity.idf(reader.docFreq(t), reader.numDocs());

            System.out.println(term + '\t' + esTerm + '\t' + q + '\t' + t + '\t' + termIDF);            
        } catch (Exception e) {
            System.err.println("Exception in fetching IDF(" + term + "): " + e.toString());
        }
        return termIDF;
    }

    public static void main(String[] args) throws Exception {
        Args qaArgs = new Args();
        CmdLineParser parser = new CmdLineParser(qaArgs, ParserProperties.defaults().withUsageWidth(90));

        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            System.err.println("Example: FetchTermIDF" + parser.printExample(OptionHandlerFilter.REQUIRED));
            return;
        }
        
        if (qaArgs.terms == null && qaArgs.vocabFile == null) {
            System.out.println("Required one of -vocabFile or -terms arguments");
            return;
        }

        FetchTermIDF idfFetcher = new FetchTermIDF(qaArgs);
        List<String> termsList = null;
        if (qaArgs.terms != null && !qaArgs.terms.isEmpty()) {
            termsList = Arrays.asList(qaArgs.terms.split("\\s+"));
            for(String t: termsList) {
                idfFetcher.getTermIDF(t);
            }
        } else {
            try(BufferedReader br = new BufferedReader(new FileReader(qaArgs.vocabFile))) {
            for(String line; (line = br.readLine()) != null; ) {
                idfFetcher.getTermIDF(line);
            }
            // line is not visible here.
        }   
        }
    }

}