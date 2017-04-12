# IDF scorer 

Download the TrecQA and WikiQA data (question-answer pairs) from[here](https://github.com/castorini/data.git)

Switch to an appropriate directory and run the following scripts:

```
python3 parse.py

python3 overlap_features.py

python3 build_vocab.py
```

After running the script, you should have the following directory structure:

```
├── raw-dev
├── raw-test
├── train
└── train-all
```
and each directory should have the following files:
```
├── a.toks
├── b.toks
├── boundary.txt
├── id.txt
├── numrels.txt
└── sim.txt
```
 
Clone and compile[Anserini](https://github.com/castorini/Anserini.git)

```
git clone https://github.com/castorini/Anserini.git
cd Anserini
mvn clean package appassembler:assemble
``` 
 
### Indexing WikiQA collection

First, download the Wikipedia dump by running the following command:

```
mkdir WikiQACollection
for line in $(cat idf_baseline/src/main/resources/WikiQA/wikidump-list.txt); do wget $line -P WikiQACollection; done
```

To index the collection:
```
cd Anserini
nohup sh target/appassembler/bin/IndexCollection -collection WikipediaCollection -input ../WikiQACollection
-generator JsoupGenerator -index lucene.index.wikipedia.pos.docvectors -threads 32 -storePositions 
-storeDocvectors -optimize > log.wikipedia.pos.docvectors & 
```

### Indexing TrecQA collection

Create a new directory called TrecQACollection
```
mkdir TrecQACollection
```

Copy the contents of disk1, disk2, disk3, disk4, and AQUAINT to TrecQACollection

To index the collection:

```
cd Anserini
nohup sh target/appassembler/bin/IndexCollection -collection TrecCollection -input [path of TrecQACollection]
-generator JsoupGenerator -index lucene.index.trecQA.pos.docvectors -threads 32 -storePositions 
-storeDocvectors -optimize > log.trecQA.pos.docvectors & 
```

### Calculating IDF overlap

Run the following command to score each answer with an IDF value:

```
sh target/appassembler/bin/GetIDF
```

Possible parameters are:

```
-index (required)
```

Path of the index

```
-config (requiered)
```
Configuration of this experiment i.e., dev, train, train-all, test etc.

```
-output (optional: file path)
```

Path of the run file to be created

```
-analyze 
```
If specified, the scorer uses  `EnglishAnalyzer` for removing stopwords and stemming. In addtion to 
the default list, the analyzer uses NLTK's stopword list obtained 
from[here](https://gist.github.com/sebleier/554280)

The above command will create a run file in the `trec_eval` format and a qrel file
at a location specified by `-output`.

### Evaluating the system:

To calculate MAP/MRR for the above run file:

- Download and install `trec_eval` from[here](https://github.com/castorini/Anserini/blob/master/eval/trec_eval.9.0.tar.gz)

```
eval/trec_eval.9.0/trec_eval -m map -m recip_rank <qrel-file> <run-file>
```