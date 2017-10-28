## Retrieve Sentences

#### 1. Clone [Anserini](https://github.com/castorini/Anserini.git) and [Castor](https://github.com/castorini/Castor.git)
```bash
git clone https://github.com/castorini/Anserini.git
git clone https://github.com/castorini/Castor.git
```

Your directory structure should look like
```
├── Anserini
└── Castor
```

#### 2. Compile Anserini

```bash
cd Anserini
mvn package
``` 

This creates `anserini-0.0.1-SNAPSHOT.jar` at `Anserini/target`

#### 3. Download Dependencies
- Download the index from [here](https://drive.google.com/open?id=0B2u_nClt6NbzcllGZTVtX2Q4bkk)
- Download the Google word2vec file from [here](https://drive.google.com/drive/folders/0B2u_nClt6NbzNWJkWExmaklYNTA?usp=sharing)

#### 4. Run the following command

```bash
python ./anserini_dependency/RetrieveSentences.py
```

Possible parameters are: 

| option         | input format | default | description |
|----------------|--------------|---------|-------------|
| `-index`   | string      | N/A     | Path of the Lucene index            |
| `-embeddings`   | string       | ""     | Path of the word2vec index            |
| `-topics`   | string       | ""     | topics file            |
| `-query`   | string       | ""     | a single query            |
| `-hits`   | [1, inf)       | 100     | max number of hits to return            |
| `-scorer`   | string       | Idf     | passage scores (Idf or Wmd)            |
| `-k`   | [1, inf)       | 1     | top-k passages to be retrieved            |
  
Note: Either a query or a topic must be passed in as an argument; they can't be both empty.
