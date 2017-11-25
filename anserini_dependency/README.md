## Setup Retrieve Sentences and end2end QA pipeline

#### 1. Clone [Anserini](https://github.com/castorini/Anserini.git), [Castor](https://github.com/castorini/Castor.git), [data](https://github.com/castorini/data.git), and [models](https://github.com/castorini/models.git):
```bash
git clone https://github.com/castorini/Anserini.git
git clone https://github.com/castorini/Castor.git
git clone https://github.com/castorini/data.git
git clone https://github.com/castorini/models.git
```

Your directory structure should look like
```
.
├── Anserini
├── Castor
├── data
└── models
```

#### 2. Compile Anserini

```bash
cd Anserini
mvn package
cd ..
``` 

This creates `anserini-0.0.1-SNAPSHOT.jar` at `Anserini/target`

We highly recommend the use of [virtualenv](https://virtualenv.pypa.io/en/stable/) as the dependencies
are subjected to frequent changes.

Install the dependency packages:

```
cd Castor
pip3 install -r requirements.txt
```
Make sure that you have PyTorch installed. For more help, follow [these](https://github.com/castorini/Castor) steps.

#### 3. Download Dependencies
- Download the TrecQA lucene index
- Download the Google word2vec file from [here](https://drive.google.com/drive/folders/0B2u_nClt6NbzNWJkWExmaklYNTA?usp=sharing)

#### 4. Additional files for pipeline:
As some of the files are too large to be uploaded onto GitHub, please download the following files from
[here](https://drive.google.com/drive/folders/0B2u_nClt6NbzNm1LdjlwUFdzQVE?usp=sharing) and place them
in the appropriate locations:

- copy the contents of `word2vec` directory to `data/word2vec`
- copy `word2dfs.p` to `data/TrecQA/`

### To run RetrieveSentences:

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


__NB:__  The speech UI cannot be run in Ubuntu. To test the pipeline in Ubuntu, make the following changes:
- Comment out the JavaScript part and run the Bash script
- Make a REST API query to the endpoint using Postman, Curl etc.

### To setup the demo

#### 1. Installing libraries for demo

```sh
cd anserini_dependency/js
npm install
cd ../..
```

#### 2. Flask

- Flask is used as the server for the API
- Copy `config.cfg.example` to `config.cfg` and make necessary changes, such as setting the index path and API keys.


#### 3. Run the Demo

```sh
./run_ui.sh
```

### Additional Notes
- This is the documentation for the API call to send a question to the model and get back the predicted answer.
- The request body fields are: question(required )num_hits(optional) and k(optional).
```

# REQUEST:
HTTP Method: POST
Endpoint: [host]:[port]/answer
Content-Type: application/json
text of body in raw format:
{
    "question": "What is the birthdate of Einstein?",
    "num_hits": 50,
    "k": 30
}
```

- The response body contains answers which is a list of objects with two fields - passage, score.
```
# RESPONSE:
Content-Type: application/json
text of body in raw format:
{
  "answers": [
                {"passage": "Einstein was born in the 1800s", 'score': 0.976},
                {"passage": "Einstein was a physicist", 'score': 0.524}
            ]
}
```
