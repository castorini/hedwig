import argparse

import jnius_config
jnius_config.set_classpath("../Anserini/target/anserini-0.0.1-SNAPSHOT.jar")
from jnius import autoclass


class RetrieveSentences:
    """Python class built to call RetrieveSentences
    Attributes
    ----------
    rs : io.anserini.qa.RetrieveSentences
        The RetrieveSentences object
    args : io.anserini.qa.RetrieveSentences$Args
        The arguments for constructing RetrieveSentences object as well as calling getRankedPassages

    """

    def __init__(self, args):
        """
        Constructor for the CallRetrieveSentences class.

        Parameters
        ----------
        args : argparse.Namespace
            Arguments needed for constructing an instance of RetrieveSentences class
        """
        RetrieveSentences = autoclass("io.anserini.qa.RetrieveSentences")
        Args = autoclass("io.anserini.qa.RetrieveSentences$Args")
        self.String = autoclass("java.lang.String")

        self.args = Args()
        index = self.String(args.index)
        self.args.index = index
        embeddings = self.String(args.embeddings)
        self.args.embeddings = embeddings
        topics = self.String(args.topics)
        self.args.topics = topics
        query = self.String(args.query)
        self.args.query = query
        self.args.hits = int(args.hits)
        scorer = self.String(args.scorer)
        self.args.scorer = scorer
        self.args.k = int(args.k)
        self.rs = RetrieveSentences(self.args)

    def getRankedPassages(self, query, index, hits, k):
        """
        Calls RetrieveSentences.getRankedPassages

        Parameters
        ----------
        query : str
            The query to be searched in the index
        index:  str
            The index
        hits:   str
            The number of document IDs to be returned
        k: str
            The number of passages to be returned 
        """

        scorer = self.rs.getRankedPassagesList(query.strip(), index, int(hits), int(k))
        candidate_passages_scores = []
        for i in range(0, scorer.size()):
            candidate_passages_scores.append(scorer.get(i))

        return candidate_passages_scores

    def getTermIdfJSON(self):
        """
        Calls RetrieveSentences.getTermIdfJSON
        """
        return self.rs.getTermIdfJSON()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve Sentences')
    parser.add_argument("-index", help="Lucene index", required=True)
    parser.add_argument("-embeddings", help="Path of the word2vec index", default="")
    parser.add_argument("-topics", help="topics file", default="")
    parser.add_argument("-query", help="a single query", default="who is newton")
    parser.add_argument("-hits", help="max number of hits to return", default=100)
    parser.add_argument("-scorer", help="passage scores", default="Idf")
    parser.add_argument("-k", help="top-k passages to be retrieved", default=1)

    args_raw = parser.parse_args()

    rs = RetrieveSentences(args_raw)
    sc = rs.getRankedPassages(args_raw.query, args_raw.index, args_raw.hits, args_raw.k)
