import heapq
from collections import Counter
import math
from indexing import InvertedIndex
from tqdm import tqdm

class Ranker:
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None, top_k: int = 100) -> None:
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict
        self.top_k = top_k
        self.cache = {}
        self.show_progress = False

    def query(self, query: str) -> list[tuple[int, float]]:
        query_tokens = [token for token in self.tokenize(query)]
        if self.stopwords:
            query_tokens = [token if token.lower() not in self.stopwords else '$$$' for token in query_tokens]
        query_word_counts = Counter(query_tokens)

        candidate_docids = set()
        for term in query_word_counts:
            postings = self.index.get_postings(term)
            docids = {docid for docid, *_ in postings}
            candidate_docids.update(docids)

        if not candidate_docids:
            return []

        stats = self.index.get_statistics()
        N = stats['number_of_documents']
        avgdl = stats['mean_document_length']
        collection_length = stats['total_token_count']

        term_stats = {}
        for term in query_word_counts:
            term_metadata = self.index.get_term_metadata(term)
            df = term_metadata['doc_frequency']
            cf = term_metadata['term_count']
            term_stats[term] = {'df': df, 'cf': cf}

        # 预计算并准备评分器
        self.scorer.prepare(
            query_word_counts=query_word_counts,
            N=N,
            avgdl=avgdl,
            collection_length=collection_length,
            term_stats=term_stats
        )

        scored_documents = []
        if self.show_progress:
            iterator = tqdm(candidate_docids, desc='Scoring documents')
        else:
            iterator = candidate_docids

        for docid in iterator:
            doc_word_counts = self.index.get_document_word_counts(docid)
            doc_length = self.index.get_doc_metadata(docid)['length']
            score = self.scorer.score(
                docid, doc_word_counts, doc_length=doc_length
            )
            scored_documents.append((docid, score))
        top_documents = heapq.nlargest(self.top_k, scored_documents, key=lambda x: (x[1], -x[0]))

        return top_documents

class RelevanceScorer:
    def __init__(self, index, parameters) -> None:
        self.index = index
        self.parameters = parameters

    def prepare(self, query_word_counts, N, avgdl, collection_length, term_stats):
        self.query_word_counts = query_word_counts
        self.N = N
        self.avgdl = avgdl
        self.collection_length = collection_length
        self.term_stats = term_stats
        self.query_terms = set(query_word_counts.keys())

    def score(self, docid: int, doc_word_counts: dict[str, int],
              query_word_counts: dict[str, int] = None, **kwargs) -> float:
        if query_word_counts is not None:
            # 如果提供了query_word_counts，重新prepare
            stats = self.index.get_statistics()
            N = stats['number_of_documents']
            avgdl = stats['mean_document_length']
            collection_length = stats['total_token_count']
            term_stats = compute_term_stats(self.index, query_word_counts.keys())
            self.prepare(query_word_counts, N, avgdl, collection_length, term_stats)
        elif not hasattr(self, 'query_word_counts'):
            raise ValueError("query_word_counts must be provided if prepare() has not been called.")
        return self._score(docid, doc_word_counts, **kwargs)

    def _score(self, docid: int, doc_word_counts: dict[str, int], **kwargs) -> float:
        raise NotImplementedError

def compute_term_stats(index, terms):
    term_stats = {}
    for term in terms:
        term_metadata = index.get_term_metadata(term)
        df = term_metadata['doc_frequency']
        cf = term_metadata['term_count']
        term_stats[term] = {'df': df, 'cf': cf}
    return term_stats

class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        super().__init__(index, parameters)

    def _score(self, docid: int, doc_word_counts: dict[str, int], **kwargs) -> float:
        matching_terms = self.query_terms & doc_word_counts.keys()
        score = sum(self.query_word_counts[term] * doc_word_counts[term] for term in matching_terms)
        return score

class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        super().__init__(index, parameters)

    def prepare(self, query_word_counts, N, avgdl, collection_length, term_stats):
        super().prepare(query_word_counts, N, avgdl, collection_length, term_stats)
        # 预计算IDF值
        self.idf_values = {}
        for term in self.query_word_counts:
            df = self.term_stats[term]['df']
            # 避免 df 为零
            df = df if df > 0 else 1
            idf = math.log((N / df)) + 1
            self.idf_values[term] = idf

    def _score(self, docid: int, doc_word_counts: dict[str, int], **kwargs) -> float:
        matching_terms = self.query_terms & doc_word_counts.keys()
        score = 0.0
        for term in matching_terms:
            tf = doc_word_counts[term]
            idf = self.idf_values[term]
            tf_weight = math.log(tf + 1)
            term_score = tf_weight * idf
            score += term_score
        return score

class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        super().__init__(index, parameters)
        self.b = parameters.get('b', 0.75)
        self.k1 = parameters.get('k1', 1.2)
        self.k3 = parameters.get('k3', 8)

    def prepare(self, query_word_counts, N, avgdl, collection_length, term_stats):
        super().prepare(query_word_counts, N, avgdl, collection_length, term_stats)
        self.idf_values = {}
        for term in self.query_word_counts:
            df = self.term_stats[term]['df']
            # 避免 df 为零
            df = df if df > 0 else 1
            idf = math.log((N - df + 0.5) / (df + 0.5))
            self.idf_values[term] = idf

    def _score(self, docid: int, doc_word_counts: dict[str, int], doc_length: int = None, **kwargs) -> float:
        if doc_length is None:
            doc_length = self.index.get_doc_metadata(docid)['length']
        avgdl = self.avgdl

        score = 0.0
        for term in self.query_terms:
            tf = doc_word_counts.get(term, 0)
            if tf == 0:
                continue
            qf = self.query_word_counts[term]
            idf = self.idf_values[term]
            tf_component = ((self.k1 + 1) * tf) / (self.k1 * (1 - self.b + self.b * (doc_length / avgdl)) + tf)
            qf_component = ((self.k3 + 1) * qf) / (self.k3 + qf)
            term_score = idf * tf_component * qf_component
            score += term_score
        return score

class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'s': 0.2}) -> None:
        super().__init__(index, parameters)
        self.s = parameters.get('s', 0.2)

    def prepare(self, query_word_counts, N, avgdl, collection_length, term_stats):
        super().prepare(query_word_counts, N, avgdl, collection_length, term_stats)
        self.idf_values = {}
        for term in self.query_word_counts:
            df = self.term_stats[term]['df']
            # 避免 df 为零
            df = df if df > 0 else 1
            idf = math.log((N + 1) / df)
            self.idf_values[term] = idf

    def _score(self, docid: int, doc_word_counts: dict[str, int], doc_length: int = None, **kwargs) -> float:
        if doc_length is None:
            doc_length = self.index.get_doc_metadata(docid)['length']
        avgdl = self.avgdl
        s = self.s

        matching_terms = self.query_terms & doc_word_counts.keys()

        normalization_factor = 1 - s + s * (doc_length / avgdl)
        score = 0.0
        for term in matching_terms:
            qf = self.query_word_counts[term]
            tf = doc_word_counts[term]
            idf = self.idf_values[term]
            tf_weight = (1 + math.log(1 + math.log(tf))) / normalization_factor
            term_score = qf * tf_weight * idf
            score += term_score
        return score

class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        super().__init__(index, parameters)
        self.mu = parameters.get('mu', 2000)

    def prepare(self, query_word_counts, N, avgdl, collection_length, term_stats):
        super().prepare(query_word_counts, N, avgdl, collection_length, term_stats)
        self.cf_values = {}
        for term in self.query_word_counts:
            cf = self.term_stats[term]['cf']
            self.cf_values[term] = cf
        self.probabilities = {}
        for term, cf in self.cf_values.items():
            # 避免 cf 为零
            cf = cf if cf > 0 else 1
            self.probabilities[term] = cf / self.collection_length

    def _score(self, docid: int, doc_word_counts: dict[str, int], doc_length: int = None, **kwargs) -> float:
        if doc_length is None:
            doc_length = self.index.get_doc_metadata(docid)['length']

        mu = self.mu
        score = 0.0

        for term in self.query_terms:
            q_count = self.query_word_counts[term]
            tf = doc_word_counts.get(term, 0)
            cf = self.cf_values[term]
            probability = self.probabilities[term]
            second_part = math.log(1 + tf / (mu * probability))
            score += q_count * second_part

        score += sum(self.query_word_counts.values()) * math.log(mu / (doc_length + mu))

        return score

class YourRanker(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2}) -> None:
        super().__init__(index, parameters)
        self.b = parameters.get('b', 0.75)
        self.k1 = parameters.get('k1', 1.2)

    def prepare(self, query_word_counts, N, avgdl, collection_length, term_stats):
        super().prepare(query_word_counts, N, avgdl, collection_length, term_stats)
        self.idf_values = {}
        for term in self.query_word_counts:
            df = self.term_stats[term]['df']
            # 避免 df 为零
            df = df if df > 0 else 1
            idf = math.log((N + 1) / (df + 1))
            self.idf_values[term] = idf

    def _score(self, docid: int, doc_word_counts: dict[str, int], doc_length: int = None, **kwargs) -> float:
        if doc_length is None:
            doc_length = self.index.get_doc_metadata(docid)['length']
        avgdl = self.avgdl

        score = 0.0

        for term in self.query_terms:
            qf = self.query_word_counts[term]
            tf = doc_word_counts.get(term, 0)
            if tf == 0:
                continue

            idf = self.idf_values[term]

            numerator = (self.k1 + 1) * tf
            denominator = self.k1 * ((1 - self.b) + self.b * (doc_length / avgdl)) + tf
            tf_component = numerator / denominator if denominator != 0 else 0

            positions = self.index.get_term_positions(docid, term)
            if positions:
                first_position = positions[0]
                position_weight = 1 / (first_position + 1)
            else:
                position_weight = 0

            term_score = idf * tf_component * position_weight
            score += term_score

        return score
