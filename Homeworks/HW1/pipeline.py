import os
import pickle
import json
import gzip
from document_preprocessor import SplitTokenizer, RegexTokenizer, SpaCyTokenizer
from indexing import Indexer, IndexType, InvertedIndex
from ranker import Ranker, BM25, WordCountCosineSimilarity, DirichletLM, PivotedNormalization, TF_IDF, YourRanker
from models import BaseSearchEngine, SearchResponse
import argparse

DATA_PATH = 'data/'
CACHE_PATH = '__cache__/'

STOPWORD_PATH = os.path.join(DATA_PATH, 'stopwords.txt')
DATASET_PATH = os.path.join(DATA_PATH, 'wikipedia_200k_dataset.jsonl.gz')
MWE_PATH = os.path.join(DATA_PATH, 'multi_word_expressions.txt')


class SearchEngine(BaseSearchEngine):
    def __init__(self, max_docs: int = -1, ranker: str = 'BM25', tokenizer_type: str = 'Regex') -> None:
        print('Initializing Search Engine...')
        self.stopwords = set()
        with open(STOPWORD_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                self.stopwords.add(line.strip())

        multiword_expressions = []
        with open(MWE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                expr = line.strip()
                if expr:
                    multiword_expressions.append(expr)

        print(f'Initializing {tokenizer_type} Tokenizer...')
        if tokenizer_type == 'Split':
            self.preprocessor = SplitTokenizer(
                lowercase=True,
                multiword_expressions=multiword_expressions
            )
        elif tokenizer_type == 'Regex':
            self.preprocessor = RegexTokenizer(
                token_regex=r'\w+',
                lowercase=True,
                multiword_expressions=multiword_expressions
            )
        elif tokenizer_type == 'SpaCy':
            self.preprocessor = SpaCyTokenizer(
                lowercase=True,
                multiword_expressions=multiword_expressions
            )
        else:
            raise ValueError("Invalid tokenizer type. Choose from 'Split', 'Regex', or 'SpaCy'.")

        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)

        if ranker == 'YourRanker':
            index_type = IndexType.PositionalIndex
        else:
            index_type = IndexType.BasicInvertedIndex

        index_cache_file = os.path.join(
            CACHE_PATH, f'index_{max_docs}_{tokenizer_type}_{index_type.value}.pkl')

        if os.path.exists(index_cache_file):
            print('Loading index from cache...')
            with open(index_cache_file, 'rb') as f:
                self.main_index = pickle.load(f)
        else:
            print('Creating indexes...')
            self.main_index = Indexer.create_index(
                index_type,
                dataset_path=DATASET_PATH,
                document_preprocessor=self.preprocessor,
                stopwords=self.stopwords,
                minimum_word_frequency=0,
                text_key='text',
                max_docs=max_docs
            )
            with open(index_cache_file, 'wb') as f:
                pickle.dump(self.main_index, f)
            print('Index saved to cache.')

        self.raw_text_dict = self.load_raw_texts(max_docs)

        self.set_ranker(ranker)
        print('Search Engine initialized!')

    def load_raw_texts(self, max_docs: int = -1) -> dict:
        raw_text_cache_file = os.path.join(
            CACHE_PATH, f'raw_text_{max_docs}.pkl')

        if os.path.exists(raw_text_cache_file):
            print('Loading raw texts from cache...')
            with open(raw_text_cache_file, 'rb') as f:
                raw_text_dict = pickle.load(f)
        else:
            print('Loading raw texts from dataset...')
            raw_text_dict = {}
            doc_count = 0
            with gzip.open(DATASET_PATH, 'rt', encoding='utf-8') as f:
                for line in f:
                    if max_docs > 0 and doc_count >= max_docs:
                        break
                    doc = json.loads(line)
                    docid = int(doc['docid'])
                    text = doc.get('text', "")
                    raw_text_dict[docid] = text
                    doc_count += 1
            with open(raw_text_cache_file, 'wb') as f:
                pickle.dump(raw_text_dict, f)
            print('Raw texts saved to cache.')
        return raw_text_dict

    def set_ranker(self, ranker: str = 'BM25') -> None:
        if ranker == 'BM25':
            self.scorer = BM25(self.main_index)
        elif ranker == "WordCountCosineSimilarity":
            self.scorer = WordCountCosineSimilarity(self.main_index)
        elif ranker == "DirichletLM":
            self.scorer = DirichletLM(self.main_index)
        elif ranker == "PivotedNormalization":
            self.scorer = PivotedNormalization(self.main_index)
        elif ranker == "TF_IDF":
            self.scorer = TF_IDF(self.main_index)
        elif ranker == "YourRanker":
            self.scorer = YourRanker(self.main_index)
        else:
            raise ValueError("Invalid ranker type")

        self.ranker = Ranker(
            index=self.main_index,
            document_preprocessor=self.preprocessor,
            stopwords=self.stopwords,
            scorer=self.scorer,
            raw_text_dict=self.raw_text_dict,
            top_k=100
        )

        self.pipeline = self.ranker

    def search(self, query: str) -> list[SearchResponse]:
        results = self.pipeline.query(query)
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]


def initialize(max_docs: int = -1, ranker: str = 'BM25', tokenizer_type: str = 'Regex'):
    search_obj = SearchEngine(max_docs=max_docs, ranker=ranker, tokenizer_type=tokenizer_type)
    return search_obj
