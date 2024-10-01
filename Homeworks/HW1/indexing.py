import os
from enum import Enum
from document_preprocessor import Tokenizer, RegexTokenizer
from collections import Counter, defaultdict
import gzip
import json
from tqdm import tqdm


class IndexType(Enum):
    PositionalIndex = 'PositionalInvertedIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    InvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        self.statistics = {}
        self.statistics['vocab'] = Counter()
        self.vocabulary = set()
        self.document_metadata = {}
        self.index = defaultdict(list)

    def remove_doc(self, docid: int) -> None:
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str], original_doc_length: int = None) -> None:
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        raise NotImplementedError

    def get_document_word_counts(self, docid: int) -> dict[str, int]:
        raise NotImplementedError

    def get_term_positions(self, docid: int, term: str) -> list[int]:
        raise NotImplementedError

    def filter_terms(self, filtered_terms: set[str]) -> None:
        terms_to_remove = set(self.index.keys()) - filtered_terms
        for term in terms_to_remove:
            del self.index[term]
            if term in self.vocabulary:
                self.vocabulary.remove(term)
            if term in self.statistics['vocab']:
                del self.statistics['vocab'][term]


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.document_count = 0
        self.doc_term_frequencies = defaultdict(dict)
        self.index = defaultdict(list)

    def add_doc(self, docid: int, tokens: list[str], original_doc_length: int = None) -> None:
        if original_doc_length is None:
            original_doc_length = len(tokens)
        term_freq = Counter(tokens)
        self.doc_term_frequencies[docid] = term_freq

        for term, freq in term_freq.items():
            if term is not None:
                self.index[term].append((docid, freq))
                self.vocabulary.add(term)
                self.statistics['vocab'][term] += freq

        unique_terms = set(tokens)
        self.document_metadata[docid] = {
            'length': original_doc_length,
            'unique_tokens': len(unique_terms)
        }

        self.document_count += 1

    def remove_doc(self, docid: int) -> None:
        if docid not in self.document_metadata:
            return

        doc_terms = []
        for term, postings in self.index.items():
            for posting in postings:
                if posting[0] == docid:
                    doc_terms.append(term)
                    break

        for term in doc_terms:
            postings = self.index[term]
            self.index[term] = [p for p in postings if p[0] != docid]
            if not self.index[term]:
                del self.index[term]
                self.vocabulary.remove(term)
                del self.statistics['vocab'][term]

        del self.document_metadata[docid]
        del self.doc_term_frequencies[docid]
        self.document_count -= 1

    def get_postings(self, term: str) -> list:
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        term_count = self.statistics['vocab'].get(term, 0)
        doc_frequency = len(self.index.get(term, []))
        return {
            'term_count': term_count,
            'doc_frequency': doc_frequency
        }

    def get_statistics(self) -> dict[str, int]:
        total_token_count = sum(meta['length'] for meta in self.document_metadata.values())
        stored_total_token_count = sum(self.statistics['vocab'].values())
        mean_document_length = total_token_count / self.document_count if self.document_count > 0 else 0

        return {
            'unique_token_count': len(self.vocabulary),
            'total_token_count': total_token_count,
            'stored_total_token_count': stored_total_token_count,
            'number_of_documents': self.document_count,
            'mean_document_length': mean_document_length
        }

    def save(self, index_directory_name: str) -> None:
        if not os.path.exists(index_directory_name):
            os.makedirs(index_directory_name)

        index_path = os.path.join(index_directory_name, 'index.json')
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump({term: postings for term, postings in self.index.items()}, f)

        metadata_path = os.path.join(index_directory_name, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': self.statistics,
                'vocabulary': list(self.vocabulary),
                'document_metadata': self.document_metadata
            }, f)

    def load(self, index_directory_name: str) -> None:
        index_path = os.path.join(index_directory_name, 'index.json')
        metadata_path = os.path.join(index_directory_name, 'metadata.json')

        with open(index_path, 'r', encoding='utf-8') as f:
            self.index = defaultdict(list, json.load(f))

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.statistics = metadata['statistics']
            self.vocabulary = set(metadata['vocabulary'])
            self.document_metadata = metadata['document_metadata']
            self.document_count = self.statistics.get('number_of_documents', 0)

    def get_document_word_counts(self, docid: int) -> dict[str, int]:
        return self.doc_term_frequencies.get(docid, {})

    def get_term_positions(self, docid: int, term: str) -> list[int]:
        return []


class PositionalInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        super().__init__()
        self.statistics['index_type'] = 'PositionalInvertedIndex'
        self.document_count = 0
        self.doc_term_frequencies = defaultdict(dict)
        self.index = defaultdict(list)  # term -> list of (docid, term_freq, positions_list)

    def add_doc(self, docid: int, tokens: list[str], original_doc_length: int = None) -> None:
        if original_doc_length is None:
            original_doc_length = len(tokens)
        self.document_metadata[docid] = {
            'length': original_doc_length,
            'unique_tokens': len(set(tokens))
        }
        self.document_count += 1

        term_positions = defaultdict(list)
        for position, term in enumerate(tokens):
            if term is not None:
                term_positions[term].append(position)
                self.vocabulary.add(term)
                self.statistics['vocab'][term] += 1

        for term, positions in term_positions.items():
            postings_list = self.index[term]
            term_freq = len(positions)
            postings_list.append((docid, term_freq, positions))
            self.doc_term_frequencies[docid][term] = term_freq

    def remove_doc(self, docid: int) -> None:
        if docid not in self.document_metadata:
            return

        for term in list(self.vocabulary):
            postings = self.index.get(term, [])
            new_postings = [p for p in postings if p[0] != docid]
            if len(new_postings) != len(postings):
                if new_postings:
                    self.index[term] = new_postings
                else:
                    del self.index[term]
                    self.vocabulary.remove(term)
                    del self.statistics['vocab'][term]

        del self.document_metadata[docid]
        del self.doc_term_frequencies[docid]
        self.document_count -= 1

    def get_postings(self, term: str) -> list:
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        postings = self.index.get(term, [])
        term_count = sum(p[1] for p in postings)
        doc_frequency = len(postings)
        return {
            'term_count': term_count,
            'doc_frequency': doc_frequency
        }

    def get_statistics(self) -> dict[str, int]:
        total_token_count = sum(meta['length'] for meta in self.document_metadata.values())
        stored_total_token_count = sum(self.statistics['vocab'].values())
        mean_document_length = float(total_token_count) / float(self.document_count) if self.document_count > 0 else 0

        return {
            'unique_token_count': len(self.vocabulary),
            'total_token_count': total_token_count,
            'stored_total_token_count': stored_total_token_count,
            'number_of_documents': self.document_count,
            'mean_document_length': mean_document_length
        }

    def get_document_word_counts(self, docid: int) -> dict[str, int]:
        return self.doc_term_frequencies.get(docid, {})

    def get_term_positions(self, docid: int, term: str) -> list[int]:
        postings = self.get_postings(term)
        for posting in postings:
            if posting[0] == docid:
                return posting[2]
        return []

    def save(self, index_directory_name: str) -> None:
        if not os.path.exists(index_directory_name):
            os.makedirs(index_directory_name)

        index_path = os.path.join(index_directory_name, 'index.json')
        index_data = {term: [(docid, term_freq, positions) for docid, term_freq, positions in postings] for term, postings in self.index.items()}
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f)

        metadata_path = os.path.join(index_directory_name, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': self.statistics,
                'vocabulary': list(self.vocabulary),
                'document_metadata': self.document_metadata
            }, f)

    def load(self, index_directory_name: str) -> None:
        index_path = os.path.join(index_directory_name, 'index.json')
        metadata_path = os.path.join(index_directory_name, 'metadata.json')

        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
            self.index = defaultdict(list, {term: [(docid, term_freq, positions) for docid, term_freq, positions in postings] for term, postings in index_data.items()})

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.statistics = metadata['statistics']
            self.vocabulary = set(metadata['vocabulary'])
            self.document_metadata = metadata['document_metadata']
            self.document_count = self.statistics.get('number_of_documents', 0)


class Indexer:
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1) -> InvertedIndex:
        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            raise ValueError("Unsupported index type")

        term_global_freq = Counter()
        total_token_count = 0

        # 根据文件扩展名选择合适的打开方式
        def open_file(path):
            if path.endswith('.gz'):
                return gzip.open(path, 'rt', encoding='utf-8')
            else:
                return open(path, 'r', encoding='utf-8')

        total_docs = max_docs if max_docs > 0 else sum(1 for _ in open_file(dataset_path))

        doc_count = 0

        with open_file(dataset_path) as f:
            for line in tqdm(f, desc="Building index", total=total_docs):
                if max_docs > 0 and doc_count >= max_docs:
                    break
                doc = json.loads(line)
                docid = int(doc['docid'])
                text = doc.get(text_key, "")
                original_tokens = document_preprocessor.tokenize(text)
                doc_length = len(original_tokens)
                total_token_count += doc_length
                tokens = [token for token in original_tokens if token.lower() not in stopwords]
                term_global_freq.update(tokens)
                index.add_doc(docid, tokens, doc_length)
                doc_count += 1

        index.statistics['total_token_count'] = total_token_count

        if minimum_word_frequency > 0:
            filtered_terms = {term for term, freq in term_global_freq.items() if freq >= minimum_word_frequency}
            index.filter_terms(filtered_terms)

        return index


class SampleIndex(InvertedIndex):
    def add_doc(self, docid, tokens, original_doc_length):
        pass

    def save(self, index_directory_name: str) -> None:
        print('Index saved!')
