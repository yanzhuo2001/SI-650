import string
import spacy
from nltk.tokenize import RegexpTokenizer as NLTKRegexpTokenizer
from spacy.symbols import ORTH

class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions if multiword_expressions is not None else []
        # Build punctuation table (excluding apostrophes)
        self.punctuation_table = str.maketrans('', '', string.punctuation.replace("'", ''))
        # Build Trie if MWEs are provided
        if self.multiword_expressions:
            self.mwe_trie = self.build_trie(self.multiword_expressions)
        else:
            self.mwe_trie = None

    def build_trie(self, expressions: list[str]) -> dict:
        trie = {}
        for expr in expressions:
            # Remove punctuation (excluding apostrophes) from MWEs
            expr_cleaned = expr.translate(self.punctuation_table)
            if self.lowercase:
                parts = expr_cleaned.lower().split()
            else:
                parts = expr_cleaned.split()
            node = trie
            for part in parts:
                node = node.setdefault(part, {})
            node['__end__'] = '__end__'
        return trie

    def match_multiword_expressions(self, tokens: list[str], original_tokens: list[str]) -> list[str]:
        result = []
        i = 0
        while i < len(tokens):
            if self.mwe_trie is None:
                break
            node = self.mwe_trie
            j = i
            last_match = None
            while j < len(tokens):
                # Remove punctuation (excluding apostrophes) from token
                token_cleaned = tokens[j].translate(self.punctuation_table)
                if token_cleaned in node:
                    node = node[token_cleaned]
                    if '__end__' in node:
                        last_match = j
                    j += 1
                else:
                    break
            if last_match is not None:
                # Merge original tokens
                mwe = ' '.join(original_tokens[i:last_match + 1])
                result.append(mwe)
                i = last_match + 1
            else:
                result.append(original_tokens[i])
                i += 1
        # Append any remaining tokens
        if i < len(tokens):
            result.extend(original_tokens[i:])
        return result

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        original_tokens = input_tokens.copy()

        # Lowercase if needed
        if self.lowercase:
            tokens = [token.lower() for token in input_tokens]
        else:
            tokens = input_tokens.copy()

        if self.mwe_trie:
            tokens = self.match_multiword_expressions(tokens, original_tokens)
        else:
            tokens = original_tokens

        if self.lowercase:
            tokens = [token.lower() for token in tokens]

        return tokens

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')

class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        tokens = text.split()
        tokens = self.postprocess(tokens)
        return tokens

class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = r'\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        super().__init__(lowercase, multiword_expressions)
        self.tokenizer = NLTKRegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        original_tokens = tokens.copy()
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
        tokens = self.postprocess(tokens)
        return tokens

class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        super().__init__(lowercase, multiword_expressions)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger'])

        # Customize tokenizer to prevent splitting on apostrophes
        special_cases = set()
        for expr in self.multiword_expressions:
            # Extract words that contain apostrophes
            for word in expr.split():
                if "'" in word:
                    special_cases.add(word)

        for case in special_cases:
            # Add special cases to prevent splitting
            self.nlp.tokenizer.add_special_case(case, [{ORTH: case}])

    def tokenize(self, text: str) -> list[str]:
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        tokens = self.postprocess(tokens)
        return tokens
