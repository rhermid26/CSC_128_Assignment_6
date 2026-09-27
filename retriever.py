"""
CSC-128 Assignment 6 starter: retrieval
Your name here
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge import DOCUMENTS

STOP_WORDS = set()

# TODO 5: set this from the evidence your tests print
#DEFAULT_THRESHOLD = 0.0
DEFAULT_THRESHOLD = 0.05
DEFAULT_TOP_K = 3



def stem(word):
    """
    TODO 1: strip common suffixes so that reserve, reserved, and
    reservation collapse to the same token. It does not have to be a real
    Porter stemmer.
    """
    if word.endswith("ation"):
        return word[:-5]

    if word.endswith("ing"):
        return word[:-3]

    if word.endswith("ed"):
        return word[:-2]

    if word.endswith("s"):
        return word[:-1]

    return word


def analyze(text):
    """Normalize, tokenize, stem, and add bigrams."""
    # Make text lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Split into words
    words = text.split()
    # Add bigrams
    bigrams = [words[i] + "_" + words[i + 1]  for i in range(len(words) - 1)]
    return words + bigrams


class Retriever:
    def __init__(self, documents=DOCUMENTS, threshold=DEFAULT_THRESHOLD):
        """TODO 3: fit a TfidfVectorizer using your analyze function."""
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_vectors = self._vectorizer.fit_transform(documents)
        self._threshold = threshold

    def search(self, question, top_k=DEFAULT_TOP_K):
        """
        TODO 4: return a list of (document, score), best first, dropping
        anything below the threshold.

        Returning an empty list is correct and important. It is what tells
        the bot to refuse instead of calling the model with nothing useful.
        """

        """Return the most relevant chunks, or an empty list if none match."""
        question_vector = self._vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self._doc_vectors)[0]

        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

        hits = []
        for index, score in ranked[:top_k]:
            if score >= self._threshold:
                hits.append(DOCUMENTS[index])
        return hits

    def build_context(self, hits):
        """Format retrieved chunks for the prompt, with their sources."""

        if not hits:
            return ""

        context = []

        for hit in hits:
            context.append(f"Source: {hit}")

        return "\n\n".join(context)
