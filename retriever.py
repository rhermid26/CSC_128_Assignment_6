import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge import DOCUMENTS


DEFAULT_THRESHOLD = 0.25
DEFAULT_TOP_K = 3


def stem(word):
    """Remove common endings from words."""

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
    """Lowercase, tokenize, stem, and add bigrams."""

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()

    words = [stem(word) for word in words]

    bigrams = [
        words[i] + "_" + words[i + 1]
        for i in range(len(words) - 1)
    ]

    return words + bigrams


class Retriever:

    def __init__(self, documents=DOCUMENTS, threshold=DEFAULT_THRESHOLD):
        self._documents = documents
        texts = [document["text"] for document in documents]
        self._vectorizer = TfidfVectorizer(
            analyzer=analyze,
            stop_words="english"
        )
        self._doc_vectors = self._vectorizer.fit_transform(texts)
        self._threshold = threshold


    def search(self, question, top_k=DEFAULT_TOP_K):
        question_vector = self._vectorizer.transform([question])
        scores = cosine_similarity(question_vector,  self._doc_vectors)[0]
        ranked = sorted(enumerate(scores),  key=lambda pair: pair[1],  reverse=True)
        hits = []
        for index, score in ranked[:top_k]:
            if score >= self._threshold:
                hits.append({
                    "id": self._documents[index]["id"],
                    "source": self._documents[index]["source"],
                    "text": self._documents[index]["text"],
                    "score": score
                })
        return hits


    def build_context(self, hits):

        if not hits:
            return ""

        context = []

        for hit in hits:

            context.append(
                f"Source: {hit['source']}\n"
                f"Fact: {hit['text']}"
            )

        return "\n\n".join(context)