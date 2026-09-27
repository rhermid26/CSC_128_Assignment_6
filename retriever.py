#	TF-IDF retrieval

# retriever.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge import DOCUMENTS

vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(DOCUMENTS)


def retrieve(question, top_k=2, threshold=0.10):
    """Return the most relevant chunks, or an empty list if none match."""
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, doc_vectors)[0]

    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

    hits = []
    for index, score in ranked[:top_k]:
        if score >= threshold:
            hits.append(DOCUMENTS[index])
    return hits


print(retrieve("what time does the library close on friday"))
##print(retrieve("how do I pay my tuition"))   # returns [] on purpose