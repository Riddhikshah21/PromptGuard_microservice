from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer, util
import string

def cosine_similarity_tfidf(prompt1, prompt2):
    vectorizer = TfidfVectorizer().fit([prompt1, prompt2])
    vectors = vectorizer.transform([prompt1, prompt2])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]


# def sentence_transformer_similarity(prompt1, prompt2):
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     embeddings = model.encode([prompt1, prompt2])
#     similarity = util.cos_sim(embeddings, embeddings)
#     return similarity.item()


def jaccard_similarity(prompt1, prompt2):
    translator = str.maketrans('', '', string.punctuation)
    set1 = set(prompt1.lower().translate(translator).split())
    set2 = set(prompt2.lower().translate(translator).split())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)