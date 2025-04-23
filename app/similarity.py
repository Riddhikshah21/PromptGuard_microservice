from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


def cosine_similarity_tfidf(prompt1, prompt2):
    vectorizer = TfidfVectorizer().fit([prompt1, prompt2])
    vectors = vectorizer.transform([prompt1, prompt2])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]


def sentence_transformer_similarity(sentence1, sentence2):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_1 = model.encode(sentence1, convert_to_tensor=True)
    embedding_2 = model.encode(sentence2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return similarity.item()


def jaccard_similarity(prompt1, prompt2):
    set1 = set(prompt1.lower().split())
    set2 = set(prompt2.lower().split())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)