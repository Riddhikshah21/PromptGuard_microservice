from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from sentence_transformers import SentenceTransformer, util

def cosine_similarity_tfidf(prompt1, prompt2):
    """Cosine Similarity (Default) - Computes the cosine similarity between two input text prompts using TF-IDF (Term Frequency - Inverse Document Frequency) vectorization."""
    vectorizer = TfidfVectorizer().fit([prompt1, prompt2])
    vectors = vectorizer.transform([prompt1, prompt2])
    similarity = cosine_similarity(vectors[0], vectors[1])
    print(similarity[0][0])
    return similarity[0][0]


def jaccard_similarity(prompt1, prompt2):
    """Jaccard Similarity - Based on the intersection over union of tokens. Simpler calculation based on word overlap."""
    translator = str.maketrans('', '', string.punctuation)
    set1 = set(prompt1.lower().translate(translator).split())
    set2 = set(prompt2.lower().translate(translator).split())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)

            
# def sentence_similarity(prompt1, prompt2):
#     """Sentence Similarity - Computes the semantic similarity between two input text prompts using Sentence Transformers."""
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings1 = model.encode(prompt1, convert_to_tensor=True)
#     embeddings2 = model.encode(prompt2, convert_to_tensor=True)
#     similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
#     print(similarity.item())
#     return similarity.item()