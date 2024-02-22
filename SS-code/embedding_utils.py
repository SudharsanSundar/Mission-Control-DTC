from sentence_transformers import SentenceTransformer


"""
Object that deals with efficiently creating relatively good text embeddings.
"""
class SBertEmbeddingModel():
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)
