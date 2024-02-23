# TODO: Fill out!

import faiss

class FaissIndex():
    def __init__(self, model, embeddings, k=5):
        self.embeddings = np.array(embeddings)
        self.index = None
        self.k = k
        self.model = model


    def build(self):
        d = self.embeddings.shape[1]
		self.index = faiss.IndexFlatIP(d)
		self.index.add(self.embeddings)

    def search(self, text):
        embedding = self.model.encode(text)
        distances, indices = self.index.search(embedding, k=self.k)
        return distances, indices
