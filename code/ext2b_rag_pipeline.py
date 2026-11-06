import numpy as np


class FairRAGPipeline:
    def __init__(self, embedder, generator, steering_vector, steering_coeff: float = 1.5):
        self.retriever = embedder
        self.generator = generator
        self.steering_vector = steering_vector
        self.steering_coeff = steering_coeff

    def retrieve_context(self, query, corpus, k: int = 3):
        # Basic dense retrieval simulation
        query_emb = self.retriever.encode(query)
        scores = [np.dot(query_emb, doc.emb) for doc in corpus]
        top_indices = np.argsort(scores)[-k:]

        return [corpus[i].text for i in top_indices]

    def generate_response(self, query, corpus):
        # 1. Retrieve potentially biased documents
        context_docs = self.retrieve_context(query, corpus)
        context_str = " ".join(context_docs)

        prompt = f"Context: {context_str}\n\nQuestion: {query}\nAnswer:"

        # 2. Apply activation addition to the LLM forward pass
        # as described by Siddique et al.
        response = self.generator.generate(
            prompt,
            intervention_direction=self.steering_vector,
            alpha=self.steering_coeff
        )

        return response
