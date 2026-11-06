class BiasEvaluator:
    def __init__(self, sensitivity: float, noise: float = 0.0):
        # Sensitivity represents how much the LLM is influenced by the embedder
        self.sensitivity = sensitivity
        self.noise = noise

    def calculate_rag_bias(self, embedder_bias: float, llm_bias: float) -> float:
        """
        Calculates the overall RAG system bias based on the linear formula.
        Rb = s * Eb + Lb + noise
        """
        rag_bias = (self.sensitivity * embedder_bias) + llm_bias + self.noise
        return rag_bias

    def find_optimal_embedder_bias(self, llm_bias: float) -> float:
        """
        Finds the required 'reverse bias' for the embedder to achieve a neutral RAG (Rb = 0).
        0 = s * Eb + Lb + noise  =>  Eb = -(Lb + noise) / s
        """
        if self.sensitivity == 0:
            raise ValueError("Sensitivity cannot be zero when trying to reverse bias the embedder.")

        optimal_bias = -(llm_bias + self.noise) / self.sensitivity
        print(f"Target reverse bias for the embedder to neutralize the system: {optimal_bias:.4f}")
        return optimal_bias


# Example usage for a highly sensitive RAG pipeline
evaluator = BiasEvaluator(sensitivity=0.95, noise=0.05)

# Assuming the Mistral-7b-instruct (steered) still retains a small bias of 0.20
generator_residual_bias = 0.20

# Calculate what bias the embedder needs to completely fix the RAG pipeline
target_embedder_bias = evaluator.find_optimal_embedder_bias(llm_bias=generator_residual_bias)