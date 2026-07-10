"""
Layer 2: Semantic Classifier
Only invoked when structural confidence < 0.85.
Uses sentence-transformers for zero-token semantic understanding.
"""
from typing import Dict, List, Tuple
from models.enums import TaskCategory
from core.logger import setup_logger

logger = setup_logger("semantic_classifier")

# Category exemplars: representative sentences for each task type.
# These are encoded ONCE at startup and cached.
CATEGORY_EXEMPLARS: Dict[TaskCategory, List[str]] = {
    TaskCategory.MATH: [
        "Calculate the result of this arithmetic expression",
        "Solve this math problem step by step",
        "What is the total when you add these numbers",
        "Compute the average of the following values",
    ],
    TaskCategory.SENTIMENT: [
        "What is the sentiment of this review",
        "Is this text positive, negative or neutral",
        "Classify the emotional tone of this sentence",
        "Determine whether this customer review is positive or negative",
    ],
    TaskCategory.SUMMARIZATION: [
        "Summarize this article in a few sentences",
        "Provide a brief summary of the following text",
        "Give me the key points from this passage",
        "Condense this document into a short paragraph",
    ],
    TaskCategory.NER: [
        "Extract all named entities from this text",
        "Find all people, places and organizations mentioned",
        "Identify the named entities in the following passage",
        "List all the locations and persons in this text",
    ],
    TaskCategory.DEBUGGING: [
        "Find and fix the bug in this code",
        "What is wrong with this Python function",
        "Debug this code and explain the error",
        "Identify the issue in this program",
    ],
    TaskCategory.LOGIC: [
        "Solve this logic puzzle step by step",
        "Use deductive reasoning to find the answer",
        "Given these constraints determine who has what",
        "Apply logical deduction to this problem",
    ],
    TaskCategory.CODE_GEN: [
        "Write a Python function that does this",
        "Generate code to accomplish this task",
        "Create a script that performs the following",
        "Implement this algorithm in Python",
    ],
    TaskCategory.FACTUAL: [
        "What is the capital city of France",
        "Tell me about the history of the internet",
        "Who invented the telephone",
        "What year did World War 2 end",
    ],
}

class SemanticClassifier:
    def __init__(self):
        self._model = None
        self._exemplar_embeddings: Dict[TaskCategory, any] = {}
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            # Pre-compute and cache all exemplar embeddings at startup
            for category, sentences in CATEGORY_EXEMPLARS.items():
                embeddings = self._model.encode(sentences)
                self._exemplar_embeddings[category] = np.mean(embeddings, axis=0)
            self._available = True
            logger.info(f"SemanticClassifier loaded. Exemplars cached for {len(self._exemplar_embeddings)} categories.")
        except ImportError:
            logger.warning("sentence-transformers not installed. Semantic layer disabled.")
        except Exception as e:
            logger.error(f"SemanticClassifier failed to load: {e}")

    def classify(self, prompt: str) -> List[Tuple[TaskCategory, float]]:
        """
        Returns a sorted list of (category, probability) tuples.
        """
        if not self._available:
            return []

        import numpy as np

        prompt_embedding = self._model.encode([prompt])[0]

        scores: Dict[TaskCategory, float] = {}
        for category, exemplar_embedding in self._exemplar_embeddings.items():
            # Cosine similarity
            cosine_sim = float(
                np.dot(prompt_embedding, exemplar_embedding) /
                (np.linalg.norm(prompt_embedding) * np.linalg.norm(exemplar_embedding) + 1e-8)
            )
            # Normalize to [0,1]
            scores[category] = (cosine_sim + 1.0) / 2.0

        # Softmax normalization across all categories
        total = sum(scores.values())
        probabilities = {cat: score / total for cat, score in scores.items()}

        return sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
