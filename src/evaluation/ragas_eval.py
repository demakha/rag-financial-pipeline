"""
RAGAS Evaluation: Automatically score RAG answer quality.
Using RAGAS 0.1.21 - stable, Python 3.9 compatible version.
"""

import os
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from dotenv import load_dotenv

load_dotenv()

# Set OpenRouter as the LLM backend for RAGAS
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'serving'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embedding'))

from retriever import retrieve_relevant_chunks
from rag_chain import generate_answer


def evaluate_rag(questions: list) -> dict:
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truths": []
    }

    print(f"Generating answers for {len(questions)} questions...")

    for question in questions:
        result = generate_answer(question)
        chunks = retrieve_relevant_chunks(question, top_k=5)
        contexts = [chunk['chunk_text'] for chunk in chunks]
        data["question"].append(question)
        data["answer"].append(result["answer"])
        data["contexts"].append(contexts)
        data["ground_truths"].append([""])

    dataset = Dataset.from_dict(data)

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    return results


if __name__ == "__main__":
    test_questions = [
        "What are Apple's main business risks?",
        "What products does Apple sell?",
        "How does Apple generate revenue?",
        "What is Apple's iPhone revenue in 2025?"
    ]

    results = evaluate_rag(test_questions)
    print("\n=== RAGAS Evaluation Results ===")
    print(results)