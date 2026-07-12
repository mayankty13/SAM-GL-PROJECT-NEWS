from __future__ import annotations

from typing import Dict


class NewsClassifierService:
    """Placeholder classifier that will later call your trained model."""

    def predict_category(
        self,
        headline: str,
        description: str,
        content: str,
        model_name: str | None = None,
    ) -> Dict[str, object]:
        category = "Technology"
        confidence = 0.91
        if model_name == "Advanced model":
            confidence = 0.94
            category = "Business"
        explanation = (
            "This placeholder result uses a mock classifier pipeline to demonstrate the UI experience. "
            "Replace this with your trained NLP model once ready."
        )
        return {
            "category": category,
            "confidence": confidence,
            "model_name": model_name or "Baseline model",
            "explanation": explanation,
        }


class GenAIClassifierService:
    """Placeholder LLM-based classifier that will later call your GenAI workflow."""

    def predict_category(self, headline: str, description: str, content: str) -> Dict[str, object]:
        return {
            "category": "Technology",
            "confidence": 0.96,
            "model_name": "GenAI classifier",
            "explanation": "This placeholder result simulates an LLM-based classifier for side-by-side comparison with conventional NLP models.",
        }


class NewsSummarizerService:
    """Placeholder summarizer that will later call your GenAI workflow."""

    def summarize(self, headline: str, description: str, content: str, max_words: int = 60) -> str:
        return (
            f"{headline} — This placeholder summary illustrates the final experience for concise, headline-style news summaries. "
            f"Replace it with your GenAI-powered summarization logic once the model is integrated."
        )


class PlaceholderAPIClient:
    """Simple placeholder for a production API layer."""

    def get_status(self) -> Dict[str, str]:
        return {
            "api_status": "Mock API connected",
            "pipeline_status": "Ready for model integration",
            "summary_status": "Ready for GenAI integration",
        }
