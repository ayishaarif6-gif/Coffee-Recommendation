from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock

import joblib
import keras
import numpy as np
import pandas as pd
from scipy import sparse

from schema import RecommendationRequest


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "model"

ATTRIBUTE_COLUMNS = (
    "temperature",
    "sweetness",
    "strength",
    "milk",
    "creaminess",
)
EXPECTED_INPUT_FEATURES = 285
EXPECTED_OUTPUT_CLASSES = 12


class CoffeeRecommender:
    """Load the AI Barista artifacts once and provide thread-safe inference."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._prediction_lock = Lock()

        paths = {
            "TF-IDF vectorizer": self.model_dir / "tfidf_vectorizer.joblib",
            "attribute encoder": self.model_dir / "attribute_encoder.joblib",
            "label encoder": self.model_dir / "label_encoder.joblib",
            "model config": self.model_dir / "ai_barista_ann" / "config.json",
            "model weights": self.model_dir
            / "ai_barista_ann"
            / "model.weights.h5",
        }
        self._ensure_artifacts_exist(paths)

        try:
            self.vectorizer = joblib.load(paths["TF-IDF vectorizer"])
            self.attribute_encoder = joblib.load(paths["attribute encoder"])
            self.label_encoder = joblib.load(paths["label encoder"])

            model_config = json.loads(paths["model config"].read_text(encoding="utf-8"))
            self.model = keras.models.model_from_json(json.dumps(model_config))
            self.model.load_weights(paths["model weights"])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load AI Barista artifacts from {self.model_dir}: {exc}"
            ) from exc

        self._validate_artifacts()

    @staticmethod
    def _ensure_artifacts_exist(paths: dict[str, Path]) -> None:
        missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
        if missing:
            raise RuntimeError("Missing AI Barista artifact(s): " + ", ".join(missing))

    def _validate_artifacts(self) -> None:
        try:
            text_features = len(self.vectorizer.get_feature_names_out())
            attribute_features = len(self.attribute_encoder.get_feature_names_out())
            combined_features = text_features + attribute_features

            fitted_columns = tuple(self.attribute_encoder.feature_names_in_)
            input_features = int(self.model.input_shape[-1])
            output_classes = int(self.model.output_shape[-1])
            encoded_classes = len(self.label_encoder.classes_)
        except Exception as exc:
            raise RuntimeError(f"Invalid AI Barista artifact metadata: {exc}") from exc

        if fitted_columns != ATTRIBUTE_COLUMNS:
            raise RuntimeError(
                "Attribute encoder columns do not match the required order: "
                f"expected {ATTRIBUTE_COLUMNS}, got {fitted_columns}"
            )

        if combined_features != EXPECTED_INPUT_FEATURES:
            raise RuntimeError(
                "Preprocessor feature count does not match the trained model: "
                f"expected {EXPECTED_INPUT_FEATURES}, got {combined_features}"
            )

        if input_features != EXPECTED_INPUT_FEATURES:
            raise RuntimeError(
                "Model input size is invalid: "
                f"expected {EXPECTED_INPUT_FEATURES}, got {input_features}"
            )

        if (
            output_classes != EXPECTED_OUTPUT_CLASSES
            or encoded_classes != EXPECTED_OUTPUT_CLASSES
        ):
            raise RuntimeError(
                "Model output size and label encoder classes must both equal "
                f"{EXPECTED_OUTPUT_CLASSES}; got model={output_classes}, "
                f"labels={encoded_classes}"
            )

    def predict(self, request: RecommendationRequest) -> tuple[str, float]:
        attributes = pd.DataFrame(
            [self._infer_attributes(request.message)],
            columns=ATTRIBUTE_COLUMNS,
        )

        text_features = self.vectorizer.transform([request.message])
        attribute_features = self.attribute_encoder.transform(attributes)
        features = sparse.hstack(
            (text_features, attribute_features),
            format="csr",
            dtype=np.float32,
        )

        if features.shape != (1, EXPECTED_INPUT_FEATURES):
            raise RuntimeError(
                "Unexpected inference feature shape: "
                f"expected (1, {EXPECTED_INPUT_FEATURES}), got {features.shape}"
            )

        with self._prediction_lock:
            output = self.model(features.toarray(), training=False)

        probabilities = np.asarray(output, dtype=np.float64)
        if probabilities.shape != (1, EXPECTED_OUTPUT_CLASSES):
            raise RuntimeError(
                "Unexpected model output shape: "
                f"expected (1, {EXPECTED_OUTPUT_CLASSES}), got {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model returned a non-finite confidence value")

        class_index = int(np.argmax(probabilities[0]))
        recommended_drink = str(
            self.label_encoder.inverse_transform([class_index])[0]
        )
        confidence = float(probabilities[0, class_index])
        return recommended_drink, confidence

    @staticmethod
    def _infer_attributes(message: str) -> dict[str, str]:
        """Infer the model's required categorical features from natural language."""
        text = message.lower()
        vanilla = CoffeeRecommender._contains_fuzzy_word(text, "vanilla")
        caramel = CoffeeRecommender._contains_fuzzy_word(text, "caramel")
        chocolate = CoffeeRecommender._contains_fuzzy_word(text, "chocolate")

        temperature = (
            "Iced" if re.search(r"\b(iced|cold|refreshing)\b", text) else "Hot"
        )

        if re.search(
            r"\b(no sugar|without sugar|not sweet)\b|don'?t want (?:a )?sweet",
            text,
        ):
            sweetness = "Low"
        elif re.search(r"\bnot too sweet\b", text):
            sweetness = "Medium"
        elif (
            vanilla
            or caramel
            or chocolate
            or re.search(r"\b(sweet|chocolatey)\b", text)
        ):
            sweetness = "High"
        else:
            sweetness = "Medium"

        if re.search(r"\b(not too strong|light|mild)\b", text):
            strength = "Mild"
        elif re.search(r"\b(strong|wake me|wake up)\b", text):
            strength = "Strong"
        elif vanilla:
            strength = "Mild"
        else:
            strength = "Medium"

        milk = (
            "No"
            if re.search(r"\b(no milk|without milk|black coffee|black)\b", text)
            else "Yes"
        )

        if milk == "No" or re.search(
            r"\b(no cream|without cream|not creamy)\b",
            text,
        ):
            creaminess = "Low"
        elif vanilla or caramel or chocolate or re.search(
            r"\b(creamy|cream|milky|milk|foam|foamy|latte|cappuccino)\b", text
        ):
            creaminess = "High"
        else:
            creaminess = "Medium"

        return {
            "temperature": temperature,
            "sweetness": sweetness,
            "strength": strength,
            "milk": milk,
            "creaminess": creaminess,
        }

    @staticmethod
    def _contains_fuzzy_word(text: str, expected: str, cutoff: float = 0.82) -> bool:
        """Match a flavor word while tolerating small user spelling mistakes."""
        words = re.findall(r"[a-z]+", text)
        return any(
            word == expected
            or (
                len(word) >= 4
                and SequenceMatcher(None, word, expected).ratio() >= cutoff
            )
            for word in words
        )
