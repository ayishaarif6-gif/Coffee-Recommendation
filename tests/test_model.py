from schema import RecommendationRequest
from train_model import (
    EXPECTED_INPUT_FEATURES,
    EXPECTED_OUTPUT_CLASSES,
    CoffeeRecommender,
)


def test_model_artifacts_and_prediction() -> None:
    recommender = CoffeeRecommender()

    assert recommender.model.input_shape[-1] == EXPECTED_INPUT_FEATURES
    assert recommender.model.output_shape[-1] == EXPECTED_OUTPUT_CLASSES
    assert len(recommender.label_encoder.classes_) == EXPECTED_OUTPUT_CLASSES

    drink, confidence = recommender.predict(
        RecommendationRequest(
            message="I want a strong coffee without milk",
        )
    )

    assert drink in recommender.label_encoder.classes_
    assert 0.0 <= confidence <= 1.0


def test_misspelled_iced_vanilla_request() -> None:
    recommender = CoffeeRecommender()

    drink, confidence = recommender.predict(
        RecommendationRequest(
            message="I want a vainlla falavoured iced coffee",
        )
    )

    assert drink == "Iced Vanilla Latte"
    assert 0.0 <= confidence <= 1.0
