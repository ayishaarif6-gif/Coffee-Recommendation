from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

import main


VALID_REQUEST = {
    "message": "I want a strong coffee without milk",
}


class FakeRecommender:
    instances = 0

    def __init__(self) -> None:
        type(self).instances += 1

    def predict(self, payload: object) -> tuple[str, float]:
        return "Americano", 0.97


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    FakeRecommender.instances = 0
    monkeypatch.setattr(main, "CoffeeRecommender", FakeRecommender)
    with TestClient(main.app) as test_client:
        yield test_client


def test_recommendation_contract_and_service_reuse(client: TestClient) -> None:
    first_response = client.get("/recommend", params=VALID_REQUEST)
    second_response = client.get("/recommend", params=VALID_REQUEST)

    assert first_response.status_code == 200
    assert first_response.json() == {
        "recommended_drink": "Americano",
        "confidence": 0.97,
    }
    assert second_response.status_code == 200
    assert FakeRecommender.instances == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "   "},
        {},
        {**VALID_REQUEST, "temperature": "Hot"},
    ],
)
def test_invalid_requests_return_422(
    client: TestClient,
    payload: dict[str, str],
) -> None:
    response = client.get("/recommend", params=payload)

    assert response.status_code == 422
