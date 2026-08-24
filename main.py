from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, Query, Request

from schema import RecommendationRequest, RecommendationResponse
from train_model import CoffeeRecommender


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.recommender = CoffeeRecommender()
    yield
    del app.state.recommender


app = FastAPI(title="AI Barista API", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello World"}


@app.get("/recommend", response_model=RecommendationResponse)
def recommend(
    payload: Annotated[RecommendationRequest, Query()],
    request: Request,
) -> RecommendationResponse:
    drink, confidence = request.app.state.recommender.predict(payload)
    return RecommendationResponse(
        recommended_drink=drink,
        confidence=confidence,
    )
