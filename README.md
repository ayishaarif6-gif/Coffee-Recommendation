# AI Barista API

A FastAPI service that recommends a coffee drink from a natural-language message. The service combines TF-IDF text features with preferences inferred from the message, then runs a trained Keras neural network to select one of 12 drinks.

## Features

- One-parameter `GET /recommend` API
- Typo-tolerant flavor recognition, including common misspellings of vanilla, caramel, and chocolate
- Model and preprocessing artifacts loaded once during application startup
- Strict request validation and confidence scores
- Automated API and real-model tests
- Docker and Docker Compose support with a non-root runtime user and health check

## Requirements

- Python 3.12 for local development
- Docker Desktop or another Docker Engine for containerized execution

## Local setup

Create and activate a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Make a recommendation

The endpoint accepts one required query parameter named `message`:

```bash
curl --get http://127.0.0.1:8000/recommend \
  --data-urlencode "message=I want a vanilla flavoured iced coffee"
```

Example response:

```json
{
  "recommended_drink": "Iced Vanilla Latte",
  "confidence": 0.7140247225761414
}
```

You can also open an encoded URL directly:

```text
http://127.0.0.1:8000/recommend?message=I%20want%20a%20vanilla%20flavoured%20iced%20coffee
```

## Run the tests

Run the complete suite:

```bash
python -m pytest -q
```

Run only the API or real-model tests:

```bash
python -m pytest tests/test_api.py -v
python -m pytest tests/test_model.py -v
```

## Run with Docker Compose

Build and start the service:

```bash
docker compose up --build
```

To run it in the background:

```bash
docker compose up --build --detach
```

Test the containerized endpoint:

```bash
curl --get http://127.0.0.1:8000/recommend \
  --data-urlencode "message=I want a strong coffee without milk"
```

View status and logs:

```bash
docker compose ps
docker compose logs --follow
```

Stop and remove the service:

```bash
docker compose down
```

Set a different host port when port 8000 is already in use:

```bash
PORT=8080 docker compose up --build
```

## Project structure

```text
.
├── main.py                       # FastAPI application and routes
├── schema.py                     # Request and response validation
├── train_model.py                # Artifact loading and inference service
├── model/                        # Keras model and fitted preprocessors
├── tests/                        # API and real-model tests
├── coffee recommendation.ipynb  # Training/reference notebook
├── requirements.txt              # Pinned Python dependencies
├── Dockerfile                    # Production container image
└── compose.yaml                  # Local container orchestration
```

The application fails during startup with a descriptive error if a model artifact is missing or its feature dimensions do not match the trained 285-input, 12-output network.
