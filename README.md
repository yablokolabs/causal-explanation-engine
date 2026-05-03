# Causal Explanation Engine

Production-style Python service for grounded, low-hallucination explanations of Commercial Real Estate (CRE) model predictions.

The default repo runs fully offline: XGBoost prediction, SHAP attribution, adjusted causal-effect estimates, hybrid graph/vector retrieval, deterministic schema-constrained narration, and validation. Hosted LLM/DoWhy/EconML/CausalNex adapters can be added behind the existing interfaces without changing API contracts.

## Architecture

1. **SHAP attribution layer** (`core/attribution.py`) computes normalized top-k feature contributions using `shap.TreeExplainer`, with deterministic ablation fallback.
2. **Causal inference layer** (`causal/engine.py`) returns a CRE causal graph plus adjusted ATE/CATE-style estimates and correlation-vs-causation labels.
3. **Knowledge retrieval layer** (`retrieval/knowledge_base.py`) performs hybrid graph + lexical/vector retrieval against governed CRE facts and constraints.
4. **Context enrichment layer** (`core/enrichment.py`) adds deterministic market, macro, and location signals.
5. **Constrained narration layer** (`llm/narrator.py`) emits Pydantic-schema JSON only and builds every sentence from retrieved facts, causal estimates, and SHAP outputs.
6. **Validation layer** (`validation/validator.py`) checks SHAP recomposition, evidence grounding, and causal alignment before explanations are trusted.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Run model inference and return a prediction with metadata |
| `POST` | `/explain` | Generate a grounded causal explanation for a prediction |
| `POST` | `/validate` | Run deterministic validation on an explanation |
| `GET` | `/health` | Liveness probe |

## Quickstart

```bash
# Clone and install
git clone https://github.com/yablokolabs/causal-explanation-engine.git
cd causal-explanation-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run example pipeline (predict → explain → validate)
python scripts/run_example.py

# Run the 1000-query regression suite
python scripts/evaluate_regression.py --n 1000

# Start the API server
uvicorn api.main:app --reload
```

Or with **Make**:

```bash
make dev          # install with dev dependencies
make test         # run test suite
make lint         # ruff lint
make typecheck    # mypy strict check
make serve        # start API server
make regression   # run 1000-query regression
```

Or with **Docker**:

```bash
docker build -t causal-explanation-engine .
docker run -p 8000:8000 causal-explanation-engine
```

## Usage Examples

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "cap_rate": 0.061,
      "occupancy_rate": 0.92,
      "noi_growth": 0.031,
      "interest_rate": 0.052,
      "unemployment_rate": 0.041,
      "population_growth": 0.018,
      "transit_score": 72,
      "crime_rate": 18,
      "property_age": 21,
      "lease_term_months": 60,
      "market_liquidity": 0.69,
      "supply_pipeline": 0.034,
      "submarket_score": 74
    }
  }'
```

### Explain

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "cap_rate": 0.061,
      "occupancy_rate": 0.92,
      "noi_growth": 0.031,
      "interest_rate": 0.052,
      "unemployment_rate": 0.041,
      "population_growth": 0.018,
      "transit_score": 72,
      "crime_rate": 18,
      "property_age": 21,
      "lease_term_months": 60,
      "market_liquidity": 0.69,
      "supply_pipeline": 0.034,
      "submarket_score": 74
    },
    "top_k": 5
  }'
```

### Validate

Pass the full explanation object returned by `/explain`:

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"explanation": <paste explanation JSON here>}'
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## Configuration

Settings are loaded from `configs/default.yaml` by default. Override the config path or individual values with environment variables:

| Variable | Description |
|----------|-------------|
| `CEE_CONFIG` | Path to a YAML config file (default: `configs/default.yaml`) |
| `CEE_MODEL_ARTIFACT` | Override the model artifact path (validated for path safety) |
| `CEE_CORS_ORIGINS` | Comma-separated list of allowed CORS origins |

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only API tests
pytest tests/test_api.py

# Run the regression suite (targets <1% hallucination rate)
python scripts/evaluate_regression.py --n 1000
```

## Hallucination Guardrails

The engine is intentionally extractive by default. A generated claim fails validation unless it cites a retrieved fact ID. Correlation-only drivers are allowed in explanations only if narrated as model attribution, not real-world causality. The included 1000-query regression suite targets `<1%` hallucination rate and writes metrics to `artifacts/regression_metrics.json`.

## Optional Production Extensions

- Replace `CausalInferenceLayer._adjusted_effect` with DoWhy/EconML estimators.
- Replace in-memory NetworkX facts with Neo4j and a vector database adapter.
- Add hosted LLM JSON-schema/function-calling behind `ConstrainedNarrationLayer`, then run validation as a hard gate.
- Add OpenTelemetry exporter in `core/logging.py` for distributed tracing.
- Add API key / bearer token middleware for authentication.
