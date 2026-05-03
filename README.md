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

- `POST /predict`
- `POST /explain`
- `POST /validate`
- `GET /health`

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_example.py
python scripts/evaluate_regression.py --n 1000
uvicorn api.main:app --reload
```

## Example Request

```json
{
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
}
```

## Hallucination Guardrails

The engine is intentionally extractive by default. A generated claim fails validation unless it cites a retrieved fact ID. Correlation-only drivers are allowed in explanations only if narrated as model attribution, not real-world causality. The included 1000-query regression suite targets `<1%` hallucination rate and writes metrics to `artifacts/regression_metrics.json`.

## Optional Production Extensions

- Replace `CausalInferenceLayer._adjusted_effect` with DoWhy/EconML estimators.
- Replace in-memory NetworkX facts with Neo4j and a vector database adapter.
- Add hosted LLM JSON-schema/function-calling behind `ConstrainedNarrationLayer`, then run validation as a hard gate.
- Add OpenTelemetry exporter in `core/logging.py` for distributed tracing.
