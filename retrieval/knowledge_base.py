from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from core.schemas import RetrievedFact, RetrievalResult
from causal.engine import CRE_CAUSAL_EDGES

TOKEN_RE = re.compile(r"[a-zA-Z_]+")


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    text: str
    source: str
    related_features: tuple[str, ...]
    relationship: str


FACTS: tuple[FactRecord, ...] = (
    FactRecord("cre-001", "Higher capitalization rates mechanically lower income-property value multiples when net operating income is held constant.", "CRE valuation identity", ("cap_rate",), "causal"),
    FactRecord("cre-002", "Higher occupancy improves realized rental income and typically raises property value score.", "CRE operating fundamentals", ("occupancy_rate",), "causal"),
    FactRecord("cre-003", "Expected NOI growth supports higher valuation because buyers underwrite future income expansion.", "CRE underwriting principle", ("noi_growth",), "causal"),
    FactRecord("cre-004", "Higher interest rates increase financing costs and discount rates, pressuring values downward.", "Macro-finance CRE linkage", ("interest_rate",), "causal"),
    FactRecord("cre-005", "Higher unemployment can reduce tenant demand and weaken occupancy-sensitive asset performance.", "Labor market demand channel", ("unemployment_rate", "occupancy_rate"), "causal"),
    FactRecord("cre-006", "Population growth is a demand tailwind for many CRE sectors and can improve absorption.", "Market demand signal", ("population_growth",), "causal"),
    FactRecord("cre-007", "Transit access can raise submarket attractiveness but must not be treated as a value driver without local evidence.", "Location accessibility constraint", ("transit_score", "submarket_score"), "constraint"),
    FactRecord("cre-008", "Elevated crime rates can reduce submarket desirability and leasing demand.", "Location risk signal", ("crime_rate", "submarket_score"), "causal"),
    FactRecord("cre-009", "Older properties can face higher capital expenditure needs, reducing risk-adjusted value unless offset by renovation or location quality.", "Asset condition constraint", ("property_age",), "causal"),
    FactRecord("cre-010", "Longer weighted lease terms can stabilize cash flows and reduce near-term rollover risk.", "Lease risk principle", ("lease_term_months",), "causal"),
    FactRecord("cre-011", "Liquid transaction markets reduce exit risk and support pricing confidence.", "Capital markets liquidity", ("market_liquidity",), "causal"),
    FactRecord("cre-012", "A large competing supply pipeline can weaken rent growth and occupancy expectations.", "Supply-demand constraint", ("supply_pipeline", "occupancy_rate", "noi_growth"), "causal"),
    FactRecord("cre-013", "Submarket score aggregates local demand, access, safety, and liquidity signals; component claims need supporting evidence.", "CRE schema definition", ("submarket_score", "transit_score", "crime_rate"), "definition"),
    FactRecord("cre-014", "Narration must distinguish model attribution from causal effect: SHAP explains the model prediction, not real-world causality by itself.", "Explanation governance", tuple(), "constraint"),
)


def _tokens(text: str) -> Counter[str]:
    return Counter(t.lower() for t in TOKEN_RE.findall(text))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


class HybridKnowledgeRetriever:
    """Hybrid graph + lexical/vector retrieval over governed CRE facts."""

    def __init__(self, top_k: int = 6, lexical_weight: float = 0.55, graph_weight: float = 0.45):
        self.top_k = top_k
        self.lexical_weight = lexical_weight
        self.graph_weight = graph_weight
        self.facts = FACTS
        self.fact_tokens = {f.fact_id: _tokens(f.text + " " + " ".join(f.related_features)) for f in self.facts}
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(CRE_CAUSAL_EDGES)
        for f in self.facts:
            self.graph.add_node(f.fact_id, kind="fact")
            for feature in f.related_features:
                self.graph.add_edge(feature, f.fact_id)

    def retrieve(self, features: Iterable[str], context_terms: Iterable[str] = ()) -> RetrievalResult:
        query_features = list(dict.fromkeys(features))
        terms = query_features + list(context_terms)
        query = _tokens(" ".join(terms).replace("_", " "))
        scored: list[tuple[float, FactRecord]] = []
        for fact in self.facts:
            lexical = _cosine(query, self.fact_tokens[fact.fact_id])
            graph_score = self._graph_score(query_features, fact)
            score = self.lexical_weight * lexical + self.graph_weight * graph_score
            scored.append((score, fact))
        scored.sort(key=lambda item: item[0], reverse=True)
        facts = [
            RetrievedFact(
                fact_id=f.fact_id,
                text=f.text,
                source=f.source,
                related_features=list(f.related_features),
                relationship=f.relationship,  # type: ignore[arg-type]
                score=round(float(score), 6),
            )
            for score, f in scored[: self.top_k]
            if score > 0
        ]
        # Always include governance constraint to bound narration.
        if not any(f.fact_id == "cre-014" for f in facts):
            gov = next(f for f in self.facts if f.fact_id == "cre-014")
            facts.append(RetrievedFact(fact_id=gov.fact_id, text=gov.text, source=gov.source, related_features=[], relationship="constraint", score=1.0))
        return RetrievalResult(facts=facts, query_terms=terms)

    def _graph_score(self, query_features: list[str], fact: FactRecord) -> float:
        if not fact.related_features:
            return 0.1 if fact.relationship == "constraint" else 0.0
        direct = len(set(query_features) & set(fact.related_features)) / max(len(set(fact.related_features)), 1)
        neighbor = 0.0
        for feature in query_features:
            for related in fact.related_features:
                if feature == related:
                    continue
                if self.graph.has_edge(feature, related) or self.graph.has_edge(related, feature):
                    neighbor = max(neighbor, 0.5)
        return max(direct, neighbor)
