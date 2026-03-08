# Design Notes

## How It Works

```
Evidence (supports / attacks)
        ↓
   Belief Node
        ↓
 TruthState (TRUE / FALSE / BOTH / NEITHER)
        ↓
 Confidence + Temporal Decay
        ↓
    Agent API
```

**TruthState** uses Belnap's four-valued logic. Instead of overwriting on conflict, the system represents the contradiction explicitly with `BOTH` — then lets you resolve it with new evidence.

**Evidence Ledger** is append-only. Evidence is never deleted, only invalidated. Every belief carries a full justification chain: what supports it, what attacks it, and what has expired.

**Temporal Decay** degrades evidence weight by belief type:

| BeliefType | Half-life |
|------------|-----------|
| FACT | 365 days |
| PREFERENCE | 90 days |
| INFERENCE | 30 days |
| PREDICTION | 3 days |

## Formal Model

MnemeBrain is grounded in two well-established theories from knowledge representation and belief revision:

- **Belnap four-valued logic (1977)** — used to represent contradictory evidence without collapsing the belief system. Instead of overwriting, the system holds `BOTH` as a valid, stable state.
- **AGM belief revision (Alchourrón, Gärdenfors, Makinson, 1985)** — defines how a rational agent updates beliefs when new evidence arrives, with minimal disturbance to existing knowledge.

**TruthState** is computed over the evidence ledger using Belnap's lattice:

```
TruthState ∈ { TRUE, FALSE, BOTH, NEITHER }

TRUE     — net supporting evidence dominates
FALSE    — net attacking evidence dominates
BOTH     — significant supporting AND attacking evidence (contradiction)
NEITHER  — insufficient evidence to determine
```

**Confidence** is derived from weighted, time-decayed evidence:

```
confidence = Σ(support_weight × decay(t)) / (Σ(support_weight × decay(t)) + Σ(attack_weight × decay(t)))
```

where `decay(t) = 0.5 ^ (t / half_life)` and `half_life` varies by belief type (3 days for PREDICTION → 365 days for FACT).

**Belief ranking** uses a composite score across three signals:

```
rank_score = 0.60 × similarity        # semantic relevance to query
           + 0.25 × confidence        # evidence strength
           + 0.15 × stability         # inverse of revision volatility
```

Stability is `1 / (1 + revision_count)` — beliefs that have been revised frequently rank lower than beliefs that have been stable, even at equal confidence. This prevents contradicted high-confidence beliefs from polluting retrieval.

**Revision policy** follows AGM minimal change: when new evidence contradicts an existing belief, the system retracts the minimum set of evidence necessary to restore consistency. Pluggable policies (recency, confidence-weighted, entrenchment-based) determine selection order.

**Counterfactual reasoning** uses copy-on-write sandbox isolation: hypothetical evidence is applied to a forked belief graph, leaving the canonical state unchanged.

## References

- Belnap, N. D. (1977). A useful four-valued logic. In *Modern Uses of Multiple-Valued Logic*. Reidel.
- Alchourrón, C. E., Gärdenfors, P., & Makinson, D. (1985). On the logic of theory change: Partial meet contraction and revision functions. *Journal of Symbolic Logic*, 50(2), 510–530.
- Lewis, D. (1973). *Counterfactuals*. Harvard University Press.
- Gutierrez, B. J., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*.
