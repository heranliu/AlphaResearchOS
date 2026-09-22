# Roadmap

AlphaResearchOS develops around a complete research cycle: propose, evaluate,
review, retain the evidence, and choose the next experiment. The items below are
planned work, ordered by their expected effect on research quality and usability.

## Research quality

- **Evidence-rich reviews.** Give the reviewer compact parent comparisons,
  execution costs, training sample counts, and feature transformations. Record
  which evidence informed each review.
- **Baseline-aware selection.** Add configurable development-stage admission
  rules and a baseline option, with per-fold excess return and turnover criteria.
- **Controlled experiments.** Run paired comparisons across markets and time
  windows with fixed data, costs, and call budgets. Evaluate review, memory, and
  parent-selection mechanisms separately.

## Research memory and efficiency

- **Structured retrieval.** Rank prior work by hypothesis, operator structure,
  evaluation context, and observed failure modes; measure retrieval effects in
  paired experiments.
- **Reusable computation.** Cache validated feature values by expression, data
  identity, and implementation version while keeping each evaluation cutoff
  explicit.
- **Provider comparison.** Record candidate quality, latency, token usage, and
  review outcomes under the same experiment budget.

## Agent interoperability and collaboration

- **MCP evidence tools.** Expose run discovery, candidate inspection, lineage,
  and verification through a typed read-only interface for external agents.
- **Research briefs.** Add versioned, machine-readable experiment specifications
  with data references, hypotheses, constraints, baselines, and budget settings.
- **Shared experiment views.** Add bilingual navigation and portable comparison
  reports that retain the protocol and provenance alongside every metric.

See [CONTRIBUTING.md](CONTRIBUTING.md) to propose a focused implementation or an
experiment that measures one of these improvements.
