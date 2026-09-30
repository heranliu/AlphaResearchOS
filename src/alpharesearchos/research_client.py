"""Typed proposer/reviewer calls with development-only context and no retries."""
from __future__ import annotations

import json

from .proposals import llm_settings
from .provider_transport import chat_json


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


PROPOSAL_SCHEMA = object_schema({
    "name": {"type": "string"}, "hypothesis": {"type": "string"},
    "features": {"type": "array", "items": {"type": "string"}},
    "model": {"type": "string", "enum": ["rank", "ridge", "hist_gbdt"]},
    "model_params": object_schema({"alpha": {"type": "number"}, "train_window": {"type": "integer"},
                                    "retrain_every": {"type": "integer"}, "horizon": {"type": "integer"},
                                    "smoothing": {"type": "integer"}}),
    "rationale": {"type": "string"},
})
REVIEW_SCHEMA = object_schema({"decision": {"type": "string", "enum": ["approve", "revise", "reject"]},
                               "critique": {"type": "string"}, "risks": {"type": "array", "items": {"type": "string"}},
                               "suggested_action": {"type": "string"}})


def reservation():
    return 65536 if llm_settings().get("provider") == "codex_cli" else 24000


def compact_trial(trial):
    """Explicit allowlist: never forward holdout, raw data, or stored credentials."""
    result = {key: trial[key] for key in ["id", "name", "hypothesis", "features", "model", "model_params", "parents", "action", "score", "status", "reason", "review"] if key in trial}
    for key in ["name", "hypothesis", "reason"]:
        if isinstance(result.get(key), str):
            result[key] = result[key][:350]
    if isinstance(result.get("review"), dict):
        verdict = result["review"]
        result["review"] = {"decision": verdict.get("decision"), "critique": str(verdict.get("critique", ""))[:400],
                            "risks": [str(value)[:180] for value in verdict.get("risks", [])[:3]],
                            "suggested_action": str(verdict.get("suggested_action", ""))[:300]}
    result["development_metrics"] = {key: trial.get("metrics", {}).get(key) for key in
                                     ["sharpe", "total_return", "excess_return", "max_drawdown", "avg_turnover"]}
    result["development_folds"] = [{key: fold.get("metrics", {}).get(key) for key in
                                     ["sharpe", "excess_return", "max_drawdown", "avg_turnover"]} for fold in trial.get("folds", [])]
    return result


def structured_call(instruction, context, schema, *, timeout):
    settings = llm_settings()
    context = json.loads(json.dumps(context, ensure_ascii=False, allow_nan=False))
    prefix = instruction + "\nDo not use tools, browse, read files, or run commands. Only return the required JSON. Write explanations in Chinese.\n"
    prompt = prefix + json.dumps(context, ensure_ascii=False, allow_nan=False)
    # Preserve the current candidate/selected parents; oldest ancillary context
    # is removed deterministically before admission, never replaced by a guess.
    for key in ["development_history", "retrieved_trajectories"]:
        while len(prompt.encode()) > 18000 and context.get(key):
            context[key].pop(0)
            context["context_trimmed"] = True
            prompt = prefix + json.dumps(context, ensure_ascii=False, allow_nan=False)
    if len(prompt.encode()) > 22000:
        raise ValueError("Research context exceeds the 22 KB budget")
    if settings.get("provider") == "codex_cli":
        from .codex_provider import codex_structured
        return codex_structured(prompt, schema, model=settings.get("codex_model", ""), timeout=timeout)
    messages = [{"role": "system", "content": "Return a JSON object matching this schema: " + json.dumps(schema)},
                {"role": "user", "content": prompt}]
    return chat_json(settings, messages, completion_tokens=1800, timeout=timeout)


def validate_proposal(value):
    """Share the executable proposal contract with the explicit connection probe."""
    from .predictive import validate_strategy

    if not isinstance(value, dict) or set(value) != set(PROPOSAL_SCHEMA["properties"]):
        raise ValueError("Strategy response has missing or unexpected fields")
    for field, limit in [("name", 100), ("hypothesis", 1000), ("rationale", 1200)]:
        if not isinstance(value.get(field), str) or not 1 <= len(value[field]) <= limit:
            raise ValueError(f"Invalid strategy {field}")
    params = value.get("model_params")
    if not isinstance(params, dict) or set(params) != set(PROPOSAL_SCHEMA["properties"]["model_params"]["properties"]):
        raise ValueError("Strategy model parameters have missing or unexpected fields")
    validate_strategy(value)
    return value


def propose(context, *, timeout):
    instruction = "You are a quantitative research proposer. Design ONE testable strategy using the attached JSON schema. " \
        "DSL fields: close,open,high,low,volume. Functions: ret(x,n),delay(x,n),delta(x,n),mean(x,n),std(x,n),rank(x),zscore(x),abs(x),log(x),sign(x). " \
        "Arithmetic + - * /. Positive integer windows <=60; std uses ddof=0. No imports, indexing, attributes or future data. " \
        "Use 1-6 different modest OHLCV expressions as features. The executor cross-sectionally ranks features. " \
        "model=rank is equal-rank aggregation; ridge is pooled causal ridge regression; hist_gbdt is bounded histogram gradient boosting. " \
        "Training predicts cross-sectional future returns using only matured historical labels. Development folds and final evaluation freeze the fit before the evaluated block. " \
        "model_params bounds: alpha 0.000001..10000; train_window 63..1008; retrain_every 5..252; horizon 1..20; smoothing 1..20. " \
        "Default useful starting values are alpha=1, train_window=504, retrain_every=63,horizon=5,smoothing=3. Follow required_model when supplied. " \
        "Evolve the whole selected parent trajectory, including hypothesis/features/model parameters and reviewer feedback. For crossover combine complementary parent mechanisms. " \
        "Honor executable constraints. Seek stable excess returns over equal weight AFTER costs; investigate turnover and weak folds; do not merely repeat prior expressions. " \
        "No future test evidence is provided. Benchmark feedback is development-only."
    value, usage = structured_call(instruction, context, PROPOSAL_SCHEMA, timeout=timeout)
    return validate_proposal(value), usage


def connection_probe(*, timeout):
    """One proposal request, validated without data access or a research run."""
    return propose({"direction": "Connection protocol check only. Propose one simple 20-day price momentum rank strategy. "
                                 "No market evidence is supplied; do not claim measured returns.",
                    "required_model": "rank", "constraints": {}}, timeout=timeout)


def review(context, *, timeout):
    instruction = "You are an independent quantitative research reviewer in a fresh session. Review only supplied development evidence. " \
        "Check hypothesis-to-feature consistency, training feasibility, cost/turnover, fold stability, redundancy, and unsupported claims. " \
        "Hard validity checks have already run. Approve a technically sound experiment even when profitability is weak; do not claim alpha without positive development evidence. " \
        "Use revise for a concrete hypothesis/implementation mismatch and reject for invalid or unjustified claims that make selection unsafe. " \
        "Do not invent tests or observations. Suggest one falsifiable next research action. You cannot change measured metrics or waive constraints."
    value, usage = structured_call(instruction, context, REVIEW_SCHEMA, timeout=timeout)
    if not isinstance(value, dict) or set(value) != set(REVIEW_SCHEMA["properties"]):
        raise ValueError("Review response has missing or unexpected fields")
    if not isinstance(value.get("decision"), str) or value["decision"] not in {"approve", "revise", "reject"}:
        raise ValueError("Invalid reviewer decision")
    for field in ["critique", "suggested_action"]:
        if not isinstance(value.get(field), str) or not 1 <= len(value[field]) <= 2000:
            raise ValueError("Invalid reviewer explanation")
    if not isinstance(value.get("risks"), list) or len(value["risks"]) > 12 or any(not isinstance(s, str) or len(s) > 1000 for s in value["risks"]):
        raise ValueError("Invalid reviewer risks")
    return value, usage
