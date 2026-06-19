---
name: ds-business-insights
description: Use when translating model predictions into the answer the project's question actually asked — rankings, segment insights, recommendations, decision support. Aggregates predictions to the decision level, applies domain formulas when needed, cross-checks against a second source, and writes an honest conclusion (positive, negative, or mixed). Triggers when user mentions "insights", "conclusiones", "recomendaciones", "respuesta al negocio", "qué hacemos con el modelo".
---

# DS Business Insights

## Overview

Predictions are useless until translated into the answer the question asked. This skill closes
that loop for any domain: aggregate predictions to the decision level (segment, geography, type,
time), apply any domain formula the question requires, cross-check with a second source when one
exists, and produce an honest conclusion — including negative or mixed ones. Domain-agnostic.

## When to use

- After the final model is locked (post-error-analysis, post-TEST eval)
- User says "insights" / "conclusiones" / "recomendaciones" / "qué responde el modelo"

Do NOT use:
- Before the model is locked (premature)
- For pure descriptive stats (that's EDA)

## Workflow

1. **Load the final model + TEST predictions** (only the held-out set).
2. **Aggregate to the decision level** the question is about (e.g. by segment, region, product,
   cohort, period). The unit of aggregation must match the unit of the decision.
3. **Apply domain formulas** if the question needs them (rates, yields, risk-adjusted scores,
   expected value, cost/benefit). DOCUMENT every parameter (assumed value + citation/justification).
4. **Cross-check with a second source** when available (user memory: dual-source cross-check).
   Use a held-out column, an external dataset, or an independent metric as a gate — not as a
   replacement. Report agreement (e.g. correlation) even when sources agree.
5. **Identify top-K recommendations + counter-examples** (cases the model is least confident on,
   or where external evidence contradicts it).
6. **Write an honest conclusion**: one label `{positive | negative | mixed}` + ≥3 evidences.
   Negative/mixed conclusions are valid and often stronger.

## Output spec

- `reports/insights.md`:
  ```markdown
  # Insights

  ## Conclusion: {positive | negative | mixed}
  Headline (1 sentence): "<the answer to the question, stated plainly>"

  ## Top-K recommendations
  | Rank | Unit (segment/region/...) | Predicted metric | Secondary metric | Notes |
  |---|---|---|---|---|

  ## Counter-examples
  - <case the model gets wrong or where evidence pushes back>

  ## Cross-check with second source
  <held-out column / external dataset / independent metric> → agreement = <stat>

  ## 3+ evidences
  1. ... 2. ... 3. ...

  ## Assumptions made
  - <formula parameter> = <value> (source / justification)
  ```
- `reports/recommendations.csv`: `rank,unit,predicted_metric,secondary_metric,confidence,evidence_ids`

## <EXTREMELY-IMPORTANT> Rules

1. **Honest conclusion.** If the answer is negative or "the model can't answer this reliably",
   say so. This is what the consigna evaluates as critical use of the model.
2. **Dual-source cross-check when applicable.** Two sources for the same quantity → both as a
   gate, not one replacing the other. Document it even if they agree.
3. **Counter-examples required.** A recommendation list with no counter-examples is suspicious;
   find at least one case the model gets wrong.
4. **Document every assumption.** Each formula parameter has a citation or explicit justification.
5. **TEST set, not val.** Aggregations come from held-out TEST predictions. Don't leak.

## Auto-review before handoff

1. `insights.md` has exactly one `Conclusion: {positive|negative|mixed}` label
2. ≥3 evidences, each traceable to an EDA finding id or a model metric
3. ≥1 counter-example (the model's blind spot)
4. Cross-check documented (or explicitly "no second source available" with reason)
5. Every formula parameter has a citation or assumption note

## Red flags

| Thought | Reality |
|---|---|
| "Top-10 looks great, ship it" | Where are the counter-examples? Every model has cases it misses. |
| "Mixed sounds boring" | Mixed is honest. Don't manufacture a stronger conclusion. |
| "Use a round number for the parameter" | Every parameter needs a citation or stated assumption. |
| "No second source, skip cross-check" | Look harder — a held-out column often works as a cross-check. |
| "Negative conclusion looks bad" | A well-defended negative conclusion is excellent. Rigor > optimism. |
