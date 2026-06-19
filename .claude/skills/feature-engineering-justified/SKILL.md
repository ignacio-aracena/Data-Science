---
name: feature-engineering-justified
description: Use when building features for an academic DS model where EACH feature must be traceable to a specific EDA finding. Outputs reports/fe_decisions.json mapping finding_id → feature → expected_impact, plus train/val/test parquets and a fitted preprocessor. Enforces no-leakage discipline (fit on train only). Triggers when user mentions "feature engineering", "FE", "preparar features".
---

# Feature Engineering Justified

## Overview

The consigna prohibits "transformaciones aplicadas por defecto o sin vínculo explícito con lo observado en los datos". This skill enforces that. It reads `eda_findings.json` and builds ONE feature per finding that has an `fe_candidate`. Each feature in `fe_decisions.json` carries its `source_finding_id`. No source → no feature.

## When to use

- After EDA, before modeling
- User says "feature engineering" / "preparar features" / "armar dataset para entrenar"

Do NOT use:
- Without `reports/eda_findings.json` existing first
- For inference-time preprocessing on new data (use the fitted `preprocessor.joblib` instead)

## Workflow

1. **Read EDA findings**: `reports/eda_findings.json`
2. **Design one feature per finding** with `fe_candidate` non-null:
   - Numéricas continuas → scaling decision (Standard / MinMax / Yeo-Johnson based on EDA skew)
   - Categóricas baja cardinalidad → one-hot
   - Categóricas alta cardinalidad → top-K + Other (K chosen from EDA cardinality)
   - Categóricas muy alta cardinalidad (>100) → embedding learnable (dim = ceil(log2(N)))
   - Geo → distancia a punto de interés (centro / transporte / etc.)
   - Multi-hot lists (tags / features / categories) → top-K filtered by EDA correlation with target
   - Texto → keyword binarias (lista derivada del EDA)
3. **Split train/val/test** (60/20/20 estratificado por el target binarizado, o random_state=42 si no clasif)
4. **Fit transformers SOLO en train**:
   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.impute import SimpleImputer
   
   preprocessor = ColumnTransformer([
       ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_COLS),
       ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
   ])
   preprocessor.fit(X_train)  # NO val, NO test
   X_train_t = preprocessor.transform(X_train)
   X_val_t = preprocessor.transform(X_val)
   X_test_t = preprocessor.transform(X_test)
   ```
5. **Generate `fe_decisions.json`**:
   ```json
   [
     {
       "feature_name": "log_price",
       "source_finding_id": "F-03",
       "transformation": "log1p",
       "rationale": "EDA shows price has skew=2.4, log transform stabilizes variance",
       "expected_impact": "medium — helps linear baselines, neutral for tree models"
     }
   ]
   ```
6. **Sanity assertions**:
   - `assert all(f["source_finding_id"] is not None for f in fe_decisions)`
   - `assert (X_train.isna().sum() == 0).all()`
   - `assert (X_val.isna().sum() == 0).all()`
   - `assert (X_test.isna().sum() == 0).all()`
   - `assert preprocessor.named_transformers_["cat"].handle_unknown == "ignore"` (no crash on OOV at val/test)
   - `assert len(set(X_train.index) & set(X_val.index)) == 0` (no split leakage)
7. **Persist**:
   - `data/processed/{train,val,test}.parquet`
   - `models/preprocessor.joblib`
   - `reports/fe_decisions.json`

## Output spec

- `data/processed/train.parquet` (60% rows)
- `data/processed/val.parquet` (20% rows)
- `data/processed/test.parquet` (20% rows)
- `models/preprocessor.joblib` (sklearn `ColumnTransformer`)
- `reports/fe_decisions.json` (list of dicts with `source_finding_id` non-null)

## <EXTREMELY-IMPORTANT> Rules

1. **Source-finding traceability is mandatory.** Every feature in `fe_decisions.json` MUST have non-null `source_finding_id`. No exceptions.
2. **No-leakage discipline.** `fit` SOLO en train. `transform` en val/test. If using SMOTE later, also only on train AFTER preprocessor fit.
3. **OOV handling explicit.** Categóricas: `handle_unknown="ignore"` para no crashear en val/test con categorías nuevas.
4. **Imputation on train only.** `SimpleImputer.fit(X_train)`; los val/test heredan la mediana/moda de train.
5. **No duplicates between splits.** Verify with `assert len(set(X_train.index) & set(X_val.index)) == 0`.
6. **`FORBIDDEN_COLUMNS` excluded.** Read from EDA leakage check; never appear in features.

## Auto-review before handoff

Before declaring FE done, verify (per user feedback `feedback_revalidate_outputs`):
1. Every feature in `fe_decisions.json` has non-null `source_finding_id` (assert programmatically over the full list, not sampled)
2. Train/val/test parquets exist with row counts that sum to original dataset
3. `preprocessor.joblib` re-loadable: `joblib.load(...).transform(X_val.head(5))` works — safe here because the file is produced by this pipeline in the same session, not from an external/untrusted source
4. Zero NaNs in any of the three splits (full check, not `.head()`)
5. `FORBIDDEN_COLUMNS` from EDA leakage check are absent from all splits

If any check fails, halt and re-fit. Do NOT pass to baselines / NN with leakage or NaNs.

## Red flags

| Thought | Reality |
|---|---|
| "Just one-hot all categóricas" | Default. The consigna penalizes. Justify per feature. |
| "Imputed with median because pandas does it" | Need explicit per-column strategy traced to EDA. |
| "Fit preprocessor on full data, then split" | LEAKAGE. Fit on train only. |
| "SMOTE on full data then split" | LEAKAGE. SMOTE on train only after split. |
| "Feature looked useful, added it" | "Looked useful" is not a finding. Source-finding required. |
