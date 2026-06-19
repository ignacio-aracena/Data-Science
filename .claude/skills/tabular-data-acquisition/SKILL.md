---
name: tabular-data-acquisition
description: Use when obtaining a tabular dataset (CSV/Parquet/Excel, Kaggle, public data portal, REST API, or SQL/warehouse) for a DS project, before EDA. Covers source identification, robust download (browser UA, retry/backoff), schema validation, and snapshot documentation for reproducibility. Triggers when user mentions "descargar dataset", "CSV", "Kaggle", "portal de datos", "API de datos", "snapshot", "fuente de datos tabular".
---

# Tabular Data Acquisition

## Overview

Get a tabular dataset onto disk and produce a validated, documented snapshot that downstream
phases can trust. The reusable discipline: robust download (some portals 403 the default Python
UA), immediate schema validation (halt loud on missing key columns), and a `SNAPSHOT_INFO.md`
so the analysis is reproducible. Domain-agnostic — works for any source.

## When to use

- A DS project needs raw tabular data from a file, Kaggle, a data portal, an API, or a DB
- An existing `data/raw/` is incomplete or stale (read `SNAPSHOT_INFO.md` first)

Do NOT use:
- For image datasets (use [[image-dataset-acquisition]])
- If the data is already validated and documented

## Workflow

1. **Identify the source** and the most recent snapshot/version. Record the exact URL/slug/query.
2. **Download robustly.**
   - HTTP files: use a browser `User-Agent` (many portals 403 the default `python-requests` UA)
     and retry with exponential backoff (e.g. 3 retries: 2s/4s/8s).
   - Kaggle: `kaggle datasets download -d <slug> --unzip` (needs `~/.kaggle/kaggle.json`, chmod 600).
   - SQL/warehouse: parameterized query; save the result + the query text.
3. **Validate schema immediately** (halt loud on failure):
   - required key columns present
   - row count ≥ a sane minimum for the problem
   - no critical nulls in key columns (id / target / join keys)
4. **Identify external/secondary sources** only if justified by the question (cross-checks,
   enrichment). Each external source must earn its inclusion.
5. **Document the snapshot** in `data/raw/SNAPSHOT_INFO.md`: exact source, date, file sizes,
   row counts, schema hash, license, and any external sources used.

## Code template (robust HTTP download, deterministic)

```python
import urllib.request, urllib.error, time
from pathlib import Path

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

def download_with_retry(url: str, out: Path, retries: int = 3) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(out, "wb") as f:
                f.write(r.read())
            print(f"OK ({k+1}/{retries}) {url} -> {out} ({out.stat().st_size:,} B)")
            return
        except urllib.error.HTTPError as e:
            last = e; wait = 2 ** (k + 1)
            print(f"HTTP {e.code} attempt {k+1}; waiting {wait}s"); time.sleep(wait)
    raise RuntimeError(f"failed after {retries} retries: {url}") from last
```

## Schema validation template

```python
import pandas as pd
def validate(df: pd.DataFrame, required: list[str], min_rows: int, keys: list[str]):
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"missing required columns: {missing}"
    assert len(df) >= min_rows, f"too few rows: {len(df)} < {min_rows}"
    for k in keys:
        assert df[k].notna().all(), f"nulls in key column {k}"
    print(f"schema OK: {len(df):,} rows, {df.shape[1]} cols")
```

## Output spec

- Raw files in `data/raw/` with non-zero size
- `data/raw/SNAPSHOT_INFO.md`: source URL/slug/query, snapshot date, row counts, schema,
  license, external sources (each justified)

## <EXTREMELY-IMPORTANT> Rules

1. **Browser UA + retry** for portal downloads — the default Python UA often returns 403.
2. **Validate schema immediately, halt loud.** A missing key column must stop the pipeline,
   not silently corrupt EDA.
3. **Document the snapshot date/version.** Quarterly-updated sources change row counts; without
   the date the analysis isn't reproducible.
4. **External sources only if justified by the question.** No "grabbed it just in case".
5. **No silent failures.** Every download prints an OK line; every validation asserts.

## Auto-review before handoff (to EDA)

1. Every required file exists with non-zero size
2. `SNAPSHOT_INFO.md` has exact source + date
3. Schema validation passed (key columns, row count, key nulls)
4. External sources each justified by the question

## Red flags

| Thought | Reality |
|---|---|
| "HTTP 403, the URL is wrong" | Usually the UA. Use a browser UA + retry. |
| "Schema looks similar, probably fine" | "Probably" corrupts EDA silently. Assert explicitly. |
| "Grab every external dataset I find" | Each must justify inclusion vs the question. |
| "2024 snapshot is close enough" | Updates shift counts 5-15%. Use the latest; record the date. |
