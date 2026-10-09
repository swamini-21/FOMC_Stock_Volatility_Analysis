# Phase 3 data pipeline

## Scope

Phase 3 implements validated ingestion and conservative preprocessing for the existing Federal Reserve corpus and VIX file. It does not download new data, run NLP models, align events to trading sessions, or estimate statistical models.

## Guarantees

- Every source row is retained unless the input fails the required file schema.
- Documents sharing a publication date are preserved as separate records.
- Canonical URLs provide stable document identity; exact content is tracked separately with SHA-256.
- Raw text remains unchanged in `raw_text`; preprocessing writes a separate `cleaned_text` field.
- Cleaning rules are anchored to document starts or complete header lines.
- Missing market values remain in the market table with an explicit exclusion reason.
- Every pipeline run emits record counts, validation issues, input hashes, and a processing version.
- Validation errors stop output creation by default.

## Outputs

Running the pipeline creates ignored local artifacts under `data/processed/`:

| Output | Purpose |
|---|---|
| `fed_documents.csv` | Document identities, metadata, raw and cleaned text, hashes, duplicate flags, and validation status |
| `vix_observations.csv` | Preserved VIX rows, normalized dates and values, usability flags, and exclusion reasons |
| `data_quality_report.json` | Dataset-level counts, warnings, errors, and processing metadata |

The outputs are excluded from Git because they are generated and the document file includes the full corpus text.

## Document identity and duplicates

`document_id` is derived from the canonical source URL. The content hash is stored independently so a source URL that unexpectedly changes content can be detected as an identity conflict. If no valid URL is available, the content hash is the fallback identity.

Duplicate flags have distinct meanings:

- `is_exact_duplicate`: every source column is repeated.
- `is_duplicate_content`: the raw text hash occurs in multiple records.
- `is_duplicate_document_id`: the canonical source identity occurs more than once.

None of these flags causes automatic deletion. A publication date is an event-grouping attribute, not a document identifier.

## Text preprocessing

The cleaner applies HTML entity decoding, Unicode normalization, control-character removal, line-wrap dehyphenation, and whitespace normalization. Boilerplate removal is restricted to recognized full headings or a transcript disclosure preamble anchored at the beginning of a document.

The broad notebook expression beginning with `Conference Call.*?` is not used. A conversational mention of a conference call therefore remains intact. Each record stores matched cleaning rules and the proportion of characters removed.

## Market observations

The source provider and retrieval date are not established by the current repository. Configuration therefore labels the file as an undocumented local source and leaves retrieval timestamps empty. Phase 3 does not silently describe these observations as newly downloaded FRED data.

Rows with missing values are retained with `is_usable = false` and `exclusion_reason = missing_or_non_numeric_value`. Negative values, duplicate trading dates, and invalid dates are validation errors.

## Running the pipeline

From the activated project environment:

```powershell
python -m fed_policy_intelligence.data.pipeline --config config/config.yaml
```

To validate without writing output files:

```powershell
python -m fed_policy_intelligence.data.pipeline --config config/config.yaml --validate-only
```

## Current limitations

- Publication timestamps and source retrieval timestamps are unavailable in the local inputs.
- The corpus currently contains only the notebook's `statement` and `transcript` categories.
- Possible encoding artifacts are reported but not automatically rewritten.
- Trading-calendar alignment belongs to the event-study phase.
- Data-provider download interfaces will be added only when newer data is introduced.

## Verified local run

The pipeline was run against the current local inputs after implementation:

| Measure | Result |
|---|---:|
| Source document rows | 773 |
| Document rows retained | 773 |
| Valid document rows | 773 |
| Unique document IDs | 773 |
| Publication dates with multiple documents | 232 |
| Documents on those dates | 470 |
| Documents losing more than 25% of text | 0 |
| Maximum cleaning removal ratio | 1.56% |
| Documents flagged for possible encoding artifacts | 59 |
| Source VIX rows | 9,320 |
| VIX rows retained | 9,320 |
| Usable VIX observations | 9,021 |
| Missing VIX values retained and flagged | 299 |

The input hashes match those recorded during the Phase 1 audit. Generated outputs remain local and ignored by Git.

