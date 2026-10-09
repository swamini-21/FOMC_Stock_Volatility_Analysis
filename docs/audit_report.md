# Phase 1 audit report

## Scope and audit status

This report audits the supplied notebook and local data without modifying the original notebook or datasets. The project prompt names `FinalFedSA+TM (3).ipynb`, but the workspace contains `FinalFedSA+TM (2).ipynb`; this report uses the file that is present.

The audit covered all 85 notebook cells, stored outputs, both CSV inputs, preprocessing rules, model-loading and inference code, topic modeling, the market-data join, and the OLS implementation. It did not rerun transformer inference or download dependencies. Consequently, the notebook's stored regression output was checked for internal consistency but was not independently regenerated.

Files examined:

| File | Role | Observed SHA-256 |
|---|---|---|
| `FinalFedSA+TM (2).ipynb` | Original analysis notebook | `54C5DF0348F8F011EF7A410BC9EADCFB00011C549B82C2F84C7F43091C426BDC` |
| `fomc_corpus.csv` | Federal Reserve document corpus | `9CC45E2E85C7016C7698E97C2AB8BACE9CA6D2F814BF45EACBF796C9B09F74E5` |
| `VIXCLS.csv` | Daily VIX observations | `1D5FA5B0C4236416BE42C4D2FA6EB9051FC97EE9A88809B3FEB849A826E20113` |

No `.xlsx` file was present in the audited workspace.

## Executive summary

The saved regression output reports 358 observations and R-squared 0.235839, matching the preliminary figures in the project prompt. These values are internally consistent with the stored ANOVA table. They should not be interpreted as verified evidence that Federal Reserve language predicts subsequent volatility.

Four issues invalidate or materially limit the current analytical conclusions:

1. **Emotion inference is invalid.** Cell 20 calls the shared `encode_chunks` function from cell 17. That function closes over the FinBERT tokenizer `tok`, not the emotion tokenizer `emo_tok`. BERT token IDs and special tokens are therefore passed to a DistilRoBERTa emotion model.
2. **Date-only deduplication deletes 238 distinct documents.** All 773 corpus URLs and texts are unique. Cell 4 nevertheless keeps only one row per date, reducing the corpus to 535 records and removing legitimate same-date statements and transcripts.
3. **The conference-call cleaning regex deletes substantive transcript content.** Pattern 9 in cell 7 is unanchored and runs with `DOTALL`. In ordinary meeting transcripts it can match a conversational mention of a conference call and consume text until a later date. It removes 118,076 characters from one May 6, 2003 meeting transcript and more than 100,000 characters from two other meeting transcripts.
4. **The regression does not test the stated research question.** It models the same-day VIX level, not a subsequent VIX change or equity return, has no conventional-market baseline, ignores publication time, and contains an exactly rank-deficient design matrix. The output lists 22 predictors but reports model degrees of freedom of 20.

The transcript word cloud is also confirmed to use statement text. Topic-label matching contains a comparator bug, and the topic categories are manually tied to one particular LDA run. These are material but secondary to the four issues above.

## Existing workflow

| Cells | Current operation | Main output or state change |
|---|---|---|
| 0-6 | Load corpus, inspect duplicates, drop duplicate dates, parse dates | Corpus falls from 773 to 535 documents |
| 7-11 | Regex cleaning and text-length calculation | The original `text` column is overwritten |
| 13-18 | Load FinBERT, tokenize long documents, average chunk logits | Sentiment label, confidence, and positive-minus-negative score |
| 19-23 | Load emotion model and score the same documents | Six-class renormalized emotion distribution plus separate neutral value |
| 24-32 | Full-corpus, statement, and transcript word clouds; CSV exports | Transcript word cloud mistakenly uses statement text |
| 34-40 | Download NLTK resources, define stop words, train LDA, map topics | Separate statement/transcript topic models and manual categories |
| 41-52 | Fit LDA models, inspect categories, save topic CSVs | 242 statement and 293 transcript topic rows |
| 54-68 | Load VIX, remove missing VIX values, concatenate NLP outputs, join on date | 9,021 usable VIX rows; 358 rows have document features |
| 70-83 | Read an intermediate CSV, delete all rows with any missing value, dummy encode, fit OLS | Stored OLS output with 358 cases and R-squared 0.235839 |

The notebook relies on mutable, sequential state. Cells 75-83 repeat pieces of the regression outside `main()`, while cells 48-49 independently refit the transcript LDA for display. Rerunning selected cells out of order can therefore produce stale or inconsistent objects.

## Data sources and record-count ledger

### Federal Reserve corpus

The local corpus has 773 rows spanning March 29, 1976 through September 17, 2025.

| Check | Result |
|---|---:|
| Raw documents | 773 |
| Transcripts | 478 |
| Statements | 295 |
| Exact duplicate rows | 0 |
| Unique URLs | 773 |
| Unique text values | 773 |
| Unique dates | 535 |
| Rows removed by date-only deduplication | 238 |
| Rows retained | 535 |
| Retained transcripts | 293 |
| Retained statements | 242 |
| Missing `meeting` values | 40, all statements |
| Invalid dates | 0 |
| `year` values inconsistent with parsed date | 0 |

There are 232 dates containing multiple records and 470 documents on those dates. Every one of those 232 groups has unique URLs and unique text. The groups comprise 185 transcript/statement pairs, 42 two-statement dates, four three-statement dates, and one four-statement date. The deduplication removes 185 transcripts and 53 statements.

The corpus contains the character `â` in 59 documents, with 93 occurrences. This suggests possible mojibake or extraction artifacts and requires inspection against source documents before correction.

The file contains source URLs, but it does not contain stable project-generated document IDs, retrieval timestamps, source publication timestamps/timezones, content hashes, or processing versions. The `meeting` filename is not a unique identifier: many older records use the same generic `default.htm` name.

### VIX data

The local VIX file has 9,320 dated rows spanning January 2, 1990 through September 22, 2025. Dates are unique and parse successfully. There are 299 missing VIX values, leaving 9,021 usable observations. Observed non-missing values range from 9.14 to 82.69.

The column name strongly resembles the FRED `VIXCLS` series, but the repository does not document the provider, retrieval date, transformation, revision policy, or terms. The source must therefore be treated as unverified from repository evidence alone.

### Transformation counts

| Stage | Rows | Exclusion or interpretation |
|---|---:|---|
| Raw Fed corpus | 773 | No exact duplicate rows |
| After `drop_duplicates(subset=['date'])` | 535 | 238 distinct documents removed |
| NLP-scored/topic-modeled documents in stored outputs | 535 | 242 statements, 293 transcripts |
| Raw VIX observations | 9,320 | Includes 299 missing values |
| VIX after dropping missing `VIXCLS` | 9,021 | Daily market table |
| Deduplicated Fed dates matching usable VIX dates | 358 | Used by the regression after listwise deletion |
| Deduplicated Fed dates not matched | 177 | 172 predate VIX; five fall within the VIX span but have no usable same-date value |
| Raw documents matching usable VIX dates before date deduplication | 594 | Shows how much same-date document information is lost |
| Stored regression sample | 358 | Same-day event/VIX matches |

The five in-span unmatched deduplicated events are September 13, 2001; January 21, 2008; February 7, 2009; May 9, 2010; and March 15, 2020. They include market closures and weekend events. The notebook does not report or realign them.

## Detailed findings

### Critical confirmed errors

#### C1. Emotion model uses the FinBERT tokenizer

**Evidence:** Cells 15 and 17 define `tok` as the `ProsusAI/finbert` tokenizer and `encode_chunks` uses that global variable. Cell 19 loads `emo_tok` for `j-hartmann/emotion-english-distilroberta-base`, but cell 20 calls `encode_chunks(text)` without passing or selecting `emo_tok`.

**Impact:** The emotion model receives BERT vocabulary IDs, BERT special-token IDs, and BERT padding rather than the RoBERTa encoding it was trained on. The IDs remain within the emotion model's embedding range, so execution can succeed while producing semantically meaningless predictions. All stored emotion labels/probabilities and regression terms derived from them are invalid.

**Required regression test:** Construct a generic chunker that takes a tokenizer explicitly. Assert that emotion chunks begin/end/pad with the emotion tokenizer's own special IDs; compare produced IDs with a direct `emo_tok` encoding; and use deliberately incompatible mock tokenizers so a shared-global-tokenizer implementation fails.

#### C2. Date-only deduplication deletes distinct records

**Evidence:** Cell 3 reports 238 duplicate-date rows and no exact duplicates. Cell 4 executes `df.drop_duplicates(subset=['date'], inplace=True)`. Independent inspection finds 773 unique URLs and texts, including all records within the 232 multi-document date groups.

**Impact:** The notebook removes 30.8% of the corpus. It drops distinct statements, implementation-note-like pages, and transcript/statement pairs. Which row survives depends on input order. Downstream class distributions, topics, NLP features, date coverage, and regression results are therefore selection-biased.

**Required remediation:** Preserve all documents. Generate stable IDs from canonical URL/document metadata, maintain content hashes for duplicate-content detection, and treat date as an event grouping field rather than an identifier.

#### C3. Conference-call regex removes substantive text

**Evidence:** Pattern 9 in cell 7 is `Conference Call.*?on <date>` and is applied with case-insensitive, multiline, and dot-all flags. It is not anchored to a header. In `FOMC20030506meeting.pdf`, the match begins at the conversational phrase `conference call. The data suggest...` and removes 118,076 characters through a later page date. It similarly removes 104,903 characters from `FOMC19810707meeting.pdf` and 101,016 from `FOMC19761221meeting.pdf`.

Across all documents, five lose more than half their characters during cleaning and 98 lose more than 25%. Some large reductions may be intended boilerplate removal, but pattern 9's removals from ordinary meeting discussion are confirmed content loss.

**Impact:** Sentiment, emotion, text length, topics, and word clouds are computed from truncated documents. Since `df['text']` is overwritten in cell 9, the notebook does not retain raw and cleaned text side by side for diagnosis.

#### C4. Regression matrix is rank deficient

**Evidence:** The stored output lists 22 predictors but reports `df_model = 20`. Six renormalized emotion columns sum exactly to one, making them collinear with the intercept. Topic categories are created by separate statement and transcript models with disjoint category names, so `doc_type_transcript` is also determined by the topic dummy structure.

**Impact:** Statsmodels fits using a generalized inverse, but individual coefficients are not uniquely identified. The large and unstable coefficients shown for emotions, document type, and statement-only topics should not be interpreted.

### High-priority confirmed errors

#### H1. Transcript word cloud displays statement text

Cell 31 creates `trans_text` but calls `WordCloud(...).generate(statement_text)`. The displayed and saved `transcriptWordCloud.png` is therefore a statement word cloud.

#### H2. Topic-category matching does not implement its stated tie-break rule

Cell 38 stores only Jaccard similarity in `best_score`, then compares future candidates against `(best_score >= 3, best_score)`. Because a Jaccard score cannot reach 3, the first comparison element is always false. Any later category with at least three overlapping tokens can replace an earlier candidate regardless of whether its Jaccard score is lower. Results can depend on dictionary insertion order.

#### H3. The model target is the same-day VIX level

Cells 63 and 71-79 merge on calendar date and set `target = 'VIXCLS'`. The model does not construct VIX changes, forward windows, abnormal changes, or equity returns. It therefore does not answer whether language provides information about *subsequent* market behavior.

#### H4. Publication timing and trading-session alignment are absent

The corpus contains dates but no release times or timezones. Same-date matching assumes a date is sufficient to define the relevant market observation. Weekend events and market-closure dates are left unmatched and later disappear. After-close releases may be aligned to a close preceding the release.

#### H5. Blanket listwise deletion is applied before model-variable selection

Cell 73 and cell 75 call `df.dropna(how='any')` on the entire merged file before choosing predictors. This can delete observations because of fields not used by the model. The current stored sample happens to contain complete NLP fields on the 358 matched rows, but the procedure is fragile and exclusions are not logged.

### Modeling and statistical concerns requiring validation

#### S1. The reported R-squared is descriptive, not incremental predictive evidence

The stored model reports R-squared 0.235839, adjusted R-squared 0.190488, 358 cases, residual degrees of freedom 337, and an overall F statistic of 5.200313. The sums of squares are internally consistent with that R-squared. However:

- there is no baseline using lagged VIX, prior market returns, or other conventional market information;
- there is no chronological train/validation/test split;
- the target is contemporaneous VIX level;
- OLS uses default non-robust uncertainty estimates;
- serial dependence, heteroskedasticity, influential observations, and structural changes are not tested;
- document type and topic families are confounded with time and corpus coverage;
- multiple coefficients are tested without a multiple-testing plan.

The output cannot support causal claims, out-of-sample prediction claims, or a claim that NLP adds information beyond conventional market variables.

#### S2. Market and document tables are combined without an explicit event design

Cell 63 uses the daily VIX table as the left side of the join. It creates 9,021 daily rows, of which only 358 have document features; listwise deletion then implicitly turns the data into an event-date sample. If the 238 deleted documents are restored, a direct date join will duplicate the same market outcome across same-date documents and violate independence unless features are explicitly aggregated or the event structure is modeled.

#### S3. VIX is treated as a stock-market outcome level

VIX is an options-implied expected-volatility index, not an equity-price series. The notebook does not calculate VIX changes and includes no equity benchmark or return series. Interpretation must remain about contemporaneous VIX levels unless the outcome construction changes.

### NLP and chunking concerns

#### N1. Chunk logits are averaged and softmaxed once

Cells 18 and 20 take an unweighted mean of chunk logits, then apply softmax. This is not equivalent to averaging chunk probabilities. Overlap means many tokens contribute more than once, short final chunks receive the same weight as full chunks, and no aggregation alternatives are evaluated. The correct choice is empirical and should be compared on labeled examples rather than assumed.

#### N2. Emotion output changes the model's label space

Cell 20 removes the supported `neutral` class, renormalizes the remaining six probabilities to sum to one, and forces every document to receive a non-neutral top emotion. `emotion_neutral` is then stored separately on the original seven-class scale. The resulting seven values are not one coherent probability distribution. This design choice requires justification and currently overstates emotional content.

#### N3. FinBERT probabilities are not retained

Cell 18 computes all class probabilities but returns only the winning label, winning confidence, and positive-minus-negative score. This prevents later calibration checks, alternative aggregation, and complete error analysis without rerunning inference.

#### N4. Document-level evaluation is absent

There is no manually labeled sample, annotation guide, confusion matrix, macro-F1, calibration analysis, or failure review for sentiment or emotion. The model cards and supported-use limitations are not documented in the repository. Stored output distributions alone are not validation.

#### N5. Inference is inefficient and not restartable

The notebook batches chunks only within one document and processes documents sequentially through two large models. Every chunk is padded to length 512. There is no persistent checkpoint/cache until the full dataframe finishes, so interruptions can require a full rerun. This is especially important for the available CPU-only device.

### Topic-modeling findings

#### T1. Topic labels are manually coupled to one run's top words

Cells 37-40 use manually entered strings of ten top words as category keys. Categories depend on approximate Jaccard overlap with those strings. Small changes to corpus composition, preprocessing, Gensim, or LDA parameters can change top words and break the mapping. The date-deduplication and cleaning defects already alter the corpus being modeled.

#### T2. Separate LDA models make topic categories inseparable from document type

Statements and transcripts are fitted separately and assigned disjoint human-readable category sets. This makes topic-category dummy variables partly encode document type and creates the confirmed regression collinearity. Cross-document topic comparisons are not on a shared latent topic space.

#### T3. Preprocessing removes economically meaningful terms

Cell 35 removes `policy`, `rate`, `rates`, `market`, `markets`, `economic`, and related domain terms. NLTK English stop words also remove negations and modal words that may distinguish policy meaning. Cell 36 removes punctuation and does not retain phrases, lemmatize, or separately inspect numbers. The displayed topics contain years, names, administrative text, and web metadata, suggesting residual artifacts.

#### T4. Topic quality is not evaluated

The number of topics is fixed at five for both corpora. There is no coherence comparison, stability analysis, held-out likelihood, representative-document review protocol, or comparison to an embedding method. One statement topic is explicitly labeled `Webpage Metadata / Non-Policy Content`, indicating preprocessing leakage rather than a substantive policy topic.

### Reproducibility and data-engineering findings

#### R1. Hardcoded Colab paths prevent local execution

Cells 2, 54, and 71 use `/content/...` paths. The notebook metadata records a Colab T4 GPU, while the intended local environment is CPU-only. Paths and compute settings are not configuration-driven.

#### R2. Dependencies and model revisions are not pinned

The repository has no `requirements.txt` or `pyproject.toml`. Cell 39 installs the current Gensim release at runtime; NLTK resources and Hugging Face models are downloaded dynamically. No exact Python, pandas, NumPy, PyTorch, Transformers, statsmodels, NLTK, Gensim, wordcloud, or model revision is recorded.

#### R3. Required intermediate artifacts are absent

The notebook writes `fomc_statement.csv`, `fomc_transcript.csv`, topic CSVs, word-cloud images, `FinalVIX.csv`, and `vix_full_summary.txt`, but none is present in the repository. The regression reads `/content/FinalVIX.csv`, so it cannot be rerun from the saved workspace without first repeating NLP and topic inference.

#### R4. Stored outputs lack trustworthy execution provenance

The notebook contains outputs while code-cell execution counts are null. There is no run manifest, package snapshot, seed manifest beyond LDA's local seed, input hash record, or timestamp tying outputs to a specific run. The stored regression can be audited but should not be described as independently reproduced.

#### R5. Raw and processed text are not separated

Cell 9 overwrites `text`. The pipeline does not preserve source text, cleaned text, cleaning version, or per-rule removal diagnostics. This makes traceability and correction difficult.

## Prioritized remediation plan

This is a recommendation for later phases; no fixes were implemented during this audit.

| Priority | Action | Reason |
|---|---|---|
| P0 | Replace date deduplication with stable document IDs, canonical URLs, and content hashes; retain all legitimate same-date documents | Restores 238 deleted records and prevents order-dependent selection |
| P0 | Refactor chunking to require an explicit tokenizer and add tokenizer/model compatibility tests | Invalidates all current emotion results |
| P0 | Remove or tightly anchor the conference-call regex; preserve raw and cleaned text; add golden-text tests | Prevents confirmed deletion of substantive content |
| P0 | Discard current emotion-derived results and rerun only after the tokenizer and cleaning fixes | Existing emotion outputs are not valid measurements |
| P0 | Redefine the statistical dataset around explicit events, publication timing, VIX changes, and a conventional-market baseline | Current OLS does not answer the research question |
| P1 | Remove exact design-matrix dependencies and pre-specify predictors | Restores coefficient identifiability |
| P1 | Build exclusion and join-cardinality reports, including non-trading-day alignment | Prevents silent record loss and duplicated outcomes |
| P1 | Fix the transcript word cloud and topic-category comparator | Confirmed implementation defects |
| P1 | Refit and evaluate LDA after corpus/preprocessing corrections; keep raw topic IDs/keywords separate from human labels | Current labels are brittle and corpus-dependent |
| P1 | Retain full sentiment and emotion probability distributions without forced removal of neutral | Enables evaluation and coherent downstream modeling |
| P2 | Add configuration, dependency constraints, model revisions, caching, and a CPU demonstration mode | Makes runs reproducible and practical |
| P2 | Add a labeled evaluation sample and error-analysis workflow | Needed before interpreting NLP features |

## Tests needed before later implementation is accepted

| Area | Minimum regression coverage |
|---|---|
| Deduplication | Same date/different URL and content must retain both; exact duplicate content must follow a documented policy |
| Tokenizers | Each model receives IDs, masks, special tokens, padding, and maximum lengths from its paired tokenizer |
| Cleaning | Golden source passages survive; every regex has positive and negative fixtures; removal ratios are bounded and logged |
| Chunking | Empty, short, exactly-at-limit, and multi-chunk documents; overlap and final-chunk behavior; finite normalized outputs |
| Aggregation | Logit, probability, and length-weighted strategies are explicit and testable |
| Topics | Deterministic seed behavior; empty dictionary handling; category matching chooses the true best score; unmapped topics remain visible |
| Joins | Duplicate dates do not create silent many-to-many joins; non-trading dates receive documented alignment; exclusions are counted |
| Outcomes | VIX changes and equity returns use correct direction, window, and trading calendar |
| Regression data | Explicit predictor list, no target leakage, full-rank design check, comparable baseline/enhanced samples |
| Reproducibility | Fresh local/Colab setup can load cached results and run tests without paid APIs |

## Claims supported and not supported

Supported by direct inspection:

- The raw corpus contains 773 distinct URL/text records and 535 unique dates.
- Date-only deduplication removes 238 distinct documents.
- The emotion model uses FinBERT tokenization.
- The transcript word cloud uses statement text.
- Pattern 9 removes substantive transcript passages.
- The stored regression output contains 358 cases and reports R-squared 0.235839.
- The stored regression design is rank deficient.

Not supported by the current implementation:

- Emotion labels are valid measurements of the documents.
- Federal Reserve language predicts subsequent VIX changes or equity returns.
- NLP features add value beyond conventional market information.
- Any observed relationship is causal.
- The saved regression has been reproduced in the current workspace.
- The manually named LDA topics are stable, comprehensive, or superior to alternatives.

## Phase 1 completion boundary

Phase 1 produced this audit only. The original notebook, CSV inputs, model logic, statistical code, and project architecture were not changed. No dependencies or model weights were downloaded, no transformer inference was rerun, and no reported metric was fabricated.
