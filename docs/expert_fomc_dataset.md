# Expert-labelled FOMC dataset

The revised policy-stance path uses Georgia Tech's
`gtfintechlab/fomc_communication` dataset as the external training and evaluation
corpus. It contains sentence-level FOMC communication labelled as dovish, hawkish,
or neutral. The source dataset and associated FOMC-RoBERTa model are licensed
CC BY-NC 4.0; downstream use must retain attribution and remain compatible with
that non-commercial license.

The project loads the dataset through the Hugging Face `datasets` library. No copy
is committed to Git. The first run stores files in the Hugging Face cache; later runs
reuse that cache.

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.data.expert_labels
```

The loader validates the presence of train and test splits, non-empty sentences,
and the published label mapping:

- `0`: dovish
- `1`: hawkish
- `2`: neutral

The published test split remains untouched for external evaluation. Model selection
and any optional fine-tuning must use only the training split. Dataset revision
`6b0283f55f0005a6d38d49f271d795c21fccc1a3` is pinned so future runs use the same
expert labels even if the Hub repository changes.

This dataset replaces manual project annotation for policy-stance model development.
It does not replace the existing 773-document corpus, which retains the document
dates and metadata needed for market-reaction analysis.

## Model training and evaluation

The authors' published FOMC-RoBERTa weights are currently gated and return HTTP 401
without a Hugging Face access grant. To keep the project reproducible without a user
account or token, the default path fine-tunes public `FacebookAI/roberta-base`
revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` on the expert training split.

Install the lightweight Trainer runtime and fine-tune using the available GPU:

```powershell
python -m pip install -e ".[nlp,training]"
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.nlp.train_policy
```

Twenty percent of the published training split is selected by a seeded, stratified
split for model selection. The published 496-row test split remains untouched during
training.

Evaluate the saved local model exactly once on that external test split:

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.evaluation.expert_policy --device auto
```

The command writes a JSON metrics report and row-level CSV predictions under the
ignored `data/processed` directory.

## Existing-corpus inference

Apply the validated model sentence by sentence and aggregate transparent stance
shares and mean probabilities for the statement documents used by the market study:

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.nlp.document_policy --device auto --document-type statement
```

The separate output `data/processed/domain_policy_predictions.csv` preserves the
Phase 4 baseline. Omitting `--document-type statement` processes all 773 documents,
including transcripts, if transcript-level research is later required.
