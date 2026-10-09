# NLP annotation workspace

`annotation_template.csv` is an empty, versioned schema for a manually reviewed
evaluation set. It deliberately contains no inferred or synthetic gold labels.

Annotators should read the full document and assign:

- financial sentiment: `positive`, `negative`, or `neutral`, using the expected
  financial implications expressed by the passage rather than policy direction;
- emotion: one of `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, or
  `surprise`, while recording ambiguity in `annotator_notes`;
- policy stance: `hawkish`, `dovish`, `neutral_or_mixed`, or
  `insufficient_evidence`, following `docs/model_evaluation.md`;
- verbatim evidence copied from the source for policy-stance decisions.

Rows used during rule or prompt development must be marked `development`. Final
performance estimates must use rows marked `evaluation`. A second annotator and
agreement measurement are preferred for subjective labels.
