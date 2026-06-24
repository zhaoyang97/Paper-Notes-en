---
title: >-
  [Paper Note] CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction
description: >-
  [ACL 2025 (Long Paper)][Interpretability][Grammatical Error Correction] This paper proposes CLEME2.0, an interpretable reference-based GEC evaluation metric. By disentangling edits into four categories (correct correction TP, wrong correction FPne, under-correction FN, and over-correction FPun) and combining them with edit weighting techniques, it achieves state-of-the-art correlation with human judgments on both GJG15 and SEEDA datasets.
tags:
  - "ACL 2025 (Long Paper)"
  - "Interpretability"
  - "Grammatical Error Correction"
  - "evaluation metric"
  - "Edit Disentangling"
  - "Reference-based Metric"
date: 2026-05-08
content_hash: d0e6a808e35594fc
---

# CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2407.00934](https://arxiv.org/abs/2407.00934)  
**Code**: [https://github.com/THUKElab/CLEME](https://github.com/THUKElab/CLEME)  
**Area**: Natural Language Processing / Grammatical Error Correction Evaluation  
**Keywords**: Grammatical Error Correction, evaluation metric, interpretability, Edit Disentangling, Reference-based Metric

## TL;DR
This paper proposes CLEME2.0, an interpretable reference-based GEC evaluation metric. By disentangling edits into four categories (correct correction TP, wrong correction FPne, under-correction FN, and over-correction FPun) and combining them with edit weighting techniques, it achieves state-of-the-art correlation with human judgments on both GJG15 and SEEDA datasets.

## Background & Motivation
- **Background**: Current mainstream GEC evaluation metrics (such as ERRANT and MaxMatch/M2) rely on Precision/Recall/F0.5 scores. Although widely used, they suffer from two core limitations:
  1. **Lack of interpretability**: P/R/F scores cannot reveal the specific weaknesses of GEC systems, making it difficult for developers to pinpoint areas for improvement.
  2. **Inability to distinguish between different types of erroneous edits**: Traditional metrics treat all False Positive (FP) edits equally. However, "proposing a correction in the correct position but with wrong content" (wrong-correction) and "making unnecessary changes in error-free positions" (over-correction) are fundamentally different error types.
- **Key Challenge**: In the LLM era, large language models exhibit a strong tendency toward over-correction in GEC tasks, changing the original meaning of sentences. Existing metrics fail to quantify this phenomenon.
- **Limitations of Prior Work**: Traditional metrics assign equal weight to all edits, ignoring variations in edit importance (e.g., punctuation changes versus content word modifications).

## Core Problem
How can an interpretable GEC evaluation metric be designed to multi-dimensionally quantify system performance characteristics (grammaticality and faithfulness) while outperforming existing metrics in terms of correlation with human judgments?

## Method

### Overall Architecture
The workflow of CLEME2.0 consists of three steps:
1. **Edit Extraction**: Utilizing CLEME's chunk partition technique to simultaneously align the source, hypothesis, and reference sentences, segmenting them into an equal number of chunk sequences.
2. **Disentangled Scoring**: Classifying hypothesis edits into four categories: TP, FPne, FPun, and FN, to compute four-dimensional scores: hit-correction, wrong-correction, under-correction, and over-correction.
3. **Comprehensive Scoring**: Merging the four scores into a single comprehensive score via weighted summation, with optional edit weighting techniques (similarity-based or LLM-based).

### Key Designs
1. **Edit Disentangling**:

    - **TP (True Positive)**: Correct corrections where the hypothesis chunk matches the reference chunk.
    - **FPne (False Positive - Necessary)**: Modifications where the hypothesis chunk differs from the reference chunk, but the reference indicates a modification was indeed necessary → correct error location but incorrect target content.
    - **FPun (False Positive - Unnecessary)**: Unnecessary over-corrections where the hypothesis makes a modification while the reference remains unchanged.
    - **FN (False Negative)**: Missed corrections where the hypothesis makes no changes but the reference indicates a modification is needed.
    - The Core Idea lies in further dividing traditional FPs into FPne and FPun, establishing a one-to-one mapping between the four edit types and four system performance traits.

2. **Four-Dimensional Disentangled Scores**:

    - $Hit = \frac{TP}{TP + FP_{ne} + FN}$ (Hit-correction rate)
    - $Wrong = \frac{FP_{ne}}{TP + FP_{ne} + FN}$ (Wrong-correction rate)
    - $Under = \frac{FN}{TP + FP_{ne} + FN}$ (Under-correction rate)
    - $Over = \frac{FP_{un}}{TP + FP_{ne} + FP_{un}}$ (Over-correction rate)
    - Comprehensive score: $Score = \alpha_1 \cdot Hit + \alpha_2 \cdot (1-Wrong) + \alpha_3 \cdot (1-Under) + \alpha_4 \cdot (1-Over)$

3. **Edit Weighting**:

    - **Similarity-based weighting**: Computes semantic importance weights for each edit using PTScore/BERTScore, measuring the impact of an edit on overall sentence quality by simulating partially corrected sentences.
    - **LLM-based weighting**: Employs Llama-2-7B to score the importance of each edit on a scale of 1-5, leveraging LLM semantic understanding to distinguish the significance of different modifications.

4. **Determining Weighting Factors**: Optimal weights are searched using cross-validation:

    - Corpus-level: $\alpha_1, \alpha_2, \alpha_3, \alpha_4 = 0.45, 0.35, 0.15, 0.05$
    - Sentence-level: $\alpha_1, \alpha_2, \alpha_3, \alpha_4 = 0.35, 0.25, 0.20, 0.20$

## Key Experimental Results

### GJG15 Dataset (Corpus-level, average correlation across 6 reference sets)

| Metric | Average Correlation |
|------|-----------|
| M2 | 0.616 |
| ERRANT | 0.625 |
| PT-M2 | 0.666 |
| CLEME-dep | 0.633 |
| CLEME-ind | 0.635 |
| **CLEME2.0-dep** | **0.734** |
| **CLEME2.0-ind** | **0.775** |
| **CLEME2.0-sim-dep** | **0.790** |
| **CLEME2.0-sim-ind** | **0.817** |

### SEEDA Dataset (Average correlation based on TrueSkill)

| Metric | SEEDA-S (γ) | SEEDA-S (ρ) | SEEDA-E (γ) | SEEDA-E (ρ) | Avg. |
|------|------------|------------|------------|------------|------|
| ERRANT | 0.557 | 0.406 | 0.697 | 0.671 | 0.583 |
| CLEME-dep | 0.633 | 0.501 | 0.755 | 0.757 | 0.662 |
| GoToScorer | 0.929 | 0.881 | 0.901 | 0.937 | 0.912 |
| SOME | 0.892 | 0.867 | 0.901 | 0.951 | 0.903 |
| **CLEME2.0-dep** | **0.937** | **0.865** | **0.945** | **0.939** | **0.922** |
| **CLEME2.0-sim-ind** | **0.921** | **0.907** | **0.953** | **0.981** | **0.941** |

### Ablation Study
- Hit-correction and under-correction scores exhibit a moderate positive correlation with human judgments.
- Wrong-correction scores show a negative correlation, but carrying a larger weight in the comprehensive score prevents evaluation bias that solely favors high-confidence edits.
- Over-correction shows a minor positive correlation at the corpus level and a minor negative correlation at the sentence level.
- Similarity-based weighting significantly outperforms LLM-based weighting (the latter uses Llama-2-7B, which has a coarse-grained scale of only 1-5).
- Even without edit weighting, CLEME2.0 achieves comparable or superior performance to other metrics.

## Highlights & Insights
- **Clear and impactful core innovation**: Decomposing FP into FPne (necessary) and FPun (unnecessary) is an elegant yet profound design that directly maps four edit categories to four distinct system performance dimensions.
- **High practical value**: The four-dimensional scores precisely identify GEC system weaknesses (e.g., assessing the CAMB system with 27.1% hit rate, 53.4% under-correction, and 47.0% over-correction), providing developers and users with actionable diagnostics.
- **Thorough and robust experimentation**: Evaluated thoroughly across 2 human-annotated datasets, 6 reference sets, at both corpus and sentence levels.
- **Balance of interpretability and performance**: Outperforms current systems by providing interpretable multi-dimensional feedback while achieving state-of-the-art correlation with human ratings.
- **Edit weighting mechanisms**: Introduces both similarity-based and LLM-based weighting methods, resolving the limitation of traditional metrics overlooking semantic importance.

## Limitations & Future Work
- **Language limitation**: Evaluation is currently restricted to English datasets; effectiveness on other languages remains untested.
- **Dataset limitations**: Experiments are mostly conducted on the CoNLL-2014 shared task reference sets (L2 learner data), lacking multi-domain and multilingual validation.
- **Interpretability lacks human verification**: Although advocating for interpretable evaluation, the paper lacks human-anchored evaluation experiments to comprehensively validate the utility of its interpretability.
- **Suboptimal LLM-based weighting performance**: The LLM-based weighting using Llama-2-7B is less effective than the similarity-based counterpart, suggesting a need for larger or more fine-grained LLMs to enhance performance.
- **Weight parameter tuning requirement**: The four $\alpha$ scaling factors are determined via cross-validation grid search, which may lack robustness across unseen test corpora.

## Related Work & Insights
- **vs ERRANT/M2**: Traditional metrics rely on P/R/F0.5 and fail to distinguish between FPne and FPun; additionally, ERRANT's edit extraction relies on language-specific linguistic pipelines.
- **vs CLEME**: CLEME2.0 inherits CLEME's chunk partition technique and double-hypothesis evaluation framework, while its core breakthroughs lie in edit disentangling and weighting.
- **vs PT-M2**: Although PT-M2 employs pre-trained models for weighting, it remains bound to the P/R/F framework, lacking the multi-dimensional interpretability of disentangled scores.
- **vs Reference-less metrics (SOME, IMPARA)**: These metrics rely on fine-tuned models, making them costly and potentially less robust (e.g., Scribendi Score exhibits inconsistent performance across datasets), whereas CLEME2.0, as a reference-based metric, balances high correlation with clear interpretability.

## Related Work & Insights
- **General paradigm for evaluation metric design**: The strategy of finely classifying error categories to enhance interpretability is generalizable to other natural language generation assessment tasks (e.g., machine translation, summarization).
- **The over-correction concern in LLM evaluation**: The research specifically highlights the over-correction issue in LLMs, which is highly pertinent given the growing usage of LLMs for writing assistance and text refinement.
- **Room for improvement in edit weighting**: The limited performance of Llama-2-7B-based weighting suggests constraints of smaller LLMs on fine-grained evaluation tasks. Future explorations could involve larger instruction-tuned LLMs or task-specific scoring models.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of decomposing FP into FPne and FPun is simple yet effective, though the framework is largely an incremental improvement over CLEME.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Thoroughly evaluated on 2 human-annotated datasets, 6 reference sets, at both corpus and sentence levels, utilizing various weighting schemes, alongside extensive ablation studies and case analyses.
- Writing Quality: ⭐⭐⭐⭐ The work is structurally sound with clear formulations, though some tables possess high information density.
- Value: ⭐⭐⭐⭐ Highly valuable for GEC evaluation, with clean opportunities to migrate the four-dimensional disentangled analysis to other NLP evaluation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CB-SLICE: Concept-Based Interpretable Error Slice Discovery](../../ICML2026/interpretability/cb-slice_concept-based_interpretable_error_slice_discovery.md)
- [\[CVPR 2025\] TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction](../../CVPR2025/interpretability/tide_domain_generalization.md)
- [\[ACL 2026\] Interpretable Coreference Resolution Evaluation Using Explicit Semantics](../../ACL2026/interpretability/interpretable_coreference_resolution_evaluation_using_explicit_semantics.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ACL 2025\] Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](shortcut_neuron_eval.md)

</div>

<!-- RELATED:END -->
