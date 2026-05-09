---
title: >-
  [Paper Note] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences
description: >-
  [AAAI 2026][LLM Pretraining][LLM Evaluation] ELSPR models pairwise preferences of LLM evaluators as tournament graphs, identifies non-transitive preferences via strongly connected components (SCCs), proposes a normalized directed graph structural entropy metric, and filters problematic training data through graph reconstruction — resulting in a 13.8% reduction in non-transitivity and a 0.088 decrease in structural entropy, while the discarded data achieves only 34.4% human agreement (vs. 52.6% for retained data).
tags:
  - AAAI 2026
  - LLM Pretraining
  - LLM Evaluation
  - Non-Transitive Preferences
  - Tournament Graph
  - Data Cleaning
  - Structural Entropy
date: 2026-05-08
content_hash: 3f1ac580ded10edc
---

# ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences

**Conference**: AAAI 2026
**arXiv**: [2505.17691](https://arxiv.org/abs/2505.17691)
**Code**: [https://github.com/yy0525/ELSPR](https://github.com/yy0525/ELSPR)
**Area**: LLM Pre-training
**Keywords**: LLM Evaluation, Non-Transitive Preferences, Tournament Graph, Data Cleaning, Structural Entropy

## TL;DR
ELSPR models pairwise preferences of LLM evaluators as tournament graphs, identifies non-transitive preferences via strongly connected components (SCCs), proposes a normalized directed graph structural entropy metric, and filters problematic training data through graph reconstruction — resulting in a 13.8% reduction in non-transitivity and a 0.088 decrease in structural entropy, while the discarded data achieves only 34.4% human agreement (vs. 52.6% for retained data).

## Background & Motivation

**State of the Field**: LLM-as-judge has been widely adopted to evaluate other models. Pairwise comparison is a common paradigm, but it may produce non-transitive preferences (A>B, B>C, C>A).

**Limitations of Prior Work**: (1) Non-transitive preferences undermine ranking reliability; (2) existing head-to-head evaluations neglect transitivity constraints; (3) the root cause of non-transitivity — low-quality or ambiguous training data — has not been systematically addressed.

**Root Cause**: Ambiguous comparison pairs in training data cause evaluators to learn contradictory preference patterns.

**Paper Goals**: Automatically identify and remove training data that induces non-transitive preferences.

**Starting Point**: Modeling preferences as a graph-theoretic problem — tournament graphs combined with SCC analysis.

**Core Idea**: Non-transitive relations are identified via SCCs in tournament graphs, which are then reconstructed into DAGs to filter inconsistent training samples.

## Method

### Overall Architecture
(1) Construct a tournament graph per question (with position-swapping to debias); (2) apply Tarjan's algorithm to find SCCs; (3) expand SCCs into a DAG by sorting nodes by in-degree → filter inconsistent training samples into "Cleaned" vs. "Discarded" sets; (4) fine-tune the evaluator on cleaned data via LoRA.

### Key Designs

1. **Non-Transitivity Detection**: An SCC with $|S|>2$ containing node pairs without bidirectional edges is classified as a non-transitive SCC.
2. **Normalized Structural Entropy**: $\tau(G) = H^2(G) / \log_2 n$, measuring hierarchical uncertainty of the graph.
3. **SCC→DAG Reconstruction**: Nodes are sorted by in-degree (as a capability estimate); contradictory intra-SCC edges are removed.
4. **LoRA Fine-tuning**: rank=8, 3 epochs, lr=1e-4, batch=16; teacher model is Qwen2.5-Max.

### Loss & Training
Standard preference learning loss. Qwen2.5-7B-Instruct and LLaMA3.1-8B-Instruct serve as backbone models.

## Key Experimental Results

### Main Results (Cross-validated Non-Transitivity Rate, averaged over 5 test sets)

| Data Type | $\rho_{\text{non-trans}}$↓ | $\tau_{\text{avg}}$↓ |
|-----------|--------------------------|---------------------|
| Raw | 64.3% | 0.811 |
| **Cleaned** | **50.5%** | **0.723** |
| Δ | **-13.8%** | **-0.088** |

### Human Validation

| Metric | Cleaned | Discarded |
|--------|---------|-----------|
| Human Agreement | **52.6%** | 34.4% |
| Model–Human Agreement | **80.6%** | 51.2% |

### Ablation Study
- Random filtering (removing an equivalent amount of data) leads to higher non-transitivity, confirming the necessity of targeted filtering.
- Cross-model validation: equally effective on LLaMA3.1 (Helpful_Base: 40.2% vs. 59.0%).
- ~80% of training data is retained — maximum quality gain at minimal data loss.

### Key Findings
- Discarded data achieves only 34.4% human agreement, confirming these are genuinely low-quality samples.
- Non-transitive preferences occur more frequently among response pairs with quality differences below the just-noticeable difference (higher Self-BLEU).
- The cleaned model exhibits stronger discriminability on MT-bench (SD improvement of 2–4%).

## Highlights & Insights
- **Graph-theoretic treatment of preference data** is an elegant design choice — SCCs precisely capture "mutually contradictory evaluation cycles."
- **Validation of discarded data quality** is highly convincing — 34.4% human agreement demonstrates that useful data is not being discarded, and model–human agreement drops from 80.6% to 51.2% (near random).
- **Structural entropy** provides a novel quantitative tool for preference evaluation, generalizable to any scenario requiring assessment of preference consistency.
- From a graph-theoretic perspective, non-transitive preferences are essentially "evaluation cycles" — SCCs are the canonical tool for capturing cycles, making the methodological choice highly natural.

## Limitations & Future Work
- Repeated inference at multiple temperatures is required to construct the graph (computational overhead), which may be impractical for large-scale preference datasets.
- The quality of cleaned data is upper-bounded by the teacher model — inconsistent preferences in the teacher model will degrade cleaning effectiveness.
- Only binary preferences are handled; the approach is not extended to rating-based or multi-candidate ranking scenarios, despite the growing prevalence of K-wise comparisons in LLM evaluation.
- Using in-degree as a capability estimate for SCC→DAG reconstruction is a heuristic that may be inaccurate for highly imbalanced graphs.
- The assumption that non-transitivity primarily stems from data quality rather than the inherent complexity of evaluation tasks may not hold for subjective tasks where a total ordering does not exist.
- Filtering ~20% of training data reduces dataset size, which may cause excessive data loss in low-resource settings.

## Related Work & Insights
- Directly relevant to RLHF data cleaning — non-transitivity in preference data may be a source of reward hacking, and cleaning preference data could fundamentally improve reward model quality.
- **vs. Xu et al. (2025)**: They identify non-transitivity in GPT-4 evaluations but propose no solution; ELSPR provides a systematic graph-theoretic remedy.
- **vs. Traditional Data Denoising Methods**: General-purpose methods lack structural priors specific to preference data; ELSPR leverages the mathematical properties of tournament graphs for targeted filtering.
- **vs. Canoe (AAAI 2026)**: Canoe addresses faithfulness (models failing to follow context), while ELSPR addresses consistency of the evaluator itself — the two are complementary, with the former improving evaluated models and the latter improving evaluators.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of graph theory, structural entropy, and preference data cleaning is highly original; SCC analysis precisely captures non-transitive preferences.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-validation across 5 test sets, human validation, and cross-model validation; rigorously designed.
- Writing Quality: ⭐⭐⭐⭐ Formally rigorous; the derivation from tournament graphs to DAGs is clearly presented.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to improving the reliability of LLM evaluation systems and the quality of RLHF data.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](../../ICLR2026/llm_pretraining/common_corpus_ethical_data_for_llm_pretraining.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] Stochastic Self-Organization in Multi-Agent Systems](../../ICLR2026/llm_pretraining/stochastic_self-organization_in_multi-agent_systems.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](../../ICLR2026/llm_pretraining/predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)

<!-- RELATED:END -->
