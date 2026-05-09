---
title: >-
  [Paper Note] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations
description: >-
  [NeurIPS 2025][LLM Evaluation][LLM-as-a-Judge] This paper proposes LAGER, a framework that aggregates score token logits from intermediate to final layers of an LLM and computes an expected score to derive the final judgment. Without any model fine-tuning, LAGER improves human alignment by up to 7.5% and matches or surpasses reasoning-based methods without requiring chain-of-thought inference.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - LLM-as-a-Judge
  - hidden representations
  - cross-layer aggregation
  - evaluation alignment
  - plug-and-play
date: 2026-05-08
content_hash: 3952ecb8646af894
---

# Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations

**Conference**: NeurIPS 2025
**arXiv**: [2508.03550](https://arxiv.org/abs/2508.03550)
**Code**: [https://github.com/sustech-nlp/LAGER](https://github.com/sustech-nlp/LAGER)
**Area**: LLM Evaluation
**Keywords**: LLM-as-a-Judge, hidden representations, cross-layer aggregation, evaluation alignment, plug-and-play

## TL;DR
This paper proposes LAGER, a framework that aggregates score token logits from intermediate to final layers of an LLM and computes an expected score to derive the final judgment. Without any model fine-tuning, LAGER improves human alignment by up to 7.5% and matches or surpasses reasoning-based methods without requiring chain-of-thought inference.

## Background & Motivation
**State of the Field**: LLM-as-a-Judge has become the dominant paradigm for automatic evaluation, yet improving its alignment with human judgments remains a central challenge. Existing approaches either rely on complex chain-of-thought reasoning (increasing computational cost) or require fine-tuning (sacrificing generalizability).

**Limitations of Prior Work**: The standard practice uses only the highest-probability score token from the final layer (vanilla score), which (a) discards rich information in the full probability distribution (e.g., when scores of 4 and 5 have similar probabilities but only 5 is selected), and (b) ignores potentially superior judgment signals encoded in intermediate layers.

**Core Observation**: Empirical analysis reveals that hidden representations from middle-to-upper layers frequently yield judgment scores more aligned with human ratings than those from the final layer — different layers encode complementary semantic and task-level information.

**Starting Point**: A better scoring distribution can be obtained by computing a weighted aggregation of score logits across all layers, followed by taking the expectation to produce a continuous, fine-grained score. Layer weights are trained lightly on a small validation set (only $L+1$ parameters total), while the model backbone remains fully frozen.

**Core Idea**: Cross-layer logit weighted aggregation + probability distribution expectation = superior judgment scores compared to final-layer argmax, with a fully plug-and-play design.

## Method

### Overall Architecture
During LLM-based evaluation, at the position of the score token, hidden representations $\mathbf{h}_n^{(l)}$ are extracted from all $L+1$ layers (from the embedding layer to the final decoder layer). These representations are projected to logits via the shared unembedding matrix, aggregated with per-layer weights, and then passed through a softmax over the candidate score tokens to obtain a probability distribution, whose expectation serves as the final score.

### Key Designs

1. **Cross-Layer Logit Aggregation**:

   - Function: $\hat{\mathbf{z}} = \sum_{i=0}^{L} w_i [\mathbf{h}_n^{(i)} \mathbf{W}_{\text{unembd}}]_{\mathcal{M}}$, where $\mathcal{M}$ is the set of candidate score tokens.
   - Design Motivation: Different layers encode information at different granularities — lower layers capture local lexical features, middle layers capture semantics, and upper layers capture task-level reasoning. Aggregation integrates perspectives from all layers.
   - Key Detail: Softmax normalization is **not** applied prior to aggregation (ablations confirm that pre-normalization discards relative scale information in the logits, leading to degraded performance).

2. **Expected Score**:

   - Function: $s^* = \sum_{s \in \mathbb{S}} s \times P(s)$, where $P(s) = \text{softmax}(\hat{\mathbf{z}})[s]$.
   - Design Motivation: Provides finer granularity than argmax — if $P(4)=0.45$ and $P(5)=0.55$, argmax assigns 5, while the expected score yields 4.55, better discriminating response quality.
   - This simple modification alone yields significant improvements (E-Score baseline).

3. **Lightweight Weight Training**:

   - Function: $L+1$ layer weight parameters are trained on a small validation set using a joint CE+MAE loss (e.g., only 33 parameters for LLaMA-3.1-8B).
   - The model backbone is completely frozen; next-token prediction is unaffected.
   - Weights are trained once and reused across all benchmarks and downstream tasks.
   - Uniform aggregation without weight training (LAGER w.o. tuning) also yields substantial improvements.

## Key Experimental Results

### Main Results (Spearman Correlation, Direct Evaluation without Reasoning Chain)

| Model | Method | Flask | HelpSteer | BIGGen | Avg. |
|-------|--------|-------|-----------|--------|------|
| LLaMA-3.1-8B | VScore | 0.442 | 0.452 | 0.333 | 0.409 |
| LLaMA-3.1-8B | E-Score | 0.454 | 0.520 | 0.403 | 0.459 |
| LLaMA-3.1-8B | **LAGER** | **0.488** | **0.560** | **0.421** | **0.490** |
| Qwen-2.5-14B | VScore | 0.489 | 0.440 | 0.420 | 0.450 |
| Qwen-2.5-14B | **LAGER** | **0.528** | **0.524** | **0.449** | **0.500** |
| LLaMA-3.3-70B | VScore | 0.501 | 0.508 | 0.445 | 0.485 |
| LLaMA-3.3-70B | **LAGER** | **0.538** | **0.548** | **0.473** | **0.520** |

### Comparison with Reasoning-Based Methods

| Method | Flask | HelpSteer | BIGGen |
|--------|-------|-----------|--------|
| VScore + Reasoning | 0.456 | 0.470 | 0.388 |
| LAGER (no reasoning) | **0.488** | **0.560** | **0.421** |

### Key Findings
- LAGER achieves up to 7.5% average Spearman correlation improvement across three benchmarks.
- LAGER matches or surpasses explicit reasoning methods without any chain-of-thought — shallow reasoning chains prove unreliable.
- Uniform aggregation (no training) already yields significant gains; trained weights provide further improvement.
- Aggregating before softmax outperforms aggregating after softmax: preserving logit scale information is critical.
- Downstream validation: using LAGER to select instruction fine-tuning data outperforms multiple baselines on AlpacaEval-2.0.

## Highlights & Insights
- The finding that **"intermediate layers understand judgment better than the final layer"** is highly instructive: the final layer may lose certain judgment-relevant semantic signals due to over-optimization toward the next-token prediction objective.
- **Minimalist design**: only 33 trainable parameters for an 8B model, fully plug-and-play without modifying the inference pipeline — likely the most lightweight approach to improving LLM-as-a-Judge currently available.
- The improvement from **expected score vs. argmax**, while seemingly minor, is conceptually significant: it converts discrete scores into continuous ones, capturing the model's "uncertainty" between adjacent ratings.

## Limitations & Future Work
- Access to intermediate hidden representations is required, making the method inapplicable to API-only models (though degrading to E-Score still provides benefit).
- Layer weights are trained on a validation set — if the validation distribution diverges significantly from the test distribution, performance may be suboptimal.
- Only pointwise evaluation is addressed; pairwise comparison settings remain unexplored.
- Weights are not transferable across models with different layer counts — separate weight training is required per model.
- No mechanistic analysis is provided to explain why intermediate layers yield better judgment — deeper investigation into what judgment-relevant signals these layers encode is lacking.

## Related Work & Insights
- **vs. G-Eval**: G-Eval computes the softmax expectation over final-layer logits (equivalent to E-Score); LAGER further aggregates across layers for superior performance.
- **vs. Prometheus/TIGERScore**: These methods fine-tune the entire LLM, limiting generalizability. LAGER freezes the model and trains only 33 parameters.
- **vs. CoT reasoning methods**: Reasoning-based approaches increase latency and yield unstable reasoning quality; LAGER requires no reasoning and is both faster and more accurate.

## Rating
- Novelty: ⭐⭐⭐⭐ Cross-layer aggregation for judgment is a novel perspective, though the core technique (weighted logit aggregation) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six models, three benchmarks, multiple ablations, and downstream application validation — comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated (the layer-wise analysis in Figure 2 is particularly convincing); methodological description is precise.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play, near-zero-cost improvement with direct practical value for any scenario employing LLM-as-a-Judge.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](../../ICLR2026/llm_evaluation/preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](../../ACL2026/llm_evaluation/contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)

<!-- RELATED:END -->
