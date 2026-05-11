---
title: >-
  [Paper Note] Annotation-Efficient Universal Honesty Alignment
description: >-
  [ICLR 2026][LLM Reasoning][honesty alignment] This paper proposes EliCal (Elicit then Calibrate), a two-stage framework that first trains an LLM to express internal confidence using annotation-free self-consistency signa…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "honesty alignment"
  - "confidence calibration"
  - "self-consistency"
  - "annotation efficiency"
  - "LLM trustworthiness"
date: 2026-05-08
content_hash: 424ddb6a390e965b
---

# Annotation-Efficient Universal Honesty Alignment

**Conference**: ICLR 2026
**arXiv**: [2510.17509](https://arxiv.org/abs/2510.17509)
**Code**: Available (GitHub link)
**Area**: LLM Reasoning
**Keywords**: honesty alignment, confidence calibration, self-consistency, annotation efficiency, LLM trustworthiness

## TL;DR
This paper proposes EliCal (Elicit then Calibrate), a two-stage framework that first trains an LLM to express internal confidence using annotation-free self-consistency signals, then calibrates with a minimal number of correctness annotations (only 1K samples, 0.18% of the full set). On HonestyBench (560K training + 70K evaluation), EliCal achieves approximately 98% of the fully-supervised upper bound and generalizes better than calibration-only baselines on unseen MMLU tasks.

## Background & Motivation
**Background**: Honesty alignment for LLMs requires models to accurately perceive their own knowledge boundaries and express calibrated confidence. Existing approaches fall into two categories: training-free confidence estimation (token probabilities, self-consistency) and training-based calibration (requiring correctness annotations).

**Limitations of Prior Work**: Training-based methods yield superior results, but achieving universal honesty alignment across tasks demands large-scale correctness annotations — every question requires a ground-truth answer to determine whether the model responds correctly. This is prohibitively expensive.

**Key Challenge**: Correctness annotations simultaneously serve two roles — (1) teaching the model to express confidence; and (2) calibrating that confidence against actual correctness. If the first role can be fulfilled by a cheaper signal, only a small number of annotations are needed for the second step.

**Goal**: How can high-quality honesty alignment be achieved with minimal correctness annotations?

**Key Insight**: The observation that self-consistency confidence (the proportion of semantically consistent responses across multiple samples) is highly correlated with actual accuracy and is freely available. This signal is used to teach the model to express confidence in Stage 1, followed by calibration with a small number of annotations in Stage 2.

**Core Idea**: "Elicit first, calibrate later" — leverage self-consistency for pretraining-level confidence learning, then use minimal annotations for fine-tuning-level calibration.

## Method

### Overall Architecture
EliCal consists of two stages: Stage 1 trains the model to express internal confidence using self-consistency signals from 560K questions (no ground truth required); Stage 2 calibrates the confidence using only 1K correctness-annotated samples. The model architecture freezes LLM parameters and trains a LoRA adapter plus a linear head to output a confidence score.

### Key Designs

1. **Stage 1: Confidence Elicitation**:

    - **Function**: Trains the model to output its internal confidence in a single forward pass, replacing the expensive multi-sample consistency estimation.
    - **Mechanism**: For each question, $k=20$ responses are sampled; the proportion of responses semantically consistent with the greedy decoding output is computed as the self-consistency target. An MSE loss trains the LoRA adapter and linear head to predict this target.
    - **Design Motivation**: Self-consistency is highly correlated with true accuracy (Figure 2) and requires no ground-truth labels. This stage teaches the model to "perceive how confident it is."

2. **Stage 2: Confidence Calibration**:

    - **Function**: Calibrates the confidence learned in Stage 1 to actual correctness using a small annotated dataset.
    - **Mechanism**: Starting from Stage 1 parameters, training continues with MSE loss but with per-instance accuracy (derived from ground truth) as the target. Only ~1K annotated samples are required.
    - **Design Motivation**: Analogous to the pretraining–fine-tuning paradigm — Stage 1 establishes a foundational ability to express confidence, and Stage 2 performs the final-mile calibration with minimal supervision.

3. **HonestyBench**:

    - **Function**: Constructs a large-scale benchmark for honesty alignment evaluation.
    - **Mechanism**: Integrates 10 free-form QA datasets into 560K training, 38K in-domain evaluation, and 33K OOD evaluation samples. Each model–question pair is annotated with self-consistency and correctness across 20 sampled responses, covering 3 LLMs (Qwen-7B/14B, Llama-8B).
    - **Design Motivation**: Prior honesty research evaluates only on small in-domain datasets, lacking generalizability tests.

### Loss & Training
Both stages use MSE loss. LLM parameters are frozen; only the LoRA adapter and linear head are trained. Stage 1 trains on all 560K questions with self-consistency targets; Stage 2 fine-tunes on 1K annotated samples with correctness targets.

## Key Experimental Results

### Main Results

| Method | Annotation Size | In-Domain Performance | OOD Performance |
|--------|----------------|----------------------|-----------------|
| Best Training-Free (Self-Consistency) | 0 | Baseline | Baseline |
| Cal-Only (full annotations) | 560K | Upper bound | — |
| EliCal + Cal-Only (full) | 560K | Upper bound (17%+ over training-free) | — |
| **EliCal (1K annotations only)** | **1K (0.18%)** | **~98% of upper bound** | **Significantly better than Cal-Only** |
| Cal-Only (1K annotations only) | 1K | Significantly below EliCal | Poor |

### Ablation Study

| Configuration | Performance | Notes |
|--------------|-------------|-------|
| Cal-Only (from scratch) | Requires >>1K annotations | Without elicitation, convergence demands many labels |
| EliCal 1K | ~98% of upper bound | Pretraining–fine-tuning paradigm greatly improves annotation efficiency |
| MMLU (OOD) | EliCal >> Cal-Only | Generalizes to unseen tasks |

### Key Findings
- EliCal achieves 98% of the best performance with only 0.18% of annotations, representing more than a 500× improvement in annotation efficiency.
- On MMLU (fully OOD), EliCal consistently outperforms Cal-Only, demonstrating that self-consistency pretraining provides a stronger generalization basis.
- Self-consistency confidence is highly correlated with correctness across multiple models; however, models are systematically overconfident — which motivates Stage 2 calibration.
- Under full annotation, EliCal and Cal-Only reach comparable performance (both at the upper bound), but EliCal is substantially better in the low-annotation regime.

## Highlights & Insights
- **Transferring the pretraining–fine-tuning paradigm to confidence learning**: The strategy of using cheap signals for "pretraining" and expensive annotations for "fine-tuning" is methodologically elegant and broadly applicable.
- **Self-consistency as a free supervisory signal**: Although self-consistency $\neq$ correctness (a model may be consistently wrong), it serves as a sufficiently good proxy signal for teaching confidence expression.
- **Practical value of HonestyBench**: With 560K samples, three LLMs, and 10 datasets, the benchmark provides the community with a standardized and large-scale platform for honesty alignment evaluation.

## Limitations & Future Work
- Self-consistency requires $k=20$ samples to generate training signals; while this is only needed during Stage 1 data construction and inference remains one-shot, the sampling overhead is non-trivial.
- Evaluation is limited to free-form QA; tasks requiring more precise confidence estimation, such as mathematical reasoning, are not covered.
- Whether the linear head + LoRA architecture is optimal has not been thoroughly explored.
- Whether the calibrated confidence is genuinely effective in downstream applications such as RAG triggering remains to be validated.

## Related Work & Insights
- **vs. Cal-Only (Zhang et al., 2024)**: Cal-Only degrades sharply under low-annotation settings; EliCal addresses this via the elicitation stage.
- **vs. Training-Free Methods (Self-Consistency)**: EliCal's one-shot inference is far more efficient than the 20-sample self-consistency approach and achieves better-calibrated outputs.
- **vs. R-Tuning (Yang et al., 2023)**: R-Tuning trains and evaluates on a single dataset; EliCal targets universal honesty alignment across diverse tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The two-stage framework design is innovative; the annotation-free elicitation stage is particularly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large-scale benchmark, multiple models, in-domain and OOD evaluation, and detailed annotation efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clear, formalization is complete, and the narrative flows naturally.
- **Value**: ⭐⭐⭐⭐⭐ HonestyBench and EliCal together constitute important infrastructure and methodological contributions for the honesty alignment research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision](../../AAAI2026/llm_reasoning/spare_single-pass_annotation_with_reference-guided_evaluation_for_automatic_proc.md)
- [\[ACL 2026\] Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment](../../ACL2026/llm_reasoning/taming_actor-observer_asymmetry_in_agents_via_dialectical_alignment.md)
- [\[AAAI 2026\] Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment](../../AAAI2026/llm_reasoning/dropouts_in_confidence_moral_uncertainty_in_human-llm_alignment.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)

</div>

<!-- RELATED:END -->
