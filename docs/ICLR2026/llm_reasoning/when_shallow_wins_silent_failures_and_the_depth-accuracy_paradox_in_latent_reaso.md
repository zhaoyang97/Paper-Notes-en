---
title: >-
  [Paper Note] When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][latent reasoning] This paper systematically analyzes the latent reasoning behavior of Qwen2.5-Math-7B on GSM8K…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "latent reasoning"
  - "faithfulness metrics"
  - "silent failures"
  - "depth-accuracy paradox"
  - "computational stability"
date: 2026-05-08
content_hash: 7b7d6fc25219c99a
---

# When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning

**Conference**: ICLR 2026
**arXiv**: [2603.03475](https://arxiv.org/abs/2603.03475)
**Code**: [github.com/SubramanyamSahoo/When-Shallow-Wins](https://github.com/SubramanyamSahoo/When-Shallow-Wins)
**Area**: LLM Reasoning
**Keywords**: latent reasoning, faithfulness metrics, silent failures, depth-accuracy paradox, computational stability

## TL;DR

This paper systematically analyzes the latent reasoning behavior of Qwen2.5-Math-7B on GSM8K, finding that 81.6% of correct predictions arise from computationally inconsistent paths, 8.8% constitute silent failures (high-confidence errors), and revealing a paradoxical relationship between reasoning depth and accuracy.

## Background & Motivation

**Background**: Chain-of-Thought (CoT) prompting has substantially improved multi-step reasoning in LLMs, yet explicit reasoning consumes context windows, introduces latency, and may not faithfully reflect the underlying computation. Recent architectures have demonstrated **latent reasoning**—completing multi-hop inference within activation space without linguistic verbalization.

**Limitations of Prior Work**: Current benchmarks focus solely on per-sample accuracy and cannot measure the reliability of a model's internal computation. A correct answer may stem from a stable reasoning path or from a fragile heuristic shortcut. In high-stakes settings such as educational tutoring and automated grading, this opacity poses deployment safety risks.

**Key Challenge**: Benchmark accuracy ≠ computational reliability. Models can achieve seemingly strong accuracy via statistical shortcuts while the underlying reasoning paths remain highly unstable, potentially yielding drastically different outputs under minor input perturbations.

**Goal**: The paper proposes a composite faithfulness metric—comprising activation stability $\mathcal{S}$, reasoning-hop alignment $\mathcal{A}$, and depth efficiency $\mathcal{E}$—that quantifies the true computational quality of latent reasoning through activation analysis across multiple forward passes, and constructs a safety evaluation framework for identifying silent failures.

## Method

### Overall Architecture

The paper is organized around three core research questions: (1) how to quantify the faithfulness of latent reasoning; (2) whether latent reasoning constitutes a compressed form of CoT or a qualitatively different computational strategy; and (3) whether models can achieve high accuracy simultaneously through both stable and unstable paths. The methodology comprises four modules: faithfulness metric design, layer-wise interpretability analysis, safety evaluation framework, and compression hypothesis testing.

### Key Design 1: Composite Faithfulness Metric

The metric is a weighted combination of three interpretable components:

$$\mathcal{F}(q) = 0.35 \cdot \mathcal{S}(q) + 0.35 \cdot \mathcal{A}(q) + 0.30 \cdot \mathcal{E}(q)$$

- **Activation Stability $\mathcal{S}$**: Two independent forward passes are executed on the same question; the mean cosine similarity across layer activations is computed and multiplied by a variance penalty term $(1 - \min(\sigma^2, 1))$, jointly capturing average consistency and cross-layer stability.
- **Reasoning-Hop Alignment $\mathcal{A}$**: Layers where activation magnitude changes exceed the 75th percentile are identified as reasoning transition points; a log-ratio measures the alignment between the observed transition frequency and the expected number of reasoning steps.
- **Depth Efficiency $\mathcal{E}$**: Integrates the proportion of active layers, hop density, and magnitude distribution, measuring deviation from the theoretically optimal depth $\mathcal{D}_{\text{opt}} = \min(s/L, 1)$.

A prediction is deemed faithful only when all three thresholds are simultaneously satisfied: $\mathcal{F} \geq 0.60$, $\mathcal{S} \geq 0.65$, and $\mathcal{E} \geq 0.60$.

### Key Design 2: Safety Evaluation Framework and Silent Failure Detection

Based on a two-dimensional classification of activation stability and correctness, model outputs are categorized into four modes:

| Mode | Condition | Risk Level |
|------|-----------|------------|
| True Positive | Correct ∧ $\mathcal{S} \geq 0.65$ | Low |
| Lucky Guess | Correct ∧ $\mathcal{S} < 0.65$ | Medium |
| True Negative | Incorrect ∧ $\mathcal{S} < 0.65$ | Expected |
| Silent Failure | Incorrect ∧ $\mathcal{S} \geq 0.65$ | **High** |

The silent failure rate $\text{SFR} = |\text{Silent Failures}| / |\mathcal{P}|$ quantifies the proportion of erroneous outputs produced with high computational confidence.

### Key Design 3: Compression Hypothesis Testing

Layer-wise activation magnitude trajectories are compared across three reasoning modes (implicit, explicit CoT, and concise reasoning) via cosine similarity:

$$\text{SR} = \frac{1}{|\mathcal{P}|} \sum_{q \in \mathcal{P}} \mathbb{I}[\text{sim}_{\text{traj}}(q, \text{impl}, \text{conc}) \geq 0.7]$$

The compression hypothesis is supported if $\text{SR} \geq 0.75$ and rejected if $\text{SR} < 0.50$.

## Key Experimental Results

### Main Results

Qwen2.5-Math-7B is evaluated on 500 GSM8K problems:

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.610 | 0.488 |
| Reasoning Depth $\mathcal{D}$ | 0.514 | 0.012 |
| Activation Entropy $H$ | 0.090 | 0.041 |
| Stability $\mathcal{S}$ | 0.600 | 0.200 |
| Alignment $\mathcal{A}$ | 0.687 | 0.139 |
| Efficiency $\mathcal{E}$ | 0.737 | 0.030 |
| Overall Fidelity $\mathcal{F}$ | 0.672 | 0.092 |

Failure mode distribution: Lucky Guess 49.8% (249 cases), True Negative 30.2% (151 cases), True Positive 11.2% (56 cases), Silent Failure 8.8% (44 cases). Only 20% of responses satisfy the strict faithfulness criteria.

### Ablation Study

| Configuration | Mean Fidelity | Correlation with Correctness |
|---------------|---------------|------------------------------|
| Full | 0.642 | -0.315 |
| No Stability | 0.718 | -0.315 |
| No Alignment | 0.611 | -0.314 |
| No Efficiency | 0.600 | -0.311 |

Cross-model comparison (7B vs. 1.5B):

| Metric | 7B | 1.5B | Δ |
|--------|----|------|---|
| Accuracy | 0.610 | 0.610 | 0.000 |
| Reasoning Depth | 0.514 | 0.479 | +0.034 |
| Activation Entropy | 0.090 | 0.169 | -0.079 |

### Key Findings

1. **Depth-Accuracy Paradox**: Fidelity exhibits a weak negative correlation with correctness ($r = -0.21$, $p = 0.002$), yet continuous-analysis AUROC reaches 0.78, suggesting this is a binary threshold artifact.
2. **No Benefit from Model Scaling**: Scaling from 1.5B to 7B (4.7× parameters) yields identical accuracy (61%) on the evaluation subset; the larger model reasons more deeply without translating depth into performance gains.
3. **Latent Reasoning ≠ Compressed CoT**: Only approximately 20% of latent reasoning trajectories achieve a cosine similarity ≥ 0.7 with CoT-mode trajectories, with a mean similarity of only 0.43, indicating that latent reasoning employs diverse computational strategies.
4. **Causal Importance of Middle Layers**: Noise intervention experiments reveal that middle layers (layers 6–9) contribute most causally ($\gamma_6 = 0.34$), while activation magnitude peaks occur in later layers (layers 20–28), suggesting a two-stage computational model.

## Highlights & Insights

- **Lucky Guess Dominance**: 81.6% of correct predictions arise from unstable paths, indicating that benchmark accuracy substantially overestimates true reasoning capability.
- **Safety Risks of Silent Failures**: 8.8% of predictions manifest as "high-confidence errors," which can have serious consequences in domains such as education and medical decision-making.
- **Discovery of a Two-Stage Computational Model**: Middle layers perform critical reasoning operations while later layers handle amplification and output formatting, consistent with findings from circuit discovery research.
- **Call for Evaluation Reform**: Single-sample accuracy is insufficient to guarantee computational reliability; multi-run consistency evaluation and stability-weighted scoring mechanisms are needed.

## Limitations & Future Work

- Evaluation is conducted on only 6% of GSM8K (500 problems); generalization of findings requires validation on the full dataset.
- The faithfulness metrics lack a theoretical foundation, and threshold selection is empirical.
- The study focuses exclusively on the Qwen model family; applicability to other architectures remains unknown.
- Stability estimation requires multiple forward passes, limiting scalability to larger models.
- The noise intervention approach is coarse-grained; more precise techniques such as activation patching may yield more informative results.

## Related Work & Insights

- **CoT Reasoning and Explanation Faithfulness**: Lanham et al. (2023) and Turpin et al. (2023) question whether verbalized reasoning reflects true computation; this paper extends that inquiry to the latent reasoning regime.
- **Mechanistic Interpretability**: The circuit discovery methodology of Wang et al. (2023) and the causal intervention approach of Meng et al. (2023) provide the methodological foundation for the paper's layer-wise analysis.
- **Information Bottleneck Theory**: The information bottleneck theory of Tishby & Zaslavsky (2015) receives empirical support in this work—the sharp compression of activation entropy in later layers coincides with high-activation regions.

## Rating

⭐⭐⭐

The paper proposes a valuable faithfulness metric framework and reveals several interesting phenomena in latent reasoning. However, the evaluation scope is limited (only 500 problems), the metrics lack theoretical grounding, and certain conclusions (e.g., no benefit from parameter scaling) may be confounded by the evaluation subset. The overall contribution is of moderate quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)
- [\[ICLR 2026\] No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[AAAI 2026\] Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models](../../AAAI2026/llm_reasoning/answering_the_unanswerable_is_to_err_knowingly_analyzing_and.md)
- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](../../ACL2026/llm_reasoning/large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)

</div>

<!-- RELATED:END -->
