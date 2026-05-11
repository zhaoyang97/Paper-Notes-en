---
title: >-
  [Paper Note] LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations
description: >-
  [ICLR 2026][Model Compression][Difficulty Prediction] This paper demonstrates that LLMs encode model-specific success probability information in their pre-generation internal activations. Training linear probes to extrac…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Difficulty Prediction"
  - "Linear Probe"
  - "Model Routing"
  - "Inference-Time Compute"
  - "Success Prediction"
date: 2026-05-08
content_hash: cccc5250f0cc929a
---

# LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations

**Conference**: ICLR 2026
**arXiv**: [2602.09924](https://arxiv.org/abs/2602.09924)
**Code**: [https://github.com/KabakaWilliam/llms_know_difficulty](https://github.com/KabakaWilliam/llms_know_difficulty)
**Area**: Model Compression
**Keywords**: Difficulty Prediction, Linear Probe, Model Routing, Inference-Time Compute, Success Prediction

## TL;DR
This paper demonstrates that LLMs encode model-specific success probability information in their pre-generation internal activations. Training linear probes to extract this signal enables efficient model routing that matches the accuracy of the strongest model while reducing inference cost by 70% on benchmarks such as MATH.

## Background & Motivation

**Background**: LLMs have achieved remarkable results on mathematical and programming tasks, yet running extended reasoning (e.g., CoT) for every query is expensive. Model routing systems require accurate estimates of a model's success probability on a given input, but low-variance estimates demand multiple costly samples.

**Limitations of Prior Work**: Prior work has shown that models contain correctness-related signals, but it remains unclear whether these signals reflect human-perceived difficulty or model-specific difficulty, and whether they are reliable enough to support practical decision-making. Existing routing methods rely on indirect proxies such as input length, perplexity, or heuristic confidence scores.

**Key Challenge**: Human judgments of difficulty and a model's internal "perception" of difficulty are fundamentally distinct—as extended reasoning capabilities scale up, models increasingly solve problems that humans find hard, widening the gap between the two notions of difficulty.

**Goal**: 1) Disentangle human difficulty signals from model-specific difficulty signals encoded in LLM activations; 2) Evaluate the reliability of these signals across different reasoning strategies; 3) Apply the probes to practical model routing to reduce inference cost.

**Key Insight**: The E2H-AMC dataset, which provides both human IRT difficulty labels and model performance records, is used to directly compare human difficulty and model difficulty signals extracted from the same pre-generation activations.

**Core Idea**: LLMs encode information about their own success probability in activations before generating any answer token. Extracting this signal via linear probes enables cost-accuracy trade-off routing with negligible overhead.

## Method

### Overall Architecture
For a given LLM, the last-layer activation vector is extracted after the instruction tokens and before the first generated token. A linear probe is trained on these activations to predict success or failure under a specific decoding strategy. The probe's predicted probability is then used to make model routing decisions.

### Key Designs

1. **Dual-Objective Linear Probe Framework**:

    - Function: Separately predict human IRT difficulty and model success probability.
    - Mechanism: Two types of probes are trained on the same pre-generation activations $\mathbf{h} \in \mathbb{R}^d$: (a) a regression probe predicting expected success rate $\hat{s}_{MC}(\pi, q) = \mathbf{w}^\top \mathbf{h} + b$ (MSE loss); (b) a binary classification probe predicting success/failure under a specific decoding strategy (BCE loss), with targets from either Greedy or Maj@K decoding.
    - Design Motivation: Human difficulty and model difficulty are distinct signals; the latter is more valuable for routing. Supervised linear probes outperform unsupervised direction extraction in discriminating reasoning task outcomes.

2. **Cascade Routing**:

    - Function: Cost-aware query allocation between a base model and a stronger model.
    - Mechanism: Given a base model $M_s$ and a stronger model $M_l$, routing follows a threshold rule: $M(x) = M_l$ if $\hat{p}_s(x) < \tau$, otherwise $M(x) = M_s$. The threshold $\tau$ controls the performance–cost trade-off.
    - Design Motivation: A simple threshold strategy suffices to leverage the probe signal without requiring complex routing learning.

3. **Utility-Based Routing**:

    - Function: Optimal model selection from a heterogeneous model pool.
    - Mechanism: Given a model pool $\{M_1, \ldots, M_K\}$ with normalized costs $\{\hat{c}_1, \ldots, \hat{c}_K\}$, the selected model is $\hat{M}(x) = \arg\max_i (\hat{p}_i(x) - \lambda \hat{c}_i)$, where $\lambda$ controls the trade-off between success probability and cost.
    - Design Motivation: When the model pool is heterogeneous (varying sizes and inference budgets), both per-model success probability and cost must be considered jointly.

### Loss & Training
Linear probes are trained with an 80/20 train–validation split; layer and token position selection is based on validation performance. Regression probes use MSE loss; classification probes use BCE loss. The probes are extremely lightweight—a single linear layer—making training cost negligible.

## Key Experimental Results

### Main Results

| Model | Reasoning | Task Accuracy | Linear Probe AUROC | TF-IDF AUROC | Length AUROC |
|-------|-----------|--------------|-------------------|--------------|-------------|
| Qwen2.5-Math-1.5B | Greedy | 0.724 | 0.84 | 0.64 | 0.61 |
| Qwen2.5-Math-1.5B | Maj@5 | 0.763 | 0.76 | 0.63 | 0.66 |
| Qwen2.5-Math-7B | Greedy | 0.809 | 0.79 | 0.68 | 0.67 |
| Qwen2.5-Math-7B | Maj@5 | 0.827 | 0.80 | 0.72 | - |
| GPT-OSS-20B (low reasoning) | Maj@5 | 0.866 | 0.78 | - | - |
| GPT-OSS-20B (high reasoning) | Maj@5 | 0.920 | 0.64 | - | - |

### Ablation Study

| Signal Type | Spearman ρ Range | Notes |
|-------------|-----------------|-------|
| Human IRT Difficulty | 0.83–0.87 | Highly linearly extractable |
| Model Success Rate (low reasoning) | 0.58 | Moderately extractable |
| Model Success Rate (high reasoning) | 0.40 | Significant degradation after reasoning scaling |

| Routing Strategy | Accuracy | Cost Savings | Benchmark |
|-----------------|----------|-------------|-----------|
| Cascade (τ=0.6) | 91.2% (matched) | 17% | MATH |
| Utility Routing (5 models) | 92% (matched) | 70% | MATH |
| Utility Routing | 93.3% (matched) | 37% | AIME 2025 |

### Key Findings
- Linear probes substantially outperform surface features (TF-IDF, question length), typically by 10–20 AUROC points.
- Extended reasoning improves task accuracy but degrades probe quality (AUROC drops from 0.78 to 0.64), suggesting that difficulty information becomes less linearly separable within longer reasoning chains.
- Human difficulty and model difficulty are distinct signals; the gap between them widens as model reasoning capability increases.
- Reasoning chain length correlates positively with human difficulty but negatively with model success—models spend more tokens on problems humans find hard, even when the model itself can solve them easily.

## Highlights & Insights
- **Pre-generation activations carry rich decision-relevant signals**: Models effectively "know" whether they will answer correctly before generating any tokens, a finding with far-reaching implications for adaptive inference systems.
- **Empirical evidence that human difficulty ≠ model difficulty**: The divergence between the two grows as reasoning capability scales, suggesting that human difficulty labels may become increasingly unreliable proxies for model evaluation.

## Limitations & Future Work
- Only linear probes at a single token position are used, potentially missing difficulty information encoded non-linearly.
- Cross-domain and cross-dataset transferability of probes remains unexplored.
- Routing strategies are relatively simple (fixed thresholds); adaptive routing policies could further narrow the gap with oracle routing.
- Probe performance is sensitive to token position, limiting practical applicability.

## Related Work & Insights
- **vs. Kadavath et al. (P(True))**: That work elicits self-assessment via explicit prompting, incurring additional generation overhead; this paper extracts signals from pre-generation activations at zero additional cost.
- **vs. Cencerrado et al. (correctness directions)**: Their unsupervised mean-difference approach achieves only 0.6–0.7 AUROC on reasoning tasks; this paper's supervised probes exceed 0.7 AUROC.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic disentanglement of human vs. model difficulty signals, with a demonstrated inverse relationship between extended reasoning and probe quality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across models, datasets, and reasoning strategies.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression with findings building incrementally on one another.
- Value: ⭐⭐⭐⭐ Direct applicability to model routing and adaptive inference systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] QKV Projections Require a Fraction of Their Memory](qkv_projections_require_a_fraction_of_their_memory.md)
- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](../../ACL2026/model_compression/from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[ICLR 2026\] Draft-based Approximate Inference for LLMs](draft-based_approximate_inference_for_llms.md)
- [\[ICLR 2026\] π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](pi-flow_policy-based_few-step_generation_via_imitation_distillation.md)

</div>

<!-- RELATED:END -->
