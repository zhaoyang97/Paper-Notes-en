---
title: >-
  [Paper Note] Parallel Token Prediction for Language Models
description: >-
  [ICLR 2026][Model Compression][parallel decoding] This paper proposes Parallel Token Prediction (PTP), which relocates sampling randomness from post-processing to model inputs via auxiliary variables, rendering future tokens deterministic functions of those variables and enabling joint prediction of multiple tokens in a single forward pass.
tags:
  - ICLR 2026
  - Model Compression
  - parallel decoding
  - speculative decoding
  - auxiliary variables
  - autoregressive models
  - inference acceleration
date: 2026-05-08
content_hash: fcf41a57f45744c1
---

# Parallel Token Prediction for Language Models

**Conference**: ICLR 2026
**arXiv**: [2512.21323](https://arxiv.org/abs/2512.21323)
**Code**: [GitHub](https://github.com/mandt-lab/ptp)
**Area**: Model Compression
**Keywords**: parallel decoding, speculative decoding, auxiliary variables, autoregressive models, inference acceleration

## TL;DR

This paper proposes Parallel Token Prediction (PTP), which relocates sampling randomness from post-processing to model inputs via auxiliary variables, rendering future tokens deterministic functions of those variables and enabling joint prediction of multiple tokens in a single forward pass.

## Background & Motivation

The sequential generation process of autoregressive Transformers is the primary bottleneck for inference latency — each token prediction requires one full forward pass. Limitations of existing acceleration methods include:
- **Speculative decoding**: employs a small draft model followed by verification, yet the draft model itself still generates sequentially.
- **Independent multi-token prediction**: assumes conditional independence among tokens, leading to semantic inconsistencies (e.g., generating "def numpy").
- **Discrete diffusion**: requires multi-step iteration with an irreducible sequential component.

The core insight of PTP is that if the random variable $u_i \sim \mathcal{U}[0,1]$ used for sampling is supplied as model input, each token $t_i$ becomes a deterministic function of $u_i$ and the preceding context, allowing the model to predict all future tokens in parallel.

## Method

### Overall Architecture

PTP has two variants: O-PTP (predicting one-hot distributions) and C-PTP (recovering full conditional distributions), both supporting multi-token prediction in a single forward pass. Training can be performed via distillation or from scratch.

### Key Designs

1. **Auxiliary Variable Sampling Mechanism**:

    - Standard sampling: $t_i = \text{Pick}(u_i, P_i)$, where $u_i \sim \mathcal{U}[0,1]$ determines the token via inverse CDF.
    - Key observation: given $u_i$, token $t_i$ is deterministic, and $u_i$ carries information equivalent to $t_i$.
    - **Theorem 1**: $t_k = f_P(t_{<i}; u_i, \ldots, u_k)$, i.e., future tokens can be expressed as deterministic functions of auxiliary variables.

2. **O-PTP (One-Hot PTP)**:

    - The model receives all auxiliary variables $u_i, \ldots, u_N$ simultaneously and predicts one-hot distributions.
    - $t_k = \arg\max(P(t_k | t_{<i}; u_i, \ldots, u_k))$
    - Advantage: efficient parallel prediction; Limitation: does not expose the underlying sampling distribution.

3. **C-PTP (Categorical PTP)**:

    - **Theorem 2**: $P(t_k | t_{<i}, u_i, \ldots, u_{k-1}) = P(t_k | t_{<k})$
    - The full conditional probability distribution is recovered by withholding $u_k$.
    - Can be trained from scratch (inverse autoregressive training) or via distillation.

4. **Partial Quadratic Decoding**:

    - Drafting and verification are executed in parallel, with branches prepared for all possible acceptance counts.
    - Branch probabilities are estimated using model confidence: $P(\#\text{correct}=m|t) \approx (1-c_{i+m})\prod_{k=i}^{i+m-1} c_k$
    - Continuation tokens are greedily allocated to high-probability branches to minimize computational waste.

### Loss & Training

- **Distillation training**: the auxiliary variable $u_k \in [F_{k,t_k-1}, F_{k,t_k})$ for each token is back-derived from the teacher model.
- O-PTP loss: $\mathcal{L}(\theta; t, i) = -\sum_{k=i}^N \log P_\theta(t_k | t_{<i}, u_i, \ldots, u_k)$
- C-PTP loss: $\mathcal{L}(\theta; t, i) = -\sum_{k=i}^N \log P_\theta(t_k | t_{<i}, u_i, \ldots, u_{k-1})$
- Auxiliary variable encoding: $\text{embed}(u) = W \cdot \text{binary}(u) + b$, mapping a float32 value to a 32-bit binary vector.

## Key Experimental Results

### Main Results (SpecBench — Vicuna-7B Distillation)

| Method | MTC | TL | SUM | QA | Math | RAG | Avg. #accepted |
|--------|-----|----|-----|----|------|-----|---------------|
| O-PTP | 2.77 | - | - | - | - | - | **4.2** |
| Autoregressive baseline | - | - | - | - | - | - | ~2.0 |
| Independent prediction | - | - | - | - | - | - | ~3.5 |

| Metric | Ours (O-PTP) | Note |
|--------|-------------|------|
| Wall-clock speedup | **2.4×** | vs. standard autoregressive decoding |
| Accepted tokens per step | **4.2** | per speculative decoding step |

### Ablation Study

| Configuration | #accepted ↑ | Note |
|---------------|------------|------|
| O-PTP (with auxiliary variables) | **7.0 ± 0.1** | coordinated across tokens |
| Independent prediction (no auxiliary variables) | 6.2 ± 0.1 | independent tokens, inconsistent pairs |
| C-PTP trained from scratch | PPL 19.88 | close to autoregressive baseline (19.81) |

### Key Findings

- PTP draft models predict multiple tokens per call, shifting the optimal model size toward larger models (including direct fine-tuning of the teacher).
- Auxiliary variables introduce coordination among tokens, substantially reducing incompatible token pairs (e.g., "def numpy" drops to <1%).
- C-PTP trained from scratch achieves perplexity comparable to autoregressive models, empirically validating the theoretical expressiveness.

## Highlights & Insights

- Strong theoretical contributions: Theorems 1 and 2 rigorously prove the feasibility of parallel sampling from a probabilistic perspective.
- The inverse autoregressive idea from Normalizing Flows is transferred to discrete sequence generation, representing a cross-domain innovation.
- The auxiliary variable mechanism naturally resolves the inconsistency problem inherent in independent multi-token prediction.
- Partial Quadratic Decoding leverages confidence estimates to allocate computation, offering strong practical utility.

## Limitations & Future Work

- Practical speedup is bounded by model capacity — limited Transformer capacity constrains the number of tokens that can be accurately predicted in a single pass.
- Distillation requires the teacher model to back-derive auxiliary variables, incurring non-trivial training cost.
- The binary encoding of auxiliary variables may not be the optimal representation.
- Effectiveness on larger-scale models (70B+) and longer contexts remains unvalidated.

## Related Work & Insights

- Distinction from Medusa/EAGLE: PTP achieves inter-token coordination via auxiliary variables rather than independent multi-head prediction.
- Connection to Normalizing Flows: PTP is essentially a discrete analogue of Inverse Autoregressive Flow.
- Compatible with efficient training techniques such as GaLore and FlashAttention.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The auxiliary-variable parallel sampling framework constitutes an entirely novel theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task validation is provided, though large-scale model experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Theorem proofs are rigorous and figures are clear.
- Value: ⭐⭐⭐⭐⭐ Opens a new design space for parallel token generation.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICLR 2026\] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ICLR 2026\] Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)

<!-- RELATED:END -->
