---
title: >-
  [Paper Note] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging
description: >-
  [ICLR 2026][LLM Reasoning][Model Merging] This paper proposes RAIN-Merging, a gradient-free two-stage model merging method: it first applies null-space projection to preserve the thinking format of Large Reasoning Models (LRMs), then employs instruction-attention-guided merging coefficients to enhance instruction following, simultaneously improving instruction compliance and reasoning quality.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Model Merging
  - Instruction Following
  - Reasoning Models
  - Null-Space Projection
  - Gradient-Free Method
date: 2026-05-08
content_hash: 27491d1dbd39d651
---

# RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging

**Conference**: ICLR 2026
**arXiv**: [2602.22538](https://arxiv.org/abs/2602.22538)
**Code**: [GitHub](https://github.com/K1nght/RAIN-Merging)
**Area**: LLM Reasoning
**Keywords**: Model Merging, Instruction Following, Reasoning Models, Null-Space Projection, Gradient-Free Method

## TL;DR

This paper proposes RAIN-Merging, a gradient-free two-stage model merging method: it first applies null-space projection to preserve the thinking format of Large Reasoning Models (LRMs), then employs instruction-attention-guided merging coefficients to enhance instruction following, simultaneously improving instruction compliance and reasoning quality.

## Background & Motivation

### The Instruction-Following Contradiction in Large Reasoning Models

Large Reasoning Models (LRMs) such as DeepSeek-R1 and OpenAI-o1 excel at multi-step reasoning but exhibit significant deficiencies in instruction following:

- They generate lengthy chain-of-thought reasoning while ignoring user-specified formats, constraints, or particular requirements.
- This inconsistency severely undermines the practical utility of LRMs in agent and professional tool deployment scenarios.

### Limitations of Prior Work

- **Continued SFT**: Requires large volumes of annotated data (including long chain-of-thought), incurring high costs and risking capability degradation.
- **Naive model merging**: LRMs and Instruction-Tuned Models (ITMs) have fundamentally different output formats—LRMs use `<think>...</think>` delimiters to separate reasoning from answers, while ITMs produce answers directly. Naive merging corrupts the thinking format.

### Key Findings from Parameter-Space Analysis

SVD analysis of the task vectors of LRMs and ITMs reveals that **the principal subspaces of the two models are nearly orthogonal across all key modules** (cosine similarity < 0.1). This suggests that reasoning capability and instruction-following capability are loosely coupled in parameter space, making interference-free merging feasible.

However, orthogonality alone does not guarantee unchanged output behavior—the generation probabilities of special tokens (`<think>`, `</think>`) may be altered during forward propagation.

## Method

### Overall Architecture

RAIN-Merging proceeds in two stages, yielding a merged model:

$$\theta_\star = \theta_R + \lambda \bigoplus_{k=1}^{K} \alpha_\star^k \Delta_I^{\perp, k}$$

where $\theta_R$ denotes the LRM parameters, $\Delta_I^{\perp, k}$ is the null-space-projected ITM task vector, $\alpha_\star^k$ is the instruction-attention-guided merging coefficient, and $\lambda$ is a global scaling factor.

### Stage 1: Reasoning-Aware Null-Space Projection

**Objective**: Preserve the LRM's thinking format (the `<think>...</think>` structure) from being disrupted by merging.

**Core Idea**: Project the ITM task vector into the null space of the forward feature operator $\Phi$ at thinking special token positions—perturbations within this subspace do not alter the intermediate representations at thinking positions.

For each submodule $k$, the null-space projection is constructed as:

$$P^\perp(\Phi^k_{\Omega_{\text{think}}}) = \text{diag}(1) - {\Phi^k_{\Omega_{\text{think}}}}^\top (\Phi^k_{\Omega_{\text{think}}} {\Phi^k_{\Omega_{\text{think}}}}^\top)^+ \Phi^k_{\Omega_{\text{think}}}$$

The projected task vector satisfies $\Phi_{\Omega_{\text{think}}} \text{vec}(\Delta_I^\perp) = 0$, ensuring that forward features at thinking token positions remain entirely unchanged.

**Theoretical Guarantee (Proposition 1)**: Via a second-order approximation of the softmax KL divergence:

$$\mathcal{L}_{\text{think}}(\theta_R + \Delta_I^\perp) = O(\|\Delta_I^\perp\|_2^2) \approx 0$$

Only 150 reasoning calibration samples are required to construct the projection.

### Stage 2: Instruction-Attention-Guided Merging Coefficients

**Objective**: Maximize instruction-following performance while preserving the thinking format.

For each attention head $\tilde{k}$, the instruction alignment score and leakage score are defined as:

$$a^{\tilde{k}}(x, \tilde{\alpha}) = \sum_{t \in \mathcal{R}(x)} \sum_{\tau \in \mathcal{I}(x)} \frac{\text{Att}^{\tilde{k}}(x, \tilde{\alpha})[t, \tau]}{|\mathcal{I}(x)| |\mathcal{R}(x)|}$$

$$u^{\tilde{k}}(x, \tilde{\alpha}) = \sum_{t \in \mathcal{U}(x)} \sum_{\tau \in \mathcal{I}(x)} \frac{\text{Att}^{\tilde{k}}(x, \tilde{\alpha})[t, \tau]}{|\mathcal{I}(x)| |\mathcal{U}(x)|}$$

where $\mathcal{I}(x)$ is the set of instruction tokens, $\mathcal{R}(x)$ is the set of instruction-constrained output tokens, and $\mathcal{U}(x)$ is the set of irrelevant output tokens.

**Optimization Objective**: Maximize the instruction attention score (high alignment − low leakage):

$$\max_{\tilde{\alpha}} \mathcal{J}_I^{\text{Proxy}}(\tilde{\alpha}) := \bar{a}(\tilde{\alpha}) - \rho \bar{u}(\tilde{\alpha})$$

A closed-form solution is obtained via second-order Taylor expansion and engineering approximations:

$$\tilde{\alpha}_\star^{\tilde{k}} = \text{clip}_{[\tilde{\alpha}_l, \tilde{\alpha}_u]} \left(\frac{g^{\tilde{k}}}{\tilde{H}^{\tilde{k}}}\right)$$

The entire process relies only on attention statistics from forward passes and is completely gradient-free. A total of 365 instruction calibration samples are used.

### Loss & Training

RAIN-Merging involves no training loss—it is a gradient-free model merging method. Its optimization objective is a constrained problem:

$$\max_\Delta \mathcal{J}_I(\theta_R + \Delta) \quad \text{s.t.} \quad \mathcal{L}_{\text{think}}(\theta_R + \Delta) \leq \delta$$

## Key Experimental Results

### Main Results: 7B Model Merging (DeepSeek-R1-Distill-Qwen-7B + Qwen2.5-7B-Instruct)

| Method | IF Avg ↑ | Reasoning Avg ↑ | Runtime |
|--------|----------|-----------------|---------|
| LRM (original) | 44.12 | 51.03 | - |
| ITM (original) | 52.92 | 43.32 | - |
| SFT | 45.08 | 49.51 | 120.32 min |
| Task Arithmetic | 45.96 | 49.59 | 0.93 min |
| SLERP | 45.95 | 50.97 | 1.12 min |
| TIES | 46.35 | 51.99 | 1.18 min |
| AIM-TIES | 47.02 | 53.10 | 18.51 min |
| **RAIN-Merging** | **48.11** | **55.59** | 20.96 min |

RAIN-Merging simultaneously improves instruction following (+4.0 vs. LRM) and reasoning capability (+4.6 vs. LRM).

### Multi-Scale and Multi-Architecture Validation

| Model Configuration | IF Relative Gain | Reasoning Relative Gain |
|--------------------|------------------|------------------------|
| Qwen2.5-1.5B | +6.09% | +8.20% |
| Qwen2.5-7B | +9.06% | +8.93% |
| Llama-3.1-8B | +5.86% | +7.78% |
| Qwen2.5-14B | +6.11% | +6.17% |
| Qwen2.5-32B | +1.57% | +3.83% |

Consistent effectiveness across scales from 1.5B to 32B and across both Qwen and Llama architectures.

### Ablation Study

| Method | IF Avg | Reasoning Avg |
|--------|--------|---------------|
| w/o Stage 2 (null-space projection only) | 46.58 | **54.92** |
| w/o Stage 1 (attention-guided only) | **47.62** | 52.44 |
| **RAIN-Merging (full)** | **48.11** | **55.59** |

- Removing Stage 2: limited improvement in instruction following, but reasoning is well preserved.
- Removing Stage 1: stronger instruction following, but notable degradation in reasoning.
- The two stages are complementary: together they achieve the best performance on both dimensions.

### Thinking Format Preservation

| Method | $\mathcal{L}_{\text{think}}$ | `</think>` Missing Rate |
|--------|------------------------------|------------------------|
| Task Arithmetic | 0.1224 | 6.4% |
| **RAIN-Merging** | **0.0065** | **0.0%** |

Null-space projection reduces KL divergence from 0.12 to 0.006, completely eliminating the `</think>` missing problem.

### Agent Scenarios

| Model | ALFWorld | WebShop |
|-------|----------|---------|
| ITM | 17.50 | 10.45 |
| LRM | 22.00 | 26.63 |
| **RAIN-Merging** | **25.00** | **29.42** |

### Key Findings

1. **Reasoning and instruction following are orthogonal capabilities**: the cosine similarity between the principal subspaces of task vectors is < 0.1.
2. **Protecting the thinking format is critical**: naive merging corrupts the `<think>` structure and causes reasoning degradation.
3. **Enhancing instruction following can reciprocally improve reasoning**: better instruction comprehension leads to higher-quality chain-of-thought.
4. **Gradient-free method vs. SFT**: RAIN-Merging completes in ~21 minutes vs. 120 minutes for SFT, while achieving superior performance.

## Highlights & Insights

1. **Precise problem identification**: Starting from a parameter-space orthogonality analysis, the method simultaneously identifies the risk of output format mismatch.
2. **Elegance of null-space projection**: Linear-algebraic tools are used to guarantee invariance of the thinking format while maximizing instruction-following capability.
3. **Interpretability of instruction attention**: Alignment and leakage scores quantify each attention head's responsiveness to instructions.
4. **Strong practicality**: No training required; only ~500 calibration samples and 20 minutes are needed to significantly improve LRM instruction following—far superior to SFT.
5. **Complete theoretical guarantees**: Proposition 1 proves that null-space projection satisfies the KL divergence constraint.

## Limitations & Future Work

1. The additional null-space computation and attention statistics collection, while faster than SFT, are approximately 20× slower than the simplest Task Arithmetic baseline.
2. The method depends on calibration dataset quality—the selection of reasoning and instruction calibration sets may affect results.
3. Only Q, K, V, O, and FFN parameters are merged; other modules (e.g., embeddings) are not considered.
4. Performance gains diminish at the 32B scale, and applicability to very large models remains to be verified.
5. The method assumes that the LRM and ITM share the same base model, making it inapplicable to models with entirely different architectures.

## Related Work & Insights

- **Task Arithmetic (Ilharco et al. 2023)**: The foundation of RAIN-Merging; however, naive linear addition corrupts the reasoning format.
- **TIES / DARE**: Data-agnostic task vector pruning methods; RAIN-Merging surpasses these through data-driven constraints.
- **AIM (Nobari et al. 2025)**: Activation-aware merging that does not account for output format mismatch.
- **Guardieiro et al. 2025**: The inspiration for instruction attention analysis; RAIN-Merging systematizes this into merging coefficients.
- **Insight**: Model merging requires not only parameter-space analysis but also consideration of output distribution format compatibility.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic treatment of thinking format preservation in LRM+ITM merging.
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Rigorous derivation of null-space projection and complete proof of the KL divergence constraint.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 IF benchmarks + 9 reasoning benchmarks + 5 model scales + agent scenarios.
- **Value**: ⭐⭐⭐⭐⭐ — Gradient-free, 20 minutes, 500 samples suffice to significantly enhance LRM instruction following.
- **Overall**: ⭐⭐⭐⭐⭐ — Important problem, elegant method, rigorous theory, and comprehensive experiments; an outstanding contribution to the model merging field.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format](rain-merging_a_gradient-free_method_to_enhance_instruction_following_in_large_re.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ICLR 2026\] On The Fragility of Benchmark Contamination Detection in Reasoning Models](on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[ICLR 2026\] Estimating the Empowerment of Language Model Agents](estimating_the_empowerment_of_language_model_agents.md)

<!-- RELATED:END -->
