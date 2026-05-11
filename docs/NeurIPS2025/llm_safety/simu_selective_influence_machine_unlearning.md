---
title: >-
  [Paper Note] SIMU: Selective Influence Machine Unlearning
description: >-
  [NeurIPS 2025][LLM Safety][Machine Unlearning] SIMU proposes a two-stage framework: it first identifies critical MLP neurons encoding forget-set information via gradient aggregation…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Second-Order Optimization"
  - "Neuron Localization"
  - "Influence Functions"
date: 2026-05-08
content_hash: b3ae477711487eb0
---

# SIMU: Selective Influence Machine Unlearning

**Conference**: NeurIPS 2025
**arXiv**: [2510.07822](https://arxiv.org/abs/2510.07822)
**Code**: Not released
**Area**: LLM Safety
**Keywords**: Machine Unlearning, LLM Safety, Second-Order Optimization, Neuron Localization, Influence Functions

## TL;DR

SIMU proposes a two-stage framework: it first identifies critical MLP neurons encoding forget-set information via gradient aggregation, then applies second-order (Sophia) optimization exclusively to those neurons, achieving effective unlearning while substantially preserving the model's original capabilities.

## Background & Motivation

Large language models (LLMs) memorize sensitive information from training data, raising serious concerns regarding data privacy and AI safety. Machine unlearning aims to make a model precisely "forget" the influence of specific data without retraining from scratch.

Existing unlearning methods face a fundamental tension:

**Gradient ascent methods** (e.g., Gradient Ascent) tend to over-forget, causing severe degradation of overall model capability.

**Fine-tuning on the retain set only** often results in insufficient forgetting, leaving residual influence of the target data.

**Regularization methods** (e.g., GradDiff, NPO) attempt to balance both objectives but still harm model utility under aggressive unlearning scenarios.

**Second-order optimization methods** (e.g., SOUL) leverage Hessian information for more precise parameter updates and perform better, yet Hessian approximation errors accumulate over iterations.

Key insight: **no prior work has combined localization-aware techniques with second-order influence function unlearning** — that is, simultaneously addressing *where* to update and *how* to update. SIMU fills this gap.

## Method

### Overall Architecture

SIMU is a two-stage framework:

- **Stage 1**: Critical Neuron Identification — locating which MLP neurons primarily encode forget-set information.
- **Stage 2**: Selective Influence Unlearning — applying second-order optimization updates exclusively to those critical neurons.

### Key Design 1: Critical Neuron Identification

Building on Meng et al.'s finding that MLP layers in Transformers function as key-value memories storing factual knowledge, SIMU extends the Privacy Neuron Detector with a gradient aggregation scheme designed for autoregressive language models:

**Step 1**: Convert forget-set QA pairs into multiple next-token prediction samples.

**Step 2**: For the $k$-th MLP down-projection neuron $w_l^k$ in layer $l$, obtain its raw activation $\beta_{l,i}^k$ on sample $i$.

**Step 3**: Uniformly scale the activation from 0 to $\beta_{l,i}^k$ in $m$ steps, computing the loss change at each step.

**Step 4**: Compute the attribution score:

$$\text{Att}(w_l^k) = \frac{1}{m} \sum_{j=1}^{m} \sum_{i=1}^{|D|} \beta_{l,i}^k \frac{\partial L_i(a_{j,l,i}^k)}{\partial a_{j,l,i}^k}$$

**Step 5**: Apply per-layer thresholding to generate binary masks. Let $M_l = \max_k \text{Att}(w_l^k)$; a neuron is marked critical if $\text{Att}(w_l^k) > t \cdot M_l$, where $t \in (0,1]$ controls the proportion of neurons selected per layer.

### Key Design 2: Selective Influence Function Unlearning

Stage 2 fine-tunes the model using the Sophia optimizer within a second-order iterative framework:

- **Frozen**: all parameters except attention projection layers and MLP down-projection layers.
- **Sparse updates**: within MLP layers, only critical neurons are updated (controlled by mask $\mathbf{M}$).
- **Full updates**: attention layers receive unrestricted updates to preserve sequence modeling capability.

Each Sophia update is a clipped quasi-Newton step:

$$\theta_{t+1} = \theta_t - \eta_t \cdot \text{clip}\left(\frac{m_t}{\max\{\gamma H_t, \epsilon\}}, 1\right)$$

where $m_t$ is the EMA of the first-order momentum and $H_t$ is the EMA of the diagonal Gauss-Newton Hessian approximation.

### Loss & Training

The mask is applied at three critical points to ensure non-critical neurons remain entirely unaffected:

1. **After first-order momentum EMA**: $m_t = \mathbf{M} \odot m_t' + \bar{\mathbf{M}} \odot m_{t-1}$
2. **After second-order curvature EMA**: $H_t = \mathbf{M} \odot H_t' + \bar{\mathbf{M}} \odot H_{t-1}$
3. **After parameter update**: $\theta_t = \mathbf{M} \odot \theta_t' + \bar{\mathbf{M}} \odot \theta_{t-1}$

This triple-masking mechanism ensures that unlearning updates are strictly confined to critical neurons, minimizing collateral damage to retained knowledge. The unlearning objective follows GradDiff: gradient ascent on the forget set (increasing loss) and gradient descent on the retain set (maintaining loss), combined with a weighting coefficient.

## Key Experimental Results

### Main Results

Evaluation is conducted on two benchmarks, TOFU and LUME, using LLaMA2-7B and OLMo-1B.

**TOFU Benchmark Results**:

| Model | Method | Aggregate ↑ | Forget EM ↓ | Retain EM ↑ | World Facts EM ↑ |
|-------|--------|:-----------:|:-----------:|:-----------:|:----------------:|
| LLaMA2-7B | FO-GradDiff | 0.4738 | 72.75% | 76.50% | 79.49% |
| LLaMA2-7B | SO-GradDiff | 0.7957 | 10.25% | 72.25% | 82.05% |
| LLaMA2-7B | **SIMU-GradDiff** | **0.7963** | 20% | **78.00%** | **82.90%** |
| OLMo-1B | FO-GradDiff | 0.7059 | 26.50% | 63.00% | 0.85% |
| OLMo-1B | SO-GradDiff | 0.8235 | 22.75% | 78.00% | 38.46% |
| OLMo-1B | **SIMU-GradDiff** | **0.8438** | **10.25%** | 75.50% | **42.74%** |

**LUME Benchmark Results**:

| Model | Method | Aggregate ↑ | Forget Overall ↓ | Utility Overall ↑ |
|-------|--------|:-----------:|:----------------:|:-----------------:|
| LLaMA2-7B | SO-GradDiff | 0.607 | 0.0187/0.00 | 0.7714/0.6212 |
| LLaMA2-7B | **SIMU** | **0.659** | **0.0025/0.00** | **0.8295/0.7149** |
| OLMo-1B | SO-GradDiff | 0.728 | 0.0055/0.0 | 0.9244/0.8499 |
| OLMo-1B | **SIMU** | **0.740** | **0.0015/0.0** | **0.9365/0.8540** |

### Ablation Study

- For LLaMA2-7B, SIMU yields approximately **5–6%** utility improvement over SO-GradDiff.
- For OLMo-1B, the utility gain is approximately **1–2%**.
- The disparity is attributed to the forget-set signal in LLaMA2-7B being more concentrated in a smaller set of critical neurons.

### Key Findings

1. **Larger models benefit more**: LLaMA2-7B gains more from SIMU than OLMo-1B, as its forget-set signal is more concentrated.
2. **Sparse MLP + full Attention is the optimal combination**: achieving an ideal balance between forgetting and utility preservation.
3. **Critical neuron masking effectively reduces propagation of Hessian approximation errors**.

## Highlights & Insights

1. **Elegant integration of theory and practice**: the theoretical insight that "MLP layers serve as factual memories" is translated into a practical unlearning strategy.
2. **Sophisticated triple-masking design**: masks are applied simultaneously at the first-order momentum, second-order curvature, and parameter update stages, strictly constraining updates.
3. **Progressive activation scaling** for attribution provides more precise attribution than naive gradient attribution.
4. **Full Attention updates + sparse MLP updates** reflects a clear design intuition: preserve contextual modeling capacity while correcting only factual storage.
5. The first work to combine localization-aware techniques with second-order influence function unlearning, filling a notable research gap.

## Limitations & Future Work

1. **Computational overhead**: the critical neuron identification stage requires multi-step activation scaling and gradient computation per neuron, incurring substantial cost on large models.
2. **Sensitivity to threshold $t$**: threshold selection has a significant impact, yet the paper provides limited discussion of automatic tuning strategies.
3. **Applicable only to fixed forget sets**: continual unlearning requests require recomputing critical neuron masks for each new request.
4. **Limited evaluation scope**: experiments are conducted on only two benchmarks, lacking validation on larger models (e.g., 70B).
5. Future work may explore extending neuron attribution to attention layers for finer-grained selective unlearning.

## Related Work & Insights

- **SOUL** (Jia et al., 2024): foundational work on second-order influence function unlearning; SIMU builds upon it by adding localization.
- **ROME/MEMIT** (Meng et al., 2022/2023): theoretical basis for MLP layers as factual memories.
- **Privacy Neuron Detector** (Wu et al., 2023): neuron-level privacy detection; SIMU extends this to autoregressive models.
- **GradDiff** (Liu et al., 2022): a classical gradient differencing unlearning method.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: combining localization-aware techniques with second-order optimization is a natural yet effective contribution.
- **Experimental Thoroughness** ⭐⭐⭐⭐: comprehensive evaluation across two benchmarks and two models with consistent improvements.
- **Theoretical Depth** ⭐⭐⭐: intuitive explanations are provided, but rigorous theoretical analysis is lacking.
- **Practical Value** ⭐⭐⭐⭐: the method is relatively straightforward and directly applicable to existing LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](../../ICLR2026/llm_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](../../ACL2026/llm_safety/forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)

</div>

<!-- RELATED:END -->
