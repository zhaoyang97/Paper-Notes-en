---
title: >-
  [Paper Note] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models
description: >-
  [CVPR 2026][LLM Alignment][jailbreak defense] This paper proposes NullSteer, an activation steering defense framework based on null-space projection, which effectively resists visual jailbreak attacks without degrading general model capability by constraining steering operations to the null space of benign activations.
tags:
  - CVPR 2026
  - LLM Alignment
  - jailbreak defense
  - activation steering
  - null-space projection
  - VLM safety
  - inference-time defense
date: 2026-05-08
content_hash: 7d0c897a4593e2d3
---

# Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.22094](https://arxiv.org/abs/2603.22094)
**Code**: None
**Area**: LLM Alignment / VLM Safety
**Keywords**: jailbreak defense, activation steering, null-space projection, VLM safety, inference-time defense

## TL;DR

This paper proposes NullSteer, an activation steering defense framework based on null-space projection, which effectively resists visual jailbreak attacks without degrading general model capability by constraining steering operations to the null space of benign activations.

## Background & Motivation

Vision-language models (VLMs) deployed in open-world scenarios are highly vulnerable to visual jailbreak attacks, where adversaries add adversarial perturbations to images or embed malicious instructions to hijack the model into generating harmful content. Existing defenses fall into three categories:

**Training-time defenses** (e.g., adversarial training, safety fine-tuning): computationally expensive, require additional annotated data.

**Inference-time defenses** (e.g., prompt rewriting, multi-turn detection): high latency, low efficiency.

**Activation Steering**: lightweight and training-free, but with a critical flaw.

Activation steering injects a "refusal direction vector" into the model's hidden states to guide safe outputs, making it an efficient inference-time defense. However, **it does not distinguish between benign and malicious inputs**—the refusal vector affects all inputs equally, causing benign requests to also be incorrectly refused (i.e., the over-refusal problem), which severely degrades general model capability. The authors observe that benign activations are also shifted after steering, which explains the root cause of over-refusal.

The core motivation is: can one design a **selective steering** mechanism that applies steering only to malicious inputs while remaining "transparent" to benign inputs?

## Method

### Overall Architecture

NullSteer constructs a linear transformation matrix $\Delta$ whose projection onto the benign activation subspace is zero (producing no perturbation), while dynamically guiding the model toward refusal semantics along malicious directions. The entire process requires no training and only needs a small number of benign and malicious samples to estimate the projection matrix.

At inference time, hidden states at each layer are modified as:
$$\mathbf{h}^{(l)'} = \mathbf{h}^{(l)} + \lambda \tilde{\Delta}^{*(l)} \mathbf{P}^{(l)} \mathbf{h}^{(l)}$$

### Key Designs

1. **Construction of the null-space projection matrix P**: the core guarantee that benign inputs are undisturbed.

    - Collect hidden states of $N_b$ benign inputs to form matrix $\mathbf{H}_b \in \mathbb{R}^{d \times N_b}$.
    - Require the transformation matrix to satisfy $\Delta \mathbf{H}_b = \mathbf{0}$, i.e., zero perturbation on benign activations.
    - Exploit the equivalence $\text{Null}(\mathbf{H}_b) = \text{Null}(\mathbf{H}_b \mathbf{H}_b^\top)$ to reduce computation from $d \times N_b$ to $d \times d$.
    - Apply SVD to the covariance matrix and select eigenvectors corresponding to near-zero eigenvalues to form the projection matrix: $\mathbf{P} = \hat{\mathbf{U}} \hat{\mathbf{U}}^\top$.
    - **Design Motivation**: provides a mathematical guarantee that benign input activations remain completely unchanged before and after steering.

2. **Steering learning for malicious directions**: guiding malicious inputs toward refusal semantics.

    - Collect hidden states $\mathbf{H}_m$ of $N_m$ malicious inputs.
    - Objective: $\tilde{\Delta} \mathbf{P} \mathbf{H}_m = \mathbf{R}$ (target activations in the refusal direction).
    - A closed-form solution exists; no iterative training is required.

3. **Harmful direction suppression term**: further eliminating residual jailbreak semantics.

    - The harmful direction $\mathbf{V}$ is extracted by masking visually salient tokens and measuring activation changes.
    - A term $\|\tilde{\Delta} \mathbf{P} \mathbf{H}_m - \mathbf{V}\|_F^2$ is added to the optimization objective to explicitly suppress jailbreak-related features.

### Loss & Training

NullSteer's optimization objective consists of three terms and admits a closed-form solution:

$$\tilde{\Delta}^* = \arg\min_{\tilde{\Delta}} \left( \|\tilde{\Delta}\mathbf{P}\mathbf{H}_m - \mathbf{R}\|_F^2 + \alpha\|\tilde{\Delta}\mathbf{P}\|_F^2 + \beta\|\tilde{\Delta}\mathbf{P}\mathbf{H}_m - \mathbf{V}\|_F^2 \right)$$

- First term: aligns malicious activations to refusal semantics.
- Second term: regularization ensuring smoothness of the transformation.
- Third term: suppresses residual jailbreak feature directions.

The closed-form solution is obtained directly via the Moore-Penrose pseudoinverse, **requiring no gradient-based training whatsoever**.

## Key Experimental Results

### Main Results

Evaluated on three VLMs (MiniGPT-4, Qwen2-VL, LLaVA-v1.5) against PGD perturbation attacks:

| Model | Metric | NullSteer | ASTRA (Prev. SOTA) | No Defense |
|------|------|-----------|---------------|--------|
| MiniGPT-4 (unconstrained) | Toxicity ↓ | **2.89%** | 4.48% | 52.12% |
| MiniGPT-4 (unconstrained) | ASR ↓ | **7.32%** | 9.09% | 53.64% |
| Qwen2-VL (ε=32/255) | Toxicity ↓ | **3.51%** | 5.45% | 51.62% |
| Qwen2-VL (ε=32/255) | ASR ↓ | **4.55%** | 5.00% | 70.46% |
| LLaVA-v1.5 (ε=32/255) | Toxicity ↓ | **31.82%** | 34.76% | 84.40% |
| LLaVA-v1.5 (ε=32/255) | ASR ↓ | **8.75%** | 10.91% | 56.36% |

General capability (Utility) preservation:

| Model | MM-Vet | MMBench | XSTest |
|------|--------|---------|--------|
| MiniGPT-4 Original | 19.40 | 35.90 | 87.60 |
| MiniGPT-4 NullSteer | **21.05** | **36.25** | **87.80** |
| Qwen2-VL Original | 49.13 | 78.00 | 73.60 |
| Qwen2-VL NullSteer | 49.02 | **78.82** | **74.50** |

### Ablation Study

| Configuration | Toxicity ↓ | ASR ↓ | Utility ↑ | Notes |
|------|-----------|-------|-----------|------|
| No defense | 30.65% | 34.55% | 35.90 | Baseline |
| Regularization + refusal alignment | 3.58% | 8.36% | 36.00 | Without harmful suppression |
| Regularization + harmful suppression | 4.02% | 8.57% | 36.00 | Without refusal alignment |
| All three terms | **2.89%** | **7.32%** | **36.25** | Complementary |

### Key Findings

- Only approximately 8 benign samples are needed to construct a stable null-space projection.
- Approximately 100 malicious samples are sufficient to estimate the harmful direction.
- A steering strength of $\lambda \approx 5$ achieves the best balance between safety and utility.
- The method remains effective under adaptive attacks (where the adversary knows the defense)—Jailbreak ASR drops from 49.1% to 19.3%.

## Highlights & Insights

1. **Theoretical elegance**: the safety alignment problem is reformulated as a null-space constrained optimization, providing a mathematical guarantee of benign representation invariance—a first in VLM defense.
2. **Training-free**: the entire method admits a closed-form solution with no fine-tuning or gradient descent, adding virtually no latency at inference time.
3. **Selective mechanism**: the over-refusal problem of conventional activation steering is cleanly resolved—benign input activations are entirely unaffected.
4. **Cross-model generalization**: consistent performance across three VLMs with different architectures demonstrates the universality of the null-space constraint.

## Limitations & Future Work

1. Relies on a linearity assumption—benign and malicious activation distributions are assumed to be separable via linear subspaces, which may fail against highly nonlinear attacks.
2. The null-space dimension $r$ must be predefined; different models and layers may require different settings.
3. Evaluation is currently limited to perturbation-based attacks; assessment against typography-based visual attacks is relatively limited.
4. The choice of steering layer $l$ depends on manual heuristics (layer 20 for 13B models, layer 14 for 7B models), lacking an adaptive selection mechanism.

## Related Work & Insights

- **AlphaEdit**: first applied null-space projection to protect existing knowledge in LLM knowledge editing, serving as the core inspiration for this work.
- **ASTRA**: the primary comparison baseline, using adaptive activation steering but lacking null-space constraints.
- Null-space constraints have been widely validated in continual learning (GNSP, NS-Net, etc.); this work is the first to introduce them into VLM safety alignment.
- Insight: null-space projection provides a general "selective control" paradigm that may extend to broader scenarios requiring "changing certain behaviors while preserving others."

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying null-space projection to VLM safety is a novel combination, though null-space methods themselves are widely used in CL and knowledge editing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three models, multiple attack strengths, adaptive attacks, complete ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are clear and motivation is well articulated.
- Value: ⭐⭐⭐⭐ — Provides a practical and theoretically interpretable solution for VLM safety defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](../../ICLR2026/llm_alignment/alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[ICLR 2026\] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](../../ICLR2026/llm_alignment/reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)
- [\[ACL 2026\] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](../../ACL2026/llm_alignment/s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)
- [\[ACL 2026\] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](../../ACL2026/llm_alignment/trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)

</div>

<!-- RELATED:END -->
