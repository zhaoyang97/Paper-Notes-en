---
title: >-
  [Paper Note] SineProject: Machine Unlearning for Stable Vision–Language Alignment
description: >-
  [CVPR 2026][LLM Safety][Machine Unlearning] To address the severe ill-conditioning of the projector Jacobian during machine unlearning in MLLMs—which causes systematic vision–language alignment drift—this paper proposes…
tags:
  - "CVPR 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Multimodal Large Language Models"
  - "Vision–Language Alignment"
  - "Projector Stability"
  - "Jacobian Condition Number"
date: 2026-05-08
content_hash: 5f68c363f8ca3bad
---

# SineProject: Machine Unlearning for Stable Vision–Language Alignment

**Conference**: CVPR 2026
**arXiv**: [2511.18444](https://arxiv.org/abs/2511.18444)  
**Code**: [Available](https://github.com/arpit2412/SineProject)  
**Area**: LLM Safety
**Keywords**: Machine Unlearning, Multimodal Large Language Models, Vision–Language Alignment, Projector Stability, Jacobian Condition Number

## TL;DR

To address the severe ill-conditioning of the projector Jacobian during machine unlearning in MLLMs—which causes systematic vision–language alignment drift—this paper proposes SineProject, which applies sinusoidal modulation ($\sin(\Delta W)$) to projector weights to constrain parameter magnitudes to $[-1, 1]$. This reduces the Jacobian condition number by 3–4 orders of magnitude, achieving complete forgetting of target knowledge while reducing the safe answer rejection rate (SARR) on benign queries by 15%.

## Background & Motivation

### 1. State of the Field

Multimodal large language models (MLLMs, e.g., LLaVA, BLIP-2, GPT-4V) are increasingly deployed in safety-sensitive settings such as medical diagnosis and content moderation. Privacy regulations (e.g., GDPR) and security requirements demand that models be capable of **selectively forgetting** specific knowledge (unsafe content, private information) without full retraining.

### 2. Limitations of Prior Work

Existing unlearning methods are primarily designed for text-only LLMs (e.g., Gradient Ascent, KL divergence minimization, Preference Optimization). When directly transferred to MLLMs, they **fail catastrophically**:

- SafeEraser reports that gradient-based methods on LLaVA-1.5-7B yield a safe answer rejection rate (SARR) of **100%**—the model refuses not only harmful queries but all benign ones as well.
- MLLMU-Bench demonstrates severe capability degradation in privacy entity unlearning tasks.

### 3. Root Cause

Unlike text-only LLMs, MLLMs achieve geometric coupling between visual and linguistic representations through carefully trained **projector layers**. Unlearning must erase target knowledge while preserving this cross-modal geometric alignment—a fundamentally conflicting objective.

### 4. Paper Goals

The authors attribute the failure to **Alignment Drift**—the systematic degradation of vision–language geometric alignment during unlearning—manifested in three interrelated phenomena:

- **Spectral instability**: The Jacobian condition number of the projector grows by 3–4 orders of magnitude during unlearning.
- **Modal decoupling**: Visual and linguistic embeddings deviate from optimal alignment.
- **Representational collapse**: The model loses the ability to distinguish harmful from benign content, leading to indiscriminate refusal.

### 5. Starting Point

Existing methods modify the language backbone or visual encoder, overlooking the **projector layer** as the sole conduit for cross-modal information flow. The authors redirect attention to the spectral conditioning properties of the projector's Jacobian.

### 6. Core Idea

A trainable residual $\Delta W$ is appended to the frozen pretrained weight $W$, and a sinusoidal transformation $\sin(\Delta W)$ is applied to $\Delta W$, bounding all updates to $[-1, 1]$. This acts as an **implicit spectral regularizer**, constraining the spectral properties of the Jacobian and preventing condition number explosion during unlearning.

## Method

### Overall Architecture

The core architecture of SineProject is minimal: it modifies only the parameterization of the projector layer within a standard MLLM (visual encoder + projector MLP + language model), leaving the architecture and loss functions unchanged. It is compatible with any existing unlearning pipeline.

The base MLLM projector is a two-layer MLP: $F(x) = W_2 \phi(W_1 x + b_1) + b_2$, where $\phi$ is a GELU/ReLU activation.

### Key Designs

**Design 1: Sine Projector**

- **Function**: Wraps the projector weight matrices with the sine function.
- **Mechanism**: Defines the regularized MLP as $G(x) = \sin(W_2)\phi(\sin(W_1)x + b_1) + b_2$, where $\sin(\cdot)$ is applied element-wise.
- **Design Motivation**: Since $\sin/\cos$ has range $[-1, 1]$, this guarantees that the three Jacobian blocks $\nabla_{W_1}G$, $\nabla_{W_2}G$, and $\nabla_{b_2}G$ are all bounded (Theorem 3.1); only $\nabla_{b_1}G$ may remain unbounded. In contrast, the Jacobian blocks of a standard MLP can grow arbitrarily as $W_1, W_2$ increase.

**Design 2: Pretrained Knowledge–Preserving Fine-tuning Strategy**

- **Function**: Freezes the original pretrained weights $W$, introduces randomly initialized $\Delta W$, and optimizes only $\Delta W$.
- **Mechanism**: The final weight is $W + \sin(\Delta W)$, i.e., $(W_2 + \sin(\Delta W_2))\phi((W_1 + \sin(\Delta W_1))x + b_1) + b_2$.
- **Design Motivation**: Directly applying $\sin$ to pretrained weights would overwrite learned knowledge. By freezing $W$ and optimizing only the $\sin(\Delta W)$ increment, the method preserves pretrained knowledge while obtaining spectral regularization benefits. This is essentially a **fully dense adapter**.

**Design 3: Prompt Decoupling (PD)**

- **Function**: Processes text-only samples and multimodal samples with separate losses during the unlearning phase.
- **Mechanism**: Inherited from SafeEraser; $D_f^{(\text{text})}$ and $D_f^{(\text{mm})}$ compute losses independently.
- **Design Motivation**: Mitigates over-forgetting; experiments demonstrate that PD yields significant improvements in SARR.

### Loss & Training

The unlearning objective follows a standard forget–retain trade-off:

$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{forget}}(\theta; D_f) + \lambda \mathcal{L}_{\text{retain}}(\theta; D_r)$$

- $\mathcal{L}_{\text{forget}}$: supports Gradient Descent, KL divergence minimization, or Preference Optimization (main experiments use PO+PD).
- $\mathcal{L}_{\text{retain}}$: preserves performance on the retain set.
- During training, the visual encoder is frozen; LoRA adapters (rank 32) and the sine projector ($\Delta W_1, \Delta W_2, b_1, b_2$) are trained.
- Parameter overhead is $<1\%$.

## Key Experimental Results

### Main Results

**Table 1: SafeEraser Benchmark (Safety Unlearning)**

Evaluated on LLaVA-v1.5-7B and 13B. Forget Quality measures unlearning effectiveness (ASR↓, RR↑); Model Utility measures retained capability (ROUGE↑, GPT-Eval↑, Specificity↑, SARR↓):

| Method | ASR(Eff.)↓ | RR(Eff.)↑ | ASR(Gen.)↓ | RR(Gen.)↑ | ROUGE↑ | GPT↑ | Spec.↑ | SARR↓ |
|--------|-----------|----------|-----------|----------|--------|------|--------|-------|
| **LLaVA-7B** | | | | | | | | |
| GA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 15.3 | **100** |
| GD+PD | 2.8 | 0.0 | 0.5 | 0.4 | 61.6 | 82.8 | 50.7 | 28.0 |
| PO (w/o PD) | 0.1 | 100 | 0.1 | 100 | 65.2 | 85.4 | 63.7 | **100** |
| SafeEraser (PO+PD) | 0.2 | 100 | 0.2 | 99.7 | 65.4 | 86.2 | 64.4 | 30.3 |
| **SineProject (PO+PD)** | **0.1** | **100** | **0.1** | **99.9** | **65.8** | **86.3** | **65.2** | **25.8** |
| **LLaVA-13B** | | | | | | | | |
| SafeEraser (PO+PD) | 2.2 | 99.5 | 2.4 | 99.1 | 62.7 | 81.7 | 65.3 | 27.3 |
| **SineProject (PO+PD)** | **1.6** | **99.8** | **0.8** | **99.9** | **63.9** | **82.9** | **65.4** | **25.1** |

Key finding: SineProject maintains 100% forgetting while reducing SARR from 30.3% to 25.8% (7B) and from 27.3% to 25.1% (13B), substantially reducing false refusals on benign queries.

**Table 2: MLLMU-Bench (Privacy Unlearning, LLaVA-7B, Avg.↑)**

| Method | 5% Deletion Avg.↑ | 10% Deletion Avg.↑ | 15% Deletion Avg.↑ |
|--------|------------------|------------------|------------------|
| GA | 45.7 | 50.4 | 50.9 |
| Grad. Diff. | 50.2 | 56.8 | 51.4 |
| NPO | 51.8 | 44.5 | 53.5 |
| MMUnlearner | 53.9 | 52.4 | 51.8 |
| **SineProject (NPO)** | **62.1** | **68.4** | **66.2** |

Key finding: SineProject achieves substantially higher aggregate scores across all deletion ratios (8–16 points above the strongest baseline MMUnlearner), with the advantage increasing as the deletion ratio grows, validating the importance of geometric stability for scalable unlearning.

### Ablation Study

- **Function selection**: $\sin(\Delta W)$ achieves a condition number of $5.40\times10^2$, far superior to spectral norm ($1.15\times10^5$), weight clipping, LoRA, tanh, and sigmoid; SARR is 25.8% vs. 34.1%.
- **Layer necessity**: Joint modulation of $W_1 + W_2$ (25.8%) outperforms modulating only $W_2$ (26.5%).
- **Loss generalization**: Consistently reduces SARR by 0.8–4.5% across GD, KL, and PO losses, with RR maintained above 99%.
- **Robustness**: SARR varies by $<0.3\%$ for $\alpha \in [1, 300]$ ($p = 0.83$); variance across 10 seeds is reduced by 74%.
- **Architecture generalization**: Reduces SARR by 14.9–20.1% on both MLP and attention-based projectors.

### Key Findings

1. **Geometric stability is central**: SafeEraser's $W_2$ Jacobian condition number exceeds $10^6$ during unlearning, and the MIR deviates from the optimal interval to $>4.5$; SineProject keeps the condition number below $10^3$ and stabilizes MIR at $\approx 2.7$ (within the optimal interval $[2.5, 3.0]$).
2. **Spectral dynamics**: Baselines exhibit explosive growth in $\sigma_{\max}$ and collapse of $\sigma_{\min}$; SineProject keeps both stable.
3. **Strong correlation between condition number and SARR**: $r = 0.89$ ($p < 0.01$), confirming the practical significance of the theoretical analysis.
4. **Training dynamics reversal**: Baselines worsen condition number by $3.3\times$; SineProject improves it by $13.4\times$.

## Highlights & Insights

1. **Precise problem localization**: This work is the first to systematically analyze the mechanism of "alignment drift" in multimodal unlearning, translating abstract alignment collapse into a quantifiable, diagnosable spectral metric via the Jacobian condition number.
2. **Minimal and elegant method**: A single $\sin(\cdot)$ transformation—requiring no architectural changes, no loss modifications, and $<1\%$ parameter overhead—yields condition number improvements of 3–4 orders of magnitude.
3. **Closed loop between theory and experiment**: Theorem 3.1 rigorously proves the Jacobian boundedness of the sine projector, and experiments precisely validate the theoretical predictions.
4. **Plug-and-play**: Compatible with multiple unlearning losses (GD/KL/PO) and can be directly integrated into existing unlearning pipelines.

## Limitations & Future Work

1. **Architectural scope**: The method is primarily optimized for MLP projector layers; while generalization experiments are conducted on Q-Former/Resampler, validation on Flamingo-style deeply interleaved cross-modal architectures remains absent.
2. **Semantic entanglement**: Geometric conditioning preserves alignment structure but does not resolve the semantic entanglement of related concepts—a capacity–forgetting trade-off unrelated to conditioning emerges when more than 25% of the knowledge base is forgotten.
3. **Lack of certified unlearning guarantees**: Adversarial fine-tuning post-unlearning may partially recover forgotten information; integration with certified defense mechanisms is needed.
4. **Projector-only scope**: Joint optimization of sinusoidal modulation with LoRA adapters is not explored (noted by the authors as future work).

## Related Work & Insights

- **Machine unlearning benchmarks**: TOFU, MUSE (unimodal) → SafeEraser, MLLMU-Bench (multimodal)—the evaluation ecosystem for multimodal unlearning is rapidly maturing.
- **Geometry of multimodal alignment**: Contrastive learning alignment in CLIP/LiT → this work reveals that such alignment is extremely fragile under unlearning.
- **NTK theory application**: Extending Jacobian condition number analysis from pretraining/fine-tuning to the unlearning setting represents a novel application of the NTK perspective.
- **Inspiration**: The bounded-transformation paradigm may generalize to other settings requiring "stable modification," such as continual learning and model editing.

## Rating

⭐⭐⭐⭐ The problem is identified with exceptional precision; the method is minimal with rigorous theoretical support; and the approach achieves comprehensive state-of-the-art results on two benchmarks. The primary limitation is the restriction to projector layers, leaving applicability to broader architectures to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Forgetting is Not Erasing: A Survey of Reversibility in Large Language Model Machine Unlearning](../../ICML2026/llm_safety/unlearning_isnt_deletion_investigating_reversibility_of_machine_unlearning_in_ll.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](../../ICLR2026/llm_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[ICML 2026\] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](../../ICML2026/llm_safety/dualoptim_bridging_shared_and_decoupled_optimizer_states_for_better_machine_unle.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/llm_safety/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)

</div>

<!-- RELATED:END -->
