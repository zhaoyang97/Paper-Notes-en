---
title: >-
  [Paper Note] RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Machine Unlearning] RAZOR selects the most critical layers and attention heads via ratio-aware gradient scoring that jointly measures forgetting pressure and retention alignment, and achieves precise, efficient targeted unlearning on CLIP, Stable Diffusion, and VLMs through a three-component constrained loss and an iterative expansion mechanism, with no performance degradation after quantization.
tags:
  - CVPR 2026
  - Image Generation
  - Machine Unlearning
  - Ratio-Aware Editing
  - Multi-Layer Selection
  - Model Unlearnability
  - Quantization Robustness
date: 2026-05-08
content_hash: 6e25791a5cd8e8ad
---

# RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2603.14819](https://arxiv.org/abs/2603.14819)
**Code**: [https://github.com/raviranjan-ai/RAZOR-cvpr2026](https://github.com/raviranjan-ai/RAZOR-cvpr2026)
**Area**: Model Security / Machine Unlearning
**Keywords**: Machine Unlearning, Ratio-Aware Editing, Multi-Layer Selection, Model Unlearnability, Quantization Robustness

## TL;DR

RAZOR selects the most critical layers and attention heads via ratio-aware gradient scoring that jointly measures forgetting pressure and retention alignment, and achieves precise, efficient targeted unlearning on CLIP, Stable Diffusion, and VLMs through a three-component constrained loss and an iterative expansion mechanism, with no performance degradation after quantization.

## Background & Motivation

**State of the Field**: Large-scale vision-language models and diffusion models are trained on massive datasets and inevitably encode sensitive or undesired information. Regulations such as GDPR require models to "forget" specific data, yet retraining from scratch is prohibitively expensive, making machine unlearning an attractive alternative.

**Limitations of Prior Work**: Existing methods each exhibit distinct shortcomings. (1) Gradient-ascent approaches (e.g., SalUn) select parameters based solely on forgetting-set gradients, ignoring conflicts with the retention set, leading to incomplete forgetting or severe degradation of retained performance. (2) Single-layer editing methods (e.g., SLUG) are efficient but fragile—one layer is insufficient when knowledge is distributed across multiple layers. (3) Full-model update methods suffer drastic degradation of unlearning effectiveness after quantization.

**Root Cause**: Parameter selection considers only forgetting-set saliency, while retention conflicts are compensated post hoc. This sequential "forget-then-repair" strategy cannot avoid the coupling between forgetting and retention dynamics.

**Paper Goals**: (1) How to jointly consider forgetting pressure and retention alignment when selecting editing locations; (2) how to explicitly control the forgetting–retention trade-off in multi-layer editing; (3) how to ensure unlearning effectiveness does not collapse after quantization.

**Starting Point**: Compute gradients from both the forgetting set and the retention set simultaneously, and measure the "editing value" of each layer as the product of forgetting gradient magnitude and the orthogonality between forgetting and retention gradients. A high score indicates strong forgetting influence with low retention harm.

**Core Idea**: Jointly localize critical layers via ratio-aware scoring of forgetting/retention gradients, and achieve precise multi-layer unlearning through a constrained multi-objective loss and iterative expansion.

## Method

### Overall Architecture

Given a pretrained model together with forgetting and retention datasets, RAZOR operates in three stages: (1) compute forgetting and retention gradients for each layer and select a subset $\mathcal{K}$ of high-impact, low-harm layers via ratio-aware scoring; (2) perform one or a few gradient update steps on the selected layers using a three-component constrained loss; (3) if the forgetting metric has not reached the threshold, iteratively expand $\mathcal{K}$ until convergence. The entire pipeline is model-agnostic and applies uniformly to CLIP, Stable Diffusion, and VLMs.

### Key Designs

1. **Ratio-Aware Gradient Scoring**:

    - Function: Computes a composite score for each layer/attention head to determine whether it requires editing.
    - Mechanism: For each layer $l$, the forgetting gradient $g_l^f = \nabla_{\theta_l}\mathcal{L}_{\text{forget}}$ and retention gradient $g_l^r = \nabla_{\theta_l}\mathcal{L}_{\text{retain}}$ are computed in a single pass, yielding the score $\phi(l) = \frac{\|g_l^f\|_2}{\|\theta_l\|_2+\varepsilon} \cdot (1-\cos(g_l^f, g_l^r))^\alpha$. The first term measures the saliency of the forgetting gradient relative to the parameter magnitude; the second measures the directional divergence between forgetting and retention gradients—greater orthogonality indicates less harm to the retention set from editing.
    - Design Motivation: Prior methods rank parameters using only forgetting gradients and ignore retention conflicts. RAZOR jointly considers both at the selection stage, fundamentally avoiding the coupling problem inherent to the "forget-then-repair" paradigm.

2. **Three-Loss Objective**:

    - Function: Performs constrained updates on selected layers to explicitly balance forgetting, retention, and stability.
    - Mechanism: $\mathcal{L}_{\text{RAZOR}} = \mathcal{L}_{\text{retain}} + \lambda_f \rho \mathcal{L}_{\text{forget}} + \lambda_m \mathcal{L}_{\text{mismatch}}$. The retention loss preserves task performance (e.g., InfoNCE for CLIP); the forgetting loss drives forgetting alignment apart via gradient ascent (cosine embedding loss); the mismatch loss regularizes the drift of embedding similarities relative to the frozen model. The ratio hyperparameter $\rho$ explicitly controls forgetting intensity.
    - Design Motivation: A single forgetting objective either yields incomplete unlearning or damages retention. The three-component design decouples the three objectives, each providing its own gradient direction.

3. **Iterative Growing of $\mathcal{K}$**:

    - Function: Dynamically adds new layers if the initial set of selected layers fails to reach the forgetting threshold.
    - Mechanism: At each round, scores $\phi_t(l)$ are recomputed on the updated parameters, the highest-scoring unedited layer is added to $\mathcal{K}$ and updated, with a maximum of 6 iterations.
    - Design Motivation: Avoids over-editing caused by selecting too many layers at once; the progressive strategy ensures precise unlearning while controlling collateral damage.

### Loss & Training

The design is modular—the specific form of each loss component varies with the underlying model: CLIP uses InfoNCE + cosine embedding loss + similarity-drift regularization; Stable Diffusion uses denoising loss + text-encoder cosine loss + generation-guidance drift regularization; VLMs use InfoNCE + visual-encoder cosine loss + neutral-QA drift regularization. The per-layer learning rate $\eta_l$ is determined via lightweight binary search to identify the maximum stable step size.

## Key Experimental Results

### Main Results

| Method | CIFAR-10 M1↓ | CIFAR-10 M4↑ | CIFAR-10 M5↑ | ImageNet M1↓ | ImageNet M4↑ | LAION M1↓ | LAION M4↑ |
|--------|-------------|-------------|-------------|-------------|-------------|----------|----------|
| SSD | 52.00 | 25.00 | 97.50 | 52.50 | 30.00 | 42.00 | 48.00 |
| SalUn | 97.00 | 83.00 | 84.50 | 88.00 | 84.00 | 48.00 | 88.00 |
| SLUG | 67.50 | 87.50 | 96.50 | 68.00 | 88.00 | 48.00 | 88.00 |
| **RAZOR** | **52.50** | **89.00** | **100.00** | **53.50** | **92.00** | **40.00** | **94.00** |

SD-V3 UnlearnCanvas style/object unlearning:

| Method | Style UA↑ | Style IRA↑ | Style CRA↑ | Object UA↑ | Object IRA↑ | Object CRA↑ |
|--------|----------|-----------|-----------|-----------|------------|------------|
| ESD | 99.62 | 89.97 | 98.86 | 97.44 | 68.47 | 82.37 |
| SalUn | 90.36 | 92.33 | 97.02 | 91.06 | 98.35 | 99.59 |
| SLUG | 88.20 | 85.59 | 91.00 | 85.44 | 79.50 | 91.00 |
| **RAZOR** | **99.40** | **98.97** | **100.00** | **98.80** | **98.35** | **100.00** |

### Ablation Study

| Configuration | Efficiency Trade-off Score↑ | Time↓ | Memory↓ | Storage↓ |
|--------------|---------------------------|-------|---------|---------|
| ESD | 11.97 | 6163s | 17.8GB | 4.30GB |
| SLUG | 59.42 | 39s | 3.6GB | 0.04GB |
| **RAZOR** | **66.86** | 78s | 4.2GB | 0.06GB |

VLM (LLaVA-1.6-8B) identity unlearning: forgetting accuracy drops to 2.2%, while MME cognition/perception is maintained at 301/1362+ and GQA at 60+.

### Key Findings

- RAZOR leads across all five CLIP metrics; M3 (privacy leakage) and M5 (retention stability) both reach near-perfect levels.
- Quantization robustness is outstanding: under 4-bit quantization, RAZOR's metrics decline by only ~0.5%, whereas full-model update methods (SSD/SalUn) decline by 5–10%.
- RAZOR achieves the highest efficiency trade-off score (66.86), 12% above SLUG's 59.42, balancing unlearning quality with computational efficiency.
- Five of six metrics are best on SD-V3, demonstrating generalization to new architectures.

## Highlights & Insights

- **Core Insight of Ratio-Aware Scoring**: Incorporating the directional divergence between forgetting and retention gradients into the layer selection criterion addresses the fundamental design flaw of "forget-then-repair." This joint scoring strategy is transferable to any scenario requiring selective parameter updates, such as avoiding catastrophic forgetting in continual learning.
- **Extremely Small Editing Subset**: $|\mathcal{K}| \ll |\mathcal{L}|$; only modified weights are stored, yielding a storage footprint of 0.06 GB—far below the 4 GB+ of full-model methods—realizing truly surgical unlearning.
- **Model-Agnostic Modular Loss Design**: The same framework spans three major model families (CLIP/SD/VLM); adaptation requires only substituting the concrete loss functions in the respective loss table.

## Limitations & Future Work

- The method involves a relatively large number of hyperparameters ($\alpha$, $\rho$, $\tau$, $\lambda_f$, $\lambda_m$), which may require tuning across different models and tasks.
- The hard limit of six iterations in the iterative expansion may be insufficient for models with extremely distributed knowledge.
- Evaluation is limited to identity, style, and object unlearning; effectiveness on concept-level or finer-grained unlearning remains unknown.
- Robustness to adversarial recovery attacks beyond quantization has not been tested.

## Related Work & Insights

- **vs. SLUG**: SLUG is restricted to single-layer editing and fails when knowledge is distributed; RAZOR's multi-layer editing and iterative expansion offer far greater flexibility and coverage.
- **vs. SalUn**: SalUn selects parameters using only forgetting gradients and compensates for retention conflicts post hoc; RAZOR jointly considers both forgetting and retention at the selection stage.
- **vs. ESD**: ESD suppresses concepts via negative guidance at sampling time without true deletion; RAZOR removes knowledge at the weight level, achieving more thorough unlearning.

## Rating

- Novelty: ⭐⭐⭐⭐ The ratio-aware scoring and model-agnostic framework design are inventive, though the overall approach remains within the gradient-selection + constrained-optimization paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three major model families (CLIP/SD/VLM), quantization robustness, and efficiency comparisons comprehensively.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and notation is complete, though the density of symbols requires frequent cross-referencing.
- Value: ⭐⭐⭐⭐ Provides a practical unlearning framework; quantization robustness is a significant additional contribution.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)
- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[CVPR 2026\] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [\[CVPR 2026\] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](segquant_diffusion_model_quantization.md)

<!-- RELATED:END -->
