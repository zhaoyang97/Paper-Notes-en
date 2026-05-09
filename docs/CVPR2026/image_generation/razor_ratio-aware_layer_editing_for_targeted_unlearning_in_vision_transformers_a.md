---
title: >-
  [Paper Note] RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models
description: >-
  [CVPR2026][Image Generation][machine unlearning] This paper proposes RAZOR, a ratio-aware multi-layer/multi-head selective editing framework that enables efficient and precise targeted unlearning in Transformer-based vision models such as CLIP, Stable Diffusion, and VLMs, while preserving overall model performance and quantization robustness.
tags:
  - CVPR2026
  - Image Generation
  - machine unlearning
  - vision transformer
  - diffusion model
  - CLIP
  - model editing
  - ratio-aware saliency
date: 2026-05-08
content_hash: c6da2a3987448e5b
---

# RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models

**Conference**: CVPR2026
**arXiv**: [2603.14819](https://arxiv.org/abs/2603.14819)
**Code**: [raviranjan-ai/RAZOR-cvpr2026](https://github.com/raviranjan-ai/RAZOR-cvpr2026)
**Area**: Image Generation
**Keywords**: machine unlearning, vision transformer, diffusion model, CLIP, model editing, ratio-aware saliency

## TL;DR

This paper proposes RAZOR, a ratio-aware multi-layer/multi-head selective editing framework that enables efficient and precise targeted unlearning in Transformer-based vision models such as CLIP, Stable Diffusion, and VLMs, while preserving overall model performance and quantization robustness.

## Background & Motivation

Large-scale vision-language models (CLIP), text-to-image diffusion models (Stable Diffusion), and vision-language assistants (LLaVA) are trained on massive datasets and inevitably encode sensitive or undesirable information (e.g., personal identities, copyrighted content). Compliance requirements such as GDPR necessitate the removal of such knowledge from deployed models. Since retraining from scratch is prohibitively expensive, **machine unlearning** has emerged as an important research direction.

An ideal unlearning method must simultaneously satisfy three criteria:

**Efficient**: low computational overhead for the unlearning process;

**Precise**: removes only the target information without affecting unrelated capabilities;

**Robust**: target knowledge remains irrecoverable even under adversarial or quantized conditions.

Existing methods each have notable limitations:
- **SSD** uses Fisher information to select parameters, incurring high computational cost and noticeable utility degradation;
- **SalUn** selects weights via gradient saliency but considers only the forget gradient, resulting in incomplete unlearning;
- **SLUG** pioneers single-layer editing for high efficiency, but becomes brittle when knowledge is distributed across multiple layers;
- **ESD/FMN/UCE** and related diffusion model approaches still face trade-offs between unlearning completeness and content fidelity.

The root cause is that existing methods drive parameter selection **solely by forget-set saliency**, with retain conflicts mitigated only post hoc, making it impossible to avoid the dynamic coupling between forgetting and retention. RAZOR addresses this by employing **ratio-aware scoring** to jointly account for forgetting pressure and retention alignment at the parameter selection stage.

## Method

### Overall Architecture

RAZOR (Ratio-Aware Zero/One-step Optimized Retentive unlearning) is a lightweight, model-agnostic unlearning framework operating in three steps:

1. **Ratio-aware layer/head selection**: Compute forget and retain gradients for each layer and use ratio-aware saliency scores to identify the set of layers $\mathcal{K}$ to be edited;
2. **Constrained multi-objective loss optimization**: Apply fused gradient updates on selected layers, jointly optimizing forgetting, retention, and stability objectives;
3. **Iterative expansion**: If unlearning is insufficient, incrementally incorporate additional layers until the threshold is met or the iteration limit is reached.

### Key Design 1: Ratio-Aware Saliency Scoring

For each layer $l$, one-step forget and retain gradients are computed:

$$g^f_l = \nabla_{\theta_l} \mathcal{L}_{\text{forget}}, \quad g^r_l = \nabla_{\theta_l} \mathcal{L}_{\text{retain}}$$

The ratio-aware saliency score is then defined as:

$$\phi(l) = \frac{\|g^f_l\|_2}{\|\theta_l\|_2 + \varepsilon} \cdot (1 - \cos(g^f_l, g^r_l))^\alpha$$

- The first term measures the layer's contribution to forgetting (normalized forget gradient norm);
- The second term measures the orthogonality between the forget and retain gradients — **greater orthogonality implies less collateral damage to retained knowledge when editing that layer**;
- $\alpha \in [0,1]$ balances magnitude and orthogonality; $\varepsilon$ ensures numerical stability.

Layers are selected as $\mathcal{K} = \{l \mid \phi(l) > \tau\}$, where $\tau$ is a threshold.

Compared to SLUG's single-layer selection and SalUn's forget-only gradient criterion, this design **jointly considers unlearning capability and retention safety at the selection stage**, constituting the paper's core contribution.

### Key Design 2: Three-Component Loss Function

$$\mathcal{L}_{\text{RAZOR}} = \mathcal{L}_{\text{retain}} + \lambda_f \rho \, \mathcal{L}_{\text{forget}} + \lambda_m \, \mathcal{L}_{\text{mismatch}}$$

| Loss Term | CLIP | Stable Diffusion | VLM (LLaVA) |
|---|---|---|---|
| $\mathcal{L}_{\text{retain}}$ (utility preservation) | Symmetric InfoNCE contrastive loss | $\varepsilon$-prediction denoising loss | Symmetric InfoNCE (visual encoder) |
| $\mathcal{L}_{\text{forget}}$ (forgetting push) | Cosine embedding repulsion loss | CE loss on text encoder | CE loss on visual encoder |
| $\mathcal{L}_{\text{mismatch}}$ (stability regularization) | Similarity Drift Regularization (SDR) | SDR on generated outputs | Logit drift regularization on neutral QA |

Here $\rho \in (0,1]$ is a ratio hyperparameter controlling the overall forgetting pressure. The three terms serve distinct roles: retain preserves utility, forget drives unlearning (gradient ascent), and mismatch regularization prevents global drift in the embedding space.

### Loss & Training: Single/Few-Step Updates with Iterative Expansion

For each selected layer $l \in \mathcal{K}$, a fused gradient update is applied:

$$\Delta\theta_l = -\eta_l(-\lambda_f \rho \, g^f_l + g^r_l + \lambda_m \nabla_{\theta_l}\mathcal{L}_{\text{mismatch}})$$

The step size $\eta_l$ is determined via lightweight binary search, selecting the largest stable step that yields the best forget-retain trade-off on a small validation set.

If the initial edit yields insufficient unlearning, **iterative expansion** is triggered: at each round, $\phi_t(l)$ is recomputed on the updated model, the highest-scoring new layer is added to $\mathcal{K}$ and updated, for a maximum of 6 rounds. This progressive strategy ensures precise unlearning without over-editing.

## Key Experimental Results

### Main Results 1: CLIP Identity Unlearning (Selected from Table 3)

| Method | CIFAR-10 M1↓ | CIFAR-10 M4↑ | CIFAR-10 M5↑ | ImageNet M1↓ | ImageNet M4↑ | LAION M1↓ | LAION M4↑ |
|---|---|---|---|---|---|---|---|
| SSD | 52.00 | 25.00 | 97.50 | 52.50 | 30.00 | 42.00 | 48.00 |
| SalUn | 97.00 | 83.00 | 84.50 | 88.00 | 84.00 | 48.00 | 88.00 |
| SLUG | 67.50 | 87.50 | 96.50 | 68.00 | 88.00 | 48.00 | 88.00 |
| **RAZOR** | **52.50** | **89.00** | **100.0** | **53.50** | **92.00** | **40.00** | **94.00** |

RAZOR achieves the highest retention accuracy (M4) and perfect retrieval stability (M5=100) while maintaining strong unlearning precision (low M1), with zero M3 privacy leakage. After 4-bit quantization, RAZOR exhibits minimal performance degradation (M5 drops only 1.4%), far outperforming SalUn and LoTUS.

### Main Results 2: Stable Diffusion Style/Object Unlearning (Selected from Table 4)

| Method | SD-V3 Style UA↑ | SD-V3 Style IRA↑ | SD-V3 Object UA↑ | SD-V1.5 Style UA↑ | SD-V1.5 Object UA↑ |
|---|---|---|---|---|---|
| ESD | 99.62 | 89.97 | 97.44 | 98.58 | 92.15 |
| SalUn | 90.36 | 92.33 | 91.06 | 86.26 | 86.91 |
| SLUG | 88.20 | 85.59 | 85.44 | 86.29 | 75.43 |
| **RAZOR** | **99.40** | **98.97** | **98.80** | **99.26** | **97.91** |

RAZOR achieves comprehensive improvements across UA/IRA/CRA metrics on both SD-V3 and SD-V1.5, with CRA approaching 100, demonstrating excellent semantic coherence after unlearning.

### Efficiency Comparison (Table 5, SD-V1.5)

| Method | Time (s)↓ | VRAM (GB)↓ | Storage (GB)↓ | Trade-off↑ |
|---|---|---|---|---|
| ESD | 6163 | 17.8 | 4.30 | 11.97 |
| SLUG | 39 | 3.6 | 0.04 | 59.42 |
| **RAZOR** | 78 | 4.2 | 0.06 | **66.86** |

Although RAZOR is approximately twice as slow as SLUG, it achieves the highest performance-efficiency trade-off score (66.86 vs. 59.42) when accounting for unlearning accuracy, while modifying only a small subset of layer weights (storage: 0.06 GB).

### Ablation Study (Table 7, CLIP on CIFAR-10)

| Configuration | M1↓ | M3→0 | M4↑ | M5↑ |
|---|---|---|---|---|
| w/o Retain loss | 52.72 | 0.40 | 82.00 | 99.0 |
| w/o Mismatch loss | 53.25 | 1.04 | 88.00 | 100.0 |
| w/o Forget loss | 96.00 | 0.40 | 86.00 | 99.0 |
| Full-layer update (no selection) | 51.00 | 1.58 | 78.00 | 96.0 |
| Selection w/o iteration | 53.00 | 0.00 | 88.82 | 100.0 |
| **Full RAZOR** | **52.50** | **0.00** | **89.00** | **100.0** |

Key findings:
- Removing the forget loss largely causes unlearning to fail (M1=96%);
- Full-layer updating achieves the lowest M1 but the highest M3 leakage and worst M4 utility — validating the necessity of selective editing;
- Iterative expansion yields a marginal improvement in M4 from 88.82 to 89.00, and more importantly ensures the unlearning threshold is reliably reached.

### VLM Unlearning (Table 6, LLaVA-1.6-8B)

After unlearning 10 celebrity identities, the average Forgetting Accuracy (FA) drops to only 2.2% (baseline: 97.25%), while GQA is maintained at 60.46 (baseline: 60.18) and MMBench at 60.9% (baseline: 61.57%), demonstrating negligible degradation to general multimodal capabilities.

## Highlights & Insights

1. **Ratio-aware selection** is an elegant and effective design: a single gradient computation simultaneously evaluates the forget contribution and retention safety of each layer, offering greater stability than sequential strategies that first forget and then compensate for retention.
2. **Model-agnosticism**: the same framework applies uniformly to CLIP, Stable Diffusion, and LLaVA, requiring only loss function instantiation to be swapped.
3. **Quantization robustness** is an underexplored but practically critical dimension — RAZOR maintains near-original unlearning effectiveness after 4-bit quantization, whereas full-model update methods such as SalUn and LoTUS degrade significantly under quantization.
4. The efficiency-accuracy trade-off score (66.86) surpasses all baselines, demonstrating that the additional computational cost of multi-layer editing is fully compensated by superior unlearning performance.

## Limitations & Future Work

1. The iterative expansion cap is fixed at 6 rounds; it remains unclear whether more complex tasks require more iterations, and no adaptive stopping criterion is provided;
2. Sensitivity analyses for the threshold $\tau$ and hyperparameters $\alpha, \rho, \lambda_f, \lambda_m$ are relegated to the appendix, with no general tuning guidelines in the main text;
3. Evaluation is limited to identity, style, and object unlearning scenarios; more complex compositional concepts (e.g., "a specific person wearing a specific outfit") are not tested;
4. Robustness against adversarial unlearning recovery attacks is not discussed;
5. Extension to modalities such as audio and video remains unexplored.

## Related Work & Insights

- **SLUG** (ICML 2025): The direct baseline for RAZOR; single-layer editing is efficient but brittle. RAZOR addresses the distributed storage of knowledge through multi-layer selective editing.
- **SalUn** (NeurIPS 2023): Uses only the forget gradient for saliency ranking; RAZOR augments this with the orthogonality of the retain gradient to form the ratio-aware criterion.
- **ESD**: Suppresses concepts via negative guided sampling; achieves slightly higher UA on SD-V3 but substantially lower IRA/CRA compared to RAZOR.
- **Inspiration**: The "forget-retain orthogonality" principle underlying ratio-aware scoring is generalizable to LLM knowledge editing and the forgetting-retention trade-off in continual learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The ratio-aware saliency scoring and iterative expansion strategy are innovative, though the overall approach remains an engineering refinement within the gradient-editing paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers three model families (CLIP/SD/VLM) × multiple datasets × quantization robustness × ablation studies × efficiency comparisons; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Framework presentation is clear, loss comparison tables are intuitive, and mathematical notation is rigorous.
- **Value**: ⭐⭐⭐⭐ — Provides a practical and scalable solution for deployment-grade unlearning in vision models.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing](care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)
- [\[CVPR 2026\] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](segquant_a_semantics-aware_and_generalizable_quantization_framework_for_diffusio.md)
- [\[CVPR 2026\] SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](seacache_spectral-evolution-aware_cache_for_accelerating_diffusion_models.md)
- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)
- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)

<!-- RELATED:END -->
