---
title: >-
  [Paper Note] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Prompt Learning] This paper proposes VaMP, a variational multi-modal prompt learning framework that models text-side prompts as latent variables and performs instance-level uncertainty mode…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Prompt Learning"
  - "Variational Inference"
  - "CLIP"
  - "Few-Shot Learning"
  - "Domain Generalization"
date: 2026-05-08
content_hash: 76c0fcfca82ba82c
---

# VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.22664](https://arxiv.org/abs/2511.22664)  
**Code**: [GitHub](https://visual-ai.github.io/vamp)  
**Area**: Multimodal VLM
**Keywords**: Prompt Learning, Variational Inference, CLIP, Few-Shot Learning, Domain Generalization

## TL;DR

This paper proposes VaMP, a variational multi-modal prompt learning framework that models text-side prompts as latent variables and performs instance-level uncertainty modeling via variational inference. Combined with a class-aware prior for regularizing the latent space, VaMP significantly improves CLIP's downstream adaptation under few-shot and domain generalization settings.

## Background & Motivation

Vision-language models (e.g., CLIP) perform well in zero-shot settings but remain challenging to adapt to downstream tasks in few-shot scenarios. Prompt learning has been widely adopted as a parameter-efficient adaptation strategy; however, existing methods (e.g., CoOp, MaPLe, MMRL) typically employ **fixed, deterministic prompts shared across all samples**, giving rise to three core limitations:

**Lack of instance-level adaptability**: Fixed prompts cannot be personalized for different inputs, limiting generalization under distribution shift.

**Lack of uncertainty modeling**: Deterministic parameters fail to capture model uncertainty in few-shot settings.

**Overly simplistic priors**: Existing variational approaches (e.g., Bayesian Prompt Learning, Any-Shift Prompting) only model the text branch, use standard Gaussian priors, and operate at a global level, precluding fine-grained token-level semantic variation.

The paper frames prompt tuning as a **variational inference** problem, introducing token-level variational modeling within a multi-modal (text + vision) framework and employing a class-aware prior to provide semantically structured regularization.

## Method

### Overall Architecture

VaMP consists of three core components: (1) image-conditioned sample-specific prompt generation; (2) a variational inference mechanism that treats text-side prompts as latent variables; and (3) a class-aware prior constructed from class prototypes. Prompts are injected into multiple Transformer layers of both the CLIP text and vision encoders.

### Key Designs

1. **Sample-Specific Prompt Generation**

   Unlike MaPLe/MMRL, which use fixed shared prompts, VaMP dynamically generates text-side prompts conditioned on the input image. Given input image $x$, a global representation $f_x$ is first extracted via the frozen CLIP image encoder. A set of $H$ layer-specific MLP generators $\{\Phi_i\}$ then produces prompt tokens for each Transformer layer:

   $z_i = \Phi_i(f_x) \in \mathbb{R}^{M \times d}, \quad i = J, \dots, J+H-1$

   On the vision side, shared learnable prompt tokens $\tilde{z}_i$ are used and remain invariant across samples. This asymmetric design enables instance-level adaptation on the text side while maintaining stable shared representations on the vision side.

2. **Variational Multi-Modal Prompt Adaptation**

   The central innovation is upgrading text-side prompts $z_i$ from deterministic parameters to **latent variables**. For each layer, an MLP $\phi_i$ predicts the posterior distribution parameters:

   $[\mu_i, \sigma_i] = \phi_i(\bar{f}_x), \quad q_\phi(z_i|x) = \mathcal{N}(\mu_i, \text{diag}(\sigma_i^2))$

   The training objective maximizes the variational evidence lower bound (ELBO):

   $\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p(y|x,t,z)] - \text{KL}(q_\phi(z|x) \| p(z))$

   End-to-end training is enabled via the reparameterization trick $z_i = \mu_i + \sigma_i \odot \epsilon_i$. At inference time, Monte Carlo sampling with $S=10$ draws followed by prediction averaging yields an uncertainty-aware ensemble.

3. **Class-Aware Prior**

   The standard Gaussian prior $\mathcal{N}(0, I)$ lacks semantic structure. VaMP constructs a class-aware prior to provide global semantic anchoring. During training, per-class prototypes $o_y$ are computed as the mean of posterior means for samples in each class, then mapped to prior distribution parameters via layer-specific prior networks $\psi_i$:

   $[\hat{\mu}_i, \hat{\sigma}_i] = \psi_i(c_y), \quad p_\psi(z_i|o_y) = \mathcal{N}(\hat{\mu}_i, \text{diag}(\hat{\sigma}_i^2))$

   The ELBO is summed across layers, encouraging prompt distributions of same-class samples to cluster in the latent space and improving intra-class consistency. At test time, where labels are unavailable, the prior degenerates to a standard Gaussian.

### Loss & Training

The variational ELBO serves as the training objective, comprising a reconstruction term (classification cross-entropy) and a KL divergence regularization term. All experiments use the 16-shot setting with ViT-B/16 CLIP, trained on a single V100 GPU. Class prototypes are computed offline with the frozen encoder prior to training.

## Key Experimental Results

### Main Results: Base-to-Novel Generalization (Average over 11 Datasets)

| Method | Base | Novel | H (Harmonic Mean) |
|--------|------|-------|-------------------|
| CLIP | 69.34 | 74.22 | 71.70 |
| MaPLe | 82.28 | 75.14 | 78.55 |
| MMRL | 85.68 | 77.16 | 81.20 |
| **VaMP** | **86.45** | **78.67** | **82.37** |

VaMP surpasses MMRL on Novel classes by **1.51%** and improves the harmonic mean by **1.17%**.

### Domain Generalization (ImageNet → 4 Variants)

| Method | ImageNet | -V2 | -Sketch | -A | -R |
|--------|----------|-----|---------|-----|-----|
| MMRL | 72.03 | 64.47 | 49.17 | 51.20 | 77.53 |
| **VaMP** | **72.83** | **64.96** | **49.69** | **51.97** | **78.01** |

### Ablation Study

| Configuration | Base | Novel | H | Note |
|---------------|------|-------|---|------|
| Task-specific prompts (MMRL baseline) | 85.68 | 77.16 | 81.20 | Fixed shared prompts |
| + Sample-specific prompts | 85.93 | 78.13 | 81.84 | Novel +0.97% |
| + Variational modeling | 86.11 | 78.45 | 82.10 | Novel +0.32% |
| + Class-aware prior | 86.45 | 78.67 | 82.37 | Novel +0.22% |

### Key Findings

- All three components (sample-specificity, variational modeling, class-aware prior) contribute independently and complementarily.
- On fine-grained texture datasets such as DTD, VaMP leads MMRL by 2.67% on Novel accuracy.
- In cross-dataset generalization (average over 10 target datasets), VaMP achieves 67.74%, surpassing MMRL by 0.49%.
- Latent space visualizations reveal that deeper layers yield more compact posterior distributions, reflecting hierarchical uncertainty refinement.

## Highlights & Insights

- **The integration of prompt learning and variational inference is taken to its fullest extent**: rather than simple global variational modeling, fine-grained token-level, multi-layer variational modeling is employed.
- The class-aware prior substantially outperforms the standard Gaussian prior, providing structured semantic guidance in the latent space.
- Monte Carlo sampling at inference time yields uncertainty-aware ensembling, improving robustness.
- The framework is general and can be adapted to both MaPLe and MMRL as multi-modal prompt baselines.

## Limitations & Future Work

- Inference requires 10 Monte Carlo samples, introducing computational overhead; mean approximation can be used as a faster alternative.
- The class-aware prior degenerates to a standard Gaussian at test time, leaving test-time class information underutilized.
- Experiments are conducted only on ViT-B/16; performance on larger backbones (e.g., ViT-L/14) remains unknown.
- Variational modeling is restricted to the text side; the vision side still employs deterministic prompts.

## Related Work & Insights

- Bayesian Prompt Learning and Any-Shift Prompting are pioneering works in this direction, but are limited to unimodal text modeling and global-level variational inference.
- MMRL's multi-modal shared representation paradigm provides the foundation for VaMP's multi-layer prompt injection.
- Classical applications of variational inference in few-shot learning (e.g., VAE-based meta-learning) provide the theoretical grounding for this work.
- Insight: the variational framework can be extended to other PETL methods such as visual prompts and adapters.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Token-level multi-layer variational prompt modeling combined with a class-aware prior constitutes a meaningful and systematic contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three generalization settings, 11 datasets, complete ablations, and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rigorous mathematical derivations.
- **Value**: ⭐⭐⭐⭐ Introduces a structured probabilistic modeling paradigm for prompt learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](../../CVPR2026/multimodal_vlm/noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)
- [\[AAAI 2026\] RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)](../../AAAI2026/multimodal_vlm/rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)
- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](../../ICCV2025/multimodal_vlm/fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)
- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](../../ICCV2025/multimodal_vlm/advancing_textual_prompt_learning_with_anchored_attributes.md)

</div>

<!-- RELATED:END -->
