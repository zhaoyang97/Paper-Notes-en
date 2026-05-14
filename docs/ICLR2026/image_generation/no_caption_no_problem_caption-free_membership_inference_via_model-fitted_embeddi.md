---
title: >-
  [Paper Note] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings
description: >-
  [ICLR 2026][Image Generation][Membership Inference Attack] This paper proposes MoFit, the first membership inference attack (MIA) framework for diffusion models under a caption-free setting. By constructing surrogate ima…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Membership Inference Attack"
  - "Diffusion Models"
  - "Caption-Free Setting"
  - "Model-Fitted Embeddings"
  - "Privacy Auditing"
date: 2026-05-08
content_hash: cf7b194d452f578f
---

# No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings

**Conference**: ICLR 2026
**arXiv**: [2602.22689](https://arxiv.org/abs/2602.22689)
**Code**: [GitHub](https://github.com/JoonsungJeon/MoFit)
**Area**: AI Security / Privacy Attacks
**Keywords**: Membership Inference Attack, Diffusion Models, Caption-Free Setting, Model-Fitted Embeddings, Privacy Auditing

## TL;DR

This paper proposes MoFit, the first membership inference attack (MIA) framework for diffusion models under a caption-free setting. By constructing surrogate images and conditional embeddings that overfit to the target model, MoFit exploits the asymmetric sensitivity of member samples to conditioning mismatch to enable effective inference.

## Background & Motivation

- The memorization tendency of diffusion models in high-fidelity generation raises concerns about privacy and intellectual property.
- Membership inference attacks (MIA) are the standard method for auditing memorization.
- **Critical assumption flaw in existing MIAs**: prior work assumes the attacker has access to ground-truth captions, which is unrealistic because:
    - Artists suspecting their work was copied typically cannot obtain the training captions.
    - Public generative AI platforms do not disclose training data sources.
- When ground-truth captions are replaced with VLM-generated alternatives, the performance of SOTA methods degrades significantly.

## Method

### Core Observation

**Member and non-member samples exhibit systematic differences in sensitivity to conditioning mismatch**:
- Member samples show a significant increase in $\mathcal{L}_{\text{cond}}$ under surrogate conditions.
- Non-member samples show little change.
- $\mathcal{L}_{\text{uncond}}$ remains stable for both groups.

### MoFit Two-Stage Framework

**Stage 1: Model-Fitted Surrogate Optimization**

A surrogate image $x_0^* = x_0 + \delta^*$ is constructed to overfit the unconditional prior of the target model:

$$\delta^* = \arg\min_\delta \mathbb{E}_{z_0', t, \hat{\epsilon}} [\|\hat{\epsilon} - \epsilon_\theta(z_t', t, \phi_{\text{null}})\|^2]$$

$\hat{\epsilon}$ and $t$ are fixed to stabilize the perturbation direction, and $\delta$ is iteratively updated along the gradient sign direction.

**Stage 2: Surrogate-Driven Embedding Extraction**

An embedding $\phi^*$ is optimized from the surrogate image $x_0^*$:

$$\phi^* = \arg\min_\phi \mathbb{E}_{z_0^*, t, \hat{\epsilon}} [\|\hat{\epsilon} - \epsilon_\theta(z_t^*, t, \phi)\|^2]$$

The VLM-generated caption embedding is used as initialization.

**Stage 3: Membership Inference**

The model-fitted embedding $\phi^*$ is used to condition the original query $x_0$:

$$\mathcal{L}_{\text{MoFit}} = \mathbb{E}[\|\hat{\epsilon} - \epsilon_\theta(z_t, t, \phi^*)\|^2] - \mathbb{E}[\|\hat{\epsilon} - \epsilon_\theta(z_t, t, \phi_{\text{null}})\|^2]$$

The final decision fuses the MoFit score with auxiliary losses ($\mathcal{L}_{\text{uncond}}$ or $\mathcal{L}_{\text{VLM}}$).

## Key Experimental Results

### MIA Performance Comparison in the Caption-Free Setting

| Method | Condition | Pokemon ASR | Pokemon TPR@1%FPR | MS-COCO ASR | MS-COCO TPR@1%FPR |
|------|------|------------|-------------------|-------------|-------------------|
| CLiD | GT | 96.52 | 90.14 | 86.50 | 68.80 |
| CLiD | VLM | 77.55 | 19.23 | 80.90 | 50.80 |
| PFAMI | VLM | 74.43 | 6.01 | 80.40 | 29.40 |
| SecMI | VLM | 78.51 | 6.97 | 57.30 | 4.20 |
| **MoFit** | **$\phi^*$** | **94.48** | **50.48** | **88.00** | **47.00** |

### Ablation Study: Surrogate Image Variants

| Input | Condition | Pokemon ASR | MS-COCO ASR | MS-COCO TPR@1%FPR |
|------|------|------------|-------------|-------------------|
| $x_0$ (original) | $\phi$ | 75.63 | 78.00 | 31.00 |
| $x_0 + \delta$ (random noise) | $\phi$ | 93.99 | 81.70 | 29.20 |
| $x_0 + \delta_{\text{MAX}}$ (reverse optimization) | $\phi$ | 75.87 | 78.00 | 34.00 |
| **MoFit** ($x_0 + \delta^*$) | **$\phi^*$** | **94.48** | **88.00** | **47.00** |

### Key Findings

1. MoFit substantially outperforms VLM-conditioned baselines in the caption-free setting (up to +25% ASR and +30–47% TPR@1%FPR).
2. On MS-COCO, MoFit even surpasses CLiD with ground-truth captions (ASR: 88.00 vs. 86.50).
3. Surrogate optimization is critical: using the original image alone or random noise for embedding optimization yields significantly worse results.
4. MoFit remains effective on SD v1.5 pretrained models (ASR: 77.61), demonstrating generalizability.

## Highlights & Insights

1. **Practical relevance of problem formulation**: The caption-free MIA setting more closely reflects real-world auditing needs.
2. **Deep theoretical insight**: The asymmetric sensitivity of member samples to conditioning mismatch provides a novel and exploitable signal.
3. **Elegant two-stage design**: Constructing an overfitted surrogate before extracting embeddings yields a tightly coupled model-fitted pair.
4. **No additional data or models required**: Only inference-level access to the target model's denoising network is needed.

## Limitations & Future Work

- Requires access to the target model's denoising network parameters (gray-box assumption).
- Surrogate optimization and embedding extraction introduce additional computational overhead.
- The fixed timestep $t=140$ is a hyperparameter that may require tuning for different models.
- Effectiveness is relatively reduced on LAION-scale pretrained models, a setting where all methods perform poorly.

## Related Work & Insights

- **Diffusion model MIA**: SecMI, PIA, PFAMI, CLiD
- **LLM MIA**: Shokri et al. (2017)
- **Caption-free generation**: Classifier-free guidance (Ho & Salimans, 2022)

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First MIA framework for diffusion models targeting the caption-free setting.
- Technical Depth: ⭐⭐⭐⭐ — Core observation is insightful; two-stage optimization design is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple datasets, multiple models, and thorough ablations.
- Value: ⭐⭐⭐⭐ — Provides a practical tool for data privacy auditing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)
- [\[ICLR 2026\] HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation](hierloc_hyperbolic_entity_embeddings_for_hierarchical_visual_geolocation.md)
- [\[ICLR 2026\] RNE: plug-and-play diffusion inference-time control and energy-based training](rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)
- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)

</div>

<!-- RELATED:END -->
