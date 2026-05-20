---
title: >-
  [Paper Note] Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection
description: >-
  [ICLR 2026][Image Generation][concept erasure] HiRM introduces a concept erasure strategy that decouples the update location from the erasure target — updating only the first-layer weights of the CLIP text encoder while…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "concept erasure"
  - "text encoder"
  - "causal localization"
  - "representation misdirection"
  - "modular safety patch"
date: 2026-05-08
content_hash: 8681c8406d78d404
---

# Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection

**Conference**: ICLR 2026
**arXiv**: [2602.19631](https://arxiv.org/abs/2602.19631)  
**Code**: [GitHub](https://github.com/Coffeeloveman/HiRM)  
**Area**: Diffusion Models / Safety / Machine Unlearning
**Keywords**: concept erasure, text encoder, causal localization, representation misdirection, modular safety patch

## TL;DR

HiRM introduces a concept erasure strategy that decouples the update location from the erasure target — updating only the first-layer weights of the CLIP text encoder while imposing erasure supervision on the high-level semantic representations at the final layer. By steering target concept representations toward random directions (HiRM-R) or semantically meaningful directions (HiRM-S), the method achieves effective erasure of styles, objects, and NSFW content on the UnlearnCanvas and NSFW benchmarks, with zero-shot transferability to the Flux architecture.

## Background & Motivation

**Background**: Concept erasure methods are broadly divided into training-based approaches (fine-tuning the U-Net, e.g., ESD, SalUn, MACE) and training-free approaches (closed-form editing or prompt manipulation, e.g., UCE, RECE, SAFREE). Both categories primarily modify the U-Net/denoiser.

**Limitations of Prior Work**: Modifying the U-Net is computationally expensive and tends to degrade the generation quality of unrelated concepts. Training-free methods struggle to balance erasure effectiveness with concept preservation.

**Key Challenge**: Causal tracing by Basu et al. identifies the first layer of the CLIP text encoder as the causal state for visual attributes, suggesting that direct intervention at this point should be feasible. However, directly editing early layers (e.g., Diff-QuickFix) performs poorly on abstract concepts (e.g., NSFW/nudity) and degrades overall model quality, because early-layer representations form a "bag of concepts" — modifications at this level inadvertently affect all shared low-level features.

**Goal**: To achieve precise concept erasure within the text encoder, simultaneously satisfying: (a) effective erasure of both concrete concepts (styles/objects) and abstract concepts (nudity); (b) preservation of generation quality for non-target concepts; (c) computational efficiency and cross-architecture transferability.

**Key Insight**: Toker et al. show that coherent high-level semantic representations emerge only in the final few layers, while early layers encode dispersed low-level features. This motivates decoupling the update location from the supervision location — applying gradient updates at the first layer (where the causal state resides) while defining the erasure loss at the final layer (where high-level semantics are formed).

**Core Idea**: By updating only the first-layer weights, HiRM remotely misdirects the high-level semantic representations of target concepts at the final layer, achieving precisely localized concept erasure.

## Method

### Overall Architecture

Given a text encoder $f_{\text{text}}$ consisting of an $L$-layer Transformer, only the first-layer parameters $\theta_1$ are updated while $\theta_{2:L}$ are frozen. For prompts containing the target concept, the final-layer representation $h^{(L)}$ is computed and a loss is applied to steer it toward a designated direction.

### Key Designs

1. **HiRM-R (Random Direction Misdirection)**:

    - Function: Drives the final-layer token representation $h_t^{(L)}$ of the target concept toward a random unit vector $\hat{r}_t$.
    - Loss: $\mathcal{L}_{\text{HiRM-R}} = \frac{1}{T} \sum_{t=1}^T \|h_t^{(L)} - c \cdot \hat{r}_t\|^2$
    - Design Motivation: A random direction is sufficient to disrupt target semantics and is generalizable across targets without requiring a predefined semantic anchor.
    - Steering coefficient $c=500$ (HiRM-R), controlling misdirection intensity.

2. **HiRM-S (Semantic Direction Misdirection)**:

    - Function: Steers the target representation toward a semantically related superordinate concept (e.g., "Van Gogh" → "Painting").
    - Loss: $\mathcal{L}_{\text{HiRM-S}} = \frac{1}{T} \sum_{t=1}^T \|h_t^{(L)} - c \cdot s_t^{(L)}\|^2$
    - For NSFW concepts: a "safe misdirection vector" is constructed by subtracting the nudity semantic vector from the representation of prompts containing nudity (inspired by the Ring-A-Bell framework), and the result is used as the target direction.
    - $c=1$ for HiRM-S.

3. **Decoupling Update Location from Supervision Target**:

    - Why update only the first layer: Causal tracing confirms that visual attributes are primarily determined by the first layer.
    - Why apply supervision at the final layer: High-level semantic representations at the final layer correspond more precisely to target concepts, avoiding representation shattering.
    - Ablation validation: Models supervised at early layers achieve good erasure but poor preservation; supervision at the final-layer $W_{\text{out}}$ yields the best erasure–preservation trade-off.

### Loss & Training

- Style erasure: lr=5e-5, 40 epochs (HiRM-R) / 30 epochs (HiRM-S), single-word prompts.
- Object erasure: lr=5e-5, 25 epochs (HiRM-R) / 15 epochs (HiRM-S).
- Nudity erasure: lr=1e-4, 50 epochs (HiRM-R) / 25 epochs (HiRM-S), joint multi-keyword training.
- Training time ~1.2s, GPU memory 1.60 GB, no retain set required.

## Key Experimental Results

### UnlearnCanvas Benchmark (Style + Object)

| Method | Training-based | Style UA↑/IRA↑/AA↑ | Object UA↑/IRA↑/AA↑ | Train Time (s) |
|--------|---------------|---------------------|----------------------|----------------|
| ESD | ✓ | 98.58/80.97/91.17 | 92.15/55.78/64.05 | 7372 |
| MACE | ✓ | 54.69/89.85/81.10 | 67.65/98.52/87.85 | 175 |
| SalUn | ✓ | 86.26/90.39/90.58 | 86.91/96.35/94.28 | 610 |
| Diff-Q | ✗ | 96.40/93.91/95.81 | 94.00/98.37/96.19 | - |
| **HiRM-R** | ✓ | 95.50/89.31/94.24 | 93.20/98.18/94.65 | **1.20** |
| **HiRM-S** | ✓ | **96.20/92.67/95.54** | **96.20/97.77/96.94** | **1.20** |

### NSFW Erasure (Adversarial Robustness)

| Method | Ring-16↓ | Ring-77↓ | MMA↓ | I2P↓ | COCO CLIP↑ |
|--------|----------|----------|------|------|------------|
| SalUn | 0.00 | 2.11 | 0.90 | 0.57 | 0.293 |
| RECE | 1.05 | 1.05 | 0.40 | 0.57 | 0.277 |
| Ediff | 2.11 | 1.05 | 4.10 | 0.85 | 0.307 |
| **HiRM-R** | **0.00** | **0.00** | 8.00 | 0.96 | 0.304 |
| **HiRM-S** | 1.05 | **0.00** | **3.30** | **0.66** | **0.306** |

### Key Findings

- HiRM-S achieves the best AA on both style and object erasure simultaneously, with a training time of only 1.2s (vs. 7372s for ESD) — 145× faster than MACE, the fastest prior training-based method.
- Synergy with denoiser-based methods: HiRM-R combined with EraseAnything on Flux reduces Ring-16 from 29.47% to 3.16% with negligible change in CLIP score.
- Zero-shot transfer to Flux: replacing only the text encoder without additional training reduces Ring-16 from 88.42% to 37.89%.
- Multi-concept erasure (S-HiRM-S = SPEED + HiRM-S): maintains MMA at 1.70% and Ring-16 at 1.05% for 50-celebrity erasure combined with nudity erasure.
- t-SNE visualization confirms that only target concept representations are displaced while non-target concept representations remain stable.

## Highlights & Insights

- **The decoupled update–supervision design is remarkably elegant**: the first layer is updated (as the causal state), while the final layer provides supervision (as the site of high-level semantics); the frozen intermediate layers naturally bridge the two.
- **Modular safety patch**: since only the text encoder is modified, the patch is plug-and-play and can be applied to any model sharing the same CLIP encoder — including LoRA fine-tuned variants and Flux — without retraining.
- **Orthogonal and complementary to U-Net methods**: HiRM modifies the text encoder while denoiser-based methods modify the U-Net, forming a dual-layer defense with significant synergistic benefits.

## Limitations & Future Work

- HiRM applies misdirection uniformly across all tokens without distinguishing token importance, potentially suppressing representations unrelated to the target concept.
- Robustness against white-box adversarial attacks (UnLearnDiffAtk) is relatively weak (22.54% ASR), underperforming SalUn and RECE.
- Multi-concept erasure currently relies on simple weight averaging to merge LoRA modules, resulting in a drop in IRA (65.56%); more refined fusion strategies are needed.
- The model-agnostic design, while clean, limits the potential to exploit model-internal structural information.

## Related Work & Insights

- **vs. Diff-QuickFix**: Both edit the text encoder, but Diff-Q uses a closed-form solution to directly modify the first-layer projection matrix; it performs poorly on NSFW tasks (I2P 7.09% but CLIP drops to 0.273). HiRM addresses this by decoupling the supervision location.
- **vs. ESD**: U-Net fine-tuning yields strong style erasure but severe degradation on object erasure (Object AA only 64.05%), with a training time of 7372s.
- **vs. SPEED**: Complementary relationship — SPEED modifies cross-attention U-Net weights for multi-concept erasure (5s/100 concepts), while HiRM modifies the text encoder for nudity erasure; the combination (S-HiRM-S) achieves the best overall performance.
- Broader implication: The paradigm of causal tracing → localization → decoupled intervention is generalizable to safety alignment in LLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The decoupled update–supervision design is highly insightful; this is the first work to systematically achieve full-category concept erasure within the text encoder.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage spans UnlearnCanvas, NSFW, adversarial attacks, Flux transfer, LoRA transfer, and synergy experiments — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; ablations are well-structured.
- Value: ⭐⭐⭐⭐⭐ 1.2s training, zero-shot cross-architecture transfer, and modular safety patching confer extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](../../CVPR2026/image_generation/groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
