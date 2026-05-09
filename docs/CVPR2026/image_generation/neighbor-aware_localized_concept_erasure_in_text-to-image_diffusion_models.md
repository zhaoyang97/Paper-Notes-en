---
title: >-
  [Paper Note] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Concept Erasure] This paper proposes NLCE, a training-free three-stage concept erasure framework for text-to-image diffusion models. It achieves precise localized erasure of target concepts through spectrally-weighted representation modulation, attention-guided spatial gating, and gated feature scrubbing, while explicitly preserving semantically neighboring concepts. NLCE outperforms existing methods on Oxford Flowers, Stanford Dogs, celebrity identity, and sensitive content erasure benchmarks.
tags:
  - CVPR 2026
  - Image Generation
  - Concept Erasure
  - Diffusion Models
  - Neighbor Preservation
  - Training-Free
  - Localized Erasure
date: 2026-05-08
content_hash: 3406b043995aa3cc
---

# Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2603.25994](https://arxiv.org/abs/2603.25994)
**Code**: [https://github.com/alirezafarashah/NLCE](https://github.com/alirezafarashah/NLCE)
**Area**: Image Generation / AI Safety
**Keywords**: Concept Erasure, Diffusion Models, Neighbor Preservation, Training-Free, Localized Erasure

## TL;DR

This paper proposes NLCE, a training-free three-stage concept erasure framework for text-to-image diffusion models. It achieves precise localized erasure of target concepts through spectrally-weighted representation modulation, attention-guided spatial gating, and gated feature scrubbing, while explicitly preserving semantically neighboring concepts. NLCE outperforms existing methods on Oxford Flowers, Stanford Dogs, celebrity identity, and sensitive content erasure benchmarks.

## Background & Motivation

1. **State of the Field**: Concept erasure methods for T2I diffusion models fall into two categories: training-based approaches (e.g., ESD, MACE, SPM, which require fine-tuning) and training-free approaches (e.g., UCE, RECE, GLoCE, which intervene only at inference time). Localized erasure methods such as GLoCE aim to restrict edits to target regions only.
2. **Limitations of Prior Work**: The **Neighbor Gap** problem — erasing a fine-grained concept inadvertently degrades semantically adjacent concepts. For example, erasing one dog breed reduces generation quality for other breeds as well.
3. **Root Cause**: Concept representations are highly entangled in the embedding space, making it impossible for simple projection or suppression operations to precisely distinguish between target and neighboring concepts.
4. **Paper Goals**: To accurately erase a target concept without training, while preserving the semantic integrity of neighboring concepts.
5. **Starting Point**: A three-stage progressive erasure pipeline — spectrally-weighted modulation in representation space to suppress the target and reinforce neighbors, followed by attention-based localization of residual activations, and finally hard scrubbing for complete removal.
6. **Core Idea**: Explicitly modeling and protecting the "concept neighborhood" structure to enable precise rather than indiscriminate concept removal.

## Method

### Overall Architecture

NLCE intervenes in the cross-attention layers of the UNet at inference time across three stages, without modifying model weights (training-free). Stage 1 operates in the embedding space (modifying Key/Value projection matrices), while Stages 2 and 3 operate in the feature space at each denoising step.

### Key Designs

1. **Stage 1: Representation Space Modulation (Spectrally-Weighted Suppression + Neighbor Enhancement)**

   - **Function**: Attenuate the target concept's semantics at the embedding level while recovering the representations of neighboring concepts.
   - **Mechanism**: SVD is applied to the target concept embedding to obtain an orthonormal basis $U_{F_c}$, from which a spectrally-weighted projection $P_{F_c} = U_{F_c}\Lambda_{F_c}U_{F_c}^T$ is constructed, with weights $\lambda_i$ modulated by singular value importance (stronger suppression along more important directions). An analogous projection $P_{\mathcal{N}_c}$ is constructed for neighbor concepts. The final operator $P_c = (I - \beta P_{F_c}) + \gamma P_{\mathcal{N}_c} P_{F_c}$ is applied globally to $W_K$ and $W_V$. Neighbors are identified via Wikipedia retrieval, filtered by RoBERTa-based specificity scoring, and ranked by CLIP visual similarity.
   - **Design Motivation**: Unlike GLoCE's gated low-rank adapter — which may miss concept reactivation through indirect attention paths — global application is more reliable. Spectral weighting ensures suppression intensity is proportional to semantic importance.

2. **Stage 2: Attention-Guided Spatial Gating**

   - **Function**: Localize spatial regions where residual activations of the target concept persist.
   - **Mechanism**: Each denoising step involves two forward passes (a dry pass followed by a real pass). The first pass extracts attention maps from DownBlock-2. Tokens with high overlap with the target subspace ($s_j = \|P_{F_c}x_j\|_2 > \delta_{\text{token}}$) are flagged as "live tokens." Their attention maps are aggregated into a spatial gating map $G_t(x,y)$. In the second forward pass, attention from live tokens is suppressed within gated regions: $A^\ell(x,y,j) \leftarrow (1-G_t)\cdot A^\ell(x,y,j)$.
   - **Design Motivation**: Since Stage 1 is a global operation and may leave residuals, Stage 2 uses spatial attention to precisely localize where the target concept remains active.

3. **Stage 3: Gated Feature Hard Scrubbing**

   - **Function**: Completely eliminate residual target signals within gated spatial regions.
   - **Mechanism**: The gating map from Stage 2 is upsampled to each UNet layer's resolution and binarized (threshold $\delta_{\text{scrub}}$). Hidden features at positions where the mask equals 1 are set to zero: $h_t^\ell(x,y) \leftarrow \mathbf{0}$. This constitutes an irreversible hard erasure.
   - **Design Motivation**: Projection-based suppression can theoretically be recovered; hard zeroing guarantees strict safety.

### Loss & Training

No training is involved. All operations are performed at inference time. Key hyperparameters:
- $\beta, \gamma \in [0,1]$: control the strength of target suppression and neighbor enhancement.
- $\delta_{\text{token}}$: threshold for live token detection.
- $\delta_{\text{scrub}}$: threshold for hard-scrubbing gating.
- In multi-concept scenarios, activated concept operators are detected per prompt and composed as $P_{\text{multi}} = \prod_{c\in\mathcal{A}} P_c$.

## Key Experimental Results

### Main Results

**Oxford Flowers / Stanford Dogs Fine-Grained Erasure**:

| Method | Alpine Sea Holly Acc_t↓/Acc_r↑/Ho↑ | Bluetick Acc_t↓/Acc_r↑/Ho↑ |
|--------|-----------------------------------|-----------------------------|
| GLoCE | 32.0/78.91/73.05 | 28.0/73.59/72.79 |
| RECE | 0.0/64.85/78.68 | 0.0/73.33/84.62 |
| **NLCE** | **0.0/82.06/90.15** | **0.0/75.91/86.31** |

**Celebrity Identity Erasure**:

| Method | Anna Kendrick Acc_t↓/Ho↑ | Elon Musk Acc_t↓/Ho↑ |
|--------|--------------------------|----------------------|
| SLD | 0.0/96.55 | 3.33/94.28 |
| GLoCE | 1.33/96.63 | 0.67/97.29 |
| **NLCE** | **0.0/96.91** | **0.0/96.55** |

### Ablation Study

Progressive effect of adding each stage (derived from Figure 9 of the paper):

| Configuration | Effect |
|---------------|--------|
| Stage 1 only | Basic erasure with possible residuals |
| Stage 1 + 2 | More thorough erasure with spatial precision |
| Stage 1 + 2 + 3 | Complete erasure with no residuals |

The degree of reliance on all three stages varies by dataset: Stage 1 alone suffices for simple scenarios, while the full pipeline is required for complex cases.

### Key Findings

- NLCE achieves the highest Acc_r and Harmonic score (Ho) across all fine-grained datasets, demonstrating superior neighbor preservation.
- GLoCE still exhibits relatively high Acc_t (e.g., 32%), indicating insufficient erasure from lightweight editing; NLCE reduces this to nearly 0% in almost all cases.
- On I2P sensitive content erasure, NLCE detects the least nudity while maintaining a relatively high CLIP Score of 29.70.
- Under simultaneous multi-concept erasure (10 breeds), NLCE maintains high Acc_r, whereas methods such as MACE, UCE, and RECE suffer severe collapse in retention accuracy.
- NLCE consistently achieves the lowest KID values, indicating the best preservation of visual quality.

## Highlights & Insights

- The **identification and formalization of the "Neighbor Gap" problem** clearly explains why existing methods fail in fine-grained settings. This insight has broad implications for the concept erasure field.
- The **three-stage progressive erasure design** transforms concept erasure from a blunt operation into a precise surgical procedure: weakening in representation space, localizing in attention space, and purging in feature space — with a well-defined objective at each stage.
- The **neighbor mining pipeline** (Wikipedia retrieval → RoBERTa specificity filtering → CLIP visual ranking) provides a practical method for constructing semantic neighborhoods, reusable in other tasks that require concept boundary delineation.

## Limitations & Future Work

- Two forward passes per denoising step (dry pass + real pass) double inference time.
- Neighbor mining relies on external resources (Wikipedia, RoBERTa), which may fail to retrieve appropriate neighbors for rare concepts.
- Hard scrubbing (zeroing) may introduce localized visual artifacts in certain cases.
- $\beta$ and $\gamma$ require manual tuning depending on the desired erasure strength, with optimal values varying across scenarios.

## Related Work & Insights

- **vs. GLoCE**: Both are localized erasure methods, but GLoCE uses a gated low-rank adapter without explicitly protecting neighbors. NLCE significantly outperforms GLoCE in neighbor retention (Acc_r gap of 3–7%) while achieving lower Acc_t.
- **vs. RECE**: Achieves thorough erasure but suffers from severe neighbor forgetting (Acc_r frequently 10–15% lower than NLCE) due to the absence of a neighbor protection mechanism.
- **vs. AdaVD**: Employs spectral suppression but lacks spatial localization and neighbor enhancement, making it less robust in multi-concept scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Neighbor-aware concept erasure is a well-motivated problem formulation, and the three-stage design is principled; however, the individual techniques (SVD projection, attention gating) are not novel in themselves.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers four distinct settings (fine-grained categories, celebrity identity, sensitive content, artistic style), includes multi-concept extension, and provides complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly presented, but the three-stage description is notation-heavy; the algorithmic flow could be conveyed more intuitively.
- **Value**: ⭐⭐⭐⭐ — Directly relevant to the safe deployment of T2I models, particularly for fine-grained concept control applications.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[ICLR 2026\] Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](../../ICLR2026/image_generation/localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)

<!-- RELATED:END -->
