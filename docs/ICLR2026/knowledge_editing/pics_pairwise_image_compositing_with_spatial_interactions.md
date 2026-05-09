---
title: >-
  [Paper Note] PICS: Pairwise Image Compositing with Spatial Interactions
description: >-
  [ICLR 2026][Knowledge Editing][image compositing] This paper proposes PICS—a parallel pairwise image compositing method that simultaneously composes two objects in a single inference pass via mask-guided MoE and adaptive α-blending within an Interaction Transformer, explicitly modeling spatial interactions such as occlusion and contact, and consistently outperforming existing sequential compositing methods.
tags:
  - ICLR 2026
  - Knowledge Editing
  - image compositing
  - diffusion model
  - Mixture-of-Experts
  - spatial interaction
  - α-blending
date: 2026-05-08
content_hash: 04dfd3951b413699
---

# PICS: Pairwise Image Compositing with Spatial Interactions

**Conference**: ICLR 2026  
**arXiv**: [2603.06873](https://arxiv.org/abs/2603.06873)  
**Code**: [github.com/RyanHangZhou/PICS](https://github.com/RyanHangZhou/PICS)  
**Area**: Knowledge Editing  
**Keywords**: image compositing, diffusion model, Mixture-of-Experts, spatial interaction, α-blending

## TL;DR

This paper proposes PICS—a parallel pairwise image compositing method that simultaneously composes two objects in a single inference pass via mask-guided MoE and adaptive α-blending within an Interaction Transformer, explicitly modeling spatial interactions such as occlusion and contact, and consistently outperforming existing sequential compositing methods.

## Background & Motivation

**State of the Field**: Recent diffusion-based methods have demonstrated strong performance in single-object compositing, supporting object insertion into diverse backgrounds using visual prompt conditioning.

**Limitations of Prior Work**: Existing methods are inherently single-round—only one object is inserted per pass. When multiple objects are inserted sequentially, later operations overwrite earlier content, disrupting occlusion ordering and physical consistency.

**Root Cause — Painter's Algorithm Failure**: When objects are composed sequentially by depth order (far to near), the first inserted object is often mistaken as part of the background, leading to partial deletion, distortion, or excessive blending.

**Lack of Explicit Inter-Object Interaction Modeling**: Real-world scenes involve fundamental spatial relations—support, containment, occlusion, and deformation—yet prevailing training data construction schemes (foreground-background dichotomy) ignore these relations.

**Pairwise Relations as the Atomic Unit of Compositional Reasoning**: The spatial plausibility of any multi-object scene can be decomposed into pairwise object relations; therefore, modeling pairwise interactions is a critical step toward multi-object compositing.

**Core Idea**: The image is partitioned into the background, exclusive regions of each object, and their overlap region, which are handled by dedicated routed experts; attention-gated α-blending resolves fusion in the overlap region.

## Method

### Overall Architecture

Built upon a latent diffusion model with ControlNet conditioning. Inputs consist of a masked background $\mathbf{x}_{bg}$, two objects $\{\mathbf{x}_a, \mathbf{x}_b\}$, and their binary masks $\{\mathbf{m}_a, \mathbf{m}_b\}$. The model generates a complete composite image containing both objects in a single forward pass.

**Data Construction** (self-supervised composition-by-decomposition):
- Objects are decomposed from target images by erasing their regions from the background
- Union mask $\mathbf{m}_u$, intersection mask $\mathbf{m}_{ab}$, and exclusive masks $\mathbf{m}_a^{ex}$, $\mathbf{m}_b^{ex}$ are computed
- Training objective: reconstruct the original image from its decomposed components

### Key Designs

**1. Interaction Transformer Block**

Each block consists of:
- **Self-attention**: captures global dependencies
- **Mask-guided MoE**: routes tokens to specialized experts based on spatial region
- **Residual aggregation + FFN**: gated outputs are merged and refined by FFN

**2. Four Expert Types in Spatially-Aware MoE**:

| Expert Type | Region | Operation |
|-------------|--------|-----------|
| Background expert | $\bar{\mathbf{m}}_{bg}$ | Identity mapping (preserves background) |
| Object-a exclusive expert | $\bar{\mathbf{m}}_a^{ex}$ | Cross-attention: background query → object-a code |
| Object-b exclusive expert | $\bar{\mathbf{m}}_b^{ex}$ | Cross-attention: background query → object-b code |
| Overlap expert | $\bar{\mathbf{m}}_{ab}$ | Attention-gated α-blending (see below) |

**3. Adaptive α-Blending Mechanism**:

Fusion in the overlap region is the central design of this work:
- A gating query $\mathbf{q}_g$ is generated from the background deep representation $\mathbf{z}^{l-1}$
- $\mathbf{q}_g$ performs cross-attention with each object code to obtain aggregated representations $\tilde{\mathbf{c}}_a$, $\tilde{\mathbf{c}}_b$
- Compatibility scores are computed: $s_p = \langle \mathbf{q}_g, \tilde{\mathbf{c}}_p \rangle / \sqrt{d}$
- Blending weights $\alpha$ are obtained via softmax with temperature $\tau$
- Fusion: $\mathbf{c}_{ab} = \alpha \tilde{\mathbf{c}}_a + (1-\alpha) \tilde{\mathbf{c}}_b$

**Key Property**: The gating query encodes learned occlusion semantics rather than appearance cues, enabling $\alpha$ to adaptively reflect which object should dominate at each spatial position. Experiments confirm that $\alpha$ aligns with actual visibility relationships and is invariant to object input order.

### Geometry-Aware Data Augmentation

1. **Multi-view shape prior**: A single-view 3D reconstruction model renders $K$ auxiliary viewpoints; encodings are fused into a compact multi-view descriptor via MLP
2. **In-plane rotation augmentation**: Object images are randomly rotated by $\theta \sim \mathcal{U}(-\pi/6, \pi/6)$

### Loss & Training

- Self-supervised recomposition loss: reconstructs the original image
- Standard latent diffusion denoising loss
- No additional annotated data required

## Key Experimental Results

### Main Results

**Object Recomposition (LVIS validation set)**:

| Method | mPSNR ↑ | mSSIM ↑ | mLPIPS ↓ | PSNR ↑ | FID ↓ | LPIPS ↓ |
|--------|---------|---------|----------|--------|-------|---------|
| PbE (CVPR'23) | 10.24 | 0.4241 | 0.4535 | 15.29 | 34.93 | 0.4138 |
| AnyDoor (CVPR'24) | 11.62 | 0.5283 | 0.4185 | 17.12 | 27.17 | 0.3302 |
| OmniPaint (ICCV'25) | 12.20 | 0.3096 | 0.4618 | 16.09 | 26.25 | 0.3542 |
| **PICS (ours)** | **13.88** | **0.5823** | **0.3221** | **18.27** | **24.99** | **0.2530** |

Gains are particularly pronounced on overlap-region metrics (mPSNR/mSSIM/mLPIPS), demonstrating the advantage of explicitly modeling the overlap region.

**Object Compositing (DreamBooth test set)**:

| Method | FID ↓ | CLIP-score ↑ | DINOv2-score ↑ | DreamSim ↓ |
|--------|-------|-------------|----------------|------------|
| ObjectStitch | 260.4 | 51.35 | 0.3203 | 0.3374 |
| AnyDoor | 274.1 | 51.24 | 0.3401 | 0.2733 |
| InsertAnything (AAAI'26) | 266.0 | 50.54 | 0.3612 | 0.2934 |
| **PICS (ours)** | **255.5** | **54.02** | 0.3631 | 0.3054 |

### Ablation Study

| Setting | Key Change | FID ↓ | CLIP-score ↑ |
|---------|-----------|-------|-------------|
| #1 MLP + single-view | Baseline | 173.1 | 74.6 |
| #2 ITB + single-view | MLP→ITB | 165.2 | 76.3 |
| #3 ITB + rotation aug. | +in-plane rotation | 162.5 | 74.9 |
| #4 ITB + multi-view | +multi-view prior | 158.2 | 77.3 |
| #5 ITB + combined data | +1M training set | **151.3** | **79.1** |

Each component contributes consistent improvements; scaling training data (LVIS→1M composite dataset) yields the largest gain.

### Key Findings

1. **Parallel vs. sequential compositing**: The parallel approach effectively avoids error accumulation from sequential compositing, especially at occlusion boundaries.
2. **α-blending learns true visibility**: The sign of $\Delta s = s_a - s_b$ aligns with actual object visibility and is invariant to input order.
3. **Evolution of α during denoising**: Coarse in early steps → decisive in mid steps → refined in late steps, consistent with the refinement dynamics of diffusion models.
4. **Scalability to 3/4 objects**: Models additionally trained for 3/4 objects maintain consistent occlusion ordering and contact relationships.
5. **User study**: PICS ranks first in realism (17.7%) and consistency (22.5%).

## Highlights & Insights

- **The design intuition of mask-guided MoE routing is elegant**: Different regions naturally require different processing strategies—the background should remain unchanged, exclusive regions receive a single object's information, and overlap regions require arbitration—this prior knowledge is hard-coded into the architecture.
- **Adaptive α-blending outperforms hard occlusion masks**: The model autonomously learns occlusion semantics rather than relying on manually specified depth ordering.
- **Self-supervised training eliminates annotation costs**: Composition-by-decomposition automatically constructs training pairs from existing images.
- **Input-order invariance**: The α-blending mechanism ensures that swapping the labels of objects a and b does not affect the output, a desirable symmetry property.

## Limitations & Future Work

- **Limited shape encoder capacity**: Geometric and texture degradation occasionally occurs in highly cluttered environments (see Figure 10 failure cases).
- **Restricted to pairwise compositing**: Although 3/4-object extensions are demonstrated, the number of MoE experts grows exponentially with the number of objects, requiring architectural redesign for larger object counts.
- **Backbone limitations**: The method is built on a standard diffusion model and does not adopt stronger flow-matching backbones such as FLUX (which gives OmniPaint an advantage on some metrics).
- **Dataset diversity**: Training is primarily conducted on LVIS; although a combined dataset is incorporated, generalization to highly specialized domains (e.g., medical image compositing) remains unverified.
- **Absence of text conditioning**: The method is purely image-prompted and does not support text descriptions to guide compositing location or style.

## Related Work & Insights

- **AnyDoor (CVPR'24)**: Uses auxiliary edge maps to preserve semantics but lacks inter-object interaction modeling, producing artifacts at occlusion boundaries.
- **FreeCompose (ECCV'24)**: Supports zero-shot compositing but does not explicitly handle spatial interactions.
- **InsertAnything (AAAI'26)**: One of the latest comparison methods; PICS surpasses it on both FID and CLIP-score.
- **Multi-round editing (Zhou et al., 2025; Avrahami et al., 2025)**: Complementary to this work—multi-round editing faces analogous cross-round consistency challenges.
- **Insights**: The mask-guided routing paradigm of MoE can be generalized to other visual generation tasks requiring region-specialized processing, such as foreground/background separation in video editing or independent manipulation of object regions in 3D scene editing.

## Rating

- ⭐ Novelty: 4/5 — The combined design of parallel compositing + mask-guided MoE + α-blending is novel, though each individual component is not entirely new.
- ⭐ Experimental Thoroughness: 4.5/5 — Covers multiple datasets, multiple metrics, user studies, ablation experiments, and multi-object extensions.
- ⭐ Writing Quality: 4/5 — Clear structure, rich illustrations, and complete derivations.
- ⭐ Value: 4/5 — Directly applicable to virtual try-on, scene editing, and similar tasks; code is publicly available.
- Occlusion ordering must be determined in advance.

## Rating
- Novelty: ⭐⭐⭐⭐ MoE + spatial masking for parallel compositing is a novel solution
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-scenario coverage
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated
- Value: ⭐⭐⭐⭐ Addresses a practical pain point in multi-round compositing

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Rote Learning Considered Useful: Generalizing over Memorized Training Examples](rote_learning_considered_useful_generalizing_over_memorized_training_examples.md)
- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [\[ICLR 2026\] Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs](rote_learning_considered_useful_generalizing_over_memorized_data_in_llms.md)
- [\[ICLR 2026\] Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)

<!-- RELATED:END -->
