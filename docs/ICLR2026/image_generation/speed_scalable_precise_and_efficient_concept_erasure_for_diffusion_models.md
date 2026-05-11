---
title: >-
  [Paper Note] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models
description: >-
  [ICLR 2026][Image Generation][Concept Erasure] SPEED proposes a closed-form model editing method based on null space constraints, refining the preservation set through three complementary techniques—Influence-based Prior…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Concept Erasure"
  - "Null Space Constraint"
  - "Model Editing"
  - "Prior Preservation"
  - "Multi-Concept Erasure"
date: 2026-05-08
content_hash: dd9c250903bad39f
---

# SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2503.07392](https://arxiv.org/abs/2503.07392)
**Code**: [GitHub](https://github.com/Ouxiang-Li/SPEED)
**Area**: Diffusion Models / Safety / Unlearning
**Keywords**: Concept Erasure, Null Space Constraint, Model Editing, Prior Preservation, Multi-Concept Erasure

## TL;DR

SPEED proposes a closed-form model editing method based on null space constraints, refining the preservation set through three complementary techniques—Influence-based Prior Filtering (IPF), Directional Prior Augmentation (DPA), and Invariance Equality Constraint (IEC)—to achieve scalable (erasing 100 concepts within 5 seconds), precise (zero semantic loss on non-target concepts), and efficient concept erasure.

## Background & Motivation

**Background**: Concept erasure in T2I diffusion models follows two major paradigms—training-based (fine-tuning, e.g., ESD, MACE) and editing-based (closed-form, e.g., UCE, RECE). Editing-based methods are naturally suited for multi-concept scenarios as they require no additional training.

**Limitations of Prior Work**: Editing-based methods such as UCE employ weighted least squares to jointly optimize the erasure error $e_1$ and the preservation error $e_0$; however, $e_0$ has a provably non-zero lower bound. As the number of erased concepts grows, the accumulated $e_0$ causes semantic degradation in non-target concepts.

**Key Challenge**: Null space methods (e.g., AlphaEdit) can enforce $e_0 = 0$ exactly, but enlarging the preservation set causes the feature matrix to approach full rank, shrinking the null space dimension $\dim = d_0 - \text{rank}(\mathbf{C}_0\mathbf{C}_0^\top)$. Resorting to an approximate null space reintroduces semantic degradation.

**Goal**: To simultaneously guarantee (a) erasure effectiveness, (b) zero loss on non-target concepts, and (c) computational efficiency in multi-concept erasure.

**Key Insight**: Rather than naively enlarging the preservation set, the paper strategically refines it—filtering out low-influence concepts to prevent full rank, and augmenting high-influence concepts to improve coverage.

**Core Idea**: By refining the preservation set with prior knowledge, the null space constraint remains accurate at scale, achieving lossless prior preservation with $e_0 = 0$.

## Method

### Overall Architecture

The method takes three concept sets as input: an erasure set $\mathbf{E}$ (target concepts), an anchor set $\mathbf{A}$ (substitute concepts, e.g., Snoopy→Dog), and a preservation set $\mathbf{R}$ (non-target concepts). A closed-form update $\bm{\Delta}\mathbf{P}$ is computed for the projection weights $\mathbf{W}$ of cross-attention layers, where $\mathbf{P}$ is the null space projection matrix corresponding to the refined preservation set $\mathbf{R}_{\text{refine}}$.

### Key Designs

1. **Influence-based Prior Filtering (IPF)**:

    - *Function*: Quantifies the degree to which each non-target concept is affected by erasure, and filters out low-influence concepts.
    - *Mechanism*: First computes the closed-form update $\bm{\Delta}_{\text{erase}}$ using only the erasure term $e_1$, then calculates the prior shift $\|\bm{\Delta}_{\text{erase}} \bm{c}_0\|^2$ for each preserved concept, retaining only those above the mean.
    - *Design Motivation*: Reduces the size of the preservation set to prevent the correlation matrix from approaching full rank, thereby preserving null space accuracy.

2. **Directional Prior Augmentation (DPA)**:

    - *Function*: Augments the filtered preservation set with directional noise.
    - *Mechanism*: Applies SVD to the parameter matrix $\mathbf{W}$ and constructs a projection $\mathbf{P}_{\text{min}}$ from the smallest singular directions. Random noise is projected onto this direction and added to the concept embedding: $\bm{c}_0' = \bm{c}_0 + \bm{\epsilon} \cdot \mathbf{P}_{\text{min}}$.
    - *Design Motivation*: Compared to random noise augmentation, directional noise yields smaller semantic distances under the mapping of $\mathbf{W}$, avoiding rank space waste caused by semantically meaningless embeddings.

3. **Invariance Equality Constraint (IEC)**:

    - *Function*: Forces the outputs of the [SOT] token and null-text embedding to remain unchanged before and after erasure.
    - *Mechanism*: Incorporates an equality constraint $(\bm{\Delta}\mathbf{P})\mathbf{C}_2 = \mathbf{0}$ into the closed-form optimization, solved via Lagrange multipliers.
    - *Design Motivation*: These invariant embeddings participate in all generation processes; protecting them naturally preserves prior knowledge.

### Loss & Training

The final closed-form solution is:

$$(\bm{\Delta}\mathbf{P})_{\text{Ours}} = \mathbf{W}(\mathbf{C}_*\mathbf{C}_1^\top - \mathbf{C}_1\mathbf{C}_1^\top)\mathbf{P}\mathbf{Q}\mathbf{M}$$

where $\mathbf{M} = (\mathbf{C}_1\mathbf{C}_1^\top\mathbf{P} + \mathbf{I})^{-1}$ and $\mathbf{Q} = \mathbf{I} - \mathbf{M}\mathbf{C}_2(\mathbf{C}_2^\top\mathbf{P}\mathbf{M}\mathbf{C}_2)^{-1}\mathbf{C}_2^\top\mathbf{P}$. The entire process requires no training and consists purely of matrix operations.

## Key Experimental Results

### Multi-Concept Erasure (Celebrities)

| # Erased | Metric | UCE | MACE | RECE | **SPEED** |
|----------|--------|-----|------|------|-----------|
| 10 | $\text{Acc}_r$↑ / $H_o$↑ | 71.19 / 83.10 | 87.73 / 92.75 | 67.43 / 80.44 | **89.09 / 93.42** |
| 50 | $\text{Acc}_r$↑ / $H_o$↑ | 31.94 / 48.41 | 84.31 / 90.03 | 19.77 / 32.95 | **88.48 / 92.34** |
| 100 | $\text{Acc}_r$↑ / $H_o$↑ | 20.92 / 34.60 | 80.20 / 87.06 | 23.71 / 38.16 | **85.54 / 89.63** |
| 100 | Runtime (s) | 2.1 | 1736 | 11.0 | **5.0** |

### Ablation Study

| Configuration | Target CS↓ | Non-target FID↓ | MS-COCO FID↓ |
|---------------|------------|-----------------|--------------|
| Baseline (Eq.3, no refinement) | 27.20 | 50.43 | 26.33 |
| +IEC | 27.20 | 48.17 | 24.95 |
| +IEC+IPF | 26.68 | 38.02 | 20.57 |
| +IEC+IPF+DPA (SPEED) | **26.29** | **29.35** | **20.36** |

### Key Findings

- IPF contributes most significantly: non-target FID drops from 48.17 to 38.02, confirming that filtering irrelevant concepts to avoid full rank is the key mechanism.
- DPA outperforms random augmentation (RPA): directional noise maintains semantic consistency, reducing non-target FID from 32.62 to 29.35.
- SPEED achieves a 350× speedup over MACE (5s vs. 1736s) with superior prior preservation.
- The method transfers to SDXL and SDv3 (DiT architecture) and supports knowledge editing (modifying anchor concepts).

## Highlights & Insights

- **Null space constraint + refinement = provable $e_0 = 0$**: Prior preservation is exact rather than approximate, an advantage that becomes increasingly pronounced as the number of concepts grows.
- **IPF's prior shift metric**: Quantifying influence via the closed-form erasure update is both elegant and computationally efficient, requiring no training.
- **DPA's directional noise design**: Projecting noise onto the smallest singular directions of $\mathbf{W}$ is a clever trick that is transferable to other model editing tasks.

## Limitations & Future Work

- Only cross-attention layer weights are modified, limiting influence on internal representations that bypass cross-attention (e.g., self-attention).
- The closed-form solution relies on linear assumptions and may not perfectly handle highly nonlinear concept interactions.
- The preservation set must still be predefined; protection of entirely unknown non-target concepts cannot be guaranteed.
- Erasure effectiveness in certain scenarios falls short of the most aggressive training-based methods, though this is offset by speed and prior preservation.

## Related Work & Insights

- **vs. UCE**: Both are editing-based methods, but UCE's weighted least squares yields a non-zero lower bound on $e_0$, causing severe degradation in the multi-concept regime ($\text{Acc}_r$ drops to only 20.92% at 100 concepts).
- **vs. MACE**: This training-based method uses LoRA for large-scale erasure with comparable quality, but requires 1736 seconds versus SPEED's 5 seconds.
- **vs. RECE**: Iterative adversarial training improves robustness but scales poorly ($\text{Acc}_r$ only 23.71% at 100 concepts).
- Transferring null space constraints from continual learning to concept erasure is a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of null space constraints and prior refinement is the core innovation; IPF/DPA/IEC are well-motivated designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers few/multi/implicit concept erasure tasks, thorough ablations, and cross-architecture validation.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear and figures are highly illustrative.
- Value: ⭐⭐⭐⭐⭐ Erasing 100 concepts in 5 seconds with state-of-the-art prior preservation makes this extremely practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](../../CVPR2026/image_generation/groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
