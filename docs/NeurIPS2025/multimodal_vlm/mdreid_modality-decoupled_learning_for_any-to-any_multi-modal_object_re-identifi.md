---
title: >-
  [Paper Note] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification
description: >-
  [NeurIPS 2025][Multimodal VLM][Multi-modal ReID] This paper proposes MDReID, a framework that decouples modality features into modality-shared and modality-specific components, enabling object re-identification under arbitrary modality combinations (any-to-any ReID) and substantially outperforming existing methods in both modality-matched and modality-mismatched scenarios.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Multi-modal ReID
  - modality decoupling
  - cross-modal retrieval
  - any-to-any matching
  - metric learning
date: 2026-05-08
content_hash: 6236c8afe6a98f7a
---

# MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification

**Conference**: NeurIPS 2025
**arXiv**: [2510.23301](https://arxiv.org/abs/2510.23301)
**Code**: [GitHub](https://github.com/stone96123/MDReID)
**Area**: Multi-Modal VLM
**Keywords**: Multi-modal ReID, modality decoupling, cross-modal retrieval, any-to-any matching, metric learning

## TL;DR

This paper proposes MDReID, a framework that decouples modality features into modality-shared and modality-specific components, enabling object re-identification under arbitrary modality combinations (any-to-any ReID) and substantially outperforming existing methods in both modality-matched and modality-mismatched scenarios.

## Background & Motivation

**Background**: Multi-modal object re-identification (ReID) leverages complementary spectral information from RGB, NIR, TIR, and other modalities to improve recognition robustness in complex scenes.

**Limitations of Prior Work**: Existing methods (e.g., TOP-ReID, EDITOR) assume strict modality alignment between query and gallery sets; however, in real deployments, camera types and environments vary, leading to modality inconsistencies.

**Key Challenge**: When modalities are missing, reconstructing absent modality representations from available ones is an ill-posed problem, as unpredictable modality-specific information leads to suboptimal learning.

**Goal**: Design a flexible framework that supports retrieval under arbitrary query–gallery modality combinations, covering both modality-matched and modality-mismatched scenarios.

**Key Insight**: Decompose modality information into predictable and transferable shared features and unpredictable specific features, handling each separately.

**Core Idea**: Introduce learnable modality-shared and modality-specific tokens into a ViT to explicitly decouple representations, and reinforce the decoupling with an orthogonality loss and a knowledge discrepancy loss.

## Method

### Overall Architecture

MDReID is built on a Vision Transformer (ViT) backbone and comprises two core components:
- **Modality Decoupled Learning (MDL)**: Splits each modality's representation into modality-shared and modality-specific parts.
- **Modality-aware Metric Learning (MML)**: Further enhances feature decoupling through metric learning.

### Key Designs

1. **Modality Decoupled Learning (MDL)**:

    - **Function**: Extracts shared and specific features for each modality.
    - **Design Motivation**: Shared features support cross-modal retrieval (modality-mismatched scenario), while specific features retain modality-exclusive discriminative information (modality-matched scenario).
    - **Mechanism**: Two learnable tokens, $I_{sp}^M$ (modality-specific) and $I_{sh}^M$ (modality-shared), are prepended to the patch embedding sequence of each modality within ViT; after ViT encoding, decoupled features are obtained. A unified feature vector is constructed as:
    $v_{full} = [I_{sp}^R, I_{sp}^N, I_{sp}^T, I_{sh}^R, I_{sh}^N, I_{sh}^T]$
      Missing modalities are filled with zero vectors, with a binary availability mask indicating which modalities are present.
    - **Novelty**: Unlike TOP-ReID, which attempts to reconstruct missing modality representations, MDReID avoids the ill-posed reconstruction problem entirely.

2. **Similarity Computation**:

    - **Modality-specific similarity** $Sim_{sp}$: Compares specific features of the same modality only, with availability masks handling missing cases.
    - **Modality-shared similarity** $Sim_{sh}$: Computes a similarity matrix over all available shared feature pairs.
    $Sim_{total}(v_q, v_g) = (Sim_{sp} + Sim_{sh}) / 2$

3. **Representation Orthogonality Loss (ROL)**:

    - **Function**: Promotes channel-level aggregation of modality-shared features and enforces orthogonality between shared and specific features.
    - **Mechanism**: An ideal $6\times6$ target similarity matrix $A$ is defined, in which specific features form an identity matrix (mutual orthogonality), shared features are all-ones (mutual consistency), and cross-group entries are all zeros (orthogonality between groups). The loss minimizes the squared error between the actual similarity and the target:
    $L_{ROL} = \sum_{i,j} (V_{sim}(i,j) - A(i,j))^2$

4. **Knowledge Discrepancy Loss (KDL)**:

    - **Function**: Ensures that the combination of shared and specific features is more discriminative than either component alone.
    - **Mechanism**: Following the triplet loss paradigm, the combined features are required to yield smaller maximum positive-pair distances and larger minimum negative-pair distances:
    $L_{KDL} = \|D_p - 0\|_1 + \|D_n - 1\|_1$

### Loss & Training

The total loss is:
$$L = L_{ce} + L_{tri} + L_{MML}$$
where $L_{MML} = w_1 \times L_{ROL} + w_2 \times L_{KDL}$, with $w_1=1.5$ and $w_2=5.25$.

Training uses the Adam optimizer with a batch size of 64, a base learning rate of $3.5 \times 10^{-4}$, a ViT fine-tuning learning rate of $5 \times 10^{-6}$, and runs for 50 epochs. The backbone is the CLIP-Base visual encoder.

## Key Experimental Results

### Main Results

**Modality-matched scenario (RNT-to-RNT)**:

| Method | RGBNT201 mAP | RGBNT201 R-1 | RGBNT100 mAP | RGBNT100 R-1 | MSVR310 mAP | MSVR310 R-1 |
|--------|-------------|-------------|-------------|-------------|------------|------------|
| TOP-ReID | 72.3 | 76.6 | 81.2 | 96.4 | 35.9 | 44.6 |
| EDITOR | - | - | 82.1 | 96.4 | 39.0 | 49.3 |
| **MDReID** | **82.1** | **85.2** | **85.3** | **95.6** | **51.0** | **68.9** |

**Modality-mismatched scenario (average over 4 settings)**:

| Method | RGBNT201 Avg mAP | RGBNT100 Avg mAP | MSVR310 Avg mAP |
|--------|-----------------|-----------------|-----------------|
| TOP-ReID | 18.2 | 26.8 | 11.2 |
| EDITOR | 8.5 | 11.9 | 2.5 |
| **MDReID** | **21.6** | **38.6** | **22.1** |

### Ablation Study

| Config | MDL | $L_{ROL}$ | $L_{KDL}$ | mAP | R-1 |
|--------|-----|-----------|-----------|-----|-----|
| 1 (single classifier) | ✕ | ✕ | ✕ | 27.8 | 27.1 |
| 2 (MDL only) | ✓ | ✕ | ✕ | 39.4 | 38.2 |
| 3 (+ROL) | ✓ | ✓ | ✕ | 41.2 | 40.8 |
| 5 (full model) | ✓ | ✓ | ✓ | **43.2** | **42.3** |

### Key Findings

- MDL contributes the most: introducing modality-specific classifiers improves mAP from 27.8% to 39.4% (+11.6%).
- ROL and KDL each provide an additional ~2% gain, validating the effectiveness of decoupling constraints.
- In modality-mismatched scenarios, MDReID surpasses TOP-ReID by 10.9% mAP on MSVR310, demonstrating strong robustness.

## Highlights & Insights

- **Clear problem formulation**: The paper is the first to systematically define the image-level any-to-any multi-modal ReID problem.
- **Elegant decoupling design**: Feature decoupling is achieved naturally via shared/specific tokens combined with ViT self-attention, avoiding complex reconstruction modules.
- **Flexible similarity computation**: Availability-mask-based similarity calculation adaptively handles arbitrary missing modalities.
- **Intuitive ROL target matrix**: The $6\times6$ ideal similarity matrix has a clear rationale—specific features are mutually orthogonal, shared features are mutually consistent, and the two groups are orthogonal to each other.

## Limitations & Future Work

- Validation is limited to three modalities (RGB/NIR/TIR); performance on additional modalities (e.g., depth, event cameras) remains unexplored.
- Training assumes all modalities are available; modality-missing scenarios at training time are not investigated.
- The degree of decoupling between shared and specific features is sensitive to the loss hyperparameters ($w_1$, $w_2$).
- Scalability to open-set ReID or large-scale datasets is not discussed.

## Related Work & Insights

- **TOP-ReID**: Aggregates multi-spectral features via cyclic token permutation, but is constrained by the modality alignment assumption.
- **RLE**: Identifies cross-spectral feature prediction as an ill-posed problem, motivating the decoupling approach in this work.
- **CLIP backbone**: Leverages the strong representational capacity of pre-trained vision–language models to enhance feature quality.
- **Insights**: The modality decoupling paradigm is broadly transferable to other multi-modal tasks, such as vision–language and vision–audio fusion.

## Rating

- Novelty: ⭐⭐⭐⭐ The decoupling idea itself is not novel, but its application and realization in any-to-any ReID is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, multiple modality-matched/mismatched/missing scenarios, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, with complete formulations and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Addresses a practically important deployment challenge in multi-modal ReID with significant performance gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Omni-Mol: Multitask Molecular Model for Any-to-Any Modalities](omni-mol_multitask_molecular_model_for_any-to-any_modalities.md)
- [\[ICCV 2025\] Generalizable Object Re-Identification via Visual In-Context Prompting](../../ICCV2025/multimodal_vlm/generalizable_object_re-identification_via_visual_in-context_prompting.md)
- [\[ICCV 2025\] MAVias: Mitigate Any Visual Bias](../../ICCV2025/multimodal_vlm/mavias_mitigate_any_visual_bias.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)

</div>

<!-- RELATED:END -->
