---
title: >-
  [Paper Note] SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification
description: >-
  [CVPR2025][Optical-SAR] SDF-Net is proposed, which leverages the physical prior of ships as rigid bodies. It extracts scale-invariant gradient energy statistics in intermediate ViT layers as cross-modal geometric anchors. In the terminal layer, features are disentangled into modal-invariant shared features and modal-specific features, which are then fused via additive residuality, achieving state-of-the-art (SOTA) performance in optical-SAR ship re-identification.
tags:
  - "CVPR2025"
  - "Optical-SAR"
  - "Ship Re-identification"
  - "Structure Consistency"
  - "Feature Disentanglement"
  - "Gradient Energy"
date: 2026-05-08
content_hash: 908167eacde19d4b
---

# SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification

**Conference**: CVPR2025  
**arXiv**: [2603.12588](https://arxiv.org/abs/2603.12588)  
**Code**: [GitHub](https://github.com/cfrfree/SDF-Net)  
**Area**: Cross-Modal Retrieval / Remote Sensing  
**Keywords**: Optical-SAR, Ship Re-identification, Structure Consistency, Feature Disentanglement, Gradient Energy

## TL;DR
SDF-Net is proposed, which leverages the physical prior of ships as rigid bodies. It extracts scale-invariant gradient energy statistics in intermediate ViT layers as cross-modal geometric anchors. In the terminal layer, features are disentangled into modal-invariant shared features and modal-specific features, which are then fused via additive residuality, achieving state-of-the-art (SOTA) performance in optical-SAR ship re-identification.

## Background & Motivation
**Task Definition**: Cross-modal ship re-identification (ReID) aims to associate the same ship identity across optical and SAR images, which is a core task in maritime surveillance.

**Key Challenge**: Severe non-linear radiation distortion (NRD) exists between optical (passive reflection) and SAR (active microwave scattering) modalities. Since the texture appearance is highly modality-dependent, direct appearance alignment is unreliable.

**Limitations of Prior Work**:
   - Methods based on statistical distribution alignment lack the utilization of physical priors.
   - Generative methods (e.g., CycleGAN) are computationally expensive and may introduce hallucinated artifacts.
   - Pedestrian ReID methods (e.g., Hi-CMD, DEEN) are designed for non-rigid body deformations and are unsuitable for rigid ships.

**Key Insight**: Ships are rigid bodies whose geometric structures (contours, aspect ratios, space layouts) remain stable across modalities, whereas textures are modality-dependent. Intermediate layers of network representations abstract away low-level noise while retaining spatial topology, making them the optimal position to extract structural information.

## Method

### Overall Architecture (Based on ViT-B/16)

**(a) Input Stage: Cross-Modal Dual-Head Tokenizer**
- Optical and SAR images are mapped to a unified $C$-dimensional space via independent linear projection heads.
- Design Motivation: To neutralize low-level sensor discrepancies, preventing the shared self-attention from being dominated by modality-specific intensity biases.

**(b) Intermediate Stage: Structure-Aware Consistency Learning (SCL)**
- **Intermediate Layer Gradient Energy Extraction**: Feature maps $\mathbf{F}^{(B_s)}$ are extracted from the $B_s=6$-th layer (out of 12 layers).
- **Spatial Gradient Computation**: $\mathbf{G}_x(h,w) = \mathbf{F}(h,w+1) - \mathbf{F}(h,w-1)$ in both horizontal and vertical directions.
- **Spatial Integration**: $\mathbf{e}_x = \frac{1}{H'W'}\sum_{h,w}|\mathbf{G}_x(h,w)|$, compressing the gradient fields into a channel-wise structural descriptor.
- **Instance Normalization**: $\hat{\mathbf{f}}_{\text{struct}} = \text{IN}(\mathbf{f}_{\text{struct}})$, eliminating absolute magnitude discrepancies between modalities.
- **Prototype-Level Consistency Loss**: $\mathcal{L}_{\text{struct}} = \frac{1}{|\mathcal{I}|}\sum_i \|\mathbf{c}_i^o - \mathbf{c}_i^s\|_2^2$, aligning optical and SAR structural prototypes of the same identity.

**(c) Terminal Stage: Disentangled Feature Learning (DFL)**
- Two parallel linear projection heads decompose the terminal representation into: shared identity features $\mathbf{f}_{\text{sh}}$ + modality-specific features $\mathbf{f}_{\text{sp}}$.
- Orthogonal Constraint: $\mathcal{L}_{\text{orth}} = \mathbb{E}[|\langle\bar{\mathbf{f}}_{\text{sh}}, \bar{\mathbf{f}}_{\text{sp}}\rangle|]$, ensuring the independence of the two subspaces.
- Additive Residual Fusion: $\mathbf{f}_{\text{fuse}} = \mathbf{f}_{\text{sh}} + \mathbf{f}_{\text{sp}}$, parameter-free, where modality-specific features serve as residual supplements.

**Joint Optimization**: $\mathcal{L} = \mathcal{L}_{\text{id}} + \lambda_{\text{orth}}\mathcal{L}_{\text{orth}} + \lambda_{\text{struct}}\mathcal{L}_{\text{struct}}$
- Where $\mathcal{L}_{\text{id}}$ includes label-smoothed cross-entropy and weighted triplet loss.
- Hyperparameter Settings: $\lambda_{\text{orth}} = 10.0$, $\lambda_{\text{struct}} = 1.0$.

### Implementation Details
- ViT-B/16 backbone initialized with TransOSS pre-trained weights.
- Input size 256×128, with random horizontal flipping, cropping, and erasing augmentation.
- Strict cross-modal P×K sampling: 32 images per batch = 8 identities × 4 images (2 optical + 2 SAR).
- SGD optimizer, lr=5e-4 with linear warmup, for 100 epochs.
- Trained on a single RTX 3090 GPU, with PyTorch 2.2.2 and CUDA 11.8.
- Hyperparameters: $\lambda_{\text{orth}}=10.0$, $\lambda_{\text{struct}}=1.0$, SCL layer $B_s=6$.

## Key Experimental Results

### SOTA Comparison on HOSS-ReID Benchmark (mAP / Rank-1, %)

| Method | Type | All mAP | All R1 | O→S mAP | O→S R1 | S→O mAP | S→O R1 |
|------|-----|---------|--------|---------|--------|---------|--------|
| TransReID (ICCV21) | Single-Modal ReID | 48.1 | 60.8 | 27.3 | 18.5 | 20.9 | 11.9 |
| VersReID (TPAMI24) | Cross-Modal ReID | 49.3 | 59.7 | 25.7 | 13.8 | 27.7 | 17.9 |
| DEEN (CVPR23) | Cross-Modal ReID | 43.8 | 58.5 | 31.3 | 21.5 | 27.4 | 22.4 |
| TransOSS (ICCV25) | RS-Specific | 57.4 | 65.9 | 48.9 | 33.8 | 38.7 | 29.9 |
| **SDF-Net** | **RS-Specific** | **60.9** | **69.9** | **50.0** | **35.4** | **46.6** | **38.8** |

- All mAP +3.5%, Rank-1 +4.0% (vs. TransOSS).
- SAR→Optical mAP +7.9% (38.7→46.6), R1 +8.9% (29.9→38.8) — the most significant improvement, validating the efficacy of structural anchors on the SAR side.
- Pedestrian ReID methods (e.g., DEEN/VersReID) are unsuitable for optical-SAR scenarios, significantly lagging in performance, particularly on the O→S task.

### Ablation Study

| SCL | DFL | All mAP | All R1 | O→S mAP | S→O mAP |
|-----|-----|---------|--------|---------|---------|
| ✗ | ✗ | 58.6 | 67.6 | 46.5 | 44.5 |
| ✓ | ✗ | 59.2 | 66.5 | 47.6 | 46.6 |
| ✗ | ✓ | 59.8 | 69.9 | 49.3 | 41.4 |
| ✓ | ✓ | **60.9** | **69.9** | **50.0** | **46.6** |

- SCL primarily contributes to structural alignment in SAR→Optical (mAP 44.5→46.6), though using it alone slightly decreases R1 (due to overly strong alignment constraints).
- DFL primarily improves discriminative accuracy (R1 67.6→69.9), but when used alone, the S→O mAP drops to 41.4 (indicating that disentanglement without structural anchors is unreliable).
- The combination of both yields the optimal performance: SCL provides cross-modal geometric anchors, on top of which DFL refines identification capability.

## Highlights & Insights
- **Physical Prior-Driven Network Design**: Systematically embeds the physical knowledge of "ships are rigid bodies" into various stages of feature learning.
- **Innovative Utilization of Intermediate Layer Gradient Energy**: Avoids raw pixels (noise) and high-level features (overly abstract) to capture structural topology in intermediate layers.
- **Physical Explanation of Instance Normalization**: Not merely a technical trick, but formulated from the physical mechanisms of SAR microwave scattering vs. optical reflection, mapping heterogeneous magnitudes to a unified unit-variance manifold.
- **Simplicity of Additive Residual Fusion**: A parameter-free fusion strategy where modality-specific features are treated as residual supplements rather than discarded as noise.
- **Open-source Code**

## Limitations & Future Work
- Validated only on a single dataset, HOSS-ReID (1,063 training images, 769 testing images); the scale is limited, and generalization remains to be verified.
- The rigid-body assumption might not fully hold in certain scenarios, such as ships carrying deformable cargo.
- The choice of the intermediate layer $B_s=6$ is empirical, and hyperparameters may require tuning for backbones with different depths.
- Training requires a strict cross-modal P×K sampling strategy, which places demands on the optical/SAR paired nature of the dataset.
- Although additive fusion is simple, it may be less flexible than adaptive gating in extreme scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of physical priors, intermediate-layer gradient energy, and disentangled fusion is novel in the SAR ReID field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three ablation dimensions (modules, fusion strategies, layer choices) plus comprehensive SOTA comparisons.
- Writing Quality: ⭐⭐⭐⭐ In-depth discussion of physical motivations and complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Provides an effective, physically driven paradigm for cross-modal retrieval in remote sensing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structure-Aware Spectral Sparsification via Uniform Edge Sampling](../../NeurIPS2025/others/structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)
- [\[CVPR 2025\] TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception](traf-align_trajectory-aware_feature_alignment_for_asynchronous_multi-agent_perce.md)
- [\[AAAI 2026\] ParaMETA: Towards Learning Disentangled Paralinguistic Speaking Styles Representations](../../AAAI2026/others/parameta_towards_learning_disentangled_paralinguistic_speaking_styles_representa.md)
- [\[AAAI 2026\] Structure-Aware Encodings of Argumentation Properties for Clique-width](../../AAAI2026/others/structure-aware_encodings_of_argumentation_properties_for_clique-width.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](../../AAAI2026/others/how_to_marginalize_in_causal_structure_learning.md)

</div>

<!-- RELATED:END -->
