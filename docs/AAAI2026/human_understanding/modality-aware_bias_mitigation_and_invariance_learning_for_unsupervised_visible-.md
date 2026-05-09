---
title: >-
  [Paper Note] Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification
description: >-
  [AAAI 2026][Human Understanding][Cross-Modality Person Re-Identification] To address the core problem of unreliable cross-modality associations in unsupervised visible-infrared person re-identification (USVI-ReID), this paper proposes modality-aware Jaccard distance correction and a "split-and-contrast" invariance learning strategy. By eliminating modality bias, the method enables reliable global cross-modality clustering and feature alignment, achieving state-of-the-art performance on SYSU-MM01 and RegDB.
tags:
  - AAAI 2026
  - Human Understanding
  - Cross-Modality Person Re-Identification
  - Unsupervised Learning
  - Modality Bias Mitigation
  - Jaccard Distance Correction
  - Global Clustering
date: 2026-05-08
content_hash: deed593ca24b1bb7
---

# Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification

**Conference**: AAAI 2026
**arXiv**: [2512.07760](https://arxiv.org/abs/2512.07760)
**Code**: [github](https://github.com/Terminator8758/BMIL)
**Area**: Human Understanding
**Keywords**: Cross-Modality Person Re-Identification, Unsupervised Learning, Modality Bias Mitigation, Jaccard Distance Correction, Global Clustering

## TL;DR

To address the core problem of unreliable cross-modality associations in unsupervised visible-infrared person re-identification (USVI-ReID), this paper proposes modality-aware Jaccard distance correction and a "split-and-contrast" invariance learning strategy. By eliminating modality bias, the method enables reliable global cross-modality clustering and feature alignment, achieving state-of-the-art performance on SYSU-MM01 and RegDB.

## Background & Motivation

Visible-infrared person re-identification (VI-ReID) involves matching the same pedestrian across daytime (visible cameras) and nighttime (infrared cameras), with important applications in nocturnal surveillance and person retrieval. The unsupervised setting (USVI-ReID) aims to accomplish this task without any annotation.

The **core challenge** lies in the substantial cross-modality gap (visible images are color RGB while infrared images are grayscale thermal), making cross-modality association estimation extremely difficult.

Key limitations of existing methods:

**Deficiencies of local matching strategies**: Mainstream methods first cluster within each modality and then match cross-modality clusters via optimal transport (e.g., the Hungarian algorithm). The drawback is that noise in intra-modality clustering propagates through the matching process, and global instance-level similarity relationships are ignored.

**Obstacles in naive global clustering**: Directly performing global clustering over all images seems reasonable, but modality bias causes failure—due to the modality gap, intra-modality image similarity is far higher than cross-modality similarity. In Jaccard distance KNN retrieval, the retrieved neighbors are dominated by same-modality instances (as shown in Figure 1(a)), further biasing distance computation and preventing global clustering from effectively associating cross-modality instances.

**Insufficient cross-modality representation learning**: Even when associations are established, visible and infrared features within the same global cluster exhibit substantially different distributions (a "bimodal" distribution), making a single centroid prototype inadequate for characterizing the mixed-modality cluster.

This paper systematically addresses cross-modality learning from two complementary perspectives: **bias mitigation** and **invariance learning**.

## Method

### Overall Architecture

The method follows a standard two-stage learning paradigm:
- **Stage 1**: Intra-modality learning — iterative clustering and prototype contrastive learning performed separately within the visible and infrared modalities.
- **Stage 2**: Cross-modality learning — on top of intra-modality learning, bias-corrected global association and modality-invariant representation learning are introduced.

A dual-stream backbone (ResNet-50 + AGW) is adopted, with independent initial convolutional blocks for the visible and infrared streams and shared parameters elsewhere.

### Key Designs

#### 1. **Intra-Modality Learning Baseline**: Subset Clustering to Alleviate Over-Clustering

Since visible images typically far outnumber infrared images (22k visible vs. 12k infrared in SYSU-MM01), DBSCAN tends to produce over-clustering on the visible modality (predicted cluster count far exceeding the true number of identities).

**Subset clustering strategy**: A fixed proportion (e.g., 0.5) of visible images is randomly sampled per epoch for clustering. This offers two advantages: reducing the average number of images per identity facilitates clustering, and random sampling ensures full-set coverage over time.

The intra-modality contrastive loss adopts standard InfoNCE:
$$\mathcal{L}_{intra}^v = -\sum_{i=1}^{N_b} \log \frac{\exp(\mathcal{M}^v[\tilde{y}_i]^T f_\theta(x_i^v)/\tau)}{\sum_{j=1}^{C^v} \exp(\mathcal{M}^v[j]^T f_\theta(x_i^v)/\tau)}$$

#### 2. **Modality-Aware Jaccard Distance Correction**: Global Association with Bias Elimination

This is the paper's most central contribution. The core idea is to enforce balanced contributions from intra-modality and inter-modality neighbors at the critical step of Jaccard distance computation.

**KNN correction**: Instead of using the conventional global top-$k_1$ neighbors, the method retrieves $k_1/2$ neighbors from within each modality and $k_1/2$ from across modalities, then merges and re-ranks:
$$N^*(x_i, k_1) = N^{intra}(x_i, k_1/2) \cup N^{inter}(x_i, k_1/2)$$

**Balanced local query expansion**: Local query expansion likewise employs modality-balanced $k_2$ neighbors:
$$\overline{Dist}(x_i) = \frac{1}{k_2} \sum_{j \in N^*(x_i, k_2)} Dist(x_j)$$

These two corrections ensure that intra-modality and inter-modality neighbors contribute fairly to distance computation, enabling global clustering to effectively associate cross-modality instances. The key distinction from prior methods is that they only consider modality balance in the local query expansion step, whereas this paper introduces a fundamental correction at the KNN retrieval stage.

#### 3. **"Split-and-Contrast" Modality Invariance Learning**: Multi-Positive Contrastive Loss

**Modality-aware global prototypes**: Exploiting modality labels as prior information, each global cluster is split into sub-clusters by modality, with separate modality-specific prototypes constructed for each. Consequently, clusters containing mixed-modality images are represented by two prototypes (one visible, one infrared), precisely capturing intra-cluster modality variation.

**Multi-positive contrastive loss**: For a query image from a mixed cluster, two positive prototypes exist (one same-modality and one cross-modality). A multi-positive InfoNCE loss ensures that features are simultaneously attracted to both positive prototypes:
$$\mathcal{L}_{glb}^v = -\sum_{i=1}^{N_b} \frac{1}{|P(z_i)|} \sum_{p \in P(z_i)} \log \frac{\exp(\mathcal{K}[p]^T f_\theta(x_i^v)/\tau)}{\sum_{j \in S(x_i^v)} \exp(\mathcal{K}[j]^T f_\theta(x_i^v)/\tau)}$$

This design is inspired by the spirit of Invariant Risk Minimization (IRM), achieving modality invariance by reducing response variance across different modalities.

### Loss & Training

- **Total loss** = intra-modality contrastive loss $\mathcal{L}_{intra}$ + global contrastive loss $\mathcal{L}_{global}$
- Two-stage training, 50 epochs per stage
- Adam optimizer, initial learning rate 3.5e-3, decayed by 10× every 20 epochs
- Batch size 128, sampling 8 pseudo-identities × 16 images per batch
- DBSCAN for clustering, eps=0.6 (SYSU) / 0.3 (RegDB)
- Temperature τ=0.05, memory bank momentum μ=0.1
- Stage 2 uses a two-step update: intra-modality loss and global loss are computed on separate batches

## Key Experimental Results

### Main Results

| Method | Type | SYSU All R1 | SYSU All mAP | SYSU Indoor R1 | RegDB V2T R1 | RegDB V2T mAP |
|--------|------|-------------|-------------|-----------------|-------------|-------------|
| CAJ | Supervised | 69.9 | 66.9 | 76.3 | 85.0 | 79.1 |
| DEEN | Supervised | 74.7 | 71.8 | 80.3 | 91.1 | 85.1 |
| PartMix | Supervised | 77.8 | 74.6 | 81.5 | 85.7 | 82.3 |
| PCLHD† | Unsupervised | 65.9 | 61.8 | 70.3 | 89.6 | 83.7 |
| RPNR | Unsupervised | 65.2 | 60.0 | 68.9 | 90.9 | 84.7 |
| **Ours** | **Unsupervised** | **67.1** | **63.1** | **75.0** | **94.3** | **89.1** |

On SYSU-MM01, All Search Rank-1 improves by 1.2% and Indoor by 4.7%; on RegDB, V2T Rank-1 improves by 3.4% and mAP by 4.4%. The proposed unsupervised method is competitive with certain supervised methods (e.g., CAJ, DART).

### Ablation Study

| Config | SC | BMGC | MIRL | All R1 | All mAP | Indoor R1 | Indoor mAP |
|--------|----|------|------|--------|---------|-----------|------------|
| M1: Intra Baseline | | | | 39.5 | 38.9 | 47.1 | 55.5 |
| M2: Global Baseline | | | | 54.9 | 51.5 | 62.9 | 68.9 |
| M3: +BMGC | | ✓ | | 64.9 | 60.0 | 68.0 | 73.5 |
| M4: +SC+BMGC | ✓ | ✓ | | 64.8 | 61.0 | 73.9 | 77.4 |
| M5: +SC+MIRL | ✓ | | ✓ | 61.1 | 58.3 | 72.1 | 76.1 |
| M6: Full | ✓ | ✓ | ✓ | **67.1** | **63.1** | **75.0** | **78.6** |

- **Bias-mitigated global clustering (BMGC)** yields a 10% Rank-1 improvement over naive global clustering (M2→M3)
- **Modality invariance learning (MIRL)** further improves Rank-1 by 2.3% on top of BMGC (M4→M6)
- **Subset clustering (SC)** substantially improves Indoor Search (+5.9%), mitigating over-clustering

### Key Findings

1. **Substantially higher clustering accuracy**: The method significantly outperforms existing methods on the ARI metric, demonstrating more reliable global clustering associations.
2. **Feature visualization**: T-SNE plots show that the proposed method clusters cross-modality images of the same identity into compact groups, whereas the baseline produces multiple modality-specific scattered clusters.
3. **Distance distribution correction**: The corrected Jaccard distance substantially narrows the gap between intra-modality and inter-modality distances (Figure 6), while the improved Jaccard distance from prior work (10833701) shows limited effect.
4. **Manageable computational overhead**: Global clustering Jaccard distance computation takes approximately 68s vs. 47s for naive global clustering—an acceptable increase.

## Highlights & Insights

- **Conceptually simple yet highly effective**: The core innovation is enforcing modality balance in KNN retrieval for Jaccard distance computation—an intuitive idea that yields a substantial ~10% improvement.
- **Framing cross-modality matching as a bias problem**: Treating modality differences as a form of "bias" rather than a "gap" to be bridged is a thought-provoking perspective shift.
- **Multi-positive contrastive learning**: Introduces IRM-inspired thinking into Re-ID, achieving invariance learning through modality-specific prototypes.
- **Complementary to camera-aware methods**: When camera labels are available, the approach can simultaneously address both modality bias and camera bias.
- **Elegant application of subset clustering**: A simple random sampling scheme resolves over-clustering while reducing computation.

## Limitations & Future Work

1. Reliance on DBSCAN's eps parameter requires manual tuning across datasets.
2. Subset clustering performs modestly on small datasets (e.g., RegDB) and is not a universally applicable strategy.
3. Two-stage training may be suboptimal—iterative alternation between intra-modality and cross-modality learning could be more effective.
4. Global clustering introduces additional computational overhead (68s vs. 47s), which may require optimization for larger-scale datasets.
5. The modality invariance learning assumes each global cluster contains exactly two modalities; handling clusters with only one modality lacks flexibility.

## Related Work & Insights

- **CA-Jaccard** (ICCV 2024): The direct inspiration for this work, which proposed camera-aware distance correction.
- **PCLHD / MMM / RPNR**: The most recent USVI-ReID SOTA methods, upon which this work improves.
- **IRM** (Invariant Risk Minimization): The source of the invariance learning concept.
- Insight: In unsupervised cross-modality learning, **correcting the distance metric** may be more effective than complex feature transformations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Modality-aware Jaccard distance is a concise and effective contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Ablation studies, parameter analysis, visualizations, and computational complexity analysis are all comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, figures are intuitive, and mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐ — The method is simple to reproduce and broadly applicable to cross-modality retrieval scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](../../ICCV2025/human_understanding/weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [\[CVPR 2026\] Vision-Language Attribute Disentanglement and Reinforcement for Lifelong Person Re-Identification](../../CVPR2026/human_understanding/vision-language_attribute_disentanglement_and_reinforcement_for_lifelong_person_.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[AAAI 2026\] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)
- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](generating_attribute-aware_human_motions_from_textual_prompt.md)

<!-- RELATED:END -->
