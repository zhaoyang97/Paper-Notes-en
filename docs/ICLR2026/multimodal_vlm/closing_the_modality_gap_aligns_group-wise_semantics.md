---
title: >-
  [Paper Note] Closing the Modality Gap Aligns Group-Wise Semantics
description: >-
  [ICLR2026][Multimodal VLM][modality gap] Demonstrates that the modality gap in CLIP is irrelevant for instance-level tasks (retrieval) but severely harms group-level tasks (clustering). Proposes a new objective function consisting of Align True Pairs loss + Centroid Uniformity loss, reducing the gap nearly to zero in bi-modal and tri-modal settings, significantly improving clustering V-Measure (+10-17 points) while maintaining retrieval performance.
tags:
  - "ICLR2026"
  - "Multimodal VLM"
  - "modality gap"
  - "contrastive learning"
  - "CLIP"
  - "clustering"
  - "multimodal alignment"
date: 2026-05-08
content_hash: ca35dcd2801dede3
---

# Closing the Modality Gap Aligns Group-Wise Semantics

**Conference**: ICLR2026  
**arXiv**: [2601.18525](https://arxiv.org/abs/2601.18525)  
**Code**: [https://github.com/ispamm/ModGap](https://github.com/ispamm/ModGap)  
**Area**: Multimodal VLM  
**Keywords**: modality gap, contrastive learning, CLIP, clustering, multimodal alignment

## TL;DR
Demonstrates that the modality gap in CLIP is irrelevant for instance-level tasks (retrieval) but severely harms group-level tasks (clustering). Proposes a new objective function consisting of Align True Pairs loss + Centroid Uniformity loss, reducing the gap nearly to zero in bi-modal and tri-modal settings, significantly improving clustering V-Measure (+10-17 points) while maintaining retrieval performance.

## Background & Motivation

**Background**: CLIP and its variants learn a cross-modal shared space via InfoNCE loss; however, embeddings of different modalities form their own clusters—known as the "modality gap". Existing perspectives on this issue are divided: some argue that narrowing the gap improves retrieval, while others suggest the gap is positively correlated with downstream performance.

**Limitations of Prior Work**: (a) Existing research focuses only on the impact of the gap on retrieval (instance-level tasks), leading to contradictory conclusions; (b) studies are limited to bi-modal (image+text) settings, excluding tri-modal or higher; (c) the existence of the gap causes the latent space to "cluster by modality" rather than "cluster by semantics," a consequence that has not been systematically analyzed.

**Key Challenge**: InfoNCE optimizes the relative ranking of positive and negative pairs (whether they are most similar) rather than absolute distance (whether they are truly close). As long as the relative ranking is correct, retrieval succeeds—even if the absolute cosine similarity of a positive pair is only 0.34. However, clustering relies on absolute distance, and the gap causes within-class scatter to expand by $\|\boldsymbol{\delta}\|^2$.

**Goal** (a) Theoretically clarify the different impacts of the gap on instance-level vs. group-level tasks; (b) propose an effective method to narrow the gap; (c) extend the approach to tri-modal settings.

**Key Insight**: Starting from the mathematical decomposition of within-class scatter—the gap vector $\boldsymbol{\delta}$ is orthogonal to semantics, thus uniformly expanding the scatter of all clusters, which is irrelevant for retrieval but fatal for clustering.

**Core Idea**: The modality gap is a harmless artifact for retrieval but a systematic obstacle for clustering. A combination of true pair alignment + centroid uniformity loss functions can simultaneously eliminate the gap and improve semantic clustering.

## Method

### Overall Architecture
This paper addresses the modality gap in CLIP-like models, where modalities cluster separately and cross-modal semantics are misaligned, without compromising retrieval performance. The logic is two-fold: first, it theoretically proves that the gap is a constant offset orthogonal to semantics. It is invisible to instance-level tasks (retrieval) that only consider relative ranking but uniformly expands the within-class scatter of every semantic cluster, degrading group-level tasks (clustering) that rely on absolute distance. Second, it introduces a lightweight training objective to eliminate the gap. While keeping standard bidirectional InfoNCE contrastive terms, it adds two explicit geometric constraints: one pulling true paired samples together ($\mathcal{L}_{\text{ATP}}$), and another pushing centroids of different semantic samples apart on the hypersphere ($\mathcal{L}_{\text{CU}}$). Combined as $\mathcal{L}_{\text{gap}}=\mathcal{L}_{\text{ATP}}+\mathcal{L}_{\text{CU}}$, the final training objective is:

$$\mathcal{L}_{\text{CL}_{\text{gap}}} = \mathcal{L}_{\text{gap}} + \tfrac{1}{2}\big(\mathcal{L}^{(m\to n)} + \mathcal{L}^{(n\to m)}\big).$$

The design is modality-agnostic; transitioning from bi-modal to $M$ modalities (e.g., audio+video+text) simply requires extending the summation across all modalities without architectural changes.

### Key Designs

**1. Align True Pairs Loss ($\mathcal{L}_{\text{ATP}}$): Pulling "correctly ranked but distant" positive pairs together**

The gap originates from InfoNCE's focus on relative ranking—as long as a positive pair is more similar than negative pairs, retrieval is successful. $\mathcal{L}_{\text{ATP}}$ addresses this by minimizing the Euclidean distance between embeddings of the same sample across modalities, using an anchor modality $a$ as a reference:

$$\mathcal{L}_{\text{ATP}} = \frac{1}{M-1}\sum_{m\neq a}\frac{1}{N}\sum_i \big\|\mathbf{z}_i^m - \mathbf{z}_i^a\big\|_2^2.$$

This term reduces the absolute distance of positive pairs, causing the gap to shrink. However, used alone, it risks collapse where all points converge to a single point, destroying semantic structure.

**2. Centroid Uniformity Loss ($\mathcal{L}_{\text{CU}}$): Stretching the space at the centroid level to prevent collapse**

To counter potential collapse, a uniformity constraint is applied to cross-modal centroids $\boldsymbol{\mu}_k = \frac{1}{M}\sum_m \mathbf{z}_k^m$, ensuring centroids of different semantic samples are dispersed on the unit hypersphere:

$$\mathcal{L}_{\text{CU}} = \log\frac{1}{N}\sum_i\sum_{j\neq i}\exp\big(-2\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2^2\big).$$

Targeting centroids instead of mono-modal embeddings preserves learned cross-modal alignment—pushing "different samples" apart without separating different modalities of the same sample. The RBF kernel ensures the centroids cover the hypersphere uniformly, providing the necessary repulsion to balance $\mathcal{L}_{\text{ATP}}$.

**3. Theoretical Analysis: Why the same gap is harmless for retrieval but fatal for clustering**

Retrieval success depends only on relative ranking—if $\text{sim}(\mathbf{z}_i^m, \mathbf{z}_i^n) > \max_{j\neq i}\text{sim}(\mathbf{z}_i^m, \mathbf{z}_j^n)$, the result is correct. Since the gap acts as a uniform offset for all samples, it does not change relative order. Clustering, however, depends on absolute distance. Decomposing within-class scatter yields:

$$\mathbb{E}\big[\|\mathbf{z}_s^m - \boldsymbol{\mu}_s^\delta\|^2\big] \approx \mathbb{E}\big[\|\mathbf{z}_s^m - \boldsymbol{\mu}_s^0\|^2\big] + \|\boldsymbol{\delta}\|^2,$$

where the gap vector $\boldsymbol{\delta}$ adds $\|\boldsymbol{\delta}\|^2$ to the scatter of every cluster. Since $\boldsymbol{\delta}$ is orthogonal to semantic directions (Zhang et al., 2023), it acts as a constant offset that inflates absolute distances without affecting rankings.

### Loss & Training
The complete objective combines the three components: $\mathcal{L}_{\text{CL}_{\text{gap}}} = \mathcal{L}_{\text{ATP}} + \mathcal{L}_{\text{CU}} + \frac{1}{2}(\mathcal{L}^{(m\to n)} + \mathcal{L}^{(n\to m)})$. During training, as the gap closes, gradients from non-matching pairs become more informative "hard negatives," while gradients for matched pairs decrease, naturally shifting optimization focus from modality alignment to semantic refinement.

## Key Experimental Results

### Main Results (Gap Value vs. Retrieval vs. Clustering)

| Method | Dataset | Gap ↓ | CM R@1 | V-Measure ↑ | kNN ↑ |
|------|--------|-------|--------|-------------|-------|
| CLIP (LT) | MSCOCO (2-modal) | 0.47 | 74.6 | 12.98 | 26.3 |
| CLIP (FT) | MSCOCO | 0.12 | 73.2 | 12.99 | 31.0 |
| **Ours** | MSCOCO | **0.03** | 70.3 | **23.63** | **36.4** |
| CLIP (LT) | MSR-VTT (3-modal) | 0.29 | 34.2/10.3 | 23.3 | 52.9 |
| **Ours** | MSR-VTT | **0.07** | 32.8/11.8 | **32.1** | **58.0** |
| CLIP (LT) | AV-MNIST (3-modal) | 0.20 | 87.1/84.2 | 77.6 | 87.0 |
| **Ours** | AV-MNIST | **0.09** | 88.7/89.1 | **82.7** | **89.2** |

### Ablation Study (Cos True Pairs Gain)

| Method | MSCOCO Gap | Cos TP ↑ | MM R@1 | CIDEr (captioning) |
|------|-----------|---------|--------|---------------------|
| CLIP (LT) | 0.47 | 0.34 | 72.5 | 153.2 |
| CLIP (FT) | 0.12 | 0.63 | 73.8 | 155.0 |
| Ours | **0.03** | **0.77** | 76.2 | **158.2** |

### Key Findings
- **Retrieval is nearly independent of the gap**: MSCOCO R@1 drops only 4.3 points despite the gap shrinking from 0.47 to 0.03, while V-Measure improves by 10.65 points—confirming theoretical predictions.
- **Clustering is strongly correlated with the gap**: On MSR-VTT, reducing the gap from 0.3 to 0 led to a +17.5 increase in V-Measure.
- **Positive pair cosine similarity increased from 0.34 to 0.77**: Standard CLIP positive pairs are surprisingly distant (0.34), showing InfoNCE guarantees ranking but not proximity.
- **Effective for Tri-modal settings**: Successfully reduced the gap and improved clustering in AV-MNIST and MSR-VTT (audio+video+text).
- **Benefits Captioning**: The improved alignment space allows the decoder to generate more accurate captions (CIDEr +5).

## Highlights & Insights
- **Redefining the significance of the modality gap**: Shifts the debate from "whether to reduce the gap" to a precise analysis of "what tasks the gap affects"—an insight that resolves contradictory conclusions in the field.
- **Complementary Design of $\mathcal{L}_{\text{ATP}}$ + $\mathcal{L}_{\text{CU}}$**: Pulling matching pairs (risk of collapse) while pushing non-matching pairs at the centroid level (preventing collapse) is more elegant than performing alignment and uniformity on raw embeddings.
- **Mechanism explanation via gradients**: Gap reduction turns non-matching pairs into more effective hard negatives, focusing gradients on semantic structure refinement.
- **Direct Extension to N Modalities**: The method naturally supports any number of modalities without architectural changes.

## Limitations & Future Work
- **Slight drop in retrieval**: MSCOCO retrieval dropped from 74.6 to 70.3. Though theoretically irrelevant, a small trade-off exists, likely because $\mathcal{L}_{\text{ATP}}$ slightly perturbs rankings.
- **Limited experimental scale**: The largest experiment used MSCOCO (EVA-CLIP ViT-G); behavior at LAION-5B pre-training scale remains unverified.
- **Focused on contrastive pipelines**: Gap behavior in non-contrastive methods like SigLIP or BLIP-2 was not explored.
- **Future Directions**: Integrating $\mathcal{L}_{\text{gap}}$ into VLM pre-training might improve group-level downstream understanding (e.g., zero-shot classification, visual commonsense reasoning).

## Related Work & Insights
- **vs. Liang et al. (2022)**: Provided a training-time solution and theoretical justification "why to reduce the gap" compared to their post-hoc translation fix.
- **vs. Yaras et al. (2025) fixed temperature**: Fixed temperature partially reduces the gap (0.12) but is less thorough than Ours (0.03) and lacks theoretical grounding.
- **vs. NotAGap (Fahim et al.)**: NotAGap reduces the gap but CosTP decreases (0.11 vs 0.34), proving that gap reduction alone is insufficient without strong positive pair alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight that "gap harms clustering but not retrieval" is a core contribution; the loss design is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across 4 datasets, 2/3 modalities, and 4 downstream tasks, though lacking large-scale pre-training.
- Writing Quality: ⭐⭐⭐⭐⭐ Seamless narrative from theory to experiments to visualization; clear mathematical derivation.
- Value: ⭐⭐⭐⭐ Provides significant theoretical and practical tools for understanding and improving multimodal latent spaces.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Is the Modality Gap a Bug or a Feature? A Robustness Perspective](../../CVPR2026/multimodal_vlm/is_the_modality_gap_a_bug_or_a_feature_a_robustness_perspective.md)
- [\[NeurIPS 2025\] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](../../NeurIPS2025/multimodal_vlm/hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)
- [\[CVPR 2026\] Bridging the Modality Gap in Compositional Zero-Shot Learning via Sparse Alignment and Unimodal Memory Bank](../../CVPR2026/multimodal_vlm/bridging_the_modality_gap_in_compositional_zero-shot_learning_via_sparse_alignme.md)
- [\[ICLR 2026\] Turning Internal Gap into Self-Improvement: Promoting the Generation-Understanding Unification in MLLMs](turning_internal_gap_into_self-improvement_promoting_the_generation-understandin.md)
- [\[CVPR 2026\] Text-Printed Image: Bridging the Image-Text Modality Gap by "Printing" Text into Images](../../CVPR2026/multimodal_vlm/text-printed_image_bridging_the_image-text_modality_gap_for_text-centric_trainin.md)

</div>

<!-- RELATED:END -->
