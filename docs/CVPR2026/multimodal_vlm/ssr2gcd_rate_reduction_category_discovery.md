---
title: >-
  [Paper Note] SSR2-GCD: Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery
description: >-
  [CVPR 2026][Multimodal VLM][Generalized Category Discovery] This paper proposes SSR2-GCD, a framework that replaces conventional contrastive losses with a Semi-Supervised Rate Reduction (SSR2) loss to learn uniformly compressed, structured representations. The work further reveals that inter-modal alignment is not only unnecessary but harmful in multi-modal GCD, achieving +3.1% and +6.3% over the prior state of the art on Stanford Cars and Flowers102, respectively.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Generalized Category Discovery
  - Maximal Coding Rate Reduction
  - Intra-modal Alignment
  - CLIP
  - Multi-Modal Representation Learning
date: 2026-05-08
content_hash: d9d16aceb94a7251
---

# SSR2-GCD: Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery

**Conference**: CVPR 2026
**arXiv**: [2602.19910](https://arxiv.org/abs/2602.19910)
**Code**: N/A
**Area**: Self-Supervised Learning / Multi-Modal VLM / Representation Learning
**Keywords**: Generalized Category Discovery, Maximal Coding Rate Reduction, Intra-modal Alignment, CLIP, Multi-Modal Representation Learning

## TL;DR
This paper proposes SSR2-GCD, a framework that replaces conventional contrastive losses with a Semi-Supervised Rate Reduction (SSR2) loss to learn uniformly compressed, structured representations. The work further reveals that inter-modal alignment is not only unnecessary but harmful in multi-modal GCD, achieving +3.1% and +6.3% over the prior state of the art on Stanford Cars and Flowers102, respectively.

## Background & Motivation

Generalized Category Discovery (GCD) requires a model to leverage partially labeled known categories to discover unknown ones. Recent multi-modal methods (CLIP-GCD, TextGCD, GET) have begun exploiting textual modalities to assist visual category discovery, yet their representation learning suffers from two fundamental issues: (1) they over-emphasize inter-modal alignment (CLIP-style) while neglecting intra-modal representation structure; (2) conventional contrastive losses induce **imbalanced compression**—labeled known categories are excessively compressed (effective rank drops sharply), while unlabeled unknown categories are under-compressed, causing blurry cluster boundaries.

## Core Problem

How can one learn **uniformly compressed**, structured representations for both known and unknown categories in multi-modal GCD? And what roles do inter-modal alignment and intra-modal alignment each play in GCD?

## Method

### Overall Architecture

A three-module pipeline: (1) **Retrieval-based Text Aggregation (RTA)** generates semantically rich pseudo-text embeddings for each image; (2) the **SSR2 module** applies semi-supervised rate reduction losses separately within the image and text modalities to learn structured representations; (3) a **dual-branch classifier** processes image/text embeddings independently and aligns pseudo-labels via co-teaching.

### Key Designs

1. **Semi-Supervised Rate Reduction Loss (SSR2)**: Designed on the principle of Maximal Coding Rate Reduction (MCR2). $\mathcal{L}_{SSR^2} = -R(\mathbf{Z}) + R_c^s(\mathbf{Z}, \mathbf{Y}^*) + R_c^u(\mathbf{Z}, \mathbf{Y})$. The first term maximizes the global coding rate of representations (encouraging a more spread-out overall distribution); the second and third terms minimize the within-class coding rates of known categories (using ground-truth labels) and unknown categories (using pseudo-labels), respectively. The key advantage is that MCR2 theory guarantees each class is compressed into an equal-rank low-dimensional subspace, avoiding the over-compression of known categories that occurs with contrastive losses.

2. **Retrieval-based Text Aggregation (RTA)**: Addresses the limitation of CLIP's inability to handle long text prompts as used in TextGCD. Rather than concatenating multiple tags into a long string, each tag and attribute is encoded separately, and a weighted aggregation produces the final text embedding: weight $\sigma_1 = 1-\alpha$ is assigned to the most similar candidate and $\sigma_i = \alpha/(c-1)$ to the remaining candidates ($\alpha=0.5, c=4$). This enables integration of richer candidate information without any token-length constraint.

3. **Finding that Inter-Modal Alignment Is Unnecessary**: Experiments demonstrate that applying $\mathcal{L}_{SSR^2}$ alone (intra-modal alignment only) outperforms the combination with $\mathcal{L}_{CLIP}$ (inter-modal alignment) on 5 out of 6 datasets. The rationale is that pre-trained CLIP already implicitly establishes cross-modal associations through similar-text retrieval; explicit inter-modal alignment introduces noise due to imprecise correspondence between pseudo-text and images, ultimately disrupting the structured intra-modal representations.

### Loss & Training

Two-stage training: a **warm-up phase** (10 epochs) using $\mathcal{L}_{SSR^2}^I + \mathcal{L}_{SSR^2}^T + \mathcal{L}_{cls}^I + \mathcal{L}_{cls}^T$; an **alignment phase** (190 epochs) that adds a co-teaching loss to align dual-branch predictions. No inter-modal alignment loss is used at any stage. Optimization is performed with SGD, learning rate 0.001, batch size 128, on a single RTX 3090.

## Key Experimental Results

| Dataset | Metric | SSR2-GCD | TextGCD | GET | Gain vs. Best |
|--------|------|------|----------|------|------|
| Stanford Cars | All ACC | **89.2** | 86.1 | 78.5 | +3.1 |
| Flowers102 | All ACC | **93.5** | 87.2 | 85.5 | +6.3 |
| CIFAR-100 | All ACC | **86.4** | 85.7 | 82.1 | +0.7 |
| ImageNet-100 | All ACC | **92.1** | 88.0 | 91.7 | +0.4 |
| Oxford Pets | All ACC | **95.7** | 93.7 | 91.1 | +2.0 |
| ImageNet-1K | All ACC | **66.7** | 64.8 | 62.4 | +1.9 |

The ACC gap between Old and New categories is substantially narrowed—e.g., on Stanford Cars, Old 93.1% vs. New 87.3% yields a gap of only 5.8% (TextGCD's gap is 7.9%).

### Ablation Study

- **SSR2 vs. contrastive loss**: SSR2 outperforms conventional supervised + unsupervised contrastive losses on 5/6 datasets, with a margin of 1.7% on Flowers102.
- **Inter-modal alignment is harmful**: $\mathcal{L}_{CLIP} + \mathcal{L}_{SSR^2}$ underperforms $\mathcal{L}_{SSR^2}$ alone on all 6 datasets; similarly, $\mathcal{L}_{CLIP} + \mathcal{L}_{con}$ underperforms $\mathcal{L}_{con}$ alone on 4/6 datasets.
- **Quantifying imbalanced compression**: Effective rank visualizations clearly show that contrastive losses cause a sharp rank drop in Old categories (over-compression), whereas SSR2 maintains uniform rank across Old and New categories.
- **RTA is effective**: Using 4 candidate tags + attributes outperforms TextGCD's top-3 tags + top-2 attributes, with $\alpha=0.5$ being optimal.
- **SSR2 generalizes to unimodal settings**: Replacing contrastive losses with SSR2 in GCD and SimGCD also yields substantial improvements on fine-grained datasets, indicating that imbalanced compression is a general problem.

## Highlights & Insights

- The finding that "inter-modal alignment is unnecessary and even harmful in multi-modal GCD" is highly counter-intuitive yet rigorously validated—it challenges the default assumption of CLIP-style contrastive learning.
- SSR2 addresses the **imbalanced compression** problem from the perspective of **coding rate theory**, yielding an approach that is both theoretically principled and practically effective.
- Two quantitative metrics—edge ratio $R_e$ and effective rank—clearly characterize the impact of different loss functions on representation structure.
- The RTA strategy is concise and effective: encoding multiple retrieved candidates separately and aggregating them with learned weights elegantly circumvents CLIP's token-length limitation.

## Limitations & Future Work

- Increasing the number of candidates incurs additional computational and memory overhead (Table B.11 reports an extra 11% memory usage).
- Image and text modalities are treated symmetrically; adaptive weighting of modality importance is not explored.
- The number of categories $K$ is assumed to be known or accurately estimable, whereas category number estimation is itself a challenging problem in practice.
- Validation is limited to CLIP-B/16; whether larger CLIP models (L/14, H/14) could yield further gains remains unknown.

## Related Work & Insights

- **vs. TextGCD**: TextGCD employs CLIP-style inter-modal alignment with co-teaching but imposes no intra-modal structural constraints. SSR2-GCD outperforms it comprehensively across all datasets.
- **vs. GET**: GET uses a textual inversion network to generate prompts combined with contrastive loss and CICO inter-modal alignment. SSR2-GCD demonstrates that both forms of inter-modal alignment (CLIP loss and CICO) degrade intra-modal learning.
- **vs. SimGCD/SelEx (unimodal)**: SSR2 loss also improves these unimodal methods, confirming that imbalanced compression is a widespread issue.

The finding that "intra-modal alignment matters more than inter-modal alignment" is broadly relevant to any work leveraging CLIP for downstream tasks—blindly adding a CLIP contrastive loss is not always beneficial. The MCR2/rate reduction principle is generalizable to a wider class of semi-supervised and open-world learning problems. Effective rank, as a measure of representation quality, provides a useful tool for monitoring and diagnosing issues in representation learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Extending MCR2 to GCD with a semi-supervised formulation is novel, and the finding that inter-modal alignment is harmful is striking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Eight datasets, extensive loss function comparisons, visualization analyses, unimodal validation, and highly detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Overall clear with thorough theoretical and empirical analysis, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐ — Offers important insights for the multi-modal GCD field; core findings are transferable to broader VLM applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)
- [\[ICLR 2026\] SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery](../../ICLR2026/multimodal_vlm/spectralgcd_spectral_concept_selection_and_cross-modal_representation_learni.md)
- [\[CVPR 2026\] IsoCLIP: Decomposing CLIP Projectors for Efficient Intra-modal Alignment](isoclip_decomposing_clip_projectors_for_efficient_intramodal_alignment.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)

</div>

<!-- RELATED:END -->
