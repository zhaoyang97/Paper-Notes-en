---
title: >-
  [Paper Note] Closing the Modality Gap Aligns Group-Wise Semantics
description: >-
  [ICLR2026][Multimodal VLM][modality gap] This paper demonstrates that the modality gap in CLIP is inconsequential for instance-level tasks (retrieval) yet severely harms group-level tasks (clustering). It proposes a nove…
tags:
  - "ICLR2026"
  - "Multimodal VLM"
  - "modality gap"
  - "contrastive learning"
  - "CLIP"
  - "clustering"
  - "multimodal alignment"
date: 2026-05-08
content_hash: 71c98a3a77afbd97
---

# Closing the Modality Gap Aligns Group-Wise Semantics

**Conference**: ICLR2026  
**arXiv**: [2601.18525](https://arxiv.org/abs/2601.18525)  
**Code**: [https://github.com/ispamm/ModGap](https://github.com/ispamm/ModGap)  
**Area**: Multimodal VLM  
**Keywords**: modality gap, contrastive learning, CLIP, clustering, multimodal alignment

## TL;DR
This paper demonstrates that the modality gap in CLIP is inconsequential for instance-level tasks (retrieval) yet severely harms group-level tasks (clustering). It proposes a novel objective comprising an Align True Pairs loss and a Centroid Uniformity loss that reduces the gap to nearly zero in both bimodal and trimodal settings, substantially improving clustering V-Measure by +10–17 points while preserving retrieval performance.

## Background & Motivation

**Background**: CLIP and its variants learn a shared cross-modal embedding space via InfoNCE loss, but embeddings from different modalities tend to form separate clusters—a phenomenon known as the "modality gap." Existing work is divided: some argue that reducing the gap improves retrieval, while others claim the gap correlates positively with downstream performance.

**Limitations of Prior Work**: (a) Prior studies focus exclusively on the gap's effect on retrieval (an instance-level task), yielding contradictory conclusions. (b) All prior methods consider only the bimodal (image + text) setting, without addressing three or more modalities. (c) The consequence that the gap causes the latent space to cluster by modality rather than by semantics has not been systematically analyzed.

**Key Challenge**: InfoNCE optimizes the relative ranking of positive and negative pairs (i.e., which pair is most similar), not absolute distances (i.e., whether positive pairs are truly close). Retrieval succeeds as long as relative rankings are correct—even if the absolute cosine similarity of positive pairs is only 0.34. Clustering, however, depends on absolute distances; the gap inflates within-class scatter by $\|\boldsymbol{\delta}\|^2$.

**Goal**: (a) Theoretically characterize the differential impact of the modality gap on instance-level versus group-level tasks. (b) Propose an effective method for reducing the gap. (c) Extend the analysis to the trimodal setting.

**Key Insight**: The analysis begins from a mathematical decomposition of within-class scatter—the gap vector $\boldsymbol{\delta}$ is orthogonal to the semantic directions and therefore inflates the scatter of all clusters equally, which is irrelevant to retrieval but detrimental to clustering.

**Core Idea**: The modality gap is a harmless artifact for retrieval but a systematic obstacle for clustering. Two loss functions—true pair alignment and centroid uniformity—can simultaneously eliminate the gap and improve semantic clustering.

## Method

### Overall Architecture
Two explicit loss terms are added on top of the standard InfoNCE contrastive objective: $\mathcal{L}_{\text{ATP}}$ (pulling positive pairs together) and $\mathcal{L}_{\text{CU}}$ (pushing centroids apart), combined as $\mathcal{L}_{\text{CL}_{\text{gap}}} = \mathcal{L}_{\text{gap}} + \frac{1}{2}(\mathcal{L}^{(m\to n)} + \mathcal{L}^{(n\to m)})$. The formulation extends directly to $M$ modalities.

### Key Designs

1. **Align True Pairs Loss ($\mathcal{L}_{\text{ATP}}$)**:

    - **Function**: Explicitly minimizes the Euclidean distance between matching (positive) pairs.
    - **Formula**: $\mathcal{L}_{\text{ATP}} = \frac{1}{M-1}\sum_{m\neq a}\frac{1}{N}\sum_i \|\mathbf{z}_i^m - \mathbf{z}_i^a\|_2^2$, where $a$ denotes the anchor modality.
    - **Design Motivation**: InfoNCE optimizes only relative rankings and does not guarantee that positive pairs are absolutely close. $\mathcal{L}_{\text{ATP}}$ directly reduces the gap by pulling positive pairs together. Used in isolation, however, it causes the entire embedding space to collapse to a single point.

2. **Centroid Uniformity Loss ($\mathcal{L}_{\text{CU}}$)**:

    - **Function**: Encourages the cross-modal centroids of distinct semantic samples to be uniformly distributed on the hypersphere, preventing collapse.
    - **Formula**: $\mathcal{L}_{\text{CU}} = \log\frac{1}{N}\sum_i\sum_{j\neq i}\exp(-2\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2^2)$, where $\boldsymbol{\mu}_k = \frac{1}{M}\sum_m \mathbf{z}_k^m$ is the cross-modal centroid of the $k$-th sample.
    - **Design Motivation**: (a) Uniformity is enforced at the centroid level rather than on unimodal embeddings, thereby preserving the cross-modal alignment already learned. (b) The RBF kernel is associated with the uniform distribution on the unit hypersphere, ensuring full spherical coverage.

3. **Theoretical Analysis (Why the Gap Hurts Clustering but Not Retrieval)**:

    - Retrieval requires only $\text{sim}(\mathbf{z}_i^m, \mathbf{z}_i^n) > \max_{j\neq i}\text{sim}(\mathbf{z}_i^m, \mathbf{z}_j^n)$—relative rankings are unaffected by the gap.
    - The within-class scatter for clustering decomposes as: $\mathbb{E}[\|\mathbf{z}_s^m - \boldsymbol{\mu}_s^\delta\|^2] \approx \mathbb{E}[\|\mathbf{z}_s^m - \boldsymbol{\mu}_s^0\|^2] + \|\boldsymbol{\delta}\|^2$—the gap inflates all clusters equally.
    - Key mathematical property: the gap vector $\boldsymbol{\delta}$ is orthogonal to semantic vectors (Zhang et al., 2023), acting as a constant offset that leaves rankings unchanged but increases absolute distances.

### Loss & Training
Total loss: $\mathcal{L}_{\text{CL}_{\text{gap}}} = \mathcal{L}_{\text{ATP}} + \mathcal{L}_{\text{CU}} + \frac{1}{2}(\mathcal{L}^{(m\to n)} + \mathcal{L}^{(n\to m)})$. A notable finding is that as the gap shrinks, gradients from non-matching pairs increase (they become more informative hard negatives) while gradients from matching pairs decrease—optimization naturally shifts toward refining semantic structure.

## Key Experimental Results

### Main Results (Gap vs. Retrieval vs. Clustering)

| Method | Dataset | Gap ↓ | CM R@1 | V-Measure ↑ | kNN ↑ |
|--------|---------|-------|--------|-------------|-------|
| CLIP (LT) | MSCOCO (2-modal) | 0.47 | 74.6 | 12.98 | 26.3 |
| CLIP (FT) | MSCOCO | 0.12 | 73.2 | 12.99 | 31.0 |
| **Ours** | MSCOCO | **0.03** | 70.3 | **23.63** | **36.4** |
| CLIP (LT) | MSR-VTT (3-modal) | 0.29 | 34.2/10.3 | 23.3 | 52.9 |
| **Ours** | MSR-VTT | **0.07** | 32.8/11.8 | **32.1** | **58.0** |
| CLIP (LT) | AV-MNIST (3-modal) | 0.20 | 87.1/84.2 | 77.6 | 87.0 |
| **Ours** | AV-MNIST | **0.09** | 88.7/89.1 | **82.7** | **89.2** |

### Ablation Study (Cosine Similarity of True Pairs)

| Method | MSCOCO Gap | Cos TP ↑ | MM R@1 | CIDEr (captioning) |
|--------|-----------|---------|--------|---------------------|
| CLIP (LT) | 0.47 | 0.34 | 72.5 | 153.2 |
| CLIP (FT) | 0.12 | 0.63 | 73.8 | 155.0 |
| Ours | **0.03** | **0.77** | 76.2 | **158.2** |

### Key Findings
- **Retrieval is largely unaffected by the gap**: CLIP (LT) with gap=0.47 versus Ours with gap=0.03 differ by only 4.3 points in MSCOCO R@1, but by 10.65 points in V-Measure—confirming the theoretical predictions.
- **Clustering is strongly correlated with the gap**: On MSR-VTT, controlling the gap from 0.3 to 0 yields a V-Measure improvement of up to +17.5 points.
- **Cosine similarity of true pairs increases from 0.34 to 0.77**: After standard CLIP training, the cosine similarity of positive pairs is merely 0.34—surprisingly low—demonstrating that InfoNCE guarantees ranking but not proximity.
- **Effectiveness in the trimodal setting**: Gap reduction and clustering improvement are consistently achieved on AV-MNIST and MSR-VTT (audio + video + text) in the three-modality setting.
- **Captioning also benefits**: The better-aligned embedding space enables decoders to generate more accurate captions (CIDEr +5).

## Highlights & Insights
- **Reframing the significance of the modality gap**: The paper shifts the debate from "should the gap be reduced?" to "which tasks are affected by the gap?"—an insight that can resolve contradictory conclusions in the literature.
- **Complementary design of $\mathcal{L}_{\text{ATP}}$ and $\mathcal{L}_{\text{CU}}$**: Pulling matching pairs together (risking collapse) combined with pushing centroids of non-matching pairs apart (preventing collapse) is more elegant than applying alignment and uniformity directly at the embedding level.
- **Gradient-based mechanistic explanation**: Gap reduction → non-matching pairs become more effective hard negatives → gradients concentrate on refining semantic structure. This finding carries important implications for understanding contrastive learning dynamics.
- **Natural extension to $N$ modalities**: The method design inherently supports an arbitrary number of modalities without architectural modifications.

## Limitations & Future Work
- **Slight retrieval degradation**: On MSCOCO, retrieval drops from 74.6 to 70.3. Although theory predicts no effect, a minor practical trade-off exists, possibly due to mild perturbation of rankings by $\mathcal{L}_{\text{ATP}}$.
- **Limited experimental scale**: The largest experiment uses MSCOCO with EVA-CLIP ViT-G; validation at the LAION-5B pretraining scale is absent. Gap behavior in large-scale contrastive learning may differ.
- **Restricted to contrastive learning pipelines**: Only CLIP-like methods are evaluated. Gap behavior in non-contrastive approaches such as SigLIP and BLIP-2 remains unexplored.
- **Future directions**: Integrating $\mathcal{L}_{\text{gap}}$ into VLM pretraining may improve downstream group-level understanding capabilities, such as zero-shot classification and visual commonsense reasoning.

## Related Work & Insights
- **vs. Liang et al. (2022)**: They first identified the modality gap and proposed a post-hoc translation to correct it. This paper offers a more principled training-time solution and provides a theoretical explanation for why the gap should be reduced.
- **vs. Yaras et al. (2025) fixed-temperature scheme**: Fixed temperature partially reduces the gap (to 0.12) but is less effective than the proposed method (0.03), and no theoretical justification is provided for its effectiveness.
- **vs. NotAGap (Fahim et al.)**: NotAGap reduces the gap but decreases Cos TP (0.11 vs. 0.34), indicating that gap reduction alone is insufficient—positive pair alignment must be achieved simultaneously.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The insight that the gap harms clustering but not retrieval is the core contribution; the loss design is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive validation across 4 datasets, 2/3 modalities, and 4 types of downstream tasks, though large-scale pretraining experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative flows seamlessly from theory to experiments to visualization, with clear mathematical derivations.
- **Value**: ⭐⭐⭐⭐ Provides important theoretical and practical tools for understanding and improving multimodal latent spaces.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](../../NeurIPS2025/multimodal_vlm/hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](../../CVPR2026/multimodal_vlm/text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](../../CVPR2026/multimodal_vlm/coat_cbm_concept_wise_attention.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](../../ICCV2025/multimodal_vlm/r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](../../AAAI2026/multimodal_vlm/aligning_the_true_semantics_constrained_decoupling_and_distr.md)

</div>

<!-- RELATED:END -->
