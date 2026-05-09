---
title: >-
  [Paper Note] DSS: Discover, Segment, and Select - A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation
description: >-
  [CVPR 2026][Segmentation][Camouflaged object segmentation] DSS is a three-stage zero-shot camouflaged object segmentation framework: (1) **Discover** candidate regions via DINOv2 feature clustering and part combination (FOD); (2) **Segment** using SAM; (3) **Select** the optimal mask via pairwise MLLM comparison (SMS). Requiring no training, DSS achieves comprehensive improvements over prior zero-shot methods on four COD benchmarks, with particularly pronounced advantages in multi-instance scenarios.
tags:
  - CVPR 2026
  - Segmentation
  - Camouflaged object segmentation
  - zero-shot
  - DINOv2 clustering
  - MLLM selection
  - SAM
  - training-free
date: 2026-05-08
content_hash: b0f0a671a825354b
---

# DSS: Discover, Segment, and Select - A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation

**Conference**: CVPR 2026
**arXiv**: [2602.19944](https://arxiv.org/abs/2602.19944)
**Code**: None
**Area**: Semantic Segmentation / Camouflaged Object Detection / Zero-Shot
**Keywords**: Camouflaged object segmentation, zero-shot, DINOv2 clustering, MLLM selection, SAM, training-free

## TL;DR
DSS is a three-stage zero-shot camouflaged object segmentation framework: (1) **Discover** candidate regions via DINOv2 feature clustering and part combination (FOD); (2) **Segment** using SAM; (3) **Select** the optimal mask via pairwise MLLM comparison (SMS). Requiring no training, DSS achieves comprehensive improvements over prior zero-shot methods on four COD benchmarks, with particularly pronounced advantages in multi-instance scenarios.

## Background & Motivation

The dominant paradigm for zero-shot camouflaged object segmentation (COS) is a two-stage "discover-then-segment" pipeline: an MLLM localizes the camouflaged object to generate bbox/point prompts, which are then fed into SAM for segmentation. However, MLLMs frequently fail in camouflaged scenarios in three ways: (1) **inaccurate localization** (a semantic gap between high-level language priors and low-contrast foreground); (2) **missed detections** (only the most salient instance is found in multi-instance scenes); (3) **false detections** (background regions misidentified as camouflaged objects). The root cause is **sole reliance on the language priors of MLLMs for discovery**, neglecting the intrinsic visual discriminative features of the image.

## Core Problem

How to achieve robust camouflaged object discovery and high-quality mask selection in a zero-shot, training-free setting by jointly leveraging visual feature priors and language reasoning capabilities?

## Method

### Overall Architecture

Three stages: (1) **Discover (FOD)**: extract patch-level features with DINOv2 → Leiden clustering → Part Combination (PC) for iterative cluster refinement → Similarity-Based bbox Generation (SBG) → multiple heterogeneous bbox prompts; QWen2.5-VL is also applied via VLOS to produce supplementary MLLM bboxes. (2) **Segment**: all bboxes are fed into SAM2 (ViT-L) to generate a candidate mask set. (3) **Select (SMS)**: heuristic scoring (relevance + boundary contact rate) filters top-K candidates → pairwise ascending MLLM comparison → final mask selection.

### Key Designs

1. **Part Combination (PC)**: Clustering may over-segment a single camouflaged object into multiple parts. PC resolves this via iterative soft assignment: the foreground probability of each patch is updated as $y_i^{(t)} = \sigma(\|x_i - \mu_b\|^2 - \|x_i - \mu_f\|^2)$, based on relative distances to foreground/background centroids, until a feature consistency energy (intra-class compactness + inter-class separation) converges, typically within 5 iterations.

2. **Similarity-Based bbox Generation (SBG)**: Rather than directly extracting bboxes from binary masks (which tend to be incomplete), SBG computes cosine similarity between the foreground centroid $\mu_f$ of each cluster and all patches to produce a continuous similarity map $\text{sim}_c$, from which bboxes are extracted via thresholding and connected component analysis. Redundancy is reduced by merging highly correlated similarity maps using Pearson correlation ($\tau=0.95$). SBG substantially improves bbox completeness over direct mask-based extraction in multi-instance scenarios.

3. **Semantics-driven Mask Selection (SMS)**: Given multiple candidate masks, a heuristic score $s_i = \text{corr}(m_i, \text{sim}_i) + (1 - \text{BC}(m_i))$ (mask–similarity-map correlation + low boundary contact rate) is first used to select top-K=5 candidates. The key design is **pairwise ascending comparison**: starting from the lowest-scored mask, pairs are compared and the MLLM judges "which looks more like a camouflaged object," with the winner advancing to the next round. Ablations show ascending order (74.4%) greatly outperforms descending (50.8%) and single-pass full selection (OOM).

### Loss & Training

The method is entirely training-free with no optimization required. The pipeline operates purely at inference time. QWen2.5-VL-Instruct serves as the MLLM, SAM2 (ViT-L) performs segmentation, and DINOv2 extracts features that are reduced to 16 dimensions via PCA before Leiden clustering (which automatically determines the number of clusters without specifying $K$). The full pipeline runs on a single RTX 3090 24 GB GPU.

## Key Experimental Results

| Method | Setting | CAMO $S_\alpha$↑ | NC4K $S_\alpha$↑ | COD10K $S_\alpha$↑ | CHAMELEON $S_\alpha$↑ |
|--------|---------|-----------------|-----------------|-------------------|----------------------|
| ProMaC (NeurIPS24) | ZS | .725 | .777 | .716 | .790 |
| RDVP-MSD (MM25) | ZS | .785 | .795 | .775 | .814 |
| IAPF | ZS | .768 | .828 | .799 | — |
| QWen+SAM2 | ZS | .741 | .846 | .827 | .785 |
| **DSS** | **ZS** | **.766** | **.870** | **.849** | **.848** |

The most pronounced advantage lies in **multi-instance scenarios**: with 3+ instances, DSS exhibits the smallest performance degradation (on NC4K, 3+ instances vs. single instance drops ~5%, whereas other methods drop 15–25%).

Efficiency: DSS processes 42 s/image (FOD 7.7 s + SAM 3.6 s + SMS 30.6 s) vs. ProMaC at 130.5 s/image; GPU memory is 17.9 GB vs. ProMaC's 32.9 GB.

### Ablation Study

- **FOD and VLOS are complementary**: clustering alone (Leiden/PC) achieves 0.82 on COD10K, VLOS alone 0.85, and their combination 0.89—visual and language-based discovery are mutually complementary.
- **SBG advantage in multi-instance settings**: in 3+ instance scenarios, similarity-map-based bboxes consistently outperform direct mask-based bboxes on both NC4K and COD10K.
- **SMS is effective but has room for improvement**: DSS with SMS scores 0.87 vs. Ideal Seg at 0.89 vs. VLOS baseline at 0.85, indicating the selection step is effective but still below the oracle upper bound.
- **Pairwise ascending comparison is optimal**: ascending pairwise (74.4%) >> descending (50.8%) >> random (54.0%) >> single-pass full selection (OOM).
- **Leiden outperforms K-means**: automatic cluster number determination avoids the need to specify $K$ as required by K-means.

## Highlights & Insights

- The three-stage "Discover–Segment–Select" paradigm is more robust than the two-stage "Discover–Segment" approach—the selection stage provides strong error tolerance and leverages MLLMs for reasoning rather than localization (MLLMs are better at judging than discovering).
- The visual feature clustering in FOD and the language-based localization of MLLMs are **orthogonally complementary**—visual priors are more reliable than language priors in low-contrast scenarios.
- The pairwise ascending MLLM selection strategy is a practical and effective trick—decomposing N-way selection into multiple binary comparisons yields greater stability.
- The combination of Leiden clustering and PCA dimensionality reduction is elegant—it automatically determines the number of clusters with high computational efficiency, with the entire FOD stage taking only 7.7 s.
- The advantage in multi-instance scenarios is significant—precisely the most critical weakness of existing methods.

## Limitations & Future Work

- The SMS module accounts for 72.9% of runtime (~30.6 s/image) due to multiple MLLM inference calls; the selection stage is the primary bottleneck.
- Ablations with Ideal Seg reveal a remaining gap in mask quality (COD10K: 0.871 vs. oracle 0.892); improved selection strategies or end-to-end optimization may help.
- Tiny camouflaged object detection is not specifically addressed—the conclusion notes that multi-scale feature aggregation is a direction for future work.
- The pipeline depends on the representation quality of DINOv2 and may degrade in domains where DINOv2 features are less discriminative.
- MLLM judgments in pairwise comparisons are not fully reliable, though this already represents the best available strategy.

## Related Work & Insights

- **vs. GenSAM (AAAI24)**: GenSAM uses BLIP2-generated categories and CLIP attention as prompts; DSS uses DINOv2 feature clustering for discovery. CAMO: .766 vs. .659.
- **vs. ProMaC (NeurIPS24)**: ProMaC leverages MLLM hallucination priors to reduce inaccurate text prompts but still relies on MLLMs for discovery. DSS shifts the MLLM's role from "discovery" to "selection."
- **vs. RDVP-MSD (MM25)**: RDVP employs step-wise decomposition to mitigate semantic ambiguity but remains limited in multi-instance scenes. DSS's FOD module naturally handles multiple instances through clustering.
- **vs. IAPF**: IAPF uses Grounding DINO for instance-level bboxes and single-foreground multi-background point prompts; DSS performs discovery via unsupervised clustering without relying on a supervised localizer.

The "Discover–Segment–Select" paradigm is generalizable to other visual understanding tasks, such as candidate region generation and filtering in open-world detection. A key insight is the role shift of MLLMs from **discoverers to evaluators**—assigning models to tasks they perform best.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-stage paradigm and visual clustering-based discovery in FOD offer a fresh perspective, though individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks, comprehensive comparisons against fully supervised/unsupervised/zero-shot methods, multi-instance analysis, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear, though the abundance of symbols and modules slightly raises the reading barrier.
- Value: ⭐⭐⭐⭐ New state of the art for zero-shot camouflaged segmentation, with the most prominent gains in multi-instance scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](../../AAAI2026/segmentation/rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)

<!-- RELATED:END -->
