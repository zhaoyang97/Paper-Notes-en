---
title: >-
  [Paper Note] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales
description: >-
  [ECCV 2024][Segmentation][Co-salient Object Detection] This paper proposes SCoSPARC, a two-stage self-supervised co-salient object detection model. By detecting co-salient objects in image groups through patch-level and region-level ViT feature correspondences, it achieves a 13.7% higher F-measure on the CoCA dataset compared to the unsupervised SOTA, even outperforming several supervised methods.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Co-salient Object Detection"
  - "Self-supervised"
  - "Feature Correspondence"
  - "Vision Transformer"
  - "Adaptive Thresholding"
date: 2026-05-08
content_hash: 5eb094fb1aca303a
---

# Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales

**Conference**: ECCV 2024  
**arXiv**: [2403.11107](https://arxiv.org/abs/2403.11107)  
**Code**: [https://github.com/sourachakra/SCoSPARC](https://github.com/sourachakra/SCoSPARC)  
**Area**: Image Segmentation  
**Keywords**: Co-salient Object Detection, Self-supervised, Feature Correspondence, Vision Transformer, Adaptive Thresholding

## TL;DR

This paper proposes SCoSPARC, a two-stage self-supervised co-salient object detection model. By detecting co-salient objects in image groups through patch-level and region-level ViT feature correspondences, it achieves a 13.7% higher F-measure on the CoCA dataset compared to the unsupervised SOTA, even outperforming several supervised methods.

## Background & Motivation

Co-salient object detection (CoSOD) aims to simultaneously detect co-occurring salient objects from a group of related images. Existing methods face two major challenges:

**Supervised methods** (e.g., GCoNet+, DCFM) rely on expensive pixel-wise segmentation annotations, limiting their scalability.

**Limitations of prior unsupervised methods**:
   - DVFDVD only utilizes local patch-level information (clustering ViT patch descriptors) and ignores region-level semantics.
   - ZS-CSD and US-CoSOD depend on heavy pre-trained components such as SAM and STEGO, leading to high computational overhead and slow inference.
   - Handcrafted feature methods (e.g., UCCDGO) exhibit significantly lagging performance.

Key Insight: **Features learned by self-supervised ViTs (such as DINO) contain both rich local semantics (patch descriptors) and global saliency information (self-attention maps)**, which can be leveraged to mine feature correspondences at different scales for unsupervised CoSOD.

## Method

### Overall Architecture

SCoSPARC consists of two stages:
- **Stage 1 (Patch-level)**: Trains a self-supervised network to compute cross-image patch-level feature correspondences, generating cross-attention maps, which are then thresholded using a confidence adaptive threshold to obtain intermediate segmentation results.
- **Stage 2 (Region-level)**: Performs connected component analysis on the intermediate segmentation results to discard regions inconsistent with global foreground features, and finally refines boundaries using denseCRF.

### Key Designs

1. **Patch-level Feature Correspondence (Stage 1)**:

    - Uses a DINO pre-trained ViT-B/8 as the feature encoder to extract patch features $\mathbf{x}^{pat}_n \in \mathbb{R}^{C \times H \times W}$.
    - Enhances features via a residual block: $\mathcal{F}_{res} = \mathcal{F}_{init} + conv^{1\times 1}(\mathcal{F}_{init})$.
    - Computes Key and Query projections to obtain the global feature similarity matrix $S = \frac{1}{\sqrt{d}} K Q^\top \in \mathbb{R}^{NHW \times NHW}$.
    - Takes the row mean for each image to obtain the cross-attention map $S_n \in \mathbb{R}^{H \times W}$, which is then binarized via a modified Sigmoid function: $\mathcal{M}_n = \frac{1}{1 + e^{-k(S_n - s_{th})}}$ (where $k=6.66$, $s_{th}=0.65$).

2. **Dual-loss Self-supervised Training**:

    - **Co-occurrence Loss $\mathcal{L}_{cooc}$**: Based on contrastive learning, it pulls closer the feature embeddings of foreground regions in different images (positive pairs) and pushes apart the feature embeddings of foreground and background within the same image (negative pairs), measured using cosine similarity: $d^+_{nm} = 1 - \cos(f(\mathcal{M}^f_n, \mathbf{x}^{pat}), f(\mathcal{M}^f_m, \mathbf{x}^{pat}))$.
    - **Saliency Loss $\mathcal{L}_{sal}$**: Leverages the DINO self-attention map (averaged across heads) as a saliency prior to maximize the average saliency of the detected regions: $\mathcal{L}_{sal} = 1 - \frac{1}{N}\sum_{n=1}^{N} \mathcal{M}_n \otimes SA_n$.
    - Total Loss: $\mathcal{L}_{total} = \mathcal{L}_{cooc} + \lambda_{sal} \mathcal{L}_{sal}$ (where $\lambda_{sal} = 0.3$).
    - Ingenuity of the design: **Requires no external saliency model**, directly reusing the self-attention maps and patch features of the ViT encoder.

3. **Confidence Adaptive Thresholding (CAT)**:

    - Core finding: High-confidence attention maps require lower thresholds, while low-confidence ones require higher thresholds; a fixed 0.5 threshold is sub-optimal.
    - Computes prediction confidence: $c_M = \frac{1}{n_{conf}} \sum_{p \geq \bar{\mathcal{M}}} \mathcal{M}_p$.
    - Adaptive threshold: $th = th_0 + \alpha_c (b_M - \overline{b_M})$, where $b_M = 1 - c_M$, $th_0 = 0.5$, and $\alpha_c = 1$.

4. **Region-level Feature Correspondence (Stage 2)**:

    - Performs connected component labeling on the intermediate segmentation mask to obtain sub-regions for each image.
    - Computes the average feature embedding $F_G$ of foreground regions across all images (global consensus representation).
    - Computes the cosine similarity between the feature embedding of each sub-region and $F_G$, keeping only regions with similarity $\geq d_f^{th}=0.75$.
    - This step effectively filters out false positives (e.g., shared background regions) caused by local feature matching in Stage 1.

### Loss & Training

- Training Data: COCO9213 (9,213 images, 65 groups) + DUTS-Class (8,250 images, 291 groups), requiring no segmentation annotations.
- Optimizer: Adam, 80 epochs, total training time of approximately 10 hours.
- Inference: All intra-group images (resized to 224×224) are input simultaneously.
- Post-processing: Dense CRF is used to ensure spatial continuity and sharp boundaries.

## Key Experimental Results

### Main Results

Comparison with unsupervised and supervised SOTA on three benchmarks (partial key results):

| Method | Type | CoCA $F_\beta^{max}$↑ | CoCA MAE↓ | Cosal2015 $F_\beta^{max}$↑ | CoSOD3k $F_\beta^{max}$↑ |
|------|------|:---:|:---:|:---:|:---:|
| US-CoSOD | Unsupervised | 0.546 | 0.116 | 0.845 | 0.779 |
| TokenCut | Unsupervised | 0.467 | 0.167 | 0.805 | 0.720 |
| DCFM | Supervised | 0.598 | 0.085 | 0.856 | 0.805 |
| GCoNet+ | Supervised | 0.637 | 0.081 | 0.891 | 0.834 |
| **SCoSPARC** | **Self-supervised** | **0.614** | **0.092** | **0.869** | **0.827** |

### Ablation Study

| ID | Co-oc. | Sal. | CAT | RFC | d-CRF | CoCA $F_\beta^{max}$ | Cosal2015 $F_\beta^{max}$ | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 1 | ✓ | | | | | 0.565 | 0.851 | Co-occurrence loss only |
| 2 | ✓ | ✓ | | | | 0.564 | 0.853 | + Saliency loss |
| 3 | ✓ | ✓ | ✓ | | | 0.567 | 0.840 | + Adaptive threshold |
| 4 | ✓ | ✓ | ✓ | ✓ | | 0.601 | 0.851 | + Region-level refinement |
| 5 | ✓ | ✓ | ✓ | ✓ | ✓ | **0.614** | **0.869** | **Full model** |

### Key Findings

1. **Self-supervised Outperforms Supervised**: SCoSPARC outperforms several supervised methods on CoCA in terms of F-measure, including DCFM (+1.6%), CoRP, and UFO, demonstrating the superiority of self-supervised methods in scenarios with limited annotations.
2. **Region-level Refinement is Crucial**: The performance jump from ID3 to ID4 (CoCA F-measure 0.567 to 0.601) demonstrates that region-level feature correspondence can effectively filter out patch-level false positives.
3. **Lightweight and Efficient**: Without CRF, the inference speed reaches 20.5 FPS (greatly exceeding 0.5 FPS for SegSwap and 0.05 FPS for Group TokenCut); with CRF, it still runs at 4.1 FPS.
4. **Low-data Supervised underperforms Self-supervised**: GCoNet+ with 50% labels underperforms label-free SCoSPARC across all metrics, and still falls behind on most metrics even with 75% labels, showing that supervised methods are prone to overfitting when annotations are insufficient.

## Highlights & Insights

1. **Multi-scale Feature Correspondence Paradigm**: The two-stage design encompassing patch-level (local semantic matching) and region-level (global consistency verification) is a generalizable paradigm for feature correspondence mining.
2. **Full Exploitation of Self-supervised ViT Knowledge**: Instead of introducing additional saliency or segmentation models, it directly leverages DINO's patch descriptors (to construct co-occurrence/contrastive signals) and self-attention maps (to construct saliency signals), keeping the model lightweight.
3. **Confidence Adaptive Thresholding**: A simple yet effective idea that links prediction confidence to the segmentation threshold, outperforming a fixed 0.5 threshold and showing potential for other binary segmentation tasks.
4. **Group TokenCut Baseline**: Extending single-image TokenCut to a group-image baseline is an exemplary experimental design that clearly demonstrates the performance gains of each proposed component.

## Limitations & Future Work

1. **Resolution Limitation**: The patch size of ViT-B/8 is 8×8 with an inference resolution of 224×224, limiting segmentation precision on small objects (performance degrades significantly when patch size increases to 16).
2. **Dense CRF Inference Overhead**: The speed drop from 20 FPS to 4 FPS is primarily due to CRF post-processing; an end-to-end trainable alternative to CRF could be considered.
3. **Simultaneous Processing of All Group Images**: GPU memory bottlenecks may occur for large groups (truncated to min(24, group size) during training).
4. **Binary Foreground/Background Only**: The model cannot distinguish between different co-salient object instances; exploring instance-level CoSOD remains a future research direction.

## Related Work & Insights

- Comparison with DVFDVD indicates that clustering patch descriptors alone is insufficient, and region-level semantic understanding is crucial.
- Comparison with US-CoSOD suggests that relying on heavy pre-trained components (e.g., SAM, STEGO) is inferior to directly mining representations from the ViT's own features.
- Insight: The self-attention maps of DINO ViT serve as a free saliency prior, representing a highly reusable signal for various unsupervised vision tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The two-stage self-supervised CoSOD using multi-scale feature correspondence is a novel solution, with an elegantly designed confidence adaptive threshold.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on three benchmarks using four metrics, featuring comprehensive ablation studies (components, backbones, datasets, inference speeds) and low-data comparisons with supervised methods.
- **Writing Quality**: ⭐⭐⭐⭐ The three-method comparison in Figure 1 intuitively highlights the contributions, and Algorithm 1 clearly outlines the Stage 2 workflow.
- **Value**: ⭐⭐⭐⭐ Lightweight model, no annotations required, and outperforming supervised methods, offering strong application value in annotation-scarce scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](../../CVPR2025/segmentation/visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[CVPR 2026\] Generalizable Co-Salient Object Detection via Mixed Content-Style Modulation](../../CVPR2026/segmentation/generalizable_co-salient_object_detection_via_mixed_content-style_modulation.md)
- [\[ECCV 2024\] FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions](frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)
- [\[ECCV 2024\] Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures](representing_topological_self-similarity_using_fractal_feature_maps_for_accurate.md)
- [\[ECCV 2024\] CoLA: Conditional Dropout and Language-Driven Robust Dual-Modal Salient Object Detection](cola_conditional_dropout_and_language-driven_robust_dual-modal_salient_object_de.md)

</div>

<!-- RELATED:END -->
