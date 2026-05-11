---
title: >-
  [Paper Note] UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting
description: >-
  [CVPR 2026][open-set defect detection] This paper proposes UniSpector, an open-set industrial defect detection framework that addresses visual prompt embedding collapse through spectral-spatial dual-domain feature fusion…
tags:
  - "CVPR 2026"
  - "open-set defect detection"
  - "frequency-domain features"
  - "contrastive prompt encoding"
  - "visual prompting"
  - "industrial quality inspection"
date: 2026-05-08
content_hash: 08e5e45e594931d4
---

# UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting

**Conference**: CVPR 2026
**arXiv**: [2604.02905](https://arxiv.org/abs/2604.02905)
**Code**: [https://geonuk-kimmm.github.io/UniSpector](https://geonuk-kimmm.github.io/UniSpector)
**Area**: Other
**Keywords**: open-set defect detection, frequency-domain features, contrastive prompt encoding, visual prompting, industrial quality inspection

## TL;DR

This paper proposes UniSpector, an open-set industrial defect detection framework that addresses visual prompt embedding collapse through spectral-spatial dual-domain feature fusion (SSPE) and angular-margin contrastive prompt encoding (CPE). On the newly constructed Inspect Anything benchmark encompassing 360 defect categories, UniSpector surpasses the best baseline by 19.7% in AP50 detection and 15.8% in segmentation.

## Background & Motivation

1. **Background**: Industrial quality inspection requires detecting a wide variety of unseen defect types. Existing open-set detection methods (e.g., GroundingDINO, T-Rex2) are primarily designed for natural images and perform poorly in industrial defect scenarios, where defects typically manifest as subtle texture or color anomalies with feature distributions vastly different from natural objects.
2. **Limitations of Prior Work**: (1) Visual prompting methods suffer from "prompt embedding collapse" in industrial settings—prompt vectors for different defect categories overlap heavily in the embedding space and cannot be distinguished; (2) existing methods rely solely on spatial-domain features, neglecting frequency-domain characteristics (e.g., periodic texture anomalies are more discriminative in the spectral domain).
3. **Key Challenge**: Industrial defects exhibit extremely subtle visual differences (often minor scratches, pits, or color deviations), making it difficult for purely spatial RoI features to capture these distinctions, causing prompt vectors of different categories to collapse into the same region.
4. **Goal**: To design a prompt encoding scheme that extracts discriminative defect features in both the frequency and spatial domains, and to explicitly maximize the embedding distance between different defect categories via contrastive constraints.
5. **Key Insight**: The observation that frequency-domain features of defects (e.g., energy concentration patterns of periodic stripes in the spectrum) are more stable and discriminative than spatial pixels—inspired by classical spectral analysis in signal processing.
6. **Core Idea**: Dual-domain prompt encoding (SSPE) + angular-margin contrastive learning (CPE) + prompt-guided query selection (PQS), forming a unified solution for open-set industrial defect detection.

## Method

### Overall Architecture

Defect RoI from reference image → SSPE extracts and fuses frequency- and spatial-domain features into prompt embeddings → CPE applies angular-margin contrastive loss to separate embeddings of different categories → Category prototypes compute similarity with backbone feature maps → PQS selects the most relevant queries for the detection/segmentation head → Output bounding boxes and masks.

### Key Designs

1. **Spectral-Spatial Prompt Encoder (SSPE)**

    - **Function**: Extracts complementary frequency- and spatial-domain features from RoI patches.
    - **Mechanism**: A 2D DFT is applied to the RoI to obtain the spectrum $F_k(u,v) = \text{DFT}(R_k)$; the radial frequency distribution $h_k(\rho) = \frac{1}{|\Gamma_\rho|}\sum_{(u,v) \in \Gamma_\rho}|F_k(u,v)|$ (direction-invariant) is then extracted and encoded by a radial frequency encoder to yield $z_k^{\text{freq}}$. The spatial branch produces $z_k^{\text{spatial}}$ via masked cross-attention. The two branches are aligned and fused through dual MLPs: $\mathbf{e}_k = f_{\text{align}}(z_k^{\text{spatial}}) + v_{\text{align}}(z_k^{\text{freq}})$.
    - **Design Motivation**: The direction invariance of radial frequency features addresses the problem of random defect orientations; spatial features capture local texture details. The two branches are complementary.

2. **Contrastive Prompt Encoding (CPE)**

    - **Function**: Explicitly constrains the embedding distances between different defect categories to prevent collapse.
    - **Mechanism**: Category prototypes $\mathbf{p}_c$ (intra-class embedding means) are computed, and an angular-margin loss based on cosine similarity is applied: $\mathcal{L}_{\text{CPE}} = -\frac{1}{N}\sum_{k=1}^N \log\frac{\exp(\alpha\cos(\theta_{y_k,k}+m))}{\exp(\alpha\cos(\theta_{y_k,k}+m))+\sum_{c\neq y_k}\exp(\alpha\cos(\theta_{c,k}))}$, where the margin $m$ enforces a minimum angular separation between categories.
    - **Design Motivation**: Standard contrastive losses may yield loose decision boundaries; the angular-margin constraint (inspired by ArcFace from face recognition) ensures compact and discriminative category clusters.

3. **Prompt-guided Query Selection (PQS)**

    - **Function**: Directs the detector to attend only to image regions highly relevant to the prompted defect.
    - **Mechanism**: Cosine similarities between visual tokens $\mathcal{F}$ and category prototype $\mathbf{p}$ are computed as relevance scores; a differentiable top-K selection via Gumbel-Softmax with a Straight-Through Estimator identifies the most relevant queries while preserving gradient flow.
    - **Design Motivation**: Both learnable parameters and heuristic top-K selection are suboptimal—learnable parameters ignore prompt information, while heuristic top-K is non-differentiable and precludes end-to-end optimization.

### Loss & Training

The CPE angular-margin contrastive loss is combined with standard detection/segmentation losses. The scaling factor $\alpha$ and margin $m$ are hyperparameters. The model is built upon the DINOv architecture and trained on the InsA training set.

## Key Experimental Results

### Main Results

| Method | GC10 | MagTile | Real-IAD | MVTec | Mean AP50↑ |
|--------|------|---------|----------|-------|------------|
| GroundingDINO | 9.6 | 26.7 | 0.3 | 1.4 | 5.4 |
| DINOv† | 16.5 | 48.4 | 21.0 | 15.9 | 17.1 |
| T-Rex2† | 32.4 | 49.0 | 25.1 | 24.4 | 32.7 |
| YOLOE† | 10.7 | 43.3 | 17.2 | 25.8 | 17.4 |
| **UniSpector†** | **38.2** | **63.3** | **69.1** | **53.5** | **40.9** |

### Ablation Study

| Component | APb | AP50b | AP75b | APm | AP50m |
|-----------|-----|-------|-------|-----|-------|
| Baseline | 13.6 | 24.0 | 14.5 | 7.7 | 20.0 |
| +SSPE | 27.9 | 43.0 | 31.0 | 17.7 | 34.8 |
| +SSPE+CPE | 43.8 | 65.8 | 48.9 | 26.0 | 53.1 |
| +SSPE+CPE+PQS | **46.3** | **69.1** | **51.9** | **28.9** | **56.7** |

### Key Findings

- SSPE contributes the largest individual gain (AP50b +19.0); CPE adds a further 22.8, and PQS contributes 3.3—their combined effect far exceeds the sum of individual contributions.
- Cross-domain generalization (3CAD=14.1, VISION=15.3, VisA=32.8) is lower than in-domain performance but still substantially outperforms baselines.
- Closed-set performance (90.0 AP50b) approaches that of dedicated closed-set detectors (YOLOv11 88.3, MaskDINO 91.7), demonstrating that the open-set design does not sacrifice accuracy.
- The differentiable top-K selection in PQS outperforms both learnable parameters and heuristic top-K (GC10 AP50b: 38.2 vs. 34.4/35.6).

## Highlights & Insights

- **Elegant introduction of frequency-domain features**: The direction invariance of radial frequency features in the industrial defect setting is a refined design choice—defect orientations are unpredictable, yet frequency characteristics remain stable, representing a strong alignment between problem and methodology.
- **Contribution of the InsA benchmark**: A unified evaluation standard comprising 67k images and 360 defect categories addresses the long-standing absence of large-scale open-set benchmarks in the industrial domain.
- **Transfer of ArcFace to detection**: Migrating angular-margin contrastive learning from face recognition to prompt encoding for defect detection is a natural and effective cross-domain transfer.

## Limitations & Future Work

- Cross-domain performance degrades noticeably (in-domain 40.9 vs. cross-domain ~20), with lighting and texture variations across factories being the primary challenge.
- Prompt quality depends on annotated reference images, which may incur high annotation costs in industrial settings.
- Frequency-domain features are sensitive to defect size—very small defects may produce insufficient spectral signals.
- Reference defect images must be provided at inference time, precluding the flexibility of language-based prompts for describing novel defect types.

## Related Work & Insights

- **vs. T-Rex2**: T-Rex2 relies on purely spatial visual prompts and suffers severe prompt collapse in industrial scenarios. UniSpector fundamentally improves prompt discriminability through frequency-domain features.
- **vs. GroundingDINO**: Text-prompt-based methods perform extremely poorly in industrial settings (AP50=5.4), as defects are difficult to describe precisely in natural language.
- **vs. YOLOE**: Recent real-time YOLO-series detectors also underperform in defect scenarios, demonstrating that detection capability on natural images does not transfer directly to the industrial domain.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of spectral prompt encoding and angular-margin contrastive learning is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation, cross-domain and closed-set comparisons, multiple baselines, and a new benchmark.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-motivated contributions.
- Value: ⭐⭐⭐⭐ Addresses practical industrial inspection needs with benchmark contributions and a deployable method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)
- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](../../ICCV2025/others/kaputt_a_large-scale_dataset_for_visual_defect_detection.md)
- [\[CVPR 2026\] SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)
- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](../../NeurIPS2025/others/zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)

</div>

<!-- RELATED:END -->
