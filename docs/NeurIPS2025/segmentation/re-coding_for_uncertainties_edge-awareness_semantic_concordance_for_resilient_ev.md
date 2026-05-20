---
title: >-
  [Paper Note] Re-coding for Uncertainties: Edge-awareness Semantic Concordance for Resilient Event-RGB Segmentation
description: >-
  [NeurIPS 2025][Segmentation][Event-RGB Fusion] This paper proposes the Edge-awareness Semantic Concordance (ESC) framework, which leverages semantic edges as an intermediate bridge between heterogeneous Event and RGB mod…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Event-RGB Fusion"
  - "Semantic Edge"
  - "Discrete Latent Space"
  - "Uncertainty Optimization"
  - "Extreme Conditions"
date: 2026-05-08
content_hash: 45297b774b470a83
---

# Re-coding for Uncertainties: Edge-awareness Semantic Concordance for Resilient Event-RGB Segmentation

**Conference**: NeurIPS 2025
**arXiv**: [2511.08269](https://arxiv.org/abs/2511.08269)  
**Code**: [https://github.com/iCVTEAM/ESC](https://github.com/iCVTEAM/ESC)  
**Area**: Semantic Segmentation / Multimodal Fusion
**Keywords**: Event-RGB Fusion, Semantic Edge, Discrete Latent Space, Uncertainty Optimization, Extreme Conditions

## TL;DR

This paper proposes the Edge-awareness Semantic Concordance (ESC) framework, which leverages semantic edges as an intermediate bridge between heterogeneous Event and RGB modalities. Through discrete latent space modeling via an edge dictionary, ESC achieves cross-modal feature alignment and uncertainty optimization, surpassing the state of the art by 2.55% mIoU under extreme conditions.

## Background & Motivation

**Background**: Under extreme conditions (low illumination, severe motion blur), RGB information degrades significantly. Event cameras provide complementary information via high dynamic range and high temporal resolution.

**Limitations of Prior Work**: Event and RGB modalities are fundamentally heterogeneous—feature-level mismatches and degraded optimization are critical issues. Existing multimodal methods rely on naive fusion strategies that cannot handle modality imbalance or modality failure.

**Key Challenge**: Establishing a unified representation space between two fundamentally different modalities remains an open challenge.

**Key Insight**: Event data is empirically observed to concentrate at semantic edge regions (statistically verified), while RGB gradients also reveal edge cues—semantic edges thus serve as a natural shared intermediate representation for both modalities.

**Core Idea**: A VQ-VAE-based edge dictionary (discrete latent space) is constructed to enable bidirectional feature transformation and distribution alignment via re-coding.

## Method

### Overall Architecture

Pre-build edge dictionary → Edge-awareness Latent Re-coding (ELR, bidirectional encoding-decoding) → Re-coded Consolidation (RC, edge information integration) → Uncertainty Optimization (UO, joint uncertainty-aware optimization).

### Key Designs

1. **Edge Dictionary**

    - **Function**: Learns discrete latent representations from semantic edges based on a VQ-VAE architecture.
    - **Mechanism**: Boundary maps $\mathbf{B}$ are extracted from semantic mask ground truth, encoded by a tokenizer, and matched to the nearest entry in a $K$-item dictionary, then reconstructed by a detokenizer.
    - **Design Motivation**: The dictionary captures fundamental elements of semantic edges, serving as a shared intermediate semantic space for both Event and RGB modalities.

2. **Edge-awareness Latent Re-coding (ELR)**

    - **Function**: Bidirectional transformation—edge embeddings to distributions, and modal distributions to edge features.
    - **Mechanism**: Edge encoders map Image/Event features to categorical probability distributions $p(\mathcal{K}|\mathcal{I})$ and $p(\mathcal{K}|\mathcal{E})$, which are aligned to the GT edge distribution $q(\mathcal{K}|\mathbf{B})$ via cross-entropy; re-coded features are retrieved via argmax and dictionary lookup.
    - **Design Motivation**: Cross-entropy supervision bridges the modality gap, aligning heterogeneous features into a unified semantic space.

3. **Re-coded Consolidation (RC)**

    - **Function**: Integrates image context using re-coded edge features.
    - **Mechanism**: Multi-head attention where Query = image features and Key/Value = [image + noise embeddings, image re-coded features, Event re-coded features].
    - **Design Motivation**: Image features capture global context but lack edge understanding; re-coded features supply complementary edge information.

4. **Uncertainty Optimization (UO)**

    - **Function**: Derives uncertainty indicators from modal distributions for joint optimization.
    - **Mechanism**: The entropy of the modal classification probability distribution serves as an uncertainty indicator to down-weight unreliable modalities during fusion.
    - **Design Motivation**: Under extreme conditions, one modality may fail entirely, necessitating dynamic weighting.

### Loss & Training

$L_{total} = L_{seg} + L_{edge} + L_{dict}$, where $L_{edge} = -\sum q(\mathcal{K}|\mathbf{B})\log(p(\mathcal{K}|\mathcal{I})p(\mathcal{K}|\mathcal{E}))$.

## Key Experimental Results

### Main Results

| Method | DERS-XS mIoU | DSEC-Xtrm mIoU | DERS-XR mIoU |
|--------|-------------|----------------|-------------|
| CMX | 51.23 | 42.15 | 38.67 |
| Any2Seg | 52.11 | 43.28 | 39.45 |
| SegFormer (RGB only) | 48.56 | 40.12 | 36.89 |
| **ESC (Ours)** | **53.78** | **44.83** | **41.22** |

### Ablation Study

| Configuration | DERS-XS mIoU | Note |
|---------------|-------------|------|
| Baseline (w/o ESC) | 48.56 | RGB only |
| + ELR | 51.23 | Edge alignment |
| + RC | 52.45 | Edge consolidation |
| + UO | 53.12 | Uncertainty optimization |
| **Full ESC** | **53.78** | **Complete framework** |

### Key Findings

- Surpasses the previous SOTA by 2.55% mIoU on the newly constructed DERS-XS dataset.
- ESC demonstrates significant robustness under spatial occlusion evaluation—the first work to assess model resilience to occlusion without fine-tuning.
- Event data is statistically confirmed to concentrate at semantic edge regions (edge pixels account for only 5–10% of the image, yet the proportion of Event responses at edges consistently exceeds the corresponding area ratio).

## Highlights & Insights

- **Edge as Bridge**: The paper identifies semantic edges as a natural common ground between Event and RGB modalities and validates this hypothesis statistically. This idea is transferable to other heterogeneous modality fusion settings (e.g., LiDAR-Camera).
- **Discrete Latent Space Modeling**: The VQ-VAE dictionary mechanism constructs a shared space that is more stable than continuous alignment approaches.
- **Dataset Contribution**: Three extreme-condition datasets are introduced—DERS-XS (synthetic), DERS-XR (real-world), and DSEC-Xtrm—filling a gap in evaluation benchmarks.

## Limitations & Future Work

- The edge dictionary requires semantic mask ground truth for training, which is unavailable during deployment.
- Performance is sensitive to the choice of dictionary size $K$.
- Evaluation is limited to driving scenarios; generalization to other extreme-condition settings (e.g., indoor environments) requires further validation.

## Related Work & Insights

- **vs. CMX/Any2Seg**: These methods adopt general cross-modal fusion without exploiting the edge-concentrated nature of Event data; ESC specifically leverages edge priors.
- **vs. ESEG**: ESEG utilizes edge semantics within single-modality Event perception; ESC extends this to dual-modality Event-RGB fusion.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of edge dictionary and re-coding constitutes a clever and well-motivated innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets and occlusion robustness evaluation.
- Writing Quality: ⭐⭐⭐⭐ Methodologically rigorous with sufficient statistical validation.
- Value: ⭐⭐⭐⭐⭐ Perception under extreme conditions is a core requirement for autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SRSR: Enhancing Semantic Accuracy in Real-World Image Super-Resolution with Spatially Re-Focused Text-Conditioning](srsr_enhancing_semantic_accuracy_in_real-world_image_super-resolution_with_spati.md)
- [\[ICCV 2025\] Online Generic Event Boundary Detection](../../ICCV2025/segmentation/online_generic_event_boundary_detection.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](../../CVPR2026/segmentation/geomprompt_rgbd_segmentation.md)
- [\[ICCV 2025\] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP](../../ICCV2025/segmentation/know_no_better_a_data-driven_approach_for_enhancing_negation_awareness_in_clip.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)

</div>

<!-- RELATED:END -->
