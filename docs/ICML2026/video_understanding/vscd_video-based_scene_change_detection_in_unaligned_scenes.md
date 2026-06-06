---
title: >-
  [Paper Note] VSCD: Video Scene Change Detection in Unaligned Scenarios
description: >-
  [ICML 2026][Video Understanding][Scene Change Detection] This paper introduces the VSCD task—detecting object-level changes per pixel between two video sequences of the same environment recorded at different times. Under…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Scene Change Detection"
  - "Video Alignment"
  - "Multi-reference Matching"
  - "Multi-view Geometry"
  - "Long-term Autonomy"
date: 2026-05-08
content_hash: 615554f33ba741b5
---

# VSCD: Video Scene Change Detection in Unaligned Scenarios

**Conference**: ICML 2026  
**arXiv**: [2605.20821](https://arxiv.org/abs/2605.20821)  
**Code**: https://github.com/AutoCompSysLab/VSCD  
**Area**: Video Understanding / Video Comparison  
**Keywords**: Scene Change Detection, Video Alignment, Multi-reference Matching, Multi-view Geometry, Long-term Autonomy

## TL;DR
This paper introduces the VSCD task—detecting object-level changes per pixel between two video sequences of the same environment recorded at different times. Under unconstrained camera motion and severe viewpoint mismatch, the method leverages temporal consistency, patch-level correspondence, and confidence-weighted fusion through a query-centric multi-reference model.

## Background & Motivation

**Background**: Change detection is a classic computer vision problem. Existing methods generally fall into two categories: image-based (RSCD, SCD), which assume relatively fixed viewpoints, and video-based (AOD), which assume reference and query videos follow identical or opposite trajectories.

**Limitations of Prior Work**: These methods fail to handle three real-world challenges: (1) unconstrained camera motion; (2) drastic viewpoint changes; and (3) simultaneous appearance or disappearance of multiple objects. These issues coexist when mobile robots detect environmental changes during long-term autonomous operation.

**Key Challenge**: Frame-level registration is unfeasible—comparing any two frames directly results in massive alignment errors due to completely different viewpoints. However, the temporal structure of video sequences contains sufficient regularity.

**Goal**: Define the new VSCD task; construct a large-scale annotated dataset (1.1M+ frames plus a real-world test set); and propose a method that utilizes temporal structure to detect changes without explicit trajectory alignment.

**Key Insight**: Although single-frame mismatch is severe, **the temporal coherence of video sequences and multi-view geometric constraints are sufficient for reliable reasoning**.

**Core Idea**: **Multi-reference matching + Temporal alignment + Patch-level correspondence + Confidence-weighted fusion**—implicitly learning robust change detection capabilities from video sequences without prior knowledge of camera motion or trajectory alignment.

## Method

### Overall Architecture
VSCDNet utilizes a query-centric multi-reference architecture consisting of three stages: (1) **Frame-level alignment**: Uniformly sampling keyframes, encoding them via ViT, computing a frame-level similarity grid, and finding candidate reference frame combinations through soft matching; (2) **Patch-level correspondence**: Computing local correlation volumes at the patch scale for each reference candidate and performing geometric compensation via differentiable warp; (3) **Confidence-weighted fusion**: Combining frame-level confidence (from the frame matching distribution) and patch-level confidence (from local matching sharpness and entropy) to fuse multi-reference change features.

### Key Designs

1.  **Temporal Consistency Frame Alignment**:
    - **Function**: To find corresponding frame segments within the reference and query videos, providing coarse-grained correspondence for subsequent patch matching.
    - **Mechanism**: Computes the cosine similarity of frame features $S_{t,s} = \cos(v_t^q, v_s^r)$ for each keyframe pair $(t, s)$, refined by a shallow convolutional head to obtain $A = S + h_\psi(S)$; then normalized via row-wise softmax as $P_{\text{frame}}(t,s) = \text{softmax}_s(A_{t,s}/\tau_f)$. Temporal coherence is used to cluster frames into segments via matching segment proposals.
    - **Design Motivation**: Temporal coherence is a key constraint for sequence-level alignment; reliable reference segments can be identified using temporal smoothing priors without explicit pose estimation.

2.  **Patch-level Correspondence + Differentiable Warp**:
    - **Function**: Locally compensates for viewpoint changes and occlusions within the feature space.
    - **Mechanism**: For each reference candidate $s$, the dot-product correlation between query patches and reference patches is computed within a $k \times k$ local window: $P_{\text{patch},i}^{(t,s)}(x,y) = \text{softmax}(dots)$; the expected displacement is calculated as $\Delta^{(t,s)}(x,y) = \sum_i P_{\text{patch},i}^{(t,s)} \delta_i$. Reference features are warped via bilinear sampling, and a lightweight convolutional head fuses them to obtain change features $F_{t,s} = g_\phi(E_t^q, E_{t,s}^{r(w)})$.
    - **Design Motivation**: Patch-level local matching is more robust than global matching; differentiable warping avoids explicit pose estimation; the soft correlation distribution retains uncertainty information.

3.  **Confidence-weighted Fusion**:
    - **Function**: Intelligently aggregates multi-reference change features while suppressing candidates with uncertain matching or failed geometric registration.
    - **Mechanism**: Frame-level confidence $C_f(t,s) = P_{\text{frame}}(t,s)$; patch-level confidence $C_{sp}^{(t,s)}(x,y) = c_p \cdot p_{\max}^{(t,s)} + c \cdot (1 - e^{(t,s)})$ (peak + normalized entropy); fusion is performed as $F_t = \sum_s C_f(t,s) \cdot C_{sp}^{(t,s)} \cdot F_{t,s} / \text{norm}$.
    - **Design Motivation**: Direct fusion of multiple references is easily contaminated by poor registrations; entropy detects "ambiguous" matches (multiple offset probabilities are similar), which usually indicates the patch cannot be reliably aligned.

## Key Experimental Results

### Main Results

| Method | Synthetic F1 | Real-world F1 | vs Prev. SOTA |
| :--- | :--- | :--- | :--- |
| TCF-LMO (AOD) | 19.7% | 10.3% | -44.5% |
| PBCD-MC (AOD) | 26.8% | 16.1% | -28.4% |
| CSCDNet (SCD) | 19.8% | 9.1% | -45.4% |
| DR-TANet (SCD) | 20.6% | 11.6% | -43.1% |
| C-3PO (SCD) | 24.1% | 11.7% | -39.9% |
| GeSCF (Prev. SOTA) | 29.5% | 17.3% | baseline |
| **VSCDNet (Ours)** | **36.6%** | **25.4%** | **+7.1% / +8.1%** |

### Hierarchical Evaluation

| Video Length | Low | Med | High | Low Graph Quality | Med | High | Few Changes | Med | Many |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| F1 | 38.1% | 36.9% | 33.9% | 40.7% | 31.7% | 32.1% | 37.7% | 39.0% | 36.6% |

### Key Findings
- Temporal consistency frame alignment converts chaotic frame-to-frame matching into ordered sequence correspondence, serving as the cornerstone of model performance.
- Patch-level correspondence is more robust than global features, maintaining a 31-40% F1 score in scenes with high viewpoint variation.
- Entropy-regularized confidence is crucial; normalized entropy provides additional detection for "flat distributions" (where multiple offsets have equal probability).
- Generalization to real-world data: The performance drop from synthetic to real is approximately 11%, demonstrating strong generalization compared to other methods (which drop 50%+).

## Highlights & Insights
- **Paradigm Shift from Frame to Sequence**: Breakthrough use of the video sequence's temporal structure as an alignment prior, eliminating the need for SLAM or motion estimation.
- **Elegance of Implicit Geometric Learning**: Implicitly learns multi-view correspondence through patch-level correlation and differentiable warping instead of explicitly estimating camera poses.
- **Ingenious Dual-layer Confidence Mechanism**: The combination of peak and entropy detects both "certain matches" and "ambiguous matches."
- **New Benchmark for Unconstrained Video Understanding**: The scale of 1.1 million frames and its authenticity exceeds existing change detection datasets by an order of magnitude.

## Limitations & Future Work
- The method depends on video temporal length; segmentation and streaming processing for extremely long videos remain unexplored.
- The real-world dataset contains only 8 video pairs, offering limited environmental diversity.
- It assumes that object states remain fixed during a single video recording pass.
- Future improvements: Sliding windows or hierarchical temporal encoding for ultra-long videos; collection of more real-world data; adaptive hyperparameter tuning; and optimization of the inference pipeline.

## Related Work & Insights
- **vs RSCD**: Aerial/satellite-based, assuming fixed viewpoints; Ours targets indoor scenes with strong mismatches.
- **vs SCD**: Handles isolated image pairs; Ours utilizes temporal coherence.
- **vs AOD**: Assumes identical/opposite trajectories; Ours tackles more difficult unconstrained motion.
- **vs Video Copy Localization**: Ours borrows the frame similarity map concept but applies it to pixel-level change detection.
- **Insight**: Multi-reference fusion + confidence weighting can be transferred to video frame interpolation, stereo matching, and optical flow estimation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The VSCD task definition fills a critical gap; the shift in thinking combined with implicit geometric learning is an industry first.
- Experimental Thoroughness: ⭐⭐⭐⭐ 1.1M synthetic frames + 8 real pairs + 4 baseline comparisons + hierarchical evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, standardized formulas, and highly informative charts.
- Value: ⭐⭐⭐⭐⭐ Addresses frontier requirements for long-term autonomous navigation, providing a practical solution + high-quality dataset + open-source implementation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection](privacy-aware_video_anomaly_detection_through_orthogonal_subspace_projection.md)
- [\[CVPR 2026\] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](../../CVPR2026/video_understanding/savax_egotoexo_imitation_error_detection_via_scene.md)
- [\[CVPR 2026\] Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting](../../CVPR2026/video_understanding/seen_to_scene_keep_the_seen_generate_the_unseen_for_video_outpainting.md)
- [\[AAAI 2026\] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](../../AAAI2026/video_understanding/r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)
- [\[ACL 2026\] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](../../ACL2026/video_understanding/response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)

</div>

<!-- RELATED:END -->
