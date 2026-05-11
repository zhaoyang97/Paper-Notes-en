---
title: >-
  [Paper Note] MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision
description: >-
  [CVPR 2026][Human Understanding][edge detection] MatchED introduces a lightweight (~21K parameter) plug-and-play module that generates crisp (single-pixel-wide) edge maps by performing one-to-one bipartite matching betwe…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "edge detection"
  - "crisp edges"
  - "bipartite matching"
  - "plug-and-play"
  - "end-to-end training"
date: 2026-05-08
content_hash: 44ecba3d9b3e6d8b
---

# MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision

**Conference**: CVPR 2026
**arXiv**: [2602.20689](https://arxiv.org/abs/2602.20689)
**Code**: [https://cvpr26-matched.github.io](https://cvpr26-matched.github.io)
**Area**: Human Understanding / Edge Detection
**Keywords**: edge detection, crisp edges, bipartite matching, plug-and-play, end-to-end training

## TL;DR

MatchED introduces a lightweight (~21K parameter) plug-and-play module that generates crisp (single-pixel-wide) edge maps by performing one-to-one bipartite matching between predicted and GT edge pixels during training, based on spatial distance and confidence. The module can be appended to any edge detector for end-to-end training, and for the first time matches or surpasses standard post-processing methods without relying on NMS and thinning.

## Background & Motivation

**Background**: Edge detection is a fundamental problem in computer vision, underpinning downstream tasks such as depth estimation, semantic segmentation, and image generation. Modern deep learning edge detectors (HED, RCF, PiDiNet, RankED, SAUGE, etc.) have achieved significant progress in detection accuracy, yet nearly all methods rely on a standard post-processing pipeline to produce the final single-pixel-wide edge maps: Non-Maximum Suppression (NMS) followed by skeleton-based thinning.

**Limitations of Prior Work**: NMS and skeleton thinning are hand-crafted, non-differentiable algorithms that completely block end-to-end optimization. This introduces three core issues: (i) training optimizes a "thick" edge probability map while testing employs post-processing to yield "thin" edges, creating a train-test protocol inconsistency; (ii) post-processing hyperparameters (NMS window size, boundary decay, etc.) require separate tuning and cannot be optimized via gradients; (iii) the few methods that attempt to directly generate crisp edges (LPCB, CATS, DiffusionEdge, CPD, etc.) still require post-processing to achieve satisfactory performance.

**Key Challenge**: Edge annotations themselves are spatially imprecise due to human labeling bias, resulting in positional offsets between predictions and GT. To compensate, models tend to produce thicker edge responses to "hedge" against annotation noise. The only prior attempt to address this, GLR, refines labels using a fixed Canny-guided preprocessing step before training, but cannot dynamically adapt to the model's evolving predictions during training.

**Goal**: (a) How to enable edge detectors to directly output single-pixel-wide crisp edges? (b) How to align training objectives with test-time evaluation? (c) How to design a universal module attachable to any existing detector?

**Key Insight**: The authors draw inspiration from matching in object detection (e.g., DETR's bipartite matching)—if a one-to-one optimal matching between predicted and GT edge pixels can be established at each training iteration, each predicted pixel is assigned to at most one GT pixel, naturally preventing multiple responses from collapsing onto the same GT and producing thick edges. The matching simultaneously considers spatial distance and confidence, with the distance threshold aligned to the evaluation protocol to ensure train-test consistency.

**Core Idea**: Replace non-differentiable post-processing with differentiable matching-based supervision, directly producing crisp edges through prediction-GT bipartite matching during training.

## Method

### Overall Architecture

The MatchED pipeline is straightforward: given any edge detector $f$ (CNN/Transformer/Diffusion-based), its output raw edge map $\mathbf{E}_r = f(I; \theta_r)$ is passed to MatchED, a lightweight CNN appended after the last layer of $f$, which refines it into a crisp edge map $\mathbf{E}_c = \text{MatchED}(\mathbf{E}_r; \theta_c)$. During training, MatchED performs bipartite matching between predictions and GT at each iteration to generate a matching-based GT, then optimizes with BCE loss. At inference, it directly outputs crisp edge maps without NMS or thinning.

### Key Designs

1. **Alignment Cost Matrix Construction**:

    - Function: Computes matching costs between predicted and GT edge pixels at each training iteration.
    - Mechanism: The cost matrix jointly considers spatial distance and confidence. When three conditions are met (predicted confidence exceeds threshold $\tau_c$, the GT pixel is an edge, and Manhattan distance is within threshold $\tau_d$), the cost is $d(\mathbf{p_c}, \mathbf{p_g}) - \alpha \cdot \mathbf{E}_c(\mathbf{p_c})$; otherwise the cost is infinity.
    - Design Motivation: The three conditions respectively ensure that only high-confidence predictions are considered, only true GT edges are matched, and only locally proximate matches are permitted. $\tau_d$ is aligned with the evaluation protocol, serving as a critical guarantee of train-test consistency. Higher confidence yields lower cost, encouraging high-confidence responses.

2. **One-to-one Bipartite Matching**:

    - Function: Solves an optimal one-to-one assignment over the cost matrix, ensuring each predicted pixel is matched to at most one GT pixel.
    - Mechanism: The Hungarian algorithm is used to find the optimal assignment minimizing total cost. After obtaining the optimal assignment, a matching-based GT is constructed. GT edge pixels with no predicted response within $\tau_d$ are directly restored into the matching GT to ensure subsequent iterations can match them.
    - Design Motivation: One-to-one matching is the core mechanism for eliminating thick edges—if multiple predicted pixels attempt to match the same GT pixel, the optimal assignment retains only one and marks the rest as non-edges, naturally suppressing edge dilation and forcing the model to precisely localize edges.

3. **Lightweight CNN Architecture of MatchED**:

    - Function: Refines the raw edge map into a crisp edge map.
    - Mechanism: Consists of only five standard convolutional blocks (Conv2D + ReLU + Normalization) followed by Sigmoid. The total parameter count is approximately 21K, adding only 3% overhead to PiDiNet and less than 0.02% to large models.
    - Design Motivation: The module must be sufficiently lightweight to serve as a plug-and-play solution. The refinement capability is driven by matching-based supervision.

4. **Two-stage Training Strategy**:

    - Function: Ensures MatchED operates on reliable raw edge maps.
    - Mechanism: During the first $N/2$ epochs, only the base detector is trained; during the remaining $N/2$ epochs, the detector and MatchED are jointly trained.
    - Design Motivation: MatchED's matching is only effective once the base model produces reasonable edge responses.

### Loss & Training

The total loss is a weighted combination of the base model loss and the MatchED loss:

$$\mathcal{L}_{\text{total}} = \beta \cdot \mathcal{L}_{\text{MatchED}} + \mathcal{L}_f$$

The MatchED loss is binary cross-entropy between the predicted edge map and the matching-based GT. $\beta$ controls the weighting. The base detector retains its original loss function (weighted BCE for PiDiNet, ranking loss for RankED, etc.), and MatchED is fully transparent to these.

## Key Experimental Results

### Main Results

Evaluated on four datasets (BSDS500 / NYUDv2 / BIPED-v2 / Multi-Cue) with four base models; CEval denotes evaluation without post-processing:

| Dataset | Model | CEval ODS | CEval OIS | CEval AC | Gain vs. Base |
|--------|------|-----------|-----------|----------|------------|
| BSDS | PiDiNet+MatchED | .800 | .811 | .866 | ODS +0.222, AC +0.717 |
| BSDS | RankED+MatchED | .789 | .795 | .600 | ODS +0.188, AC +0.438 |
| BSDS | SAUGE+MatchED | .809 | .813 | .818 | ODS +0.156, AC +0.631 |
| BSDS | DiffEdge+MatchED | .830 | .839 | .875 | ODS +0.084, AC +0.474 |
| NYUDv2 | PiDiNet+MatchED | .736 | .749 | .930 | ODS +0.337, AC +0.757 |
| NYUDv2 | RankED+MatchED | .775 | .784 | .886 | ODS +0.298, AC +0.740 |
| NYUDv2 | DiffEdge+MatchED | .759 | .762 | .937 | ODS +0.032, AC +0.074 |
| BIPED-v2 | PiDiNet+MatchED | .900 | .905 | .971 | Surpasses post-processed version |

Comparison with crisp edge detection state of the art (AC metric):

| Method | BSDS AC | Multi-Cue AC | BIPED AC |
|------|---------|-------------|----------|
| PiDiNet+Dice | .306 | .208 | .340 |
| DiffusionEdge | .401 | .498 | .879 |
| **MatchED** | **.875** | **.846** | **.971** |

### Ablation Study

Hyperparameter analysis on BSDS with PiDiNet (baseline ODS=.789, OIS=.803):

| Hyperparameter | Setting | ODS | OIS | AP | Note |
|--------|------|------|------|----|------|
| Confidence threshold | 0.01 | .800 | .811 | .866 | Optimal |
| Confidence threshold | 0.10 | .799 | .808 | .836 | Still outperforms baseline |
| Confidence threshold | 0.30 | .761 | .771 | .803 | Degrades when too high |
| Distance threshold | 2 | .797 | .807 | .856 | Relatively robust |
| Distance threshold | 4 | .800 | .811 | .866 | Optimal |
| Confidence weight | 5 | .630 | .639 | .653 | Too low |
| Confidence weight | 25 | .800 | .811 | .866 | Optimal range |

Runtime comparison (654 NYUD test images, CPU):

| Method | Time (s) |
|------|---------|
| NMS | 25.69 |
| NMS + Thinning (x100) | 1875.57 |
| **MatchED** | **32.98** |

### Key Findings

- **2–4× improvement in AC**: MatchED substantially sharpens edge crispness; on BSDS, PiDiNet's AC improves from 0.149 to 0.866.
- **First to match or surpass standard post-processing**: SAUGE+MatchED achieves CEval ODS of .809 on BSDS, exceeding SEval .808.
- **Applying post-processing on top of MatchED outputs degrades performance**, confirming the outputs are already crisp.
- **Effective across four architectures (CNN/Transformer/Diffusion/SAM)**, validating generalizability.
- **Minimal 21K parameter overhead**; inference time is far lower than full post-processing pipelines.

## Highlights & Insights

- **Matching-based supervision is the core innovation.** Transferring bipartite matching from object detection to pixel-level edge detection, the one-to-one constraint naturally eliminates edge dilation. The matching cost encodes spatial distance and confidence, with the distance threshold aligned to the evaluation protocol, achieving train-test consistency.
- **Truly plug-and-play design.** Only a 21K-parameter CNN and a matching loss are added, without modifying the original model architecture or loss. Universality is validated across CNN, Transformer, and Diffusion paradigms.
- **A "post-processing killer."** NMS+thinning has been standard practice in edge detection for decades; MatchED is the first to demonstrate it is unnecessary. This has broad implications for pixel-level prediction tasks that rely on non-differentiable post-processing.
- **Unmatched GT recovery strategy.** GT edges with no predicted response are retained in the matching GT to prevent information loss and ensure subsequent iterations can learn from them.

## Limitations & Future Work

- **GPU memory overhead**: At 320×320 input resolution, the matching matrix reaches 28.32 GB, requiring patch-wise processing, limiting scalability to high-resolution inputs.
- **Retraining required**: Hyperparameter adjustments necessitate retraining, unlike NMS which can be tuned instantly.
- **RankED downsampling artifacts**: 0.25× resolution produces interpolation artifacts that affect matching quality.
- **Only four standard benchmarks**: Evaluation on complex real-world scenarios (autonomous driving, industrial inspection, etc.) is absent.
- **Hungarian algorithm complexity**: $O(n^3)$ may become a bottleneck in extremely dense edge scenarios.

## Related Work & Insights

- **vs. DiffusionEdge**: The strongest crisp edge SOTA, but still requires post-processing. MatchED as a plug-in further improves its AC (+0.474 on BSDS).
- **vs. LPCB (Dice loss)**: Encourages crisp edges through loss design, but still produces thick predictions. MatchED fundamentally resolves the many-to-one problem at the matching level.
- **vs. GLR**: Refines labels with a fixed Canny-guided step prior to training, without dynamic adaptation. MatchED performs dynamic matching at every iteration.
- **vs. DETR**: The matching concept originates from DETR, but is extended from instance-level to pixel-level; the cost matrix design is entirely different.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Transfers bipartite matching from object detection to pixel-level edge detection, achieving post-processing-free crisp edges for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets, four architectures, detailed ablations, runtime and parameter analysis, and qualitative visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, methodology is concise, and contributions are well-defined.
- **Value**: ⭐⭐⭐⭐ A post-processing-free end-to-end solution; matching-based supervision is generalizable to other pixel-level tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)
- [\[CVPR 2026\] FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision](flexavatar_learning_complete_3d_head_avatars_with_partial_supervision.md)
- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)
- [\[CVPR 2026\] RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised HOI Detection](regformer_transferable_relational_grounding_for_weakly-supervised_hoi_detection.md)
- [\[CVPR 2026\] All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark](all_in_one_unifying_deepfake_detection_tampering_localization_and_source_tracing.md)

</div>

<!-- RELATED:END -->
