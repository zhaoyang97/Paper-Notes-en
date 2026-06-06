---
title: >-
  [Paper Note] FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT
description: >-
  [CVPR 2026][Video Understanding][Referring Multi-Object Tracking] This paper proposes FlexHook, a novel two-stage Referring-by-Tracking framework that redefines feature construction via a sampling-based Conditioning Hook…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Referring Multi-Object Tracking"
  - "Two-Stage RBT"
  - "Sampling-Based Feature Construction"
  - "Pairwise Correspondence Decoding"
  - "Language-Conditioned Enhancement"
date: 2026-05-08
content_hash: 7a7994c289e0eec1
---

# FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT

**Conference**: CVPR 2026
**arXiv**: [2503.07516](https://arxiv.org/abs/2503.07516)  
**Code**: [GitHub](https://github.com/buptLwz/FlexHook)  
**Area**: Video Understanding
**Keywords**: Referring Multi-Object Tracking, Two-Stage RBT, Sampling-Based Feature Construction, Pairwise Correspondence Decoding, Language-Conditioned Enhancement

## TL;DR
This paper proposes FlexHook, a novel two-stage Referring-by-Tracking framework that redefines feature construction via a sampling-based Conditioning Hook (C-Hook) and replaces CLIP cosine similarity matching with a Pairwise Correspondence Decoder (PCD), making a two-stage method comprehensively surpass current state-of-the-art one-stage methods for the first time.

## Background & Motivation
Referring Multi-Object Tracking (RMOT) aims to track multiple targets in video based on natural language expressions. Existing methods fall into three paradigms:

**Tracking-by-Referring (TBR)**: Localizes bounding boxes via GroundingDINO and associates trajectories, relying on large-scale VLMs.

**One-Stage RBT**: Decodes trajectory queries based on MOTR and computes matching scores, requiring end-to-end joint optimization.

**Two-Stage RBT**: Proposed by iKUN, fully decouples tracking from referring, offering low training cost and support for incremental deployment.

**Key Challenge**: Although two-stage RBT holds irreplaceable advantages in training efficiency and incremental deployment, its performance lags far behind one-stage methods (iKUN achieves only 10.32 HOTA on Refer-KITTI-v2). The authors identify two fundamental limitations:

- **Over-heuristic feature construction**: Existing methods apply dual encoding of full images and cropped patches using a shared encoder, ignoring the spatial gradient flow and context aggregation capabilities already present in modern visual backbones. Furthermore, feature construction is language-agnostic, preventing adaptive focus based on different semantic cues (e.g., position, direction).
- **Fragile correspondence modeling**: Reliance on cosine similarity in CLIP's pre-trained alignment space causes alignment collapse whenever additional modules are introduced or the backbone is replaced, effectively imposing a performance ceiling.

## Method

### Overall Architecture
FlexHook operates analogously to hook functions in programming—it intercepts features within the forward pass of an existing visual backbone without introducing additional encoding stages. Given a $p$-frame trajectory segment $\mathcal{B}^i_{t:t+p}$, the image is globally encoded only once; the C-Hook sampling → temporal integration → PCD decoding workflow is then applied at each backbone layer. Multi-scale results are aggregated via a feature pyramid, and $\hat{N}$ matching scores are produced as output.

### Key Designs

1. **Conditioning Hook (C-Hook)**: Samples target features and language-conditioned cues directly from the backbone's raw feature stream.

    - **Neighboring Grid Sampling**: The bounding box $B^i_t = \langle x_0, y_0, w_b, h_b \rangle$ is converted into a coordinate grid $P^i_t \in \mathbb{R}^{h \times w \times 2}$, from which target features $J$ are sampled from the feature map $F_v$ via differentiable grid sampling (bilinear interpolation). To bridge the distribution gap between GT trajectories used in training and tracker outputs used at inference, three data augmentation strategies are introduced: (i) random trajectory truncation to simulate target loss, (ii) Gaussian noise injection to simulate localization inaccuracy, and (iii) intra-batch grid sequence shuffling to simulate ID switches.
    - **Conditioning Enhancement**: Learns language-conditioned reference points to inject linguistic priors. Learnable query vectors $Q_{LR} \in \mathbb{R}^{\hat{N} \times M \times C}$ perform cross-attention with text features $F_l$; an MLP followed by sigmoid generates normalized 2D reference points $P_r$, which are repeated along the temporal dimension and sampled together with the coordinate grid.
    - **Design Motivation**: The sampling strategy preserves the gradient flow from backbone pre-training and avoids redundant encoding; language-conditioned sampling enables the model to adaptively attend to different regions depending on the semantic expression (e.g., "person in red" vs. "person on the left").

2. **Temporal Integration**: Extracts explicit optical flow information via coordinate grid differencing.

    - **Function**: Frame-wise grid displacements $\Delta Grid = \text{Cat}(\{P^i_{t+k} - P^i_{t+k-1}\}_{k=1}^p)$ are concatenated with multi-frame features $J$ along the channel dimension and compressed by an MLP to obtain trajectory features $F_J \in \mathbb{R}^{h \times w \times C}$.
    - **Mechanism**: The coordinate grids already constructed by C-Hook are reused to derive object-level optical flow without any additional network.
    - Reference features $F_r$, which carry no motion information, are obtained directly via temporal pooling.

3. **Pairwise Correspondence Decoder (PCD)**: A learnable correspondence decoder that replaces CLIP cosine similarity.

    - **Function**: Forms $\hat{N}$ sample pairs from $\hat{N}$ referring expressions and a shared trajectory segment, using learnable queries $Q \in \mathbb{R}^{\hat{N} \times C}$ to extract a matching score for each pair.
    - **Mechanism**: The flattened trajectory features $F_J$, reference features $F_r$, and language features $F_l$ are concatenated along the first dimension as Key/Value, while $Q$ serves as Query in masked cross-attention. The attention mask $A$ allows all queries to share the trajectory features while restricting each query to its own corresponding language and reference features, enabling pairwise output and implicitly supporting cross-pair contrastive learning.
    - After decoding, an FFN branches into two paths: one MLP predicts matching scores $S \in \mathbb{R}^{\hat{N} \times 2}$, and the other passes representations to the next PCD layer for multi-scale decoding.

### Loss & Training
- **Focal Loss**: Supervises the averaged output $\bar{S}$ across all layers, enhancing multi-scale capability and alleviating sample imbalance.
- **Reference Point Boundary Penalty $\mathcal{L}_r$**: Prevents the learned reference point coordinates from collapsing to the boundary of the normalized space $[-1,1]^2$. The minimum boundary distance is defined as $d_{uv} = \min(1-|u|, 1-|v|)$, with a softplus penalty: $\mathcal{L}_r = \frac{1}{|P_r|}\sum_{u,v}\text{softplus}(\alpha(\delta - d_{uv}))$.
- **Total Loss**: $\mathcal{L} = \mathcal{L}_{\text{Focal}}(\bar{S}, S_{\text{gt}}) + \lambda \mathcal{L}_r$
- **Training Setup**: Input resolution $224 \times 672$ (far smaller than iKUN's dual-encoding scheme), AdamW optimizer with lr=3e-5, 20 epochs, 2× RTX 4090.

## Key Experimental Results

### Main Results

| Dataset | Metric | FlexHook-best | Prev. SOTA | Gain |
|--------|------|---------------|----------|------|
| Refer-KITTI | HOTA | 53.83 | 52.41 (HFF-Tracker) | +1.42 |
| Refer-KITTI-V2 | HOTA | 42.53 | 36.18 (HFF-Tracker) | +6.35 |
| Refer-Dance | HOTA | 32.17 | 29.06 (iKUN) | +3.11 |
| LaMOT | HOTA | 56.77 | 48.45 (LaMOTer) | +8.32 |

FlexHook is the first two-stage method to comprehensively surpass one-stage state-of-the-art across all benchmarks.

### Ablation Study

| Configuration | HOTA | DetA | AssA | Note |
|------|------|------|------|------|
| iKUN (baseline) | 10.32 | 2.17 | 49.77 | Two-stage baseline |
| +C-Hook | 34.49 | 22.51 | 52.97 | Sampling-based construction yields large gain |
| +C-Hook+PCD | 38.62 | 27.92 | 53.58 | Pairwise decoding replaces cosine similarity |
| +C-Hook+PCD+TI | 39.19 | 28.47 | 54.11 | Optical flow temporal integration |

### Key Findings
- C-Hook yields the largest improvement (HOTA +24.17), confirming that sampling-based feature construction is the key innovation.
- PCD is effective not only in non-aligned spaces but also outperforms cosine similarity in CLIP-aligned spaces.
- The number of reference points $M=10$ for Conditioning Enhancement is empirically optimal and consistently effective across different backbones.
- Freezing the encoder incurs only a minor performance drop (40.86 vs. 42.53), offering a favorable training efficiency trade-off.
- FlexHook achieves the fastest overall inference speed (51.47 min), attributed to the elimination of redundant encoding and PCD's parallel processing.

## Highlights & Insights
1. **"Hook" Philosophy**: Rather than modifying the backbone, the framework intercepts and samples from the forward pass, preserving pre-trained capabilities while supporting plug-and-play replacement of any visual or text encoder.
2. **Breaking CLIP Dependency**: PCD transforms passive similarity comparison into active correspondence modeling, freeing the framework from the constraints of any specific pre-trained alignment space.
3. **Training–Inference Consistency in Neighboring Grid Sampling**: Noise injection elegantly bridges the distribution gap between GT and tracker outputs by simulating the uncertainty inherent in tracker predictions.
4. **Revival of the Two-Stage Paradigm**: The proposed method surpasses one-stage approaches requiring 51.68 hours of training, while incurring only 1.91 hours of training cost.

## Limitations & Future Work
- Performance is upper-bounded by the quality of the external detector-tracker; a weak detector will degrade results accordingly.
- The current benchmarks in autonomous driving scenarios (Refer-KITTI) cover a limited range of expression types (primarily car/pedestrian); generalization to more complex expression scenarios (e.g., indoor environments, complex motion descriptions) remains to be validated.
- Reference point collapse requires additional regularization, suggesting potential optimization instability during training.

## Related Work & Insights
- Shares the two-stage RBT paradigm with iKUN (CVPR'24), but completely redesigns both feature construction and correspondence modeling.
- The grid sampling approach in C-Hook is conceptually analogous to deformable attention in Deformable DETR.
- The masked cross-attention design in PCD follows the query-based decoding lineage of the DETR family.
- The framework is particularly well-suited for lightweight incremental deployment scenarios, such as edge vehicles equipped with mature existing trackers.

## Rating
- Novelty: ⭐⭐⭐⭐ — Both core modules are uniquely designed; the hook philosophy is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated on 4 datasets with multiple encoder combinations and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is clear and well-illustrated.
- Value: ⭐⭐⭐⭐ — Revives the two-stage paradigm with practical significance for industrial deployment.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)
- [\[ACL 2026\] Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning](../../ACL2026/video_understanding/rethinking_the_idiomaticity_decomposability_hypothesis_evidence_from_distributio.md)
- [\[NeurIPS 2025\] Two Causally Related Needles in a Video Haystack](../../NeurIPS2025/video_understanding/two_causally_related_needles_in_a_video_haystack.md)
- [\[AAAI 2026\] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective](../../AAAI2026/video_understanding/rethinking_progression_of_memory_state_in_robotic_manipulation_an_object-centric.md)
- [\[CVPR 2026\] Drift-Resilient Temporal Priors for Visual Tracking](drift-resilient_temporal_priors_for_visual_tracking.md)

</div>

<!-- RELATED:END -->
