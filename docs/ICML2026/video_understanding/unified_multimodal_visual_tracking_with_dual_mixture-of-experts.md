---
title: >-
  [Paper Note] Unified Multimodal Visual Tracking with Dual Mixture-of-Experts
description: >-
  [ICML 2026][Video Understanding][Visual Tracking] OneTrackerV2 unifies five tracking tasks—RGB, RGB+D, RGB+T, RGB+E, RGB+N—into a single network trained end-to-end. It uses a Meta Merger for modality fusion…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Visual Tracking"
  - "RGB+X"
  - "Mixture-of-Experts"
  - "Feature Decoupling"
  - "Modality-Missing Robustness"
date: 2026-05-08
content_hash: 5fe06d4355e894a6
---

# Unified Multimodal Visual Tracking with Dual Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2605.03716](https://arxiv.org/abs/2605.03716)  
**Code**: None  
**Area**: Video Understanding / Multimodal Visual Tracking / Mixture-of-Experts  
**Keywords**: Visual Tracking, RGB+X, Mixture-of-Experts, Feature Decoupling, Modality-Missing Robustness

## TL;DR
OneTrackerV2 unifies five tracking tasks—RGB, RGB+D, RGB+T, RGB+E, RGB+N—into a single network trained end-to-end. It uses a Meta Merger for modality fusion, and Dual MoE to explicitly decouple "spatiotemporal matching" and "modality fusion" into T-MoE and M-MoE, respectively. Dissimilarity loss and router clustering ensure these do not collapse into the same subspace.

## Background & Motivation
**Background**: Visual object tracking is divided by input modality into RGB and RGB+X (X=Depth/Thermal/Event/Language). Mainstream approaches include: (a) designing architectures and training for each X task independently; (b) adapting pretrained RGB trackers via fine-tuning (e.g., OneTracker); (c) preliminary unified models like SUTrack, which concatenate multimodal tokens and use a shared backbone.

**Limitations of Prior Work**: (1) Multi-stage training (pretrained → finetune) often converges suboptimally; (2) Lack of unified architecture, still requiring manual task-specific branches; (3) Parameters are grouped by task even in shared architectures, not truly "unified params"; (4) Performance collapses if any modality is missing at inference; (5) Feature conflict—simple token concatenation forces the same parameter space to learn both spatiotemporal matching and modality-specific patterns, causing interference.

**Key Challenge**: Tracking inherently requires two distinct capabilities: spatiotemporal matching (template ↔ search, cross-frame motion) and modality fusion (complementary cues from RGB ↔ X). Forcing both into a single backbone or MoE leads to zero-sum parameter competition.

**Goal**: (1) Single-step, end-to-end training with shared parameters and architecture; (2) Modality fusion as modality-agnostic, missing-modality-robust "meta embedding"; (3) Structural decoupling to resolve feature conflict between spatiotemporal matching and modality fusion; (4) Scalable capacity without exploding inference cost.

**Key Insight**: Use learnable meta embedding as a central modality hub; introduce Dual MoE so two expert groups handle spatiotemporal and modality tasks separately, enforced to be orthogonal via explicit decoupling loss.

**Core Idea**: Meta Merger + Dual MoE = one network, one training, one parameter set for five tracking tasks, robust to missing modalities and model compression.

## Method

### Overall Architecture
Input consists of template and search regions, each containing RGB and an X modality frame (for RGB-only tasks, X is replaced by RGB). Both streams share patch embedding to obtain $F_{rgb},F_x$, which are fused by the Meta Merger using a learnable meta embedding $F_{meta}$ via spatial + channel attention and centralized convolution, yielding modality-agnostic token sequences. These tokens are fed into a Vision Transformer backbone, where each block replaces the FFN with Dual MoE: each token is processed in parallel by a shared expert, T-MoE (top-$k$), and M-MoE (top-$k$), with outputs summed. The SUTrack-style classification + IoU + L1 detection head outputs the bounding box. Four model variants are provided: B224 / B384 / L224 / L384, with 80M–271M parameters and inference FPS of 23.4–72.4.

### Key Designs

1. **Meta Merger: Modality-Agnostic Central Hub**:

    - **Function**: Compresses heterogeneous RGB and X modality features into a unified space, inherently robust to missing modalities.
    - **Mechanism**: For $F_{rgb}$ and $F_x$, compute $W^{spatial}=\sigma(\mathrm{Conv}(F^{avg})+\mathrm{Conv}(F^{max}))$ and $W^{channel}=\sigma(\mathrm{Linear}(F^{avg})+\mathrm{Linear}(F^{max}))$ for enhancement; introduce learnable $F_{meta}$, and use $F_{meta}'=\mathrm{Conv}(\mathrm{Conv}(F_{meta}+F'_{rgb})+\mathrm{Conv}(F_{meta}+F'_x)+F_{meta})$ so meta embedding acts as a cross-modal intermediary, outputting globally aligned tokens. If X is missing, it degrades to RGB-only interaction without changing the fusion pipeline.
    - **Design Motivation**: Compared to SUTrack's direct token concatenation, meta embedding avoids the doubled computation of multi-branch designs and uses a global variable as a "modality translator," naturally adapting to any modality combination.

2. **Dual MoE: Explicit Decoupling of T-MoE and M-MoE**:

    - **Function**: Separates spatiotemporal matching and modality fusion into two independent expert groups, avoiding heterogeneous objectives in the same parameter space.
    - **Mechanism**: For each token $x$, DMoE outputs $y=E_{shared}(x)+\sum_{i\in S^T_k}\hat g_i^T(x)E_i^T(x)+\sum_{i\in S^M_k}\hat g_i^M(x)E_i^M(x)$, where $S^T_k,S^M_k$ are top-$k$ expert sets and $\hat g$ are renormalized softmax weights. Each expert projects to rank $r$, applies nonlinearity, then projects back to $d$, offering high capacity at controlled cost. An expert decoupling loss $\mathcal L_{dis}=(\cos(y^T,y^M))^2$ enforces orthogonality between the two outputs.
    - **Design Motivation**: Tracking requires temporal consistency, so T-MoE, once pushed away from M-MoE's subspace, naturally attracts motion features; M-MoE absorbs modality-specific signals. Table 4 shows D-MoE significantly outperforms single MoE, proving the necessity of decoupling.

3. **Multimodal Router Cluster: Enforcing Modality-Specific M-MoE Routing**:

    - **Function**: Ensures M-MoE routing logits are highly similar within the same modality and dissimilar across modalities, enabling truly modality-specific expert selection.
    - **Mechanism**: Uses a batch-wise routing similarity matrix $S_{ij}=\langle g^M(x_i),g^M(x_j)\rangle$ with margin $\delta$ to construct $\mathcal L_{same}=\frac{1}{|M_{same}|}\sum_{(i,j)\in M_{same}}\max(0,(1/K+\delta)-S_{ij})$ and $\mathcal L_{diff}=\frac{1}{|M_{diff}|}\sum_{(i,j)\in M_{diff}}\max(0,S_{ij}-(\delta-1/K))$, combining into $\mathcal L_{cluster}=\mathcal L_{same}+\mathcal L_{diff}$.
    - **Design Motivation**: $\mathcal L_{dis}$ alone ensures T/M outputs are orthogonal but does not guarantee modality clustering within M-MoE; router cluster loss provides modality-level hierarchical preference, allowing certain experts to specialize in Depth, others in Thermal, etc.

### Loss & Training
The total loss is $\mathcal L=\mathcal L_{class}+\lambda_G\mathcal L_{IoU}+\lambda_{L_1}\mathcal L_{L_1}+\mathcal L_{task}+\lambda_{dis}\mathcal L_{dis}+\lambda_{cluster}\mathcal L_{cluster}+\lambda_{balance}\mathcal L_{balance}$, with default $\lambda_G\!=\!2,\lambda_{L_1}\!=\!5,\lambda_{dis}\!=\!0.1,\lambda_{cluster}\!=\!1$; $\mathcal L_{balance}$ constrains MoE load balancing. The entire network is trained end-to-end in a single stage, with no pretrain → finetune phases.

## Key Experimental Results

### Main Results

| Task / Benchmark | Metric | OneTrackerV2-L384 | SUTrack-L384 (Strong Baseline) | Notes |
|------------------|--------|-------------------|-------------------------------|-------|
| LaSOT | AUC | 76.1 | 75.2 | Long-term single-object, unified architecture still leads |
| LaSOT_ext | AUC | 55.2 | 53.6 | Significant improvement on OOD classes |
| TrackingNet | AUC / P | 88.6 / 89.0 | 87.7 / 88.7 | Large-scale online tracking |
| GOT-10k | AO | 81.3 | 81.5 | Comparable, but with unified parameters |
| UAV123 | AUC | 71.0 | 70.4 | UAV perspective |
| Model Specs | Params (M) / FLOPs (G) / FPS | 80.2 / 23.8 / 72.4 (B224) | — | DMoE adds negligible cost |

### Ablation Study

| Design | Key Observation | Interpretation |
|--------|-----------------|---------------|
| Full OneTrackerV2 | SOTA on all 5 tasks and 12 benchmarks | Single model unifies RGB + RGB+X |
| Remove Dual MoE / Use single MoE | Significant drop (Table 4: D-MoE > single MoE) | Heterogeneous objectives must be explicitly decoupled |
| Remove $\mathcal L_{dis}$ | T-MoE / M-MoE output similarity increases, performance drops | Orthogonality constraint is key for decoupling |
| Remove router cluster | M-MoE degrades to generic FFN, cross-modal generalization worsens | Modality-specific expert selection is lost |
| Missing modality inference | Performance remains stable, much better than SUTrack | Meta Merger provides modality robustness |
| Model compression | Main accuracy retained after compression | DMoE's structural redundancy allows sparsification |

### Key Findings
- T-MoE's expert selection correlates strongly with target motion intensity (Fig. 5), confirming it learns motion-related features; M-MoE experts show clear preferences for different X modalities, validating router cluster effectiveness.
- Single MoE collapses into a generative but weakly discriminative extractor when handling both tasks; after decoupling, expert groups specialize, improving both performance and robustness.
- In engineering-critical scenarios—model compression and missing modalities—OneTrackerV2's advantage grows, indicating unified + decoupled design is inherently robust.

## Highlights & Insights
- Explicitly optimizing for "feature conflict": using simple $\cos^2$ dissimilarity as an orthogonalization loss enables dual MoE specialization—a highly ROI-efficient design.
- Router cluster provides modality-level inductive bias: treating "routing similarity" as an observable and applying margin loss constrains routing more precisely than expert capacity loss.
- Meta embedding as a "modality intermediary" is inherently robust to missing modalities—a broadly applicable design pattern (transferable to RGB+X detection/segmentation/multimodal reasoning).
- Single-stage training + shared parameters + SOTA on 12 benchmarks makes this one of the most "industry-ready" multimodal tracking solutions.

## Limitations & Future Work
- Still relies on ImageNet-style ViT backbone; plug-and-play capability for modalities far from RGB (e.g., pure event, radar, point cloud) is not fully discussed.
- DMoE replaces FFN with multiple experts; while FLOPs increase is limited, memory and training time rise significantly, which may be unfriendly to small teams.
- The paper uses two manually set weights (dissimilarity and router cluster), lacking automatic scheduling (e.g., dynamically adjusting weights by task difficulty).
- Multimodal training data is still aggregated by task; cross-task positive/negative transfer is not fully explored.

## Related Work & Insights
- **vs SUTrack (Chen et al. 2025)**: SUTrack uses naive token concatenation and fails under missing modalities; OneTrackerV2 uses Meta Merger as a central hub + DMoE explicit decoupling, comprehensively surpassing it.
- **vs OneTracker (Hong et al. 2024)**: The original follows a pretrain → finetune path with task-grouped parameters; this work achieves truly unified params and single-stage training.
- **vs MoE Trackers (Tan et al. 2025, Cai et al. 2025)**: Prior works use MoE for capacity expansion or domain adaptation; this work uses MoE as a "structural container for task decoupling," a novel application in tracking.
- **vs Multimodal Fusion in OneTracker / SUTrack**: The Meta Merger here is a general module, transferable to any detection/segmentation task requiring "primary + auxiliary modality."

## Rating
- Novelty: ⭐⭐⭐⭐ Dual MoE + router cluster structurally resolve "feature conflict"—a fresh approach in tracking.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 tasks, 12 benchmarks, 4 model variants, model compression, missing modalities, multiple ablations—extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear illustrations, well-organized loss formulas, each design's rationale is evident.
- Value: ⭐⭐⭐⭐ Currently the most practically valuable unified baseline for multimodal tracking; the structural modality hub + dual MoE pattern is extensible to other multimodal vision tasks.

## Related Papers

- [\[ICLR 2026\] Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](../../ICLR2026/video_understanding/coupling_experts_and_routers_in_mixture-of-experts_via_an_auxiliary_loss.md)
- [\[ICML 2026\] RELO: Reinforcement Learning to Localize for Visual Object Tracking](relo_reinforcement_learning_to_localize_for_visual_object_tracking.md)
- [\[CVPR 2026\] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](../../CVPR2026/video_understanding/utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)
- [\[ECCV 2024\] Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective](../../ECCV2024/video_understanding/occluded_gait_recognition_with_mixture_of_experts_an_action_detection_perspectiv.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)

</div>

<!-- RELATED:END -->
