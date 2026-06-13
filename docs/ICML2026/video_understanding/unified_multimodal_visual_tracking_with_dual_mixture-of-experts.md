---
title: >-
  [Paper Note] Unified Multimodal Visual Tracking with Dual Mixture-of-Experts
description: >-
  [ICML 2026][Video Understanding][Visual Tracking] OneTrackerV2 unifies five tracking tasks (RGB / RGB+D / RGB+T / RGB+E / RGB+N) into a single network for end-to-end training. It utilizes a Meta Merger for modality fusio…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Visual Tracking"
  - "RGB+X"
  - "Mixture-of-Experts"
  - "Feature Decoupling"
  - "Robustness to Missing Modalities"
date: 2026-05-08
content_hash: 760f5366aa6c8a62
---

# Unified Multimodal Visual Tracking with Dual Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2605.03716](https://arxiv.org/abs/2605.03716)  
**Code**: None  
**Area**: Video Understanding / Multimodal Visual Tracking / Mixture-of-Experts  
**Keywords**: Visual Tracking, RGB+X, Mixture-of-Experts, Feature Decoupling, Robustness to Missing Modalities  

## TL;DR
OneTrackerV2 unifies five tracking tasks (RGB / RGB+D / RGB+T / RGB+E / RGB+N) into a single network for end-to-end training. It utilizes a Meta Merger for modality fusion and a Dual MoE to explicitly decouple heterogeneous features into "spatial-temporal matching" (T-MoE) and "modality fusion" (M-MoE), using a dissimilarity loss and router clustering to prevent subspace collapse.

## Background & Motivation
**Background**: Visual object tracking is categorized into RGB and RGB+X (X=Depth/Thermal/Event/Language) based on input modalities. Current approaches follow three main routes: (a) independent architecture design and training for each X task; (b) fine-tuning pre-trained RGB trackers for adaptation (e.g., OneTracker); (c) preliminary unified models that concatenate multimodal tokens within a shared backbone (e.g., SUTrack).

**Limitations of Prior Work**: (1) Multi-step training (pre-train → fine-tune) often converges to sub-optimal solutions; (2) Lack of unified architectures necessitates manual task-branch design; (3) Parameters in shared architectures remain grouped by task rather than being truly "unified"; (4) Performance collapses if a modality is missing during inference; (5) Feature interference — simple token concatenation forces the same parameter space to learn both spatial-temporal motion matching and modality-specific patterns simultaneously.

**Key Challenge**: Tracking essentially requires two distinct capabilities: spatial-temporal matching (inter-frame motion between template ↔ search) and modality fusion (complementary cues between RGB ↔ X). Forcing both into a single backbone or a single MoE leads to zero-sum parameter competition.

**Goal**: (1) Single-step end-to-end training with shared parameters and architecture; (2) Modality-agnostic and missing-modality-robust "meta embedding" for fusion; (3) Structural decoupling to resolve feature conflicts between matching and fusion; (4) Scalable capacity without exploding inference costs.

**Key Insight**: Utilize a learnable meta embedding as a central multimodal hub and introduce Dual MoE to assign spatial-temporal and modality tasks to two sets of experts, enforced by an explicit decoupling loss for orthogonality.

**Core Idea**: Meta Merger + Dual MoE = One network, one training session, and one set of parameters to handle five tracking tasks, maintaining robustness to missing modalities and model compression.

## Method

### Overall Architecture
The system inputs template and search regions, each containing RGB and an X modality frame (for RGB-only tasks, RGB is used as X). $F_{rgb}$ and $F_x$ are obtained via shared patch embeddings. The Meta Merger uses a learnable meta embedding $F_{meta}$ to perform spatial/channel attention and centralized convolution for fusion, resulting in a modality-agnostic token sequence. This sequence is fed into a Vision Transformer backbone where the FFN in each block is replaced by a Dual MoE: tokens pass through a shared expert, T-MoE (top-$k$), and M-MoE (top-$k$) simultaneously. Finally, a detection head (SUTrack-style) outputs the bounding box via classification, IoU, and L1 heads. Four versions (B224/B384/L224/L384) are provided, with parameters ranging from 80M to 271M and inference speeds of 23.4–72.4 FPS.

### Key Designs

1.  **Meta Merger: Modality-Agnostic Central Hub**:
    - **Function**: Compresses heterogeneous RGB and X features into a unified space while providing inherent robustness to missing modalities.
    - **Mechanism**: First, $F_{rgb}$ and $F_x$ are enhanced using $W^{spatial}=\sigma(\mathrm{Conv}(F^{avg})+\mathrm{Conv}(F^{max}))$ and $W^{channel}=\sigma(\mathrm{Linear}(F^{avg})+\mathrm{Linear}(F^{max}))$. Then, a learnable $F_{meta}$ acts as an intermediary via $F_{meta}'=\mathrm{Conv}(\mathrm{Conv}(F_{meta}+F'_{rgb})+\mathrm{Conv}(F_{meta}+F'_x)+F_{meta})$ to output globally aligned tokens. If X is missing, it degrades to interacting only with RGB without requiring pipeline modifications.
    - **Design Motivation**: Compared to token concatenation in SUTrack, meta embedding avoids doubled computation from multiple branches and uses a global variable as a "modal translator" to adapt naturally to any modality combination.

2.  **Dual MoE: Explicit Decoupling of T-MoE and M-MoE**:
    - **Function**: Separates spatial-temporal matching and modality fusion into two sets of independent experts to avoid heterogeneous objectives within the same parameter space.
    - **Mechanism**: For each token $x$, the DMoE outputs $y=E_{shared}(x)+\sum_{i\in S^T_k}\hat g_i^T(x)E_i^T(x)+\sum_{i\in S^M_k}\hat g_i^M(x)E_i^M(x)$, where $S^T_k$ and $S^M_k$ are top-$k$ expert sets and $\hat g$ represents renormalized softmax weights. Each expert is implemented as a "projection to rank $r$ → non-linearity → projection to $d$" bottleneck. An expert decoupling loss $\mathcal L_{dis}=(\cos(y^T,y^M))^2$ forces the outputs to be orthogonal.
    - **Design Motivation**: Tracking requires temporal consistency; thus, once T-MoE is pushed away from the M-MoE subspace, it naturally gravitates toward motion features, while M-MoE absorbs modality-specific signals.

3.  **Multimodal Router Cluster: Specific Modal Allocation for M-MoE**:
    - **Function**: Ensures M-MoE routing logits maintain high similarity for samples of the same modality and low similarity for different modalities, enabling modality-specific expert selection strategies.
    - **Mechanism**: Using an intra-batch routing similarity matrix $S_{ij}=\langle g^M(x_i),g^M(x_j)\rangle$ and a margin $\delta$, the model constructs $\mathcal L_{same}=\frac{1}{|M_{same}|}\sum_{(i,j)\in M_{same}}\max(0,(1/K+\delta)-S_{ij})$ and $\mathcal L_{diff}=\frac{1}{|M_{diff}|}\sum_{(i,j)\in M_{diff}}\max(0,S_{ij}-(\delta-1/K))$. The final clustering loss is $\mathcal L_{cluster}=\mathcal L_{same}+\mathcal L_{diff}$.
    - **Design Motivation**: While $\mathcal L_{dis}$ ensures T/M orthogonality, it does not guarantee modality-based clustering within M-MoE. The router cluster loss provides hierarchical modality-level preferences, specializing certain experts for Depth, Thermal, etc.

### Loss & Training
The total loss is defined as $\mathcal L=\mathcal L_{class}+\lambda_G\mathcal L_{IoU}+\lambda_{L_1}\mathcal L_{L_1}+\mathcal L_{task}+\lambda_{dis}\mathcal L_{dis}+\lambda_{cluster}\mathcal L_{cluster}+\lambda_{balance}\mathcal L_{balance}$, with default weights $\lambda_G\!=\!2,\lambda_{L_1}\!=\!5,\lambda_{dis}\!=\!0.1,\lambda_{cluster}\!=\!1$. $\mathcal L_{balance}$ ensures MoE load balancing. The entire network is trained end-to-end in a single stage.

## Key Experimental Results

### Main Results

| Task / Benchmark | Metric | OneTrackerV2-L384 (Ours) | SUTrack-L384 (Prev. SOTA) | Description |
| :--- | :--- | :--- | :--- | :--- |
| LaSOT | AUC | 76.1 | 75.2 | Long-term single-target tracking |
| LaSOT_ext | AUC | 55.2 | 53.6 | Significant gain on OOD classes |
| TrackingNet | AUC / P | 88.6 / 89.0 | 87.7 / 88.7 | Large-scale online tracking |
| GOT-10k | AO | 81.3 | 81.5 | Comparable but unified |
| UAV123 | AUC | 71.0 | 70.4 | UAV perspective |
| Model Specs | Params(M)/FLOPs(G)/FPS | 80.2 / 23.8 / 72.4 (B224) | — | DMoE adds minimal cost |

### Ablation Study

| Design | Key Findings | Insight |
| :--- | :--- | :--- |
| Full OneTrackerV2 | SOTA across 5 tasks and 12 benchmarks | Single model can unify RGB + RGB+X |
| Remove Dual MoE / Use single MoE | Significant performance drop | Heterogeneous goals require explicit decoupling |
| Remove $\mathcal L_{dis}$ | T/M similarity increases, performance drops | Orthogonal constraint is key to decoupling |
| Remove router cluster | M-MoE degrades to general FFN | Loss of modality-specific expert selection |
| Missing Modality Inference | Performance remains stable | Meta Merger provides modality robustness |
| Model Compression | Accuracy largely preserved after compression | DMoE redundancy allows for sparsification |

### Key Findings
- The expert selection patterns of T-MoE correlate highly with target motion intensity, proving it learns motion-related features. In M-MoE, different experts show clear preferences for specific X modalities, validating the router cluster.
- A single MoE attempting both tasks collapses into a generative but weakly discriminative feature extractor. Decoupled experts allow for improved performance and robustness.
- OneTrackerV2 shows a significantly wider margin of advantage in engineering-critical scenarios like model compression and missing modalities.

## Highlights & Insights
- **Explicit Optimization of "Feature Conflict"**: Using a simple $\cos^2$ dissimilarity loss to specialize the Dual MoE is a high-ROI design choice.
- **Modality-Level Inductive Bias**: Imposing margin losses on "routing similarity" as an observable variable constrains routing behavior more precisely than standard expert capacity losses.
- **Meta-Embedding as an Intermediary**: This inherently robust hub for missing modalities is a widely applicable design pattern suitable for detection, segmentation, or multimodal reasoning.
- **Practicality**: Single-stage training with shared parameters across 12 benchmarks makes it one of the most industrially viable multimodal tracking solutions.

## Limitations & Future Work
- Still relies on ImageNet-style ViT backbones; the "plug-and-play" capability for modalities with large domain gaps (e.g., pure Event, Radar, Point Cloud) requires further discussion.
- Replacing FFN with multiple experts increases VRAM and training time, which may be less accessible for smaller teams.
- Manual weighting for dissimilarity and router clusters lacks automatic scheduling (e.g., dynamic adjustment based on task difficulty).
- Cross-task positive/negative transfer during aggregate multimodal training has not been fully explored.

## Related Work & Insights
- **vs. SUTrack (Chen et al. 2025)**: SUTrack uses naive token concatenation and collapses in missing modality scenarios; Ours uses Meta Merger and DMoE for explicit decoupling.
- **vs. OneTracker (Hong et al. 2024)**: The original used pre-train → fine-tune paths and task-grouped parameters; this work achieves truly unified parameters in a single training session.
- **vs. MoE Trackers (Tan et al. 2025, Cai et al. 2025)**: Previous works used MoE for capacity expansion or domain adaptation; this work uses MoE as a "structural container for task decoupling."

## Rating
- **Novelty**: ⭐⭐⭐⭐ Dual MoE + router cluster provides a structural solution to feature conflict.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High coverage across tasks, benchmarks, model scales, and robustness tests.
- **Writing Quality**: ⭐⭐⭐⭐ Clear diagrams and organized loss formulations.
- **Value**: ⭐⭐⭐⭐ A highly practical unified baseline for multimodal tracking with transferable design patterns.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RELO: Reinforcement Learning to Localize for Visual Object Tracking](relo_reinforcement_learning_to_localize_for_visual_object_tracking.md)
- [\[CVPR 2026\] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](../../CVPR2026/video_understanding/utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)
- [\[ICML 2026\] AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes](avtrack_audio-visual_tracking_in_human-centric_complex_scenes.md)
- [\[CVPR 2026\] Drift-Resilient Temporal Priors for Visual Tracking](../../CVPR2026/video_understanding/drift-resilient_temporal_priors_for_visual_tracking.md)
- [\[CVPR 2026\] SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking](../../CVPR2026/video_understanding/spiketrack_a_spike-driven_framework_for_efficient_visual_tracking.md)

</div>

<!-- RELATED:END -->
