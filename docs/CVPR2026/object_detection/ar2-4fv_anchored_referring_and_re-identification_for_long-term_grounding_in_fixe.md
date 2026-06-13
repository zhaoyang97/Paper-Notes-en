---
title: >-
  [Paper Note] AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos
description: >-
  [CVPR 2026][Object Detection][Long-term referring] By exploiting the temporal invariance of background structure in fixed-view videos, the paper constructs an offline Anchor Bank and an online Anchor Map as persistent la…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Long-term referring"
  - "fixed-view video"
  - "background anchor"
  - "re-entry detection"
  - "identity re-identification"
date: 2026-05-08
content_hash: 8c4a7ec7b7c532d2
---

# AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos

**Conference**: CVPR 2026
**arXiv**: [2603.07758](https://arxiv.org/abs/2603.07758)  
**Code**: To be confirmed  
**Area**: Object Detection / Video Understanding / Language-Guided Object Grounding
**Keywords**: Long-term referring, fixed-view video, background anchor, re-entry detection, identity re-identification

## TL;DR

By exploiting the temporal invariance of background structure in fixed-view videos, the paper constructs an offline Anchor Bank and an online Anchor Map as persistent language–scene memory. Combined with an anchor-guided re-entry prior and a ReID-Gating identity verification mechanism, the system achieves robust re-capture of targets after occlusion or departure, yielding a 10.3% improvement in RCR and a 24.2% reduction in RCL.

## Background & Motivation

**Background**: Language-guided video object grounding (referring) has become a core technology for surveillance, behavior analysis, and related applications. Existing methods (MTTR, ReferFormer, OnlineRefer, etc.) are primarily designed for short-horizon scenarios, assuming targets are visible in most frames and maintaining identity consistency through inter-frame appearance propagation.

**Limitations of Prior Work**: In long-duration fixed-view videos (e.g., surveillance cameras, average length >120 s), targets frequently become occluded, exit the field of view, and re-enter. Existing methods face three critical problems:
   - **Semantic memory loss**: When a target is invisible, the semantic memory in framewise pipelines is interrupted, preventing re-association upon target re-entry.
   - **Appearance drift**: Over long time spans, illumination changes and pose variations render appearance features unreliable, causing pure ReID-based appearance matching to drift.
   - **Near-semantic interference**: Distractors with similar appearance (e.g., pedestrians wearing similar clothing) are incorrectly identified during target absence.

**Key Challenge**: Existing methods anchor semantic alignment entirely to the target's own appearance features. Once the target is invisible, the "text–target" semantic chain breaks. Yet the background structure in fixed-view videos is stable—a fact that has been entirely overlooked.

**Key Insight**: Fixed camera → invariant background layout → a set of spatial anchors can be distilled from the background → text queries are aligned to these anchors → even when the target disappears, the "text–scene" spatial memory remains persistently valid → the spatial prior enables rapid re-capture upon target re-entry.

**Core Idea**: **Compensate for the temporal variability of target appearance with the temporal invariance of background structure**—upgrading referring from "finding the target" to "localizing the spatial region corresponding to the query within a scene coordinate system."

## Method

### Overall Architecture

The input is a sequence of fixed-view video frames $\{I_t\}_{t=1}^{T}$ and a natural language query $q$; the output is per-frame bounding boxes $\{y_t\}$. The pipeline consists of two stages:

- **Offline stage**: Static background structure is extracted from the first $T_0$ frames and distilled into an Anchor Bank.
- **Online stage**: The query is aligned to the Anchor Bank to generate an Anchor Map → anchor-guided candidate filtering and fusion scoring → re-entry prior maintained during search mode → ReID-Gating for identity verification.

A key design principle is that **the target is not assumed to be visible in the first frame**; the system must locate the target from scratch.

### Key Designs

#### 1. Anchor Bank (Offline Background Structure Distillation)

- **Function**: Extracts $K$ static background region anchors $\mathcal{B} = \{(M_k, p_k, c_k)\}_{k=1}^{K}$ from the first $T_0$ frames of a fixed-view video.
- **Mechanism**: The median-brightness frame $t^\star$ is selected; a segmentation model (SAM) extracts persistent region masks $M_k$; mask-aware mean pooling is applied over the visual encoder feature map to obtain prototype vectors:
  $$p_k = \text{Norm}\left(\frac{1}{|M_k|}\sum_x M_k(x) F_{t^\star}(x)\right)$$
  with centroid $c_k = \frac{1}{|M_k|}\sum_x M_k(x) \cdot x$.
- **Design Motivation**: Under a fixed viewpoint, the background structure is invariant, so anchors need only be extracted once and are reusable indefinitely. These anchors define a **scene coordinate system** in which all subsequent operations are performed, inherently providing spatial invariance. Default settings: $K=64$, $T_0 \in [30, 120]$.

#### 2. Anchor Map (Online Language–Scene Alignment)

- **Function**: Maps the text query into scene space to produce a query-conditioned spatial heatmap.
- **Mechanism**: Lightweight alignment heads $\phi_l, \phi_v$ project the text embedding $e_q$ and anchor prototypes $p_k$ into a shared subspace; cosine similarities and softmax weights are computed:
  $$s_k = \cos(\phi_l(e_q), \phi_v(p_k)), \quad \omega_k = \frac{\exp(\tau \cdot s_k)}{\sum_j \exp(\tau \cdot s_j)}$$
  The Anchor Map is formed by weighted aggregation of anchor masks:
  $$A(x) = \sum_{k=1}^{K} \omega_k M_k(x) \in [0,1]$$
- **Design Motivation**: The Anchor Map is **fixed** for a given query—$\{M_k\}$ and $\{\omega_k\}$ do not change during inference. Even when the target is absent for hundreds of frames, the system retains a memory of which scene region the query most likely refers to. **This is the central design of the paper**: replacing transient "target appearance memory" with persistent "scene spatial memory."

#### 3. Anchor-Guided Candidate Filtering and Fusion Scoring

- **Spatial filtering**: An open-vocabulary detector $\mathcal{D}$ (GroundingDINO) generates candidate regions $\mathcal{R}_t$; only candidates whose Anchor Map response exceeds threshold $\eta$ are retained:
  $$\tilde{\mathcal{R}}_t = \{r \in \mathcal{R}_t \mid \bar{A}_{bb}(r) \geq \eta\}$$
- **Fusion scoring**: Mask-aware feature pooling $g_v(r)$ is applied to filtered candidates; the final score integrates text–visual similarity and anchor evidence:
  $$\text{Score}(r) = \lambda \cos(g_v(r), g_l(q)) + (1-\lambda) \bar{A}_m(r)$$
  When the highest score falls below threshold $\theta$, the system enters search mode (target not visible); otherwise, it proceeds to ReID-Gating verification.

#### 4. Anchor-Based Re-entry Prior

- **Function**: Maintains a spatial probability distribution during target absence to predict the most likely re-entry location.
- **Mechanism**: The re-entry prior $P_t^{re}$ is initialized as the Anchor Map $A$ and iteratively updated via EMA, Gaussian smoothing, and $\ell_1$ normalization:
  $$\tilde{P}_t^{re} = \beta(G_\sigma * \tilde{P}_{t-1}^{re}) + (1-\beta) A$$
  Candidates receive a multiplicative weight $W(r) \propto A(x) \cdot P_t^{re}(x)$, biasing scores toward high-probability re-entry regions. When the target is confirmed at anchor $k^\star$, the prior is redirected:
  $$\tilde{P}_{t+1}^{re} = \rho \cdot G_\sigma(\cdot - c_{k^\star}) + (1-\rho) A$$
- **Design Motivation**: Target re-entry is not random—in fixed-view settings, pedestrians tend to reappear from specific locations such as entrances or corridors. The re-entry prior encodes this "spatial habit," accelerating re-capture.

#### 5. ReID-Gating (Identity Verification Gate)

- **Function**: Verifies whether a candidate target's identity is consistent with the previously tracked target, preventing identity drift.
- **Mechanism**: A gating decision is made by combining three signals—appearance ReID similarity, anchor consistency, and displacement in the anchor coordinate system:
  $$G(r) = \sigma(\alpha_1 \cdot \text{sim}_{\text{ReID}}(r) + \alpha_2 \cdot \bar{A}_m(r) - \alpha_3 \cdot \hat{\Delta}(r) + b)$$
  where $\text{sim}_{\text{ReID}}(r)$ is stabilized via a momentum queue $\mathcal{Q}$, and $\hat{\Delta}(r)$ is the normalized displacement between the candidate and the last confirmed anchor. A candidate is accepted if $G(r) \geq \gamma$.
- **Design Motivation**: Pure appearance ReID is unreliable over long durations (illumination and pose changes). Incorporating anchor evidence and displacement constraints effectively performs identity verification in the scene coordinate system. The displacement penalty prevents false matches to visually similar targets at distant locations.

### Implementation Details

The system operates in a fully zero-shot manner with frozen encoders: GroundingDINO for proposals, a RexSeek-style refiner for cross-modal disambiguation, SAM for masks, a CLIP-family encoder for identity embeddings, and spaCy for query preprocessing. Key hyperparameters: $K=64$, $\tau=10$, $\lambda=0.6$, $\theta=0.4$, $\beta=0.8$, $\gamma=0.5$.

## Key Experimental Results

### AR²-4FV-Bench (New Benchmark)

The first dedicated benchmark for long-term referring and ReID in fixed-view videos:

| Dimension | Scale |
|-----------|-------|
| Number of videos | 1,684 |
| Average duration | >120 seconds |
| Scene types | School gates / lobbies / community intersections / indoor corridors, etc. |
| Annotations | Per-frame visibility + bbox + re-entry timestamps |
| Difficulty stratification | Absence duration (short/medium/long) × re-entry count (single/multiple) |
| Query types | Anchor-reference type / attribute disambiguation type / paraphrase type |

### Main Results (Re-entry Performance)

| Method | Conference | IDF1↑ | RCR↑ | RCL↓ |
|--------|-----------|-------|------|------|
| MTTR | CVPR'22 | 56.3 | 0.60 | 33.8 |
| ReferFormer | CVPR'22 | 57.9 | 0.63 | 31.2 |
| OnlineRefer | ICCV'23 | 58.6 | 0.64 | 29.9 |
| SOC | NeurIPS'23 | 58.7 | 0.64 | 30.3 |
| DsHmp | CVPR'24 | 60.4 | 0.66 | 28.6 |
| SSA | CVPR'25 | 61.5 | 0.68 | 26.5 |
| DUTrack | CVPR'25 | 62.3 | 0.69 | 25.8 |
| **AR²-4FV** | **—** | **64.8** | **0.75** | **20.1** |

AR²-4FV vs. best baseline (DUTrack): RCR +8.7% (0.69→0.75), RCL −22.1% (25.8→20.1 frames).

### Grounding Performance

| Method | mAP↑ | mIoU↑ |
|--------|------|-------|
| OnlineRefer | 46.1 | 64.2 |
| DUTrack | 46.5 | 63.7 |
| SSA | 45.2 | 64.0 |
| **AR²-4FV** | **49.2** | **66.9** |

mAP +6.7%, mIoU +4.2%; advantages become more pronounced at higher IoU thresholds (P@0.8, P@0.9).

### Ablation Study

| Anchor Bank | Anchor Map | Re-entry Prior | ReID-Gating | mIoU | mAP | IDF1 | RCR | RCL |
|:-----------:|:----------:|:--------------:|:-----------:|------|-----|------|-----|-----|
| ✓ | — | — | — | 63.2 | 45.2 | 61.2 | 0.67 | 27.1 |
| ✓ | ✓ | ✓ | — | 64.7 | 46.3 | 62.2 | 0.70 | 26.9 |
| ✓ | ✓ | — | ✓ | 63.8 | 45.5 | 61.3 | 0.68 | 21.3 |
| ✓ | ✓ | ✓ | ✓ | **66.9** | **49.2** | **64.8** | **0.75** | **20.1** |

### Key Findings

- **Anchor Map is foundational**: It provides spatial memory and is a prerequisite for all subsequent modules.
- **Re-entry Prior primarily improves RCR**: Its inclusion raises RCR from 0.67 to 0.70, enabling faster re-capture of re-entering targets.
- **ReID-Gating primarily reduces RCL**: Its inclusion dramatically lowers RCL from 27.1 to 21.3, reducing latency caused by false positives.
- **Strong complementarity across all three modules**: The full configuration achieves mIoU 66.9, far exceeding the Anchor Bank-only baseline of 63.2; each module addresses a distinct aspect of the problem.

## Highlights & Insights

- The concept of **using background as a "spatial identity card"** is particularly elegant: in fixed-view settings, "the person near the entrance" is a more reliable description than "the person in red"—the former is temporally invariant while the latter varies with illumination and pose. This shifts referring from appearance space to scene space.
- **Zero-shot operation**: All encoders are frozen; the Anchor Bank requires only a single one-time extraction; no training is needed at inference time. This dramatically lowers the barrier to deployment in real surveillance scenarios.
- **The dynamic update mechanism of re-entry prior $P^{re}$** is transferable to other tasks requiring prediction of "where an object will reappear" (e.g., obstacle reappearance prediction in robot navigation).
- **The three-signal fusion in ReID-Gating** (appearance + spatial + displacement) is more robust than pure appearance ReID and offers a generalizable approach applicable to pedestrian re-identification.

## Limitations & Future Work

1. **Strong dependence on the fixed-view assumption**: Minor camera jitter or PTZ motion will invalidate the Anchor Bank. A background registration module could be introduced to accommodate quasi-fixed viewpoints.
2. **Fixed anchor count $K=64$**: Scene complexity varies considerably; adaptive anchor count selection may be more appropriate.
3. **Linear time complexity with large constants**: Each frame requires multiple forward passes through GroundingDINO, SAM, and CLIP, raising concerns about real-time feasibility.
4. **No handling of cross-scenario appearance changes**: The authors explicitly exclude cross-session ReID scenarios such as clothing changes, limiting the applicable scope.
5. **Re-entry prior assumes spatial regularity**: In highly random behavioral scenarios (e.g., animal behavior analysis), the background-structure-based re-entry assumption may not hold.

## Related Work & Insights

- **vs. ReferFormer/MTTR**: These methods assume continuous target visibility and propagate semantics via inter-frame Transformers; AR²-4FV makes no such assumption, replacing inter-frame propagation with scene structure.
- **vs. OVTrack**: OVTrack performs open-vocabulary tracking via text retrieval but still relies on appearance continuity; AR²-4FV substitutes spatial priors when appearance becomes unreliable.
- **vs. ByteTrack/BoT-SORT**: These MOT methods target short-horizon scenarios and lack language guidance or long-term re-entry handling.
- **vs. background modeling methods (MOG/ViBe)**: AR²-4FV draws on the "foreground–background separation" idea from classical background modeling but elevates it to semantic-level "anchor–query alignment."

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using background structure for language-guided referring is novel, though individual sub-modules are combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ A new benchmark with comprehensive ablations is provided, but cross-dataset generalization and inference speed analysis are absent.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear; algorithm pseudocode and equations are complete; some tables have formatting issues.
- Value: ⭐⭐⭐⭐ Practical application value in fixed-view surveillance scenarios; the zero-shot design lowers deployment barriers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification](sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)
- [\[CVPR 2026\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching.md)
- [\[CVPR 2026\] HeROD: Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection](herod_heuristic_inspired_reasoning_data_efficient_rod.md)
- [\[CVPR 2026\] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training](pet-dino_unifying_visual_cues_into_grounding_dino_with_prompt-enriched_training.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)

</div>

<!-- RELATED:END -->
