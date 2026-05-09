---
title: >-
  [Paper Note] x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space
description: >-
  [ICLR 2026][Autonomous Driving][Optical Flow Estimation] x²-Fusion introduces Event Edge Space — the first edge-based isomorphic latent space — that unifies image, LiDAR, and event camera features into a shared edge-centric representation. Combined with reliability-aware adaptive fusion and cross-dimension contrastive learning, it achieves state-of-the-art joint 2D optical flow and 3D scene flow estimation under both standard and degraded conditions.
tags:
  - ICLR 2026
  - Autonomous Driving
  - Optical Flow Estimation
  - Scene Flow Estimation
  - Event Camera
  - Multimodal Fusion
  - Edge Space
date: 2026-05-08
content_hash: a917c0d75b82bba5
---

# x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space

**Conference**: ICLR 2026
**arXiv**: [2603.16671](https://arxiv.org/abs/2603.16671)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: Optical Flow Estimation, Scene Flow Estimation, Event Camera, Multimodal Fusion, Edge Space

## TL;DR
x²-Fusion introduces Event Edge Space — the first edge-based isomorphic latent space — that unifies image, LiDAR, and event camera features into a shared edge-centric representation. Combined with reliability-aware adaptive fusion and cross-dimension contrastive learning, it achieves state-of-the-art joint 2D optical flow and 3D scene flow estimation under both standard and degraded conditions.

## Background & Motivation
1. **Background**: Optical flow and scene flow estimation are fundamental to dynamic scene understanding, with broad applications in autonomous driving, tracking, and 3D reconstruction. Recent methods fusing images, LiDAR, and event cameras have surpassed single-modality baselines.
2. **Limitations of Prior Work**:
    - **High Complexity**: Existing methods retain each modality in its native feature space without a shared channel-level foundation, requiring multiple pairwise alignments — RPEFlow employs staged fusion blocks, CMX uses pairwise correction/attention units, and VisMoFlow relies on multiple hand-crafted physical spaces — resulting in unwieldy, difficult-to-train, and hard-to-scale models.
    - **Information Erosion**: Processing features in heterogeneous spaces defers fusion to late stages, by which point modality-specific distortions are difficult to correct through cross-modal interaction.
    - **High Fragility**: Without a common representational foundation, modalities cannot provide stable priors for one another; under degraded conditions such as extreme exposure, LiDAR sparsity, or motion blur, alignment itself breaks down.
3. **Key Challenge**: The representational heterogeneity among images (2D grids), LiDAR (point clouds), and events (asynchronous streams) makes simple, robust, and efficient cross-modal interaction fundamentally difficult.
4. **Goal**: To achieve effective image–LiDAR–event fusion within a unified isomorphic space while jointly estimating 2D optical flow and 3D scene flow.
5. **Key Insight**: Leveraging the spatiotemporal edge signals naturally provided by event cameras as anchors to construct an edge-centric isomorphic latent space.
6. **Core Idea**: Edges encode modality-agnostic structural information (object boundaries and scene discontinuities). Event cameras are intrinsically spatiotemporal edge detectors (firing at motion edges), sharing 2D pixel coordinates with images and exhibiting a sparse sampling structure analogous to LiDAR. This dual correspondence makes events an ideal anchor for an isomorphic space.

## Method

### Overall Architecture
Image + Event + LiDAR → Pre-trained and frozen event edge encoder → Align image and LiDAR encoders to Event Edge Space using event embeddings as edge prototypes → Reliability-aware adaptive fusion → Cross-dimension contrastive learning → Joint output of 2D optical flow and 3D scene flow.

### Key Designs
1. **Event Edge Space Construction**:
    - **Event Edge Encoder Pre-training**: Event streams are voxelized and fed into a sparse 3D CNN to produce a multi-scale event feature pyramid $\{F_s^E\}$.
    - Event edge intensity is defined as: $e^E(x,y) = \tilde{A}^E(x,y)(1 - \tilde{\sigma}_t(x,y)) \in [0,1]$, where $\tilde{A}^E$ is the normalized event activity and $\tilde{\sigma}_t$ is the normalized temporal variance.
    - Self-supervised pre-training predicts future edge intensity from past events, with loss $\mathcal{L}_{\text{edge}}^E = \sum_s \lambda_s \|g_s(F_s^{E,\text{past}}) - e_s^{E,\text{future}}\|_1$.
    - **Design Motivation**: Explicitly distills motion-aware edge features to serve as fixed edge prototypes guiding the alignment of other modalities.

    - **Image–LiDAR Alignment**:
    - The event encoder is frozen; its features $Z_s^E \equiv F_s^E$ serve as edge prototypes.
    - Projection heads $Z_s^I = h_s^I(F_s^I)$ and $Z_s^L = h_s^L(F_s^L)$ map image and LiDAR features to the same dimension $C_s$.
    - Edge-anchored symmetric regularization: $D_s^{2/3D}(p) = \sum_{(m,n) \in \{I,E,L\}} \|Z_s^m(p) - Z_s^n(p)\|_1$.
    - The alignment loss is weighted by the event edge map: $\mathcal{L}_{\text{align}}^{2/3D} = \sum_s \sum_p e_s^{E}(p) D_s^{2/3D}(p)$.
    - Gradients are stopped at $Z_s^E$ to keep edge prototypes fixed.
    - **Design Motivation**: Fixed edge prototypes provide stable alignment anchors, while symmetric regularization pulls image and LiDAR features toward them.

2. **Reliability-aware Adaptive Fusion**:
    - **Global Reliability Score**: Per-modality reliability is estimated via spatiotemporal decomposition.
      - Temporal stream $\mathcal{T}(\hat{Z}) = \sigma(\mathbb{L}(\Delta_t(\text{Conv}(\hat{Z}))))$: captures fine-grained temporal variation.
      - Spatial stream $\mathcal{S}(\hat{Z}) = \|\nabla(\text{DConv}(\hat{Z}))\|_2$: encodes spatial structure.
      - Global reliability score $\omega_m = \text{softmax}_m((\mathcal{T} \otimes \mathcal{S})\hat{Z})$.
    - **Local Attention Mechanism**: $\mathcal{A}_m(x) = \text{softmax}((\mathcal{H} \oplus \mathcal{P} \oplus \mathcal{G})\tilde{Z})_m$, utilizing high-pass filtering, average pooling, and grouped convolution.
    - Fused features: $F_{\text{fused}}(x) = \sum_m \frac{\omega_m \mathcal{A}_m(x)}{\sum_n \omega_n \mathcal{A}_n(x)} Z_m(x)$.
    - A cross-attention Transformer further enhances multi-modal interaction.
    - **Design Motivation**: The isomorphic space enables fusion through lightweight weighted summation and unified cross-attention, replacing stacked modality-specific modules.

3. **Cross-dimension Contrastive Learning (CCL)**:
    - **Cross-temporal contrast (pull)**: 3D features are projected into 2D space; temporal motion vectors $M^{2/3D}$ are computed and cosine similarity loss encourages 2D–3D motion consistency: $\mathcal{L}_{\text{pull}} = 1 - \frac{\langle \phi(M^{2D}), \psi(M_{\text{proj}}^{3D}) \rangle}{\|\phi(M^{2D})\|_2 \cdot \|\psi(M_{\text{proj}}^{3D})\|_2}$.
    - **Cross-task contrast (push)**: 2D/3D features are encoded into latent distributions via variational encoding; mutual information is minimized to preserve intra-frame complementarity: $\mathcal{L}_{\text{push}} = \frac{1}{2}\sum_t \text{BCE}(\sigma(\mathbf{z}_t^{2D}), \sigma(\mathbf{z}_t^{3D}))$.
    - **Design Motivation**: Pull enforces inter-frame motion consistency (2D and 3D perceive the same motion); push enforces intra-frame complementarity (2D and 3D each capture distinct useful information).

### Loss & Training
- Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{align}} \mathcal{L}_{\text{align}} + \lambda_{\text{contra}} \mathcal{L}_{\text{contra}}$
- Task loss follows PWC-style coarse-to-fine supervision.
- The event edge encoder is pre-trained with $\mathcal{L}_{\text{edge}}^E$ and then frozen.
- Implementation: PyTorch; 4× RTX A6000 GPUs; Adam optimizer; lr=$10^{-4}$; weight decay=$10^{-6}$; batch size 8; MultiStepLR scheduler; synchronized BN and mixed-precision training.

## Key Experimental Results

### Main Results

| Method | Modality | EKubric EPE2D↓ | EKubric EPE3D↓ | DSEC EPE2D↓ | DSEC EPE3D↓ |
|--------|----------|---------------|---------------|-------------|-------------|
| RAFT | Img | 0.838 | - | 0.586 | - |
| FlowFormer | Img | 0.702 | - | 0.567 | - |
| CamLiFlow | Img+PC | 0.770 | 0.035 | 0.399 | 0.129 |
| RPEFlow | Img+PC+EV | 0.439 | 0.027 | 0.326 | 0.103 |
| **x²-Fusion** | **Img+PC+EV** | **0.430** | **0.024** | **0.322** | **0.094** |

### Performance Under Degraded Conditions (vs. RPEFlow)

| Degradation Type | EPE2D Improvement | EPE3D Improvement | Notes |
|------------------|------------------|------------------|-------|
| Under-exposure (EKubric) | ↓2.520 (3.663→1.143) | ↓0.010 | Most significant gain |
| Over-exposure (EKubric) | ↓1.807 (2.801→0.994) | ↓0.007 | |
| LiDAR Sparsity (DSEC) | ↓0.162 | ↓0.009 | |
| LiDAR Drift (DSEC) | ↓0.642 (1.051→0.409) | ↓0.172 | ACC.10 improved by 26.41% |

### Ablation Study

| Configuration | EPE2D↓ | EPE3D↓ | Notes |
|---------------|--------|--------|-------|
| w/o Event Edge Space | 0.491 | 0.146 | No isomorphic space |
| w/o Edge-anchored Regularization | 0.393 | 0.119 | Space present but no regularization |
| w/o Event Edge Encoder Pre-training | 0.378 | 0.114 | Event encoder not pre-trained |
| **Full Model** | **0.322** | **0.094** | All components |
| Independent 2D+3D Tasks | 0.404 | 0.119 | No joint training |
| Joint 2D&3D Tasks | 0.386 | 0.113 | No CCL |
| Joint + CCL | 0.325 | 0.103 | Cross-dimension contrastive learning effective |

### Key Findings
- Event Edge Space is the most critical component; its removal increases EPE2D by 52.5% and EPE3D by 55.3%.
- Advantages are more pronounced under degraded conditions, particularly under under-exposure where EPE2D drops from 3.663 to 1.143 (68.8% reduction).
- Joint 2D+3D estimation outperforms independent estimation; CCL yields further significant gains.
- t-SNE visualization confirms that edge-anchored regularization effectively tightens cross-modal feature clustering.
- Three-modality fusion provides greater advantages over dual- and single-modality alternatives under degraded conditions.

## Highlights & Insights
- The core insight behind Event Edge Space is elegant: edges constitute a modality-agnostic structural language, event cameras are natural spatiotemporal edge detectors, and anchoring an isomorphic space to events reframes fusion as a representation unification problem.
- Freezing the event encoder as a stable prototype cleverly reduces the alignment problem to a one-way mapping.
- The global–local dual-layer weighting in reliability-aware adaptive fusion is particularly effective under degraded conditions.
- The pull (consistency) + push (complementarity) design in cross-dimension contrastive learning exhibits strong theoretical elegance.
- With only 8.2M parameters — fewer than RPEFlow's 9.8M — the model achieves superior performance.

## Limitations & Future Work
- Validation is currently limited to the image–LiDAR–event three-modality setting; although EES is claimed to be modality-agnostic, extension to other sensors remains to be verified.
- The real-world dataset (DSEC) covers limited scenario diversity, primarily urban driving.
- Pre-training the event edge encoder requires an additional training stage.
- Degraded-condition simulations rely on synthetic noise models; the gap with real-world degradation warrants further investigation.
- The EES framework could potentially be generalized to text–image–video fusion and broader cross-domain feature alignment tasks.

## Related Work & Insights
- RPEFlow's staged fusion and VisMoFlow's multi-physical-space design are the primary baselines for comparison.
- CamLiFlow's bidirectional camera–LiDAR fusion provides a foundation for interpolation strategies.
- The edge properties of event cameras have been exploited in video frame interpolation, stereo matching, and related tasks.
- Contrastive learning in multimodal fusion is increasingly mature, but this work is the first to apply it to joint 2D–3D flow estimation.
- The Event Edge Space concept is generalizable to other tasks requiring unified representations of heterogeneous sensors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](../../CVPR2026/autonomous_driving/x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[ICLR 2026\] SEAL: Segment Any Events with Language](segment_any_events_with_language.md)
- [\[ICLR 2026\] SiMO: Single-Modality-Operable Multimodal Collaborative Perception](simo_single-modality-operable_multimodal_collaborative_perceptio.md)
- [\[AAAI 2026\] MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation](../../AAAI2026/autonomous_driving/mambaseg_harnessing_mamba_for_accurate_and_efficient_image-e.md)
- [\[ICLR 2026\] NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](nemo-map_neural_implicit_flow_fields_for_spatio-temporal_motion_mapping.md)

<!-- RELATED:END -->
