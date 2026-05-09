---
title: >-
  [Paper Note] ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions
description: >-
  [ICCV 2025][Autonomous Driving][3D Semantic Occupancy Prediction] ALOcc is proposed as a framework that achieves state-of-the-art performance on multiple 3D semantic occupancy and occupancy flow prediction benchmarks through three innovations: an occlusion-aware adaptive lifting mechanism, a semantic prototype-based occupancy head, and a BEV cost volume-based flow prediction module, while offering multiple model variants ranging from real-time to high-accuracy configurations.
tags:
  - ICCV 2025
  - Autonomous Driving
  - 3D Semantic Occupancy Prediction
  - Occupancy Flow Prediction
  - View Transformation
  - BEV Perception
  - Semantic Prototypes
date: 2026-05-08
content_hash: 180f6085356d5214
---

# ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions

**Conference**: ICCV 2025
**arXiv**: [2411.07725](https://arxiv.org/abs/2411.07725)
**Code**: [https://github.com/cdb342/ALOcc](https://github.com/cdb342/ALOcc)
**Area**: 3D Scene Understanding / Autonomous Driving
**Keywords**: 3D Semantic Occupancy Prediction, Occupancy Flow Prediction, View Transformation, BEV Perception, Semantic Prototypes

## TL;DR
ALOcc is proposed as a framework that achieves state-of-the-art performance on multiple 3D semantic occupancy and occupancy flow prediction benchmarks through three innovations: an occlusion-aware adaptive lifting mechanism, a semantic prototype-based occupancy head, and a BEV cost volume-based flow prediction module, while offering multiple model variants ranging from real-time to high-accuracy configurations.

## Background & Motivation
3D semantic occupancy prediction is a core task in autonomous driving scene understanding, aiming to transform multi-camera images into dense voxel grid representations where each voxel encodes occupancy status, semantic label, and motion flow. Compared to traditional bounding box representations, occupancy grids provide more complete and fine-grained scene descriptions.

Existing methods face three key challenges:

**Limitations of view transformation**: LSS-based depth methods suffer from inductive biases in depth priors, are prone to premature convergence, and handle occlusions poorly; cross-attention-based methods avoid these issues but lack explicit geometric grounding, resulting in suboptimal performance.

**Severe impact of class imbalance**: Empty voxels dominate the scene, and the semantic class distribution follows a long-tail pattern, causing the model to underlearn rare categories.

**Conflict in joint semantic-motion prediction**: Simultaneously encoding static semantics and dynamic motion information imposes contradictory demands on feature representations, increasing the representational burden.

ALOcc addresses each stage of the pipeline with targeted improvements: an occlusion-aware mechanism in the view transformation stage, semantic prototype enhancement in the decoding stage, and BEV cost volume in the flow prediction stage to decouple semantics and motion.

## Method

### Overall Architecture
ALOcc follows the classical paradigm of "2D feature extraction → view transformation → 3D encoding → task decoding." Multi-view images are processed by a backbone to extract 2D features $\mathbf{f}_I$, which are transformed into 3D space via the adaptive lifting mechanism to obtain $\mathbf{f}_{Lift}$. Together with historical frame features, these are encoded by a 3D encoder, and pluggable task heads predict semantic occupancy and motion flow respectively. The overall architecture is purely convolutional, with no Transformer components.

### Key Designs

1. **Occlusion-Aware Adaptive Lifting**:

    - Function: Improves the 2D-to-3D view transformation so that features are projected not only onto visible surfaces but also propagated into occluded regions.
    - Mechanism: Trilinear interpolation first replaces hard rounding to achieve differentiable soft splatting. The key innovation is a probability transfer matrix designed to propagate information from visible surfaces to occluded regions. For intra-object occlusion, depth probabilities are converted to occlusion-length probabilities via Bayesian conditional probability $P(o_{ol}^j) = \sum_{i=1}^D P(o_{ol}^j|o_d^i) \cdot P(o_d^i)$; for inter-object occlusion, an MLP predicts offsets and weights to propagate probabilities to neighboring points.
    - Depth denoising for training stability: Ground-truth and predicted depth are blended via cosine annealing as $P(o_d) = \frac{1}{2}[(1+\cos(\frac{\pi e}{E})) \cdot P_{gt} + (1-\cos(\frac{\pi e}{E})) \cdot P_{pred}]$, relying on ground-truth depth in early training to prevent premature convergence.
    - Design Motivation: Standard LSS depth estimation follows a $\delta$ distribution, concentrating most weight on surface points and leaving occluded regions severely underrepresented.

2. **Semantic Prototype-Based Occupancy Head**:

    - Function: Bridges the 2D and 3D feature domains through shared semantic prototypes to enhance semantic consistency.
    - Mechanism: Per-class prototypes are randomly initialized and serve as class weights for both 2D and 3D loss computation. At inference, prediction is computed as $\hat{\mathbf{o}}_v = \arg\max_c(\text{MLP}(P_c) \cdot \mathbf{f}_v)$.
    - Conditional training strategy: Loss is computed only for classes present in each GT sample, avoiding wasted training on absent categories.
    - Uncertainty-guided sampling: Predicted logits are used as a model uncertainty measure; combined with class priors, they form a sampling distribution from which $K$ hard voxels are sampled to focus training on low-confidence regions and minority classes.
    - Loss function: $\mathcal{L}_{3D} = \alpha\mathcal{L}_{Dice} + \beta\mathcal{L}_{BCE}$, supplemented by a 2D projection loss $\mathcal{L}_{2D}$.
    - Design Motivation: Severe class imbalance in scenes makes direct voxel-to-prototype similarity computation ineffective.

3. **BEV Cost Volume-Based Flow Head**:

    - Function: Predicts motion flow by constructing explicit cross-frame correspondences, decoupling the representational burden of semantics and motion.
    - Mechanism: Volumetric features are collapsed to the BEV plane and downsampled; ego-motion warps the previous frame's BEV to the current coordinate system; cosine similarity is computed within a local search window to build the cost volume $\mathrm{cv}(\mathbf{f}_v^{(t)};k) = \frac{\hat{\mathbf{f}}_v^{(t)} \cdot \mathrm{warp}(\hat{\mathbf{f}}_v^{(t-1)}(\Delta p_k))}{\|\hat{\mathbf{f}}_v^{(t)}\|_2 \cdot \|\mathrm{warp}(\hat{\mathbf{f}}_v^{(t-1)}(\Delta p_k))\|_2}$.
    - Hybrid classification-regression: The continuous flow space is discretized into bins; a probability distribution over bins is predicted and the continuous flow is recovered as $\hat{\mathbf{o}}_f = \sum_{n=1}^{N_b} p_b^n \cdot \mathbf{b}^n$.
    - Flow loss: $\mathcal{L}_{flow} = \mathcal{L}_{flow}^{reg} + \mathcal{L}_{flow}^{cls}$, comprising an L2 loss for magnitude accuracy, cosine similarity for directional accuracy, and classification cross-entropy.
    - Design Motivation: Conventional methods predict flow from single-frame features, requiring features to simultaneously encode static semantics and dynamic motion, creating a representational bottleneck.

### Loss & Training
- Total semantic occupancy loss: $\mathcal{L}_{sem} = \mathcal{L}_{3D} + \mathcal{L}_{2D} + \mathcal{L}_{depth}$
- Joint semantic-flow prediction: $\mathcal{L}_{sem-flow} = \mathcal{L}_{3D} + \mathcal{L}_{2D} + \mathcal{L}_{depth} + \mathcal{L}_{flow}$
- Training configuration: AdamW optimizer, learning rate $2\times10^{-4}$, batch size 16; 12 epochs for the semantic task, 18 epochs for flow prediction.
- Three model variants are provided: ALOcc-2D-mini (real-time), ALOcc-2D (default), and ALOcc-3D (high-accuracy).

## Key Experimental Results

### Main Results (Occ3D Semantic Occupancy Prediction, with visibility mask)

| Method | Backbone | Input | mIoU_D | mIoU | FPS |
|------|------|------|--------|------|-----|
| FB-Occ | R50 | C | 34.2 | 39.8 | 10.3 |
| COTR | R50 | C | 38.6 | 44.5 | 0.5 |
| FlashOCC | R50 | C | 24.7 | 32.0 | 29.6 |
| **ALOcc-2D-mini** | R50 | C | 35.4 | 41.4 | **30.5** |
| **ALOcc-2D** | R50 | C | **38.7** | **44.8** | 8.2 |
| **ALOcc-3D** | R50 | C | **39.3** | **45.5** | 6.0 |
| FusionOcc | Swin-B | C+L | 53.1 | 56.6 | - |
| **ALOcc-3D** | Swin-B | C+D | **57.8** | **60.0** | 1.5 |

### Ablation Study

| Configuration | mIoU_D | mIoU | Note |
|------|--------|------|------|
| Full model ALOcc-2D-40 | 38.5 | 44.5 | Baseline |
| w/o Adaptive Lifting (AL) | 37.5 | 43.5 | mIoU_D drops by 1.0 |
| w/o Semantic Prototype (SP) | 36.0 | 42.1 | mIoU drops by 2.4, larger impact |
| w/o both AL and SP | 34.9 | 41.2 | Total drop of 3.3 |

| Flow Prediction Ablation | Occ Score | mAVE | RayIoU |
|-----------|-----------|------|--------|
| Semantics only (no Flow) | - | - | 42.4 |
| +Flow head | 40.7 | 0.597 | 39.7 |
| +Bin classification | 39.9 | 0.565 | 38.3 |
| +BEV cost volume | 41.1 | 0.588 | 40.2 |
| +Channel expansion (final) | **42.1** | **0.537** | **40.5** |

### Key Findings
- Adaptive lifting and semantic prototypes each contribute independently, and their combination yields the best results.
- Adding the flow prediction head degrades semantic occupancy performance; the BEV cost volume effectively mitigates this conflict.
- The real-time variant ALOcc-2D-mini achieves near-SOTA accuracy at 30.5 FPS.
- When ground-truth depth is used, camera-only input surpasses multi-modal fusion methods.

## Highlights & Insights
- The physical motivation behind the occlusion-aware mechanism is clear: it simulates the human ability to infer complete shapes from partial observations.
- The depth denoising technique elegantly addresses premature convergence in depth estimation priors.
- The BEV cost volume reuses cached features from previous frames with negligible additional computation.
- The complete model family—from real-time to high-accuracy—demonstrates the practical flexibility of the framework.

## Limitations & Future Work
- There is a relatively high dependence on ground-truth depth (performance improves substantially with GT depth), leaving room for improvement in vision-only mode.
- The BEV cost volume has limited modeling capacity for motion along the Z-axis, focusing primarily on X-Y plane motion.
- Long-tail categories show improvement but remain challenging, particularly extremely rare classes.
- The purely convolutional architecture may be less effective than Transformers at modeling long-range dependencies.

## Related Work & Insights
- **vs. FB-Occ**: ALOcc achieves a mIoU improvement of 4.5+ under identical conditions at comparable speed.
- **vs. COTR**: Performance is comparable, but COTR runs at only 0.5 FPS; ALOcc is 16× faster.
- **vs. FlashOCC**: ALOcc-2D-mini is slightly faster with a mIoU gain of 9.4.
- **vs. multi-modal methods**: With GT depth, ALOcc surpasses methods that use additional LiDAR/Radar inputs.

## Rating
- Novelty: ⭐⭐⭐⭐ Each of the three modules has a clear technical contribution; the occlusion-aware lifting mechanism is particularly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three benchmarks, detailed ablation design, and complete speed-accuracy trade-off analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-organized structure, rigorous mathematical derivations, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐ Establishes a new baseline in autonomous driving 3D perception; the real-time variant has practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)

<!-- RELATED:END -->
