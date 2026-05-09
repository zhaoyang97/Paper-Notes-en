---
title: >-
  [Paper Note] What You Have is What You Track: Adaptive and Robust Multimodal Tracking
description: >-
  [ICCV 2025][Video Understanding][Multimodal tracking] This paper proposes FlexTrack—the first framework to systematically study tracking under **temporally incomplete multimodal data**—achieving adaptive computational complexity via a Heterogeneous Mixture-of-Experts fusion module (HMoE) combined with a video-level masking training strategy. FlexTrack achieves state-of-the-art performance on 9 benchmarks, with gains of 2.6% under complete modalities and 10.2% under missing-modality scenarios.
tags:
  - ICCV 2025
  - Video Understanding
  - Multimodal tracking
  - missing modality
  - Mixture-of-Experts
  - video-level masking
  - adaptive complexity
date: 2026-05-08
content_hash: e9133a28dce461b5
---

# What You Have is What You Track: Adaptive and Robust Multimodal Tracking

**Conference**: ICCV 2025
**arXiv**: [2507.05899](https://arxiv.org/abs/2507.05899)
**Code**: Coming soon
**Area**: Video Understanding / Object Tracking / Multimodal Fusion
**Keywords**: Multimodal tracking, missing modality, Mixture-of-Experts, video-level masking, adaptive complexity

## TL;DR

This paper proposes FlexTrack—the first framework to systematically study tracking under **temporally incomplete multimodal data**—achieving adaptive computational complexity via a Heterogeneous Mixture-of-Experts fusion module (HMoE) combined with a video-level masking training strategy. FlexTrack achieves state-of-the-art performance on 9 benchmarks, with gains of 2.6% under complete modalities and 10.2% under missing-modality scenarios.

## Background & Motivation

Multimodal tracking (e.g., RGB+Depth, RGB+Thermal, RGB+Event camera) leverages complementary information to improve robustness under occlusion, illumination changes, and similar challenges. In practice, however, **sensor synchronization failures** are common:

### Core Problem: Temporal Modality Absence

Differences in **exposure time and frame rate** across sensors make perfect synchronization extremely difficult. Existing datasets (e.g., LasHeR, DepthTrack) assume modalities are always paired, an assumption that does not hold in real-world deployments—modalities may intermittently disappear within a temporal window.

### Limitations of Prior Work

**ViPT, UnTrack**: Unified architectures that entirely ignore missing-modality scenarios.

**IPT**: The first method to handle missing modalities, but it designs **separate prompt strategies** for each missing pattern, resulting in a rigid architecture.

**Key Challenge**: Should a model maintain the same computational complexity as when modalities are complete, even when modalities are missing? The answer is no.

The core philosophy of FlexTrack: **the tracker should dynamically allocate computational resources according to data availability**—activating simpler experts when more modalities are absent, and more complex experts when data is complete.

### Comparison with Prior Methods

| Method | Unified Architecture | Unified Parameters | Missing Modality | Adaptive Complexity |
|--------|---------------------|--------------------|------------------|---------------------|
| IPT | ✗ | ✗ | ✓ | ✗ |
| ViPT | ✓ | ✗ | ✗ | ✗ |
| UnTrack | ✓ | ✓ | ✗ | ✗ |
| **FlexTrack** | **✓** | **✓** | **✓** | **✓** |

## Method

### Overall Architecture

FlexTrack takes video clips and search regions from RGB and an auxiliary modality (depth/thermal/event, collectively denoted X) as input, and consists of two core components:

1. **HMoE Fusion Module**: dynamically adjusts complexity at inference time.
2. **Video-Level Masking Strategy**: used only during training to enhance temporal robustness.

### Heterogeneous Mixture-of-Experts Fusion (HMoE-Fuse)

Unlike conventional MoE where all experts are **homogeneous** (identical architecture and size), each expert in HMoE has a **different hidden dimension** $2^d$, where $d \in \{2, \ldots, D-1\}$.

**Routing Mechanism**: Rather than token-level routing, HMoE routes **entire video clips** to experts to ensure temporal consistency:

$$g_n = \begin{cases} \text{Softmax}(G(T_v))_n, & \text{if } G(T_v)_n \in \text{Top-K}(G(T_v)) \\ 0, & \text{otherwise} \end{cases}$$

$$T_y^1 = \sum_{n=1}^{M} g_n E_n(T_v)$$

where $K=2$, and the output is a weighted sum over the activated experts.

**Intuition**: When modalities are complete (high information content), the gating function tends to activate **complex experts** (high-dimensional hidden layers); when modalities are missing (low information content), it activates **simpler experts**—realizing adaptive computational complexity.

The full HMoE-Fuse pipeline further includes a linear attention transformation:

$$T_y^2 = T_v W_1 (T_v W_2)^T$$
$$T_y^3 = T_y^1 (T_y^2)^T$$
$$T_y^4 = \text{Softmax}(T_v W_3) T_y^3$$
$$T_y^5 = T_y^4 W_4$$

### Multimodal Video-Level Masking Strategy

Existing masking approaches are ill-suited for multimodal tracking:

- **Random/MAE-style masking**: disrupts spatial integrity and temporal continuity.
- **VideoMAE tube masking**: preserves temporal consistency but persistently masks the same spatial locations across all frames.
- **Key Insight**: Tracking simultaneously requires **spatial integrity** (understanding target appearance) and **temporal continuity** (capturing motion dynamics).

FlexTrack's solution (Algorithm 1):

1. **Search region masking**: randomly selects from 5 predefined patterns (both modalities retained / RGB retained–X missing / X retained–RGB missing, etc.), retaining complete data with probability 3/5.
2. **Video clip masking**: triggered with probability $\alpha$; for each clip, a modality-missing pattern (complete / RGB only / X only) is independently sampled.
3. **Core guarantee**: at every time step, at least one modality retains complete spatial information.

### Loss & Training

$$\mathcal{L}_{all} = \lambda_1 \mathcal{L}_{aux} + \lambda_2 \mathcal{L}_{cls} + \lambda_3 \mathcal{L}_{l1} + \lambda_4 \mathcal{L}_{GIoU}$$

where $\mathcal{L}_{aux} = \mathcal{L}_{balance} + \mathcal{L}_{important}$ (from Switch Transformer), ensuring expert load balancing. Default weights: $\lambda_1=1, \lambda_2=5, \lambda_3=2, \lambda_4=1$.

## Key Experimental Results

### Main Results (Complete Modalities)

**Table 2: RGB-Event Tracking (VisEvent)**

| Method | P | AUC |
|--------|---|-----|
| ViPT | 75.8 | 59.2 |
| UnTrack | 75.5 | 58.9 |
| STTrack | 78.6 | 61.9 |
| **FlexTrack** | **81.4** | **64.1** |

**Table 3: RGB-Thermal Tracking**

| Method | LasHeR AUC | RGBT234 MSR |
|--------|------------|-------------|
| ViPT | 52.5 | 61.7 |
| STTrack | 60.3 | 66.7 |
| SUTrack | 59.5 | 69.5 |
| **FlexTrack** | **62.0** | **69.9** |

**Table 4: RGB-Depth Tracking (DepthTrack)**

| Method | F-score | Re | Pr |
|--------|---------|----|----|
| SUTrack | 65.1 | 65.7 | 64.5 |
| STTrack | 63.3 | 63.4 | 63.2 |
| **FlexTrack** | **67.0** | **66.9** | **67.1** |

FlexTrack achieves state-of-the-art results across all 9 complete-modality benchmarks, surpassing the previous best by an average of **2.6%**.

### Missing Modality Results

On the missing-modality benchmarks introduced by IPT (random/switched/prolonged missing), FlexTrack improves upon the previous best method by **10.2%**, validating the effectiveness of the HMoE adaptive mechanism.

### Ablation Study

The contribution of video-level masking is confirmed by substantial performance drops on missing-modality tests when masking is removed, demonstrating the necessity of simulating missing modalities during training. The heterogeneous design of HMoE also proves its value: uniform-size experts underperform heterogeneous ones, indicating that experts of varying complexity do develop meaningful specialization.

## Highlights & Insights

1. **First systematic study**: Temporally incomplete multimodal tracking is a real and important problem, previously obscured by dataset design assumptions.
2. **The "unequal complexity" design of HMoE** is intuitively motivated: less information → simpler processing; more information → more complex processing. The model adapts not only to missing rates but also to scene complexity.
3. **Three-fold guarantee of video-level masking**: spatial integrity + temporal continuity + at least one modality always available.
4. A unified framework handles three modality combinations (RGB-D, RGB-T, RGB-Event), with **joint training** further improving generalization.

## Limitations & Future Work

1. The number of experts and their dimension configurations in HMoE depend on hyperparameters that may require adjustment for different modality combinations.
2. The masking strategy during training (5 search-region patterns, 3 clip patterns) is manually designed; ideally, it should be learned adaptively from the data distribution.
3. Validation is limited to dual-modality scenarios (RGB+1); scalability to three or more modalities remains unexplored.
4. Masking is disabled at inference time, yet in real deployments modality absence occurs online—the model's real-time adaptive capability warrants further evaluation.

## Related Work & Insights

- **Multimodal tracking**: ViPT, UnTrack, SDSTrack, OneTracker, STTrack
- **Missing modality**: IPT, FuseMoE, Flex-MoE
- **MoE**: DeepSeekMoE, Switch Transformer

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of heterogeneous MoE and video-level masking establishes a new paradigm for multimodal tracking.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 9 benchmarks, both complete and missing modality settings, detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; the pseudocode for the masking strategy is intuitive.
- Value: ⭐⭐⭐⭐⭐ — Directly addresses the reality of imperfect sensors, with high practical engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[ICCV 2025\] XTrack: Multimodal Training Boosts RGB-X Video Object Trackers](xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)
- [\[ICCV 2025\] UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions](umdatrack_unified_multi-domain_adaptive_tracking_under_adverse_weather_condition.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](aim_adaptive_inference_of_multi-modal_llms_via_token_merging_and_pruning.md)
- [\[ICCV 2025\] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception](egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)

</div>

<!-- RELATED:END -->
