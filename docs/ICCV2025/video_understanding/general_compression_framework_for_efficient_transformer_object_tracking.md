---
title: >-
  [Paper Note] General Compression Framework for Efficient Transformer Object Tracking
description: >-
  [ICCV 2025][Video Understanding][Object Tracking] This paper proposes CompressTracker, a general Transformer tracker compression framework that achieves architecture-agnostic efficient compression through three progressive innovations—stage division, replacement training, and feature mimicking—delivering a 2.42× speedup while retaining approximately 99% of SUTrack's accuracy.
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Object Tracking"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Transformer"
  - "Efficient Inference"
date: 2026-05-08
content_hash: ed100e6001fefb2d
---

# General Compression Framework for Efficient Transformer Object Tracking

**Conference**: ICCV 2025
**arXiv**: [2409.17564](https://arxiv.org/abs/2409.17564)  
**Code**: [GitHub](https://github.com/honglyhly/CompressTracker) (code available as stated in the paper)  
**Area**: Video Understanding
**Keywords**: Object Tracking, Model Compression, Knowledge Distillation, Transformer, Efficient Inference

## TL;DR

This paper proposes CompressTracker, a general Transformer tracker compression framework that achieves architecture-agnostic efficient compression through three progressive innovations—stage division, replacement training, and feature mimicking—delivering a 2.42× speedup while retaining approximately 99% of SUTrack's accuracy.

## Background & Motivation

Transformer-based trackers (e.g., OSTrack, SUTrack) achieve outstanding performance on standard benchmarks, yet their deployment on resource-constrained devices remains challenging. Existing acceleration approaches suffer from three core issues:

**Accuracy degradation**: Lightweight designs (e.g., HiT, SMAT) underfit due to limited parameters.

**Training complexity**: MixFormerV2's multi-stage distillation requires 120 hours (8×RTX8000), and suboptimality at intermediate stages accumulates.

**Architectural constraints**: Existing distillation paradigms require the student model to share the same architecture as the teacher.

CompressTracker targets a **single-step end-to-end, architecture-agnostic, high-fidelity** general compression solution.

## Method

### Overall Architecture

CompressTracker comprises three progressive innovations:
1. Stage Division → 2. Replacement Training → 3. Prediction Guidance & Feature Mimicking

These form a coherent knowledge transfer chain: stage division serves as the foundation, replacement training builds upon it, and prediction guidance with feature mimicking further refine the knowledge transfer.

### Key Designs

1. **Stage Division**:

    - The teacher model's $N_t$ layers are evenly partitioned into $N$ stages ($N$ = number of student layers).
    - Each student stage (1 layer) learns to replicate the functionality of the corresponding teacher stage (multiple layers).
    - Linear projection layers are added before and after each student stage to align feature dimensions (removed at inference).
    - This breaks the conventional paradigm of treating the model as an indivisible whole, enabling fine-grained knowledge transfer.
    - Supports student models of arbitrary Transformer architectures.

2. **Replacement Training**:

    - During training, student stages are dynamically and randomly replaced by the corresponding frozen teacher stages.
    - Each stage uses Bernoulli sampling to decide whether to use the teacher or the student:
    $h_i = \begin{cases} stage_i^t(h_{i-1}), & r_i = 0 \\ stage_i^s(h_{i-1}), & r_i = 1 \end{cases}, \quad r_i \sim \text{Bernoulli}(p)$
    - Core advantage: unsubstituted teacher stages provide contextual supervision for the substituted student stages.
    - The student does not learn in isolation but directly participates in the teacher's behavior.
    - At inference, student stages are concatenated directly.

3. **Prediction Guidance & Stage-wise Feature Mimicking**:

    - **Prediction Guidance**: Uses the teacher's predictions as additional supervision to accelerate convergence.
    - **Stage-wise Feature Mimicking**: Computes the L2 distance between corresponding stage outputs as a loss term.
    - Simple L2 distance is adopted over complex losses to highlight the effectiveness of stage division and replacement training.

4. **Progressive Replacement**:

    - $p$ grows progressively from $p_{init}$ to 1.0, implementing easy-to-hard learning.
    - Eliminates the need for a separate fine-tuning step, enabling truly end-to-end training.
    - Three-phase schedule: warmup ($p_{init}$) → linear growth → full student ($p=1.0$).

### Loss & Training

$$L = \lambda_{track} L_{track} + \lambda_{pred} L_{pred} + \lambda_{feat} L_{feat}$$

- $\lambda_{track} = 1$, $\lambda_{pred} = 1$, $\lambda_{feat} = 0.2$
- $p_{init} = 0.5$, $\alpha_1 = \alpha_2 = 0.1$
- AdamW optimizer, learning rate $4 \times 10^{-5}$, 500 epochs
- Search/template image resolution: 256×256 / 128×128
- Student initialized with teacher pretrained weights (skip-layer strategy marginally outperforms consecutive layers)

## Key Experimental Results

### Main Results (Tables)

**Compression results across teacher models**:

| Method | LaSOT AUC | Retention | GPU FPS | Speedup |
|--------|-----------|-----------|---------|---------|
| SUTrack (Teacher) | 73.2 | 100% | 55 | 1.0× |
| **CT-SUTrack** | **72.2** | **99%** | **134** | **2.42×** |
| OSTrack (Teacher) | 69.1 | 100% | 105 | 1.0× |
| **CT-OSTrack** | **66.1** | **96%** | **228** | **2.17×** |
| ODTrack (Teacher) | 73.2 | 100% | 32 | 1.0× |
| **CT-ODTrack** | **70.5** | **96%** | **87** | **2.71×** |

**Comparison with lightweight trackers**:

| Method | LaSOT AUC | TNL2K AUC | TrackingNet AUC | GPU FPS |
|--------|-----------|-----------|-----------------|---------|
| MixFormerV2-S | 60.6 | 48.3 | 75.8 | 325 |
| HCAT | 59.0 | — | 76.6 | 195 |
| HiT-Base | 64.6 | — | 80.0 | 175 |
| **CT-OSTrack-4** | **66.1** | **53.6** | **82.1** | **228** |

### Ablation Study (Tables)

**Ablation of supervision strategies (LaSOT AUC)**:

| # | Prediction Guidance | Feature Mimicking | Replacement Training | AUC |
|---|:---:|:---:|:---:|-----|
| 1 | | | | 62.8% |
| 4 | | | ✓ | 63.7% |
| 5 | ✓ | | ✓ | 64.1% |
| 6 | | ✓ | ✓ | 64.5% |
| 8 | ✓ | ✓ | ✓ | **65.2%** |

**Comparison with other compression techniques**:

| Method | AUC | FPS |
|--------|-----|-----|
| Pruning (MixFormerV2-S) | 60.6% | 325 |
| Distillation | 63.8% | 228 |
| **CompressTracker-4** | **66.1%** | **228** |

### Key Findings

- The three components contribute progressively: RT (+0.9%), PG (+0.4%), FM (+0.7%), totaling +2.4% AUC.
- Replacement probability performs optimally in the range of 0.5–0.7; values too low lead to insufficient training, while values too high reduce teacher–student interaction.
- Uniform stage partition performs comparably to non-uniform partition (62.8% vs. 62.7%); the simpler scheme is adopted.
- Initializing the student with skip-layer teacher weights (62.3%) marginally outperforms consecutive-layer initialization (62.0%).
- Training requires only 20 hours (8×RTX3090), far less than MixFormerV2-S's 120 hours.
- The framework generalizes to varying numbers of layers (2–8), resolutions, and teacher models.

## Highlights & Insights

- **Truly general**: Compatible with arbitrary teacher models, layer counts, resolutions, and student architectures—a capability unattainable by prior methods.
- The replacement training mechanism is elegantly designed: by dynamically involving teacher stages during training, each student stage learns within authentic contextual conditions.
- Progressive replacement eliminates multi-stage training, enabling end-to-end optimization.
- CT-SUTrack achieves 72.2% AUC on LaSOT, a post-compression performance that surpasses many uncompressed trackers.

## Limitations & Future Work

- The number of student layers still requires manual selection; automatic architecture search could be explored.
- Feature mimicking relies solely on L2 distance; more advanced distribution-matching methods may yield further gains.
- Validation is limited to Transformer trackers; applicability to CNN–Transformer hybrid architectures remains unexplored.
- The impact of progressive replacement schedule parameters $\alpha_1, \alpha_2$ on performance is not thoroughly analyzed.

## Related Work & Insights

- The stage division concept is transferable to compression of other Transformer models (detection, segmentation, etc.).
- Replacement training can be viewed as a more elegant form of progressive distillation, with potential future application to large language model compression.
- Comparison with MixFormerV2 demonstrates that single-step end-to-end training is superior to complex multi-stage distillation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Replacement training and progressive replacement strategies are novel and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four teacher models, five benchmarks, comprehensive ablations, and multi-dimensional generalization validation.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with a fluent, progressively developed framework presentation.
- **Value**: ⭐⭐⭐⭐⭐ The general-purpose framework offers strong practical utility and is directly applicable to industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](../../CVPR2026/video_understanding/uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)
- [\[CVPR 2026\] An Efficient Token Compression Framework for Visual Object Tracking](../../CVPR2026/video_understanding/an_efficient_token_compression_framework_for_visual_object_tracking.md)
- [\[ICCV 2025\] Towards Efficient General Feature Prediction in Masked Skeleton Modeling](towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [\[ICCV 2025\] ResidualViT for Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)
- [\[ICCV 2025\] MobileViCLIP: An Efficient Video-Text Model for Mobile Devices](mobileviclip_an_efficient_video-text_model_for_mobile_devices.md)

</div>

<!-- RELATED:END -->
