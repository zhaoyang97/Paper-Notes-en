---
title: >-
  [Paper Note] SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow
description: >-
  [ECCV 2024][Video Understanding][Optical Flow] SEA-RAFT achieves SOTA accuracy while maintaining a simple architecture through three improvements: Mixture of Laplace (MoL) loss, direct regression of initial optical flow, and rigid-flow pre-training, achieving a speedup of over 2.3× compared to existing methods.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Optical Flow"
  - "RAFT"
  - "Mixture of Laplace"
  - "Iterative Optimization"
  - "Efficient Inference"
date: 2026-05-08
content_hash: 9da4f61751dd3aa7
---

# SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow

**Conference**: ECCV 2024  
**arXiv**: [2405.14793](https://arxiv.org/abs/2405.14793)  
**Code**: [https://github.com/princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)  
**Area**: Video Understanding / Optical Flow Estimation  
**Keywords**: Optical Flow, RAFT, Mixture of Laplace, Iterative Optimization, Efficient Inference

## TL;DR

SEA-RAFT achieves SOTA accuracy while maintaining a simple architecture through three improvements: Mixture of Laplace (MoL) loss, direct regression of initial optical flow, and rigid-flow pre-training, achieving a speedup of over 2.3× compared to existing methods.

## Background & Motivation

**Background**: Optical flow estimation is a foundational task in low-level vision, used in downstream tasks such as action recognition, video inpainting, frame interpolation, and 3D reconstruction. Most current SOTA methods are based on the RAFT architecture, which iteratively refines the optical flow field through recurrent networks.

**Limitations of Prior Work**:
   - RAFT-like methods require a large number of iterations (12 during training, up to 32 during inference), leading to severe latency.
   - The standard $L_1$ loss performs poorly on **ambiguous cases** caused by occlusions, where high errors in these samples dominate the training loss.
   - Zero-initialized optical flow deviates significantly from the ground truth, leading to slow convergence.
   - The custom encoder and ConvGRU designs of original RAFT are complex and lack scalability.

**Key Challenge**: The trade-off between accuracy and efficiency—existing high-accuracy methods (e.g., MS-RAFT+) are extremely slow (with SEA-RAFT achieving up to 24× speedup), whereas efficient methods suffer from significant accuracy degradation.

**Goal**: To achieve or surpass SOTA accuracy while significantly improving efficiency, and to enhance cross-dataset generalization.

**Key Insight**: Simultaneously improve RAFT from four orthogonal dimensions: loss function design (probabilistic regression), flow field initialization strategy, pre-training data strategy, and architectural simplification.

**Core Idea**: Model optical flow uncertainty with a Mixture of Laplace distributions, enabling the network to distinguish "normal pixels" from "ambiguous pixels". Combined with direct regression of the initial flow and rigid-flow pre-training, SOTA is achieved with only 4 iterations.

## Method

### Overall Architecture

SEA-RAFT inherits the iterative refinement framework of RAFT: feature encoders extract features → construct multi-scale 4D correlation volumes → RNN iteratively refines the optical flow. The key improvements comprise three parts: (1) Mixture of Laplace loss replacing $L_1$; (2) direct regression of initial optical flow by the context encoder; (3) rigid-flow pre-training on TartanAir.

### Key Designs

1. **Mixture of Laplace (MoL) Loss**:

   **Function**: Models optical flow prediction as a mixture of two Laplace distributions, one handling normal cases, the other handling ambiguous cases such as occlusions.

   **Mechanism**: For each pixel, predict the mixture coefficient $\alpha$, the scale parameter $\beta_2$, and the *mean* $\mu$. The key innovation is **fixing the scale of the first component $\beta_1 = 0$**, making it equivalent to the standard $L_1$ loss:

    $$MixLap(x; \alpha, 0, \beta_2, \mu) = \alpha \cdot \frac{e^{-|x-\mu|}}{2} + (1-\alpha) \cdot \frac{e^{-\frac{|x-\mu|}{e^{\beta_2}}}}{2e^{\beta_2}}$$

   The training loss is a sequence-weighted formulation: $\mathcal{L}_{all} = \sum_{i=1}^{N} \gamma^{N-i} \mathcal{L}_{MoL}^i$

   **Design Motivation**: 
    - Normal pixels are dominated by the first component where $\alpha$ is close to 1, which is equivalent to the $L_1$ loss, aligning with the evaluation metrics.
    - Ambiguous pixels (e.g., heavily occluded) are handled by the second component, which reduces the penalty on these unpredictable samples through a large $\beta_2$.
    - Regress $\beta$ in the log space to avoid numerical instability.
    - Unlike probabilistic methods in keypoint matching, optical flow requires accurate correspondences for **every pixel**, making it necessary to align one of the mixture components with $L_1$.

2. **Direct Regression of Initial Flow**:

   **Function**: Uses the context encoder $C$ to receive the stacked two-frame input and directly predict the initial optical flow, replacing the zero-initialization of RAFT.

   **Mechanism**: Stack the two image frames and feed them into the context encoder to regress an initial optical flow estimate along with its MoL parameters. This introduces minimal additional computational overhead by reusing the existing encoder.

   **Design Motivation**: Zero initialization can deviate heavily from the ground truth, requiring numerous iterations to converge. Providing a reasonable initial estimate through FlowNet-style direct regression significantly reduces the required number of iterations (from 32 down to 4-12).

3. **Rigid-Flow Pre-Training**:

   **Function**: Pre-trains on the TartanAir dataset. TartanAir provides optical flow annotations in static scenes generated by camera motion.

   **Mechanism**: Despite the limited motion diversity of TartanAir (camera motion only), its scene realism and diversity far exceed those of synthetic datasets (FlyingChairs/Things), which helps improve generalization.

   **Design Motivation**: The scale and realism of existing training datasets (FlyingChairs, FlyingThings3D) are limited. Although TartanAir only contains rigid-flow, it provides higher scene realism, serving as a low-cost data augmentation strategy.

### Architectural Simplification

- **Encoder**: Replace RAFT's custom encoder with a standard ImageNet pre-trained ResNet (obviating the need for different normalization layers).
- **RNN**: Replace ConvGRU with 2 ConvNeXt blocks, which have fewer parameters and more stable training.
- **Iterations**: Training and inference require only $N=4$ for SEA-RAFT(S/M), and up to $N=12$ for SEA-RAFT(L).

### Loss & Training

- Pre-training: TartanAir 300K steps → FlyingChairs 100K steps → FlyingThings3D 120K steps (i.e., "C+T")
- Fine-tuning: Sintel+Things+KITTI+HD1K 300K steps ("C+T+S+K+H")
- Additional fine-tuning for Spring/KITTI
- $\gamma < 1$ exponentially decays the weights of early iterations
- The upper bound of $\beta_2$ is set to 10 to ensure training stability

## Key Experimental Results

### Main Results — Spring Benchmark

| Method | Extra Data | Fine-tune | Spring(test) 1px↓ | Spring(test) EPE↓ | Spring(test) WAUC↑ |
|------|---------|------|-------------------|-------------------|-------------------|
| RAFT | None | ✗ | 6.790 | 1.476 | 90.920 |
| FlowFormer | None | ✗ | 6.510 | 0.723 | 91.679 |
| MS-RAFT+ | VIPER | ✗ | 5.724 | 0.643 | 92.888 |
| CroCoFlow | CroCo | ✓ | 4.565 | 0.498 | 93.660 |
| **SEA-RAFT(S)** | TartanAir | ✓ | 3.904 | 0.377 | 94.182 |
| **SEA-RAFT(M)** | TartanAir | ✓ | **3.686** | **0.363** | **94.534** |

### Ablation Study — Spring subval

| Configuration | Initial Flow | TartanAir Pre-training | RNN Type | Loss Function | EPE |
|---------|--------|--------------|---------|---------|-----|
| SEA-RAFT (w/o Tar.) | ✓ | ✗ | 2×ConvNeXt | MoL ($\beta_1$=0) | 0.187 |
| SEA-RAFT (w/ Tar.) | ✓ | ✓ | 2×ConvNeXt | MoL ($\beta_1$=0) | **0.179** |
| w/o Direct Reg. | ✗ | ✗ | 2×ConvNeXt | MoL ($\beta_1$=0) | 0.201 |
| RAFT GRU | ✓ | ✗ | GRU | MoL ($\beta_1$=0) | 0.189 |
| Naive Laplace | ✓ | ✗ | 2×ConvNeXt | Single Laplace | 0.217 |
| Naive MoL | ✓ | ✗ | 2×ConvNeXt | MoL (both $\beta$ free) | 0.248 |
| $L_1$ Loss | ✓ | ✗ | 2×ConvNeXt | $L_1$ | 0.206 |
| Mixture of Gaussian | ✓ | ✗ | 2×ConvNeXt | MoG | 0.210 |

### Key Findings

- **The $\beta_1=0$ constraint in MoL loss is crucial**: Naive MoL with free $\beta_1$ (0.248) performs worse than $L_1$ (0.206), while MoL with fixed $\beta_1=0$ (0.187) yields the best results.
- **Direct regression of the initial flow shows significant effects**: EPE drops from 0.201 to 0.187, with an increase of only ~7G MACs.
- **Iteration bottleneck is eliminated**: RAFT's iterations account for 82-86% of total latency, whereas for SEA-RAFT, it is only 26-39%.
- **Stunning efficiency advantage**: SEA-RAFT(S) processes 1080p at 21fps (RTX3090), which is 3× faster than RAFT and 24× faster than MS-RAFT+.
- Spring benchmark EPE is reduced by 22.9% (0.363 vs 0.471), and 1px error is reduced by 17.8% (3.686 vs 4.482).
- KITTI cross-dataset generalization is optimal: Fl-epe 3.62, Fl-all 12.9.

## Highlights & Insights

1. **Clever design of probabilistic modeling aligned with evaluation metrics**: Fixing $\beta_1=0$ degrades MoL to $L_1$ in normal cases and automatically tolerates ambiguous cases—a simple and elegant solution.
2. **The "less is more" philosophy**: Achieving superior results to RAFT with only 4 iterations instead of 32 proves that a good initialization is more important than brute-force iteration.
3. **Orthogonality of the three improvements**: The loss function, initialization, and data strategies are complementary and can be independently integrated into other RAFT variants.
4. **The courage to simplify architecture**: Replacing custom modules with standard ResNet and ConvNeXt reduces complexity while enhancing performance.

## Limitations & Future Work

1. **Anomalous performance on Sintel Final pass**: Under the C+T setting, Sintel Final performance is suboptimal (4.04 vs 2.40 of competing methods). The authors could not explain the reason, and it only improved after incorporating KITTI+HD1K.
2. **Hyperparameter sensitivity of the MoL loss**: The upper bound of $\beta_2$ (10) and the specific mixture distribution form require experimental tuning.
3. **Limitations of TartanAir pre-training**: It only supports rigid-flow, lacking independent object motion.
4. **Unexplored large-scale real-data pre-training**: E.g., the approach of using diffusion models for pre-training like DDVM.
5. **Lack of downstream application validation for uncertainty estimation**: MoL provides uncertainty outputs ($\alpha$, $\beta_2$), but their value has not been verified in downstream tasks.

## Related Work & Insights

- **RAFT Family**: GMA, FlowFormer, CRAFT, etc., focus on replacing modules (e.g., Transformers), whereas SEA-RAFT focuses on loss functions and training strategies—making them orthogonal and complementary.
- **Probabilistic Regression**: PDC-Net+ uses MoL in keypoint matching but does not require alignment with $L_1$; SEA-RAFT's $\beta_1=0$ constraint is an original adaptation designed for the "per-pixel accuracy" requirement in optical flow.
- **Efficient Inference**: EMD-L, DIFT, etc., reduce iterations through efficient implementations but at the cost of noticeable accuracy degradation; SEA-RAFT achieves "substituting quantity with quality" via initial flow regression.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Although the combination of the $\beta_1=0$ constraint in MoL loss and the direct regression of initial flow is simple, the combined outcome is excellent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on Spring/Sintel/KITTI, with ablation studies covering all dimensions of loss, initialization, pre-training, and architecture.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, well-articulated motivation, and concise mathematical derivations.
- **Value**: ⭐⭐⭐⭐⭐ Open-source code, 2.3×-24× acceleration, and 1080p@21fps provide practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient All-Pairs Correlation Volume Sampling for Optical Flow Estimation](../../CVPR2026/video_understanding/efficient_all-pairs_correlation_volume_sampling_for_optical_flow_estimation.md)
- [\[ECCV 2024\] LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow](layeredflow_a_real-world_benchmark_for_non-lambertian_multi-layer_optical_flow.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](../../ICCV2025/video_understanding/memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[CVPR 2026\] FlowFM: Advancing Dark Optical Flow Estimation with Flow Matching](../../CVPR2026/video_understanding/flowfm_advancing_dark_optical_flow_estimation_with_flow_matching.md)
- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](../../ICCV2025/video_understanding/prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)

</div>

<!-- RELATED:END -->
