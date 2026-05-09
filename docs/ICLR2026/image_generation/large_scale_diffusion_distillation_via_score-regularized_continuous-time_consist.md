---
title: >-
  [Paper Note] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency
description: >-
  [ICLR 2026][Image Generation][Continuous-time consistency models] This paper proposes rCM (score-regularized continuous-time consistency model), which for the first time scales continuous-time consistency distillation to 14B-parameter text-to-image/video models. By combining forward KL divergence (consistency) with reverse KL divergence (score distillation), rCM matches DMD2 in quality while preserving diversity, achieving 15–50× inference speedup.
tags:
  - ICLR 2026
  - Image Generation
  - Continuous-time consistency models
  - score distillation
  - large-scale distillation
  - JVP
  - few-step generation
date: 2026-05-08
content_hash: 8ef3e70d71dfad61
---

# Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency

**Conference**: ICLR 2026
**arXiv**: [2510.08431](https://arxiv.org/abs/2510.08431)
**Code**: [Project Page](https://research.nvidia.com/labs/dir/rcm)
**Area**: Image Generation
**Keywords**: Continuous-time consistency models, score distillation, large-scale distillation, JVP, few-step generation

## TL;DR
This paper proposes rCM (score-regularized continuous-time consistency model), which for the first time scales continuous-time consistency distillation to 14B-parameter text-to-image/video models. By combining forward KL divergence (consistency) with reverse KL divergence (score distillation), rCM matches DMD2 in quality while preserving diversity, achieving 15–50× inference speedup.

## Background & Motivation
- **Background**: sCM (continuous-time consistency models) is theoretically elegant, but its applicability to large-scale text-to-image/video models remains unclear — JVP computation is incompatible with FlashAttention and distributed training.
- **Limitations of Prior Work**: sCM suffers from quality degradation in fine-grained generation (error accumulation and quality diffusion caused by the mode-covering nature of forward KL divergence). Score/adversarial distillation methods (e.g., DMD2) achieve superior quality but are prone to mode collapse and lack diversity.
- **Key Challenge**: Forward KL divergence (consistency models) and reverse KL divergence (score distillation) have complementary characteristics.
- **Goal**: Combine the two divergence types to achieve both high quality and high diversity at large scale.

## Method

### Overall Architecture
rCM = sCM (forward KL consistency distillation) + DMD (reverse KL score distillation) + infrastructure optimization. Training alternates between optimizing the student model (rCM loss) and the fake score network (flow matching loss).

### Key Designs

1. **FlashAttention-2 JVP Kernel**: A Triton kernel is developed to integrate JVP into the forward pass of FlashAttention-2, supporting both self-attention and cross-attention, and compatible with FSDP and Context Parallelism, enabling sCM training to scale to models with 10B+ parameters.

2. **Score Regularization**: The DMD loss is introduced as a long-jump regularizer to complement sCM. The final objective is:
$$\mathcal{L}_{\text{rCM}}(\theta) = \mathcal{L}_{\text{sCM}}(\theta) + \lambda \mathcal{L}_{\text{DMD}}(\theta)$$
   with $\lambda=0.01$ being universal across models and tasks. sCM provides mode-covering (high diversity), while DMD provides mode-seeking (high quality).

3. **Stable Time-Derivative Computation**: To address JVP training instability in large models, two strategies are proposed:
   - Semi-continuous time: JVP is used for spatial components, while finite differences approximate the temporal component ($\Delta t = 10^{-4}$).
   - High-precision time: FP32 precision is enforced on time-embedding layers.

4. **Rollout Strategy**: The student can perform arbitrary-step sampling; the number of steps $N \in [1, N_{\max}]$ is randomly selected. DMD loss is backpropagated only through the final step, with random timesteps ensuring coverage of the full time range.

### Loss & Training
- sCM loss (tangent normalization): $\mathcal{L}_{\text{sCM}} = \mathbb{E}\left[\left\|\mathbf{F}_\theta - \mathbf{F}_{\theta^-} - \frac{\mathbf{g}}{\|\mathbf{g}\|_2^2 + c}\right\|_2^2\right]$
- DMD loss: guides the student using the discrepancy between fake score and teacher score.
- The fake score network is trained on student-generated data using a flow matching loss, with alternating optimization.

## Key Experimental Results

### Main Results (GenEval T2I)

| Model | Params | NFE | Overall | Counting | Position |
|-------|--------|-----|---------|----------|----------|
| FLUX.1-dev | 12B | 50 | 0.66 | 0.74 | 0.22 |
| Cosmos-Predict2 14B (teacher) | 14B | 70 | 0.84 | 0.79 | 0.64 |
| Cosmos-Predict2 + DMD2 | 2B | 4 | 0.80 | 0.70 | 0.57 |
| **Cosmos-Predict2 + rCM** | **2B** | **4** | **0.81** | **0.73** | **0.58** |
| **Cosmos-Predict2 + rCM** | **14B** | **4** | **0.83** | **0.80** | **0.59** |
| **Cosmos-Predict2 + rCM** | **14B** | **1** | **0.82** | **0.84** | **0.49** |

### VBench Video Experiments

| Model | Params | NFE | Total Score | Throughput (FPS) |
|-------|--------|-----|-------------|------------------|
| Wan2.1 14B (teacher) | 14B | 100 | 83.58 | 0.18 |
| Wan2.1 + DMD2 | 1.3B | 4 | 84.56 | 14.6 |
| **Wan2.1 + rCM** | **1.3B** | **4** | **84.43** | **14.6** |
| **Wan2.1 + rCM** | **14B** | **2** | **85.05** | **8.3** |

### Key Findings
- rCM matches or exceeds DMD2 in quality while significantly outperforming DMD2 in diversity (Figure 1 shows DMD2 outputs converging in object position and pose).
- The 14B rCM achieves a GenEval score of 0.83 in 4 steps, approaching the teacher's 0.84 in 70 steps.
- In video generation, rCM achieves near-teacher VBench scores in just 2 steps.
- $\lambda=0.01$ achieves the best balance between quality and diversity.
- Pure sCM exhibits notable quality defects in fine-grained scenarios such as text rendering, which rCM successfully resolves.

## Highlights & Insights
- This is the first work to scale JVP-based continuous-time consistency distillation to 14B parameters and 5-second video generation.
- The paper offers a unified framework for understanding distillation methods through the complementarity of forward and reverse KL divergences.
- No GAN tuning or extensive hyperparameter search is required; $\lambda=0.01$ generalizes across tasks.
- The diversity advantage of rCM is particularly valuable for scenarios requiring diverse outputs, such as interactive world models.

## Limitations & Future Work
- An additional fake score network introduces extra memory overhead.
- JVP computation remains slower than standard forward passes, resulting in high training costs.
- Single-step video generation still shows noticeable quality degradation (VBench drops from 85.05 to 83.02).
- Extension to autoregressive video diffusion is mentioned only as future work.

## Related Work & Insights
- sCM and MeanFlow provide the theoretical foundation.
- DMD/DMD2 provide practical solutions for reverse KL distillation.
- The philosophy of jointly combining forward and reverse KL divergences in DDO and DDRL underpins the design of rCM.
- rCM provides a practical acceleration solution for deploying large-scale visual generation models.

## Technical Details
- TrigFlow noise schedule: $\alpha_t = \cos(t), \sigma_t = \sin(t)$, interconvertible with rectified flow via SNR matching.
- The fake score network is trained on student-generated data with a flow matching loss via alternating optimization.
- Selective Activation Checkpointing (SAC) is employed to reduce memory consumption.
- The teacher uses CFG, which is jointly distilled into the student.
- Full-parameter fine-tuning is used (no LoRA), emphasizing the stability of rCM.
- Experiments cover Cosmos-Predict2 (0.6B/2B/14B T2I) and Wan2.1 (1.3B/14B T2V).
- Wan2.1 14B with 2-step rCM achieves 8.3 FPS versus the teacher's 0.18 FPS (~46× speedup).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The theoretical insight of combining forward and reverse KL divergences is valuable, though individual components are known.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Validation at an unprecedented scale (14B parameters, T2I+T2V, multi-step ablations).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical analysis is clear and engineering details are thorough.
- **Value**: ⭐⭐⭐⭐⭐ Addresses the core challenge of accelerating large-scale diffusion models with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](../../ICCV2025/image_generation/sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[ICLR 2026\] Compositional amortized inference for large-scale hierarchical Bayesian models](compositional_amortized_inference_for_large-scale_hierarchical_bayesian_models.md)
- [\[ICLR 2026\] Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening](motion_prior_distillation_in_time_reversal_sampling_for_generative_inbetweening.md)
- [\[ICLR 2026\] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)
- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)

<!-- RELATED:END -->
