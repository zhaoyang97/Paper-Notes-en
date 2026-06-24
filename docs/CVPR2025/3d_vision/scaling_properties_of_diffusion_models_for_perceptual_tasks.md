---
title: >-
  [Paper Note] Scaling Properties of Diffusion Models for Perceptual Tasks
description: >-
  [CVPR 2025][3D Vision][Diffusion Model Scaling] This paper systematically studies the scaling properties of diffusion models on perceptual tasks such as depth estimation, optical flow prediction, and amodal segmentation. It establishes power-law scaling relations for both training and inference, and demonstrates that increasing test-time compute (via more denoising steps and multi-prediction ensembling) significantly boosts performance, achieving competitive results while usi…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Diffusion Model Scaling"
  - "Depth Estimation"
  - "Optical Flow Prediction"
  - "Perceptual Tasks"
  - "Test-Time Compute"
date: 2026-05-08
content_hash: 9678428fd3a207a0
---

# Scaling Properties of Diffusion Models for Perceptual Tasks

**Conference**: CVPR 2025  
**arXiv**: [2411.08034](https://arxiv.org/abs/2411.08034)  
**Code**: [https://scaling-diffusion-perception.github.io](https://scaling-diffusion-perception.github.io)  
**Area**: Diffusion Models / Visual Perception  
**Keywords**: Diffusion Model Scaling, Depth Estimation, Optical Flow Prediction, Perceptual Tasks, Test-Time Compute

## TL;DR

This paper systematically studies the scaling properties of diffusion models on perceptual tasks such as depth estimation, optical flow prediction, and amodal segmentation. It establishes power-law scaling relations for both training and inference, and demonstrates that increasing test-time compute (via more denoising steps and multi-prediction ensembling) significantly boosts performance, achieving competitive results while using far less training data and computation than previous SOTA.

## Background & Motivation

**Background**: Diffusion models have demonstrated outstanding scaling properties in image/video generation, but their scaling behavior in visual perception (discriminative) tasks remains understudied. Prior works like Marigold (for depth estimation), FlowDiffuser (for optical flow), and pix2gestalt (for amodal segmentation) have proven the feasibility of adapting image diffusion models for perceptual tasks, but these were conducted independently and lack a unified framework and systematic scaling analysis.

**Limitations of Prior Work**: Current methods adapted from diffusion models for perceptual tasks primarily rely on heavy large-scale pre-training (e.g., Stable Diffusion trained on internet-scale data) and lack systematic research on how to scale compute efficiently. In practice, researchers often operate under tight compute budgets without clarity on whether to scale up the model size, expand training data, increase resolution, or increase inference compute.

**Key Challenge**: The iterative denoising nature of diffusion models inherently supports test-time compute scaling (e.g., increasing steps or ensembling multiple predictions). However, a systematic scaling law is missing to guide optimal training and inference configurations, particularly regarding the unclear trade-off between training compute and test-time compute.

**Goal**: To unify diverse visual perception tasks into an image-to-image translation framework, systematically establish the training and inference scaling laws of diffusion models on these tasks, and provide compute-optimal solutions.

**Key Insight**: Drawing an analogy to OpenAI o1's test-time compute scaling in large language models (LLMs)—"letting the model think for 20 more seconds during inference can be equivalent to scaling the model size by 100,000 times." The iterative denoising process of diffusion models naturally aligns with this philosophy.

**Core Idea**: Unify depth estimation, optical flow, and amodal segmentation as conditional denoising diffusion, and establish scaling power laws across multiple dimensions, including model size, pre-training compute, resolution, MoE upcycling, denoising steps, ensemble size, and noise scheduling.

## Method

### Overall Architecture

All perceptual tasks are unified as conditional image-to-image translation. Given an input RGB image $I$ and optional conditioning images, they are encoded into the latent space via the Stable Diffusion VAE. The RGB latent $i_0$ is concatenated channel-wise with the randomly noised ground-truth latent $d_t$ and fed into a DiT model for conditional denoising. During inference, starting from pure noise, perceptual predictions are generated via iterative DDIM denoising. Pre-training is conducted on ImageNet-1K for class-conditional image generation, followed by fine-tuning on specific perceptual tasks.

### Key Designs

1. **Training Stage Scaling Analysis**:

    - **Function**: To dissect the power-law relationship of model size, pre-training compute, resolution, and MoE with downstream performance.
    - **Mechanism**: (a) **Model Size**: Six dense DiT models ranging from 14.8M to 1.9B parameters are trained, revealing that pre-training loss scales with compute according to a power law $L(C) = 0.23 \times C^{-0.0098}$. (b) **Pre-training Compute**: Holding the a4 model (458M parameters) constant, pre-training with varying steps (60K to 120K) highlights that more pre-training consistently boosts fine-tuning performance. (c) **Resolution**: Increasing resolution from 256 to 512 scales token count by 4×, leading to a power-law performance improvement in depth estimation. (d) **MoE Upcycling**: Upcycling fine-tuned dense models to sparse MoEs and continuing training improves AbsRel by 5.3% on average.
    - **Design Motivation**: To establish scaling laws that enable researchers to select the optimal configuration under a given budget.

2. **Test-time Compute Scaling Strategy**:

    - **Function**: To leverage the iterative and stochastic properties of diffusion models to improve precision by injecting more compute during inference.
    - **Mechanism**: Three complementary strategies are utilized. (a) **Increasing Denoising Steps**: Denoising steps $T \in \{1,2,5,10,20,50,100\}$ show power-law performance gains. (b) **Test-time Ensembling**: Running $N$ independent predictions ($N \in \{1,2,5,10,15,20\}$) and merging them using pixel-wise median or Marigold median compilation also results in a power-law boost. (c) **Noise Scheduling**: A cosine schedule, which allocates more compute to early denoising steps (global structures), is shown to be more effective than a linear schedule.
    - **Design Motivation**: Analogs to LLM test-time scaling; the coarse-to-fine nature of each denoising step in diffusion models provides an inherent "thinking at inference" mechanism.

3. **Unified Multi-task Model**:

    - **Function**: To perform depth estimation, optical flow, and amodal segmentation simultaneously using a single DiT-XL.
    - **Mechanism**: A PatchEmbedRouter routes tasks to different convolutional layers based on task types. After fine-tuning on a mixed dataset, upcycling is used to convert the model to an MoE for continued training.
    - **Design Motivation**: To validate the generalizability of the scaling strategies across diverse tasks.

### Loss & Training

A standard MSE denoising loss is employed. Fine-tuning uses an exponentially decaying learning rate from $1.2 \times 10^{-4}$ to $1.2 \times 10^{-6}$. The first convolutional layer of DiT is doubled in channel capacity to accommodate the concatenated RGB and noise latents, with weights initialized to half their original value. Inference utilizes DDIM with a cosine beta schedule. The optimal inference configuration is set to 200 denoising steps with 5 ensemble runs.

## Key Experimental Results

### Main Results

**Depth Estimation**:

| Method | Hypersim AbsRel↓ | ETH3D AbsRel↓ | NYUv2 AbsRel↓ | Pre-training Data |
|------|-----------------|---------------|--------------|----------|
| DPT | - | 7.8 | 9.8 | Large-scale |
| Marigold | 13.5 | 6.5 | 5.5 | Internet-scale |
| **Ours** | **13.6** | **4.8** | 6.8 | ImageNet-1K |

**Optical Flow** (FlyingChairs): Ours w/ ensemble achieves 3.08 EPE vs DeepFlow 3.53

**Amodal Segmentation**: Ours achieves 63.9 mIoU on MP3D vs pix2gestalt 61.5

### Ablation Study

| Scaling Dimension | Observed Power Law | Gain |
|---------|-----------------|---------|
| Model Size (14.8M→1.9B) | $L(C) \propto C^{-0.0098}$ | Continuous |
| Pre-training Steps (60K→120K) | Clear power law | Continuous |
| Resolution (256→512) | 4× tokens → power-law gain | Significant |
| MoE Upcycling | Matches/surpasses larger dense models | AbsRel -5.3% |
| Denoising Steps (1→100) | Clear power law | Significant |
| Ensemble Runs (1→20) | Clear power law | Moderate |
| Cosine vs Linear schedule | Cosine is significantly better | Significant |

### Key Findings

- Using only ImageNet-1K for pre-training, Ours surpasses Marigold on ETH3D (4.8 vs 6.5) which relies on internet-scale data—demonstrating that scaling strategies can outweigh data scale.
- Test-time compute is highly cost-effective; without extra training, scaling denoising steps and ensemble runs yields significant performance boosts.
- The cosine schedule is more effective than the linear schedule by allocating more compute to global structural reconstruction (the early steps).
- MoE upcycling serves as a "free lunch," cheaply scaling up the capacity of already-fine-tuned models to match or exceed larger dense models.

## Highlights & Insights

- This work is the first to establish a systematic scaling power law for diffusion models on perceptual tasks, providing compute-optimal guidance.
- The validation of "test-time compute scaling" in visual perception is a key contribution, suggesting that diffusion models are not just generative tools, but also a general-purpose "iterative compute" paradigm.
- Achieving competitive performance with far less data than prior SOTA highlights the high impact of proper scaling strategies.
- The analysis of the trade-off between training and inference compute offers highly practical guidance.

## Limitations & Future Work

- Perceptual performance still has room for improvement; specifically, there remains a gap between this method and specialized models in optical flow and amodal segmentation.
- Inference speed remains a deployment bottleneck, as 100 DDIM steps combined with multiple ensemble runs require heavy computation.
- The current scaling laws are established on ImageNet-1K, and their transferability to larger-scale datasets remains to be thoroughly validated.
- Future work could explore consistency distillation to reduce inference steps, as well as validation on a broader range of perceptual tasks.

## Related Work & Insights

- Relationship with Marigold: This paper serves as a "scaled and generalized" version of Marigold, unifying multi-task execution and systematically studying scaling.
- Relationship with DiT: It utilizes standard DiT architectures and scaling methodologies, generalizing from generative modeling to perceptual tasks.
- Inspiration: The "thinking during inference" phase of diffusion models is analogous to the chain-of-thought in LLMs—more steps lead to deeper reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — While the individual components are not fundamentally new, the systematic scaling analysis and the test-time scaling perspective are highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — The scale of multi-dimensional scaling experiments is massive, with comprehensive validation across three distinct tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — The structure is clear, and the power-law fitting results are presented intuitively.
- **Value**: ⭐⭐⭐⭐ — It provides key practical guidance for applying diffusion models to perceptual downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Simple yet Mighty Hartley Diffusion Versatilist for Generalizable Dense Vision Tasks](../../ICCV2025/3d_vision/a_simple_yet_mighty_hartley_diffusion_versatilist_for_genera.md)
- [\[CVPR 2025\] Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [\[CVPR 2025\] Olympus: A Universal Task Router for Computer Vision Tasks](olympus_a_universal_task_router_for_computer_vision_tasks.md)

</div>

<!-- RELATED:END -->
