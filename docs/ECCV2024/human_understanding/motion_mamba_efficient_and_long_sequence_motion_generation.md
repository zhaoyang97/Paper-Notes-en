---
title: >-
  [Paper Note] Motion Mamba: Efficient and Long Sequence Motion Generation
description: >-
  [ECCV 2024][Human Understanding][Human Motion Generation] This paper proposes Motion Mamba, which is the first to introduce Selective State Space Models (Mamba) to human motion generation. Through two core components, Hierarchical Temporal Mamba (HTM) and Bidirectional Spatial Mamba (BSM), it reduces FID by 50% (0.473 $\rightarrow$ 0.281) on HumanML3D while achieving a 4x inference speedup (0.217s $\rightarrow$ 0.058s).
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Human Motion Generation"
  - "State Space Models"
  - "Mamba"
  - "Latent Diffusion Models"
  - "Long Sequence Modeling"
date: 2026-05-08
content_hash: 5a4f4639f0a3cd1d
---

# Motion Mamba: Efficient and Long Sequence Motion Generation

**Conference**: ECCV 2024  
**arXiv**: [2403.07487](https://arxiv.org/abs/2403.07487)  
**Code**: [https://github.com/steve-zeyu-zhang/MotionMamba](https://github.com/steve-zeyu-zhang/MotionMamba)  
**Area**: Human Understanding  
**Keywords**: Human Motion Generation, State Space Models, Mamba, Latent Diffusion Models, Long Sequence Modeling

## TL;DR
This paper proposes Motion Mamba, which is the first to introduce Selective State Space Models (Mamba) to human motion generation. Through two core components, Hierarchical Temporal Mamba (HTM) and Bidirectional Spatial Mamba (BSM), it reduces FID by 50% (0.473 $\rightarrow$ 0.281) on HumanML3D while achieving a 4x inference speedup (0.217s $\rightarrow$ 0.058s).

## Background & Motivation
- **Background**: Human motion generation is an important direction in generative computer vision. Context-conditioned (e.g., text-driven) motion generation has employed various methods including autoencoders, GANs, autoregressive models, and diffusion models. Among them, diffusion models have become mainstream due to their advantages in generation quality and diversity.
- **Limitations of Prior Work**: 
  1. Convolution or Transformer-based diffusion methods face quadratic computational growth bottlenecks in long-sequence motion generation; Transformers are not naturally designed for temporal modeling.
  2. Even when performing diffusion in the latent space (e.g., MLD), the quadratic complexity of the attention mechanism still limits inference efficiency.
- **Key Challenge**: How to handle long-range dependencies and maintain near-linear computational complexity while preserving high-quality generation.
- **Key Insight**: State Space Models (SSMs), particularly Mamba, feature efficient hardware-aware designs and long-range sequence modeling capabilities, but they lack specialized architectures for motion data.
- **Core Idea**: Design an SSM-based architecture specifically tailored for motion generation, using hierarchical scanning in the temporal dimension and bidirectional scanning in the spatial dimension, combined with latent diffusion models to achieve efficient, high-quality motion generation.

## Method

### Overall Architecture
Motion Mamba is based on the Latent Motion Diffusion model and adopts a denoising U-Net architecture. First, a Motion VAE compresses motion sequences into a latent space, and then the diffusion process operates within this latent space. The denoiser $\epsilon_\theta$ consists of $N$ encoder blocks $E_{1..N}$, $N$ decoder blocks $D_{1..N}$, and a central Transformer attention-mixing block $M$. Each encoder/decoder block is composed of two core modules: HTM and BSM. Text conditions are embedded using a frozen CLIP-VIT-L-14 encoder.

### Key Designs
1. **Hierarchical Temporal Mamba (HTM)**:

    - **Function**: Processes latent representations along the temporal dimension to capture temporal dependencies at different depths.
    - **Design Motivation**: The authors observe that motion information is denser in low-level feature spaces, requiring more scans to capture details. A fixed number of scans cannot satisfy both efficiency and quality.
    - **Mechanism**: Distributes varying numbers of SSM scans hierarchically across the U-Net encoder and decoder. Outer layers (closer to input/output) utilize more scans $S_{2N-1}$, while inner layers use fewer scans $S_1$, forming a symmetric hierarchical structure $K=\{S_{2N-1}, S_{2(N-1)-1}, ..., S_1\}$. Each scan contains an independent SSM module (1D convolution $\rightarrow$ linear projection to obtain $B,C,\Delta$ $\rightarrow$ discretization $\rightarrow$ SSM calculation). The outputs of all scans are aggregated and linearly projected to obtain the final output.
    - **Comparison with Prior Methods**: Unlike the fixed number of attention heads in Transformers, HTM leverages the low-parameter nature of SSMs to increase the number of scans, balancing efficiency and quality through hierarchical distribution.

2. **Bidirectional Spatial Mamba (BSM)**:

    - **Function**: Processes latent poses along the channel/spatial dimension to enhance the accuracy of motion generation within a single frame.
    - **Design Motivation**: The structural information flow of the latent skeleton contains critical information in both forward and backward directions; unidirectional scanning would miss backward-direction dependencies.
    - **Mechanism**: First, the input dimensions are permuted from $(T, B, C)$ to $(C, B, T)$, swapping the temporal and channel dimensions. Then, forward and backward SSM scans are performed along the channel dimension. Finally, the bidirectional outputs are fused via a GateAndSum operation.
    - **Comparison with Prior Methods**: Unlike visual bidirectional SSMs like Vim, BSM is specifically designed for the channel dimension in motion latent spaces, ensuring bidirectional flow of spatial information through dimension permutation.

3. **Attention-Mixing Block**:

    - A Transformer attention block $M$ is inserted at the bottom of the U-Net to enhance condition fusion capability, serving as a hub for interaction between temporal and conditional information.

### Loss & Training
- A standard latent diffusion training objective is adopted: minimizing the MSE between target noise and predicted noise in the latent space.
- AdamW optimizer is used with a learning rate of $10^{-4}$ and a batch size of 512 (parallelized across 4 GPUs).
- Trained for 2000 epochs, with 1000 diffusion steps during training and 50 steps during inference.
- Model configuration: 11 layers, latent dimension $z \in \mathbb{R}^{2,d}$.

## Key Experimental Results

### Main Results

| Dataset | Metric | Motion Mamba | MLD (Prev. SOTA) | Gain |
|--------|------|------|----------|------|
| HumanML3D | FID↓ | **0.281** | 0.473 | -40.6% |
| HumanML3D | R-Precision Top3↑ | **0.792** | 0.772 | +2.6% |
| HumanML3D | MM Dist↓ | **3.060** | 3.196 | -4.3% |
| KIT-ML | FID↓ | **0.307** | 0.404 | -24.0% |
| KIT-ML | R-Precision Top3↑ | **0.765** | 0.734 | +4.2% |
| HumanML3D-LS(Long Sequence) | FID↓ | **0.668** | 0.952 | -29.8% |

### Ablation Study

| Configuration | FID | Description |
|------|---------|------|
| MM $\{S_1,...,S_N\}$ (low to high) | 1.278 | Baseline scan order, worst performance |
| MM $\{S_N,...,S_1\}$ (high to low) | 0.962 | Reverted order, significant improvement |
| MM $\{S_{2N_n-1},...,S_1\}$ (Hierarchical) | **0.281** | Optimal, hierarchical design brings substantial improvement |
| SingleScan | 1.063 | Unidirectional scan has limited effect |
| BiScan, block | **0.281** | Block-level bidirectional scan is optimal |
| Dim=1 | 0.652 | Dimension too low limits representation |
| Dim=2 | **0.281** | Optimal dimension |
| 9 layers | 1.080 | Insufficient layers |
| 11 layers | **0.281** | Optimal layers |
| 27/37 layers | 0.975/0.809 | Going too deep leads to degradation |

### Key Findings
- The hierarchical scanning strategy is the largest contributor, reducing the FID from 1.278 to 0.281.
- The optimal latent dimension is 2 (rather than 1 as in MLD), as the multi-scan mechanism of HTM requires additional dimensions to carry information.
- The parameters of each Mamba layer are about 25% of a Transformer encoder block, allowing high efficiency to be maintained even when increasing the number of layers.
- The advantage is even more pronounced on the long-sequence subset HumanML3D-LS (FID 0.668 vs. MLD 0.952).
- The inference time is only 0.058s per sequence, representing a ~4x speedup over MLD (0.217s).

## Highlights & Insights
- **First to apply Mamba to motion generation**: Opens up a new direction for SSMs in the motion domain, proving that SSMs can replace Transformers as the backbone of diffusion models.
- **Elegant design of hierarchical temporal scanning**: Leverages the observation that "lower-level features contain denser motion information," naturally fitting the symmetric architecture of U-Net.
- **Win-win for efficiency and quality**: The linear complexity of SSMs makes it computationally feasible to increase the number of scans, whereas Transformers cannot achieve a similar "head increase" while keeping efficiency intact.
- User studies further validate generation quality: outperforms MLD by 62% and 59% in text-to-motion alignment and quality assessments, respectively.

## Limitations & Future Work
- Validated only on text-to-motion tasks, and not extended to other conditional generation tasks such as music-to-dance or action-to-motion.
- Still relies on a frozen Motion VAE for compression; the reconstruction error of the VAE itself forms an upper bound on generation quality.
- The latent dimension of 2 is small, which may limit the representational capacity for highly complex motions.
- Lacks comprehensive comparison with the latest non-diffusion methods like MoMask.

## Related Work & Insights
- **MLD** [Chen et al.] proposed motion diffusion in the latent space, which is the direct baseline of this paper.
- **Mamba** [Gu & Dao]'s selective SSM provides a hardware-efficient foundation for long-sequence modeling.
- **VMamba / Vim** explored the application of SSMs in 2D vision. This paper is the first to extend it to motion sequences, which consist of 1D temporal + structured pose data.
- Insight: SSMs may also replace Transformers in other sequential generation tasks (e.g., speech synthesis, trajectory prediction) to achieve efficiency gains.

## Rating
- Novelty: ⭐⭐⭐⭐ For the first time, Mamba is introduced to motion generation, and the HTM hierarchical design is unique.
- Experimental Thoroughness: ⭐⭐⭐⭐ The ablation study is comprehensive and includes user studies, but lacks a full comparison with the latest SOTA.
- Writing Quality: ⭐⭐⭐⭐ Overall clear, using pseudocode to assist in understanding the algorithm.
- Value: ⭐⭐⭐⭐ Breakthroughs are achieved in both efficiency and quality, opening up a new direction of SSM in motion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ICLR 2026\] EasyTune: Efficient Step-Aware Fine-Tuning for Diffusion-Based Motion Generation](../../ICLR2026/human_understanding/easytune_efficient_step-aware_fine-tuning_for_diffusion-based_motion_generation.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)
- [\[ECCV 2024\] CoMo: Controllable Motion Generation Through Language Guided Pose Code Editing](como_controllable_motion_generation_through_language_guided_pose_code_editing.md)
- [\[ECCV 2024\] FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis](freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)

</div>

<!-- RELATED:END -->
