---
title: >-
  [Paper Note] DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture
description: >-
  [CVPR 2025][Model Compression][Data-Free Knowledge Distillation] This paper proposes the DKDM paradigm, achieving data-free knowledge distillation for diffusion models for the first time. By substituting the real data distribution with the reverse denoising process of a pre-trained teacher model, and incorporating a dynamic iterative distillation strategy to efficiently generate diverse training knowledge, it supports student models of any architecture…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Data-Free Knowledge Distillation"
  - "Diffusion Models"
  - "Dynamic Iterative Distillation"
  - "Cross-Architecture Distillation"
  - "Generative Model Compression"
date: 2026-05-08
content_hash: f43911d7d379571e
---

# DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture

**Conference**: CVPR 2025  
**arXiv**: [2409.03550](https://arxiv.org/abs/2409.03550)  
**Code**: [https://github.com/qianlong0502/DKDM](https://github.com/qianlong0502/DKDM)  
**Area**: Diffusion Models  
**Keywords**: Data-Free Knowledge Distillation, Diffusion Models, Dynamic Iterative Distillation, Cross-Architecture Distillation, Generative Model Compression

## TL;DR

This paper proposes the DKDM paradigm, achieving data-free knowledge distillation for diffusion models for the first time. By substituting the real data distribution with the reverse denoising process of a pre-trained teacher model, and incorporating a dynamic iterative distillation strategy to efficiently generate diverse training knowledge, it supports student models of any architecture, achieving generative performance comparable to or even better than data-driven training without any access to the original data.

## Background & Motivation

**Background**: Diffusion models have shown exceptional capabilities in generative tasks such as image and video generation, but their training relies on massive datasets. For instance, Stable Diffusion requires billions of image-text pairs, and the training dataset for DALL·E 2 is as large as 5.6 billion samples. Various organizations have open-sourced a large number of pre-trained diffusion models, which have already "memorized" the distribution of the training data.

**Limitations of Prior Work**: Existing diffusion model distillation methods suffer from three main limitations: (1) almost all methods require access to the original training data or a subset of it; (2) student models typically must share the same architecture as the teacher and be initialized from teacher weights, severely restricting architectural flexibility; (3) an intuitive data-free alternative of pre-generating a complete synthetic dataset using the teacher to train the student is practically unfeasible due to prohibitive time and storage costs of generating billions of synthetic images.

**Key Challenge**: In data-free scenarios, the training objective of diffusion models inherently depends on real data $x^0$—specifically, the posterior distribution $q$ and the forward noising process $q(x^t|x^0)$ in the standard diffusion training KL divergence objective $D_{KL}(q(x^{t-1}|x^t,x^0) \| p_\theta(x^{t-1}|x^t))$ both require $x^0$. Eliminating the dependency on $x^0$ while maintaining valid training signals is the core challenge.

**Goal**: Propose a comprehensive Data-Free Knowledge Distillation for Diffusion Models (DKDM) paradigm that satisfies three strict requirements: fully data-free, support for arbitrary architectures, and highly efficient knowledge transfer.

**Key Insight**: The authors observe that, compared to the final generated "clean samples," the "noisy samples" at various timesteps of the diffusion process are more directly related to the diffusion optimization objective. Therefore, the transferred knowledge should take the form of intermediate states from the teacher's denoising process itself, rather than the final generation. The student learns from each denoising step of the teacher, rather than its final output.

**Core Idea**: Replace the real posterior $q$ with the reverse process distribution $p_{\theta_T}$ of the teacher diffusion model, utilize the teacher's generative power to substitute for the forward noising process, design a theoretically equivalent DKDM objective function, and reduce the time complexity of knowledge acquisition from $O(Tb)$ to $O(b)$ via dynamic iterative distillation.

## Method

### Overall Architecture

The DKDM framework consists of a pre-trained teacher model $\theta_T$ and a randomly initialized student model $\theta_S$ of arbitrary architecture. The training pipeline is as follows: (1) Sample from Gaussian noise to construct an initial enlarged batch set; (2) Perform a random number of teacher denoising steps (shuffle denoise) on each sample in the batch to make the timestep distribution uniform; (3) In each training iteration, randomly draw a sub-batch from the enlarged set, perform forward inference on this batch with both the teacher and the student, and calculate the DKDM loss to update the student; (4) Replace the corresponding old samples in the enlarged set with the denoised samples to enable dynamic updates. The entire process requires no access to real data.

### Key Designs

1. **DKDM Optimization Objective**:

    - **Function**: Provides optimization signals equivalent to standard diffusion training for student models under data-free settings.
    - **Mechanism**: Eliminates the dependency on real data $x^0$ in two steps. Step 1: eliminate the posterior. Since the conditional distribution $p_{\theta_T}(x^{t-1}|x^t)$ of a well-trained teacher closely approximates the true posterior $q(x^{t-1}|x^t, x^0)$, the posterior is directly replaced by the teacher distribution. This formulation reframes the original KL divergence objective to $D_{KL}(p_{\theta_T}(x^{t-1}|x^t) \| p_{\theta_S}(x^{t-1}|x^t))$, which correspondingly modifies the noise prediction MSE loss to $\|\epsilon_{\theta_T}(x^t,t) - \epsilon_{\theta_S}(x^t,t)\|^2$. Step 2: eliminate the forward prior. The teacher model is utilized to generate noisy samples $\hat{x}^t$ from pure noise $\epsilon$ via the reverse denoising chain $G_{\theta_T}(T-t)$, replacing the forward noising process $q(x^t|x^0)$ that relies on $x^0$. The final DKDM objective is formulated as $L_{DKDM} = L'_{simple} + \lambda L'_{vlb}$.
    - **Design Motivation**: It can be proven via the triangle inequality that minimizing $D_{KL}(p_{\theta_T} \| p_{\theta_S})$ indirectly minimizes $D_{KL}(q \| p_{\theta_S})$, theoretically ensuring that the student model can approximate the true data distribution. This design avoids structural dependencies on the model architecture, naturally supporting cross-architecture distillation.

2. **Dynamic Iterative Distillation Strategy**:

    - **Function**: Reduces the time complexity of constructing training batches from $O(Tb)$ to $O(b)$ while ensuring the diversity of training data.
    - **Mechanism**: Resolves efficiency and diversity challenges through three progressive tiers. **Basic Iterative Distillation**: instead of executing the full denoising chain from scratch for each training iteration, the teacher continuously denoises samples. It performs only a single step $g_{\theta_T}(x^t, t)$ at a time, feeding the output directly into the next step, and resets by resampling noise once it iterates from $t=T$ to $t=1$, forming an infinite data stream. **Shuffle Denoise**: each sample in the initial batch is randomly denoised a different number of times so that the inside-batch timesteps $t_i$ follow a uniform distribution, preventing training instability caused by identical $t$ values within the same batch. **Dynamic Mechanism**: an enlarged batch set $\hat{B}^+$ of size $\rho T |\hat{B}^s|$ is constructed. In each iteration, a subset $\hat{B}^s$ is randomly drawn for training, and the denoised samples replace the corresponding entries in $\hat{B}^+$, breaking the limitation where samples are consistently coupled within the same batch.
    - **Design Motivation**: In a naive implementation, acquiring $\hat{x}^t = G_{\theta_T}(T-t)$ requires $T-t$ denoising steps, which in the worst case takes $O(Tb)$ teacher inference steps per batch—$T$ times slower than standard training. The three-tier progressive strategy maintains an $O(b)$ complexity while enhancing diversity across both timestep distributions and sample compositions.

3. **Cross-Architecture Distillation**:

    - **Function**: Distills the generative capabilities of a CNN teacher model to a ViT student model, or vice versa.
    - **Mechanism**: The DKDM objective only compares the noise predictions and variance outputs of the teacher and student given the same noisy input $\hat{x}^t$. It does not involve intermediate feature matching or weight initialization, naturally supporting any architectural combinations. Experiments validate four combinations: CNN→CNN, CNN→ViT, ViT→CNN, and ViT→ViT.
    - **Design Motivation**: Existing methods (e.g., Xie et al.) find that distillation performance drops drastically when student models are randomly initialized because their distillation objectives implicitly depend on the same architecture or weights. DKDM eliminates this constraint by design through its objective function.

### Loss & Training

The final loss is $L_{DKDM}^{\star} = L_{simple}^{\star} + \lambda L_{vlb}^{\star}$. $L_{simple}^{\star} = \mathbb{E}_{(\hat{x}^t,t) \sim \hat{B}^+}[\|\epsilon_{\theta_T}(\hat{x}^t,t) - \epsilon_{\theta_S}(\hat{x}^t,t)\|^2]$ guides mean learning, while $L_{vlb}^{\star} = \mathbb{E}_{(\hat{x}^t,t) \sim \hat{B}^+}[D_{KL}(p_{\theta_T}(\hat{x}^{t-1}|\hat{x}^t) \| p_{\theta_S}(\hat{x}^{t-1}|\hat{x}^t))]$ guides variance learning. In latent space experiments, only $L_{simple}^{\star}$ is used because the teacher model was not trained with variance prediction. Sampling uses 50-step Improved DDPM in pixel space and 200-step DDIM in latent space. The scaling factor is $\rho = 0.4$, and $\lambda = 0.001$.

## Key Experimental Results

### Main Results

**Pixel Space**:

| Dataset | Method | IS↑ | FID↓ | sFID↓ |
|--------|------|-----|------|-------|
| CIFAR10 32² | Data-driven training | 8.73 | 7.84 | 7.38 |
| CIFAR10 32² | Data-free training (0%) | 8.28 | 12.06 | 13.23 |
| CIFAR10 32² | Data-limited (20%) | 8.49 | 9.76 | 11.30 |
| CIFAR10 32² | **DKDM** | **8.60** | **9.56** | **11.77** |
| CelebA 64² | Data-driven training | 3.04 | 5.39 | 7.23 |
| CelebA 64² | **DKDM** | **2.91** | **7.07** | **8.78** |

**Latent Space**:

| Dataset | Method | FID↓ | sFID↓ |
|--------|------|------|-------|
| CelebA-HQ 256² | Data-driven training | 9.09 | 12.10 |
| CelebA-HQ 256² | Data-free training | 15.36 | 17.56 |
| CelebA-HQ 256² | **DKDM** | **8.69** | **12.50** |
| FFHQ 256² | Data-driven training | 8.91 | 8.75 |
| FFHQ 256² | **DKDM** | **11.53** | **10.29** |

### Ablation Study

Contribution of components in dynamic iterative distillation (CIFAR10 FID):

| Configuration | FID↓ |
|------|------|
| Basic Iterative Distillation | ~13.5 |
| + Shuffle Denoise | ~11.2 |
| + Dynamic (ρ=0.4) | **9.56** |

Cross-architecture distillation (CIFAR10 FID):

| Teacher→Student | Data-Free | DKDM |
|-----------|-----------|------|
| CNN→CNN | 9.64 | **6.85** |
| ViT→CNN | 44.62 | **13.17** |
| CNN→ViT | 17.11 | **17.11** |
| ViT→ViT | 63.15 | **17.86** |

### Key Findings

- DKDM achieves an FID of 8.69 on CelebA-HQ 256, outperforming data-driven training (9.09). This indicates that learning from the teacher's denoising process might be easier than learning from original raw data, as the pre-trained model "smooths" the noise and outliers in the data distribution.
- The three components of dynamic iterative distillation work progressively, with each yielding clear improvements in FID.
- DKDM shows significant improvements in cross-architecture distillation (e.g., ViT→CNN FID drops from 44.62 to 13.17), proving that designing proper knowledge transfer formats is much more critical than merely generating synthetic data.
- The method introduces only minor GPU memory overhead and is even faster during training in the latent space.

## Highlights & Insights

- Pioneeringly defines the completely new paradigm of DKDM, providing a comprehensive solution both theoretically and practically.
- The core insight of "learning from the generative process" is highly elegant—intermediate noisy states are more valuable for diffusion model training than clean generated results. This essentially allows the student to directly mimic the teacher's "reasoning process" rather than its "reasoning outcome."
- The design of dynamic iterative distillation demonstrates strong engineering intuition—by shuffling timesteps, expanding the sampling pool, and utilizing random compositions, it maximizes training data diversity under an $O(b)$ computational budget.
- Outperforming data-driven training in certain experiments suggests that the pre-trained model may encode learning signals that are "friendlier" than the original raw data.

## Limitations & Future Work

- The performance upper bound of the student model is constrained by the quality of the teacher—if the teacher model's generation quality is poor, the distillation effectiveness will also be limited.
- Experiments are primarily validated on low-to-medium resolutions (up to 256²), lacking evaluation on high-resolution and large-scale conditional generation tasks (such as text-to-image Stable Diffusion).
- The hyperparameter $\rho$ in dynamic iterative distillation needs dataset-specific tuning.
- Future directions: (1) application to conditional diffusion model distillation; (2) co-designing with model compression to reduce both parameter sizes and data requirements; (3) exploring applications in data privacy-constrained scenarios.

## Related Work & Insights

- Core difference from traditional knowledge distillation (Hinton et al.): Instead of using the teacher's soft labels to train the student, DKDM replaces the data distribution with the teacher's reverse process distribution.
- Difference from diffusion model acceleration (e.g., Progressive Distillation, Consistency Models): The latter focuses on reducing sampling steps and typically requires identical model architectures; DKDM focus on data-free training while supporting arbitrary architectures.
- Insight: The concept of using pre-trained models as "data proxies" can be extended to other generative models such as GANs and Flow Models, holding promising application prospects in data copyright and privacy-sensitive scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneering definition and solution to the data-free distillation of diffusion models, with rigorous and complete theoretical derivations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage across pixel/latent spaces, multiple datasets, cross-architecture configurations, and ablations, though lacking large-scale experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Step-by-step progression from problem definition to solutions, logically clear.
- **Value**: ⭐⭐⭐⭐ — Highly practical significance for data-constrained and model architecture migration scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] When Data-Free Knowledge Distillation Meets Non-Transferable Teacher: Escaping Out-of-Distribution](../../ICML2025/model_compression/when_data-free_knowledge_distillation_meets_non-transferable_teacher_escaping_ou.md)
- [\[CVPR 2026\] DMGD: Train-Free Dataset Distillation with Semantic-Distribution Matching in Diffusion Models](../../CVPR2026/model_compression/dmgd_train-free_dataset_distillation_with_semantic-distribution_matching_in_diff.md)
- [\[ICLR 2026\] DVD-Quant: Data-free Video Diffusion Transformers Quantization](../../ICLR2026/model_compression/dvd-quant_data-free_video_diffusion_transformers_quantization.md)
- [\[CVPR 2026\] ManifoldGD: Training-Free Hierarchical Manifold Guidance for Diffusion-Based Dataset Distillation](../../CVPR2026/model_compression/manifoldgd_training-free_hierarchical_manifold_guidance_for_diffusion-based_data.md)
- [\[CVPR 2025\] What Makes a Good Dataset for Knowledge Distillation?](what_makes_a_good_dataset_for_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
