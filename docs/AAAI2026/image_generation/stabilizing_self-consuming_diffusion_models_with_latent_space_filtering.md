---
title: >-
  [Paper Note] Stabilizing Self-Consuming Diffusion Models with Latent Space Filtering
description: >-
  [AAAI2026][Image Generation][self-consuming training] This paper proposes Latent Space Filtering (LSF), a method that analyzes the degradation of low-dimensional structure in the latent representations of self-consuming diffusion models and uses confidence scores from a probing classifier to filter low-quality synthetic data. Under a fixed training budget, LSF effectively mitigates model collapse without requiring additional real data or an enlarged training set.
tags:
  - AAAI2026
  - Image Generation
  - self-consuming training
  - model collapse
  - diffusion models
  - latent space filtering
  - data quality
date: 2026-05-08
content_hash: 1f46d9c6310132c7
---

# Stabilizing Self-Consuming Diffusion Models with Latent Space Filtering

**Conference**: AAAI2026  
**arXiv**: [2511.12742](https://arxiv.org/abs/2511.12742)  
**Authors**: Zhongteng Cai, Yaxuan Wang, Yang Liu, Xueru Zhang  
**Code**: [GitHub](https://github.com/osu-srml/Latent-Space-Filtering)  
**Area**: Image Generation  
**Keywords**: self-consuming training, model collapse, diffusion models, latent space filtering, data quality  

## TL;DR

This paper proposes Latent Space Filtering (LSF), a method that analyzes the degradation of low-dimensional structure in the latent representations of self-consuming diffusion models and uses confidence scores from a probing classifier to filter low-quality synthetic data. Under a fixed training budget, LSF effectively mitigates model collapse without requiring additional real data or an enlarged training set.

## Background & Motivation

### State of the Field
As synthetic data proliferates on the internet, new-generation generative models inevitably train on datasets containing synthetic samples, forming a "self-consuming loop." Research has shown that this loop leads to model collapse—degraded generation quality, reduced diversity, and the forgetting of rare samples.

### Limitations of Prior Work
- **Accumulating historical data**: Storage and computation costs grow linearly (e.g., CelebA requires 582 MB by generation 5).
- **Injecting fresh real data**: Acquiring labeled data is costly in practice.
- **Modifying the training procedure** (e.g., SIMS score extrapolation, self-correction): These approaches become unstable over multiple generations or sacrifice diversity.
- **Existing analyses focus on input space**: The structural changes in latent representations are largely overlooked.

### Core Motivation
The paper seeks to understand model collapse from a latent space perspective—examining how the low-dimensional structure of latent representations degrades across self-consuming generations—and to develop a filtering mechanism that **requires no additional real data and incurs no extra training cost**.

## Core Problem

1. How does the low-dimensional structure of a self-consuming diffusion model's latent space evolve across training generations?
2. Can the quality of latent representations be exploited to filter low-quality synthetic data and thereby mitigate model collapse?
3. How can a theoretical quantitative connection between latent space degradation and filtering criteria be established?

## Method

### Latent Representation Extraction
The U-Net encoder $e^{(0)}$ of the initial model (trained solely on real data) is used to extract latent representations. For samples $\mathbf{x}^{(k)}$ generated at generation $k$, the latent representation at denoising timestep $t$ is:

$$\mathbf{h}_t^{(k)} = e^{(0)}(\mathbf{x}^{(k)}, t)$$

### OLE Metric for Low-Dimensional Structure
The Orthogonal Low-rank Embedding (OLE) score measures inter-class subspace orthogonality:

$$\text{OLE}_t^{(k)} = \sum_{c \in \mathcal{C}} \|\mathbf{M}_{c,t}^{(k)}\|_* - \|\mathbf{M}_t^{(k)}\|_*$$

where $\|\cdot\|_*$ denotes the nuclear norm. A lower OLE indicates more orthogonal subspaces and better structure.

**Key Findings**: (1) At a fixed timestep, OLE increases with generation index, indicating progressive degradation of latent structure. (2) At a fixed generation, OLE follows a U-shape—intermediate timesteps exhibit the best structure.

### Theoretical Analysis 1: Lower Bound on OLE (Theorem 1)
Assuming two-class latent representations follow a noisy low-rank Gaussian distribution $\mathcal{N}(\mathbf{0}, \mathbf{U}_c\mathbf{U}_c^\top + \sigma^2\mathbf{I}_d)$, when the maximum principal angle $\tilde{\theta}$ between subspaces decreases (i.e., subspaces become more aligned), the lower bound on OLE increases:

$$\mathbb{E}[\text{OLE}(\mathbf{M}_0, \mathbf{M}_1)] \geq C_1 - C_2 \cdot \phi(\tilde{\theta})$$

where $\phi(\tilde{\theta}) = \sqrt{2n}\sqrt{\ell\cos\tilde{\theta}} + \sqrt{2n(2n-1)}\sqrt{1-\cos\tilde{\theta}}$, which decreases as $\theta$ decreases, causing the lower bound to increase.

### Latent Space Filtering (LSF)
**Mechanism**: OLE is a batch-level metric and cannot filter individual samples. The method instead uses the confidence of a probing classifier as a per-sample proxy.

1. Extract latent representations from real data and train a softmax regression classifier.
2. For any sample $(\mathbf{x}, y)$, compute the confidence for the correct class:

$$\xi(\mathbf{x}, y) = \frac{\exp\{o_y(\mathbf{x})\}}{\sum_{c \in \mathcal{C}} \exp\{o_c(\mathbf{x})\}}$$

3. Select the $N$ samples with the highest confidence from the accumulated dataset to form the training set.

### Theoretical Analysis 2: Upper Bound on Confidence (Theorem 2)
The expected confidence of the Bayes-optimal classifier satisfies:

$$\xi(\theta) \leq \frac{1}{2\sigma^2(\sigma^2+1)} \varsigma(r\sin^2\theta)$$

As the subspace angle $\theta \to 0$, the upper bound decreases, confirming that classifier confidence genuinely reflects subspace orthogonality.

### Algorithm
1. Train the probing classifier on real data (required only once; 65K parameters).
2. At each generation: generate synthetic data → construct accumulated/mixed dataset → compute per-sample confidence → select top-$N$ high-confidence samples → train the new model.

## Key Experimental Results

### Datasets and Setup
- MNIST (28×28, 10 classes), CIFAR-10 (32×32, 10 classes), CelebA (64×64, 4 classes)
- Fine-tune for 3 epochs with 1,000 samples per generation; generate 10,000 samples for evaluation
- Self-consuming training over 5 generations

### Main Results (MNIST & CelebA)

| Method | Extra Real Data | Fixed Budget | FID Trend | Precision | Recall |
|--------|:---:|:---:|------|------|------|
| SYN (synthetic only) | ✗ | ✓ | Continuously worsens | Decreasing | Decreasing |
| SYN-ADD (30% real) | ✓ | ✓ | Partially mitigated | Moderate | Moderate |
| ACU (accumulation) | ✗ | ✗ | Stable | High | High |
| ACUR (random sampling) | ✗ | ✓ | Close to ACU | Moderate–high | High |
| ACUR-SIMS | ✗ | ✓ | Unstable fluctuation | Fluctuating | Fluctuating |
| ACUR-SC | ✗ | ✓ | Increasing | Moderate | Low |
| **ACU-LSF (Ours)** | **✗** | **✓** | **Lowest / stable** | **Highest** | **Comparable to ACU** |

### Key Advantages
- ACU-LSF achieves the lowest FID and highest Precision under a fixed budget, with Recall comparable to ACU.
- Compared to ACU: the probing classifier requires only 65K parameters and 109 MB of features (vs. 32M parameters and 582 MB of data).
- ACUR-SIMS becomes unstable due to repeated score extrapolation.
- ACUR-SC maintains acceptable Precision but suffers a significant drop in Recall, indicating reduced diversity.

### Filtering Efficacy Validation
- Confidence scores for later-generation samples are systematically lower, confirming that confidence discriminates between real and synthetic samples.
- Filtered datasets contain samples from earlier generations on average, with a higher proportion of real data.
- Larger accumulated data pools yield more effective filtering.

## Highlights & Insights

- **Novel latent space perspective**: This is the first systematic study of low-dimensional structural degradation in latent representations under self-consuming diffusion model training.
- **Complete theoretical framework**: Theorem 1 establishes a quantitative link between OLE and subspace orthogonality; Theorem 2 proves that classifier confidence is a valid proxy for OLE.
- **High practicality**: No additional real data or enlarged training sets are required, and the probing classifier incurs minimal training cost (65K parameters).
- **U-shaped OLE finding**: Reveals that representation quality first improves and then declines across the denoising trajectory, consistent with the unimodal trajectory of representation quality.
- **Compatible with multiple training paradigms**: LSF can be integrated into purely synthetic or accumulation-based self-consuming loops.

## Limitations & Future Work

- **Fixed class assumption**: Dynamic class changes in continual learning or unlearning scenarios are not addressed.
- **Dependence on the initial model**: If the initial model is of poor quality, the evaluation baseline becomes unreliable.
- **Validated only on DDPM**: Applicability to more advanced diffusion models (Latent Diffusion, DiT) remains to be verified.
- **Low-resolution experiments**: The maximum resolution tested is 64×64 (CelebA); high-resolution images are not evaluated.
- **Unconditional/simple conditional generation**: Complex conditional generation scenarios such as text-to-image are not validated.
- **Filtering may introduce selection bias**: The method consistently favors samples close to the training distribution, potentially suppressing beneficial distributional exploration.

## Related Work & Insights

- **vs. MAD (Alemohammad et al.)**: MAD primarily defines the problem and provides theoretical analysis; this paper offers a concrete and actionable filtering solution.
- **vs. SIMS**: SIMS is effective within a single generation via score function extrapolation but becomes unstable across multiple generations; LSF remains consistently effective.
- **vs. Self-Correction**: SC maps synthetic samples toward real-distribution cluster centers, sacrificing diversity (low Recall); LSF preserves diversity.
- **vs. Data Accumulation (ACU)**: ACU requires storing all historical data with linearly growing training costs; LSF achieves comparable quality under a fixed budget.
- **vs. Variance Filtering (Hallucination)**: Variance filtering requires access to sampling trajectories, which is impractical; LSF requires only a single forward pass.

The latent space quality assessment paradigm can be generalized to self-consuming training of other generative models such as GANs and VAEs. The use of a probing classifier as a data quality proxy is applicable to large-scale data cleaning and selection. The OLE metric can serve as an online diagnostic tool for monitoring generative model degradation. The approach is highly aligned with the Data-Centric AI philosophy—optimizing data quality rather than model architecture.

## Rating

- Novelty: ⭐⭐⭐⭐ — Analyzing model collapse from a latent space perspective is a genuinely new starting point with solid theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Three datasets with multi-baseline comparisons, but limited by low resolution and the absence of large-scale experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, tight integration of theory and experiments, and rich intuitive figures.
- Value: ⭐⭐⭐⭐ — Highly practical and addresses an important problem (a core challenge in the era of synthetic data); the method is concise and efficient.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](../../ICCV2025/image_generation/whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](../../ICLR2026/image_generation/generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](../../CVPR2026/image_generation/dip_taming_diffusion_models_in_pixel_space.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](../../ICCV2025/image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)

<!-- RELATED:END -->
