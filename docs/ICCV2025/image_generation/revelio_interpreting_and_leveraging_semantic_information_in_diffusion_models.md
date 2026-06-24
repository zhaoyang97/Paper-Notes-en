---
title: >-
  [Paper Note] Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models
description: >-
  [ICCV 2025][Image Generation][Diffusion model interpretability] Revelio employs k-sparse autoencoders (k-SAE) to uncover monosemantic, interpretable features encoded across different layers and timesteps of diffusion models, and validates the transfer learning utility of these features via a lightweight classifier, Diff-C, enabling a systematic interpretation of black-box diffusion models.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Diffusion model interpretability"
  - "k-sparse autoencoders"
  - "representation learning"
  - "transfer learning"
  - "monosemantic features"
date: 2026-05-08
content_hash: 166bf04a90d762da
---

# Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models

**Conference**: ICCV 2025
**arXiv**: [2411.16725](https://arxiv.org/abs/2411.16725)  
**Code**: [GitHub](https://github.com/revelio-diffusion/revelio)  
**Area**: Diffusion Models · Representation Learning
**Keywords**: Diffusion model interpretability, k-sparse autoencoders, representation learning, transfer learning, monosemantic features

## TL;DR

Revelio employs k-sparse autoencoders (k-SAE) to uncover monosemantic, interpretable features encoded across different layers and timesteps of diffusion models, and validates the transfer learning utility of these features via a lightweight classifier, Diff-C, enabling a systematic interpretation of black-box diffusion models.

## Background & Motivation

### Limitations of Prior Work

**Background**: Diffusion models excel at generating high-quality images, yet how their internal representations encode rich visual semantic information remains poorly understood. The core research questions include:

1. What **granularity** of visual information is captured at different layers and timesteps?
2. What are the differences in **inductive biases** between convolutional UNet and Transformer DiT architectures?
3. How do pretraining data and language model conditioning affect representations?

Existing interpretation methods (e.g., attention map visualization, PCA) analyze only single images and lack holistic, systematic coverage.

## Method

### 1. k-Sparse Autoencoder (k-SAE)

A k-SAE is trained on intermediate features of pretrained diffusion models (spatially pooled to $\mathbf{x} \in \mathbb{R}^d$):

**Encoder**: $z = \text{TopK}(W_{enc}(x - b_{pre}) + b_{enc})$

The TopK activation retains the top $k$ largest neuron activations and sets the rest to zero ($k=32$).

**Decoder**: $\hat{x} = W_{dec} z + b_{pre}$

**Loss**: $L_{mse} = \|x - \hat{x}\|_2^2$

The semantic concept captured by each k-SAE neuron is revealed by inspecting its **highest-activating images**.

### 2. Diffusion Classifier (Diff-C)

A lightweight classifier consisting of 4 convolutional layers with progressive downsampling of diffusion features, followed by pooling and a fully connected layer. It validates the transfer learning effectiveness of diffusion features and runs $10^4$ times faster than the Diffusion Classifier, which requires a full denoising pass.

### Evaluation Metrics

- **Label purity $\sigma_{label}$**: Average standard deviation of class labels among the top 10 images for each of the top 1000 highly activating k-SAE features. Lower values indicate greater class discriminability.
- **GPT-4o evaluation**: Highly activating images are presented to GPT-4o as multiple-choice questions to assess semantic granularity.

## Key Experimental Results

### Information Granularity Across Layers

### Main Results

| Layer | Oxford-IIIT Pet σ↓ | Caltech-101 σ↓ |
|---|---|---|
| bottleneck | 9.48 | **9.35** |
| up_ft0 | 9.90 | 15.65 |
| **up_ft1** | **8.59** | 21.33 |
| up_ft2 | 9.67 | 25.61 |

- **Fine-grained task** (Pet breed classification): up_ft1 performs best, capturing breed-level features.
- **Coarse-grained task** (Caltech-101 object classification): bottleneck performs best, where shape information suffices.
- up_ft2 tends to capture high-frequency texture/pixel-level information, resulting in the poorest transferability.

### Effect of Timestep

### Ablation Study

| Timestep $t$ | Pet σ↓ (up_ft1) | Caltech-101 σ↓ (bottleneck) |
|---|---|---|
| 0 | 8.99 | 11.91 |
| **25** | **8.59** | 9.35 |
| 100 | 8.87 | 8.72 |
| **200** | 8.94 | **8.17** |
| 500 | 9.53 | 16.65 |

- Fine-grained task: $t=25$ (low noise) is optimal.
- Coarse-grained task: $t=200$ (moderate noise) is optimal; the additional noise may enhance feature generalization.

### SD 1.5 vs. SD 2.1

| Model | Pet σ↓ |
|---|---|
| **SD 1.5** | **8.59** |
| SD 2.1 | 9.67 |

SD 1.5 (using CLIP ViT-L/14 text encoder) outperforms SD 2.1 (using OpenCLIP ViT-H) on fine-grained classification.

### Analysis of DiT Blocks

| Block | Pet σ↓ |
|---|---|
| 6 | 10.18 |
| 10 | 9.44 |
| **14** | **9.05** |
| 18 | 9.55 |
| 22 | 9.84 |

The middle block of DiT (block 14) captures the most class-discriminative features, analogous to up_ft1 in UNet.

### Classification Performance

On Oxford-IIIT Pet, Diff-C (SD 1.5 up_ft1, $t=25$) achieves competitive classification accuracy with a minimal architecture, while running four orders of magnitude faster than the Diffusion Classifier.

## Highlights & Insights

1. **k-SAE is applied to visual diffusion models for mechanistic interpretation for the first time**, revealing monosemantic features.
2. **Interaction between representation granularity and task granularity**: different tasks benefit from features extracted from different layers and timesteps.
3. **Interesting architectural comparison**: convolutional UNet and Transformer DiT exhibit distinct inductive biases.
4. Later layers of diffusion models (up_ft2) focus more on pixel-level reconstruction and transfer poorly — analogous to the behavior of later layers in LLMs.

## Limitations & Future Work

- Interpretation of k-SAE features relies on manual inspection and is inherently subjective.
- GPT-4o-based granularity evaluation is noisy.
- Diff-C requires training a classifier (albeit lightweight) and is not truly zero-shot.
- Analysis of large-scale text-to-image models (e.g., SDXL, Flux) is absent.

## Related Work & Insights

- Diffusion features for discriminative tasks: DDAE, Diffusion Classifier, DiffusionDet
- Model interpretation: sparse autoencoders in LLMs, PCA analysis in Plug-and-Play diffusion
- CLIP feature interpretation: CLIP-SAE

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| **Overall** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Looking in the Mirror: A Faithful Counterfactual Explanation Method for Interpreting Deep Image Classification Models](looking_in_the_mirror_a_faithful_counterfactual_explanation_method_for_interpret.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](../../NeurIPS2025/image_generation/information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](../../NeurIPS2025/image_generation/information-theoretic_discrete_diffusion.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](../../ICLR2026/image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)

</div>

<!-- RELATED:END -->
