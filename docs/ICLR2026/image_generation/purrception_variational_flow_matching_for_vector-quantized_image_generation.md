---
title: >-
  [Paper Note] Purrception: Variational Flow Matching for Vector-Quantized Image Generation
description: >-
  [ICLR 2026][Image Generation][variational flow matching] This paper proposes Purrception, an image generation method that adapts Variational Flow Matching (VFM) to vector-quantized (VQ) latent spaces. By simultaneously c…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "variational flow matching"
  - "vector quantization"
  - "discrete diffusion"
  - "categorical posterior"
date: 2026-05-08
content_hash: 8fe67aec25c8ccfe
---

# Purrception: Variational Flow Matching for Vector-Quantized Image Generation

**Conference**: ICLR 2026
**arXiv**: [2510.01478](https://arxiv.org/abs/2510.01478)
**Code**: None
**Area**: Image Generation
**Keywords**: variational flow matching, vector quantization, discrete diffusion, categorical posterior, image generation

## TL;DR

This paper proposes Purrception, an image generation method that adapts Variational Flow Matching (VFM) to vector-quantized (VQ) latent spaces. By simultaneously computing a velocity field in the continuous embedding space and learning a categorical posterior distribution over codebook indices, Purrception bridges continuous transport dynamics with discrete supervision, achieving faster training convergence and FID scores competitive with state-of-the-art methods on ImageNet-1k 256×256.

## Background & Motivation

The field of image generation is undergoing a fundamental paradigm shift. Performing generation within a latent space has become the dominant approach, and **how to model the generative process within that latent space** is a central design choice. Two major technical paradigms currently exist, each with distinct trade-offs:

### Continuous Methods

Exemplified by Flow Matching and diffusion models, these methods define transport paths from noise to data in a continuous space.

- **Advantages**: Geometric awareness; transport paths exhibit favorable mathematical properties in continuous space; gradient estimates are smooth.
- **Disadvantages**: Cannot provide explicit supervision signals over discrete codebook indices; a fundamental mismatch arises when the underlying latent space is discrete (e.g., the codebook of a VQ-VAE).

### Discrete Methods

Exemplified by Discrete Flow Matching and masked language models, these methods directly model generation in discrete token space.

- **Advantages**: Provide explicit categorical supervision over codebook indices, naturally compatible with VQ latent spaces.
- **Disadvantages**: Lack geometric structural information from continuous space, potentially leading to less efficient training.

The core motivation of this paper is: **can the advantages of both approaches be combined?** Specifically, can one maintain continuous transport dynamics while simultaneously providing explicit categorical supervision over discrete codebook indices?

Variational Flow Matching (VFM) offers a natural framework for this bridging — it extends continuous flow matching with variational inference, enabling the definition of posterior distributions over discrete variables. Purrception represents the first attempt to adapt VFM for VQ-based image generation.

## Method

### Overall Architecture

Purrception's generative pipeline operates over the latent space of a **vector-quantized autoencoder (VQ-VAE/VQ-GAN)**. Unlike existing approaches, Purrception operates simultaneously at two levels:

- **Continuous level**: A velocity field is defined and learned in the continuous vector space of codebook embeddings, transporting samples from a noise distribution to the data distribution.
- **Discrete level**: A categorical posterior distribution is learned over codebook indices, providing a discrete supervision signal.

These two levels are **jointly modeled** within the variational flow matching framework, sharing underlying parameters and the optimization process.

### Key Designs

1. **Learning the Categorical Posterior**:

    - *Function*: Learn a probability distribution over codebook indices at each spatial location, rather than making a deterministic selection.
    - *Mechanism*: Given the current intermediate state (a point along the noise-to-data transport path), the model predicts the most probable distribution over codebook indices at each location, parameterized as a categorical distribution.
    - *Design Motivation*:
        - Multiple semantically similar codebook entries may exist in a VQ codebook; deterministic selection can introduce unnecessary hard-decision noise.
        - The categorical posterior provides **uncertainty quantification** — the model can express ambiguity such as "this position is likely code 42 or code 87."
        - This probabilistic treatment yields smoother training signals, facilitating faster convergence.

2. **Velocity Field in Continuous Space**:

    - *Function*: Define and learn a velocity field from noise to data in the continuous vector space of codebook embeddings.
    - *Mechanism*: The core mathematical framework of Flow Matching is applied to the VQ embedding space. The velocity field describes the "flow direction" along which samples are transported from the noise distribution toward the data distribution.
    - *Design Motivation*:
        - A continuous velocity field preserves the geometric advantages of Flow Matching — smooth transport paths and stable training.
        - Operating in the embedding space rather than the index space avoids the difficulty of gradient estimation in discrete token spaces.
        - Complements the discrete supervision from the categorical posterior, forming a dual learning signal.

3. **Temperature-Controlled Generation**:

    - *Function*: Control the diversity–quality trade-off in generation by adjusting the temperature parameter of the categorical posterior.
    - *Mechanism*:
        - Low temperature ($T \to 0$): The categorical posterior approaches a deterministic selection, producing more certain outputs with reduced diversity.
        - High temperature ($T > 1$): The posterior becomes more uniform, exploring more codebook combinations and increasing diversity at a potential quality cost.
        - Intermediate temperature: Achieves a balance between quality and diversity.
    - *Design Motivation*: This is a natural advantage of probabilistic models — the temperature parameter provides an interpretable and controllable generation mechanism without requiring additional post-processing steps (e.g., a classifier-free guidance scale).

4. **Adapting Variational Flow Matching**:

    - *Function*: Adapt the original VFM framework to the specific structure of vector-quantized latent spaces.
    - *Key Adaptations*:
        - **Exploiting codebook structure**: VFM is originally defined over general continuous/discrete mixed spaces; Purrception leverages the finite discrete structure of the codebook to efficiently parameterize the categorical posterior.
        - **Exploiting embedding geometry**: Geometric relationships among codebook embedding vectors (e.g., Euclidean distances) are used to define targets for the continuous transport path.
        - **Joint optimization**: Parameters for the categorical posterior and the velocity field are learned jointly within a single model, sharing underlying visual feature representations.

### Loss & Training

The training objective comprises two complementary components:

1. **Continuous Flow Matching Loss**: A standard velocity field regression loss that drives the model to learn the correct transport direction:
$$\mathcal{L}_{FM} = \mathbb{E}\left[\|v_\theta(x_t, t) - u(x_t | x_1)\|^2\right]$$

2. **Categorical Posterior Loss**: A cross-entropy loss that drives the model to correctly predict the distribution over codebook indices at each spatial location:
$$\mathcal{L}_{cat} = -\mathbb{E}\left[\sum_k q(k|x_t) \log p_\theta(k|x_t, t)\right]$$

Both losses are naturally unified within the variational flow matching framework and optimized jointly.

## Key Experimental Results

### Main Results

Evaluation on unconditional/class-conditional image generation on ImageNet-1k 256×256:

| Method | FID↓ | Training Convergence | Type |
|--------|------|----------------------|------|
| Continuous Flow Matching | Baseline | Slow | Pure continuous |
| Discrete Flow Matching | Baseline | Slow | Pure discrete |
| **Purrception** | **Competitive with SOTA** | **Faster** | Continuous + discrete bridge |
| Other SOTA models | Reference | — | Various |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Continuous FM only | Worse FID, slow convergence | No discrete supervision |
| Discrete supervision only | Worse FID | No continuous geometric information |
| Purrception (full) | Best FID + fastest convergence | Dual signals complement each other |
| Temperature = 0.5 | High quality, low diversity | Low-temperature deterministic selection |
| Temperature = 1.0 | Quality–diversity balance | Standard setting |
| Temperature = 1.5 | High diversity, slight quality drop | High-temperature softened posterior |
| Without categorical posterior | Slower convergence | Validates the acceleration effect of discrete supervision |

### Key Findings

1. **Accelerated Training Convergence**: Purrception converges faster than both the pure continuous flow matching and pure discrete flow matching baselines. The discrete supervision signal from the categorical posterior provides a sharper learning target, improving the directionality of parameter updates.

2. **Quality Competitive with SOTA**: FID scores are competitive with current state-of-the-art methods, demonstrating that bridging continuous and discrete approaches does not sacrifice generation quality.

3. **Temperature Controllability**: A single temperature parameter smoothly controls the diversity–quality trade-off, providing an intuitive generation adjustment mechanism.

4. **Uncertainty Quantification**: The categorical posterior naturally provides uncertainty estimates over possible codebook assignments at each spatial location, a capability generally unavailable in alternative methods.

## Highlights & Insights

1. **Elegant Theoretical Bridging**: Purrception elegantly bridges continuous and discrete generative paradigms through the variational flow matching framework. Rather than naively combining two losses, it achieves a natural fusion within a unified variational inference framework.

2. **Practical Speed Improvements**: The accelerated training convergence is a highly practical contribution — at the scale of ImageNet training, gains in training efficiency directly translate to savings in GPU hours and associated costs.

3. **Probabilistic Codebook Selection**: Replacing the deterministic nearest-codebook-entry lookup (argmin distance) with a probabilistic categorical posterior not only improves training efficiency but also introduces uncertainty quantification. This "soft quantization" paradigm has broader potential within the VQ research community.

4. **Interpretability of Temperature Control**: Compared to the guidance scale in classifier-free guidance, the temperature parameter carries a more intuitive physical meaning — it directly controls the "sharpness" of the posterior distribution — and is therefore more readily interpretable.

5. **Naming Ingenuity**: "Purrception" (Purr, the sound of a cat, + Perception) is a clever name, suggesting that the model's "perception" of the codebook is gentle (a purr) rather than a hard decision.

## Limitations & Future Work

1. **Validation Limited to ImageNet 256×256**: The current experimental scope is relatively narrow. Performance at higher resolutions (e.g., 512×512 or 1024×1024) and on larger-scale datasets has not been verified.

2. **Gap with the Latest SOTA**: Although FID is described as "competitive with SOTA," a gap may still exist. More detailed quantitative comparisons are needed for accurate positioning.

3. **Sensitivity to Codebook Size**: The computational complexity of the categorical posterior scales linearly with codebook size. For very large codebooks (e.g., 16,384 entries), efficiency may become a concern.

4. **Text-Conditional Generation**: The method is primarily validated on class-conditional generation; its effectiveness for text-to-image generation remains unexplored.

5. **Compatibility with Modern Encoder Architectures**: How Purrception integrates with the latest VQ encoders (e.g., improved VQGAN, FSQ) and continuous latent VAEs (e.g., the KL-VAE used in Stable Diffusion) warrants further investigation.

6. **Extension to Video Generation**: Extending the Purrception framework to video VQ latent spaces may yield additional benefits for temporal consistency.

## Related Work & Insights

- **Flow Matching**: Lipman et al.'s Flow Matching framework, which provides the theoretical foundation for continuous transport paths.
- **Variational Flow Matching**: The framework that introduces variational inference into Flow Matching, serving as the theoretical cornerstone of Purrception.
- **Discrete Flow Matching / Discrete Diffusion**: Generative modeling directly in discrete token space.
- **VQ-VAE / VQ-GAN**: Methods for constructing vector-quantized latent spaces, providing the underlying space in which Purrception operates.
- **Masked Image Modeling (MIM)**: Methods such as MaskGIT that employ masked prediction over VQ tokens.
- **Autoregressive VQ Generation**: Methods such as VQVAE + Transformer that generate VQ tokens sequentially.
- Broader Inspiration: **Variational inference as a unifying framework** enables a single model to simultaneously handle continuous and discrete structures. This paradigm may prove valuable in other generative tasks involving mixed continuous-discrete spaces, such as molecular generation and program synthesis.

## Rating

- Novelty: ⭐⭐⭐⭐ (Adapting variational flow matching to VQ spaces is a natural yet non-trivial contribution with a clear bridging rationale.)
- Experimental Thoroughness: ⭐⭐⭐ (Validation is limited to ImageNet-1k 256×256; broader benchmarks and comparisons are needed.)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear and motivations are well-articulated.)
- Value: ⭐⭐⭐⭐ (The training efficiency gains are practically meaningful, and the method introduces a new paradigm for VQ-based generation; broader impact depends on the continued development of VQ approaches.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[ICLR 2026\] Diffusion Alignment as Variational Expectation-Maximization](diffusion_alignment_as_variational_expectation-maximization.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)
- [\[CVPR 2026\] Unified Vector Floorplan Generation via Markup Representation](../../CVPR2026/image_generation/unified_vector_floorplan_generation_via_markup_representation.md)

</div>

<!-- RELATED:END -->
