---
title: >-
  [Paper Note] DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling
description: >-
  [NeurIPS 2025 (Spotlight)][Image Generation][ConvNet] This paper finds that the global self-attention in pretrained DiTs primarily captures local patterns and thus exhibits substantial redundancy in generative tasks. It…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "Image Generation"
  - "ConvNet"
  - "diffusion models"
  - "channel attention"
  - "efficient generation"
  - "U-shaped architecture"
date: 2026-05-08
content_hash: 2a391c113d1e7992
---

# DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2505.11196](https://arxiv.org/abs/2505.11196)
**Code**: [https://github.com/shallowdream204/DiCo](https://github.com/shallowdream204/DiCo)
**Area**: Image Generation
**Keywords**: ConvNet, diffusion models, channel attention, efficient generation, U-shaped architecture

## TL;DR
This paper finds that the global self-attention in pretrained DiTs primarily captures local patterns and thus exhibits substantial redundancy in generative tasks. It proposes DiCo, a purely convolutional diffusion model built from standard convolution modules and a Compact Channel Attention (CCA) mechanism. DiCo achieves an FID of 2.05 on ImageNet-256, surpassing DiT-XL/2, with 2.7× faster inference at 256 resolution and 3.1× faster at 512 resolution.

## Background & Motivation

**Architectural evolution of diffusion models**: Diffusion models have transitioned from U-Net-based designs (ADM, Stable Diffusion) to fully Transformer-based architectures (DiT). DiT achieves outstanding performance on ImageNet generation and has become the backbone of mainstream models such as Stable Diffusion 3, FLUX, and Sora. However, the quadratic computational complexity of self-attention introduces severe bottlenecks at high resolutions, particularly at 512×512.

**Limitations of Prior Work on alternative architectures**: To mitigate this issue, a series of linear-complexity alternatives have been explored, including Mamba-based DiS/DiM and gated linear attention-based DiG. However, the causal design of these models conflicts with the inherently bidirectional nature of visual generation, and their practical speed advantages remain limited at high resolutions even with highly optimized CUDA implementations.

**A counterintuitive key finding**: The authors systematically analyze the attention maps of three representative pretrained DiT models—DiT-XL/2, PixArt-α, and FLUX—and uncover a surprising phenomenon: across nearly all layers, self-attention weights are heavily concentrated on spatial positions adjacent to the anchor token, with negligible weights assigned to distant tokens. This sharply contradicts the common understanding that attention captures global dependencies in visual recognition tasks, and suggests that global attention computation is largely redundant in generative tasks—local spatial modeling is essentially sufficient.

**Root Cause of naive replacement failure**: Motivated by this finding, a natural approach is to replace self-attention with convolutions, which are inherently suited for local pattern capture. However, direct substitution leads to degraded generation quality. Through channel activation score analysis, the authors identify the root cause as **channel redundancy** in ConvNets: compared to Transformers, a large proportion of channels in ConvNets exhibit near-zero activations, resulting in severely insufficient feature diversity. Self-attention, as a dynamic and content-dependent operation, naturally promotes channel diversification, whereas the static weights of convolutions lack this capacity. DiCo is designed to preserve the efficiency of convolutions while compensating for this representational gap via a lightweight channel attention mechanism.

## Method

### Overall Architecture
DiCo adopts a three-stage U-shaped architecture composed of stacked DiCo Blocks. The input image is encoded by a VAE encoder into a spatial representation $z$ (a 32×32×4 latent for 256×256 images), which is first mapped to an initial feature map $z_0$ with $D$ channels via a 3×3 convolution. Conditioning signals (timestep $t$ and class label $y$) are processed by an MLP and an embedding layer, respectively. Within each stage, intermediate features are passed from encoder to decoder via skip connections (concatenated and projected by a 1×1 convolution). Multi-scale processing across stages employs pixel-unshuffle downsampling and pixel-shuffle upsampling. The final output feature $z_L$ is normalized and passed through a 3×3 convolutional head to predict noise and covariance. The entire model contains no self-attention or cross-attention operations and is composed entirely of standard convolutional modules.

### Key Designs

1. **Conv Module**:

    - **Function**: Replaces the self-attention module in DiT for efficient spatial and channel feature extraction.
    - **Mechanism**: A 1×1 pointwise convolution aggregates cross-channel information per pixel, followed by a 3×3 depthwise convolution to capture within-channel spatial context, and a GELU nonlinearity. The full process is formulated as $Y = W_{p_2} \text{CCA}(\text{GELU}(W_d W_{p_1} X))$, where $W_{p_1}$ and $W_{p_2}$ are pointwise convolutions and $W_d$ is a depthwise convolution. Unlike modern recognition ConvNets that use large kernels (e.g., 31×31), DiCo relies solely on standard 1×1 and 3×3 convolutions, resulting in an extremely simple design.
    - **Design Motivation**: The attention locality analysis establishes that the effective receptive field in generative tasks is small; a 3×3 depthwise convolution is thus sufficient to capture the critical local spatial patterns while maintaining excellent hardware friendliness and inference efficiency.

2. **Compact Channel Attention (CCA)**:

    - **Function**: Dynamically activates more informative channels to resolve the channel redundancy of ConvNets and enhance feature diversity.
    - **Mechanism**: CCA first applies global average pooling (GAP) over the spatial dimensions to compress features into channel descriptors, then generates channel-wise attention weights via a 1×1 convolution and Sigmoid activation, and finally multiplies them element-wise with the input: $\text{CCA}(X) = X \odot \text{Sigmoid}(W_p \text{GAP}(X))$. This is a lightweight global channel modeling approach that introduces negligible computational overhead.
    - **Design Motivation**: Channel activation score analysis reveals that directly replacing attention with convolutions causes many channels to become "dead" (near-zero activations), resulting in far fewer effective feature channels than in Transformers. CCA employs data-adaptive channel reweighting to compel the network to activate more diverse channels, thereby restoring Transformer-level feature diversity. Experiments confirm that adding CCA significantly reduces channel redundancy.

3. **U-shaped Multi-scale Architecture**:

    - **Function**: Constructs a three-stage hierarchical encoder–decoder structure that leverages multi-scale feature representations to enhance denoising capability.
    - **Mechanism**: Unlike the isotropic architecture of DiT, DiCo adopts a U-shaped design with resolution changes between stages via pixel-shuffle/unshuffle. Features are transferred from encoder to decoder through skip connections, concatenated and fused by a 1×1 convolution. The model offers five variants—S/B/L/XL/H—with parameter counts aligned to the corresponding DiT scales, but with GFLOPs reduced to only 70.1%–74.6% of DiT.
    - **Design Motivation**: Multi-scale features play a critical role in image denoising—low-resolution features capture global structure while high-resolution features preserve fine-grained textures. Through ablation studies systematically comparing isotropic, isotropic with skip connections, and U-shaped architectures, the U-shaped design consistently achieves the best results across all model scales.

### Loss & Training
The standard diffusion training pipeline from DiT is followed. The noise predictor $\epsilon_\theta$ is trained with the simplified loss $\mathcal{L}_{simple}(\theta) = \|\epsilon_\theta(x_t) - \epsilon_t\|_2^2$, and the covariance $\Sigma_\theta$ is optimized with the full variational lower bound loss $\mathcal{L}$. Classifier-free guidance (CFG) is applied to enhance sample quality. Training configurations include a learning rate of $1 \times 10^{-4}$, batch size of 256, no weight decay, and EMA decay of 0.9999. For the largest model DiCo-H (1B parameters), the learning rate is increased to $2 \times 10^{-4}$ and batch size to 1024 to accelerate training.

## Key Experimental Results

### Main Results

| Model | Type | GFLOPs | Throughput (img/s) | FID↓ | IS↑ |
|------|------|--------|-------------|------|-----|
| DiT-XL/2 (w/ CFG) | Attn | 118.66 | 76.90 | 2.27 | 278.24 |
| DiG-XL/2 (w/ CFG) | Conv+Attn | 89.40 | 71.74 | 2.07 | 278.95 |
| **DiCo-XL (w/ CFG)** | **Conv** | **87.30** | **208.47** | **2.05** | **282.17** |
| DiCo-H (w/ CFG) | Conv | 194.15 | 117.57 | **1.90** | **284.31** |

**ImageNet 512×512 Results**:

| Model | GFLOPs | Throughput (img/s) | FID↓ | IS↑ |
|------|--------|-------------|------|-----|
| DiT-XL/2 (w/ CFG) | 524.70 | 18.58 | 3.04 | 240.82 |
| DiS-H/2 | - | 8.59 | 2.88 | 272.33 |
| **DiCo-XL (w/ CFG)** | **349.78** | **57.45** | **2.53** | **275.74** |

### Ablation Study

| Configuration | FID↓ | Notes |
|------|------|------|
| Isotropic architecture | 58.23 | DiT-style flat architecture |
| Isotropic + Skip | 54.10 | With long skip connections |
| **U-shaped architecture** | **49.97** | Three-stage hierarchical, best |
| Conv replacing attention (w/o CCA) | 62.06 | Many channels become "dead" |
| **Conv + CCA** | **49.97** | Channel redundancy significantly reduced |

**Per-scale comparison at 400K steps without CFG**:

| Scale | DiT FID | DiCo FID | FID Gain | Speedup |
|------|---------|----------|---------|---------|
| S | 68.40 | 49.97 | -18.43 | 1.37× |
| B | 43.47 | 27.20 | -16.27 | 2.17× |
| L | 23.33 | 13.66 | -9.67 | 2.51× |
| XL | 19.47 | 11.67 | -7.80 | 2.71× |

### Key Findings
- DiCo outperforms DiT at all model scales, with larger margins at greater scales.
- The speedup increases with resolution: 2.7× at 256 → 3.1× at 512, stemming from convolution's $O(n)$ vs. attention's $O(n^2)$ complexity.
- DiCo-XL is 6.7× faster than Mamba-based DiS-H/2 and 7.8× faster than DiM-H.
- DiCo-H (1B parameters) further pushes FID to 1.90, demonstrating the architecture's scalability.
- In MS-COCO text-to-image experiments, replacing cross-attention with dynamic depthwise convolutions remains competitive.

## Highlights & Insights
- **Revival of ConvNets in the Transformer era**: Against the prevailing assumption that Transformers are the optimal architecture for diffusion models, this work provides rigorous theoretical grounding for the return of ConvNets through attention locality analysis. The NeurIPS Spotlight recognition is well deserved.
- **The key insight of channel redundancy**: The paper precisely identifies the root cause of performance degradation when replacing attention with convolutions—not insufficient receptive field size, but inadequate channel diversity. The CCA module addresses this at negligible cost (only a GAP + 1×1 Conv + Sigmoid), and its elegance is remarkable.
- **Resolution-friendly efficiency advantage**: The $O(n)$ complexity of convolutions causes the speedup ratio to grow with resolution, making DiCo particularly valuable for high-resolution text-to-image applications.

## Limitations & Future Work
- The inherent locality of convolutions may limit performance in scenarios requiring global spatial relationship modeling, such as maintaining spatial layout consistency among complex objects.
- Validation is limited to class-conditional ImageNet and small-scale MS-COCO text-to-image experiments; training on large-scale text-to-image datasets has not been explored.
- Compatibility with the MM-DiT paradigm (e.g., multi-modal DiT in FLUX/SD3) remains unexplored.
- The dynamic depthwise convolution text injection scheme reshapes CLIP's 77 tokens (padded to 81) into a 9×9 kernel, and this fixed reshape may limit the flexibility of text conditioning.

## Related Work & Insights
- **vs. DiT**: DiT uses Transformers as the diffusion backbone; DiCo uses a pure ConvNet—achieving equivalent quality at 2.7× (256) to 3.1× (512) faster speeds, with GFLOPs reduced by 26.4%–33.3%.
- **vs. DiG**: DiG uses gated linear attention and still relies on global token mixing; DiCo completely abandons global interaction in favor of local convolutions, yet achieves better FID and higher inference speed.
- **vs. ConvNeXt**: ConvNeXt demonstrates that modern ConvNets can compete with ViTs on recognition tasks; DiCo extends this philosophy to generative tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Reestablishes the value of ConvNets in a Transformer-dominated era; the channel redundancy insight is precise and the solution is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparisons on ImageNet-256/512 with multi-scale ablations; large-scale text-to-image experiments are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — The logical chain from problem discovery → root cause identification → solution is clear and coherent.
- **Value**: ⭐⭐⭐⭐⭐ — NeurIPS Spotlight; opens a new ConvNet pathway for efficient diffusion model architecture design.

---

## Related Papers

- [\[CVPR 2025\] DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention](../../CVPR2025/image_generation/dig_scalable_and_efficient_diffusion_models_with_gated_linear_attention.md)
- [\[NeurIPS 2025\] GSPN-2: Efficient Parallel Sequence Modeling](gspn-2_efficient_parallel_sequence_modeling.md)
- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](../../ICML2025/image_generation/efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GSPN-2: Efficient Parallel Sequence Modeling](gspn-2_efficient_parallel_sequence_modeling.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](../../CVPR2026/image_generation/reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [\[NeurIPS 2025\] Flatten Graphs as Sequences: Transformers are Scalable Graph Generators](flatten_graphs_as_sequences_transformers_are_scalable_graph_generators.md)

</div>

<!-- RELATED:END -->
