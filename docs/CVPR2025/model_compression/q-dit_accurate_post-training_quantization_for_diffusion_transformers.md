---
title: >-
  [Paper Note] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers
description: >-
  [CVPR 2025][Model Compression][Diffusion Transformer Quantization] Proposed Q-DiT, a post-training quantization method for Diffusion Transformers (DiTs). It automatically allocates quantization group sizes via evolutionary search and utilizes sample-wise dynamic activation quantization, achieving high-fidelity image/video generation under the W4A8 configuration.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Diffusion Transformer Quantization"
  - "Post-Training Quantization"
  - "Dynamic Activation Quantization"
  - "Group-grain Allocation"
  - "DiT Acceleration"
date: 2026-05-08
content_hash: 1f1360c1326fd19b
---

# Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers

**Conference**: CVPR 2025  
**arXiv**: [2406.17343](https://arxiv.org/abs/2406.17343)  
**Code**: [GitHub](https://github.com/Juanerx/Q-DiT)  
**Area**: Image Generation / Model Compression  
**Keywords**: Diffusion Transformer Quantization, Post-Training Quantization, Dynamic Activation Quantization, Group-grain Allocation, DiT Acceleration

## TL;DR

Proposed Q-DiT, a post-training quantization method for Diffusion Transformers (DiTs). It automatically allocates quantization group sizes via evolutionary search and utilizes sample-wise dynamic activation quantization, achieving high-fidelity image/video generation under the W4A8 configuration.

## Background & Motivation

The architecture of diffusion models has evolved from UNet to Diffusion Transformers (DiT), such as Stable Diffusion 3 and Sora, significantly improving generation quality and scalability. However, DiT models are massive and their iterative denoising process is computationally intensive, which limits real-time deployment.

Post-training quantization (PTQ) is a model compression scheme that does not require retraining. However, existing PTQ methods (e.g., PTQ4DM, Q-diffusion, PTQD) are primarily designed for UNet architectures, and directly applying them to DiT leads to significant performance degradation. This is due to two unique characteristics of DiT: (1) **significant variance along the input channel dimension**—the distribution differences of weights and activations across input channels are much larger than those across output channels, with outliers concentrated in specific channels; (2) **activation distribution drift across timesteps**—activation distributions vary drastically across different denoising timesteps, and the variation patterns also differ among individual samples.

Existing reconstruction-based PTQ methods struggle to handle both the transformer architecture characteristics and the dynamic denoising processes. Q-DiT addresses both challenges simultaneously through automatic quantization granularity allocation and sample-wise dynamic quantization.

## Method

### Overall Architecture

Q-DiT performs group quantization on each layer of the DiT model, where group sizes are automatically allocated via evolutionary search, and activations are dynamically quantized at runtime. Uniform quantization is used for weights and activations: $\hat{\mathbf{x}} = s \cdot (\text{clip}(\lfloor \frac{\mathbf{x}}{s}\rceil + Z, 0, 2^b - 1) - Z)$.

### Key Designs

1. **Automatic Quantization Granularity Allocation**:
    - **Function**: Automatically determines the optimal quantization group size for each layer.
    - **Mechanism**: Split matrices along input channels into groups of size $g_{ll}$, calibrating each independently. Key Finding: The relationship between group size and performance is **non-monotonic**—decreasing the group size (increasing the number of groups) does not necessarily yield better results (e.g., shrinking from 128 to 96 actually degrades FID from 17.87 to 19.97). Thus, an evolutionary algorithm is employed to search for the layer-wise group size configuration $\mathbf{g}^* = \arg\min_\mathbf{g} \text{FID}(R, G_\mathbf{g})$, subject to a constraint that BitOps do not exceed a preset threshold $N_{bitops}$. Final generation metrics (FID/FVD) are directly used as the search objective rather than layer-wise MSE.
    - **Design Motivation**: Fixed group sizes are sub-optimal; layer-wise MSE does not necessarily correlate with the final generation quality (due to non-monotonicity); and evolutionary search can directly optimize final generation metrics.

2. **Sample-wise Dynamic Activation Quantization**:
    - **Function**: Adapts to changes in activation distributions across different timesteps and individual samples.
    - **Mechanism**: Rather than using a calibration set to pre-compute fixed activation quantization parameters, quantization parameters (scale factors and zero points) are dynamically computed at runtime based on the actual activation distribution of each sample. For the input activation of each linear layer, min/max statistics are calculated at the group level to instantly determine quantization parameters. The computational overhead is minimal—only requiring extra min/max statistic collection.
    - **Design Motivation**: Quantization parameters calibrated at fixed timesteps fail to generalize across all timesteps; activations in DiT exhibit significant variance across both time and sample dimensions, making static quantization inherently prone to large errors.

3. **Integration with DiT Characteristic Analysis**:
    - **Function**: Guides the design of quantization strategies through in-depth analysis of DiT model characteristics.
    - **Mechanism**: (i) Observed that the variance of DiT weights and activations along the input channel direction is much larger than that of output channels, proving the necessity of group quantization along input channels; (ii) observed that standard deviations of activations fluctuate drastically across different blocks and timesteps (particularly in MLP layers), demonstrating the necessity of dynamic quantization; (iii) found that outliers are concentrated in specific channels, which group quantization can effectively handle at the group level.
    - **Design Motivation**: Data-driven method design—making empirical observations first, and then designing target-oriented solutions.

### Loss & Training

- **No Training Required**: It is a PTQ method that does not require retraining or fine-tuning.
- **Evolutionary Search**: Population Initialization $\rightarrow$ Mutation/Crossover $\rightarrow$ FID/FVD Evaluation $\rightarrow$ Selection $\rightarrow$ Iteration.
- **Search Space**: Group sizes are selected from $\{32, 64, 96, 128, ...\}$.
- **Quantization Configuration**: W6A8 (weight 6-bit, activation 8-bit) and W4A8 (weight 4-bit, activation 8-bit).

## Key Experimental Results

### Main Results

DiT-XL/2 on ImageNet 256×256 (W4A8 / W6A8):

| Method | W/A | FID↓ | sFID↓ | IS↑ | Description |
|------|-----|------|-------|-----|------|
| Full Precision | 32/32 | 9.62 | 6.17 | 278.24 | Baseline |
| Q-DiT (W6A8) | 6/8 | ~10 | ~6.5 | ~275 | Near-lossless |
| **Q-DiT (W4A8)** | **4/8** | **Lower** | — | — | Lowest FID |
| PTQ4DiT (W4A8) | 4/8 | +1.09 | — | — | Q-DiT reduces by 1.09 |

Group size non-monotonicity verification (ImageNet 256×256, W4A8):

| Group Size | FID↓ | sFID↓ |
|--------|------|-------|
| 128 | 17.87 | 20.45 |
| 96 | 19.97 | 21.42 |

### Ablation Study

| Configuration | Description |
|------|------|
| Fixed Group Size (128) | Baseline, higher FID |
| Automatic Group Size Allocation | FID significantly reduced |
| Static Activation Quantization | Large quantization error due to timestep variations |
| Dynamic Activation Quantization | Significant improvement, particularly on high-variance timesteps |
| Both Combined | Best representation |

### Key Findings

- Achieves nearly lossless compression under W6A8, while maintaining high-fidelity generation under W4A8.
- The non-monotonicity of group sizes is a unique phenomenon in DiT quantization, distinguishing it from LLM/ViT quantization.
- Directly optimizing FID via evolutionary search is more effective than layer-wise MSE-based methods.
- The extra overhead of sample-wise dynamic quantization is negligible but yields significant improvements.
- The method generalizes well to both image (ImageNet) and video (VBench) generation.

## Highlights & Insights

- **Deep Analysis of Unique DiT Quantization Characteristics**: It is the first to systematically point out the fundamental differences between DiT and UNet regarding input channel variance and timestep activation drift, providing empirical foundations for subsequent DiT quantization research.
- **Discovery of Group Size Non-monotonicity**: This is highly inspiring, breaking the intuition that "finer-grained quantization is always better" and demonstrating that DiT exhibits complex quantization behaviors that require data-driven search rather than manual heuristics.

## Limitations & Future Work

- The evolutionary search requires generating multiple samples to compute FIDs, incurring high search costs.
- Validated only on limited models like DiT-XL/2; effectiveness on larger-scale DiTs has yet to be verified.
- Dynamic quantization requires runtime min/max computation during deployment, which may affect latency.
- The integration with orthogonal compression techniques like weight pruning and knowledge distillation has not been explored.

## Related Work & Insights

- **vs PTQ4DiT**: PTQ4DiT uses channel saliency balancing and Spearman calibration, but does not consider non-monotonic group sizes or sample-wise dynamics. Q-DiT reduces FID by 1.09 under W4A8.
- **vs Q-diffusion**: Q-diffusion is designed for UNet and uses layer-wise reconstruction, which scales poorly to massive models. Q-DiT does not rely on reconstruction and directly optimizes generation metrics.
- **vs LLM Quantization (AWQ, GPTQ)**: LLM quantization methods assume performance scales monotonically with group size, which does not hold for the non-monotonic behavior in DiT.

## Rating

- Novelty: ⭐⭐⭐⭐ In-depth analysis of DiT quantization characteristics; highly targeted designs for automatic group allocation and dynamic quantization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual validation on both ImageNet and VBench, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear reasoning transitioning from observations to methods.
- Value: ⭐⭐⭐⭐ Provides a practical quantization solution for DiT deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [\[ICLR 2026\] SliderQuant: Accurate Post-Training Quantization for LLMs](../../ICLR2026/model_compression/sliderquant_accurate_post-training_quantization_for_llms.md)
- [\[ICLR 2026\] Gradient-Aligned Calibration for Post-Training Quantization of Diffusion Models](../../ICLR2026/model_compression/gradient-aligned_calibration_for_post-training_quantization_of_diffusion_models.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)

</div>

<!-- RELATED:END -->
