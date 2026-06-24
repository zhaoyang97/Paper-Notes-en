---
title: >-
  [Paper Note] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge
description: >-
  [CVPR 2025][Model Compression][Post-training quantization] This paper proposes QuartDepth, a post-training quantization framework for ASIC edge devices. By employing LogNP activation polishing (transforming abnormally distributed activation values into quantization-friendly distributions), activation quantization compensation (updating weights to compensate for activation quantization errors), and Fisher information-guided weight reconstruction…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Post-training quantization"
  - "depth estimation"
  - "edge deployment"
  - "ASIC accelerator"
  - "4-bit quantization"
date: 2026-05-08
content_hash: db0f61aeacdbd085
---

# QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge

**Conference**: CVPR 2025  
**arXiv**: [2503.16709](https://arxiv.org/abs/2503.16709)  
**Code**: [GitHub](https://github.com/shawnricecake/quart-depth)  
**Area**: 3D Vision  
**Keywords**: Post-training quantization, depth estimation, edge deployment, ASIC accelerator, 4-bit quantization

## TL;DR

This paper proposes QuartDepth, a post-training quantization framework for ASIC edge devices. By employing LogNP activation polishing (transforming abnormally distributed activation values into quantization-friendly distributions), activation quantization compensation (updating weights to compensate for activation quantization errors), and Fisher information-guided weight reconstruction, depth estimation foundation models are quantized to W4A4/W4A8. A programmable hardware accelerator is also designed to achieve real-time inference.

## Background & Motivation

- Foundation depth estimation models (such as Metric3D and DepthAnything) achieve excellent performance but require massive computation, making deployment on edge devices challenging.
- ASICs are ideal platforms for edge deployment, but they require low bit-width quantization (e.g., 4-bit) to fully utilize the hardware bandwidth.
- The datasets for large-scale foundation monocular depth estimation (MDE) models are massive, making full-model retraining impractical; thus, post-training quantization (PTQ) is required.
- Severe outlying activation distributions exist in the decoders of depth estimation models, characterized by large differences in outliers across channels and distributions that deviate significantly from normality.
- Per-tensor quantization cannot effectively handle the variability of outliers, while outliers still lead to significant quantization errors in per-channel quantization.
- Existing PTQ methods are primarily targeted at classification or language models, failing to specifically address the distribution characteristics of depth estimation models.
- Matrix multiplications and convolution operations account for the vast majority (~90%+) of inference time; therefore, quantizing these operations is critical.
- There is a lack of specialized hardware accelerator designs tailored for quantized depth estimation models.

## Method

### Overall Architecture

The QuartDepth pipeline consists of three steps: (1) First, apply LogNP polishing to transform the distribution of activation values, followed by activation quantization; (2) Update weights to compensate for the error introduced by activation quantization; (3) Quantize the updated weights using Fisher information-guided AdaRound. Concurrently, a flexible hardware accelerator supporting kernel fusion and custom instruction programmability is designed, containing dedicated compute cores for W4A4/W4A8 and a programmable vector computation array.

### Key Designs

**Design 1: LogNP Activation Polishing**
- **Function**: Transforms the anomalous activation distributions in the depth estimation decoder into quantization-friendly normal distributions.
- **Mechanism**: For the activation value $x$ in each channel $i$, a logarithmic transform $\Phi(x, \alpha) = \text{sign}(x) \cdot [\log_2(|x| + \alpha) - \log_2(\alpha)]$ is applied, where the polishing factor $\alpha_i = P_\epsilon(\mathbf{x}_i)$ is determined by the 95th percentile. After quantization, it is recovered via the inverse transform $\Phi^{-1}$. LogNP effectively compresses outliers while preserving the discriminability of the bulk distribution.
- **Design Motivation**: Directly quantizing activations containing outliers loses substantial information. Logarithmic transforms naturally compress larger values and expand smaller ones, rendering the distribution more concentrated and symmetric.

**Design 2: Activation Quantization Error Compensation**
- **Function**: Minimizes the output errors introduced by activation quantization by updating the weights.
- **Mechanism**: For each layer, the optimization problem $\min_{\Delta\mathbf{W}} \|\mathbf{W}\mathbf{x} - (\mathbf{W} + \Delta\mathbf{W})\hat{\mathbf{x}}\|_2^2$ is solved, yielding the closed-form solution $\Delta\mathbf{W}^* = -\mathbf{W}(\mathbf{x} - \hat{\mathbf{x}})\hat{\mathbf{x}}^T(\hat{\mathbf{x}}\hat{\mathbf{x}}^T)^{-1}$. A dampening technique is utilized when $\hat{\mathbf{x}}\hat{\mathbf{x}}^T$ is not of full rank.
- **Design Motivation**: By treating activation quantization and weight quantization separately, this design compensates for the activation error prior to quantizing the weights, minimizing the cumulative error over both stages.

**Design 3: Fisher Information-Guided Weight Reconstruction**
- **Function**: Minimizes the degradation in loss caused by weight quantization using second-order information.
- **Mechanism**: The impact of quantization error on the loss is approximated using a Taylor expansion as $\frac{1}{2}\Delta\mathbf{w}^T\mathbf{H}_\mathbf{w}\Delta\mathbf{w}$, and the layer-wise Fisher matrix is approximated using KFAC as $\mathbf{F}_l = \mathbf{G}_l \otimes \mathbf{A}_l$. Using this as the optimization objective for AdaRound, the rounding parameter $\mathbf{v}$ is learned to minimize $\sum_l (\mathbf{w}^{(l)} - \hat{\mathbf{w}}^{(l)})^T\mathbf{F}_l(\mathbf{w}^{(l)} - \hat{\mathbf{w}}^{(l)}) + \lambda h(\mathbf{v})$.
- **Design Motivation**: Traditional round-to-nearest ignores the varying sensitivity of different weights to the loss. The Fisher information matrix yields a more practical and computable second-order approximation than the Hessian.

### Loss & Training

Activation compensation: $\|\mathbf{W}\mathbf{x} - (\mathbf{W} + \Delta\mathbf{W})\hat{\mathbf{x}}\|_2^2$ (layer-wise closed-form solution); Weight reconstruction: Fisher-weighted AdaRound objective + regularization term, learning the rounding direction via gradient optimization.

## Key Experimental Results

### Main Results: Quantization Comparison of Depth Estimation on KITTI/NYUv2

| Model | Quantization Configuration | NYUv2 $\delta_1$↑ | NYUv2 AbsRel↓ | KITTI $\delta_1$↑ | KITTI AbsRel↓ |
|------|---------|-------------------|--------------|-------------------|---------------|
| Metric3D (ViT-L) FP32 | W32A32 | 0.977 | 0.064 | 0.975 | 0.052 |
| Metric3D (ViT-L) | W8A8 | 0.975 | 0.065 | 0.974 | 0.053 |
| Metric3D (ViT-L) | W4A8 | 0.970 | 0.069 | 0.970 | 0.056 |
| Metric3D (ViT-L) | **W4A4** | **0.960** | **0.076** | **0.963** | **0.061** |

### Ablation Study: Contribution of Each Component (Metric3D ViT-L, W4A4, NYUv2 $\delta_1$↑)

| Method | $\delta_1$↑ | AbsRel↓ |
|------|------------|---------|
| Baseline (Direct Quantization) | 0.891 | 0.118 |
| + LogNP polishing | 0.938 | 0.088 |
| + Activation Compensation | 0.949 | 0.081 |
| + Fisher Weight Reconstruction | **0.960** | **0.076** |

### Key Findings
- In the W4A8 configuration, $\delta_1$ degrades by only 0.7% (0.977 $\rightarrow$ 0.970), and W4A4 by 1.7%.
- LogNP polishing is the most critical component, improving the metric from 0.891 with direct quantization to 0.938 (+4.7%).
- Embodied Road Depth accuracy is independent of the choice of segmentation model (with a gap of <1% compared to GT segmentation).
- The ASIC hardware achieves real-time inference of 30+ FPS on the ViT-L model.
- Compared to AdaRound using only the MSE objective, Fisher-guided weight reconstruction delivers more accurate quantization.

## Highlights & Insights

1. **Intuitiveness of the LogNP Transform**: The logarithmic transform is naturally suited for handling long-tailed distributions, and the design of the percentile-adaptive polishing factor is simple and effective.
2. **Step-by-Step Decoupled Quantization Pipeline**: Succession of polishing $\rightarrow$ activation compensation $\rightarrow$ weight quantization, backed by clear mathematical derivations at each step.
3. **Hardware-Software Co-design**: The computational overhead of LogNP polishing is completely hidden by the parallel execution of the programmable vector computation array.
4. **Generality**: Applicable to multiple ViT-based depth estimation models (such as Metric3D and DepthAnything).

## Limitations & Future Work

- W4A4 still incurs a ~2% $\delta_1$ loss in certain scenarios, indicating that precision-sensitive applications may require W4A8.
- Currently, only linear and convolutional layers are quantized, while non-linear operations such as softmax in attention layers remain in floating-point format.
- The ASIC design is tailored to specific models, which limits its general applicability.
- Future work could explore mixed-precision quantization strategies or integrate knowledge distillation to further reduce accuracy degradation.

## Related Work & Insights

- Unlike SmoothQuant, which migrates outliers to weights, LogNP directly transforms the activation distribution.
- Fisher-guided weight reconstruction incorporates ideas from OBQ/GPTQ while avoiding the high overhead of column-by-column solving.
- The closed-form solution design for activation compensation can be generalized to the PTQ workflows of other models.

## Rating

⭐⭐⭐⭐ — A systematic quantization framework that forms a complete closed loop from problem analysis to solution and hardware design; LogNP polishing is a valuable technical contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[CVPR 2025\] Towards Practical Real-Time Neural Video Compression](towards_practical_real-time_neural_video_compression.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](../../AAAI2026/model_compression/quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)
- [\[ACL 2025\] PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](../../ACL2025/model_compression/ptq161_low_bit_quantization.md)

</div>

<!-- RELATED:END -->
