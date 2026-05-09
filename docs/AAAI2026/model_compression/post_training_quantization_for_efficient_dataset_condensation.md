---
title: >-
  [Paper Note] Post Training Quantization for Efficient Dataset Condensation
description: >-
  [AAAI 2026][Model Compression][Dataset Distillation] This work is the first to apply post-training quantization (PTQ) to dataset distillation, proposing a patch-based quantization framework (PAQ + grouping + refinement) that nearly doubles test accuracy of distilled datasets at the extreme 2-bit regime (e.g., DM IPC=1 improves from 26.0% to 54.1%). The framework is plug-and-play and can be applied to various distillation methods.
tags:
  - AAAI 2026
  - Model Compression
  - Dataset Distillation
  - Post-Training Quantization
  - Low-Bit Storage
  - Image Compression
  - Patch Quantization
date: 2026-05-08
content_hash: 993fd633443c404d
---

# Post Training Quantization for Efficient Dataset Condensation

**Conference**: AAAI 2026
**arXiv**: [2603.13346](https://arxiv.org/abs/2603.13346)
**Code**: None
**Area**: Model Compression
**Keywords**: Dataset Distillation, Post-Training Quantization, Low-Bit Storage, Image Compression, Patch Quantization

## TL;DR
This work is the first to apply post-training quantization (PTQ) to dataset distillation, proposing a patch-based quantization framework (PAQ + grouping + refinement) that nearly doubles test accuracy of distilled datasets at the extreme 2-bit regime (e.g., DM IPC=1 improves from 26.0% to 54.1%). The framework is plug-and-play and can be applied to various distillation methods.

## Background & Motivation

### State of the Field
Dataset Condensation (DC) accelerates training and reduces storage by distilling the knowledge of a large dataset into a small synthetic one. Existing methods (gradient matching, distribution matching, trajectory matching) primarily focus on generation quality while neglecting storage efficiency — each synthetic sample is still stored at full precision. Parameterized DC (PDC) methods such as IDC (spatial downsampling), AutoPalette (color reduction), and DDiF (neural fields) improve compression ratios but still rely on 32-bit representations.

### Limitations of Prior Work

**Storage redundancy**: Synthetic images are stored as 32-bit floats, wasting substantial space.

**High computational cost of existing PDC methods**: AutoPalette requires training a palette encoder; DDiF requires a neural field network with decoding at inference time.

**Bit-level redundancy unexploited**: Under the same storage budget, reducing bit-width allows more samples to be stored, yet full-image quantization degrades severely at extremely low bit-widths.

**PTQ never explored for DC**: Despite widespread adoption of PTQ in model compression, its potential for compressing synthetic data has remained entirely unexplored.

### Root Cause
How can synthetic images be quantized at extremely low bit-widths (e.g., 2-bit) while preserving their effectiveness for downstream model training?

### Starting Point
The paper proposes patch-level quantization: images are divided into non-overlapping patches, each quantized independently to preserve local detail. Clustering-based grouping shares quantization parameters across similar patches to reduce overhead, and a refinement module aligns feature distributions before and after quantization.

## Method

### Overall Architecture
Pipeline: (I) Synthetic images → quantization-aware refinement → (II) patch extraction → k-means clustering (over quantization parameters) → grouped quantization → (III) intra-group quantization + entropy coding → final compressed dataset. At inference: decode → dequantize → directly used for training.

### Key Designs

1. **Patch-wise Asymmetric Quantization (PAQ)**:

    - An image $x$ is divided into $P$ non-overlapping patches $\{x_i\}_{i=1}^P$, where $x_i \in \mathbb{R}^{h \times w \times C}$.
    - Each patch is quantized independently: $x_i^q = Q(x_i, \theta_i)$, with $\theta_i = (\alpha_i, z_i)$.
    - Asymmetric quantization formulas:
        - Scale factor: $\alpha = \frac{\max(x) - \min(x)}{Q_{max} - Q_{min}}$
        - Zero point: $z = \lfloor Q_{min} - \frac{\min(x)}{\alpha} \rceil$
        - Quantization / dequantization: $x^q = \lfloor \frac{x}{\alpha} + z \rceil$, $x^{deq} = (x^q - z) \cdot \alpha$
    - Compared to full-image quantization: PAQ achieves 47.5% at 2-bit vs. 48.9% at full precision — nearly lossless.
    - Design Motivation: Full-image quantization uses a single parameter set for the entire image and cannot adapt to spatial variation in texture and detail.

2. **Grouping-based Asymmetric Quantization (GAQ)**:

    - PAQ stores independent parameters per patch, increasing storage overhead.
    - k-means clustering is performed in the quantization parameter space $(\alpha_i, z_i)$.
    - Objective: minimize intra-group variance of quantization parameters.
    - $\{\mathcal{C}_g^*, \theta_g^*\}_{g=1}^G = \arg\min \sum_{g=1}^G \sum_{\theta_i \in \mathcal{C}_g} \|\theta_i - \hat{\theta}_g\|^2$
    - **Intra-group recalibration**: Rather than using cluster centers directly as quantization parameters, all patches within a group are concatenated and quantization parameters are recomputed.
    - $x_g = \text{concat}(\{x_i\}_{i \in \mathcal{C}_g})$; $\theta_g$ is calibrated on the flattened $x_g^{flat}$.
    - Design Motivation: Balances storage overhead and quantization quality — similar patches share parameters.

3. **Quantization-Aware Refinement Module**:

    - A refined image $x^{ft}$ is optimized so that its features after quantization align with those of the original image.
    - Feature extraction: $\mathbf{f} = f(x)$, $\tilde{\mathbf{f}} = f((x^{ft})^{deq})$.
    - Feature-space MSE minimization: $\mathcal{L}_{quant} = \mathbb{E}_{x \sim S}[\|\mathbf{f} - \tilde{\mathbf{f}}\|_2^2]$
    - Three strategies: (1) refinement before grouping only, (2) refinement after grouping only, (3) refinement both before and after.
    - Experiments show that pre-grouping refinement performs best, as it provides more accurate quantization parameters for subsequent grouping.
    - Design Motivation: Directly compensates for feature drift caused by quantization noise.

4. **Storage Measurement and Entropy Coding**:

    - Total storage = group indices $\mathcal{G}$ + quantization parameters $\mathcal{Q}$ + quantized images $\mathcal{X}^q$.
    - Entropy coding (EC) is additionally applied to $\mathcal{X}^q$ to exploit statistical redundancy.
    - Constraint: $size(\mathcal{G}) + size(\mathcal{Q}) + size(EC(\mathcal{X}^q)) \leq size(\text{IPC})$
    - Under the same budget, more quantized samples can be stored, increasing the representational density of the dataset.

### Loss & Training
- Default setting: 2-bit quantization, 5×5 non-overlapping patches.
- Grid search determines the maximum number of groups satisfying the storage constraint.
- Refinement iterations: 500 for CIFAR-10/100, 2000 for ImageNet subsets.
- Evaluation: models are trained on the compressed dataset and tested on the original test set.
- Plug-and-play: applicable to synthetic images generated by various distillation methods including DM, DSA, and DATM.

## Key Experimental Results

### Main Results

| Method | CIFAR-10 IPC=1 | IPC=10 | IPC=50 | CIFAR-100 IPC=1 | IPC=10 |
|--------|---------------|--------|--------|----------------|--------|
| DM | 26.0 | 48.9 | 63.0 | 11.4 | 29.7 |
| DSA | 28.8 | 52.1 | 60.6 | 13.9 | 32.3 |
| DATM | 46.9 | 66.8 | 76.1 | 27.9 | 47.2 |
| AutoPalette | 58.6 | 74.3 | 79.4 | 38.0 | 52.6 |
| **DM+Ours** | **54.1** | 68.2 | 77.1 | 34.0 | 51.2 |
| **DSA+Ours** | **55.3** | 58.3 | 73.4 | 34.7 | 41.1 |
| **DATM+Ours** | **68.9** | **79.0** | **83.8** | **48.0** | **56.5** |

| Dataset | I-Nette | I-Woof | I-Fruit | I-Meow | I-Squawk | I-Yellow |
|---------|---------|--------|---------|--------|----------|----------|
| DATM | 65.8 | 38.8 | 41.2 | 45.7 | 56.3 | 61.1 |
| AutoPalette | 73.2 | 44.3 | 48.4 | 53.6 | 68.0 | 72.0 |
| **DATM+Ours** | **81.1** | **53.0** | **56.6** | **61.2** | **80.6** | **78.9** |

### Ablation Study

| GAQ | Refinement | EC | CIFAR-10 (IPC=10) | I-Nette |
|-----|-----------|----|--------------------|---------|
| ✗ | ✗ | ✗ | 71.8 | 75.2 |
| ✓ | ✗ | ✗ | 76.1 (+4.3) | 76.5 |
| ✓ | ✓ | ✗ | 77.2 (+1.1) | 77.2 |
| ✓ | ✓ | ✓ | **79.0** (+1.8) | **81.1** |

| Refinement Timing | CIFAR-10 IPC=1 | Note |
|-------------------|---------------|------|
| Before grouping only | **68.9** | Best |
| After grouping only | 68.7 | Slightly worse |
| Both before and after | 68.9 | No additional gain |

### Key Findings

1. **Doubled performance under extreme compression**: DM IPC=1 improves from 26.0%→54.1% and DSA from 28.8%→55.3%, demonstrating that 2-bit patch quantization is highly effective under extremely low storage budgets.
2. **DATM+Ours achieves overall SOTA**: Surpasses PDC methods such as AutoPalette across all IPC settings and datasets without requiring any additional networks.
3. **Clear contribution of each component**: AQ→GAQ (+4.3)→refinement (+1.1)→EC (+1.8); each step contributes positively.
4. **Cross-architecture generalization**: Substantially outperforms the DATM baseline on ConvNet, AlexNet, VGG11, and ResNet18.
5. **Cross-modality generalization**: Effective on audio data (MobileNet/SqueezeNet) and 3D voxel data as well.
6. **Cross-dataset generalization**: Advantages are especially pronounced on large-scale real-world datasets such as CC3M and Places365.
7. **Refinement timing analysis**: Pre-grouping refinement performs best, as it supplies more accurate quantization parameters to the subsequent grouping step.
8. **Visualization comparison**: Median Cut preserves texture but loses color; AQ preserves color but loses texture; GAQ achieves a better balance between the two.

## Highlights & Insights
1. This work is the first to introduce PTQ into dataset distillation, opening an entirely new research direction.
2. The design intuition behind patch-level quantization is clear: local adaptation to spatial variation preserves more detail than global quantization.
3. The grouping strategy clusters in quantization parameter space rather than pixel space, precisely capturing similarity in quantization behavior.
4. The plug-and-play design enables combination with any distillation method, offering high practical utility.
5. Sustained effectiveness at the extreme 2-bit regime demonstrates that information in synthetic images is highly compressible.
6. The storage budget is formulated as an explicit constrained optimization problem, facilitating engineering practice.

## Limitations & Future Work
1. Patch size is fixed at 5×5; adaptive patch sizes are worth exploring.
2. The number of k-means groups is determined via grid search; more efficient automatic selection methods could be developed.
3. The refinement module relies on a neural network for feature extraction, introducing a dependency on the choice of network.
4. Validation is limited to 2-bit and 4-bit; other bit-widths (e.g., 3-bit) warrant further investigation.
5. Testing on CIFAR-100 IPC=50 is infeasible as it exceeds the original 500 images per class limit.
6. Comparison with learning-based compression methods (e.g., variational autoencoders) is absent.

## Related Work & Insights
- **AutoPalette (Yuan 2024a)**: color redundancy reduction → this work directly reduces bit-level redundancy, operating at a more fundamental and general level.
- **DDiF (Shin 2025)**: neural field encoding → high computational cost; the proposed PTQ requires no additional networks.
- **IDC (Kim 2022)**: spatial downsampling → patch-level quantization in this work preserves more spatial information.
- **Broad adoption of PTQ in model compression** → first transferred to data compression in this work.
- SPEED, FreD, Spectral, and other frequency/spectral-domain methods → still use 32-bit representation; this work reduces directly to 2-bit.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First application of PTQ to dataset distillation, pioneering a new research direction)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets, distillation methods, architectures, modalities, and comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and good visualizations, though some formula formatting could be improved)
- Value: ⭐⭐⭐⭐⭐ (Plug-and-play framework with doubled performance under extreme compression; high practical utility)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training](eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)
- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](../../ICLR2026/model_compression/ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/model_compression/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](../../ICLR2026/model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)

<!-- RELATED:END -->
