---
title: >-
  [Paper Note] ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect
description: >-
  [CVPR 2025][Segmentation][Large kernel] This paper reveals that the effectiveness of large kernel convolutions can be decoupled into two factors: "feature extraction at a specific granularity" and "multi-path feature fusion." Based on this insight, the authors propose ShiftwiseConv (SW Conv)—a plug-and-play CNN module that uses standard $3 \times 3$ convolutions through spatial shift operations and multi-path connections to simulate the effect of large kernels. SW Conv outper…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Large kernel"
  - "small kernel substitution"
  - "shift operation"
  - "CNN architecture design"
  - "long-range dependency"
date: 2026-05-08
content_hash: 7d213421b2aba3ce
---

# ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect

**Conference**: CVPR 2025  
**arXiv**: [2401.12736](https://arxiv.org/abs/2401.12736)  
**Code**: [https://github.com/lidc54/shift-wiseConv](https://github.com/lidc54/shift-wiseConv)  
**Area**: Image Segmentation  
**Keywords**: Large kernel, small kernel substitution, shift operation, CNN architecture design, long-range dependency

## TL;DR

This paper reveals that the effectiveness of large kernel convolutions can be decoupled into two factors: "feature extraction at a specific granularity" and "multi-path feature fusion." Based on this insight, the authors propose ShiftwiseConv (SW Conv)—a plug-and-play CNN module that uses standard $3 \times 3$ convolutions through spatial shift operations and multi-path connections to simulate the effect of large kernels. SW Conv outperforms large-kernel CNNs such as SLaK and UniRepLKNet, as well as various Transformer architectures, across tasks such as classification, detection, and segmentation.

## Background & Motivation

Vision Transformers (ViTs) have surpassed traditional CNNs on multiple vision tasks due to their superior long-range dependency modeling. To incorporate the advantages of ViTs, works like ConvNeXt have enhanced long-range dependency modeling by increasing the kernel size, and subsequent works like RepLKNet ($31 \times 31$), SLaK ($51 \times 5$), and UniRepLKNet ($13 \times 13$) have further pushed the performance limits of large-kernel CNNs.

However, researchers have observed a critical phenomenon: **the performance gains from simply increasing kernel sizes have shown diminishing returns or even stagnated**. Further performance improvements require numerous carefully designed tricks. This suggests that the effectiveness of large kernels does not stem solely from a "larger receptive field."

Inspired by the structure of the human retina—where photoreceptor cells transmit visual signals to ganglion cells via multiple pathways—this paper proposes a new perspective: **the key factors of large-kernel convolution can be decoupled into two independent components: (1) extracting basic features at a specific granularity, and (2) fusing features through multi-path connections**. Based on this insight, the authors demonstrate that standard $3 \times 3$ convolutions, combined with spatial shifting and multi-path fusion, can completely replace large-kernel convolutions and even yield further performance gains.

## Method

### Overall Architecture

ShiftwiseConv is built upon the SLaK architecture, replacing its $M \times N$ large strip convolutions with multiple $N \times N$ (default $3 \times 3$) group convolutions. The outputs of these small convolutions are spatially shifted and superimposed along the spatial dimension to simulate the receptive field of large convolutions. Supplemented by multi-path feature fusion (multi-edge), re-parameterization (Rep), and coarse-grained pruning strategies, a complete plug-and-play module is formed.

### Key Designs

1. **Spatial Stacking Substitution Experiment (Small Kernel to Large Kernel Equivalence)**:
    - **Function**: To prove that an $M \times N$ large convolution can be equivalently replaced by multiple $N \times N$ small convolutions stacked spatially.
    - **Mechanism**: Replaces the $M \times N$ strip convolution with a group convolution (number of groups = input channels $C$, output channels = $\lceil M/N \rceil$). A spatial offset is applied to the output of each small convolution according to its index so that it covers the corresponding position of the large convolution. Utilizing SLaK pre-trained parameters, the inference accuracy remains unchanged (82.5%) after the substitution.
    - **Design Motivation**: This critical experiment directly validates that large-kernel convolutions can be equivalently replaced by "small kernels + offsets," providing a foundation for subsequent designs.

2. **Multi-edge Feature Shift & Fusion**:
    - **Function**: To improve feature map utilization and simulate diverse long-range connections.
    - **Mechanism**: Analysis reveals that feature map utilization is low and predictable under a single shift path (edge). By introducing multiple edges, where each edge employs a different channel order mapping (shuffling channel permutations), coverage is significantly increased. Four edges are used by default. As the number of edges increases and the channel order is randomized, the utilization rate improves from approximately 35% to nearly 100%.
    - **Design Motivation**: Analogous to photoreceptors in the retina connecting to ganglion cells through multiple pathways, multi-path fusion increases the diversity of feature interactions.

3. **Redundancy Elimination and Parameter Optimization**:
    - **Function**: To streamline the architecture and reduce the number of parameters while maintaining or improving performance.
    - **Mechanism**:
        - Merges the two strip convolution branches of SLaK into a single shared convolution output (using reverse offsets to maintain diversity), reducing the parameters by half.
        - Adopts a Ghost-like method: introduces a ratio $G$ to allow a portion of the channels to bypass the large convolution directly ($G=0.23$ offsets SLaK's $1.3\times$ width expansion).
        - Further shrinks the kernel size from $5 \times 5$ to $3 \times 3$ (experiment #7 shows a slight accuracy improvement: 81.44% vs 81.34%, indicating that a finer granularity is more beneficial).
        - Shifts Batch Normalization (BN) in the re-parameterization branch from after-convolution to after-shift (due to greater variance between shift branches than re-parameterized branches).
    - **Design Motivation**: In large-kernel convolutions, sliding windows at peripheral positions extensively cover padded areas. Eliminating this redundancy can concurrently reduce parameters and improve performance.

### Loss & Training

- Inherits the training settings of SLaK: initially 120 epochs to explore hyperparameter trends, followed by 300 epochs of full training.
- Uses coarse-grained pruning (prune-and-grow strategy): prunes on a filter level, ranked by the sum of absolute parameter values, rather than the fine-grained element-wise pruning used in SLaK.
- Sparse mask sharing frequency: Reducing the synchronization frequency of masks among different Rep branches can boost performance (allowing the exploration of different filter combinations).
- The architectural hyperparameters adopt UniRepLKNet's depth-first strategy (a [3, 3, 18, 3] block configuration), which outperforms the width-first strategy of SLaK.

## Key Experimental Results

### Main Results

ImageNet-1K Classification:

| Method | Type | Params(M) | FLOPs(G) | Top-1 Acc(%) |
|------|------|-----------|----------|-------------|
| SW-tiny | CNN | 31 | 5.0 | **83.4** |
| UniRepLKNet-T | CNN | 31 | 4.9 | 83.2 |
| SLaK-T | CNN | 30 | 5.0 | 82.5 |
| SwinV2-T | Transformer | 28 | 6 | 81.8 |
| SW-small | CNN | 56 | 9.4 | **83.9** |
| UniRepLKNet-S | CNN | 56 | 9.1 | 83.9 |
| SLaK-S | CNN | 55 | 9.8 | 83.8 |

COCO Object Detection (Cascade Mask R-CNN):

| Method | Params(M) | AP^box | AP^mask |
|------|-----------|--------|---------|
| SW-tiny | 87 | **52.21** | **45.19** |
| UniRepLKNet-T | 89 | 51.8 | 44.9 |
| SLaK-T | - | 51.3 | 44.3 |

ADE20K Semantic Segmentation (UPerNet):

| Method | Params(M) | mIoU(SS) | mIoU(MS) |
|------|-----------|----------|----------|
| SW-tiny | 62 | **49.22** | **50.06** |
| UniRepLKNet-T | 61 | 48.6 | 49.1 |
| SLaK-T | 65 | 47.6 | - |
| SW-small | 88 | 49.79 | 50.83 |
| UniRepLKNet-S | 86 | **50.5** | **51.0** |

### Ablation Study

Progressive evolution from SLaK to SW (120 epochs, ImageNet-1K):

| Configuration | Acc(%) | Description |
|------|--------|------|
| #0 SLaK-tiny | 81.6 | Baseline ($51 \times 5$ large kernel) |
| #1 laid-out(train) | 82.27 | Small-kernel spatial stacking substitution |
| #3 SW-(pad=N//2) | 81.26 | Branch merging + unified padding |
| #4 Rep×2 | 81.52 | Adding re-parameterization |
| #7 N=5→3 | 81.44 | $3 \times 3$ kernel (finer granularity) |
| #11 rep2 E4 | 81.82 | 2Rep + 4Edge |
| #15 rep2 E4 mean | 81.94 | Optimizing initial sparsity |
| #19 architecture | 82.25 | UniRepLKNet architectural hyperparameters |
| #20 +SE | 82.27 | Adding SE module |

### Key Findings

- $3 \times 3$ convolutions can not only replace large kernels, but their finer granularity also offers a slight advantage (#6 81.34% -> #7 81.44%).
- Multi-path fusion (multiple edges) brings significant improvements over a single path, but the marginal gains of multiple Rep branches overlap with those of multiple edges.
- The depth-first strategy (UniRepLKNet-style) outperforms the width-first strategy (SLaK-style), validating the design philosophy of VGG.
- Data-driven sparsity analysis shows that deeper layers tend to prune more filters (primarily transmitting information), and the last layer of each stage experiences the most pruning (stage transition).

## Highlights & Insights

- **Meta-Insight: The True Essence of Large Kernels**: Decoupling large kernels into "granularity extraction" and "multi-path fusion" breaks the conventional perception focused purely on large kernel sizes, offering new directions for CNN design.
- **The Return of VGG**: The conclusion of replacing large kernels with $3 \times 3$ small kernels aligns with the design philosophy of VGG, but is endowed with a new meaning in modern CNN architectures: not merely stacking depth, but employing spatial stacking and multi-path connectivity.
- **Plug-and-Play Design**: SW Conv can directly replace large-kernel convolutions in existing architectures without needing to modify the overall framework.
- **Data-Driven Sparse Structure Analysis**: Analyzing patterns after coarse-grained pruning reveals inter-layer sparsity variations, which can guide future architectural design.
- **Systematic Evolution Experiments**: The progressive experimental methodology from #0 to #20 is highly exemplary, with every architectural tweak and its corresponding effect being clearly traceable.

## Limitations & Future Work

- Only results on tiny and small scales are demonstrated; the hyperparameter search cost when scaling to base/large models remains high.
- The equivalent large kernel sizes are inherited from SLaK ($51 \times 3$). Whether these sizes are also optimal for SW warrants further exploration.
- Although inference speed is optimized via re-parameterization, the memory I/O overhead of multiple edges requires attention.
- Coarse-grained pruning is more hardware-friendly on general computing devices, but the interaction between pruning rates and model architectures requires in-depth analysis.
- Future directions include exploring adaptive edge counts and connectivity patterns, as well as a more profound integration with attention mechanisms.

## Related Work & Insights

- **vs SLaK**: SLaK utilizes a $51 \times 5$ large strip kernel with fine-grained sparsity, whereas this work demonstrates that $3 \times 3$ kernels paired with spatial shifts achieve superior performance while nearly halving the parameter count.
- **vs UniRepLKNet**: UniRepLKNet employs $13 \times 13$ dilated convolutions with Squeeze-and-Excitation (SE). SW-tiny under the same architecture outperforms it using $3 \times 3$ kernels, while offering a more fundamental understanding of large-kernel design.
- **vs VAN (Visual Attention Network)**: VAN simulates a large kernel by stacking small kernels across multiple layers, which is stacking in the depth dimension. SW represents stacking in the width (spatial) dimension, expanding the effective receptive field more efficiently.

## Rating

- Novelty: ⭐⭐⭐⭐ Understanding the large kernel effect from a decoupling perspective and reconstructing it with small kernels is highly novel and convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers classification, detection, segmentation, and 3D detection, with extremely detailed progressive ablations.
- Writing Quality: ⭐⭐⭐ The logic is clear, but the experimental numbering system is relatively complex, and the figures/tables are highly dense.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for CNN architecture design; the conclusion that $3 \times 3$ can replace large kernels has highly significant implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Scale Efficient Training for Large Datasets](scale_efficient_training_for_large_datasets.md)
- [\[CVPR 2025\] F-LMM: Grounding Frozen Large Multimodal Models](f-lmm_grounding_frozen_large_multimodal_models.md)
- [\[CVPR 2025\] StoryGPT-V: Large Language Models as Consistent Story Visualizers](storygpt-v_large_language_models_as_consistent_story_visualizers.md)
- [\[NeurIPS 2025\] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](../../NeurIPS2025/segmentation/finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)
- [\[NeurIPS 2025\] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](../../NeurIPS2025/segmentation/fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)

</div>

<!-- RELATED:END -->
