---
title: >-
  [Paper Note] Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models
description: >-
  [ICML 2026][Segmentation][MetaFormer] This paper constructs PolyMLP, PolyConv, and PolyAttn using Hadamard products to replace pointwise activations/softmax in MLP, convolution, and attention. Without conventional activation functions, these modules allow MetaFormer-style backbones to reach or exceed the performance of activation-based models on ImageNet,
tags:
  - ICML 2026
  - Segmentation
  - MetaFormer
  - PolyNeXt
date: 2026-05-08
content_hash: aee6b205ae4daa7a
---
# Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models

**Conference**: ICML2026  
**arXiv**: [2605.20839](https://arxiv.org/abs/2605.20839)  
**Code**: https://github.com/jjwang8/PolyNeXt  
**Area**: Vision Backbone / Image Recognition / Semantic Segmentation Transfer  
**Keywords**: Activation Function Alternatives, Polynomial Networks, Hadamard Product, MetaFormer, PolyNeXt  

## TL;DR
This paper constructs PolyMLP, PolyConv, and PolyAttn using Hadamard products to replace pointwise activations/softmax in MLP, convolution, and attention. Without conventional activation functions, these modules allow MetaFormer-style backbones to reach or exceed the performance of activation-based models on ImageNet, robustness benchmarks, and ADE20K segmentation.

## Background & Motivation
**Background**: Modern vision backbones almost by default rely on pointwise activation functions like ReLU, GELU, and SiLU, as well as softmax exponential normalization in self-attention. Architectures such as ConvFormer, CAFormer, ConvNeXt, and ViT treat these nonlinearities as fundamental components for high-performance visual representation.

**Limitations of Prior Work**: Activation functions are not the sole source of nonlinearity. Existing polynomial networks demonstrate that multiplicative interactions can express complex functions. However, many methods require designing specialized architectures from scratch, making it difficult to reuse improvements in MetaFormer/attention/convolution. Furthermore, deep polynomial networks are prone to training instability due to multiplicative amplification.

**Key Challenge**: Directly removing activation functions may result in a lack of nonlinearity or training collapse. If complex custom polynomial structures are retained, it is difficult for them to serve as general-purpose vision modules. The paper aims to prove whether replacing nonlinear operators in standard modules while maintaining the same interfaces is sufficient to train a competitive backbone.

**Goal**: The authors aim to design a set of activation-free channel mixing, spatial convolution mixing, and attention mixing modules. These should be insertable into MetaFormer-style architectures, balancing ImageNet classification, OOD robustness, ADE20K semantic segmentation, and the potential for FHE-oriented polynomial inference.

**Key Insight**: The Hadamard product naturally produces a second-order polynomial of the input. When stacked in layers, the polynomial degree grows exponentially with depth. By controlling residual magnitudes and gradient flow, deep and narrow polynomial networks can achieve sufficient expressivity without pointwise activation functions.

**Core Idea**: Replace standard activation functions with "element-wise multiplication of parallel linear/convolutional branches + stabilized residual design," deriving the vision backbone's nonlinearity from composable polynomial interactions.

## Method
The core of this paper is the step-by-step transformation of three nonlinearity sources in common vision backbones into polynomial modules. GELU in MLP is replaced by the Hadamard product of two linear projections; activation in separable convolution is replaced by the multiplicative fusion of coarse and fine convolutional branches; and the softmax exponential kernel in attention is replaced by a polynomial kernel. The authors then assemble these modules into PolyNeXt, incorporating stabilization strategies like Sigmoid-Scale, multi-input skip connections, and a depth-over-width configuration.

### Overall Architecture
PolyNeXt adopts a four-stage hierarchical vision backbone following the MetaFormer template: each cell receives outputs from the previous two cells, passing them through a spatial mixer followed by a PolyMLP. CPolyNeXt uses PolyConv in all stages, while APolyNeXt uses PolyConv for high-resolution local information in the first two stages and PolyAttn for low-resolution global information in the last two stages. The stem is a stride-4 $7\times7$ convolution, with stride-2 convolutions used for downsampling between stages.

A cell can contain multiple stacks, each being a "spatial mixer + PolyMLP." The authors emphasize depth-over-width: rather than broadening a single layer, they stack more narrow polynomial layers, as the polynomial degree grows faster with the number of layers. To prevent numerical explosion caused by multiplicative chains, each residual branch uses a learnable sigmoid scalar (Sigmoid-Scale) to limit the output magnitude. The following diagram illustrates the data flow of a PolyNeXt cell, corresponding to the four key designs:

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Input Image → Stem(7×7 stride-4 Conv)"] --> B["4-Stage Hierarchical Backbone<br/>Stride-2 Conv Downsampling"]
    B --> C["Multi-Input Skip: Take outputs of prior two cells<br/>Per-channel Scaling & Sum → LayerNorm"]
    C --> STACK
    subgraph STACK["PolyNeXt Stack (X per cell, depth-over-width)"]
        direction TB
        D["Spatial Mixer"] -->|First 2 Stages: High Res| E["PolyConv<br/>Coarse(Dilated)/Fine branches element-wise multiplication after channel-flip"]
        D -->|Last 2 Stages: Low Res| F["PolyAttn<br/>Polynomial kernel vs. Softmax + ℓ1 Norm"]
        E --> G["PolyMLP<br/>Element-wise multiplication of two linear projections"]
        F --> G
    end
    STACK -->|Res. branch y=x+σ(λ)·f(x)| H["Sigmoid-Scale Limiting"]
    H --> I["Global Head / ADE20K UperNet"]
```

### Key Designs
**1. PolyMLP: Replacing Channel Mixing Activations with Hadamard Products**  
Standard FFNs insert a GELU between two linear projections to provide nonlinearity. GLU variants add multiplicative interaction but retain activations in one branch. PolyMLP removes activations entirely: $\text{PolyMLP}(x)=W_o((W_a x)*(W_b x))$—the input is projected into two branches $W_a x$ and $W_b x$ in an intermediate dimension, which are multiplied element-wise (followed by LayerNorm) before being projected back. This multiplication yields a second-order polynomial of the input, and the degree grows rapidly with depth, accumulating sufficient nonlinearity without pointwise activations. An intuitive reason for its effectiveness is "mutual gradient coupling": during backpropagation, the gradient of $W_a$ is scaled by $W_b x$ and vice versa. Adding GELU to a branch would cut this coupling with near-zero derivatives in the negative range, explaining why adding activations back can actually decrease performance.

**2. PolyConv: Replacing Separable Convolution Activations with Heterogeneous Multi-branch Multiplication**  
ConvFormer in MetaFormer uses separable convolutions (depthwise spatial filtering + pointwise mixing) with an internal activation. To replace this with a Hadamard product, multiplying two isomorphic branches is insufficient. PolyConv uses a pointwise convolution to generate hidden features, which then split into two depthwise branches with different receptive fields: a coarse branch using dilated convolution ($5\times5$ kernel, dilation 2, covering $9\times9$) captures wide context, while a fine branch uses standard $3\times3$ for local details. One branch undergoes a channel-flip before element-wise multiplication to further decouple the features. This design is crucial because heterogeneous receptive field multiplication explicitly generates cross-scale interaction terms, providing more expressivity than isomorphic branches (as used in MONet/DTTN).

**3. PolyAttn: Replacing Softmax Exponentials with Polynomial Kernels**  
Softmax in self-attention relies on exponential functions, which are necessary nonlinearities but hinder fully polynomial (FHE-friendly) inference. PolyAttn defines unnormalized weights as $A=(s\cdot QK^\top+1)^p$ (where $p=4$ and $s=\sigma(\lambda)$ is a learnable per-head scale), using $\ell_1$ normalization instead of softmax. Following PolyConv, it adds depthwise convolutions to $Q,K,V$ for local spatial context and shares $Q/K$ projections to save parameters. It retains the attention semantics of query-key similarity weighting while avoiding exponentials. Since it only modifies the kernel while keeping the interface unchanged, it remains compatible with window/sparse attention.

**4. Stabilization Recipe for Deep Polynomial Networks**  
Unlike ReLU, Hadamard products multiply two large values into an even larger one, leading to amplification that accumulates with depth. The authors use three techniques to stabilize training for nearly 200 layers: ① Sigmoid-Scale defines each residual branch as $y=x+\sigma(\lambda)f(x)$, using a sigmoid-limited learnable scalar to constrain residual magnitude, with smaller initial contributions for deeper layers. ② Multi-input skip (mirroring NASNet) allows each cell to receive outputs from both the previous and the second-to-last cell, summed via learnable per-channel scaling followed by LayerNorm to improve gradient flow. ③ Depth-over-width: At a similar parameter count, stacking more narrow layers is preferred over widening single layers, as polynomial degree grows exponentially with depth.

### Loss & Training
Models are trained using supervised classification on ImageNet-1K. The training recipe is based on MetaFormer/MONet but utilizes smaller batch sizes and stronger regularization. Semantic segmentation transfer uses UperNet on ADE20K for 160K iterations, following the ConvNeXt recipe with specific weight decay groupings for Sigmoid-Scale, multi-input skip, and normalization parameters. The paper also explores a fully polynomial variant where LayerNorm is replaced by polynomial-compatible BatchNorm for FHE-friendly inference.

## Key Experimental Results

### Main Results
ImageNet-1K results indicate that PolyNeXt matches or exceeds activation-based MetaFormers across scales and significantly outperforms prior polynomial networks.

| Model | Params | FLOPs | Top-1 | Note |
|------|--------|-------|-------|------|
| DTTN-T | 7.1M | 2.4G | 77.9 | prior polynomial tiny |
| MONet-T | 10M | 2.8G | 77.0 | prior polynomial tiny |
| **CPolyNeXt-T (Ours)** | 6.4M | 1.2G | 80.2 | 2-3% higher with fewer params/FLOPs |
| ConvFormer-S18 | 27M | 3.9G | 83.0 | Activation-based MetaFormer conv baseline |
| **CPolyNeXt-S (Ours)** | 26M | 4.8G | 83.9 | +0.9 Gain |
| DTTN-B | 36M | 12.3G | 82.4 | prior polynomial base |
| **CPolyNeXt-B (Ours)** | 40M | 8.5G | 84.7 | +2.3 over DTTN-B with lower FLOPs |
| CAFormer-S18 | 26M | 4.1G | 83.6 | Activation-based hybrid baseline |
| **APolyNeXt-S (Ours)** | 26M | 5.3G | 84.3 | +0.7 Gain |
| CAFormer-M36 | 56M | 13.2G | 85.2 | Large hybrid baseline |
| **APolyNeXt-L (Ours)** | 57M | 13.3G | 85.2 | Comparable |

Robustness and downstream segmentation results also support the generalization of the polynomial backbone.

| Task | Model | Clean / Main Metric | OOD / Downstream | Conclusion |
|------|------|----------------|----------------|------|
| ImageNet-C/A/R/Sketch | CAFormer-S18 | 83.6 clean, IN-C 47.4, IN-A 33.5 | IN-R 48.7, IN-Sk 36.6 | Strong hybrid baseline |
| ImageNet-C/A/R/Sketch | **APolyNeXt-S** | 84.3 clean, IN-C 45.0, IN-A 39.6 | IN-R 49.7, IN-Sk 37.5 | Simultaneous clean and robustness gains |
| ADE20K UperNet | ConvFormer-S18 | 54M, 925G | 48.6 mIoU | MetaFormer conv baseline |
| ADE20K UperNet | **CPolyNeXt-S** | 54M, 941G | 50.6 mIoU | +2.0 over ConvFormer-S18 |
| ADE20K UperNet | **APolyNeXt-S** | 55M, 1121G | 49.9 mIoU | +1.0 over CAFormer-S18 |

### Ablation Study
Ablations examine the necessity of activation functions and the criticality of stabilization.

| Configuration | Δ Acc | Note |
|------|-------|------|
| CPolyNeXt-T baseline | 80.2 | Full polynomial convolutional model |
| PolyMLP → MLP+GELU | -0.1 to -0.4 | Adding MLP activations does not help |
| PolyConv → SepConv+GELU | -0.9 | Standard separable conv is worse |
| Add GELU to one branch | -0.4 | Disrupts mutual gradient coupling |
| Add GELU after product | -1.0 | Blocking gradients for both branches |
| Hadamard → Addition | -22.3 | Multiplicative interaction is the core nonlinearity |

| Stabilization/Arch Ablation | Δ Acc | Note |
|-------------------|-------|------|
| Sigmoid-Scale → free scalar | -0.5 | Initialization geometry is key |
| Sigmoid-Scale → LayerScale init=1e-6 | -0.8 | Traditional LayerScale is less compatible |
| Sigmoid-Scale → LayerScale init=1.0 | -12.8 | Training near collapse |
| Width over Depth (1 stack/cell) | -1.5 | Insufficient polynomial degree |

### Key Findings
- Activation functions are not "the more the better" in this design. Adding GELU back often reduces performance, suggesting that mutual gradient coupling between multiplicative branches is a superior source of nonlinearity.
- The Hadamard product is irreplaceable. Replacing it with addition results in a 22.3% drop, proving that the model relies on multiplicative interaction rather than structural shells.
- Stabilization is the key to success. Without proper residual scaling, deep polynomial networks become unstable; Sigmoid-Scale and multi-input skip connections enable training for nearly 200 layers.
- Segmentation transfer gains are more significant than classification. CPolyNeXt-S outperforms ConvFormer-S18 by 2.0 mIoU on ADE20K, suggesting that polynomial backbones learn highly transferable representations.

## Highlights & Insights
- The most valuable contribution is not creating a "new" backbone but providing an interface-level replacement for standard MLP/Conv/Attention modules. This allows PolyNeXt to inherit the MetaFormer ecosystem.
- The explanation of why activations can "hurt" is insightful: the two projected branches modulate each other in backpropagation, and GELU's negative region cuts this coupling.
- The FHE perspective makes this more than just a performance paper. The fully polynomial BN version still achieves 82.7% (CPolyNeXt-S BN), exceeding ConvNeXt-T and showing that privacy-friendly networks do not have to sacrifice significant accuracy.
- The depth-over-width conclusion is transferable to other multiplicative architectures. The power of multiplicative networks comes from composable degrees rather than single-layer width.

## Limitations & Future Work
- The training recipe is not completely universal. It requires smaller batches, stronger regularization, and careful initialization; standard configurations may lead to instability.
- Depth-over-width designs incur throughput overhead. Even with similar FLOPs, actual speed may be slower than shallower MetaFormers.
- Sensitivity to learning rates and multiplicative amplification makes hyperparameter tuning more fragile than with ReLU networks.
- The fully polynomial version is a step toward FHE, but end-to-end encrypted inference still needs to address normalization, hardware, and numerical range issues.

## Related Work & Insights
- **vs MONet / DTTN**: Prior polynomial networks relied on custom architectures; PolyNeXt replaces standard modules for higher performance and better transferability.
- **vs ConvFormer / CAFormer**: These rely on separable conv, gated MLP, and softmax attention. Ours retains the template but swaps activations for polynomial interactions, matching or exceeding them at scale.
- **vs StarNet / GLU**: While these use element-wise multiplication, they retain activations; this work emphasizes that multiplication alone is sufficient for nonlinearity.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Activation replacement is simple yet systematic.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid coverage of ImageNet, robustness, ADE20K, and FHE variants.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with insightful analysis on activation interference.
- Value: ⭐⭐⭐⭐☆ High impact for vision backbones and privacy-preserving computation.

## Related Papers

- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](../../NeurIPS2025/segmentation/instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[CVPR 2025\] Style-Editor: Text-driven Object-Centric Style Editing](../../CVPR2025/segmentation/style-editor_text-driven_object-centric_style_editing.md)
- [\[CVPR 2026\] PIX-TAB: Efficient PIXel-Precise TABle Structure Recognition Approach with Speculative Decoding and Region-Based Image Segmentation](../../CVPR2026/segmentation/pix-tab_efficient_pixel-precise_table_structure_recognition_approach_with_specul.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/gkd_generalizable_knowledge_distillation_vfm.md)
- [\[CVPR 2026\] The Missing Point in Vision Transformers for Universal Image Segmentation](../../CVPR2026/segmentation/the_missing_point_in_vision_transformers_for_universal_image_segmentation.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SPROUT: Supervise Less, See More — Training-free Nuclear Instance Segmentation with Prototype-Guided Prompting](supervise_less_see_more_training-free_nuclear_instance_segmentation_with_prototy.md)
- [\[ICLR 2026\] How Well Does GPT-4o Understand Vision? Evaluating Multimodal Foundation Models on Standard Computer Vision Tasks](../../ICLR2026/segmentation/how_well_does_gpt-4o_understand_vision_evaluating_multimodal_foundation_models_o.md)
- [\[CVPR 2026\] The Missing Point in Vision Transformers for Universal Image Segmentation](../../CVPR2026/segmentation/the_missing_point_in_vision_transformers_for_universal_image_segmentation.md)
- [\[ICML 2026\] Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models](geometry-preserving_unsupervised_alignment_for_heterogeneous_foundation_models.md)
- [\[ICML 2026\] What Makes Synthetic Data Effective in Image Segmentation](what_makes_synthetic_data_effective_in_image_segmentation.md)

</div>

<!-- RELATED:END -->
