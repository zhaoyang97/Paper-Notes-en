---
title: >-
  [Paper Note] Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models
description: >-
  [ICML2026][Segmentation][Activation Function Alternatives] This paper constructs PolyMLP, PolyConv, and PolyAttn using Hadamard products to replace point-wise activations and softmax in MLP, convolution…
tags:
  - "ICML2026"
  - "Segmentation"
  - "Activation Function Alternatives"
  - "Polynomial Networks"
  - "Hadamard Product"
  - "MetaFormer"
  - "PolyNeXt"
date: 2026-05-08
content_hash: 64cb55f6472c4db7
---

# Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models

**Conference**: ICML2026  
**arXiv**: [2605.20839](https://arxiv.org/abs/2605.20839)  
**Code**: https://github.com/jjwang8/PolyNeXt  
**Area**: Vision Backbone / Image Recognition / Semantic Segmentation Transfer  
**Keywords**: Activation Function Alternatives, Polynomial Networks, Hadamard Product, MetaFormer, PolyNeXt  

## TL;DR
This paper constructs PolyMLP, PolyConv, and PolyAttn using Hadamard products to replace point-wise activations and softmax in MLP, convolution, and attention mechanisms. Within MetaFormer-style backbones, these activation-free models match or exceed activation-based models in ImageNet classification, robustness, and ADE20K segmentation.

## Background & Motivation
**Background**: Modern vision backbones default to point-wise activation functions like ReLU, GELU, and SiLU, alongside softmax exponential normalization in self-attention. Architectures such as ConvFormer, CAFormer, ConvNeXt, and ViT treat these nonlinearities as fundamental components for high-performance visual representation.

**Limitations of Prior Work**: Activation functions are not the sole source of nonlinearity. Existing polynomial networks demonstrate that multiplicative interactions can express complex functions. However, many methods require specialized architectures from scratch, making it difficult to reuse improvements from MetaFormer, attention, or convolution. Additionally, deep polynomial networks are prone to training instability due to multiplicative amplification.

**Key Challenge**: Simply removing activation functions leads to a lack of nonlinearity or training collapse. Conversely, maintaining complex custom polynomial structures hinders their use as general-purpose vision modules. The core problem is whether replacing standard nonlinear operators while keeping interfaces unchanged is sufficient to train a competitive backbone.

**Goal**: The authors aim to design a set of activation-free channel mixing, spatial convolution mixing, and attention mixing modules that can be inserted into MetaFormer-style architectures. These should balance ImageNet classification, OOD robustness, ADE20K semantic segmentation, and the potential for FHE-oriented polynomial inference.

**Key Insight**: The Hadamard product naturally generates second-order polynomials of the input, and the polynomial degree grows exponentially with depth when stacked. By controlling residual magnitudes and gradient flow, deep and narrow polynomial networks can achieve sufficient expressivity without point-wise activations.

**Core Idea**: Replace standard activation functions with element-wise multiplication of parallel linear/convolutional branches and stabilized residual designs, allowing the nonlinearity of the vision backbone to emerge from composable polynomial interactions.

## Method
The core of this paper is the transformation of three nonlinearity sources in common vision backbones into polynomial modules. GELU in MLP is replaced by the Hadamard product of two linear projections; activation in separable convolution is replaced by the multiplicative fusion of coarse and fine convolutional branches; and the softmax exponential kernel in attention is replaced by a polynomial kernel. The authors then assemble these into PolyNeXt, incorporating stabilization strategies like Sigmoid-Scale, multi-input skips, and depth-over-width.

### Overall Architecture
PolyNeXt adopts a four-stage hierarchical vision backbone following the MetaFormer template: each cell receives outputs from the previous two cells, passing through a spatial mixer and then a PolyMLP. CPolyNeXt uses PolyConv across all stages, while APolyNeXt uses PolyConv for high-resolution local information in the first two stages and PolyAttn for low-resolution global information in the last two stages. The stem uses a stride-4 $7\times7$ convolution, with stride-2 convolutions for downsampling between stages.

A cell can contain multiple stacks, each consisting of a "spatial mixer + PolyMLP." The authors emphasize depth-over-width: rather than making a single layer wider, it is more effective to stack more narrow polynomial layers, as the polynomial degree grows faster with the number of layers. To prevent numerical explosion caused by multiplicative chains, each residual branch uses a learnable sigmoid scalar to limit output magnitude.

### Key Designs
1. **Activation Replacement in PolyMLP and PolyConv**:
    - **Function**: Provides activation-free nonlinearity in channel mixing and local spatial mixing.
    - **Mechanism**: PolyMLP computes $W_o((W_a x)*(W_b x))$, where two linear projections are multiplied element-wise before being projected back. PolyConv uses pointwise convolution for hidden features, then extracts different receptive fields through a dilation depthwise coarse branch and a $3\times3$ fine branch. After flipping the channels of one branch, it performs element-wise multiplication and integrates the result with convolution.
    - **Design Motivation**: Multiplicative branches explicitly generate second-order interactions, forming high-order polynomials when stacked. PolyConv uses heterogeneous receptive field multiplication to facilitate cross-scale interactions more effectively than two identical branches.

2. **Polynomial Attention Kernel in PolyAttn**:
    - **Function**: Replaces the exponential nonlinearity of softmax in self-attention to maintain an activation-free structure.
    - **Mechanism**: PolyAttn uses $A=(s\cdot QK^\top+1)^p$ as unnormalized weights, where $s=\sigma(\lambda)$ is a learnable scale for each head and $p=4$. Subsequently, $\ell_1$ normalization replaces softmax. Depthwise convolutions are added to $Q, K, V$ to inject local spatial context, and $Q/K$ projections are shared to save parameters.
    - **Design Motivation**: The exponential function in softmax hinders purely polynomial inference and is not the only viable kernel for attention. The polynomial kernel maintains query-key similarity weighting while avoiding exponential activation and remains compatible with improvements like window or sparse attention.

3. **Stabilization of Deep Polynomial Networks**:
    - **Function**: Enables stable training of networks with hundreds of Hadamard-product layers.
    - **Mechanism**: Sigmoid-Scale formulates residuals as $y=x+\sigma(\lambda)f(x)$, with smaller residual contributions initialized for deeper layers. Multi-input skip connections allow each cell to receive outputs from both the previous and the one-before-previous cells, combined via learnable channel scales and LayerNorm. Depth-over-width increases the number of stacks under similar parameter constraints.
    - **Design Motivation**: Multiplication amplifies large values, making simple deepening prone to gradient and activation instability. Residual magnitude control and cross-cell skips act as numerical safety valves, while deep designs release polynomial expressivity.

### Loss & Training
The models are trained using ImageNet-1K supervised classification. The training recipe is based on MetaFormer/MONet but utilizes smaller batch sizes and stronger regularization. For semantic segmentation transfer, UperNet is trained on ADE20K for 160K iterations using the ConvNeXt recipe, with specific weight decay groups for Sigmoid-Scale, multi-input skip, and normalization parameters. A fully polynomial variant, replacing LayerNorm with polynomial-compatible BatchNorm, is also trained to explore FHE-friendly inference.

## Key Experimental Results

### Main Results
Main results on ImageNet-1K indicate that PolyNeXt approaches or exceeds activation-based MetaFormers across scales and significantly outperforms prior polynomial networks.

| Model | Params | FLOPs | Top-1 | Description |
|------|--------|-------|-------|------|
| DTTN-T | 7.1M | 2.4G | 77.9 | prior polynomial tiny |
| MONet-T | 10M | 2.8G | 77.0 | prior polynomial tiny |
| CPolyNeXt-T| 6.4M | 1.2G | 80.2 | 2-3 points higher with fewer params/FLOPs |
| ConvFormer-S18 | 27M | 3.9G | 83.0 | Activation-based MetaFormer conv baseline |
| CPolyNeXt-S | 26M | 4.8G | 83.9 | 0.9 points higher |
| DTTN-B | 36M | 12.3G | 82.4 | prior polynomial base |
| CPolyNeXt-B | 40M | 8.5G | 84.7 | 2.3 points higher than DTTN-B with lower FLOPs |
| CAFormer-S18 | 26M | 4.1G | 83.6 | Activation-based hybrid baseline |
| APolyNeXt-S | 26M | 5.3G | 84.3 | 0.7 points higher |
| CAFormer-M36 | 56M | 13.2G | 85.2 | Large hybrid baseline |
| APolyNeXt-L | 57M | 13.3G | 85.2 | Comparable |

Robustness and downstream segmentation results also support the generalization of the polynomial backbone.

| Task | Model | Clean / Main Metric | OOD / Downstream Metric | Conclusion |
|------|------|----------------|----------------|------|
| ImageNet-C/A/R/Sketch | CAFormer-S18 | 83.6 clean, IN-C 47.4, IN-A 33.5 | IN-R 48.7, IN-Sk 36.6 | Strong hybrid baseline |
| ImageNet-C/A/R/Sketch | APolyNeXt-S | 84.3 clean, IN-C 45.0, IN-A 39.6 | IN-R 49.7, IN-Sk 37.5 | Simultaneous clean and robustness gain, lower mCE |
| ADE20K UperNet | ConvFormer-S18 | 54M, 925G | 48.6 mIoU | MetaFormer conv baseline |
| ADE20K UperNet | CAFormer-S18 | 54M, 1024G | 48.9 mIoU | MetaFormer hybrid baseline |
| ADE20K UperNet | CPolyNeXt-S | 54M, 941G | 50.6 mIoU | 2.0 higher than ConvFormer-S18 |
| ADE20K UperNet | APolyNeXt-S | 55M, 1121G | 49.9 mIoU | 1.0 higher than CAFormer-S18 |

### Ablation Study
The ablation directly examines "whether activation functions are necessary" and "whether stabilization is critical."

| Configuration | Δ Acc | Description |
|------|-------|------|
| CPolyNeXt-T baseline | 80.2 | Full polynomial convolutional model |
| PolyMLP → MLP+GELU | -0.1 to -0.4 | Adding MLP activation back does not help |
| PolyConv → SepConv+GELU | -0.9 | Standard separable conv is worse |
| Add GELU to one multiply branch | -0.4 | Disrupts mutual gradient coupling |
| Add GELU after product | -1.0 | Single gate blocks gradients for both branches simultaneously |
| Hadamard → Addition | -22.3 | Multiplicative interaction is the core nonlinearity source |
| APolyNeXt-T baseline | 80.9 | Full polynomial attention model |
| PolyAttn → Std Attn | -1.3 | Standard attention replacement is significantly worse |
| polynomial kernel → softmax | -0.1 | Kernel itself is not the sole contributor; Q/K sharing and local conv are also important |

| Stabilization/Architecture Ablation | Δ Acc | Description |
|-------------------|-------|------|
| Sigmoid-Scale → free scalar | -0.5 | Initialization geometry is critical; Sigmoid yields secondary optimization gains |
| Sigmoid-Scale → LayerScale init=1e-6 | -0.8 | Traditional LayerScale is insufficient |
| Sigmoid-Scale → LayerScale init=1.0 | -12.8 | Training almost collapses |
| Remove multi-input skip | -0.6 | Cross-cell gradient flow contributes |
| Remove norm before cell | -0.4 | Normalization position matters |
| Wider 2 stacks/cell | -0.7 | Depth is superior to width |
| Wider 1 stack/cell | -1.5 | Insufficient polynomial degree |

### Key Findings
- Activation functions in this design are not "the more the better." Adding GELU back often reduces performance, suggesting that mutual gradient coupling between multiplicative branches is an effective source of nonlinearity.
- The Hadamard product is irreplaceable. Replacing it with addition results in a 22.3-point drop, essentially proving the model relies on multiplicative interaction rather than structural shells.
- Stabilization is the key to success. Without reasonable residual scales, deep polynomial networks become unstable due to multiplicative amplification. Sigmoid-Scale and multi-input skips make training nearly 200 layers possible.
- The gains in segmentation transfer are more pronounced than in classification. CPolyNeXt-S outperforms ConvFormer-S18 by 2.0 mIoU on ADE20K, indicating that the representations learned by the polynomial backbone serve more than just classification.

## Highlights & Insights
- The most valuable contribution is not "reinventing a new backbone" but making activation-free designs plug-and-play at the interface level for standard MLP/Conv/Attention. This allows the model to inherit the MetaFormer ecosystem.
- The explanation for "why activation hurts" is insightful: the two projected branches in multiplication modulate each other during backpropagation; the negative region of GELU cuts off this coupling, which contradicts the intuition that activations always increase expressivity.
- The FHE perspective moves the work beyond a performance paper. The fully polynomial BN version still achieves 82.7%, surpassing ConvNeXt-T, demonstrating that privacy-preserving networks do not necessarily sacrifice massive accuracy.
- The depth-over-width conclusion is transferable to other multiplicative architectures. The power of multiplicative networks comes from composable degrees rather than layer width, suggesting a design space different from conventional ReLU networks.

## Limitations & Future Work
- The training recipe is not entirely universal. The authors admit to needing smaller batches, stronger regularization, progressive dropout, and cautious initialization; standard configurations may be unstable.
- The deep and narrow design incurs throughput overhead. Even with similar FLOPs, the actual speed may be slower than shallower and wider MetaFormers.
- The Hadamard product is sensitive to learning rates, and multiplicative amplification makes hyperparameter tuning fragile.
- The fully polynomial version is only a step toward FHE; true end-to-end encrypted inference requires addressing normalization, attention normalization, hardware, and numerical range issues.
- The paper primarily validates on ImageNet/ADE20K; transfer to detection, instance segmentation, multimodal vision encoders, or video backbones requires further experimentation.

## Related Work & Insights
- **vs MONet / DTTN**: These prior polynomial networks rely more on custom architectures. PolyNeXt replaces only the nonlinearities within standard modules, achieving higher performance and easier migration to MetaFormer.
- **vs ConvFormer / CAFormer**: Both rely on separable conv, gated MLP, and softmax attention. This work keeps the overall template but swaps activations for polynomial interactions, matching or exceeding them at comparable scales.
- **vs StarNet / GLU**: StarNet and GLU also use element-wise multiplication but retain activations. This work emphasizes that once point-wise activations are removed, multiplication alone can provide sufficient nonlinearity.
- **vs linear attention / efficient attention**: PolyAttn does not purely seek linear complexity but replaces the exponential kernel of softmax with a polynomial kernel; it can be further combined with window/sparse attention structures.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The activation replacement logic is simple yet systematic, with distinctive combinations in PolyConv/PolyAttn and stabilization.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ ImageNet, robustness, ADE20K, FHE variants, and ablations are comprehensive; detection/video tasks could further strengthen it.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure, solid tables, and insightful analysis of activation interference.
- Value: ⭐⭐⭐⭐☆ Insightful for vision backbones, privacy-preserving networks, and multiplicative architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/gkd_generalizable_knowledge_distillation_vfm.md)
- [\[AAAI 2026\] Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation](../../AAAI2026/segmentation/causal-tune_mining_causal_factors_from_vision_foundation_mod.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](../../AAAI2026/segmentation/symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](../../NeurIPS2025/segmentation/instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)
- [\[AAAI 2026\] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV](../../AAAI2026/segmentation/otter_mitigating_background_distractions_of_wide-angle_few-shot_action_recogniti.md)

</div>

<!-- RELATED:END -->
