---
title: >-
  [Paper Note] Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models
description: >-
  [ICML 2026][Segmentation][MetaFormer] This paper constructs PolyMLP, PolyConv, and PolyAttn using Hadamard products to replace point-wise activations or softmax in MLPs, convolutions, and attention mechanisms. Without conventional activation functions, these MetaFormer-style backbones match or outperform activation-based models on ImageNet, robustness benc
tags:
  - ICML 2026
  - Segmentation
  - MetaFormer
  - PolyNeXt
date: 2026-05-08
content_hash: 0805ea214216785d
---
# Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models

**Conference**: ICML2026  
**arXiv**: [2605.20839](https://arxiv.org/abs/2605.20839)  
**Code**: https://github.com/jjwang8/PolyNeXt  
**Area**: Vision Backbone / Image Recognition / Semantic Segmentation Transfer  
**Keywords**: Activation Function Replacement, Polynomial Networks, Hadamard Product, MetaFormer, PolyNeXt  

## TL;DR
This paper constructs PolyMLP, PolyConv, and PolyAttn using Hadamard products to replace point-wise activations or softmax in MLPs, convolutions, and attention mechanisms. Without conventional activation functions, these MetaFormer-style backbones match or outperform activation-based models on ImageNet, robustness benchmarks, and ADE20K segmentation.

## Background & Motivation
**Background**: Modern vision backbones almost by default rely on point-wise activation functions such as ReLU, GELU, and SiLU, as well as the exponential normalization of softmax in self-attention. Architectures like ConvFormer, CAFormer, ConvNeXt, and ViT treat these nonlinearities as fundamental components for high-performance visual representation.

**Limitations of Prior Work**: Activation functions are not the sole source of nonlinearity. Existing polynomial networks demonstrate that multiplicative interactions can express complex functions. However, many methods require specialized architectures from scratch, making it difficult to reuse advancements from MetaFormer, attention, or convolutions. Furthermore, deep polynomial networks are prone to training instability due to multiplicative amplification.

**Key Challenge**: Directly removing activation functions may result in a lack of nonlinearity or training collapse. Conversely, retaining complex custom polynomial structures prevents them from becoming general-purpose vision modules. The paper seeks to prove whether simply replacing nonlinear operators within standard modules—while maintaining the interface—is sufficient to train competitive backbones.

**Goal**: The authors aim to design a set of activation-free channel mixing, spatial convolution mixing, and attention mixing modules that can be inserted into MetaFormer-style architectures, balancing ImageNet classification, OOD robustness, ADE20K semantic segmentation, and the potential for FHE-oriented (Fully Homomorphic Encryption) polynomial inference.

**Key Insight**: The Hadamard product itself generates a second-order polynomial of the input. When stacked across layers, the polynomial degree grows exponentially with depth. By controlling residual magnitudes and gradient flow, deep yet narrow polynomial networks can achieve sufficient expressivity without point-wise activations.

**Core Idea**: Use "element-wise multiplication of parallel linear/convolutional branches + stabilized residual design" to replace standard activation functions, ensuring the nonlinearity of the vision backbone arises from composable polynomial interactions.

## Method
The core of this paper is the step-by-step transformation of the three sources of nonlinearity in common vision backbones into polynomial modules. GELU in MLPs is replaced by the Hadamard product of two linear projections; activation in separable convolutions is replaced by the multiplicative fusion of coarse and fine convolutional branches; and the exponential kernel of softmax in attention is replaced by a polynomial kernel. The authors then assemble these modules into PolyNeXt, incorporating stabilization strategies such as Sigmoid-Scale, multi-input skips, and a depth-over-width configuration.

### Overall Architecture
PolyNeXt adopts a four-stage hierarchical vision backbone following the MetaFormer template: each cell receives outputs from the previous two cells, passing through a spatial mixer followed by a PolyMLP. CPolyNeXt uses PolyConv in all stages, while APolyNeXt uses PolyConv in the first two stages for high-resolution local information and PolyAttn in the last two stages for low-resolution global information. The stem is a stride-4 $7\times7$ convolution, with stride-2 convolutions used for downsampling between stages.

A cell can contain multiple stacks, each being a "spatial mixer + PolyMLP." The authors emphasize depth-over-width: rather than widening a single layer, it is more effective to stack more narrow polynomial layers, as the polynomial degree grows faster with depth. To prevent numerical explosion caused by multiplicative chains, each residual branch uses a learnable sigmoid scalar (Sigmoid-Scale) to limit the output magnitude. The following diagram illustrates the data flow of a PolyNeXt cell:

```mermaid
graph TD
    A["Input Image → Stem (7×7 stride-4 Conv)"] --> B["4-Stage Hierarchical Backbone<br/>Stride-2 Conv Downsampling"]
    B --> C["Multi-Input Skip: Take previous two cell outputs<br/>Per-channel Scale & Sum → LayerNorm"]
    C --> STACK
    subgraph STACK["PolyNeXt Stack (X per cell, depth-over-width)"]
        direction TB
        D["Spatial Mixer"] -->|Stages 1-2| E["PolyConv<br/>Coarse/Fine branches, channel-flip, element-wise multiplication"]
        D -->|Stages 3-4| F["PolyAttn<br/>Polynomial kernel instead of Softmax + ℓ1 norm"]
        E --> G["PolyMLP<br/>Element-wise multiplication of two linear projections"]
        F --> G
    end
    STACK -->|Residual branch y=x+σ(λ)·f(x)| H["Sigmoid-Scale Limiting"]
    H --> I["Classification Head / ADE20K UperNet"]
```

### Key Designs
**1. PolyMLP: Replacing Channel Mixing Activations with Hadamard Product**  
Standard FFNs insert a GELU between two linear projections to provide nonlinearity. GLU variants add multiplicative interactions but still retain activation in one branch. PolyMLP removes it entirely: $\text{PolyMLP}(x)=W_o((W_a x)*(W_b x))$—the input is projected into two branches $W_a x$ and $W_b x$, which are multiplied element-wise (followed by LayerNorm) before being projected back. This multiplication yields a second-order polynomial of the input, and through stacking, the degree grows rapidly. A key counter-intuitive finding is "mutual gradient coupling": the gradient of $W_a$ is scaled by $W_b x$ and vice versa. Adding a GELU cuts this coupling in the negative range, explaining why adding activations back actually degrades performance.

**2. PolyConv: Multi-receptive Field Branching over Separable Conv Activations**  
MetaFormer's ConvFormer uses separable convolutions with an intermediate activation. To replace this with a Hadamard product, two identical branches are insufficient. PolyConv first uses a pointwise convolution for hidden features, then splits into two depthwise branches with different receptive fields: a coarse branch using dilated convolution ($5\times5$ kernel, dilation 2, covering $9\times9$) for context, and a fine branch using standard $3\times3$ for local details. One branch undergoes a channel-flip to further decouple them before element-wise multiplication and final projection. Heterogeneous receptive field multiplication explicitly generates cross-scale interaction terms, outperforming homogeneous branches used in prior work like MONet.

**3. PolyAttn: Polynomial Kernels instead of Softmax Exponentials**  
Softmax relies on exponential functions, which are nonlinear but hinder fully polynomial (FHE-friendly) inference. PolyAttn defines unnormalized weights as $A=(s\cdot QK^\top+1)^p$ (where $p=4$ and $s=\sigma(\lambda)$ is a learnable per-head scale), and uses $\ell_1$ normalization instead of softmax. Following PolyConv, it adds depthwise convolutions to $Q, K, V$ for local context and shares $Q/K$ projections to save parameters. It retains the similarity-weighted semantics of attention while avoiding exponentials. Ablations show that while the kernel change has a minor impact (0.1 points), the shared projections and depthwise convolutions are critical to maintaining performance.

**4. Stabilization Recipe for Deep Polynomial Networks**  
Unlike ReLU, the Hadamard product can cause numerical values to explode as they are multiplied. The authors utilize three strategies to enable training of nearly 200 layers: ① Sigmoid-Scale, where each residual branch is $y=x+\sigma(\lambda)f(x)$, using a learnable scalar constrained by sigmoid to regulate residual magnitude (standard LayerScale initialization leads to a 12.8 point drop or training collapse); ② Multi-input skips (inspired by NASNet) where each cell receives outputs from the two preceding cells to improve gradient flow; ③ Depth-over-width: stacking narrower layers rather than widening them, leveraging the exponential growth of polynomial degree with depth.

### Loss & Training
The models are trained using supervised classification on ImageNet-1K. The recipe is based on MetaFormer/MONet but utilizes smaller batch sizes and stronger regularization. For semantic segmentation, UperNet is used on ADE20K for 160K iterations with the ConvNeXt recipe, applying specific weight decay groupings for Sigmoid-Scale, multi-input skip, and normalization parameters. A "fully polynomial" variant replacing LayerNorm with BatchNorm is also tested for FHE compatibility.

## Key Experimental Results

### Main Results
ImageNet-1K results indicate that PolyNeXt matches or exceeds activation-based MetaFormer models across scales and significantly outperforms prior polynomial networks.

| Model | Params | FLOPs | Top-1 | Note |
|------|--------|-------|-------|------|
| DTTN-T | 7.1M | 2.4G | 77.9 | prior polynomial tiny |
| MONet-T | 10M | 2.8G | 77.0 | prior polynomial tiny |
| CPolyNeXt-T | 6.4M | 1.2G | 80.2 | +2-3 pts with fewer Params/FLOPs |
| ConvFormer-S18 | 27M | 3.9G | 83.0 | Activation MetaFormer conv baseline |
| CPolyNeXt-S | 26M | 4.8G | 83.9 | +0.9 pts |
| DTTN-B | 36M | 12.3G | 82.4 | prior polynomial base |
| CPolyNeXt-B | 40M | 8.5G | 84.7 | +2.3 pts over DTTN-B with lower FLOPs |
| CAFormer-S18 | 26M | 4.1G | 83.6 | Activation hybrid baseline |
| APolyNeXt-S | 26M | 5.3G | 84.3 | +0.7 pts |

Robustness and ADE20K segmentation results also demonstrate strong generalization.

| Task | Model | Clean / Metric | OOD / Downstream | Conclusion |
|------|------|----------------|----------------|------|
| ImageNet-C/A/R/Sk | CAFormer-S18 | 83.6 / IN-C 47.4 | IN-A 33.5 / IN-R 48.7 | Strong hybrid baseline |
| ImageNet-C/A/R/Sk | APolyNeXt-S | 84.3 / IN-C 45.0 | IN-A 39.6 / IN-R 49.7 | Lower mCE, higher robustness |
| ADE20K UperNet | ConvFormer-S18 | 54M, 925G | 48.6 mIoU | MetaFormer conv baseline |
| ADE20K UperNet | CPolyNeXt-S | 54M, 941G | 50.6 mIoU | +2.0 over ConvFormer-S18 |

### Ablation Study
The ablation study validates the necessity of removing activations and the importance of stabilization.

| Configuration | Δ Acc | Conclusion |
|------|-------|------|
| CPolyNeXt-T baseline | 80.2 | Full polynomial conv model |
| PolyMLP → MLP+GELU | -0.1 to -0.4 | Adding MLP activation does not help |
| PolyConv → SepConv+GELU | -0.9 | Standard separable conv is worse |
| Add GELU to one branch | -0.4 | Disrupts mutual gradient coupling |
| Add GELU after product | -1.0 | Blocks gradients for both branches |
| Hadamard → Addition | -22.3 | Multiplication is the core nonlinearity |

| Stabilization/Arch Ablation | Δ Acc | Conclusion |
|-------------------|-------|------|
| Sigmoid-Scale → LayerScale (1e-6) | -0.8 | Standard LayerScale is suboptimal |
| Sigmoid-Scale → LayerScale (1.0) | -12.8 | Training near collapse |
| Remove multi-input skip | -0.6 | Gradient flow across cells matters |
| Depth vs Width (Wider 1 stack/cell)| -1.5 | Insufficient polynomial degree |

### Key Findings
- **Activations are not "more is better"**: Adding GELU back often reduces performance, confirming that mutual gradient coupling between multiplicative branches is a superior source of nonlinearity.
- **Hadamard product is indispensable**: Replacing it with addition results in a 22.3-point drop, proving the model relies on multiplicative interaction rather than structural depth alone.
- **Stabilization is non-negotiable**: Without proper residual scaling, deep polynomial networks fail due to multiplicative amplification.
- **Segmentation gains are more significant**: CPolyNeXt-S outperforms ConvFormer-S18 by 2.0 mIoU, suggesting polynomial representations generalize exceptionally well to downstream tasks.

## Highlights & Insights
- The value of this work lies in providing interface-level replacements for standard MLP/Conv/Attention modules, allowing these activation-free designs to inherit the MetaFormer ecosystem.
- The explanation of "why activations hurt" is insightful: the two branches modulate each other's gradients; GELU breaks this synergy.
- The Fully Polynomial version (with BN) achieves 82.7%, outperforming ConvNeXt-T, demonstrating that privacy-friendly networks do not have to sacrifice significant accuracy.
- The depth-over-width conclusion is transferable to other multiplicative architectures: expressivity comes from composable degrees, not layer width.

## Limitations & Future Work
- **Training Recipe**: Requires smaller batches, stronger regularization, and specific initialization; standard training hyper-parameters may be unstable.
- **Throughput**: Despite similar FLOPs, deep and narrow designs may have lower practical throughput than shallower MetaFormers.
- **Complexity**: Hadamard products are sensitive to learning rates and hyperparameter tuning.
- **FHE Practicality**: While polynomial, fully homomorphic encryption inference still faces challenges with normalization, hardware efficiency, and numerical range.

## Related Work & Insights
- **vs MONet / DTTN**: These rely on custom architectures; PolyNeXt uses modular replacements within the MetaFormer template, yielding better performance and transferability.
- **vs ConvFormer / CAFormer**: While these rely on separable convs and softmax, PolyNeXt matches them by replacing activations with polynomial interactions.
- **vs StarNet / GLU**: While these use element-wise multiplication, they retain activations; this paper proves multiplication alone is sufficient.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Simple yet systematic activation replacement with high recognition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Strong results across ImageNet, robustness, and segmentation; could expand to detection.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure and insightful analysis of gradient coupling.
- **Value**: ⭐⭐⭐⭐☆ Significant for vision backbones, privacy-preserving AI, and multiplicative network design.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](../../NeurIPS2025/segmentation/instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[CVPR 2025\] Style-Editor: Text-driven Object-Centric Style Editing](../../CVPR2025/segmentation/style-editor_text-driven_object-centric_style_editing.md)
- [\[CVPR 2026\] PIX-TAB: Efficient PIXel-Precise TABle Structure Recognition Approach with Speculative Decoding and Region-Based Image Segmentation](../../CVPR2026/segmentation/pix-tab_efficient_pixel-precise_table_structure_recognition_approach_with_specul.md)
- [\[CVPR 2026\] The Missing Point in Vision Transformers for Universal Image Segmentation](../../CVPR2026/segmentation/the_missing_point_in_vision_transformers_for_universal_image_segmentation.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/gkd_generalizable_knowledge_distillation_vfm.md)

</div>

<!-- RELATED:END -->
