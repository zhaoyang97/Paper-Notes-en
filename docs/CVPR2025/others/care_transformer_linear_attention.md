---
title: >-
  [Paper Note] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction
description: >-
  [CVPR 2025][Linear Attention] This paper proposes CARE Transformer, which decouples the learning of local inductive bias and long-range dependencies through asymmetrical feature decoupling. Fueled by a dynamic memory unit and a dual interaction module that fully exploit feature complementarity, it delivers a mobile-friendly linear-complexity vision Transformer. It achieves 78.4% top-1 accuracy on ImageNet with only 0.7 GMACs.
tags:
  - "CVPR 2025"
  - "Linear Attention"
  - "Lightweight Transformer"
  - "Mobile Deployment"
  - "Feature Decoupling"
  - "Dual Interaction"
date: 2026-05-08
content_hash: fed312d508e91672
---

# CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction

**Conference**: CVPR 2025  
**arXiv**: [2411.16170](https://arxiv.org/abs/2411.16170)  
**Code**: [https://github.com/zhouyuan888888/CARE-Transformer](https://github.com/zhouyuan888888/CARE-Transformer)  
**Area**: Others  
**Keywords**: Linear Attention, Lightweight Transformer, Mobile Deployment, Feature Decoupling, Dual Interaction

## TL;DR
This paper proposes CARE Transformer, which decouples the learning of local inductive bias and long-range dependencies through asymmetrical feature decoupling. Fueled by a dynamic memory unit and a dual interaction module that fully exploit feature complementarity, it delivers a mobile-friendly linear-complexity vision Transformer. It achieves 78.4% top-1 accuracy on ImageNet with only 0.7 GMACs.

## Background & Motivation
1. **Background**: The design of efficient vision Transformers mainly follows two technical routes: local attention (restricting receptive fields) and linear attention (reducing complexity). However, existing linear attention models either offer limited efficiency gains or suffer from significant accuracy degradation, making them difficult to deploy on mobile devices.
2. **Limitations of Prior Work**: The inherent high-entropy property of linear attention makes it difficult to suppress the influence of irrelevant tokens. While MLLA mitigates this by stacking local enhancement, this stacked design couples the learning and fusion of local and global information together, restricting both flexibility and efficiency.
3. **Key Challenge**: The stacked learning process requires the input to pass through all convolution and linear attention operations, posing a computational bottleneck. Additionally, the coupled design hinders the development of more effective feature fusion modules.
4. **Goal**: To design a linear attention mechanism that is both highly efficient and accurate, suitable for mobile deployment.
5. **Key Insight**: The authors propose a key premise—the local enhancement process can be explicitly split into two steps: learning and interaction. Separating these two steps allows them to be optimized independently.
6. **Core Idea**: Asymmetrical decoupling + dual interaction = divide-and-conquer learning of local/global information, followed by thorough fusion to exploit their complementarity.

## Method

### Overall Architecture
Input image → $4 \times 4$ convolutional stem → four-stage hierarchical structure (each stage containing multiple CARE blocks) → inside each CARE block: feature channels are split into two parts → the part with fewer channels ($d_1$) captures global dependencies via linear attention → the part with more channels ($d_2$) learns local biases via depthwise convolution → fusion via the dual interaction module → output.

### Key Designs

1. **Asymmetrical Feature Decoupling**:

    - **Function**: Splits the input features along the channel dimension into two parts to learn local and global information separately.
    - **Mechanism**: Splits $\mathbf{X} \in \mathbb{R}^{hw \times d}$ into $\bar{\mathbf{X}} \in \mathbb{R}^{hw \times d_1}$ (fed into linear attention) and $\tilde{\mathbf{X}} \in \mathbb{R}^{hw \times d_2}$ (fed into depthwise convolution). The key is setting $d_1 < d_2$ (asymmetrical). Since the complexity of linear attention is quadratic with respect to the channel dimension $O(hwd^2)$, the asymmetrical configuration reduces the complexity to $O(hw d_1^2)$, where $d_1 = d/3$.
    - **Design Motivation**: The computational bottleneck of linear attention lies in the quadratic overhead with respect to the channel dimension. Using fewer channels for capturing global information and more channels for local information guarantees both efficiency and information richness.

2. **Dynamic Memory Unit**:

    - **Function**: Preserves critical information along the network pipeline to achieve cross-layer feature interaction.
    - **Mechanism**: The first CARE block of each stage concatenates the final feature map and memory from the previous stage, then constructs the initial memory $\mathbf{Z}_0^s = \text{CONV}_{2\times2}(\mathbf{X}_{-1}^{s-1} \oplus \mathbf{Z}_{-1}^{s-1})$ via a $2 \times 2$ convolution (stride=2) for downsampling. In subsequent blocks, the memory is dynamically updated through the dual interaction module.
    - **Design Motivation**: Features from different layers have unique and complementary advantages. The memory unit replicates and utilizes information from earlier layers in later layers.

3. **Dual Interaction Module**:

    - **Function**: Enables feature interactions at two levels—between local and global features, and between features from different layers.
    - **Mechanism**: Consists of two interaction blocks. $\text{Inter}_1$ fuses local biases and long-range dependent features, i.e., $\text{Inter}_1(\bar{\mathbf{X}}, \tilde{\mathbf{X}})$, while $\text{Inter}_2$ further interacts with the memory unit, i.e., $\text{Inter}_2(\cdot, \mathbf{Z})$. Each interaction block is implemented as: concatenation $\rightarrow$ normalization $\rightarrow$ $1 \times 1$ convolution (channel interaction + 4$\times$ expansion) $\rightarrow$ $3 \times 3$ depthwise convolution (spatial interaction) $\rightarrow$ $1 \times 1$ convolution (mapping back to the original space).
    - **Design Motivation**: Decoupling the learning process necessitates an effective fusion mechanism to exploit feature complementarity. Dual interaction simultaneously accounts for both intra-layer local-to-global communication and cross-layer information exchange.

### Loss & Training
Standard ImageNet classification training. There are three sizes of CARE: S0, S1, and S2, with block configurations of ⟨2,4,8,4⟩, ⟨3,6,10,6⟩, and ⟨3,6,10,6⟩ respectively, featuring progressively increasing channel dimensions. The local bias learner utilizes $3 \times 3$ and $7 \times 7$ dual-scale depthwise convolutions, processing them in a decoupled manner as well. The first two stages do not utilize linear attention; instead, they employ $1 \times 11$ and $11 \times 1$ large-kernel depthwise convolutions. The computational complexity of linear attention is quadratic with respect to the channel dimension, $O(hwd_1^2)$, and the asymmetrical configuration $d_1=d/3$ reduces the computation to approximately $1/3$ of the symmetrical counterpart.

## Key Experimental Results

### Main Results

| Model | GMACs | Params | iPhone 13 Latency | Top-1 Acc (%) |
|------|-------|--------|-------------|---------------|
| MobileNetV2-1.0 | 0.3 | 3.5M | 1.0ms | 71.8 |
| EMO-2M | 0.4 | 2.3M | 2.0ms | 75.1 |
| **CARE-S0** | **0.7** | **3.5M** | **1.1ms** | **78.4** |
| EfficientFormerV2-S1 | 0.7 | 6.2M | 1.6ms | 79.0 |
| **CARE-S1** | **1.0** | **6.2M** | **1.5ms** | **80.6** |
| **CARE-S2** | **1.9** | **12.7M** | **2.0ms** | **82.1** |

### Ablation Study

| Configuration | GMACs | Top-1 (%) | Description |
|------|-------|-----------|------|
| Full CARE-S1 | 1.0 | 80.6 | Full model |
| Symmetrical decoupling ($d_1=d_2$) | 1.2 | 80.3 | Asymmetrical is superior and more efficient |
| w/o Memory unit | 1.0 | 79.8 | Memory unit contributes +0.8% |
| w/o Dual interaction | 0.9 | 79.2 | Interaction module contributes +1.4% |
| Stacked (MLLA) | 1.3 | 80.1 | CARE is more efficient and accurate |

### Key Findings
- Asymmetrical decoupling is theoretically and experimentally proven to be more efficient than symmetrical decoupling: $\Omega(\Delta_1) < \Omega(0)$.
- CARE also delivers outstanding results in downstream COCO detection and ADE20K segmentation: CARE-S1 achieves $40.7$ AP$^b$ in COCO object detection, $37.5$ AP$^m$ in instance segmentation, and $41.9$ mIoU in ADE20K semantic segmentation, outperforming EfficientFormer and MobileViG of similar scales.
- The measured latency on the iPhone 13 is superior to other methods with the same GMACs, illustrating that the design is highly friendly to mobile hardware.
- The dual interaction module is the main driver of the accuracy improvement (+1.4%), while the decoupling strategy is key to the efficiency boost.
- CARE-S2 achieves 82.1% Top-1 accuracy with only 1.9 GMACs, matching Swin-T (4.5 GMACs) and ConvNeXt-T (4.5 GMACs) with less than half of their computational budget.

## Highlights & Insights
- **Theoretical Proof of Asymmetrical Decoupling**: Mathematically proves the efficiency advantage of the asymmetrical configuration, validating the design choice from both theoretical and experimental perspectives.
- **Divide-and-Conquer Design Philosophy**: Decouples the "learning" and "fusion" steps, allowing each to be optimized independently and significantly enhancing design flexibility.
- **Transferability to Other Linear Attention Models**: The concepts of asymmetrical decoupling and dual interaction can be applied to any efficient model that needs to balance local and global information.
- **Design Details**: The first two stages bypass linear attention in favor of $1 \times 11$ and $11 \times 1$ large-kernel depthwise convolutions (also processed in a decoupled manner). The local bias learner uses $3 \times 3$ and $7 \times 7$ dual-scale depthwise convolutions. The asymmetrical ratio $d_1 = d/3$ obtains an optimal balance between efficiency and information preservation.
- **Comprehensive Downstream Validation**: CARE-S1 achieves $40.7$ AP$^b$ on COCO detection and $41.9$ mIoU on ADE20K semantic segmentation, matching the performance of Swin-T and ConvNeXt-T, which possess 2$-$5 times higher GMACs.

## Limitations & Future Work
- Currently, validation has only been performed on three core vision tasks (classification, detection, and segmentation), leaving generative tasks unexplored.
- Bypassing linear attention in favor of large-kernel convolutions in the first two stages represents a certain degree of engineering compromise.
- The selection of dimensions and update strategies for the dynamic memory unit requires further theoretical guidance.
- Although the high-entropy property of linear attention is alleviated through decoupling, limitations may persist for extremely long sequences.
- The choice of the asymmetrical ratio $d_1 = d/3$ is empirical; different tasks may require different proportions.
- The dynamic memory unit constructs initial memory via $2 \times 2$ convolution with stride=2 downsampling to transfer crucial information across stages. The memory is then dynamically updated via the dual interaction module in subsequent CARE blocks, enabling cross-layer feature reuse.
- The latency of CARE-S1 on an RTX 4090 is only 1.5ms, which is comparable to EfficientFormerV2-S1's 1.6ms but delivers an accuracy gain of +1.6%.

## Related Work & Insights
- **vs MLLA**: While MLLA stacks convolutions and linear attention, CARE decouples learning and then performs interaction and fusion, yielding higher efficiency and better accuracy.
- **vs FLatten/SLAB**: These methods improve the focusing capability of the linear attention mechanism itself, whereas CARE addresses the local-global balance from an architectural perspective.
- **vs EfficientFormerV2**: Although EfficientFormerV2 achieves close accuracy under identical GMACs, its latency is higher (1.6ms vs 1.5ms of CARE-S1), making CARE more suitable for practical mobile deployment.
- **vs MobileNetV2**: MobileNetV2 achieves only 71.8% accuracy at 0.3 GMACs, while CARE-S0 delivers 78.4% at 0.7 GMACs, offering a superior efficiency-accuracy trade-off.
- **vs Swin-T/ConvNeXt-T**: These models reach 82.1% top-1 accuracy under 4.5 GMACs, while CARE-S2 matches this performance at only 1.9 GMACs (less than half the computation).

## Rating

### Implementation Details
Block configurations for CARE-S0/S1/S2 are ⟨2,4,8,4⟩/⟨3,6,10,6⟩/⟨3,6,10,6⟩. The asymmetrical ratios are $d_1 = d/3$ and $d_2 = 2d/3$.
- Novelty: ⭐⭐⭐⭐ The combination of asymmetrical decoupling and dual interaction is novel, supported by solid theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across three tasks, along with mobile latency and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The mathematical derivations are clear, and the charts are intuitive.
- Value: ⭐⭐⭐⭐ High practical value for deploying vision Transformers on mobile devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RNN as Linear Transformer: A Closer Investigation into Representational Potentials of Visual Mamba Models](../../CVPR2026/others/rnn_as_linear_transformer_a_closer_investigation_into_representational_potential.md)
- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [\[CVPR 2025\] Scene-Agnostic Pose Regression for Visual Localization](scene-agnostic_pose_regression_for_visual_localization.md)
- [\[CVPR 2025\] VinaBench: Benchmark for Faithful and Consistent Visual Narratives](vinabench_benchmark_for_faithful_and_consistent_visual_narratives.md)
- [\[ICCV 2025\] LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer](../../ICCV2025/others/layertracer_cognitive-aligned_layered_svg_synthesis_via_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
