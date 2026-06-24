---
title: >-
  [Paper Note] MambaVision: A Hybrid Mamba-Transformer Vision Backbone
description: >-
  [CVPR 2025][Segmentation][Mamba] NVIDIA proposes MambaVision, the first systematic study of hybrid Mamba-Transformer formulations for vision backbones. By redesigning the MambaVision Mixer and adding self-attention in the final blocks, it addresses the limitation of SSMs in capturing global context. It achieves a new Pareto front for accuracy-throughput on ImageNet-1K, while also outperforming comparable competitors in downstream detection and segmentation tasks.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Mamba"
  - "Transformer"
  - "Hybrid Architecture"
  - "SSM"
  - "Vision Backbone"
date: 2026-05-08
content_hash: 48015212d1a28ad6
---

# MambaVision: A Hybrid Mamba-Transformer Vision Backbone

**Conference**: CVPR 2025  
**arXiv**: [2407.08083](https://arxiv.org/abs/2407.08083)  
**Code**: [https://github.com/NVlabs/MambaVision](https://github.com/NVlabs/MambaVision)  
**Area**: Vision Backbone / Image Classification / Object Detection / Semantic Segmentation  
**Keywords**: Mamba, Transformer, Hybrid Architecture, SSM, Vision Backbone

## TL;DR
NVIDIA proposes MambaVision, the first systematic study of hybrid Mamba-Transformer formulations for vision backbones. By redesigning the MambaVision Mixer and adding self-attention in the final blocks, it addresses the limitation of SSMs in capturing global context. It achieves a new Pareto front for accuracy-throughput on ImageNet-1K, while also outperforming comparable competitors in downstream detection and segmentation tasks.

## Background & Motivation

1. **Background**: Transformers have become the dominant backbone in the vision field, but the quadratic complexity of attention remains a significant bottleneck. Mamba offers a linear-complexity alternative based on SSMs, which has proven effective in NLP. Models like Vision Mamba (Vim) and VMamba have integrated Mamba into vision tasks.
2. **Limitations of Prior Work**: (1) The autoregressive formulation of Mamba has inherent disadvantages when dealing with images, as pixels lack sequential dependencies and spatial relationships are inherently local and parallel; (2) Autoregressive models cannot effectively capture global context in a single forward pass; (3) Bidirectional SSMs (e.g., Vim) introduce extra latency and training difficulties; (4) Consequently, ViTs and CNN backbones still outperform the best Mamba-based vision models.
3. **Key Challenge**: Mamba is efficient for sequence modeling but lacks a global receptive field, while Transformers have global receptive fields but are computationally expensive. Seamlessly combining their strengths to achieve mutual complementation is the key challenge.
4. **Goal**: Systematically design a hybrid Mamba-Transformer architecture that simultaneously outperforms both pure Mamba and pure Transformer models in terms of accuracy and throughput.
5. **Key Insight**: The authors systematically experiment with different hybridization patterns (e.g., Transformer in early layers, middle layers, final layers, or uniformly interleaved) and find that placing self-attention in the final blocks yields the best performance. In this setup, SSMs extract local features in early stages, while attention restores global context in latter stages.
6. **Core Idea**: Use CNNs for rapid feature extraction in the first two stages, employ modified Mamba mixers in the middle stages, and utilize self-attention in the final stages to restore global information, forming a hierarchical "CNN → Mamba → Transformer" hybrid network.

## Method

### Overall Architecture
MambaVision adopts a hierarchical 4-stage architecture. The input image is converted into patches of size H/4 x W/4 x C by a stem consisting of two 3x3 convolutions with stride 2. Stages 1 and 2 employ CNN residual blocks (BN + 3x3 conv + GELU). Stages 3 and 4 feature a hybrid layout of MambaVision Mixer and Transformer blocks. Within Stages 3 and 4, the first half of the blocks utilize the MambaVision Mixer, while the second half employ self-attention. Downsampling between stages is performed using 3x3 convolutions with stride 2.

### Key Designs

1. **MambaVision Mixer (Redesigned Vision SSM Block)**:
    - **Function**: Replaces the original Mamba block to make it more suitable for vision tasks.
    - **Mechanism**: Splits the input $X_{in}$ into two parallel branches: (1) SSM branch: The input is linearly projected to $C/2$ dimensions, processed through a regular convolution (replacing the casual conv) + SiLU + selective scan to yield $X_1$; (2) Symmetric convolutional branch: The input is similarly projected to $C/2$ dimensions + processed via conv + SiLU to obtain $X_2$ (without SSM). The outputs of both branches are concatenated and linearly projected back to $C$ dimensions. Formula: $X_{out} = \text{Linear}(\text{Concat}(X_1, X_2))$.
    - **Design Motivation**: (1) Replacing causal convs with regular convs is done because vision tasks do not require causal constraints; (2) Adding a symmetric branch without SSM compensates for the sequential memory loss inherent in SSMs, ensuring that global spatial information is not lost; (3) Reducing each branch's dimensionality to $C/2$ maintains a parameter count comparable to the original Mamba block.

2. **Hierarchical Hybridization Strategy (Transformer in the Final Blocks)**:
    - **Function**: Restores global context information in the final stages of the model.
    - **Mechanism**: For a given $N$ blocks in Stages 3 and 4, the first $N/2$ blocks utilize the MambaVision Mixer + MLP, while the remaining $N/2$ blocks utilize self-attention + MLP. The self-attention mechanism adopts window partitioning (window size 14 for Stage 3, and 7 for Stage 4).
    - **Design Motivation**: Systematic ablation studies reveal that placing Transformer blocks in the final stages outperforms configurations with early, middle, or uniformly distributed Transformers. This is because SSMs efficiently extract local features in early phases, allowing self-attention to subsequently capture global dependencies in a compact token space.

3. **Fast Feature Extraction with CNN Front-end**:
    - **Function**: Replaces Mamba/Transformer blocks with CNNs in high-resolution stages to achieve high throughput.
    - **Mechanism**: Stages 1 and 2 utilize simple residual CNN blocks (two 3x3 convs + BN + GELU + residual connections) to process features at H/4 and H/8 resolutions.
    - **Design Motivation**: In high-resolution stages where the token count is massive, using either Mamba or attention turns into a performance bottleneck. CNN blocks are computationally dense and hardware-friendly. Using CNNs in the first two stages substantially boosts overall throughput.

### Loss & Training
Standard ImageNet-1K training scheme: 300 epochs on 32 A100 GPUs, using the DeiT training recipe. Downstream detection tasks are trained with Cascade Mask R-CNN using a 3x schedule, and semantic segmentation tasks use UperNet.

## Key Experimental Results

### Main Results

| Model | Params | FLOPs | Throughput(Img/s) | Top-1 Acc |
|------|--------|-------|-------------|-----------|
| MambaVision-T | 31.8M | 4.4G | 6298 | 82.3% |
| Swin-T | 28.3M | 4.4G | 2758 | 81.3% |
| VMamba-T | 30.0M | 4.9G | 1282 | 82.6% |
| MambaVision-S | 50.1M | 7.5G | 4700 | 83.3% |
| Swin-S | 49.6M | 8.5G | 1720 | 83.2% |
| MambaVision-B | 97.7M | 15.0G | 3670 | 84.2% |
| ConvNeXt-B | 88.6M | 15.4G | 1485 | 83.8% |
| VMamba-B | 89.0M | 15.4G | 645 | 83.9% |
| MambaVision-L2 | 241.5M | 37.5G | 1021 | 85.3% |

MambaVision vastly outperforms competitors in throughput at comparable accuracy levels (e.g., MambaVision-T is ~5x faster than VMamba-T).

### Ablation Study

**COCO Object Detection (Cascade Mask R-CNN)**:

| Backbone | AP_box | AP_mask |
|----------|--------|---------|
| MambaVision-T | 51.1 | 44.3 |
| Swin-T | 50.4 | 43.7 |
| ConvNeXt-T | 50.4 | 43.7 |
| MambaVision-S | 52.3 | 45.2 |
| MambaVision-B | 52.8 | 45.7 |

**ADE20K Semantic Segmentation (UperNet)**:

| Backbone | mIoU |
|----------|------|
| MambaVision-T | 46.0 |
| Swin-T | 44.5 |
| MambaVision-B | 49.1 |
| Swin-B | 48.1 |

### Key Findings
- **Massive Throughput Advantage**: MambaVision-T (6298 img/s) is nearly 5 times faster than VMamba-T (1282 img/s) and more than twice as fast as Swin-T (2758 img/s).
- **Transformer in Final Stages is the Optimal Hybrid Strategy**: Experiments confirm that the combination of SSMs for front-end feature extraction and attention for back-end global aggregation is optimal.
- **Symmetric Branch in MambaVision Mixer is Crucial**: Removing the symmetric conv branch without SSM leads to a significant decrease in accuracy.
- **Consistent Downstream Task Performance**: It consistently outperforms ConvNeXt and Swin in both object detection and semantic segmentation.

## Highlights & Insights
- **Systematic Study of Hybridization**: Instead of arbitrarily stacking Mamba and Transformer blocks, this work thoroughly explores various hybrid designs and delivers the optimal configuration. The "SSMs first, attention last" conclusion provides valuable design guidelines for future hybrid architectures.
- **Symmetric SSM-free Branch**: Adding a pure convolutional branch parallel to the SSM branch effectively compensates for the sequential information loss. This design pattern of "main path + compensation path" is transferrable to other tasks utilizing sequence models on non-sequential data.
- **A100 Real-world Throughput**: The paper consistently emphasizes actual throughput rather than FLOPs, offering stronger practical reference value.

## Limitations & Future Work
- The paper lacks fine-grained ablation studies on different components of the MambaVision Mixer (e.g., dimension ratio of the symmetric branch, conv kernel size, etc.).
- Though MambaVision-L2 achieves 85.3% accuracy, its parameter count (241M) and FLOPs (37.5G) are relatively large.
- The window sizes (14/7) for window attention seem to be manually selected without exhaustive architectural search.
- It draws an interesting contrast with the findings of MambaOut—while MambaOut suggests that image classification tasks do not require SSMs, MambaVision demonstrates that a hybrid of SSM and attention achieves a highly competitive accuracy-speed trade-off.

## Related Work & Insights
- **vs MambaOut**: While MambaOut demonstrates that pure Gated CNNs suffice for classification, MambaVision proves that a hybrid layout of SSMs and attention achieves a superior accuracy-speed trade-off while maintaining high throughput.
- **vs VMamba**: VMamba uses cross-scan in four directions but suffers from low throughput (645 img/s). MambaVision redesigns it into a single-direction SSM with a symmetric branch, boosting throughput by approximately 5 times.
- **vs Swin Transformer**: MambaVision secures a win-win scenario in both accuracy and throughput across all scales. Specifically, MambaVision-T achieves 1% higher accuracy than Swin-T while being 2.3 times faster.
- **vs EfficientVMamba**: EfficientVMamba employs SSMs for high-resolution stages and CNNs for low-resolution stages. MambaVision adopts the exact opposite hierarchy—using CNNs in high-resolution stages and SSM+attention in low-resolution ones—resulting in superior performance.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic study of hybrid Mamba-Transformer vision backbones; the MambaVision Mixer design is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage of classification, detection, and segmentation, incorporating multi-scale comparisons, physical throughput measurements, and proper ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Complete structure with clear illustrations, though technical detail sections could be slightly more concise.
- **Value**: ⭐⭐⭐⭐ Provides a practical, highly efficient vision backbone along with systematic guidelines for designing hybrid architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MambaOut: Do We Really Need Mamba for Vision?](mambaout_do_we_really_need_mamba_for_vision.md)
- [\[ICCV 2025\] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](../../ICCV2025/segmentation/tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[ICML 2025\] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [\[CVPR 2025\] Rethinking Query-Based Transformer for Continual Image Segmentation](rethinking_query-based_transformer_for_continual_image_segmentation.md)

</div>

<!-- RELATED:END -->
