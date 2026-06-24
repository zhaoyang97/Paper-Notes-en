---
title: >-
  [Paper Note] MambaOut: Do We Really Need Mamba for Vision?
description: >-
  [CVPR 2025][Segmentation][Mamba] Through conceptual analysis, this paper points out that the SSM mechanism in Mamba is suited for long-sequence + autoregressive tasks, neither of which is satisfied by ImageNet image classification. Consequently, the authors construct the MambaOut series (a pure Gated CNN) by removing SSM. MambaOut completely outperforms all state-of-the-art vision Mamba models on image classification, thoroughly demonstrating that SSM is unnecessary for visua…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Mamba"
  - "SSM"
  - "Gated CNN"
  - "Visual Recognition"
  - "Image Classification"
date: 2026-05-08
content_hash: b4b2b21c4c78346a
---

# MambaOut: Do We Really Need Mamba for Vision?

**Conference**: CVPR 2025  
**arXiv**: [2405.07992](https://arxiv.org/abs/2405.07992)  
**Code**: [https://github.com/yuweihao/MambaOut](https://github.com/yuweihao/MambaOut)  
**Area**: Image Classification / Vision Backbones  
**Keywords**: Mamba, SSM, Gated CNN, Visual Recognition, Image Classification

## TL;DR
Through conceptual analysis, this paper points out that the SSM mechanism in Mamba is suited for long-sequence + autoregressive tasks, neither of which is satisfied by ImageNet image classification. Consequently, the authors construct the MambaOut series (a pure Gated CNN) by removing SSM. MambaOut completely outperforms all state-of-the-art vision Mamba models on image classification, thoroughly demonstrating that SSM is unnecessary for visual classification.

## Background & Motivation

1. **Background**: As an RNN-like architecture based on SSM, Mamba has excelled in NLP due to its linear complexity. Subsequently, it was introduced to vision tasks (e.g., Vision Mamba, VMamba, PlainMamba) in an attempt to replace the quadratic complexity attention mechanism of Transformers.
2. **Limitations of Prior Work**: However, vision Mamba models underperform in practice, consistently lagging behind convolutional and attention-based models. For instance, CAFormer-M36, using 7-year-old separable convolution and vanilla attention, outperforms all vision Mamba models of comparable size by over 1%.
3. **Key Challenge**: While the community has been constantly adding vision-specific modifications to Mamba (such as bidirectional scanning and local inductive biases), no one has fundamentally questioned: Is SSM truly necessary for vision tasks?
4. **Goal**: Starting from the RNN essence of Mamba, this paper aims to analyze the task characteristics for which SSM is suited and then verify whether vision tasks align with these characteristics.
5. **Key Insight**: The authors start from the memory mechanism: SSM's fixed-size hidden state constitutes lossy memory, which only demonstrates its advantage in long-sequence scenarios (where attention runs out of memory). Meanwhile, the recurrent essence of SSM implies a causal mode (only seeing historical tokens), making it suited for autoregressive tasks rather than understanding tasks.
6. **Core Idea**: SSM is suited for long-sequence and autoregressive tasks, neither of which is present in ImageNet classification. Therefore, SSM is unnecessary; removing SSM and using a Gated CNN can outperform vision Mambas.

## Method

### Overall Architecture
MambaOut adopts a ResNet-like 4-stage hierarchical architecture, with each stage stacking Gated CNN blocks. After patch embedding, the input image sequentially passes through 4 stages, with downsampling between stages to reduce resolution and increase channel dimensions. The final prediction is output via global average pooling and a classification head. The core focus is not on proposing a novel architecture, but on **verifying the hypothesis by removing the SSM**.

### Key Designs

1. **Gated CNN Block (Core Component)**:
    - **Function**: Serves as the "SSM-free version" of the Mamba block, using pure convolution for token mixing.
    - **Mechanism**: Input $X$ is first processed with LayerNorm, then projected via a linear layer into two branches: one undergoes depthwise convolution (7x7 kernel, partial channels) for token mixing, and the other undergoes Gating with a GELU activation function. The two branches are element-wise multiplied, projected back to the original dimension via a linear layer, and added to the residual. The formula is $Y = (\text{TokenMixer}(X'W_1) \odot \sigma(X'W_2))W_3 + X$.
    - **Design Motivation**: The Mamba block is essentially built on top of a Gated CNN by adding SSM. Removing the SSM while keeping the Gated CNN block preserves the representative capacity of gated convolutions, allowing a direct evaluation of the SSM's contribution.

2. **Partial Channel Convolution Strategy**:
    - **Function**: Performs depthwise convolution on only a subset of channels to improve actual inference speed.
    - **Mechanism**: Drawing inspiration from InceptionNeXt, the hidden dimension is split into three parts: the gating branch, the identity branch, and the convolutional branch. Only the convolutional branch performs depthwise convolution, while others pass through directly, followed by concatenation.
    - **Design Motivation**: Although full-channel depthwise convolution has low FLOPs, it is slow in practice due to memory access bottlenecks. The partial channel strategy significantly improves throughput with negligible loss in accuracy.

3. **Conceptual Analysis Framework (Two-Feature Criteria)**:
    - **Function**: Theoretically determines whether Mamba is necessary for a specific type of task.
    - **Mechanism**: (1) Long-sequence characteristics: The quadratic term of attention dominates the computational cost only when the token sequence length $L > 6D$ (where $D$ represents channel dimension). For ImageNet at 224² resolution, there are only 196 tokens, far below the threshold of 2304 (e.g., in ViT-S). In contrast, object detection and segmentation have around 4K tokens, approaching the threshold. (2) Autoregressive characteristics: Visual understanding tasks belong to a fully-visible mode (where the model sees the entire image at once), whereas SSMs are inherently in causal mode (restricted to looking at current and past tokens), which actually harms understanding tasks—experiments show that adding causal masks to ViT reduces accuracy.
    - **Design Motivation**: To provide clear theoretical expectations for experiments: classification does not require SSM (Hypothesis 1), while detection and segmentation warrant exploration of SSM (Hypothesis 2).

### Loss & Training
The standard DeiT training recipe (without distillation) is adopted: RandAugment, Mixup, CutMix, Random Erasing, label smoothing, stochastic depth, etc. The optimizer is AdamW with learning rate $lr = \frac{batchsize}{1024} \times 10^{-3}$, and a batch size of 4096 on TPU v3.

## Key Experimental Results

### Main Results

| Model | Token Mixer | Params | MACs | Top-1 Acc |
|------|------------|--------|------|-----------|
| MambaOut-Femto | Conv | 7M | 1.2G | 78.9% |
| EfficientVMamba-S | Conv+SSM | 11M | 1.3G | 78.7% |
| MambaOut-Tiny | Conv | 27M | 4.5G | 82.7% |
| VMamba-T | Conv+SSM | 22M | 5.6G | 82.2% |
| LocalVMamba-T | Conv+SSM | 26M | 5.7G | 82.7% |
| MambaOut-Small | Conv | 48M | 9.0G | 84.1% |
| VMamba-S | Conv+SSM | 44M | 11.2G | 83.5% |
| LocalVMamba-S | Conv+SSM | 50M | 11.4G | 83.7% |
| MambaOut-Base | Conv | 85M | 15.8G | 84.2% |
| VMamba-B | Conv+SSM | 75M | 18.0G | 83.7% |

MambaOut outperforms vision Mamba models across all scales while requiring fewer MACs.

### Ablation Study

| Experimental Setup | Conclusion |
|---------|------|
| ViT causal vs fully-visible | Adding a causal mask significantly reduces ViT accuracy, demonstrating that visual understanding does not require causal mixing |
| Detection/Segmentation Tasks | MambaOut cannot match SOTA Mamba models (e.g., VMamba-T: 47.3 AP vs. MambaOut: lower), supporting Hypothesis 2 |
| ImageNet Classification | MambaOut completely outperforms Mamba models, supporting Hypothesis 1 |

### Key Findings
- **SSM is totally unnecessary for ImageNet classification.** Utilizing a pure Gated CNN by removing SSM actually yields better performance, suggesting that SSM introduces negative effects in short-sequence understanding tasks.
- **SSM remains valuable in detection/segmentation tasks**, as these tasks involve longer token sequences (~4K), where the linear complexity advantage of SSM can be leveraged.
- **There remains a gap of >1%** between existing vision Mamba models and SOTA hybrid CNN-attention models (e.g., CAFormer-M36 at 85.2%).

## Highlights & Insights
- **Analysis from first principles**: Instead of blindly following the trend of modifying Mamba for vision, this work takes a step back to ask "what tasks is Mamba's SSM truly suited for", establishing clear criteria across both RNN memory mechanisms and token mixing paradigms. This line of thinking is generalizable to any "XX for YY" cross-domain transfer problems.
- **Practice of Occam's Razor**: Operating as a minimalist baseline, MambaOut advocates for "subtraction" rather than "addition" to prove its point. Achieving superior performance by removing SSM is far more convincing than introducing complex improvements.
- **Long-sequence threshold formula**: The $L > 6D$ criterion is succinct and practical, allowing quick determination of whether any vision task benefits from a linear complexity token mixer.

## Limitations & Future Work
- The authors only evaluate three vision tasks (classification, detection, and segmentation), leaving true long-sequence vision tasks like video understanding and point clouds unverified.
- The Gated CNN block in MambaOut trailing behind SSM-based models on downstream tasks (detection/segmentation) underscores the inherent limitations of pure convolution in global modeling.
- The paper does not discuss whether advanced SSMs (e.g., Mamba-2) would alter the conclusions.
- While bidirectional SSM (bidirectional branches) has its flaws, the paper's analysis of it is somewhat brief—the fact that each branch remains causal does not mean their combination is still strictly causal.

## Related Work & Insights
- **vs. VMamba**: VMamba uses native Cross-Scan (four-way scanning), whereas MambaOut simply discards the SSM in favor of depthwise convolution. On classification, MambaOut wins with lower MACs, though VMamba remains stronger on object detection.
- **vs. MetaFormer/CAFormer**: The MetaFormer series proves that even simple pooling can serve as a token mixer, aligned with MambaOut's conclusion that "SSM is unnecessary". CAFormer, using simple convolution and attention, far outperforms all Mamba models.
- **vs. ConvNeXt**: While both are pure convolutional backbones, MambaOut's Gated CNN block is structurally closer to Mamba's design (gating + depthwise conv). However, the major contribution of MambaOut consists in conceptual analysis rather than structural innovation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The conceptual analysis framework is novel, though the model architecture itself is not highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers classification, detection, and segmentation with comprehensive multi-scale comparisons, yet lacks validation on more diverse tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Exceptionally clear logic, tightly coupling theory with experiments. The analogy paying tribute to Kobe's "Mamba Out" is also highly engaging.
- **Value**: ⭐⭐⭐⭐ Provides a critical reflection and a strong baseline for Mamba in vision, without undermining the value of Mamba in long-sequence scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MambaVision: A Hybrid Mamba-Transformer Vision Backbone](mambavision_a_hybrid_mamba-transformer_vision_backbone.md)
- [\[ICCV 2025\] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](../../ICCV2025/segmentation/tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [\[AAAI 2026\] Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation](../../AAAI2026/segmentation/do_we_need_perfect_data_leveraging_noise_for_domain_generalized_segmentation.md)
- [\[ICML 2025\] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [\[ICCV 2025\] VSSD: Vision Mamba with Non-Causal State Space Duality](../../ICCV2025/segmentation/vssd_vision_mamba_with_non-causal_state_space_duality.md)

</div>

<!-- RELATED:END -->
