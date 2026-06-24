---
title: >-
  [Paper Note] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains
description: >-
  [CVPR 2025][Segmentation][SAM adaptation] SAM FTI-FDet proposes an automatic-prompt instance segmentation framework based on a lightweight SAM. By utilizing a Transformer decoder-style prompt generator to automatically generate task-specific prompts, an adaptive feature dispatcher to fuse multi-scale features, and a TinyViT backbone to reduce computational overhead, it achieves 74.6 $AP^{box}$ / 74.2 $AP^{mask}$ on a freight train fault detection dataset.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "SAM adaptation"
  - "automatic prompt generation"
  - "freight train fault detection"
  - "instance segmentation"
  - "TinyViT"
  - "lightweight"
date: 2026-05-08
content_hash: 7eababa2201af839
---

# Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains

**Conference**: CVPR 2025  
**arXiv**: [2603.12624](https://arxiv.org/abs/2603.12624)  
**Code**: [GitHub](https://github.com/MVME-HBUT/SAM_FTI-FDet.git)  
**Area**: Segmentation / Industrial Inspection  
**Keywords**: SAM adaptation, automatic prompt generation, freight train fault detection, instance segmentation, TinyViT, lightweight

## TL;DR
SAM FTI-FDet proposes an automatic-prompt instance segmentation framework based on a lightweight SAM. By utilizing a Transformer decoder-style prompt generator to automatically generate task-specific prompts, an adaptive feature dispatcher to fuse multi-scale features, and a TinyViT backbone to reduce computational overhead, it achieves 74.6 $AP^{box}$ / 74.2 $AP^{mask}$ on a freight train fault detection dataset.

## Background & Motivation
**Background**: Fault detection in freight trains (e.g., brake shoes, bearing seats, caliper bolts) is crucial for traffic safety. Although deep learning methods have been widely deployed, they face challenges such as poor generalization and performance degradation when shifting across stations.

**Limitations of Prior Work**: (1) CNN/Transformer methods experience a severe drop in performance when applied to new stations after training on a one specific station, making domain adaptation difficult. (2) Although SAM possesses strong generalization ability, it relies on manual prompts (clicks/boxes), which is unsuitable for fully automatic detection. (3) Object detection only provides bounding boxes, failing to quantitatively evaluate tasks that require pixel-level analysis, such as the degree of brake shoe wear.

**Key Challenge**: How to transfer SAM's general segmentation knowledge to the specific domain of freight trains without relying on manual prompts, while maintaining real-time performance?

**Key Insight**: Designing an automatic prompt generator to replace manual prompts, combined with a lightweight TinyViT-SAM backbone to reduce deployment costs, adapts the foundation model to industrial scenarios.

## Method

### Overall Architecture
Freight train image → TinyViT-SAM encoder extracts multi-layer features → Adaptive feature dispatcher (aggregator + splitter) fuses multi-scale features → Prompt generator generates query-based prompts → Prompts are integrated into the SAM mask decoder → Outputs instance segmentation masks and bounding boxes.

### Key Designs

1. **Prompt Generator**:

    - **Function**: Automatically generate class-related semantic prompts to replace manual point/box inputs.
    - **Mechanism**: Initialize $N_q$ learnable query vectors $Q_0$, which are refined layer-by-layer through $L$ Transformer decoder layers. Each layer consists of multi-head self-attention (modeling semantic dependency among queries) and multi-head cross-attention (interaction between queries and image features). The final queries are input into the mask decoder as prompts.
    - **Difference from RSPrompter**: RSPrompter uses anchor-based or query-based setups with complex hand-crafted transformations, whereas the proposed method employs a more straightforward dual-path Transformer prompt generation, leading to faster convergence.
    - **Design Motivation**: Query prompts can dynamically adapt to different component types and scenario conditions, overcoming the reliance of traditional prompts on predefined target regions.

2. **Adaptive Feature Dispatcher**:

    - **Function**: Fuse multi-layer features from the TinyViT backbone and distribute them to different scales.
    - **Mechanism**: Composed of two parts—(1) Feature Aggregator: Features from each layer are first dimensionally reduced to 32 channels using 1×1 conv + BN + ReLU + 3×3 conv, and then aggregated recursively via residual aggregation: $m_i = m_{i-1} + Conv2D(m_{i-1}) + \tilde{F}_i$. After aggregation, a FusionConv (1×1 + two 3×3 convs) is applied to obtain a unified feature $F_{agg}$. (2) Feature Splitter: Decomposes $F_{agg}$ into multi-resolution branches.
    - **Design Motivation**: TinyViT has only 4 layers. Utilizing all-layer feature extraction maximizes the utilization of representation capabilities.

3. **Mask Decoder**:

    - **Function**: Map prompt tokens to pixel-level segmentation masks.
    - **Mechanism**: Structurally similar to the prompt generator (stacked Transformer blocks) but serving a different function—it takes the prompt embedding $E_{dense}^i$ and the image feature $F_{img}^i$ to compute cross-attention, which is refined layer-by-layer to generate the mask. During inference, only the prediction of the final layer is retained, and morphological post-processing is applied to obtain the objective masks and bounding boxes.
    - **Design Motivation**: A prompt-sensitive executor that grounds high-level semantic reasoning to pixel-level spatial outputs.

4. **Freezing Strategy**:

    - Encoder fine-tuning + decoder freezing (uf/f) achieves the best performance.
    - The encoder learns task-specific representations, while freezing the decoder acts as a regularization to prevent overfitting.

## Key Experimental Results

### Main Results (Freight Train Dataset, 4410 images, 15 classes, 6 scenarios)

| Method | Backbone | $AP^{box}$ | $AP^{mask}$ | Parameters | GFLOPs |
|------|----------|-----------|------------|--------|--------|
| Mask R-CNN | ResNet50 | 70.1 | 70.7 | 44.0M | 234 |
| Mask2Former | ResNet50 | 74.2 | 72.6 | 46.3M | 245 |
| Mask2Former | Swin-T | 74.3 | 73.8 | 49M | 252 |
| RSPrompter-query | SAM-B | 72.7 | 71.9 | 131M | 425 |
| **SAM FTI-FDet** | **TinyViT** | **74.6** | **74.2** | **36.3M** | **244** |
| SAM FTI-FDet-PF | TinyViT | 73.2 | 72.9 | 30.1M | 196 |

- Surpasses all CNN/Transformer/SAM methods, with an $AP^{mask}$ of 74.2, leading Mask2Former (Swin-T) by +0.4.
- The parameter count is only 36.3M, which is far lower than RSPrompter's 131M.
- The prompt-free version (SAM FTI-FDet-PF) still achieves 73.2 $AP^{box}$ with the fewest parameters (30.1M).

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| Prompt type | query prompt > bbox prompt > gd-bbx (SAM original), $AP^{mask}$ 74.2 vs 66.3 |
| Backbone | TinyViT-5m (SA-1B pretrain) > Swin-T (COCO pretrain) > ResNet50 (ImageNet) |
| Feature layer selection | layers [2,3] are optimal; using all layers [0,1,2,3] actually degrades performance |
| Freezing strategy | Encoder fine-tuning + decoder freezing (uf/f) is optimal; full unfreezing drops performance by 2.4 $AP^{box}$ |
| Channel count | 256 > 128 > 64, wider channels extract richer features |

### Key Findings
- **Advantage of SA-1B Pretraining**: TinyViT-5m has only 5M parameters but outperforms the 45M ResNet101 due to SA-1B pretraining.
- **Training Convergence Speed**: The training loss of SAM FTI-FDet decreases faster than RSPrompter, demonstrating that the automatic prompt mechanism provides more efficient optimization signals.
- **Regularization Effect of Decoder Freezing**: Freezing SAM's mask decoder preserves generic decoding capabilities and prevents overfitting on small datasets.

## Highlights & Insights
- **Replacing Manual Interaction with Automatic Prompts**: Converts SAM from an interactive tool to a fully automatic detector. Query-based prompts offer greater flexibility than geometric prompts.
- **Industrial Deployment-Oriented**: The combination of TinyViT backbone and low-parameter design is explicitly tailored for deployment on railway edge devices.
- **Instance Segmentation Enabling Quantitative Analysis**: Pixel-level masks enable calculation of the brake shoe wear area, allowing not only the detection of faults but also the evaluation of their severity.
- **Leverage Effect of SA-1B Pretraining**: TinyViT-5m has only 5M parameters but outperforms ResNet101 (45M) due to SA-1B pretraining, indicating that the quality of pretraining data is more important than model size.

## Limitations & Future Work
- The dataset scale is limited (4,410 images, 15 classes); generalization on larger-scale industrial datasets has not been validated.
- It is only verified in freight train scenarios; transferability to other industrial inspections (e.g., high-speed trains, aerospace components) has not been tested.
- The FPS is 16, and the latency on actual edge devices (such as Jetson) remains to be verified.
- The number of prompts ($N_q=10$, $K_p=4$) is fixed, which may lack flexibility in scenarios with highly varying target counts.

## Related Work & Insights
- **Ours vs RSPrompter**: RSPrompter uses anchor-based/query-based prompts combined with complex hand-crafted transformations, whereas ours uses a more direct dual-path Transformer prompt that converges faster. Our advantages include fewer parameters (36.3M vs. 131M) and higher accuracy.
- **Ours vs Mask2Former**: Mask2Former is a strong baseline for general segmentation. Ours still achieves a +0.4 $AP^{mask}$ gain over its Swin-T version, indicating that SAM pretraining features possess unique advantages in industrial scenarios.
- **Ours vs MobileSAM**: MobileSAM achieves a lightweight SAM but still relies on manual prompts. Ours solves the automation issue further on top of lightweight adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐ The automatic prompt generator is well-designed, but the overall framework lacks breakthrough innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ The ablation study is comprehensive, but there is only a single industrial dataset, lacking cross-domain validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich charts, and detailed technical descriptions.
- Value: ⭐⭐⭐⭐ The industrial deployment orientation is clear, providing a practical solution for adapting foundation models to industrial scenarios.
- **Transfer Value of SA-1B Pretraining**: Even though there is a large gap between the target domain (railway) and the pretraining domain (general), the representation learned from SA-1B still shows significant advantages.

## Limitations & Future Work
- The dataset contains only 4,410 images and 6 scenarios. The actual number of railway inspection stations far exceeds this, and cross-station generalization requires larger-scale validation.
- Only tested on freight train scenarios; transferability to other industrial inspections (e.g., wind turbine blades, pipelines) is unknown.
- The FPS is 16.0 (with prompts), which may not suffice for real-time demanding scenarios.
- Lacks comparison with newer models like SAM2/SAM3.

## Related Work & Insights
- **Ours vs RSPrompter**: 3.6× fewer parameters, 2.3 higher $AP^{mask}$, and faster training convergence.
- **Ours vs Mask2Former(Swin-T)**: 0.4 higher $AP^{mask}$ with 26% fewer parameters, demonstrating the advantages of transferring knowledge from foundation models.
- **Ours vs Direct SAM**: The $AP^{mask}$ of SAM with manual box prompts is 66.3, whereas it rises to 74.2 (+7.9) with automatic prompts, proving the importance of prompt quality.

## Rating
- Novelty: ⭐⭐⭐ The idea of automatic prompt generation is not entirely new (pioneered by previous works like RSPrompter), but the adaptation for the railway domain is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies (prompt/backbone/layers/freezing/channels) validating from multiple perspectives.
- Writing Quality: ⭐⭐⭐ The structure is complete, but some descriptions are overly detailed.
- Value: ⭐⭐⭐ Practical contribution to the field of railway industrial inspection, though the generality of the method is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Foveated Instance Segmentation](foveated_instance_segmentation.md)
- [\[CVPR 2025\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearth-sar_a_sar-centric_and_billion-scale_geospatial_foundation_model_for_d.md)
- [\[CVPR 2026\] BiPA: Bilevel Prompt Adaptation for Underwater Instance Segmentation](../../CVPR2026/segmentation/bipa_bilevel_prompt_adaptation_for_underwater_instance_segmentation.md)
- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](../../ICCV2025/segmentation/hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[ECCV 2024\] ProMerge: Prompt and Merge for Unsupervised Instance Segmentation](../../ECCV2024/segmentation/promerge_prompt_and_merge_for_unsupervised_instance_segmentation.md)

</div>

<!-- RELATED:END -->
