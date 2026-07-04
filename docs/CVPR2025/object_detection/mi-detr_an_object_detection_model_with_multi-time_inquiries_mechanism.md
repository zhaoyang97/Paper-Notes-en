---
title: >-
  [Paper Note] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism
description: >-
  [CVPR 2025][Object Detection][DETR] MI-DETR proposes a parallel Multi-time Inquiries (MI) mechanism to replace the traditional cascaded decoder architecture of DETR. This allows object queries to learn multi-modal information from image features in parallel through multiple parameter-independent inquiry heads. Combined with U-like Feature Interaction (UFI), it achieves 52.7 AP on COCO with a ResNet-50 backbone, outperforming all existing DETR variants.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "DETR"
  - "Parallel Decoder"
  - "Multi-time Inquiries"
  - "Feature Utilization"
date: 2026-05-08
content_hash: 3bc1bd22a6a1f11d
---

# MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism

**Conference**: CVPR 2025  
**arXiv**: [2503.01463](https://arxiv.org/abs/2503.01463)  
**Code**: To be released  
**Area**: Object Detection  
**Keywords**: DETR, Parallel Decoder, Multi-time Inquiries, Feature Utilization, Object Detection

## TL;DR
MI-DETR proposes a parallel Multi-time Inquiries (MI) mechanism to replace the traditional cascaded decoder architecture of DETR. This allows object queries to learn multi-modal information from image features in parallel through multiple parameter-independent inquiry heads. Combined with U-like Feature Interaction (UFI), it achieves 52.7 AP on COCO with a ResNet-50 backbone, outperforming all existing DETR variants.

## Background & Motivation

1. **Background**: DETR-like models adopt a cascaded decoder architecture where object queries interrogate image features layer-by-layer to obtain progressively refined information. Since 2020, various DETR variants have continuously pushed performance by improving query initialization, attention mechanisms, and matching strategies.

2. **Limitations of Prior Work**: The cascaded architecture constrains query representation updates to the cascade direction only—the representation of the next layer directly depends on the current layer. This causes object queries to learn relatively limited information patterns. Excessively refined information in deep decoders can even be redundant or harmful.

3. **Key Challenge**: Objects in natural scenes can be extremely small, severely occluded, or confused with the background, which requires fully utilizing image features to learn comprehensive information. However, the single-direction update of the cascaded architecture limits the adequacy of feature utilization.

4. **Goal**: How to make object queries learn more comprehensive and multi-modal information from image features to improve detection performance.

5. **Key Insight**: Inspired by parallel architectures in traditional CNN methods that enhance feature utilization, it is proposed to let queries retrieve image features in parallel through multiple parameter-independent branches.

6. **Core Idea**: Replace the cascaded single inquiry with parameter-independent parallel inquiry heads, allowing object queries to learn multi-modal information and then fuse them to achieve more complete feature utilization.

## Method

### Overall Architecture
MI-DETR maintains the standard DETR framework: a backbone + a multi-layer transformer encoder to extract image features $E = \{E_0, ..., E_L\}$, an MI decoder utilizing image features to adapt to detection tasks, and a prediction head to predict target locations and categories. The core innovation lies in replacing traditional decoder layers with MI decoder layers and introducing a UFI module to connect encoder features at different levels.

### Key Designs

1. **Multi-time Inquiries (MI) Mechanism**:

    - Function: Enables object queries to learn multi-modal information in each decoder layer through multiple parallel and parameter-independent branches.
    - Mechanism: Each MI decoder layer feeds the input queries $Q_{i-1}$ into $M$ independent inquiry heads (each being a standard SA+CA+FFN) to obtain $M$ sets of queries $\{Q_i^1, ..., Q_i^M\}$, which are then fused into the output $Q_i = \text{Linear}(\text{Concat}(Q_i^1, ..., Q_i^M))$ via concatenation and a linear layer. Each inquiry head has its own parameters, learning information of different patterns.
    - Design Motivation: Unlike parameter-sharing parallel architectures (where primary and auxiliary queries share the same inquiry head as in Group-DETR), parameter independence allows each branch to truly learn different patterns of information. Experiments demonstrate that parameter sharing is "pseudo-parallel," as the learned information patterns are identical. The optimal number of inquiry heads is 4.

2. **Lite Multi-time Inquiries（Lite-MI）**:

    - Function: A lightweight version of MI that reduces parameters while maintaining performance.
    - Mechanism: Different inquiry heads share the parameters of the self-attention layer, while only cross-attention and FFN remain independent. Formally, $Q_i^k = \text{FFN}_i^k(\text{CrossAtt}_i^k(\text{SelfAtt}_i(Q_{i-1}), E_j))$.
    - Design Motivation: The primary function of self-attention is to eliminate duplicate candidates, and it does not involve the image features $E_j$. Therefore, configuring an independent self-attention for each inquiry head might be redundant. Lite-MI exchanges a minimal performance cost (50.1 vs 50.2 AP) for fewer parameters.

3. **U-like Feature Interaction（UFI）**:

    - Function: Fully utilizes features from all layers of the encoder rather than only using the last layer.
    - Mechanism: Inspired by U-Net, the $j$-th layer encoder feature is fused with the last layer feature and then used as Key&Value for the $i$-th layer decoder, where $j = L - i + 1$. The fusion is formulated as $E_j = \text{linear}(\text{concat}(E_j, E_L))$. Consequently, shallow decoders utilize abstract features from deep encoders, whereas deep decoders utilize detailed features from shallow encoders.
    - Design Motivation: DETR is a classic encoder-decoder architecture, where the encoder goes from detailed to abstract layer-by-layer, and the decoder goes from abstract to detailed layer-by-layer. A U-like cross-layer connection can simultaneously utilize low-level and high-level information, similar to the skip connections in U-Net.

### Loss & Training
The loss functions and training strategies of the baseline models (DINO / Relation-DETR) are adopted. The optimizer is AdamW, with a learning rate of $1 \times 10^{-4}$ and weight decay of $1 \times 10^{-4}$. It supports 1x (12 epochs) and 2x (24 epochs) training.

## Key Experimental Results

### Main Results

**COCO val2017 (ResNet-50 backbone)**:

| Method | Conference | Epochs | AP | AP_50 | AP_75 | AP_S | AP_M | AP_L |
|------|------|--------|-----|-------|-------|------|------|------|
| DINO | ICLR'23 | 12 | 49.0 | 66.6 | 53.5 | 32.0 | 52.3 | 63.0 |
| Relation-DETR | ECCV'24 | 12 | 51.7 | 69.1 | 56.3 | 36.1 | 55.6 | 66.1 |
| **MI-DETR** | - | 12 | **52.4** | **69.8** | **57.0** | 35.6 | **56.1** | **67.2** |
| Relation-DETR | ECCV'24 | 24 | 52.1 | 69.7 | 56.6 | 36.1 | 56.0 | 66.5 |
| **MI-DETR** | - | 24 | **52.7** | **70.4** | **57.2** | **36.7** | **56.7** | **66.7** |

**Swin-L backbone (12 epochs)**:

| Method | AP | AP_50 | AP_S |
|------|-----|-------|------|
| Relation-DETR | 57.8 | 76.1 | 41.2 |
| **MI-DETR** | **58.2** | **76.5** | **42.5** |

### Ablation Study

| ID | MI | Lite-MI | UFI | AP |
|----|-----|---------|-----|------|
| #1 (baseline) | - | - | - | 49.0 |
| #2 | ✓ | | | 49.8 (+0.8) |
| #3 | | ✓ | | 49.6 (+0.6) |
| #4 | | | ✓ | 49.5 (+0.5) |
| #5 | | ✓ | ✓ | 50.1 (+1.1) |
| #6 | ✓ | | ✓ | **50.2 (+1.2)** |

**Impact of the Number of Inquiry Heads**:

| IHN | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| AP | 49.5 | 49.6 | 49.9 | **50.2** | 49.9 |

### Key Findings
- **MI contributes the most**: Using MI alone brings an improvement of +0.8 AP, contributing the most among the three components.
- **MI is plug-and-play for any DETR variant**: It consistently achieves improvements of +1.2 AP and +0.7 AP on DINO and Relation-DETR respectively, indicating that MI is orthogonal to existing techniques such as one-to-many matching.
- **4 inquiry heads is optimal**: Too many heads (e.g., 5) lead to performance degradation, as too many patterns of information might interfere with each other.
- **MI accelerates convergence**: The result of MI + DINO at 12 epochs (50.2 AP) is almost comparable to DINO at 24 epochs (50.4 AP), and MI + Relation-DETR at 12 epochs (52.4 AP) already surpasses Relation-DETR at 24 epochs (52.1 AP).
- **A single inquiry head is insufficient**: Using the output of any single head alone (approx. 41-42 AP) is far below the baseline, indicating that single-pattern information is incomplete and fusion is required.

## Highlights & Insights
- **Key Distinction between Parameter Independence and Parameter Sharing**: The paper clearly demonstrates that the "parameter-sharing parallel" design in methods like Group-DETR is essentially pseudo-parallel, as shared parameters lead different branches to learn identical information patterns. This observation is valuable for understanding the nature of parallel architectures.
- **Analogical Pedagogy in Design**: Analogizing the decoder layer to "a student asking a teacher questions" and multi-time inquiries to "asking multiple questions to get answers from different perspectives" is intuitively very clear.
- **Plug-and-Play Design**: The MI mechanism does not alter the input-output interfaces of the baseline model, allowing it to be directly inserted into the decoder of any DETR-like model. This modular design gives the method high practical value.

## Limitations & Future Work
- **Computational Overhead**: Multiple inquiry heads increase the computational load of the decoder. The paper does not thoroughly analyze the trade-off between inference speed and memory overhead.
- **Verification Only on COCO**: There is a lack of experimental validation on other datasets (e.g., LVIS, Objects365), leaving generalizability questionable.
- **Simplistic Design of UFI**: Merely using linear layers to fuse encoder features across different layers may not be the optimal way for cross-layer interaction.
- **Unexplored Combinations with Other Recent Techniques**: Such as DETR joint training strategies, query denoising, etc.
- Future directions: Explore adaptive selection of the number of inquiry heads; extend MI to the encoder side; combine with dynamic query selection.

## Related Work & Insights
- **vs DINO [Zhang et al., ICLR'23]**: DINO is the most representative DETR variant, utilizing a cascaded decoder. Inserting MI directly into DINO yields a +1.2 AP improvement, demonstrating room for improvement in feature utilization.
- **vs Relation-DETR [ECCV'24]**: A current SOTA model employing techniques like one-to-many matching. MI-DETR still achieves a +0.7 AP improvement on top of it, showing that MI is orthogonal to existing optimization techniques.
- **vs Group-DETR / H-DETR**: These methods also introduce parallel structures but are pseudo-parallel due to parameter sharing, mainly solving the shortage of positive samples. The parameter-independent parallel design of MI-DETR addresses insufficient feature utilization, differing in both motivation and technical essence.

## Rating
- Novelty: ⭐⭐⭐⭐ Parallel multi-time inquiry idea is simple and effective, and the distinction from pseudo-parallelism is very clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation (components, number of heads, plug-and-play capability), but only validated on COCO.
- Writing Quality: ⭐⭐⭐⭐⭐ Vivid analogies, rigorous logic, and exceptionally clear distinction from related works.
- Value: ⭐⭐⭐⭐ Achieves new SOTA in the highly competitive DETR field, with high practical value due to the plug-and-play design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](mr_detr_instructive_multi-route_training_for_detection_transformers.md)
- [\[NeurIPS 2025\] Test-Time Adaptive Object Detection with Foundation Model](../../NeurIPS2025/object_detection/test-time_adaptive_object_detection_with_foundation_model.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](../../NeurIPS2025/object_detection/scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[ECCV 2024\] LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction](../../ECCV2024/object_detection/lami-detr_open-vocabulary_detection_with_language_model_instruction.md)
- [\[CVPR 2025\] Test-Time Backdoor Detection for Object Detection Models](test-time_backdoor_detection_for_object_detection_models.md)

</div>

<!-- RELATED:END -->
