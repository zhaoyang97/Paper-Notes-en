---
title: >-
  [Paper Note] Optimizing Factorized Encoder Models: Time and Memory Reduction for Scalable and Efficient Action Recognition
description: >-
  [ECCV 2024][Video Understanding][Video Transformer] By freezing the spatial Transformers in the ViViT factorised encoder and introducing a rational temporal Transformer initialization strategy along with compact adapters, this paper significantly reduces training costs and memory consumption while preserving or even slightly improving accuracy, offering a more efficient action recognition training solution for resource-constrained researchers.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Video Transformer"
  - "ViViT"
  - "Factorised Encoder"
  - "Training Efficiency"
  - "Action Recognition"
date: 2026-05-08
content_hash: a3d8720b8acba346
---

# Optimizing Factorized Encoder Models: Time and Memory Reduction for Scalable and Efficient Action Recognition

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Video Understanding / Action Recognition  
**Keywords**: Video Transformer, ViViT, Factorised Encoder, Training Efficiency, Action Recognition

## TL;DR
By freezing the spatial Transformers in the ViViT factorised encoder and introducing a rational temporal Transformer initialization strategy along with compact adapters, this paper significantly reduces training costs and memory consumption while preserving or even slightly improving accuracy, offering a more efficient action recognition training solution for resource-constrained researchers.

## Background & Motivation

**Background**: Video Transformers (such as ViViT) demonstrate outstanding performance in action recognition tasks. The Factorised Encoder variant of ViViT adopts a late-fusion strategy, which first processes spatial features of each frame using a spatial Transformer, and then models the temporal relationships between frames using a temporal Transformer. This strategy has been adopted by many SOTA methods because it provides a good trade-off between speed and accuracy.

**Limitations of Prior Work**: Although the factorised encoder is already the most efficient option among different ViViT variants, its training time and memory consumption remain extremely high. The spatial Transformer needs to process each frame independently; as the number of frames increases and model size grows, the training overhead scales linearly or superlinearly. This creates a significant barrier to entry for researchers with limited GPU memory or computation budgets.

**Key Challenge**: Spatial Transformers are typically based on strong image pre-trained models (such as ViT), which already possess powerful spatial feature extraction capabilities. Continuing to update the spatial Transformer parameters during video training is not only computationally expensive but also yields marginal returns—the core challenge of video understanding lies more in temporal modeling than in spatial representation. However, simply freezing the spatial Transformer leads to a significant drop in accuracy.

**Goal**: How to freeze the spatial Transformer (saving training costs drastically) without sacrificing action recognition accuracy?

**Key Insight**: The authors observe that simple freezing fails due to: (1) irrational initialization of the temporal Transformer, making it unable to handle frozen spatial features effectively; and (2) the lack of temporal task adaptation in the frozen spatial representations. As long as these two issues are addressed, freezing the spatial Transformer is highly viable.

**Core Idea**: Utilizing rational temporal Transformer initialization and compact spatial-temporal adapters allows the factorised encoder with a frozen spatial Transformer to outperform conventional full-parameter training in both training efficiency and accuracy.

## Method

### Overall Architecture
The method is based on the ViViT factorised encoder architecture. The input video is divided into multiple frames, with each frame passing through the frozen spatial Transformer to extract spatial token features. These spatial features are adapted through a lightweight adapter module and then fed into a trainable temporal Transformer for temporal modeling, ultimately outputting the action classification results. During training, only the parameters of the adapter and the temporal Transformer are updated.

### Key Designs

1. **Spatial Transformer Freezing Strategy**:

    - **Function**: Dramatically reduces training time and memory by freezing all parameters of the spatial Transformer.
    - **Mechanism**: The spatial Transformer uses ImageNet pre-trained ViT weights and is completely frozen during video training. Since gradient computation of the spatial Transformer is avoided, the computational load and memory footprint of backpropagation are substantially reduced. For the spatial encoder that processes each frame independently, freezing also allows spatial features to be cached and reused.
    - **Design Motivation**: The spatial Transformer's parameters typically occupy over 50% of the total model parameters; freezing it directly cuts training computational cost by more than half. Furthermore, the pre-trained spatial representations are already robust, making the marginal utility of continued training very small.

2. **Temporal Transformer Initialization Strategy**:

    - **Function**: Provides a better initialization for the temporal Transformer so that it can effectively handle frozen spatial features.
    - **Mechanism**: The conventional approach randomly initializes the temporal Transformer, which yields poor performance when the spatial Transformer is frozen. This paper explores various initialization schemes, including transferring initialization from spatial Transformer weights or directly initializing with ImageNet pre-trained weights. A key discovery is that initializing the temporal Transformer with image pre-trained weights, even though it processes temporal rather than spatial details, performs much better than random initialization because the general capabilities of attention mechanisms are transferable.
    - **Design Motivation**: After freezing the spatial Transformer, the temporal Transformer becomes the only trainable backbone and needs to converge rapidly within limited training epochs. A good initialization accelerates convergence and enhances final performance.

3. **Compact Adapter**:

    - **Function**: Bridges the gap between frozen spatial representations and the trainable temporal Transformer.
    - **Mechanism**: A lightweight adapter is inserted between the token sequence output by the spatial Transformer and the input of the temporal Transformer. The adapter contains a small number of trainable parameters (such as linear projection + LayerNorm) to selectively focus on different spatial regions of input frames and perform task-specific adaptational adjustments to the frozen spatial features, making them more suitable for temporal modeling.
    - **Design Motivation**: Frozen spatial features are general image representations and might not be optimally aligned with specific action recognition tasks. The adapter provides task-specific representation adjustments with negligible parameter overhead (compared to full-tuning of the spatial Transformer).

### Loss & Training
Standard cross-entropy loss is used for action classification training. During training, only the parameters of the adapter and the temporal Transformer are updated, while the spatial Transformer remains fully frozen. Due to memory savings, larger spatial Transformer models or more video frames can be processed under the same hardware conditions, further boosting performance.

## Key Experimental Results

### Main Results
Evaluated on six action recognition benchmarks:

| Dataset | Metric | Ours | ViViT Baseline | Gain |
|--------|------|----------|---------------|------|
| Kinetics-400 | Top-1 Acc | +0.5~1.79% | Baseline | Accuracy match or improvement |
| Something-SomethingV2 | Top-1 Acc | Comparable | Baseline | Maintained accuracy |
| Epic-Kitchens | Acc | Comparable/Slightly better | Baseline | Maintained accuracy |
| Moments in Time | Top-1 Acc | Comparable | Baseline | Maintained accuracy |
| Training Time | GPU hours | Reduced by ~43% | Baseline | Significant reduction in training time |
| Memory Consumption | GB | Significantly reduced | Baseline | Larger models / more frames enabled |

### Ablation Study

| Configuration | Top-1 Acc | Training Time | Description |
|------|-----------|----------|------|
| Full-parameter training (baseline) | Baseline | 100% | Standard ViViT training |
| Naive freezing (random initialization) | Significant drop | ~57% | Excessive accuracy loss, unfeasible |
| Freezing + Rational initialization | Close to baseline | ~57% | Initialization strategy is key |
| Freezing + Initialization + Adapter | Comparable or better | ~57% | Adapter bridges the final gap |
| Using larger spatial model | Outperforms baseline | ~57-65% | Saved memory traded for a larger model |

### Key Findings
- The initialization strategy of the temporal Transformer is a key factor in the success of the freezing scheme—random initialization leads to an accuracy drop of over 3%, whereas a rational initialization almost completely closes the gap.
- Although the adapter has very few parameters, it provides a 0.5-1% accuracy boost, making it indispensable in freezing scenarios.
- The freezing strategy can be generalized to other factorised encoder models and is not limited to ViViT.
- Saved memory can be utilized to accommodate larger spatial models or process more frames, indirectly leading to performance gains.

## Highlights & Insights
- **The Trifecta of Freezing, Initialization, and Adapter**: Though seemingly simple, their combination works exceptionally well. This "freeze the large component, compensate with a small adapter" philosophy has been widely validated in NLP (e.g., LoRA/Adapter), and this paper successfully translates it to the video understanding domain.
- **Trading Saved Resources for Performance**: Freezing the spatial Transformer is not merely about "saving money"; more importantly, the released memory can be utilized to scale up models or increase frames, forming a new source of performance improvement. This resource relocation mindset is highly worth adopting.
- **Effectiveness of Cross-Modal Initialization**: Initializing a Transformer designed to process temporal information with pre-trained weights that processed spatial information surprisingly works. This implies that the attention computation capabilities learned by Transformers are highly generalizable.

## Limitations & Future Work
- The method is coupled with factorised encoder architectures and is not directly applicable to non-factorised video Transformers (such as joint space-time attention).
- The design of the adapter is relatively simple (linear projection); exploring more complex adapter architectures (such as cross-attention adapters) might further improve performance.
- The experiments do not cover other video tasks, such as video generation or video retrieval, so the generalizability of the method remains to be verified.
- After freezing the spatial encoder, the model depends entirely on the pre-trained spatial representations. If the visual distribution of the target task differs significantly from ImageNet (e.g., medical videos), performance might be limited.

## Related Work & Insights
- **vs. Full-Parameter Fine-Tuning of ViViT**: Full-parameter fine-tuning yields slightly better accuracy but doubles the training cost. The proposed method represents a better cost-effectiveness trade-off.
- **vs. TimeSFormer**: TimeSFormer employs divided space-time attention and shares a similar decomposed structure, but it did not explore the feasibility of freezing the spatial component.
- **vs. Adapter-Based Methods (e.g., AIM, ST-Adapter)**: The adapter concepts in NLP/multimodal domains are similar. The contribution of this paper lies in systematically verifying the effectiveness of this concept in video factorised encoders, additionally uncovering the critical role of the initialization strategy.

## Rating
- Novelty: ⭐⭐⭐ The method components (freezing, adapter, initialization) are not entirely new, but their systematic combination within a video factorised encoder is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 benchmarks, detailed ablations, and generalization to other factorised models.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, rational experimental design, and straightforward conclusions.
- Value: ⭐⭐⭐⭐ Provides direct assistance to video understanding researchers with limited resources, lowering the training barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)
- [\[ECCV 2024\] On the Utility of 3D Hand Poses for Action Recognition](on_the_utility_of_3d_hand_poses_for_action_recognition.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)

</div>

<!-- RELATED:END -->
