---
title: >-
  [Paper Note] Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention
description: >-
  [CVPR 2025][LLM (Other)][Exposure Correction] This paper proposes the Exposure-slot framework, which extends the Slot Attention algorithm into a hierarchical slot-in-slot structure. Guided by learnable exposure prompts for feature clustering, it achieves exposure-centric region-aware representation learning, obtaining SOTA performance in under-/over-exposed image correction tasks.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Exposure Correction"
  - "Slot Attention"
  - "Hierarchical Clustering"
  - "Region-aware"
  - "Prompt Learning"
date: 2026-05-08
content_hash: b1fdd709ffb4dab8
---

# Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention

**Conference**: CVPR 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Exposure Correction, Slot Attention, Hierarchical Clustering, Region-aware, Prompt Learning

## TL;DR

This paper proposes the Exposure-slot framework, which extends the Slot Attention algorithm into a hierarchical slot-in-slot structure. Guided by learnable exposure prompts for feature clustering, it achieves exposure-centric region-aware representation learning, obtaining SOTA performance in under-/over-exposed image correction tasks.

## Background & Motivation

### Background

**Background**: Image exposure correction aims to restore detail loss and color deviation caused by under- or over-exposure. Recently, deep learning-based methods (such as RetinexNet, Uformer, etc.) have made significant progress and are widely applied in scenarios like mobile photography and security monitoring.

### Limitations of Prior Work

**Limitations of Prior Work**: (1) Failure of global processing strategies: Existing methods typically apply uniform enhancement operations to the entire image. However, in real-world images, the exposure levels of different regions vary significantly (e.g., under-exposed shadow regions and over-exposed sky regions), which global operations fail to balance. (2) Lack of targeted feature learning: Features extracted by CNNs/Transformers lack explicit modeling of exposure states, leading to suboptimal correction performance in mixed-exposure scenes. (3) Difficulty in region partitioning: Boundaries between regions with different exposure levels are blurry, making them unsuitable for hard segmentation.

**Key Challenge**: Effective exposure correction requires applying different enhancement strategies to regions with different exposure levels. However, automatically discovering and distinguishing these regions without region-level annotations remains a significant challenge.

**Goal**: How to learn exposure-centric feature representations to automatically discover and targetedly process regions with different exposure levels?

**Key Insight**: Leveraging the Slot Attention concept from the object discovery field—using a competitive attention mechanism to "assign" features to different slots, where each slot corresponds to an exposure pattern, thereby achieving soft region clustering.

**Core Idea**: Utilizing hierarchical slot-in-slot attention to progressively cluster image features based on exposure levels, combined with learnable exposure prompts to achieve region-aware correction.

## Method

### Overall Architecture

Exposure-slot adopts an encoder-decoder architecture, introducing a slot-in-slot attention module between the encoder and the decoder. After the encoder extracts multi-scale features, the outer slots first coarsely divide the features into several major exposure categories (such as severe under-exposure, mild under-exposure, normal, and over-exposure), while the inner slots further subdivide local regions within each category. Learnable exposure prompts are injected into slot initialization to guide the clustering process. Finally, the outputs of all slots are reconstructed into the corrected image through weighted fusion.

### Key Designs

1. **Slot-in-Slot Hierarchical Attention**:
    - **Function**: Progressively clustering image features by exposure levels into multi-level representations.
    - **Mechanism**: Extending the original Slot Attention into a two-layer structure. The outer slots ($K_{\text{outer}}$) assign features from all spatial positions into $K$ exposure clusters through iterative competitive attention; the inner slots ($K_{\text{inner}}$) further subdivide within each outer slot to capture spatial distribution differences under the same broad exposure category. Attention weights achieve soft assignment via softmax normalization.
    - **Design Motivation**: A single-layer slot might be too coarse to distinguish regions with similar exposure levels but different spatial locations; the hierarchical design progresses from coarse to fine, improving clustering accuracy.

2. **Learnable Exposure Prompts**:
    - **Function**: Guiding slot initialization to make different slots focus on various exposure conditions.
    - **Mechanism**: Training a learnable prompt vector for each outer slot to encode prior knowledge of specific exposure conditions (e.g., "severely under-exposed", "mildly over-exposed"). During inference, these prompts serve as initial values for the slots, guiding the attention mechanism to converge rapidly to the correct exposure regions.
    - **Design Motivation**: Standard Slot Attention utilizes random initialization or learnable mean initializations, which lacks exposure-related semantic guidance, resulting in slow and unstable convergence.

3. **Region-Aware Reconstruction and Fusion**:
    - **Function**: Adaptively fusing the correction results from each slot into the final output.
    - **Mechanism**: Each slot generates local corrected features through an independent lightweight decoding branch, while the weight maps of the slot attention naturally form region fusion masks. The final corrected image is generated via weighted summation, where weights are determined by the attention intensity of the slot at each position.
    - **Design Motivation**: Different regions require distinct correction intensities and color adjustment strategies; hard masks would introduce boundary artifacts, whereas soft attention weights enable smooth transitions.

### Loss & Training
The model is trained end-to-end, with the optimization objective comprehensively considering task losses and regularization terms.


## Key Experimental Results

### Key Findings

- On the MSEC dataset, Exposure-slot outperforms SOTA methods such as RetinexFormer and FECNet in metrics like PSNR and SSIM.
- The processing effect on mixed-exposure images (where both under-exposed and over-exposed regions coexist) is particularly prominent.
- Ablation studies demonstrate that the hierarchical slot structure improves PSNR by approximately $1.2\text{ dB}$ compared to a single-layer slot.
- Learnable exposure prompts achieve an improvement of approximately $0.8\text{ dB}$ PSNR compared to random initialization.
- Visualizations indicate that different slots indeed learn attention patterns corresponding to regions with different exposure levels.

## Highlights & Insights

- **Innovative Application of Slot Attention**: Ingeniously transferring Slot Attention from the object discovery field to the image enhancement field; discovering exposure regions is fundamentally an unsupervised clustering problem.
- **Rational Hierarchical Design**: The coarse-to-fine two-level clustering aligns with human intuition for exposure evaluation (first assessing the global exposure state, then processing local differences).
- **No Region Annotations Required**: No explicit exposure region segmentation annotations are needed; slots automatically learn region partitioning through end-to-end training.

## Limitations & Future Work

- The number of slots is a preset hyperparameter, and different scenes may require different numbers of slots.
- The effectiveness in extremely under-exposed (almost entirely black) scenes may be limited by information loss.
- Hierarchical attention increases computational overhead; real-time application scenarios may require lightweight acceleration.
- Future work can incorporate RAW domain information to further improve correction quality.


## Related Work & Insights
- **vs. Representative Methods in the Same Field**: Ours makes unique contributions to method design and is complementary to existing methods.
- **vs. Traditional Methods**: Compared to traditional solutions, Ours achieves significant improvements in key metrics.
- **Insights**: The technical route of Ours offers significant reference value for subsequent related work.


## Rating
- Novelty: ⭐⭐⭐⭐ Unique contributions to method design
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation across multiple datasets
- Writing Quality: ⭐⭐⭐⭐ Well-organized and clear
- Value: ⭐⭐⭐⭐ Promotes advancement in the field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] INJONGO: A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages](../../ACL2025/llm_nlp/injongo_a_multicultural_intent_detection_and_slot-filling_dataset_for_16_african.md)
- [\[ACL 2025\] TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure](../../ACL2025/llm_nlp/testcase_eval_llm_test_gen.md)
- [\[ACL 2025\] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](../../ACL2025/llm_nlp/uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)
- [\[CVPR 2025\] Spiking Transformer with Spatial-Temporal Attention](spiking_transformer_with_spatial-temporal_attention.md)
- [\[CVPR 2025\] Learning Textual Prompts for Open-World Semi-Supervised Learning](learning_textual_prompts_for_open-world_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
