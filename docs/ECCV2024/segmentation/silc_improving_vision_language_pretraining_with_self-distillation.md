---
title: >-
  [Paper Note] SiLC: Improving Vision Language Pretraining with Self-Distillation
description: >-
  [ECCV 2024][Segmentation][Vision-Language Pretraining] Introduces the SiLC framework, which incorporates local-to-global self-distillation into CLIP-style image-text contrastive learning, significantly boosting performance on dense prediction tasks (detection, segmentation) while also improving classification and retrieval.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Vision-Language Pretraining"
  - "Self-Distillation"
  - "CLIP Improvement"
  - "Dense Prediction"
  - "Local Feature Learning"
date: 2026-05-08
content_hash: 2a8fbf8fdd727d18
---

# SiLC: Improving Vision Language Pretraining with Self-Distillation

**Conference**: ECCV 2024  
**arXiv**: [2310.13355](https://arxiv.org/abs/2310.13355)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Vision-Language Pretraining, Self-Distillation, CLIP Improvement, Dense Prediction, Local Feature Learning

## TL;DR

Introduces the SiLC framework, which incorporates local-to-global self-distillation into CLIP-style image-text contrastive learning, significantly boosting performance on dense prediction tasks (detection, segmentation) while also improving classification and retrieval.

## Background & Motivation

Image-text contrastive pretraining, represented by CLIP, has become the standard paradigm for open-vocabulary visual understanding. By aligning global image and text representations on massive image-text pairs, these methods demonstrate powerful zero-shot classification and retrieval capabilities. Subsequent work has also shown that CLIP features exhibit open-vocabulary capabilities in dense prediction tasks (such as detection and segmentation).

However, a core problem is that **the contrastive objective of CLIP only focuses on global image-text alignment and does not directly incentivize the model to learn image features suitable for dense prediction**. The optimization goal of contrastive learning is to bring matching image-text pairs closer and push mismatched pairs further apart, which primarily affects global representations (CLS tokens), whereas the quality of local features (spatial tokens) is not directly optimized.

Concurrently, self-supervised learning methods (such as DINO) learn powerful local features through self-distillation but lack language alignment capabilities. The question arises: can the local feature learning capability of self-distillation be integrated into CLIP's image-text contrastive framework?

SiLC is the answer to this question—by simply adding a local-to-global self-distillation objective to CLIP training, it simultaneously improves performance on both global (classification, retrieval) and local (detection, segmentation) tasks.

## Method

### Overall Architecture

SiLC adds a self-distillation branch to the standard CLIP training pipeline: (1) Image-text contrastive branch: standard CLIP contrastive learning, aligning global image and text representations; (2) Self-distillation branch: supervises the local features of the student model using the global features of the EMA teacher model to learn local-to-global correspondences. The two branches share the image encoder and are trained jointly.

### Key Designs

1. **Local-to-Global Self-Distillation**:
    - **Function**: Significantly improves the quality of the model's local features.
    - **Mechanism**: Maintains an EMA (Exponential Moving Average) teacher model. The teacher model processes the complete image to produce global features, while the student model processes local crops of the image to produce local features. The training objective is to let the student's local features predict the teacher's global features. This encourages local features to encode global semantic information.
    - **Design Motivation**: Self-distillation in DINO has proven to be highly effective in learning local correspondences; introducing this mechanism into CLIP can compensate for the lack of local feature optimization in contrastive learning.

2. **EMA Teacher Model**:
    - **Function**: Provides stable global features as distillation targets.
    - **Mechanism**: The parameters of the teacher model are an exponential moving average of the student model's parameters, and do not directly participate in gradient updates. The EMA update allows the teacher model to smoothly track the student's training progress, avoiding target instability. The teacher only processes global views (full images or large crops), keeping computational overhead manageable.
    - **Design Motivation**: Directly matching local features to the student's own global features would cause representation collapse; the EMA teacher provides an asynchronous and stable distillation target.

3. **Multi-Task Joint Training**:
    - **Function**: Simultaneously optimizes global alignment and local feature quality.
    - **Mechanism**: The total loss consists of two parts—the image-text contrastive loss (InfoNCE) and the self-distillation loss. The two losses are summed with appropriate weights. The image-text contrastive branch ensures language alignment capability, while the self-distillation branch ensures local feature quality. The two share the encoder and reinforce each other.
    - **Design Motivation**: Neither contrastive learning alone nor self-distillation alone can satisfy both global and local task requirements; joint training achieves the optimal combination of both.

### Loss & Training

- Image-text contrastive loss (InfoNCE): Standard CLIP contrastive learning loss, aligning global image and text representations.
- Self-distillation loss (DINO-style): Cross-entropy loss between the student's local features and the teacher's global features.
- EMA update momentum: Starts at 0.996 and cosine decays to 1.0.
- Training data: The same large-scale image-text pair dataset as CLIP.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | CLIP (Baseline) | Gain |
|--------|------|------|----------|------|
| ImageNet | Zero-shot Top-1 | SOTA | CLIP | +1-3% |
| COCO | Zero-shot Detection AP | SOTA | CLIP | +3-5% |
| ADE20K | Zero-shot Segmentation mIoU | SOTA | CLIP | +5-8% |
| Retrieval (Flickr30k) | R@1 | SOTA | CLIP | +1-2% |
| Few-shot Classification | Average Acc | SOTA | CLIP | +2-4% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Contrastive Learning Only | Weak Dense Tasks | Standard CLIP performance |
| Self-distillation Only | No Language Alignment | Similar to DINO |
| Contrastive + Self-distillation | Best Overall | Mutually complementary |
| Different EMA Momentum | Larger momentum is more stable | 0.996-0.999 yield similar results |

### Key Findings

- Adding simple self-distillation significantly improves CLIP's dense prediction capabilities.
- Self-distillation not only improves dense tasks but also simultaneously boosts global tasks such as classification and retrieval.
- SiLC achieves comprehensive SOTA across zero-shot classification, few-shot classification, image-text retrieval, zero-shot segmentation, and open-vocabulary detection.
- The method introduces minimal training overhead (the EMA teacher only processes global views).

## Highlights & Insights

- The core contribution is simple yet powerful—adding self-distillation into CLIP, which offers a highly straightforward modification with comprehensive performance gains.
- Proves the complementary rather than competitive nature of contrastive learning and self-distillation.
- Comprehensive SOTA results cover multiple tasks, including classification, retrieval, detection, segmentation, and VQA.
- Provides a simple and effective improvement direction for vision-language pretraining.

## Limitations & Future Work

- The method is conceptually simple, functioning more as a combination of existing technologies (CLIP + DINO).
- The computational cost of large-scale pretraining remains high.
- The cropping strategies and hyperparameters of self-distillation may need to be adjusted for different tasks.
- More complex local-to-global correspondences (e.g., pixel-level instead of patch-level) could be explored.
- Combining this with other pretraining objectives (such as MAE) is also worth exploring.

## Related Work & Insights

- **CLIP**: OpenAI's image-text contrastive pretraining paradigm, the foundation of SiLC.
- **DINO/DINOv2**: Meta's self-distilled visual pretraining, which provides the inspiration for self-distillation.
- **OpenCLIP / SigLIP**: Open-source reproductions and improvements of CLIP.
- Insights: The combination of different pretraining paradigms (contrastive / self-distillation / masked reconstruction) can potentially be stronger than a single paradigm.

## Rating

- Novelty: ⭐⭐⭐ The method is a direct combination of CLIP and DINO, showing limited novelty but effective integration.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive experiments covering more than 7 types of downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐ High practical value, offering a simple and effective improvement scheme for pretraining.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](../../CVPR2025/segmentation/cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)
- [\[ECCV 2024\] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [\[ECCV 2024\] GiT: Towards Generalist Vision Transformer through Universal Language Interface](git_towards_generalist_vision_transformer_through_universal_language_interface.md)
- [\[ECCV 2024\] ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](controlnet_improving_conditional_controls_with_efficient_consistency_feedback.md)
- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](../../NeurIPS2025/segmentation/vision_transformers_with_self-distilled_registers.md)

</div>

<!-- RELATED:END -->
