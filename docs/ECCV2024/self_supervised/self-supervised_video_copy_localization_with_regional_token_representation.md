---
title: >-
  [Paper Note] Self-supervised Video Copy Localization with Regional Token Representation
description: >-
  [ECCV 2024][Self-Supervised Learning][Video Copy Localization] This paper proposes a self-supervised video copy localization framework. By introducing Regional Tokens into Vision Transformers to capture local regional information and utilizing the transitivity property to automatically generate training data, the proposed method outperforms supervised approaches without requiring manual annotations.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Video Copy Localization"
  - "Regional Token"
  - "Vision Transformer"
  - "Transitivity"
date: 2026-05-08
content_hash: d265eb473f515118
---

# Self-supervised Video Copy Localization with Regional Token Representation

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Self-Supervised Learning  
**Keywords**: Video Copy Localization, Self-Supervised Learning, Regional Token, Vision Transformer, Transitivity

## TL;DR
This paper proposes a self-supervised video copy localization framework. By introducing Regional Tokens into Vision Transformers to capture local regional information and utilizing the transitivity property to automatically generate training data, the proposed method outperforms supervised approaches without requiring manual annotations.

## Background & Motivation

**Background**: Video Copy Localization aims to locate the start and end timestamps of all copied segments given a pair of untrimmed videos. This is essential for applications such as copyright protection, content auditing, and video source tracking. Current mainstream approaches typically extract frame-level features, construct a frame-to-frame similarity map, and train a detector to identify copy patterns within this similarity map.

**Limitations of Prior Work**: (1) Frame-level features are typically unified global representations that fail to capture local information, leading to poor performance in common video copy editing scenarios like "picture-in-picture" where the copied content occupies only a small portion of the frame. (2) Training the detector requires a massive amount of manually labeled data (annotated copied video pairs and their timestamps), which is expensive and time-consuming to obtain.

**Key Challenge**: Video copy detection requires both global awareness (determining if the entire frame is copied) and local awareness (localizing copied regions in picture-in-picture scenarios), yet existing methods rely solely on global features. At the same time, the heavy reliance of supervised learning on annotated data severely limits the scalability of these methods.

**Goal**: (1) How to incorporate local regional information into frame-level features to handle complex video copy editing? (2) How to eliminate the dependency on manually labeled data?

**Key Insight**: The authors present two key insights: First, the patch tokens of a Vision Transformer naturally contain spatial positional information and can learn to focus on specific local regions; Second, the transitivity of video copies (i.e., if A is a copy of B, and B is a copy of C, then A and C are also copies) can be leveraged to automatically construct training data without manual annotations.

**Core Idea**: Extend ViT with Regional Tokens to learn local regional features, and automatically generate copied video pairs using transitivity for self-supervised training, achieving annotation-free video copy localization.

## Method

### Overall Architecture
The input consists of a pair of videos to be examined, and the output is the start and end timestamps of all copied segments. The framework consists of three stages: (1) Feature extraction: utilizing ViT with Regional Tokens to extract global and local features for each frame; (2) Similarity map construction: calculating the feature similarity matrix between all frame pairs of the two videos; (3) Copy detection: identifying diagonal patterns of copy segments on the similarity map using a self-supervised detector.

### Key Designs

1. **Regional Token Representation**:

    - **Function**: Introduces extra learnable tokens in ViT to learn to focus on specific local regions within a frame, enhancing robustness against editing scenarios like picture-in-picture.
    - **Mechanism**: In addition to standard CLS and patch tokens, several Regional Tokens are appended to the input sequence of the ViT. These Regional Tokens interact with all patch tokens via self-attention, naturally learning to attend to different spatial regions during training. An asymmetric training strategy is adopted, where the teacher network uses the full image to generate features and the student network uses cropped local regions, training the Regional Token to maintain consistency between global and local views. The final frame-level representation is an aggregation of the CLS token (global) and Regional Tokens (local).
    - **Design Motivation**: Traditional CLS tokens aggregate information from all patches, representing a purely global feature. In picture-in-picture scenarios, the copied content might only occupy 20% of the frame, causing the similarity of global features to be heavily diluted by irrelevant areas. Regional Tokens learn to focus on meaningful local areas, effectively matching local copies during similarity computations.

2. **Transitivity-based Self-supervised Data Generation**:

    - **Function**: Automatically generates copy video pairs with timestamp annotations using the transitivity of video copies, completely eliminating the need for manual annotations.
    - **Mechanism**: Given a source video A, copy video B is generated through random temporal cropping, speed variation, etc., and the corresponding timestamps between A and B are recorded. Then, B is subjected to different copying operations to generate C, and the timestamps between B and C are recorded. Based on transitivity, the copied timestamp relationship between A and C can be accurately deduced. This automatically generates a training triplet with precise timestamp annotations. By applying diverse data augmentations in both temporal and spatial dimensions (e.g., speed variation, picture-in-picture overlay, color jittering, blurring), large-scale and diversified training data can be generated.
    - **Design Motivation**: Although existing video copy detection datasets (e.g., VCSL) provide manual annotations, they are highly expensive to annotate and limited in scale. Transitivity is an inherent mathematical property of video copying, and leveraging it allows unlimited training data generation. More importantly, these automatically generated data cover various combinations of copy edit types, rendering the trained detector more generalizable.

3. **Similarity Map Detector**:

    - **Function**: Identifies diagonal patterns corresponding to copy segments within the frame-to-frame similarity map.
    - **Mechanism**: First, the cosine similarity between all frame pairs of two videos is computed to construct a 2D similarity map. Copy segments manifest as high-similarity diagonal stripes in this map. A lightweight CNN detector is used to slide a window across the similarity map to detect these diagonal patterns, outputting the start and end timestamps of the copy segments. The detector is trained completely using the data generated by the transitivity strategy, without any manual labels.
    - **Design Motivation**: The similarity map transforms the video copy localization problem into a pattern detection problem in 2D images, allowing the utilization of mature object detection techniques. Self-supervised training data provide the detector with a sufficiently diverse set of pattern samples.

### Loss & Training
Training is split into two stages: (1) Self-supervised pre-training of the Regional Tokens, using a DINO-style asymmetric distillation loss where the student network's Regional Token features are trained to match the teacher network's CLS token features; (2) Training of the detector, using standard binary cross-entropy loss on the similarity maps generated via the transitivity strategy.

## Key Experimental Results

### Main Results

**VCSL Dataset (Video Copy Segment Localization)**:

| Method | Supervision Type | F1↑ | Precision↑ | Recall↑ |
|------|---------|-----|-----------|---------|
| TN+DTW | Supervised | Low | Medium | Low |
| ViT-CLS | Supervised Features | Medium | Medium | Medium |
| Ours (Annotation-free) | **Self-supervised** | **Highest** | **Highest** | **Highest** |

### Ablation Study

| Configuration | F1↑ | Description |
|------|-----|------|
| Full model | Optimal | Regional Token + Transitivity training |
| w/o Regional Token | Decreased by 3-5% | Only CLS token used, poor performance in picture-in-picture scenarios |
| w/o Transitivity | Decreased more | Replaced with simple augmented data |
| CLS-only + Supervised | Lower than Ours | Supervised but lacks local features |

### Key Findings
- Regional Tokens contribute the most in local copy scenarios like picture-in-picture, where global features struggle to detect copying relationships while Regional Tokens accurately match local regions.
- The transitivity-based data generation strategy is crucial for the self-supervised framework. It not only provides large-scale training data but also covers various complex editing combinations, granting the detector stronger generalization capability.
- The proposed method outperforms supervised methods that rely on manually labeled data without requiring any manual annotations, demonstrating that data diversity can be more critical than label completeness.

## Highlights & Insights
- **Ingenious exploitation of transitivity**: This is an inherent mathematical property of video copying, but prior works have not used it to automatically generate training data with precise annotations. This idea can be extended to other matching problems with transitivity (e.g., hard negative mining in image retrieval, multi-hop reasoning).
- **Elegant and simple design of Regional Tokens**: Without modifying the core architecture of ViT, adding only a few learnable tokens enables local regional awareness through the natural mechanism of self-attention. This design can be directly transferred to other visual retrieval tasks requiring localized perception.
- The entire framework is fully self-supervised, eliminating dependence on expensive manual annotations, thus holding great value for practical deployments.

## Limitations & Future Work
- The number of Regional Tokens is a hyperparameter requiring manual tuning; too many will increase computational overhead, while too few may fail to cover complex local edits.
- The transitivity strategy assumes copy operations are deterministic, but real-world copy chains may involve lossy compression and other irreversible operations, where accumulated errors might affect annotation precision.
- The scale of the similarity map is proportional to video length, leading to significant computation and storage overheads for long videos.
- Future work could consider extending Regional Tokens to a variable number, adaptively increasing or decreasing local attention based on video content.

## Related Work & Insights
- **vs TransNet/ViSiL**: These methods use standard global frame features, performing poorly in picture-in-picture scenarios. The proposed Regional Token directly addresses the local awareness issue.
- **vs VCSL benchmark**: Supervised methods on this benchmark rely on expensive manual annotations; this work demonstrates that a self-supervised approach can outperform them.
- **vs DINO/DINOv2**: The proposed Regional Token can be viewed as a task-specific extension of DINO self-supervised ViT—retaining the self-distillation framework of DINO while introducing local awareness capability tailored for video copy localization.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the Regional Token and transitivity-based data generation are novel and highly effective designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Outperforms supervised methods on standard benchmarks, with ablation studies thoroughly verifying the contribution of each component.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation, with well-aligned solutions to the two key challenges.
- Value: ⭐⭐⭐⭐ Fully self-supervised video copy localization holds high practical value, with direct applicability in copyright protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[CVPR 2026\] Progressive Mask Distillation for Self-supervised Video Representation](../../CVPR2026/self_supervised/progressive_mask_distillation_for_self-supervised_video_representation.md)
- [\[CVPR 2026\] VideoSSR: Video Self-Supervised Reinforcement Learning](../../CVPR2026/self_supervised/videossr_video_self-supervised_reinforcement_learning.md)
- [\[ECCV 2024\] SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning](scpnet_unsupervised_cross-modal_homography_estimation_via_intra-modal_self-super.md)
- [\[CVPR 2025\] AutoSSVH: Automated Frame Sampling for Self-Supervised Video Hashing](../../CVPR2025/self_supervised/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_.md)

</div>

<!-- RELATED:END -->
